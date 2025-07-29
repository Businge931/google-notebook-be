"""
Response Synthesis Service Implementation

Synthesizes multiple search results into comprehensive, detailed responses
with proper contextual filtering and citation integration.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

# Import core domain entities - using try/except for flexibility
try:
    from src.core.domain.services.advanced_search_service import SearchResult
    from src.core.domain.entities.citation import Citation
    from src.core.domain.value_objects.document_id import DocumentId
    from src.core.domain.value_objects.page_number import PageNumber
except ImportError:
    # Fallback for missing domain entities
    SearchResult = None
    Citation = None
    DocumentId = None
    PageNumber = None


class ResponseSynthesisService:
    """
    Service for synthesizing multiple search results into comprehensive responses.
    
    Provides contextual filtering, information aggregation, and citation integration
    to create detailed, accurate responses similar to NotebookLM's analysis style.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    async def synthesize_response(
        self,
        query: str,
        search_results: List[SearchResult],
        max_response_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Synthesize multiple search results into a comprehensive response.
        
        Args:
            query: Original user query
            search_results: List of search results to synthesize
            max_response_length: Maximum length of synthesized response
            
        Returns:
            Dictionary containing synthesized response and metadata
        """
        try:
            self._logger.info(f"Synthesizing response from {len(search_results)} search results")
            
            if not search_results:
                return {
                    "response": "I couldn't find relevant information to answer your question.",
                    "citations": [],
                    "confidence": 0.0,
                    "sources_used": 0
                }
            
            # Extract query context for filtering
            query_context = self._extract_query_context(query)
            
            # Filter and rank results by relevance and context
            filtered_results = self._filter_by_context(search_results, query_context)
            ranked_results = self._rank_results_by_relevance(filtered_results, query_context)
            
            # Deduplicate similar content
            deduplicated_results = self._deduplicate_content(ranked_results)
            
            # Aggregate information by topic/entity
            aggregated_info = self._aggregate_information(deduplicated_results, query_context)
            
            # Synthesize comprehensive response
            synthesized_response = self._synthesize_comprehensive_response(
                query, aggregated_info, query_context, max_response_length
            )
            
            # Extract citations from used results
            citations = self._extract_citations(deduplicated_results[:5])  # Top 5 sources
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(deduplicated_results, query_context)
            
            self._logger.info(f"Synthesized response with {len(citations)} citations, confidence: {confidence:.2f}")
            
            return {
                "response": synthesized_response,
                "citations": citations,
                "confidence": confidence,
                "sources_used": len(deduplicated_results),
                "query_context": query_context
            }
            
        except Exception as e:
            self._logger.error(f"Failed to synthesize response: {e}")
            return {
                "response": "I encountered an error while processing your question. Please try again.",
                "citations": [],
                "confidence": 0.0,
                "sources_used": 0
            }
    
    def _extract_query_context(self, query: str) -> Dict[str, Any]:
        """Extract contextual information from query."""
        context = {
            'years': [],
            'entities': [],
            'keywords': [],
            'question_type': 'general',
            'focus_areas': []
        }
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        context['years'] = [int(year) for year in years]
        
        # Extract company/organization names
        company_patterns = [
            r'\b([A-Z][A-Z\s&.]+(?:LTD|LLC|INC|CORP|COMPANY|CO\.?))\b',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            context['entities'].extend(matches)
        
        # Determine question type and focus
        query_lower = query.lower()
        if any(word in query_lower for word in ['which', 'what', 'where', 'when', 'who']):
            context['question_type'] = 'specific'
        elif any(word in query_lower for word in ['company', 'work', 'job', 'position']):
            context['question_type'] = 'employment'
            context['focus_areas'].append('employment')
        elif any(word in query_lower for word in ['skill', 'technology', 'tool']):
            context['question_type'] = 'technical'
            context['focus_areas'].append('technical_skills')
        elif any(word in query_lower for word in ['education', 'degree', 'university']):
            context['question_type'] = 'education'
            context['focus_areas'].append('education')
        
        # Extract important keywords
        important_words = ['company', 'work', 'job', 'position', 'role', 'experience', 'skill', 'technology', 'education']
        context['keywords'] = [word for word in important_words if word in query_lower]
        
        return context
    
    def _filter_by_context(self, results: List[SearchResult], context: Dict[str, Any]) -> List[SearchResult]:
        """Filter results based on query context."""
        if not results:
            return results
        
        filtered = []
        
        for result in results:
            content_lower = result.content.lower()
            relevance_score = result.relevance_score
            
            # Filter by year if specified
            if context['years']:
                year_found = False
                for year in context['years']:
                    if str(year) in result.content:
                        year_found = True
                        relevance_score += 0.4  # Boost for year match
                        break
                
                # Skip results with non-matching years
                if not year_found:
                    other_years = re.findall(r'\b(19|20)\d{2}\b', result.content)
                    if other_years and all(int(year) not in context['years'] for year in other_years):
                        continue
            
            # Boost relevance for entity matches
            for entity in context['entities']:
                if entity.lower() in content_lower:
                    relevance_score += 0.3
            
            # Boost for keyword matches
            for keyword in context['keywords']:
                if keyword in content_lower:
                    relevance_score += 0.1
            
            # Update relevance score
            result.relevance_score = relevance_score
            
            # Only include results above minimum threshold
            if relevance_score >= 0.15:
                filtered.append(result)
        
        return filtered
    
    def _rank_results_by_relevance(self, results: List[SearchResult], context: Dict[str, Any]) -> List[SearchResult]:
        """Rank results by relevance score."""
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _deduplicate_content(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate or very similar content."""
        if not results:
            return results
        
        deduplicated = []
        seen_content_hashes = set()
        
        for result in results:
            # Create content hash for exact duplicates
            content_hash = hash(' '.join(result.content.lower().split()))
            
            if content_hash in seen_content_hashes:
                continue
            
            # Check similarity with existing results
            is_duplicate = False
            for existing in deduplicated:
                words1 = set(result.content.lower().split())
                words2 = set(existing.content.lower().split())
                
                if words1 and words2:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.7:  # 70% similarity threshold
                        if result.relevance_score > existing.relevance_score:
                            deduplicated.remove(existing)
                            break
                        else:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_content_hashes.add(content_hash)
        
        return deduplicated
    
    def _aggregate_information(self, results: List[SearchResult], context: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate information from results by topic/entity."""
        aggregated = {
            'employment_history': [],
            'technical_skills': [],
            'education': [],
            'projects': [],
            'other_info': []
        }
        
        for result in results:
            content = result.content
            
            # Categorize content
            if any(keyword in content.lower() for keyword in ['developer', 'engineer', 'job', 'work', 'company']):
                aggregated['employment_history'].append({
                    'content': content,
                    'source': result,
                    'relevance': result.relevance_score
                })
            elif any(keyword in content.lower() for keyword in ['skill', 'technology', 'programming', 'framework']):
                aggregated['technical_skills'].append({
                    'content': content,
                    'source': result,
                    'relevance': result.relevance_score
                })
            elif any(keyword in content.lower() for keyword in ['education', 'degree', 'university', 'college']):
                aggregated['education'].append({
                    'content': content,
                    'source': result,
                    'relevance': result.relevance_score
                })
            elif any(keyword in content.lower() for keyword in ['project', 'developed', 'built', 'created']):
                aggregated['projects'].append({
                    'content': content,
                    'source': result,
                    'relevance': result.relevance_score
                })
            else:
                aggregated['other_info'].append({
                    'content': content,
                    'source': result,
                    'relevance': result.relevance_score
                })
        
        return aggregated
    
    def _synthesize_comprehensive_response(
        self,
        query: str,
        aggregated_info: Dict[str, Any],
        context: Dict[str, Any],
        max_length: int
    ) -> str:
        """Synthesize a comprehensive response from aggregated information."""
        
        response_parts = []
        
        # Start with direct answer if it's a specific question
        if context['question_type'] == 'specific' and context.get('years'):
            year = context['years'][0]
            
            # Look for employment information for the specific year
            employment_info = aggregated_info.get('employment_history', [])
            relevant_employment = []
            
            for emp in employment_info:
                if str(year) in emp['content']:
                    relevant_employment.append(emp)
            
            if relevant_employment:
                # Sort by relevance
                relevant_employment.sort(key=lambda x: x['relevance'], reverse=True)
                best_match = relevant_employment[0]
                
                # Extract company name and details
                content = best_match['content']
                
                # Parse employment details
                company_match = re.search(r'([A-Z][A-Z\s&.]*(?:LTD|LLC|INC|CORP|COMPANY|CO\.?)?)', content, re.IGNORECASE)
                position_match = re.search(r'(Developer|Engineer|Programmer|Analyst|Manager|Designer|Consultant)', content, re.IGNORECASE)
                
                if company_match:
                    company = company_match.group(1).strip()
                    position = position_match.group(1) if position_match else "a position"
                    
                    response_parts.append(f"Based on the document content, in {year}, the candidate worked for **{company}** as a {position}.")
                    
                    # Add more details from the content
                    if len(content) > 200:
                        # Extract key responsibilities or details
                        sentences = content.split('•')
                        if len(sentences) > 1:
                            key_details = [s.strip() for s in sentences[1:3] if s.strip()]  # First 2 bullet points
                            if key_details:
                                response_parts.append(f"\nKey responsibilities and achievements included:")
                                for detail in key_details:
                                    response_parts.append(f"• {detail}")
        
        # Add additional context from other relevant information
        if len(response_parts) == 0:
            # Fallback to general information synthesis
            all_info = []
            for category, items in aggregated_info.items():
                all_info.extend(items)
            
            if all_info:
                all_info.sort(key=lambda x: x['relevance'], reverse=True)
                best_info = all_info[0]
                response_parts.append(f"Based on the available information: {best_info['content'][:300]}...")
        
        # Combine response parts
        response = '\n'.join(response_parts)
        
        # Ensure response doesn't exceed max length
        if len(response) > max_length:
            response = response[:max_length-3] + "..."
        
        return response
    
    def _extract_citations(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Extract citations from search results."""
        citations = []
        
        for i, result in enumerate(results):
            citation = {
                "id": f"citation_{i+1}",
                "document_id": result.document_id.value if result.document_id else None,
                "document_title": result.document_title,
                "page_number": result.page_number,
                "snippet": result.snippet[:200] if result.snippet else result.content[:200],
                "relevance_score": result.relevance_score
            }
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence_score(self, results: List[SearchResult], context: Dict[str, Any]) -> float:
        """Calculate confidence score for the synthesized response."""
        if not results:
            return 0.0
        
        # Base confidence on average relevance score
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        
        # Boost confidence if context matches (e.g., year found)
        context_boost = 0.0
        if context.get('years') and any(str(year) in r.content for year in context['years'] for r in results):
            context_boost += 0.2
        
        # Boost confidence based on number of supporting sources
        source_boost = min(len(results) * 0.05, 0.2)  # Max 0.2 boost
        
        confidence = min(avg_relevance + context_boost + source_boost, 1.0)
        return confidence
