"""
Simple Response Synthesis Service

Creates comprehensive, detailed responses from search results
similar to NotebookLM's analysis style.
"""
import re
import logging
from typing import List, Dict, Any, Optional


class SimpleResponseSynthesis:
    """
    Simple service for synthesizing comprehensive responses from search results.
    
    Provides contextual filtering, deduplication, and response synthesis
    to create detailed, accurate responses like NotebookLM.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def synthesize_comprehensive_response(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        max_response_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Synthesize a comprehensive response from search results.
        
        Args:
            query: Original user query
            search_results: List of search result dictionaries
            max_response_length: Maximum length of synthesized response
            
        Returns:
            Dictionary containing synthesized response and metadata
        """
        try:
            self._logger.info(f"Synthesizing comprehensive response from {len(search_results)} results")
            
            if not search_results:
                return {
                    "response": "I couldn't find relevant information to answer your question.",
                    "citations": [],
                    "confidence": 0.0,
                    "sources_used": 0
                }
            
            # Extract query context
            query_context = self._extract_query_context(query)
            
            # Filter results by context (especially year filtering)
            filtered_results = self._filter_by_context(search_results, query_context)
            
            # Deduplicate similar content
            deduplicated_results = self._deduplicate_results(filtered_results)
            
            # Synthesize comprehensive response
            synthesized_response = self._create_detailed_response(
                query, deduplicated_results, query_context, max_response_length
            )
            
            # Create citations
            citations = self._create_citations(deduplicated_results[:3])  # Top 3 sources
            
            # Calculate confidence
            confidence = self._calculate_confidence(deduplicated_results, query_context)
            
            self._logger.info(f"Synthesized response: {len(synthesized_response)} chars, {len(citations)} citations")
            
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
                "response": "I encountered an error while analyzing the document. Please try again.",
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
            'question_type': 'general'
        }
        
        # Extract years (4-digit numbers) - complete year extraction
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        context['years'] = [int(year) for year in years]
        
        # Extract company/organization names
        company_patterns = [
            r'\\b([A-Z][A-Z\\s&.]+(?:LTD|LLC|INC|CORP|COMPANY|CO\\.?)?)\\b',
            r'\\b([A-Z][a-z]+\\s+[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            context['entities'].extend(matches)
        
        # Determine question type (document-agnostic)
        query_lower = query.lower()
        if any(word in query_lower for word in ['which', 'what', 'where', 'when', 'who', 'how']):
            context['question_type'] = 'specific'
        elif any(word in query_lower for word in ['define', 'explain', 'describe', 'meaning']):
            context['question_type'] = 'definition'
        elif any(word in query_lower for word in ['list', 'enumerate', 'identify', 'find']):
            context['question_type'] = 'enumeration'
        elif any(word in query_lower for word in ['compare', 'contrast', 'difference', 'similar']):
            context['question_type'] = 'comparison'
        else:
            context['question_type'] = 'general'
        
        # Extract keywords dynamically from query (no hardcoded domain-specific terms)
        # Focus on important question words and temporal indicators
        temporal_words = ['year', 'date', 'time', 'period', 'during', 'before', 'after', 'since', 'until']
        question_words = ['what', 'which', 'who', 'where', 'when', 'how', 'why']
        action_words = ['describe', 'explain', 'define', 'list', 'identify', 'compare', 'analyze']
        
        context['keywords'] = []
        for word_list in [temporal_words, question_words, action_words]:
            context['keywords'].extend([word for word in word_list if word in query_lower])
        
        return context
    
    def _filter_by_context(self, results: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results based on query context."""
        if not results:
            return results
        
        filtered = []
        
        for result in results:
            content = result.get('content', '')
            relevance_score = result.get('relevance_score', 0.0)
            
            # Filter by year if specified in query
            if context['years']:
                year_found = False
                for year in context['years']:
                    if str(year) in content:
                        year_found = True
                        relevance_score += 0.4  # Boost for year match
                        break
                
                # Skip results with non-matching years
                if not year_found:
                    other_years = re.findall(r'\b(19\d{2}|20\d{2})\b', content)
                    if other_years and all(int(year) not in context['years'] for year in other_years):
                        continue  # Skip this result
            
            # Boost relevance for entity matches
            for entity in context['entities']:
                if entity.lower() in content.lower():
                    relevance_score += 0.3
            
            # Boost for keyword matches
            for keyword in context['keywords']:
                if keyword in content.lower():
                    relevance_score += 0.1
            
            # Update relevance score and include if above threshold
            result['relevance_score'] = relevance_score
            if relevance_score >= 0.15:
                filtered.append(result)
        
        # Sort by relevance
        filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return filtered
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar content."""
        if not results:
            return results
        
        deduplicated = []
        seen_content_hashes = set()
        
        for result in results:
            content = result.get('content', '')
            
            # Create content hash
            content_hash = hash(' '.join(content.lower().split()))
            
            if content_hash in seen_content_hashes:
                continue
            
            # Check similarity with existing results
            is_duplicate = False
            for existing in deduplicated:
                existing_content = existing.get('content', '')
                
                # Calculate similarity
                words1 = set(content.lower().split())
                words2 = set(existing_content.lower().split())
                
                if words1 and words2:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.7:  # 70% similarity threshold
                        if result.get('relevance_score', 0) > existing.get('relevance_score', 0):
                            deduplicated.remove(existing)
                            break
                        else:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_content_hashes.add(content_hash)
        
        return deduplicated
    
    def _create_detailed_response(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context: Dict[str, Any],
        max_length: int
    ) -> str:
        """Create a detailed, comprehensive response like NotebookLM analysis."""
        
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        response_parts = []
        
        # Handle temporal context (year-based filtering) - UNIVERSAL for any document
        if context.get('years') and len(context['years']) > 0:
            target_year = context['years'][0]
            
            # Find content relevant to the specific year
            year_relevant_results = []
            for result in results:
                content = result.get('content', '')
                if str(target_year) in content:
                    year_relevant_results.append(result)
            
            # Use year-filtered results if available, otherwise use all results
            relevant_results = year_relevant_results if year_relevant_results else results
        else:
            relevant_results = results
        
        # Generate comprehensive response based on question type - UNIVERSAL approach
        if context['question_type'] == 'specific' and relevant_results:
            # Handle specific questions (what, which, who, where, when, how)
            best_result = relevant_results[0]
            content = best_result.get('content', '')
            
            # Create comprehensive analysis
            response_parts.append(f"Based on the document analysis:")
            
            # Extract key information from the most relevant content
            sentences = content.split('.')
            key_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
            
            if key_sentences:
                response_parts.append(f"\n\n{key_sentences[0]}.")
                
                # Add supporting details if available
                if len(key_sentences) > 1:
                    response_parts.append(f"\n\n**Additional Details:**")
                    for i, sentence in enumerate(key_sentences[1:], 1):
                        response_parts.append(f"\n{i}. {sentence}.")
            
            # Add temporal context if year was specified
            if context.get('years'):
                year = context['years'][0]
                response_parts.append(f"\n\n**Temporal Context:** This information is specifically related to {year} as requested.")
            
            return ''.join(response_parts)
        
        elif context['question_type'] == 'definition' and relevant_results:
            # Handle definition questions (define, explain, describe, meaning)
            best_result = relevant_results[0]
            content = best_result.get('content', '')
            
            response_parts.append(f"Based on the document content:")
            
            # Extract definition or explanation
            sentences = content.split('.')
            definition_sentences = [s.strip() for s in sentences[:2] if len(s.strip()) > 15]
            
            if definition_sentences:
                response_parts.append(f"\n\n{definition_sentences[0]}.")
                if len(definition_sentences) > 1:
                    response_parts.append(f" {definition_sentences[1]}.")
            
            return ''.join(response_parts)
        
        elif context['question_type'] == 'enumeration' and relevant_results:
            # Handle list/enumeration questions (list, enumerate, identify, find)
            response_parts.append(f"Based on the document analysis, here are the key items identified:")
            
            # Extract list items from multiple results
            items_found = []
            for result in relevant_results[:3]:  # Use top 3 results
                content = result.get('content', '')
                
                # Look for bullet points or numbered items
                if '•' in content:
                    bullet_items = content.split('•')
                    for item in bullet_items[1:4]:  # First 3 items
                        cleaned = item.strip()
                        if len(cleaned) > 10:
                            items_found.append(cleaned)
                elif '\n' in content:
                    lines = content.split('\n')
                    for line in lines[:3]:
                        cleaned = line.strip()
                        if len(cleaned) > 15:
                            items_found.append(cleaned)
            
            # Format the items
            if items_found:
                for i, item in enumerate(items_found[:5], 1):  # Max 5 items
                    response_parts.append(f"\n{i}. {item}")
            
            return ''.join(response_parts)
        
        elif context['question_type'] == 'comparison' and len(relevant_results) >= 2:
            # Handle comparison questions (compare, contrast, difference, similar)
            response_parts.append(f"Based on the document analysis, here is a comparison:")
            
            # Compare content from multiple results
            result1 = relevant_results[0]
            result2 = relevant_results[1]
            
            response_parts.append(f"\n\n**First aspect:** {result1.get('content', '')[:200]}...")
            response_parts.append(f"\n\n**Second aspect:** {result2.get('content', '')[:200]}...")
            
            return ''.join(response_parts)
        
        # Fallback to general comprehensive analysis
        if len(response_parts) == 0:
            # Create a detailed analysis from available results
            response_parts.append("Based on the document analysis:")
            
            for i, result in enumerate(results[:2], 1):  # Use top 2 results
                content = result.get('content', '')
                if len(content) > 50:
                    # Extract key information
                    sentences = content.split('.')
                    key_info = sentences[0] if sentences else content[:200]
                    response_parts.append(f"\n\n{i}. {key_info.strip()}.")
        
        # Combine response parts
        response = ''.join(response_parts)
        
        # Ensure response doesn't exceed max length
        if len(response) > max_length:
            response = response[:max_length-3] + "..."
        
        return response
    
    def _create_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create citations from search results."""
        citations = []
        
        for i, result in enumerate(results):
            citation = {
                "id": f"citation_{i+1}",
                "document_id": result.get('document_id'),
                "document_title": result.get('document_title', 'Unknown Document'),
                "page_number": result.get('page_number', 1),
                "snippet": result.get('snippet', result.get('content', '')[:200]),
                "relevance_score": result.get('relevance_score', 0.0)
            }
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence(self, results: List[Dict[str, Any]], context: Dict[str, Any]) -> float:
        """Calculate confidence score for the synthesized response."""
        if not results:
            return 0.0
        
        # Base confidence on average relevance score
        avg_relevance = sum(r.get('relevance_score', 0) for r in results) / len(results)
        
        # Boost confidence if context matches (e.g., year found)
        context_boost = 0.0
        if context.get('years') and any(str(year) in r.get('content', '') for year in context['years'] for r in results):
            context_boost += 0.2
        
        # Boost confidence based on number of supporting sources
        source_boost = min(len(results) * 0.05, 0.2)  # Max 0.2 boost
        
        confidence = min(avg_relevance + context_boost + source_boost, 1.0)
        return confidence
