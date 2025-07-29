
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .enhanced_pdf_parser import EnhancedPDFParser


class EnhancedResponseSynthesis:
    """
    Enhanced response synthesis service for Google NotebookLM-style analysis.
    
    Provides coherent, detailed responses with proper citations and context.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self._logger = logging.getLogger(__name__)
        self._pdf_parser = EnhancedPDFParser()
        
        # Initialize OpenAI client if available
        self._openai_client = None
        if OPENAI_AVAILABLE and openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=openai_api_key)
        
    async def synthesize_comprehensive_response(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        max_response_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Synthesize a comprehensive, coherent response from search results.
        
        Phase 1 improvements:
        - Uses OpenAI for natural language synthesis
        - Enhanced citation quality with context
        - Better content analysis and organization
        """
        try:
            self._logger.info(f"Synthesizing enhanced response from {len(search_results)} results")
            
            if not search_results:
                return {
                    "response": "I couldn't find relevant information in the document to answer your question.",
                    "citations": [],
                    "confidence": 0.0,
                    "sources_used": 0,
                    "analysis_type": "no_results"
                }
            
            # Step 1: Analyze and organize search results
            organized_results = self._organize_search_results(search_results, query)
            
            # Step 2: Create enhanced citations with context
            enhanced_citations = self._pdf_parser.create_enhanced_citations(search_results, query)
            
            # Step 3: Generate coherent response using OpenAI (if available)
            if self._openai_client:
                response_text = await self._generate_ai_response(query, organized_results, enhanced_citations)
            else:
                response_text = self._generate_structured_response(query, organized_results)
            
            # Step 4: Calculate enhanced confidence score
            confidence = self._calculate_enhanced_confidence(search_results, query, response_text)
            
            # Step 5: Determine analysis type
            analysis_type = self._determine_analysis_type(query, search_results)
            
            self._logger.info(f"Enhanced synthesis complete: {len(response_text)} chars, {len(enhanced_citations)} citations")
            
            return {
                "response": response_text,
                "citations": enhanced_citations,
                "confidence": confidence,
                "sources_used": len(search_results),
                "analysis_type": analysis_type,
                "query_analysis": self._analyze_query(query),
                "synthesis_metadata": {
                    "used_ai_synthesis": self._openai_client is not None,
                    "total_content_analyzed": sum(len(r.get('content', '')) for r in search_results),
                    "synthesis_timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self._logger.error(f"Enhanced synthesis failed: {e}")
            return {
                "response": "I encountered an error while analyzing the document. The content may be complex or require different processing approaches.",
                "citations": [],
                "confidence": 0.0,
                "sources_used": 0,
                "analysis_type": "error"
            }
    
    def _organize_search_results(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Organize search results by relevance and content type."""
        organized = {
            "high_relevance": [],
            "medium_relevance": [],
            "low_relevance": [],
            "definitions": [],
            "examples": [],
            "evidence": [],
            "context": []
        }
        
        query_terms = set(query.lower().split())
        
        for result in results:
            content = result.get('content', '').lower()
            relevance_score = result.get('relevance_score', 0.0)
            
            # Categorize by relevance
            if relevance_score > 0.7:
                organized["high_relevance"].append(result)
            elif relevance_score > 0.4:
                organized["medium_relevance"].append(result)
            else:
                organized["low_relevance"].append(result)
            
            # Categorize by content type
            if any(word in content for word in ['definition', 'define', 'means', 'refers to']):
                organized["definitions"].append(result)
            elif any(word in content for word in ['example', 'instance', 'such as', 'including']):
                organized["examples"].append(result)
            elif any(word in content for word in ['result', 'conclusion', 'finding', 'shows']):
                organized["evidence"].append(result)
            else:
                organized["context"].append(result)
        
        return organized
    
    async def _generate_ai_response(
        self,
        query: str,
        organized_results: Dict[str, Any],
        citations: List[Dict[str, Any]]
    ) -> str:
        """Generate coherent response using OpenAI."""
        try:
            # Prepare context from search results
            context_parts = []
            
            # Add high relevance content first
            for result in organized_results["high_relevance"][:3]:
                content = result.get('content', '')
                page = result.get('page_number', 1)
                context_parts.append(f"[Page {page}] {content}")
            
            # Add medium relevance content
            for result in organized_results["medium_relevance"][:2]:
                content = result.get('content', '')
                page = result.get('page_number', 1)
                context_parts.append(f"[Page {page}] {content}")
            
            context_text = "\n\n".join(context_parts)
            
            # Create prompt for coherent response
            prompt = f"""You are an AI assistant analyzing a document to answer user questions. Provide a comprehensive, coherent response based on the document content.

User Question: {query}

Document Content:
{context_text}

Instructions:
1. Provide a detailed, flowing response that directly answers the question
2. Use information from the document content provided
3. Be specific and reference page numbers when relevant
4. If the document doesn't contain enough information, say so clearly
5. Write in a natural, conversational tone similar to Google NotebookLM
6. Organize information logically with clear structure
7. Maximum length: 500 words

Response:"""

            response = await self._openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful document analysis assistant that provides clear, accurate responses based on document content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self._logger.error(f"AI response generation failed: {e}")
            return self._generate_structured_response(query, organized_results)
    
    def _generate_structured_response(self, query: str, organized_results: Dict[str, Any]) -> str:
        """Generate structured response without AI (fallback)."""
        response_parts = []
        
        # Start with high relevance content
        high_rel = organized_results["high_relevance"]
        if high_rel:
            response_parts.append("Based on the document analysis:")
            
            for i, result in enumerate(high_rel[:2], 1):
                content = result.get('content', '')
                page = result.get('page_number', 1)
                
                # Clean and format content
                cleaned_content = self._clean_content_for_response(content)
                if cleaned_content:
                    response_parts.append(f"\n{i}. {cleaned_content} (Page {page})")
        
        # Add definitions if available
        definitions = organized_results["definitions"]
        if definitions:
            response_parts.append(f"\n\nKey definitions found:")
            for result in definitions[:2]:
                content = self._clean_content_for_response(result.get('content', ''))
                page = result.get('page_number', 1)
                if content:
                    response_parts.append(f"• {content} (Page {page})")
        
        # Add examples if available
        examples = organized_results["examples"]
        if examples:
            response_parts.append(f"\n\nRelevant examples:")
            for result in examples[:2]:
                content = self._clean_content_for_response(result.get('content', ''))
                page = result.get('page_number', 1)
                if content:
                    response_parts.append(f"• {content} (Page {page})")
        
        # Fallback if no good content
        if not response_parts:
            response_parts.append("The document contains information related to your query, but it may require more specific questions to provide detailed answers.")
        
        return "".join(response_parts)
    
    def _clean_content_for_response(self, content: str) -> str:
        """Clean content for inclusion in response."""
        if not content:
            return ""
        
        # Use the enhanced text cleaning from PDF parser
        cleaned = self._pdf_parser._enhanced_text_cleaning(content)
        
        # Additional response-specific cleaning
        # Remove very short fragments
        if len(cleaned) < 20:
            return ""
        
        # Limit length for readability
        if len(cleaned) > 200:
            # Try to cut at sentence boundary
            sentences = cleaned.split('. ')
            if len(sentences) > 1:
                cleaned = '. '.join(sentences[:2]) + '.'
            else:
                cleaned = cleaned[:200] + "..."
        
        return cleaned
    
    def _calculate_enhanced_confidence(
        self,
        results: List[Dict[str, Any]],
        query: str,
        response: str
    ) -> float:
        """Calculate enhanced confidence score."""
        if not results:
            return 0.0
        
        # Base confidence on search result scores
        avg_relevance = sum(r.get('relevance_score', 0) for r in results) / len(results)
        
        # Boost for query term coverage
        query_terms = set(query.lower().split())
        response_lower = response.lower()
        query_coverage = sum(1 for term in query_terms if term in response_lower) / len(query_terms)
        
        # Boost for response length and structure
        length_factor = min(len(response) / 500, 1.0)  # Optimal around 500 chars
        
        # Boost for number of sources
        source_factor = min(len(results) / 5, 1.0)  # Optimal around 5 sources
        
        # Penalty for very short responses
        if len(response) < 100:
            length_factor *= 0.5
        
        confidence = (
            avg_relevance * 0.4 +
            query_coverage * 0.3 +
            length_factor * 0.2 +
            source_factor * 0.1
        )
        
        return min(confidence, 1.0)
    
    def _determine_analysis_type(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Determine the type of analysis performed."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return "definition"
        elif any(word in query_lower for word in ['how', 'process', 'steps']):
            return "explanation"
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            return "reasoning"
        elif any(word in query_lower for word in ['when', 'date', 'time']):
            return "temporal"
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return "spatial"
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return "comparison"
        elif any(word in query_lower for word in ['list', 'examples', 'types']):
            return "enumeration"
        else:
            return "general_analysis"
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to understand user intent."""
        query_lower = query.lower()
        
        return {
            "length": len(query),
            "word_count": len(query.split()),
            "question_type": self._determine_analysis_type(query, []),
            "contains_temporal": bool(re.search(r'\b\d{4}\b', query)),
            "contains_numbers": bool(re.search(r'\d+', query)),
            "complexity": "complex" if len(query.split()) > 10 else "simple",
            "key_terms": [word for word in query.split() if len(word) > 3]
        }


# Factory function for easy integration
def create_enhanced_response_synthesis(openai_api_key: Optional[str] = None) -> EnhancedResponseSynthesis:
    """Create an enhanced response synthesis service."""
    return EnhancedResponseSynthesis(openai_api_key)
