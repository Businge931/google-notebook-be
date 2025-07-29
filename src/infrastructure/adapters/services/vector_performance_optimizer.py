"""
Vector Search Performance Optimizer

"""
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime, timedelta

from src.core.domain.services.embedding_service import SimilaritySearchService
from src.core.domain.value_objects import DocumentId
from src.shared.exceptions import VectorOptimizationError


@dataclass
class OptimizationMetrics:
    """Performance optimization metrics following Single Responsibility Principle."""
    search_time_ms: float
    index_size_mb: float
    memory_usage_mb: float
    cache_hit_rate: float
    throughput_qps: float
    accuracy_score: float
    optimization_applied: List[str]
    timestamp: datetime


@dataclass
class BatchSearchRequest:
    """Batch search request for performance optimization."""
    queries: List[str]
    embeddings: Optional[List[List[float]]] = None
    max_results_per_query: int = 10
    similarity_threshold: float = 0.7
    document_filters: Optional[List[DocumentId]] = None


@dataclass
class BatchSearchResult:
    """Batch search result with performance metrics."""
    results: List[List[Dict[str, Any]]]
    total_time_ms: float
    average_time_per_query_ms: float
    cache_hits: int
    cache_misses: int
    optimization_stats: Dict[str, Any]


class VectorIndexOptimizer(ABC):
    """Abstract interface for vector index optimization following Interface Segregation Principle."""
    
    @abstractmethod
    async def optimize_index(
        self,
        index_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize vector index configuration."""
        pass
    
    @abstractmethod
    async def tune_parameters(
        self,
        performance_history: List[OptimizationMetrics]
    ) -> Dict[str, Any]:
        """Tune index parameters based on performance history."""
        pass


class VectorCacheManager(ABC):
    """Abstract interface for vector caching following Interface Segregation Principle."""
    
    @abstractmethod
    async def get_cached_results(
        self,
        query_hash: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        pass
    
    @abstractmethod
    async def cache_results(
        self,
        query_hash: str,
        results: List[Dict[str, Any]],
        ttl_seconds: int = 3600
    ) -> None:
        """Cache search results."""
        pass
    
    @abstractmethod
    async def invalidate_cache(
        self,
        document_ids: Optional[List[DocumentId]] = None
    ) -> None:
        """Invalidate cache entries."""
        pass


class BatchProcessor(ABC):
    """Abstract interface for batch processing following Interface Segregation Principle."""
    
    @abstractmethod
    async def process_batch(
        self,
        batch_request: BatchSearchRequest
    ) -> BatchSearchResult:
        """Process batch of search requests."""
        pass


class FAISSIndexOptimizer(VectorIndexOptimizer):
    """
    FAISS index optimizer implementation following Single Responsibility Principle.
    
    Optimizes FAISS index configuration and parameters for better performance.
    """
    
    def __init__(self):
        """Initialize FAISS index optimizer."""
        self._logger = logging.getLogger(__name__)
        self._optimization_history = []
    
    async def optimize_index(
        self,
        index_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize FAISS index configuration.
        
        Args:
            index_config: Current index configuration
            
        Returns:
            Optimized index configuration
            
        Raises:
            VectorOptimizationError: If optimization fails
        """
        try:
            self._logger.info("Optimizing FAISS index configuration")
            
            # Analyze current configuration
            current_type = index_config.get("index_type", "IndexFlatIP")
            dimension = index_config.get("dimension", 1536)
            num_vectors = index_config.get("num_vectors", 0)
            
            # Determine optimal index type based on data size
            optimized_config = index_config.copy()
            
            if num_vectors < 1000:
                # Small dataset - use flat index for accuracy
                optimized_config["index_type"] = "IndexFlatIP"
                optimized_config["nprobe"] = None
            elif num_vectors < 10000:
                # Medium dataset - use IVF with small number of centroids
                optimized_config["index_type"] = "IndexIVFFlat"
                optimized_config["nlist"] = min(100, max(10, num_vectors // 100))
                optimized_config["nprobe"] = min(10, optimized_config["nlist"])
            elif num_vectors < 100000:
                # Large dataset - use IVF with PQ compression
                optimized_config["index_type"] = "IndexIVFPQ"
                optimized_config["nlist"] = min(1000, max(100, num_vectors // 100))
                optimized_config["nprobe"] = min(20, optimized_config["nlist"] // 4)
                optimized_config["m"] = min(64, max(8, dimension // 8))
                optimized_config["nbits"] = 8
            else:
                # Very large dataset - use HNSW for speed
                optimized_config["index_type"] = "IndexHNSWFlat"
                optimized_config["M"] = 32
                optimized_config["efConstruction"] = 200
                optimized_config["efSearch"] = 50
            
            # Add performance tuning parameters
            optimized_config.update({
                "use_gpu": num_vectors > 50000,  # Use GPU for large datasets
                "batch_size": min(1000, max(10, num_vectors // 100)),
                "prefetch_factor": 2,
                "num_threads": min(8, max(1, num_vectors // 10000))
            })
            
            self._logger.info(f"Index optimization completed: {current_type} -> {optimized_config['index_type']}")
            
            return optimized_config
            
        except Exception as e:
            self._logger.error(f"Index optimization failed: {e}")
            raise VectorOptimizationError(f"Index optimization failed: {str(e)}")
    
    async def tune_parameters(
        self,
        performance_history: List[OptimizationMetrics]
    ) -> Dict[str, Any]:
        """
        Tune index parameters based on performance history.
        
        Args:
            performance_history: Historical performance metrics
            
        Returns:
            Tuned parameters
            
        Raises:
            VectorOptimizationError: If parameter tuning fails
        """
        try:
            if not performance_history:
                return {}
            
            self._logger.info(f"Tuning parameters based on {len(performance_history)} metrics")
            
            # Analyze performance trends
            recent_metrics = performance_history[-10:]  # Last 10 measurements
            avg_search_time = sum(m.search_time_ms for m in recent_metrics) / len(recent_metrics)
            avg_accuracy = sum(m.accuracy_score for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            
            tuned_params = {}
            
            # Tune based on search time
            if avg_search_time > 500:  # Slow searches
                tuned_params.update({
                    "reduce_nprobe": True,
                    "increase_batch_size": True,
                    "enable_parallel_search": True
                })
            elif avg_search_time < 50:  # Very fast searches, can trade speed for accuracy
                tuned_params.update({
                    "increase_nprobe": True,
                    "increase_ef_search": True
                })
            
            # Tune based on accuracy
            if avg_accuracy < 0.8:  # Low accuracy
                tuned_params.update({
                    "increase_nprobe": True,
                    "reduce_compression": True,
                    "increase_ef_search": True
                })
            
            # Tune based on cache performance
            if avg_cache_hit_rate < 0.3:  # Low cache hit rate
                tuned_params.update({
                    "increase_cache_size": True,
                    "extend_cache_ttl": True,
                    "improve_cache_key_strategy": True
                })
            
            self._logger.info(f"Parameter tuning completed: {len(tuned_params)} adjustments")
            
            return tuned_params
            
        except Exception as e:
            self._logger.error(f"Parameter tuning failed: {e}")
            raise VectorOptimizationError(f"Parameter tuning failed: {str(e)}")


class RedisCacheManager(VectorCacheManager):
    """
    Redis-based vector cache manager following Single Responsibility Principle.
    
    Manages caching of vector search results using Redis.
    """
    
    def __init__(self, redis_client=None, default_ttl: int = 3600):
        """
        Initialize Redis cache manager.
        
        Args:
            redis_client: Redis client instance
            default_ttl: Default TTL for cache entries
        """
        self._redis_client = redis_client
        self._default_ttl = default_ttl
        self._logger = logging.getLogger(__name__)
        self._cache_prefix = "vector_search:"
    
    async def get_cached_results(
        self,
        query_hash: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search results.
        
        Args:
            query_hash: Hash of the search query
            
        Returns:
            Cached results if available, None otherwise
        """
        try:
            if not self._redis_client:
                return None
            
            cache_key = f"{self._cache_prefix}{query_hash}"
            cached_data = await self._redis_client.get(cache_key)
            
            if cached_data:
                import json
                results = json.loads(cached_data)
                self._logger.debug(f"Cache hit for query hash: {query_hash}")
                return results
            
            self._logger.debug(f"Cache miss for query hash: {query_hash}")
            return None
            
        except Exception as e:
            self._logger.error(f"Cache retrieval failed: {e}")
            return None
    
    async def cache_results(
        self,
        query_hash: str,
        results: List[Dict[str, Any]],
        ttl_seconds: int = None
    ) -> None:
        """
        Cache search results.
        
        Args:
            query_hash: Hash of the search query
            results: Search results to cache
            ttl_seconds: TTL for cache entry
        """
        try:
            if not self._redis_client:
                return
            
            cache_key = f"{self._cache_prefix}{query_hash}"
            ttl = ttl_seconds or self._default_ttl
            
            import json
            serialized_results = json.dumps(results, default=str)
            
            await self._redis_client.setex(
                cache_key,
                ttl,
                serialized_results
            )
            
            self._logger.debug(f"Cached results for query hash: {query_hash}")
            
        except Exception as e:
            self._logger.error(f"Cache storage failed: {e}")
    
    async def invalidate_cache(
        self,
        document_ids: Optional[List[DocumentId]] = None
    ) -> None:
        """
        Invalidate cache entries.
        
        Args:
            document_ids: Document IDs to invalidate cache for
        """
        try:
            if not self._redis_client:
                return
            
            if document_ids:
                # Invalidate specific document caches
                for doc_id in document_ids:
                    pattern = f"{self._cache_prefix}*{doc_id.value}*"
                    keys = await self._redis_client.keys(pattern)
                    if keys:
                        await self._redis_client.delete(*keys)
                        self._logger.info(f"Invalidated {len(keys)} cache entries for document {doc_id.value}")
            else:
                # Invalidate all vector search caches
                pattern = f"{self._cache_prefix}*"
                keys = await self._redis_client.keys(pattern)
                if keys:
                    await self._redis_client.delete(*keys)
                    self._logger.info(f"Invalidated {len(keys)} cache entries")
            
        except Exception as e:
            self._logger.error(f"Cache invalidation failed: {e}")


class VectorBatchProcessor(BatchProcessor):
    """
    Vector batch processor implementation following Single Responsibility Principle.
    
    Processes multiple vector search requests in batches for improved performance.
    """
    
    def __init__(
        self,
        similarity_service: SimilaritySearchService,
        cache_manager: Optional[VectorCacheManager] = None,
        max_batch_size: int = 100,
        max_workers: int = 4
    ):
        """
        Initialize vector batch processor.
        
        Args:
            similarity_service: Similarity search service
            cache_manager: Optional cache manager
            max_batch_size: Maximum batch size
            max_workers: Maximum worker threads
        """
        self._similarity_service = similarity_service
        self._cache_manager = cache_manager
        self._max_batch_size = max_batch_size
        self._max_workers = max_workers
        self._logger = logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(
        self,
        batch_request: BatchSearchRequest
    ) -> BatchSearchResult:
        """
        Process batch of search requests.
        
        Args:
            batch_request: Batch search request
            
        Returns:
            Batch search result with performance metrics
            
        Raises:
            VectorOptimizationError: If batch processing fails
        """
        try:
            start_time = time.time()
            queries = batch_request.queries
            
            self._logger.info(f"Processing batch of {len(queries)} search requests")
            
            # Split into smaller batches if needed
            batches = [
                queries[i:i + self._max_batch_size]
                for i in range(0, len(queries), self._max_batch_size)
            ]
            
            all_results = []
            cache_hits = 0
            cache_misses = 0
            
            # Process batches in parallel
            batch_tasks = []
            for batch in batches:
                task = self._process_single_batch(
                    batch,
                    batch_request.max_results_per_query,
                    batch_request.similarity_threshold,
                    batch_request.document_filters
                )
                batch_tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Collect results and metrics
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    self._logger.error(f"Batch processing error: {batch_result}")
                    # Add empty results for failed batch
                    batch_size = len(batches[len(all_results)])
                    all_results.extend([[] for _ in range(batch_size)])
                else:
                    results, hits, misses = batch_result
                    all_results.extend(results)
                    cache_hits += hits
                    cache_misses += misses
            
            # Calculate performance metrics
            total_time = (time.time() - start_time) * 1000  # Convert to ms
            avg_time_per_query = total_time / len(queries) if queries else 0
            
            optimization_stats = {
                "batches_processed": len(batches),
                "parallel_workers": self._max_workers,
                "cache_enabled": self._cache_manager is not None,
                "total_queries": len(queries),
                "successful_queries": len([r for r in all_results if r])
            }
            
            self._logger.info(
                f"Batch processing completed: {len(queries)} queries in {total_time:.2f}ms "
                f"(avg: {avg_time_per_query:.2f}ms per query)"
            )
            
            return BatchSearchResult(
                results=all_results,
                total_time_ms=total_time,
                average_time_per_query_ms=avg_time_per_query,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                optimization_stats=optimization_stats
            )
            
        except Exception as e:
            self._logger.error(f"Batch processing failed: {e}")
            raise VectorOptimizationError(f"Batch processing failed: {str(e)}")
    
    async def _process_single_batch(
        self,
        queries: List[str],
        max_results: int,
        similarity_threshold: float,
        document_filters: Optional[List[DocumentId]]
    ) -> Tuple[List[List[Dict[str, Any]]], int, int]:
        """Process a single batch of queries."""
        try:
            results = []
            cache_hits = 0
            cache_misses = 0
            
            for query in queries:
                # Check cache first
                query_hash = self._generate_query_hash(query, max_results, similarity_threshold)
                cached_result = None
                
                if self._cache_manager:
                    cached_result = await self._cache_manager.get_cached_results(query_hash)
                
                if cached_result:
                    results.append(cached_result)
                    cache_hits += 1
                else:
                    # Perform actual search
                    search_results = await self._similarity_service.similarity_search(
                        query=query,
                        max_results=max_results,
                        similarity_threshold=similarity_threshold,
                        document_filters=document_filters
                    )
                    
                    # Convert to serializable format
                    serializable_results = []
                    for result in search_results:
                        serializable_results.append({
                            "document_id": result.document_id.value,
                            "chunk_id": result.chunk_id,
                            "content": result.content,
                            "similarity_score": result.similarity_score,
                            "metadata": result.metadata
                        })
                    
                    results.append(serializable_results)
                    cache_misses += 1
                    
                    # Cache the result
                    if self._cache_manager:
                        await self._cache_manager.cache_results(query_hash, serializable_results)
            
            return results, cache_hits, cache_misses
            
        except Exception as e:
            self._logger.error(f"Single batch processing failed: {e}")
            raise
    
    def _generate_query_hash(
        self,
        query: str,
        max_results: int,
        similarity_threshold: float
    ) -> str:
        """Generate hash for query caching."""
        import hashlib
        
        cache_key = f"{query}:{max_results}:{similarity_threshold}"
        return hashlib.md5(cache_key.encode()).hexdigest()


class VectorPerformanceOptimizer:
    """
    Main vector performance optimizer orchestrating all optimization strategies.
    
    Follows Single Responsibility Principle by coordinating optimization components.
    """
    
    def __init__(
        self,
        index_optimizer: VectorIndexOptimizer,
        cache_manager: VectorCacheManager,
        batch_processor: BatchProcessor
    ):
        """
        Initialize vector performance optimizer.
        
        Args:
            index_optimizer: Index optimization service
            cache_manager: Cache management service
            batch_processor: Batch processing service
        """
        self._index_optimizer = index_optimizer
        self._cache_manager = cache_manager
        self._batch_processor = batch_processor
        self._logger = logging.getLogger(__name__)
        self._metrics_history = []
    
    async def optimize_performance(
        self,
        current_config: Dict[str, Any],
        performance_target: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Optimize vector search performance.
        
        Args:
            current_config: Current configuration
            performance_target: Target performance metrics
            
        Returns:
            Optimized configuration
            
        Raises:
            VectorOptimizationError: If optimization fails
        """
        try:
            self._logger.info("Starting vector performance optimization")
            
            # Set default performance targets
            if not performance_target:
                performance_target = {
                    "max_search_time_ms": 200,
                    "min_accuracy_score": 0.85,
                    "min_cache_hit_rate": 0.5,
                    "min_throughput_qps": 100
                }
            
            # Optimize index configuration
            optimized_index_config = await self._index_optimizer.optimize_index(current_config)
            
            # Tune parameters based on history
            if self._metrics_history:
                tuned_params = await self._index_optimizer.tune_parameters(self._metrics_history)
                optimized_index_config.update(tuned_params)
            
            # Create final optimized configuration
            optimized_config = {
                **current_config,
                **optimized_index_config,
                "performance_target": performance_target,
                "optimization_timestamp": datetime.utcnow().isoformat(),
                "cache_enabled": True,
                "batch_processing_enabled": True
            }
            
            self._logger.info("Vector performance optimization completed")
            
            return optimized_config
            
        except Exception as e:
            self._logger.error(f"Performance optimization failed: {e}")
            raise VectorOptimizationError(f"Performance optimization failed: {str(e)}")
    
    async def measure_performance(
        self,
        test_queries: List[str],
        config: Dict[str, Any]
    ) -> OptimizationMetrics:
        """
        Measure current performance metrics.
        
        Args:
            test_queries: Test queries for measurement
            config: Current configuration
            
        Returns:
            Performance metrics
            
        Raises:
            VectorOptimizationError: If measurement fails
        """
        try:
            self._logger.info(f"Measuring performance with {len(test_queries)} test queries")
            
            start_time = time.time()
            
            # Create batch request
            batch_request = BatchSearchRequest(
                queries=test_queries,
                max_results_per_query=10,
                similarity_threshold=0.7
            )
            
            # Process batch and measure performance
            batch_result = await self._batch_processor.process_batch(batch_request)
            
            # Calculate metrics
            search_time_ms = batch_result.average_time_per_query_ms
            cache_hit_rate = (
                batch_result.cache_hits / 
                (batch_result.cache_hits + batch_result.cache_misses)
                if (batch_result.cache_hits + batch_result.cache_misses) > 0 else 0
            )
            
            # Mock additional metrics (would be measured from actual system)
            metrics = OptimizationMetrics(
                search_time_ms=search_time_ms,
                index_size_mb=config.get("estimated_index_size_mb", 100),
                memory_usage_mb=config.get("estimated_memory_mb", 500),
                cache_hit_rate=cache_hit_rate,
                throughput_qps=1000 / search_time_ms if search_time_ms > 0 else 0,
                accuracy_score=0.9,  # Would be measured against ground truth
                optimization_applied=config.get("optimization_applied", []),
                timestamp=datetime.utcnow()
            )
            
            # Store metrics in history
            self._metrics_history.append(metrics)
            
            # Keep only recent metrics
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-50:]
            
            self._logger.info(
                f"Performance measurement completed: "
                f"search_time={search_time_ms:.2f}ms, "
                f"cache_hit_rate={cache_hit_rate:.2f}, "
                f"throughput={metrics.throughput_qps:.2f}qps"
            )
            
            return metrics
            
        except Exception as e:
            self._logger.error(f"Performance measurement failed: {e}")
            raise VectorOptimizationError(f"Performance measurement failed: {str(e)}")
    
    async def get_optimization_recommendations(
        self,
        current_metrics: OptimizationMetrics,
        target_metrics: Dict[str, float]
    ) -> List[str]:
        """
        Get optimization recommendations based on current performance.
        
        Args:
            current_metrics: Current performance metrics
            target_metrics: Target performance metrics
            
        Returns:
            List of optimization recommendations
        """
        try:
            recommendations = []
            
            # Analyze search time
            if current_metrics.search_time_ms > target_metrics.get("max_search_time_ms", 200):
                recommendations.extend([
                    "Consider using a faster index type (e.g., HNSW for large datasets)",
                    "Increase batch processing to amortize overhead",
                    "Enable GPU acceleration for large vector operations",
                    "Reduce vector dimensions if possible",
                    "Implement query result caching"
                ])
            
            # Analyze cache performance
            if current_metrics.cache_hit_rate < target_metrics.get("min_cache_hit_rate", 0.5):
                recommendations.extend([
                    "Increase cache size to store more results",
                    "Extend cache TTL for stable queries",
                    "Implement smarter cache key strategies",
                    "Use query clustering to improve cache efficiency"
                ])
            
            # Analyze throughput
            if current_metrics.throughput_qps < target_metrics.get("min_throughput_qps", 100):
                recommendations.extend([
                    "Implement parallel query processing",
                    "Use connection pooling for database operations",
                    "Optimize vector index for your query patterns",
                    "Consider horizontal scaling with multiple index replicas"
                ])
            
            # Analyze accuracy
            if current_metrics.accuracy_score < target_metrics.get("min_accuracy_score", 0.85):
                recommendations.extend([
                    "Use higher precision index types",
                    "Increase search parameters (nprobe, efSearch)",
                    "Reduce vector quantization if using compressed indices",
                    "Implement query expansion or reranking"
                ])
            
            # Analyze memory usage
            if current_metrics.memory_usage_mb > target_metrics.get("max_memory_mb", 1000):
                recommendations.extend([
                    "Use compressed index types (PQ, SQ)",
                    "Implement index sharding across multiple nodes",
                    "Use memory-mapped files for large indices",
                    "Implement LRU cache eviction policies"
                ])
            
            return recommendations
            
        except Exception as e:
            self._logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to analysis error"]
