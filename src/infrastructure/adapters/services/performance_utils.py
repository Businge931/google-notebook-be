"""
Performance optimization utilities for AI-enhanced PDF processing.
"""
import time
import functools
from typing import Any, Callable, Dict
import logging

logger = logging.getLogger(__name__)


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"⚡ {func.__name__} completed in {processing_time:.1f}ms")
            return result
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ {func.__name__} failed after {processing_time:.1f}ms: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"⚡ {func.__name__} completed in {processing_time:.1f}ms")
            return result
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ {func.__name__} failed after {processing_time:.1f}ms: {e}")
            raise
    
    return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper


class PerformanceCache:
    """Simple in-memory cache for performance optimization."""
    
    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        if len(self._cache) >= self._max_size:
            # Remove oldest item
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_times.clear()


# Global performance cache instance
performance_cache = PerformanceCache()
