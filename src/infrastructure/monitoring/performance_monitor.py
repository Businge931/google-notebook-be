"""
Performance Monitoring System

"""
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import logging
import time
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import json
from enum import Enum


class MetricType(str, Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(str, Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data structure following Single Responsibility Principle."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str]
    unit: str = ""
    description: str = ""


@dataclass
class AlertRule:
    """Alert rule configuration following Single Responsibility Principle."""
    name: str
    metric_name: str
    condition: str  # e.g., "> 1000", "< 0.5"
    threshold: float
    level: AlertLevel
    duration_seconds: int = 60
    message_template: str = ""


@dataclass
class Alert:
    """Alert instance following Single Responsibility Principle."""
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False


class MetricsCollector(ABC):
    """Abstract interface for metrics collection following Interface Segregation Principle."""
    
    @abstractmethod
    async def collect_metric(self, metric: PerformanceMetric) -> None:
        """Collect a performance metric."""
        pass
    
    @abstractmethod
    async def get_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        time_range: Optional[tuple] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[PerformanceMetric]:
        """Get collected metrics."""
        pass


class AlertManager(ABC):
    """Abstract interface for alert management following Interface Segregation Principle."""
    
    @abstractmethod
    async def check_alerts(self, metrics: List[PerformanceMetric]) -> List[Alert]:
        """Check metrics against alert rules."""
        pass
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> None:
        """Send alert notification."""
        pass


class PerformanceTracker:
    """
    Performance tracker for measuring operation performance.
    
    Follows Single Responsibility Principle by focusing on performance measurement.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize performance tracker.
        
        Args:
            metrics_collector: Metrics collection service
        """
        self._metrics_collector = metrics_collector
        self._logger = logging.getLogger(__name__)
        self._active_timers = {}
    
    @asynccontextmanager
    async def track_operation(
        self,
        operation_name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for tracking operation performance.
        
        Args:
            operation_name: Name of the operation
            labels: Additional labels for the metric
        """
        start_time = time.time()
        operation_labels = labels or {}
        operation_labels["operation"] = operation_name
        
        try:
            yield
            # Success - record timing
            duration_ms = (time.time() - start_time) * 1000
            
            await self._metrics_collector.collect_metric(
                PerformanceMetric(
                    name="operation_duration_ms",
                    value=duration_ms,
                    metric_type=MetricType.HISTOGRAM,
                    timestamp=datetime.utcnow(),
                    labels=operation_labels,
                    unit="milliseconds",
                    description=f"Duration of {operation_name} operation"
                )
            )
            
            await self._metrics_collector.collect_metric(
                PerformanceMetric(
                    name="operation_success_count",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    timestamp=datetime.utcnow(),
                    labels=operation_labels,
                    description=f"Successful {operation_name} operations"
                )
            )
            
        except Exception as e:
            # Error - record failure
            duration_ms = (time.time() - start_time) * 1000
            error_labels = {**operation_labels, "error_type": type(e).__name__}
            
            await self._metrics_collector.collect_metric(
                PerformanceMetric(
                    name="operation_duration_ms",
                    value=duration_ms,
                    metric_type=MetricType.HISTOGRAM,
                    timestamp=datetime.utcnow(),
                    labels=error_labels,
                    unit="milliseconds",
                    description=f"Duration of failed {operation_name} operation"
                )
            )
            
            await self._metrics_collector.collect_metric(
                PerformanceMetric(
                    name="operation_error_count",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    timestamp=datetime.utcnow(),
                    labels=error_labels,
                    description=f"Failed {operation_name} operations"
                )
            )
            
            raise
    
    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        unit: str = "",
        description: str = ""
    ) -> None:
        """
        Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Additional labels
            unit: Unit of measurement
            description: Metric description
        """
        try:
            metric = PerformanceMetric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.utcnow(),
                labels=labels or {},
                unit=unit,
                description=description
            )
            
            await self._metrics_collector.collect_metric(metric)
            
        except Exception as e:
            self._logger.error(f"Failed to record metric {name}: {e}")


class InMemoryMetricsCollector(MetricsCollector):
    """
    In-memory metrics collector implementation.
    
    Follows Single Responsibility Principle by focusing on metric storage and retrieval.
    """
    
    def __init__(self, max_metrics: int = 10000):
        """
        Initialize in-memory metrics collector.
        
        Args:
            max_metrics: Maximum number of metrics to store
        """
        self._metrics = []
        self._max_metrics = max_metrics
        self._logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def collect_metric(self, metric: PerformanceMetric) -> None:
        """
        Collect a performance metric.
        
        Args:
            metric: Performance metric to collect
        """
        try:
            async with self._lock:
                self._metrics.append(metric)
                
                # Maintain size limit
                if len(self._metrics) > self._max_metrics:
                    # Remove oldest metrics
                    self._metrics = self._metrics[-self._max_metrics:]
                
                self._logger.debug(f"Collected metric: {metric.name}={metric.value}")
                
        except Exception as e:
            self._logger.error(f"Failed to collect metric: {e}")
    
    async def get_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        time_range: Optional[tuple] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[PerformanceMetric]:
        """
        Get collected metrics.
        
        Args:
            metric_names: Filter by metric names
            time_range: Filter by time range (start, end)
            labels: Filter by labels
            
        Returns:
            Filtered metrics
        """
        try:
            async with self._lock:
                filtered_metrics = self._metrics.copy()
            
            # Filter by metric names
            if metric_names:
                filtered_metrics = [
                    m for m in filtered_metrics 
                    if m.name in metric_names
                ]
            
            # Filter by time range
            if time_range:
                start_time, end_time = time_range
                filtered_metrics = [
                    m for m in filtered_metrics
                    if start_time <= m.timestamp <= end_time
                ]
            
            # Filter by labels
            if labels:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if all(
                        m.labels.get(key) == value
                        for key, value in labels.items()
                    )
                ]
            
            return filtered_metrics
            
        except Exception as e:
            self._logger.error(f"Failed to get metrics: {e}")
            return []


class SimpleAlertManager(AlertManager):
    """
    Simple alert manager implementation.
    
    Follows Single Responsibility Principle by focusing on alert processing.
    """
    
    def __init__(self, alert_rules: List[AlertRule]):
        """
        Initialize simple alert manager.
        
        Args:
            alert_rules: List of alert rules
        """
        self._alert_rules = {rule.name: rule for rule in alert_rules}
        self._active_alerts = {}
        self._logger = logging.getLogger(__name__)
    
    async def check_alerts(self, metrics: List[PerformanceMetric]) -> List[Alert]:
        """
        Check metrics against alert rules.
        
        Args:
            metrics: Metrics to check
            
        Returns:
            Triggered alerts
        """
        try:
            triggered_alerts = []
            
            # Group metrics by name
            metrics_by_name = {}
            for metric in metrics:
                if metric.name not in metrics_by_name:
                    metrics_by_name[metric.name] = []
                metrics_by_name[metric.name].append(metric)
            
            # Check each alert rule
            for rule_name, rule in self._alert_rules.items():
                if rule.metric_name not in metrics_by_name:
                    continue
                
                # Get recent metrics for this rule
                recent_metrics = [
                    m for m in metrics_by_name[rule.metric_name]
                    if m.timestamp >= datetime.utcnow() - timedelta(seconds=rule.duration_seconds)
                ]
                
                if not recent_metrics:
                    continue
                
                # Calculate aggregate value (average for now)
                avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
                
                # Check condition
                if self._evaluate_condition(avg_value, rule.condition, rule.threshold):
                    alert = Alert(
                        rule_name=rule_name,
                        metric_name=rule.metric_name,
                        current_value=avg_value,
                        threshold=rule.threshold,
                        level=rule.level,
                        message=rule.message_template.format(
                            value=avg_value,
                            threshold=rule.threshold,
                            metric=rule.metric_name
                        ) if rule.message_template else f"{rule.metric_name} is {avg_value}, threshold: {rule.threshold}",
                        timestamp=datetime.utcnow()
                    )
                    
                    triggered_alerts.append(alert)
                    self._active_alerts[rule_name] = alert
                    
                    self._logger.warning(f"Alert triggered: {alert.message}")
                elif rule_name in self._active_alerts:
                    # Alert resolved
                    resolved_alert = self._active_alerts[rule_name]
                    resolved_alert.resolved = True
                    triggered_alerts.append(resolved_alert)
                    del self._active_alerts[rule_name]
                    
                    self._logger.info(f"Alert resolved: {rule_name}")
            
            return triggered_alerts
            
        except Exception as e:
            self._logger.error(f"Alert checking failed: {e}")
            return []
    
    async def send_alert(self, alert: Alert) -> None:
        """
        Send alert notification.
        
        Args:
            alert: Alert to send
        """
        try:
            # For now, just log the alert
            # In production, this would send to notification systems
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }.get(alert.level, logging.WARNING)
            
            self._logger.log(
                log_level,
                f"ALERT [{alert.level.upper()}] {alert.rule_name}: {alert.message}"
            )
            
        except Exception as e:
            self._logger.error(f"Failed to send alert: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        try:
            if condition == ">":
                return value > threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<":
                return value < threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return abs(value - threshold) < 0.001  # Float comparison
            elif condition == "!=":
                return abs(value - threshold) >= 0.001
            else:
                self._logger.warning(f"Unknown condition: {condition}")
                return False
        except Exception:
            return False


class ChatPerformanceMonitor:
    """
    Chat-specific performance monitor.
    
    Follows Single Responsibility Principle by focusing on chat performance metrics.
    """
    
    def __init__(self, performance_tracker: PerformanceTracker):
        """
        Initialize chat performance monitor.
        
        Args:
            performance_tracker: Performance tracking service
        """
        self._tracker = performance_tracker
        self._logger = logging.getLogger(__name__)
    
    async def track_chat_session_start(self, session_id: str, user_id: str = None) -> None:
        """Track chat session start."""
        labels = {"session_id": session_id}
        if user_id:
            labels["user_id"] = user_id
        
        await self._tracker.record_metric(
            name="chat_session_started",
            value=1,
            metric_type=MetricType.COUNTER,
            labels=labels,
            description="Chat sessions started"
        )
    
    async def track_message_processing(
        self,
        session_id: str,
        message_length: int,
        processing_time_ms: float,
        success: bool = True,
        error_type: str = None
    ) -> None:
        """Track message processing performance."""
        labels = {
            "session_id": session_id,
            "success": str(success)
        }
        
        if error_type:
            labels["error_type"] = error_type
        
        # Record processing time
        await self._tracker.record_metric(
            name="chat_message_processing_time_ms",
            value=processing_time_ms,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            unit="milliseconds",
            description="Chat message processing time"
        )
        
        # Record message length
        await self._tracker.record_metric(
            name="chat_message_length",
            value=message_length,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            unit="characters",
            description="Chat message length"
        )
        
        # Record success/failure count
        counter_name = "chat_message_success" if success else "chat_message_error"
        await self._tracker.record_metric(
            name=counter_name,
            value=1,
            metric_type=MetricType.COUNTER,
            labels=labels,
            description=f"Chat message {'successes' if success else 'errors'}"
        )
    
    async def track_streaming_performance(
        self,
        session_id: str,
        chunks_sent: int,
        total_time_ms: float,
        bytes_sent: int
    ) -> None:
        """Track streaming response performance."""
        labels = {"session_id": session_id}
        
        await self._tracker.record_metric(
            name="chat_streaming_chunks",
            value=chunks_sent,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            description="Number of streaming chunks sent"
        )
        
        await self._tracker.record_metric(
            name="chat_streaming_time_ms",
            value=total_time_ms,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            unit="milliseconds",
            description="Total streaming time"
        )
        
        await self._tracker.record_metric(
            name="chat_streaming_bytes",
            value=bytes_sent,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            unit="bytes",
            description="Bytes sent in streaming response"
        )


class SearchPerformanceMonitor:
    """
    Search-specific performance monitor.
    
    Follows Single Responsibility Principle by focusing on search performance metrics.
    """
    
    def __init__(self, performance_tracker: PerformanceTracker):
        """
        Initialize search performance monitor.
        
        Args:
            performance_tracker: Performance tracking service
        """
        self._tracker = performance_tracker
        self._logger = logging.getLogger(__name__)
    
    async def track_search_request(
        self,
        search_type: str,
        query_length: int,
        results_count: int,
        processing_time_ms: float,
        cache_hit: bool = False,
        success: bool = True,
        error_type: str = None
    ) -> None:
        """Track search request performance."""
        labels = {
            "search_type": search_type,
            "cache_hit": str(cache_hit),
            "success": str(success)
        }
        
        if error_type:
            labels["error_type"] = error_type
        
        # Record processing time
        await self._tracker.record_metric(
            name="search_processing_time_ms",
            value=processing_time_ms,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            unit="milliseconds",
            description="Search processing time"
        )
        
        # Record query length
        await self._tracker.record_metric(
            name="search_query_length",
            value=query_length,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            unit="characters",
            description="Search query length"
        )
        
        # Record results count
        await self._tracker.record_metric(
            name="search_results_count",
            value=results_count,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            description="Number of search results returned"
        )
        
        # Record search count
        await self._tracker.record_metric(
            name="search_requests_total",
            value=1,
            metric_type=MetricType.COUNTER,
            labels=labels,
            description="Total search requests"
        )
    
    async def track_vector_operation(
        self,
        operation_type: str,
        vector_count: int,
        processing_time_ms: float,
        memory_usage_mb: float = None
    ) -> None:
        """Track vector operation performance."""
        labels = {"operation_type": operation_type}
        
        await self._tracker.record_metric(
            name="vector_operation_time_ms",
            value=processing_time_ms,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            unit="milliseconds",
            description="Vector operation processing time"
        )
        
        await self._tracker.record_metric(
            name="vector_operation_count",
            value=vector_count,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            description="Number of vectors processed"
        )
        
        if memory_usage_mb is not None:
            await self._tracker.record_metric(
                name="vector_memory_usage_mb",
                value=memory_usage_mb,
                metric_type=MetricType.GAUGE,
                labels=labels,
                unit="megabytes",
                description="Memory usage for vector operations"
            )
    
    async def track_citation_extraction(
        self,
        extraction_method: str,
        citations_found: int,
        processing_time_ms: float,
        accuracy_score: float = None
    ) -> None:
        """Track citation extraction performance."""
        labels = {"extraction_method": extraction_method}
        
        await self._tracker.record_metric(
            name="citation_extraction_time_ms",
            value=processing_time_ms,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            unit="milliseconds",
            description="Citation extraction processing time"
        )
        
        await self._tracker.record_metric(
            name="citations_extracted_count",
            value=citations_found,
            metric_type=MetricType.HISTOGRAM,
            labels=labels,
            description="Number of citations extracted"
        )
        
        if accuracy_score is not None:
            await self._tracker.record_metric(
                name="citation_extraction_accuracy",
                value=accuracy_score,
                metric_type=MetricType.GAUGE,
                labels=labels,
                description="Citation extraction accuracy score"
            )


class PerformanceMonitoringService:
    """
    Main performance monitoring service orchestrating all monitoring components.
    
    Follows Single Responsibility Principle by coordinating monitoring services.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        performance_tracker: PerformanceTracker
    ):
        """
        Initialize performance monitoring service.
        
        Args:
            metrics_collector: Metrics collection service
            alert_manager: Alert management service
            performance_tracker: Performance tracking service
        """
        self._metrics_collector = metrics_collector
        self._alert_manager = alert_manager
        self._performance_tracker = performance_tracker
        self._chat_monitor = ChatPerformanceMonitor(performance_tracker)
        self._search_monitor = SearchPerformanceMonitor(performance_tracker)
        self._logger = logging.getLogger(__name__)
        self._monitoring_task = None
    
    async def start_monitoring(self, check_interval_seconds: int = 60) -> None:
        """Start continuous monitoring."""
        try:
            self._logger.info("Starting performance monitoring")
            
            self._monitoring_task = asyncio.create_task(
                self._monitoring_loop(check_interval_seconds)
            )
            
        except Exception as e:
            self._logger.error(f"Failed to start monitoring: {e}")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        try:
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
                
                self._logger.info("Performance monitoring stopped")
                
        except Exception as e:
            self._logger.error(f"Failed to stop monitoring: {e}")
    
    async def get_performance_summary(
        self,
        time_range_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get performance summary for the specified time range."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=time_range_minutes)
            
            # Get metrics for the time range
            metrics = await self._metrics_collector.get_metrics(
                time_range=(start_time, end_time)
            )
            
            # Calculate summary statistics
            summary = {
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_minutes": time_range_minutes
                },
                "total_metrics": len(metrics),
                "chat_performance": self._calculate_chat_summary(metrics),
                "search_performance": self._calculate_search_summary(metrics),
                "system_performance": self._calculate_system_summary(metrics)
            }
            
            return summary
            
        except Exception as e:
            self._logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    @property
    def chat_monitor(self) -> ChatPerformanceMonitor:
        """Get chat performance monitor."""
        return self._chat_monitor
    
    @property
    def search_monitor(self) -> SearchPerformanceMonitor:
        """Get search performance monitor."""
        return self._search_monitor
    
    @property
    def performance_tracker(self) -> PerformanceTracker:
        """Get performance tracker."""
        return self._performance_tracker
    
    async def _monitoring_loop(self, check_interval_seconds: int) -> None:
        """Continuous monitoring loop."""
        while True:
            try:
                # Get recent metrics
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(seconds=check_interval_seconds * 2)
                
                metrics = await self._metrics_collector.get_metrics(
                    time_range=(start_time, end_time)
                )
                
                # Check for alerts
                alerts = await self._alert_manager.check_alerts(metrics)
                
                # Send alerts
                for alert in alerts:
                    await self._alert_manager.send_alert(alert)
                
                # Wait for next check
                await asyncio.sleep(check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(check_interval_seconds)
    
    def _calculate_chat_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate chat performance summary."""
        chat_metrics = [m for m in metrics if "chat" in m.name]
        
        if not chat_metrics:
            return {}
        
        # Group by metric name
        by_name = {}
        for metric in chat_metrics:
            if metric.name not in by_name:
                by_name[metric.name] = []
            by_name[metric.name].append(metric.value)
        
        summary = {}
        for name, values in by_name.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    def _calculate_search_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate search performance summary."""
        search_metrics = [m for m in metrics if "search" in m.name or "vector" in m.name or "citation" in m.name]
        
        if not search_metrics:
            return {}
        
        # Group by metric name
        by_name = {}
        for metric in search_metrics:
            if metric.name not in by_name:
                by_name[metric.name] = []
            by_name[metric.name].append(metric.value)
        
        summary = {}
        for name, values in by_name.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    def _calculate_system_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate system performance summary."""
        system_metrics = [m for m in metrics if "operation" in m.name]
        
        if not system_metrics:
            return {}
        
        success_metrics = [m for m in system_metrics if "success" in m.name]
        error_metrics = [m for m in system_metrics if "error" in m.name]
        
        total_operations = len(success_metrics) + len(error_metrics)
        success_rate = len(success_metrics) / total_operations if total_operations > 0 else 0
        
        return {
            "total_operations": total_operations,
            "successful_operations": len(success_metrics),
            "failed_operations": len(error_metrics),
            "success_rate": success_rate
        }
