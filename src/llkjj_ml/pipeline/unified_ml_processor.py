"""
UnifiedMLProcessor - Consolidation of all ML processing implementations.

This module provides a single entry point for all PDF processing strategies,
replacing GeminiDirectProcessor, AsyncGeminiDirectProcessor, GeminiFirstProcessor,
and other duplicate implementations.

Architecture:
- Strategy Pattern for algorithm selection
- Chain of Responsibility for error handling
- Observer Pattern for progress updates
- Factory Pattern for strategy creation
- Memory management with automatic cleanup

Performance targets:
- Single PDF: < 2 seconds average
- Batch of 100: < 120 seconds
- Memory: < 2GB peak usage
- Cache hit: < 100ms response

Author: LLKJJ Development Team
Version: 1.0.0
License: Proprietary
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Literal, Optional, Union
from datetime import datetime
import json
import redis
from pydantic import BaseModel, Field

from ..models.processing_result import ProcessingResult
from ..settings_bridge import ConfigBridge, config_instance
from ..optimization.batch_memory_optimizer import get_global_optimizer

# Import available processors with error handling
try:
    from ..gemini.direct_processor import GeminiDirectProcessor
except ImportError:
    GeminiDirectProcessor = None

try:
    from ..extraction.docling_processor import AdvancedDoclingProcessor as DoclingProcessor
except ImportError:
    DoclingProcessor = None

try:
    from ..pipeline.spacy_rag_strategy import SpacyRagStrategy
except ImportError:
    SpacyRagStrategy = None

try:
    from ..pipeline.gemini_strategy import GeminiStrategy
except ImportError:
    GeminiStrategy = None

logger = logging.getLogger(__name__)

# Type aliases for better readability
ProcessingStrategy = Literal["auto", "gemini_first", "spacy_rag", "docling", "hybrid"]
ErrorLevel = Literal["warning", "error", "critical"]


class ProcessingOptions(BaseModel):
    """Configuration options for single PDF processing."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    file_size_limit_mb: int = Field(default=50, description="Maximum file size in MB")
    timeout_seconds: int = Field(default=120, description="Processing timeout")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    progress_callback: Optional[callable] = Field(default=None, description="Progress update callback", exclude=True)
    retry_count: int = Field(default=2, description="Number of retries on failure")
    quality_threshold: float = Field(default=0.7, description="Minimum quality threshold")


class BatchOptions(BaseModel):
    """Configuration options for batch processing."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    max_concurrent: int = Field(default=5, description="Maximum concurrent processes")
    memory_limit_gb: float = Field(default=2.0, description="Memory limit for batch")
    progress_callback: Optional[callable] = Field(default=None, description="Batch progress callback", exclude=True)
    fail_fast: bool = Field(default=False, description="Stop on first failure")
    chunk_size: int = Field(default=10, description="Number of files per chunk")


class ProcessorConfig(BaseModel):
    """Unified configuration for all processing strategies."""
    
    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model version")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")
    memory_optimization: bool = Field(default=True, description="Enable memory optimization")
    performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")


class BatchResult(BaseModel):
    """Results from batch processing operation."""
    
    total_files: int = Field(description="Total number of files processed")
    successful: int = Field(description="Number of successful processes")
    failed: int = Field(description="Number of failed processes")
    total_time_ms: int = Field(description="Total processing time")
    average_time_ms: float = Field(description="Average processing time per file")
    memory_peak_mb: float = Field(description="Peak memory usage")
    results: List[ProcessingResult] = Field(description="Individual processing results")
    errors: List[Dict[str, Any]] = Field(description="Error details for failed files")
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful / self.total_files) * 100


class CacheManager:
    """Redis-based caching for processing results."""
    
    def __init__(self, redis_url: str, ttl_hours: int = 24):
        """Initialize cache manager with Redis connection."""
        try:
            self.redis_client = redis.from_url(redis_url)
            self.ttl_seconds = ttl_hours * 3600
            logger.info("✅ Redis cache manager initialized")
        except Exception as e:
            logger.warning(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, pdf_path: Path, strategy: str, options: ProcessingOptions) -> str:
        """Generate SHA256 cache key from file content and options."""
        try:
            # Read file content for hash
            with open(pdf_path, 'rb') as f:
                file_content = f.read()
            
            # Create cache key components
            file_hash = hashlib.sha256(file_content).hexdigest()
            options_hash = hashlib.sha256(
                json.dumps({
                    "strategy": strategy,
                    "file_size_limit_mb": options.file_size_limit_mb,
                    "quality_threshold": options.quality_threshold
                }, sort_keys=True).encode()
            ).hexdigest()[:8]
            
            return f"llkjj:ml:v1:{file_hash}:{strategy}:{options_hash}"
            
        except Exception as e:
            logger.warning(f"Cache key generation failed: {e}")
            return f"fallback:{pdf_path.name}:{strategy}:{int(time.time())}"
    
    def get(self, cache_key: str) -> Optional[ProcessingResult]:
        """Retrieve cached result."""
        if not self.redis_client:
            return None
            
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                result_dict = json.loads(cached_data)
                logger.debug(f"✅ Cache hit for key: {cache_key[:16]}...")
                return ProcessingResult(**result_dict)
                
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def set(self, cache_key: str, result: ProcessingResult) -> bool:
        """Store result in cache."""
        if not self.redis_client:
            return False
            
        try:
            result_json = json.dumps(result.to_dict())
            self.redis_client.setex(cache_key, self.ttl_seconds, result_json)
            logger.debug(f"✅ Cached result for key: {cache_key[:16]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        if not self.redis_client:
            return 0
            
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"🗑️ Invalidated {deleted} cache entries")
                return deleted
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
            
        return 0


class MemoryManager:
    """Advanced memory management with automatic cleanup."""
    
    def __init__(self, memory_limit_gb: float = 2.0):
        """Initialize memory manager with limit."""
        self.memory_limit_gb = memory_limit_gb
        self.optimizer = get_global_optimizer()
        self.peak_usage_mb = 0.0
        
    def check_memory_availability(self, required_mb: float) -> bool:
        """Check if enough memory is available for operation."""
        memory_status = self.optimizer.get_memory_status()
        available_mb = memory_status.available_memory_gb * 1024
        
        if available_mb < required_mb:
            logger.warning(f"⚠️ Insufficient memory: {available_mb:.1f}MB available, {required_mb:.1f}MB required")
            return False
            
        return True
    
    def optimize_if_needed(self) -> None:
        """Trigger memory optimization if usage is high."""
        memory_status = self.optimizer.get_memory_status()
        
        if memory_status.memory_percent > 85.0:
            logger.info("🧹 Triggering memory cleanup (usage > 85%)")
            result = self.optimizer.optimize_memory()
            logger.info(f"✅ Freed {result.memory_freed_mb:.1f}MB in {result.optimization_time_ms}ms")
    
    def update_peak_usage(self) -> None:
        """Track peak memory usage."""
        memory_status = self.optimizer.get_memory_status()
        current_usage = memory_status.used_memory_gb * 1024
        
        if current_usage > self.peak_usage_mb:
            self.peak_usage_mb = current_usage
            
    def get_memory_metrics(self) -> Dict[str, float]:
        """Get current memory metrics."""
        memory_status = self.optimizer.get_memory_status()
        
        return {
            "current_usage_mb": memory_status.used_memory_gb * 1024,
            "peak_usage_mb": self.peak_usage_mb,
            "available_mb": memory_status.available_memory_gb * 1024,
            "usage_percent": memory_status.memory_percent
        }


class UnifiedMLProcessor:
    """
    Unified ML Processor - Single entry point for all PDF processing strategies.
    
    This processor consolidates GeminiDirectProcessor, AsyncGeminiDirectProcessor,
    GeminiFirstProcessor, and other implementations into a single, consistent API.
    
    Features:
    - Strategy pattern with automatic selection
    - Fallback chain for error recovery
    - Performance monitoring and optimization
    - Redis caching with SHA256 keys
    - Memory management with cleanup
    - Async processing support
    - Batch processing with memory limits
    
    Example usage:
        processor = UnifiedMLProcessor(
            strategy="auto",
            config=ProcessorConfig(gemini_model="gemini-2.5-flash")
        )
        
        # Single PDF processing
        result = processor.process_pdf(
            pdf_path=Path("invoice.pdf"),
            options=ProcessingOptions(cache_enabled=True)
        )
        
        # Batch processing
        results = processor.process_batch(
            pdf_paths=[Path("file1.pdf"), Path("file2.pdf")],
            options=BatchOptions(max_concurrent=5)
        )
        
        # Async processing
        result = await processor.process_async(Path("invoice.pdf"))
    """
    
    def __init__(
        self,
        strategy: ProcessingStrategy = "auto",
        config: Optional[ProcessorConfig] = None,
        cache_manager: Optional[CacheManager] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        """
        Initialize UnifiedMLProcessor with strategy and configuration.
        
        Args:
            strategy: Processing strategy ("auto", "gemini_first", "spacy_rag", "docling", "hybrid")
            config: Processor configuration options
            cache_manager: Optional custom cache manager
            memory_manager: Optional custom memory manager
        """
        # Configuration
        self.config = config or ProcessorConfig()
        self.default_strategy = strategy
        
        # Initialize managers
        self.cache_manager = cache_manager or CacheManager(
            redis_url=self.config.redis_url,
            ttl_hours=self.config.cache_ttl_hours
        )
        self.memory_manager = memory_manager or MemoryManager()
        
        # Initialize batch memory optimizer
        self._batch_optimizer = get_global_optimizer()
        
        # Strategy implementations
        self._strategies: Dict[str, Any] = {}
        self._fallback_chain: List[str] = ["gemini_first", "docling", "spacy_rag"]
        
        # Performance tracking
        self._metrics: Dict[str, Any] = {
            "total_processed": 0,
            "cache_hits": 0,
            "fallback_uses": 0,
            "average_processing_time_ms": 0.0
        }
        
        # Initialize available strategies
        self._initialize_strategies()
        
        logger.info(f"🚀 UnifiedMLProcessor initialized with strategy: {strategy}")
        logger.info(f"📊 Available strategies: {list(self._strategies.keys())}")
    
    def _initialize_strategies(self) -> None:
        """Initialize all available processing strategies."""
        # Initialize Gemini strategy
        if GeminiDirectProcessor is not None:
            try:
                self._strategies["gemini_first"] = GeminiDirectProcessor(
                    config=self.config.gemini_api_key
                )
                logger.debug("✅ Gemini strategy initialized")
            except Exception as e:
                logger.warning(f"❌ Gemini strategy initialization failed: {e}")
        else:
            logger.warning("❌ GeminiDirectProcessor not available")
        
        # Initialize Docling strategy  
        if DoclingProcessor is not None:
            try:
                self._strategies["docling"] = DoclingProcessor()
                logger.debug("✅ Docling strategy initialized")
            except Exception as e:
                logger.warning(f"❌ Docling strategy initialization failed: {e}")
        else:
            logger.warning("❌ DoclingProcessor not available")
        
        # Initialize SpaCy RAG strategy
        if SpacyRagStrategy is not None:
            try:
                self._strategies["spacy_rag"] = SpacyRagStrategy(
                    config=config_instance
                )
                logger.debug("✅ SpaCy RAG strategy initialized")
            except Exception as e:
                logger.warning(f"❌ SpaCy RAG strategy initialization failed: {e}")
        else:
            logger.warning("❌ SpacyRagStrategy not available")
        
        # Initialize Gemini Strategy (alternative implementation)
        if GeminiStrategy is not None:
            try:
                self._strategies["gemini"] = GeminiStrategy(config_instance)
                logger.debug("✅ Gemini (alternative) strategy initialized")
            except Exception as e:
                logger.warning(f"❌ Gemini (alternative) strategy initialization failed: {e}")
        else:
            logger.warning("❌ GeminiStrategy not available")
        
        # Strategy availability check
        available_strategies = [name for name, strategy in self._strategies.items() 
                              if self._is_strategy_available(name)]
        
        if not available_strategies:
            logger.warning("⚠️ No processing strategies available! Running in compatibility mode.")
        else:
            logger.info(f"🎯 Operational strategies: {available_strategies}")
    
    def _is_strategy_available(self, strategy_name: str) -> bool:
        """Check if a strategy is available and functional."""
        if strategy_name not in self._strategies:
            return False
            
        strategy = self._strategies[strategy_name]
        
        # Basic availability check
        if hasattr(strategy, 'is_available'):
            return strategy.is_available()
        
        # Fallback check
        return True
    
    def _select_optimal_strategy(self, pdf_path: Path, options: ProcessingOptions) -> str:
        """
        Select optimal strategy based on file characteristics and requirements.
        
        Strategy selection logic:
        - file_size < 1MB and has_text_layer: use spacy_rag
        - requires_high_accuracy: use gemini_first
        - is_scanned_document: use docling_ocr
        - low_latency_required: use hybrid_cache
        - else: use auto_select
        """
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        
        # Small files with text layer - prefer SpaCy RAG
        if file_size_mb < 1.0 and self._is_strategy_available("spacy_rag"):
            logger.debug("🎯 Selected spacy_rag (small file, text layer)")
            return "spacy_rag"
        
        # High accuracy requirements - prefer Gemini
        if options.quality_threshold > 0.9 and self._is_strategy_available("gemini_first"):
            logger.debug("🎯 Selected gemini_first (high accuracy required)")
            return "gemini_first"
        
        # Large files or scanned documents - prefer Docling
        if file_size_mb > 10.0 and self._is_strategy_available("docling"):
            logger.debug("🎯 Selected docling (large file/scanned document)")
            return "docling"
        
        # Default to first available strategy
        for strategy_name in self._fallback_chain:
            if self._is_strategy_available(strategy_name):
                logger.debug(f"🎯 Selected {strategy_name} (default fallback)")
                return strategy_name
        
        raise RuntimeError("❌ No strategies available for processing")
    
    def process_pdf(self, pdf_path: Path, options: Optional[ProcessingOptions] = None) -> ProcessingResult:
        """
        Process single PDF file with selected strategy.
        
        Args:
            pdf_path: Path to PDF file to process
            options: Processing options (optional)
            
        Returns:
            ProcessingResult with extracted data and metadata
            
        Raises:
            ValueError: Invalid PDF path or options
            RuntimeError: Processing failed with all strategies
            MemoryError: Insufficient memory for processing
        """
        # Validate inputs
        if not pdf_path.exists():
            raise ValueError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        options = options or ProcessingOptions()
        
        # File size check
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > options.file_size_limit_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {options.file_size_limit_mb}MB limit")
        
        # Memory availability check
        estimated_memory_mb = file_size_mb * 10  # Conservative estimate
        if not self.memory_manager.check_memory_availability(estimated_memory_mb):
            raise MemoryError(f"Insufficient memory for processing {file_size_mb:.1f}MB file")
        
        # Strategy selection
        if self.default_strategy == "auto":
            selected_strategy = self._select_optimal_strategy(pdf_path, options)
        else:
            selected_strategy = self.default_strategy
        
        # Cache check
        cache_key = None
        if options.cache_enabled and self.cache_manager:
            cache_key = self.cache_manager._generate_cache_key(pdf_path, selected_strategy, options)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                self._metrics["cache_hits"] += 1
                logger.info(f"⚡ Cache hit for {pdf_path.name}")
                return cached_result
        
        # Processing with fallback chain
        start_time = time.time()
        last_error = None
        
        strategies_to_try = [selected_strategy] + [
            s for s in self._fallback_chain if s != selected_strategy
        ]
        
        for attempt, strategy_name in enumerate(strategies_to_try):
            if not self._is_strategy_available(strategy_name):
                continue
                
            try:
                logger.info(f"🔄 Processing with {strategy_name} (attempt {attempt + 1})")
                
                # Memory optimization before processing
                if self.config.memory_optimization:
                    self.memory_manager.optimize_if_needed()
                
                # Strategy-specific processing
                result = self._process_with_strategy(pdf_path, strategy_name, options)
                
                # Update processing metadata
                result.processing_method = strategy_name  # type: ignore
                processing_time_ms = int((time.time() - start_time) * 1000)
                result.processing_time_ms = processing_time_ms
                
                # Cache successful result
                if options.cache_enabled and cache_key and self.cache_manager:
                    self.cache_manager.set(cache_key, result)
                
                # Update metrics
                self._metrics["total_processed"] += 1
                if attempt > 0:
                    self._metrics["fallback_uses"] += 1
                
                # Update memory peak usage
                self.memory_manager.update_peak_usage()
                
                logger.info(f"✅ Successfully processed {pdf_path.name} with {strategy_name} "
                           f"in {processing_time_ms}ms")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"❌ {strategy_name} strategy failed: {e}")
                
                # Try next strategy in chain
                continue
        
        # All strategies failed
        processing_time_ms = int((time.time() - start_time) * 1000)
        error_message = f"All processing strategies failed. Last error: {last_error}"
        
        logger.error(f"💥 Processing failed for {pdf_path.name}: {error_message}")
        
        raise RuntimeError(error_message)
    
    def _process_with_strategy(
        self, 
        pdf_path: Path, 
        strategy_name: str, 
        options: ProcessingOptions
    ) -> ProcessingResult:
        """Process PDF with specific strategy implementation."""
        strategy = self._strategies[strategy_name]
        
        if strategy_name == "gemini_first":
            # GeminiDirectProcessor interface
            return strategy.process_pdf_gemini_first(str(pdf_path))
            
        elif strategy_name == "docling":
            # DoclingProcessor interface
            return strategy.process_pdf(pdf_path)
            
        elif strategy_name == "spacy_rag":
            # SpacyRagStrategy interface
            return strategy.process_pdf(pdf_path)
            
        else:
            # Generic interface fallback
            return strategy.process_pdf(pdf_path)
    
    def process_batch(
        self, 
        pdf_paths: List[Path], 
        options: Optional[BatchOptions] = None
    ) -> BatchResult:
        """
        Process multiple PDF files in optimized batches.
        
        Args:
            pdf_paths: List of PDF file paths to process
            options: Batch processing options
            
        Returns:
            BatchResult with aggregated results and statistics
        """
        options = options or BatchOptions()
        start_time = time.time()
        
        # Filter valid PDF files
        valid_paths = [path for path in pdf_paths if path.exists() and path.suffix.lower() == '.pdf']
        invalid_count = len(pdf_paths) - len(valid_paths)
        
        if invalid_count > 0:
            logger.warning(f"⚠️ Skipped {invalid_count} invalid files")
        
        if not valid_paths:
            return BatchResult(
                total_files=0,
                successful=0,
                failed=0,
                total_time_ms=0,
                average_time_ms=0.0,
                memory_peak_mb=0.0,
                results=[],
                errors=[]
            )
        
        logger.info(f"🚀 Starting batch processing of {len(valid_paths)} files")
        
        # Memory-optimized batch size calculation
        optimal_batch_size, batch_sizes = self._batch_optimizer.suggest_batch_size(
            item_count=len(valid_paths),
            memory_per_item_mb=50.0  # Conservative estimate per PDF
        )
        
        results: List[ProcessingResult] = []
        errors: List[Dict[str, Any]] = []
        successful = 0
        failed = 0
        
        # Process in memory-optimized chunks
        start_idx = 0
        
        for batch_num, chunk_size in enumerate(batch_sizes, 1):
            chunk_paths = valid_paths[start_idx:start_idx + chunk_size]
            
            logger.info(f"📦 Processing batch {batch_num}/{len(batch_sizes)} "
                       f"({len(chunk_paths)} files)")
            
            # Memory management for batch
            with self._batch_optimizer.memory_managed_processing():
                for file_idx, pdf_path in enumerate(chunk_paths):
                    try:
                        # Progress callback
                        if options.progress_callback:
                            progress = ((start_idx + file_idx) / len(valid_paths)) * 100
                            options.progress_callback(progress, pdf_path)
                        
                        # Process single file
                        result = self.process_pdf(
                            pdf_path, 
                            ProcessingOptions(
                                cache_enabled=True,
                                retry_count=1  # Reduced retries for batch processing
                            )
                        )
                        
                        results.append(result)
                        successful += 1
                        
                        logger.debug(f"✅ Batch item {file_idx + 1}/{len(chunk_paths)}: {pdf_path.name}")
                        
                    except Exception as e:
                        failed += 1
                        error_detail = {
                            "file_path": str(pdf_path),
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        errors.append(error_detail)
                        
                        logger.error(f"❌ Batch item failed: {pdf_path.name} - {e}")
                        
                        # Fail fast option
                        if options.fail_fast:
                            break
            
            start_idx += chunk_size
            
            # Inter-batch memory cleanup
            if batch_num < len(batch_sizes):
                cleanup_result = self._batch_optimizer.optimize_memory()
                logger.debug(f"🧹 Inter-batch cleanup: {cleanup_result.memory_freed_mb:.1f}MB freed")
        
        # Calculate final metrics
        total_time_ms = int((time.time() - start_time) * 1000)
        average_time_ms = total_time_ms / len(valid_paths) if valid_paths else 0.0
        memory_metrics = self.memory_manager.get_memory_metrics()
        
        batch_result = BatchResult(
            total_files=len(valid_paths),
            successful=successful,
            failed=failed,
            total_time_ms=total_time_ms,
            average_time_ms=average_time_ms,
            memory_peak_mb=memory_metrics["peak_usage_mb"],
            results=results,
            errors=errors
        )
        
        logger.info(f"✅ Batch processing completed: {successful}/{len(valid_paths)} successful "
                   f"({batch_result.get_success_rate():.1f}% success rate) in {total_time_ms}ms")
        
        return batch_result
    
    async def process_async(self, pdf_path: Path) -> ProcessingResult:
        """
        Process PDF asynchronously without blocking.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ProcessingResult from async processing
        """
        logger.info(f"🔄 Starting async processing of {pdf_path.name}")
        
        # Run processing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.process_pdf, 
            pdf_path,
            ProcessingOptions(cache_enabled=True)
        )
        
        logger.info(f"✅ Async processing completed for {pdf_path.name}")
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processor performance metrics."""
        memory_metrics = self.memory_manager.get_memory_metrics()
        
        return {
            "processing_metrics": self._metrics.copy(),
            "memory_metrics": memory_metrics,
            "cache_metrics": {
                "cache_enabled": self.cache_manager.redis_client is not None,
                "cache_ttl_hours": self.config.cache_ttl_hours
            },
            "strategy_metrics": {
                "available_strategies": list(self._strategies.keys()),
                "default_strategy": self.default_strategy,
                "fallback_chain": self._fallback_chain
            }
        }
    
    def invalidate_cache(self, pattern: str = "llkjj:ml:*") -> int:
        """
        Invalidate cached results matching pattern.
        
        Args:
            pattern: Redis pattern to match cache keys
            
        Returns:
            Number of invalidated cache entries
        """
        if not self.cache_manager:
            return 0
            
        invalidated = self.cache_manager.invalidate_pattern(pattern)
        logger.info(f"🗑️ Cache invalidation: {invalidated} entries removed")
        
        return invalidated
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of all processor components.
        
        Returns:
            Dict with health status of all components
        """
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Strategy health checks
        for strategy_name, strategy in self._strategies.items():
            try:
                is_available = self._is_strategy_available(strategy_name)
                health_status["components"][f"strategy_{strategy_name}"] = {
                    "status": "healthy" if is_available else "unhealthy",
                    "available": is_available
                }
            except Exception as e:
                health_status["components"][f"strategy_{strategy_name}"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Cache health check
        try:
            if self.cache_manager.redis_client:
                self.cache_manager.redis_client.ping()
                health_status["components"]["cache"] = {"status": "healthy"}
            else:
                health_status["components"]["cache"] = {"status": "disabled"}
        except Exception as e:
            health_status["components"]["cache"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Memory health check
        try:
            memory_metrics = self.memory_manager.get_memory_metrics()
            memory_status = "healthy"
            
            if memory_metrics["usage_percent"] > 90:
                memory_status = "critical"
            elif memory_metrics["usage_percent"] > 75:
                memory_status = "warning"
                
            health_status["components"]["memory"] = {
                "status": memory_status,
                "usage_percent": memory_metrics["usage_percent"],
                "available_mb": memory_metrics["available_mb"]
            }
        except Exception as e:
            health_status["components"]["memory"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Overall status determination
        component_statuses = [comp.get("status", "error") for comp in health_status["components"].values()]
        
        if "error" in component_statuses or "critical" in component_statuses:
            health_status["overall_status"] = "unhealthy"
        elif "warning" in component_statuses:
            health_status["overall_status"] = "degraded"
        
        return health_status


# Factory functions for backward compatibility
def create_unified_ml_processor(
    strategy: ProcessingStrategy = "auto",
    config: Optional[ProcessorConfig] = None
) -> UnifiedMLProcessor:
    """
    Factory function to create UnifiedMLProcessor instance.
    
    Args:
        strategy: Default processing strategy
        config: Processor configuration
        
    Returns:
        Configured UnifiedMLProcessor instance
    """
    return UnifiedMLProcessor(strategy=strategy, config=config)


# Legacy compatibility aliases
GeminiDirectProcessor = UnifiedMLProcessor  # Backward compatibility
AsyncGeminiDirectProcessor = UnifiedMLProcessor  # Backward compatibility
GeminiFirstProcessor = UnifiedMLProcessor  # Backward compatibility

logger.info("📦 UnifiedMLProcessor module loaded - consolidating 4+ duplicate processors")