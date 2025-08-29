#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Batch Memory Optimizer
==========================================

Advanced memory optimization for batch processing operations.
Reduces memory footprint and improves processing efficiency.

Features:
- Dynamic memory allocation adjustment
- Batch size optimization based on available memory
- Memory leak prevention
- Garbage collection optimization

Author: LLKJJ ML Pipeline Team
Version: 1.0.0
Date: 27. August 2025
"""

import gc
import logging
import psutil
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStatus:
    """Current memory status information."""
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    recommended_batch_size: int


@dataclass
class OptimizationResult:
    """Result of memory optimization operation."""
    memory_freed_mb: float
    garbage_collected: int
    cache_cleared: bool
    optimization_time_ms: float


class BatchMemoryOptimizer:
    """
    Advanced memory optimizer for batch processing operations.
    
    Automatically adjusts processing parameters based on available memory
    and provides optimization utilities for large-scale operations.
    """
    
    def __init__(self, target_memory_usage_percent: float = 80.0):
        """
        Initialize memory optimizer.
        
        Args:
            target_memory_usage_percent: Target memory usage percentage (default: 80%)
        """
        self.target_memory_usage = target_memory_usage_percent
        self.baseline_memory = self._get_memory_info()
        self._optimization_history: List[OptimizationResult] = []
        
        logger.info(f"🔧 BatchMemoryOptimizer initialized with {target_memory_usage_percent}% target usage")
    
    def _get_memory_info(self) -> MemoryStatus:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        
        return MemoryStatus(
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            used_memory_gb=memory.used / (1024**3),
            memory_percent=memory.percent,
            recommended_batch_size=self._calculate_optimal_batch_size(memory.available)
        )
    
    def _calculate_optimal_batch_size(self, available_memory_bytes: int) -> int:
        """Calculate optimal batch size based on available memory."""
        # Estimate memory usage per item (conservative estimate)
        memory_per_item_mb = 50  # MB per PDF processing item
        available_memory_mb = available_memory_bytes / (1024**2)
        
        # Reserve 20% of available memory for other processes
        usable_memory_mb = available_memory_mb * 0.8
        
        optimal_batch_size = int(usable_memory_mb / memory_per_item_mb)
        
        # Ensure reasonable bounds
        return max(1, min(optimal_batch_size, 100))
    
    def get_memory_status(self) -> MemoryStatus:
        """Get current memory status."""
        return self._get_memory_info()
    
    def optimize_memory(self, force_gc: bool = True) -> OptimizationResult:
        """
        Perform memory optimization.
        
        Args:
            force_gc: Whether to force garbage collection
            
        Returns:
            OptimizationResult with optimization details
        """
        import time
        start_time = time.perf_counter()
        
        memory_before = self._get_memory_info()
        
        # Force garbage collection
        gc_collected = 0
        if force_gc:
            gc_collected = gc.collect()
            logger.debug(f"🗑️ Garbage collection freed {gc_collected} objects")
        
        # Clear Python's internal caches
        cache_cleared = False
        try:
            # Clear various Python caches
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            cache_cleared = True
        except Exception as e:
            logger.warning(f"⚠️ Could not clear caches: {e}")
        
        memory_after = self._get_memory_info()
        optimization_time = (time.perf_counter() - start_time) * 1000
        
        memory_freed = memory_before.used_memory_gb - memory_after.used_memory_gb
        memory_freed_mb = memory_freed * 1024
        
        result = OptimizationResult(
            memory_freed_mb=memory_freed_mb,
            garbage_collected=gc_collected,
            cache_cleared=cache_cleared,
            optimization_time_ms=optimization_time
        )
        
        self._optimization_history.append(result)
        
        logger.info(f"✅ Memory optimization completed: {memory_freed_mb:.1f}MB freed in {optimization_time:.1f}ms")
        
        return result
    
    def suggest_batch_size(self, item_count: int, memory_per_item_mb: float = 50.0) -> Tuple[int, List[int]]:
        """
        Suggest optimal batch size and create batch indices.
        
        Args:
            item_count: Total number of items to process
            memory_per_item_mb: Estimated memory usage per item in MB
            
        Returns:
            Tuple of (optimal_batch_size, list_of_batch_sizes)
        """
        memory_status = self.get_memory_status()
        
        # Calculate optimal batch size based on current memory
        usable_memory_mb = memory_status.available_memory_gb * 1024 * 0.8
        optimal_batch_size = max(1, int(usable_memory_mb / memory_per_item_mb))
        
        # Create batch sizes list
        batch_sizes = []
        remaining_items = item_count
        
        while remaining_items > 0:
            current_batch_size = min(optimal_batch_size, remaining_items)
            batch_sizes.append(current_batch_size)
            remaining_items -= current_batch_size
        
        logger.info(f"📊 Suggested batch processing: {len(batch_sizes)} batches of ~{optimal_batch_size} items")
        
        return optimal_batch_size, batch_sizes
    
    @contextmanager
    def memory_managed_processing(self, cleanup_threshold_percent: float = 85.0):
        """
        Context manager for memory-managed processing.
        
        Args:
            cleanup_threshold_percent: Memory threshold for automatic cleanup
        """
        try:
            initial_memory = self.get_memory_status()
            logger.debug(f"🎯 Starting memory-managed processing at {initial_memory.memory_percent:.1f}% usage")
            
            yield self
            
        finally:
            current_memory = self.get_memory_status()
            
            if current_memory.memory_percent > cleanup_threshold_percent:
                logger.warning(f"⚠️ Memory usage high ({current_memory.memory_percent:.1f}%), performing cleanup")
                self.optimize_memory(force_gc=True)
            
            final_memory = self.get_memory_status()
            logger.debug(f"🏁 Memory-managed processing completed at {final_memory.memory_percent:.1f}% usage")
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get history of optimization operations."""
        return self._optimization_history.copy()
    
    def reset_optimization_history(self) -> None:
        """Reset optimization history."""
        self._optimization_history.clear()
        logger.debug("🔄 Optimization history reset")


# Convenience function for easy access
def create_memory_optimizer(target_usage_percent: float = 80.0) -> BatchMemoryOptimizer:
    """Create and return a configured BatchMemoryOptimizer."""
    return BatchMemoryOptimizer(target_usage_percent)


# Global optimizer instance for module-level access
_global_optimizer: Optional[BatchMemoryOptimizer] = None


def get_global_optimizer() -> BatchMemoryOptimizer:
    """Get or create global memory optimizer instance."""
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = create_memory_optimizer()
    
    return _global_optimizer