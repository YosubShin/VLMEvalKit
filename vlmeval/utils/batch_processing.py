"""
Batch processing utilities for VLMEvalKit.

This module provides smart batch collection and processing capabilities
for VLLM-enabled models to improve inference throughput.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BatchItem:
    """Represents a single item in a batch."""
    index: int
    message: List[Dict[str, Any]]
    dataset: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BatchCollector:
    """Smart batch collector that accumulates samples for efficient processing."""
    
    def __init__(self, 
                 max_batch_size: int = 4,
                 batch_timeout: float = 5.0,
                 enable_smart_batching: bool = True,
                 verbose: bool = False):
        """Initialize BatchCollector.
        
        Args:
            max_batch_size: Maximum number of items per batch
            batch_timeout: Maximum time to wait for batch completion (seconds)
            enable_smart_batching: Whether to use intelligent batching heuristics
            verbose: Enable verbose logging
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.enable_smart_batching = enable_smart_batching
        self.verbose = verbose
        
        # Internal state
        self._current_batch: List[BatchItem] = []
        self._batch_start_time: Optional[float] = None
        self._total_collected = 0
        self._total_batches_sent = 0
        
        # Smart batching state
        self._dataset_batches: Dict[str, List[BatchItem]] = defaultdict(list)
        
    def add_item(self, index: int, message: List[Dict[str, Any]], dataset: str, 
                 metadata: Dict[str, Any] = None) -> Optional[List[BatchItem]]:
        """Add an item to the collector.
        
        Args:
            index: Unique identifier for the item
            message: The message to process
            dataset: Dataset name
            metadata: Additional metadata
            
        Returns:
            A batch ready for processing, or None if still collecting
        """
        item = BatchItem(index=index, message=message, dataset=dataset, metadata=metadata)
        self._total_collected += 1
        
        if self.enable_smart_batching:
            return self._add_item_smart(item)
        else:
            return self._add_item_simple(item)
    
    def _add_item_simple(self, item: BatchItem) -> Optional[List[BatchItem]]:
        """Simple batching: just accumulate until max size or timeout."""
        if not self._current_batch:
            self._batch_start_time = time.time()
        
        self._current_batch.append(item)
        
        # Check if batch is full
        if len(self._current_batch) >= self.max_batch_size:
            return self._finalize_current_batch()
        
        # Check if timeout reached
        if self._batch_start_time and (time.time() - self._batch_start_time) >= self.batch_timeout:
            return self._finalize_current_batch()
        
        return None
    
    def _add_item_smart(self, item: BatchItem) -> Optional[List[BatchItem]]:
        """Smart batching: group by dataset and similar characteristics."""
        dataset = item.dataset
        
        # Add to dataset-specific batch
        self._dataset_batches[dataset].append(item)
        
        # Check if any dataset batch is ready
        for ds, batch in self._dataset_batches.items():
            if len(batch) >= self.max_batch_size:
                # Remove and return this batch
                ready_batch = batch[:self.max_batch_size]
                self._dataset_batches[ds] = batch[self.max_batch_size:]
                self._total_batches_sent += 1
                
                if self.verbose:
                    logging.info(f"Smart batch ready: {len(ready_batch)} items from dataset {ds}")
                
                return ready_batch
        
        # Check for timeout-based batching
        oldest_time = float('inf')
        oldest_dataset = None
        
        for ds, batch in self._dataset_batches.items():
            if batch and batch[0].metadata.get('add_time', 0) < oldest_time:
                oldest_time = batch[0].metadata.get('add_time', 0)
                oldest_dataset = ds
        
        if (oldest_dataset and 
            oldest_time > 0 and 
            (time.time() - oldest_time) >= self.batch_timeout and
            self._dataset_batches[oldest_dataset]):
            
            # Return the oldest partial batch
            ready_batch = self._dataset_batches[oldest_dataset]
            del self._dataset_batches[oldest_dataset]
            self._total_batches_sent += 1
            
            if self.verbose:
                logging.info(f"Timeout batch ready: {len(ready_batch)} items from dataset {oldest_dataset}")
            
            return ready_batch
        
        # Store timestamp for timeout tracking
        item.metadata['add_time'] = time.time()
        
        return None
    
    def _finalize_current_batch(self) -> List[BatchItem]:
        """Finalize and return the current batch."""
        batch = self._current_batch
        self._current_batch = []
        self._batch_start_time = None
        self._total_batches_sent += 1
        
        if self.verbose:
            logging.info(f"Batch finalized: {len(batch)} items")
        
        return batch
    
    def flush_all(self) -> List[List[BatchItem]]:
        """Flush all remaining items as batches.
        
        Returns:
            List of batches ready for processing
        """
        batches = []
        
        if self.enable_smart_batching:
            # Flush all dataset batches
            for dataset, batch in self._dataset_batches.items():
                if batch:
                    batches.append(batch)
                    self._total_batches_sent += 1
            self._dataset_batches.clear()
        else:
            # Flush current batch if any
            if self._current_batch:
                batches.append(self._finalize_current_batch())
        
        if self.verbose and batches:
            total_items = sum(len(batch) for batch in batches)
            logging.info(f"Flushed {len(batches)} batches with {total_items} total items")
        
        return batches
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        pending_items = len(self._current_batch)
        if self.enable_smart_batching:
            pending_items = sum(len(batch) for batch in self._dataset_batches.values())
        
        return {
            'total_collected': self._total_collected,
            'total_batches_sent': self._total_batches_sent,
            'pending_items': pending_items,
            'avg_batch_size': self._total_collected / max(1, self._total_batches_sent),
            'smart_batching_enabled': self.enable_smart_batching
        }
    
    def is_empty(self) -> bool:
        """Check if collector has no pending items."""
        if self.enable_smart_batching:
            return all(len(batch) == 0 for batch in self._dataset_batches.values())
        else:
            return len(self._current_batch) == 0


class BatchProcessor:
    """Processes batches using VLLM-enabled models."""
    
    def __init__(self, model, verbose: bool = False):
        """Initialize BatchProcessor.
        
        Args:
            model: The model instance (should support batch processing)
            verbose: Enable verbose logging
        """
        self.model = model
        self.verbose = verbose
        
        # Check if model supports batching
        self.supports_batching = getattr(model, 'supports_batch_processing', lambda: False)()
        
        if not self.supports_batching:
            logging.warning(f"Model {type(model).__name__} does not support batch processing, will use sequential")
    
    def process_batch(self, batch: List[BatchItem]) -> List[Tuple[int, str]]:
        """Process a batch of items.
        
        Args:
            batch: List of BatchItem objects to process
            
        Returns:
            List of (index, result) tuples in the same order as input
        """
        if not batch:
            return []
        
        if self.supports_batching and hasattr(self.model, 'generate_batch_vllm'):
            return self._process_batch_vllm(batch)
        else:
            return self._process_batch_sequential(batch)
    
    def _process_batch_vllm(self, batch: List[BatchItem]) -> List[Tuple[int, str]]:
        """Process batch using VLLM batch generation."""
        try:
            # Extract messages and dataset info
            messages = [item.message for item in batch]
            dataset = batch[0].dataset  # Assume same dataset for smart batching
            
            if self.verbose:
                logging.info(f"Processing VLLM batch: {len(batch)} items from dataset {dataset}")
            
            # Generate batch results
            results = self.model.generate_batch_vllm(messages, dataset=dataset)
            
            # Pair with indices
            indexed_results = [(batch[i].index, results[i]) for i in range(len(batch))]
            
            return indexed_results
            
        except Exception as e:
            logging.error(f"VLLM batch processing failed: {e}, falling back to sequential")
            return self._process_batch_sequential(batch)
    
    def _process_batch_sequential(self, batch: List[BatchItem]) -> List[Tuple[int, str]]:
        """Process batch sequentially as fallback."""
        results = []
        
        for item in batch:
            try:
                if hasattr(self.model, 'generate'):
                    result = self.model.generate(message=item.message, dataset=item.dataset)
                elif hasattr(self.model, 'generate_inner'):
                    result = self.model.generate_inner(item.message, dataset=item.dataset)
                else:
                    result = "ERROR: No generation method found"
                
                results.append((item.index, result))
                
            except Exception as e:
                logging.error(f"Sequential processing failed for item {item.index}: {e}")
                results.append((item.index, "ERROR: Generation failed"))
        
        return results


def estimate_batch_processing_benefit(dataset_size: int, avg_batch_size: float = 3.0) -> Dict[str, float]:
    """Estimate the potential speedup from batch processing.
    
    Args:
        dataset_size: Total number of items to process
        avg_batch_size: Expected average batch size
        
    Returns:
        Dictionary with speedup estimates
    """
    if dataset_size <= 0:
        return {'speedup': 1.0, 'time_saved_percent': 0.0}
    
    # Conservative estimates based on VLLM batching efficiency
    # Assumes batch processing overhead of ~20% but 60-80% per-item speedup
    batch_overhead = 0.2
    per_item_speedup = 0.7  # 70% faster per item when batched
    
    sequential_time = dataset_size  # Normalized to 1 unit per item
    
    num_batches = dataset_size / avg_batch_size
    batch_time = num_batches * (1 + batch_overhead) * avg_batch_size * (1 - per_item_speedup)
    
    speedup = sequential_time / batch_time
    time_saved_percent = (1 - 1/speedup) * 100
    
    return {
        'speedup': round(speedup, 2),
        'time_saved_percent': round(time_saved_percent, 1),
        'estimated_batches': int(num_batches),
        'avg_batch_size': avg_batch_size
    }