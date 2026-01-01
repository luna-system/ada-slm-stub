"""
Data Loading Utilities
======================

DRY utilities for loading and preparing training data.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from datasets import Dataset


@dataclass
class DataStats:
    """Statistics about loaded data."""
    total_examples: int
    train_examples: int
    val_examples: int
    data_types: Dict[str, int]


class DataLoader:
    """
    Handles loading and preparing training data for the harness.
    
    Supports:
    - JSONL files with {"text": ...} format
    - Automatic train/val splitting
    - Data type tracking for distribution analysis
    """
    
    def __init__(
        self,
        data_file: str,
        train_split: float = 0.9,
        shuffle: bool = True,
        seed: int = 42,
        data_dir: str = "data",
    ):
        # Check if data_file is absolute or relative
        data_path = Path(data_file)
        if not data_path.is_absolute() and not data_path.exists():
            # Try looking in data_dir
            data_path = Path(data_dir) / data_file
        self.data_file = data_path
        
        self.train_split = train_split
        self.shuffle = shuffle
        self.seed = seed
        
        self.data: List[Dict] = []
        self.train_data: List[Dict] = []
        self.val_data: List[Dict] = []
        self.stats: Optional[DataStats] = None
    
    def load(self) -> "DataLoader":
        """Load data from JSONL file."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.data = []
        with open(self.data_file) as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        print(f"📦 Loaded {len(self.data)} examples from {self.data_file.name}")
        return self
    
    def split(self) -> "DataLoader":
        """Split data into train/val sets."""
        data = self.data.copy()
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(data)
        
        split_idx = int(len(data) * self.train_split)
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
        
        # Compute stats
        data_types: Dict[str, int] = {}
        for item in self.data:
            dtype = item.get("type", "unknown")
            data_types[dtype] = data_types.get(dtype, 0) + 1
        
        self.stats = DataStats(
            total_examples=len(self.data),
            train_examples=len(self.train_data),
            val_examples=len(self.val_data),
            data_types=data_types,
        )
        
        print(f"   Train: {self.stats.train_examples}")
        print(f"   Val: {self.stats.val_examples}")
        
        if data_types and len(data_types) > 1:
            print(f"   Types: {data_types}")
        
        return self
    
    def get_datasets(self) -> tuple[Dataset, Dataset]:
        """Get HuggingFace datasets for training."""
        if not self.train_data:
            self.split()
        
        train_dataset = Dataset.from_list(self.train_data)
        val_dataset = Dataset.from_list(self.val_data)
        
        return train_dataset, val_dataset
    
    def tokenize(
        self,
        tokenizer,
        max_length: int = 512,
        text_column: str = "text",
    ) -> tuple[Dataset, Dataset]:
        """
        Tokenize datasets.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            text_column: Column containing text to tokenize
            
        Returns:
            Tokenized train and val datasets
        """
        train_dataset, val_dataset = self.get_datasets()
        
        def tokenize_fn(batch):
            return tokenizer(
                batch[text_column],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        
        # Get columns to remove (everything except the tokenizer outputs)
        remove_cols = train_dataset.column_names
        
        print(f"🔤 Tokenizing (max_length={max_length})...")
        train_dataset = train_dataset.map(
            tokenize_fn, 
            batched=True, 
            remove_columns=remove_cols,
        )
        val_dataset = val_dataset.map(
            tokenize_fn, 
            batched=True, 
            remove_columns=remove_cols,
        )
        
        return train_dataset, val_dataset


def load_jsonl(path: str) -> List[Dict]:
    """Simple JSONL loader."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Simple JSONL saver."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
