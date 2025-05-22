"""
Parameter Memory Bank Implementation for Quasar Model

This module implements the Parameter Memory Bank, a key component of the Quasar
architecture that enables unlimited context window by storing and retrieving
information using key-value pairs organized in memory blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any


class ParameterMemoryBank(nn.Module):
    """
    Parameter Memory Bank for storing and retrieving information.
    
    The Parameter Memory Bank is organized into multiple blocks, each with a fixed
    capacity. It uses a key-value storage system where keys are derived from input
    embeddings and values are stored for later retrieval.
    """
    
    def __init__(
        self,
        num_blocks: int = 32,
        block_capacity: int = 1024,
        key_dim: int = 256,
        value_dim: int = 256,
        hidden_size: int = 768,
        init_scale: float = 0.02,
    ):
        """
        Initialize the Parameter Memory Bank.
        
        Args:
            num_blocks: Number of memory blocks
            block_capacity: Capacity of each memory block
            key_dim: Dimension of key vectors
            value_dim: Dimension of value vectors
            hidden_size: Hidden size of the model
            init_scale: Scale for initializing memory bank parameters
        """
        super().__init__()
        
        self.num_blocks = num_blocks
        self.block_capacity = block_capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_size = hidden_size
        
        # Key projection
        self.key_proj = nn.Linear(hidden_size, key_dim)
        
        # Value projection
        self.value_proj = nn.Linear(hidden_size, value_dim)
        
        # Output projection
        self.output_proj = nn.Linear(value_dim, hidden_size)
        
        # Initialize memory blocks
        # Each block contains keys and values
        self.register_parameter(
            "memory_keys",
            nn.Parameter(torch.randn(num_blocks, block_capacity, key_dim) * init_scale)
        )
        
        self.register_parameter(
            "memory_values",
            nn.Parameter(torch.randn(num_blocks, block_capacity, value_dim) * init_scale)
        )
        
        # Usage tracking for each block
        self.register_buffer(
            "block_usage",
            torch.zeros(num_blocks, dtype=torch.long)
        )
        
        # Last accessed position for round-robin updates
        self.register_buffer(
            "last_position",
            torch.zeros(num_blocks, dtype=torch.long)
        )
        
        # Initialize layer normalization for keys and queries
        self.key_norm = nn.LayerNorm(key_dim)
        self.query_norm = nn.LayerNorm(key_dim)
    
    def reset_memory(self):
        """Reset the memory bank usage tracking."""
        self.block_usage.zero_()
        self.last_position.zero_()
    
    def store(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Store information in the memory bank.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple of (block_indices, positions) indicating where the information was stored
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project hidden states to keys and values
        keys = self.key_proj(hidden_states)  # [batch_size, seq_len, key_dim]
        values = self.value_proj(hidden_states)  # [batch_size, seq_len, value_dim]
        
        # Normalize keys
        keys = self.key_norm(keys)
        
        # Flatten batch and sequence dimensions
        keys = keys.view(-1, self.key_dim)  # [batch_size * seq_len, key_dim]
        values = values.view(-1, self.value_dim)  # [batch_size * seq_len, value_dim]
        
        # Determine which blocks to use for each item (round-robin)
        num_items = batch_size * seq_len
        block_indices = torch.arange(num_items, device=hidden_states.device) % self.num_blocks
        
        # Determine positions within each block (round-robin with tracking)
        positions = torch.zeros(num_items, dtype=torch.long, device=hidden_states.device)
        
        for i in range(num_items):
            block_idx = block_indices[i].item()
            if self.block_usage[block_idx] < self.block_capacity:
                # Block has empty slots
                pos = self.block_usage[block_idx]
                self.block_usage[block_idx] += 1
            else:
                # Block is full, use round-robin
                pos = self.last_position[block_idx]
                self.last_position[block_idx] = (pos + 1) % self.block_capacity
            
            positions[i] = pos
            
            # Update memory at the determined position
            with torch.no_grad():
                self.memory_keys.data[block_idx, pos] = keys[i]
                self.memory_values.data[block_idx, pos] = values[i]
        
        return block_indices.view(batch_size, seq_len), positions.view(batch_size, seq_len)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Query the memory bank with input hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor after retrieving and processing from memory bank
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project hidden states to queries
        queries = self.key_proj(hidden_states)  # [batch_size, seq_len, key_dim]
        
        # Normalize queries
        queries = self.query_norm(queries)
        
        # Reshape for batch matrix multiplication
        queries = queries.view(batch_size * seq_len, 1, self.key_dim)
        
        # Initialize output tensor
        retrieved_values = torch.zeros(
            batch_size * seq_len, self.value_dim, device=hidden_states.device
        )
        
        # Process each memory block
        for block_idx in range(self.num_blocks):
            # Get keys and values for this block
            block_keys = self.memory_keys[block_idx, :self.block_usage[block_idx]]
            block_values = self.memory_values[block_idx, :self.block_usage[block_idx]]
            
            if block_keys.size(0) == 0:
                # Skip empty blocks
                continue
            
            # Calculate attention scores
            scores = torch.matmul(queries, block_keys.unsqueeze(0).expand(batch_size * seq_len, -1, -1).transpose(1, 2))
            scores = scores / math.sqrt(self.key_dim)  # Scale by sqrt(key_dim)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(scores, dim=-1)
            
            # Get weighted sum of values
            block_output = torch.matmul(attention_weights, block_values.unsqueeze(0).expand(batch_size * seq_len, -1, -1))
            
            # Add to retrieved values
            retrieved_values = retrieved_values + block_output.squeeze(1)
        
        # Project back to hidden size
        output = self.output_proj(retrieved_values)
        
        # Reshape to original dimensions
        output = output.view(batch_size, seq_len, self.hidden_size)
        
        return output


class MemoryBankManager:
    """
    Manager class for the Parameter Memory Bank.
    
    This class provides utility functions for working with the Parameter Memory Bank,
    including saving and loading the memory bank state, analyzing memory usage,
    and optimizing memory access patterns.
    """
    
    def __init__(self, memory_bank: ParameterMemoryBank):
        """
        Initialize the Memory Bank Manager.
        
        Args:
            memory_bank: The Parameter Memory Bank to manage
        """
        self.memory_bank = memory_bank
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory bank usage.
        
        Returns:
            Dictionary containing memory bank statistics
        """
        total_capacity = self.memory_bank.num_blocks * self.memory_bank.block_capacity
        used_slots = self.memory_bank.block_usage.sum().item()
        usage_percent = (used_slots / total_capacity) * 100 if total_capacity > 0 else 0
        
        # Get per-block statistics
        block_stats = []
        for i in range(self.memory_bank.num_blocks):
            block_stats.append({
                "block_id": i,
                "capacity": self.memory_bank.block_capacity,
                "used": self.memory_bank.block_usage[i].item(),
                "usage_percent": (self.memory_bank.block_usage[i].item() / self.memory_bank.block_capacity) * 100
            })
        
        return {
            "total_capacity": total_capacity,
            "used_slots": used_slots,
            "usage_percent": usage_percent,
            "blocks": block_stats
        }
    
    def save_memory_state(self, path: str):
        """
        Save the current memory bank state to a file.
        
        Args:
            path: Path to save the memory bank state
        """
        state_dict = {
            "memory_keys": self.memory_bank.memory_keys.data,
            "memory_values": self.memory_bank.memory_values.data,
            "block_usage": self.memory_bank.block_usage,
            "last_position": self.memory_bank.last_position,
            "config": {
                "num_blocks": self.memory_bank.num_blocks,
                "block_capacity": self.memory_bank.block_capacity,
                "key_dim": self.memory_bank.key_dim,
                "value_dim": self.memory_bank.value_dim,
                "hidden_size": self.memory_bank.hidden_size
            }
        }
        
        torch.save(state_dict, path)
    
    def load_memory_state(self, path: str):
        """
        Load the memory bank state from a file.
        
        Args:
            path: Path to load the memory bank state from
        """
        state_dict = torch.load(path)
        
        # Verify configuration matches
        config = state_dict["config"]
        assert config["num_blocks"] == self.memory_bank.num_blocks, "Number of blocks mismatch"
        assert config["block_capacity"] == self.memory_bank.block_capacity, "Block capacity mismatch"
        assert config["key_dim"] == self.memory_bank.key_dim, "Key dimension mismatch"
        assert config["value_dim"] == self.memory_bank.value_dim, "Value dimension mismatch"
        assert config["hidden_size"] == self.memory_bank.hidden_size, "Hidden size mismatch"
        
        # Load state
        self.memory_bank.memory_keys.data.copy_(state_dict["memory_keys"])
        self.memory_bank.memory_values.data.copy_(state_dict["memory_values"])
        self.memory_bank.block_usage.copy_(state_dict["block_usage"])
        self.memory_bank.last_position.copy_(state_dict["last_position"])
    
    def optimize_memory_layout(self):
        """
        Optimize the memory bank layout by reorganizing entries.
        
        This function analyzes access patterns and reorganizes memory entries
        to improve retrieval efficiency.
        """
        # This is a placeholder for memory optimization logic
        # In a real implementation, this would analyze access patterns and
        # reorganize memory entries for better efficiency
        pass
