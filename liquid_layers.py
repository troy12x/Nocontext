"""
Liquid Neural Network Implementation for Quasar Model

This module implements the Liquid Neural Network architecture, which features
dynamic weight updates and adaptive computation based on input data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any


class LiquidFunction(nn.Module):
    """
    Base class for liquid neural network functions.
    
    Liquid functions have weights that dynamically update based on the input
    and the current step in the computation.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config):
        """
        Initialize the liquid function.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            config: Configuration object
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Base weights (static component)
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Dynamic weight modulation
        self.dynamic_weight_scale = nn.Parameter(torch.ones(output_dim, input_dim) * 0.01)
        self.dynamic_bias_scale = nn.Parameter(torch.ones(output_dim) * 0.01)
        
        # Time-dependent modulation
        self.time_freq = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.time_phase = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        
        # Update interval for liquid dynamics
        self.update_interval = config.liquid_update_interval
    
    def get_dynamic_weights(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get dynamically updated weights based on the current step.
        
        Args:
            step: Current computation step
            
        Returns:
            Tuple of (weights, biases) with dynamic updates applied
        """
        # Apply time-dependent modulation
        time_factor = torch.sin(step * self.time_freq + self.time_phase)
        
        # Compute dynamic weights
        dynamic_weights = self.weight + self.dynamic_weight_scale * time_factor
        dynamic_bias = self.bias + self.dynamic_bias_scale * torch.sin(step * 0.01)
        
        return dynamic_weights, dynamic_bias
    
    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Forward pass with dynamic weight updates.
        
        Args:
            x: Input tensor
            step: Current computation step
            
        Returns:
            Output tensor
        """
        # Get dynamic weights
        weights, bias = self.get_dynamic_weights(step)
        
        # Apply linear transformation
        return F.linear(x, weights, bias)


class LiquidAttention(nn.Module):
    """
    Liquid Attention mechanism for the Quasar model.
    
    This attention mechanism features dynamic weight updates based on the
    liquid neural network paradigm.
    """
    
    def __init__(self, config):
        """
        Initialize the liquid attention mechanism.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Ensure hidden size is divisible by number of attention heads
        assert config.hidden_size % config.num_attention_heads == 0, \
            f"Hidden size {config.hidden_size} not divisible by number of attention heads {config.num_attention_heads}"
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, key, and value projections using liquid functions
        self.query = LiquidFunction(config.hidden_size, self.all_head_size, config)
        self.key = LiquidFunction(config.hidden_size, self.all_head_size, config)
        self.value = LiquidFunction(config.hidden_size, self.all_head_size, config)
        
        # Output projection
        self.output = LiquidFunction(self.all_head_size, config.hidden_size, config)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape the tensor for multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, all_head_size]
            
        Returns:
            Reshaped tensor of shape [batch_size, num_heads, seq_len, head_size]
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for liquid attention.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            step: Current computation step
            
        Returns:
            Output tensor after attention
        """
        # Apply layer normalization to input
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Project query, key, and value
        query_layer = self.query(hidden_states, step)
        key_layer = self.key(hidden_states, step)
        value_layer = self.value(hidden_states, step)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        # Get context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape back to [batch_size, seq_len, all_head_size]
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        output = self.output(context_layer, step)
        
        # Add residual connection
        output = output + residual
        
        return output


class LiquidMLP(nn.Module):
    """
    Liquid MLP for the Quasar model.
    
    This MLP features dynamic weight updates based on the liquid neural network paradigm.
    """
    
    def __init__(self, config):
        """
        Initialize the liquid MLP.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Intermediate layer
        self.intermediate = LiquidFunction(config.hidden_size, config.intermediate_size, config)
        
        # Output layer
        self.output = LiquidFunction(config.intermediate_size, config.hidden_size, config)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Activation function
        self.act_fn = self._get_activation_fn(config.hidden_act)
    
    def _get_activation_fn(self, activation: str):
        """
        Get activation function by name.
        
        Args:
            activation: Name of activation function
            
        Returns:
            Activation function
        """
        if activation == "gelu":
            return F.gelu
        elif activation == "relu":
            return F.relu
        elif activation == "silu" or activation == "swish":
            return F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, hidden_states: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Forward pass for liquid MLP.
        
        Args:
            hidden_states: Input hidden states
            step: Current computation step
            
        Returns:
            Output tensor after MLP
        """
        # Apply layer normalization to input
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Apply intermediate layer
        intermediate_output = self.intermediate(hidden_states, step)
        
        # Apply activation function
        intermediate_output = self.act_fn(intermediate_output)
        
        # Apply output layer
        output = self.output(intermediate_output, step)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Add residual connection
        output = output + residual
        
        return output


class LiquidLayerNorm(nn.Module):
    """
    Liquid Layer Normalization for the Quasar model.
    
    This layer normalization features dynamic parameter updates based on the
    liquid neural network paradigm.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-12, config=None):
        """
        Initialize the liquid layer normalization.
        
        Args:
            hidden_size: Hidden size
            eps: Epsilon for numerical stability
            config: Configuration object
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.config = config
        
        # Base parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Dynamic parameter modulation
        self.dynamic_weight_scale = nn.Parameter(torch.ones(hidden_size) * 0.01)
        self.dynamic_bias_scale = nn.Parameter(torch.ones(hidden_size) * 0.01)
        
        # Time-dependent modulation
        self.time_freq = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.time_phase = nn.Parameter(torch.randn(hidden_size) * 0.01)
        
        # Update interval for liquid dynamics
        self.update_interval = config.liquid_update_interval if config else 100
    
    def get_dynamic_params(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get dynamically updated parameters based on the current step.
        
        Args:
            step: Current computation step
            
        Returns:
            Tuple of (weight, bias) with dynamic updates applied
        """
        # Apply time-dependent modulation
        time_factor = torch.sin(step * self.time_freq + self.time_phase)
        
        # Compute dynamic parameters
        dynamic_weight = self.weight + self.dynamic_weight_scale * time_factor
        dynamic_bias = self.bias + self.dynamic_bias_scale * torch.sin(step * 0.01)
        
        return dynamic_weight, dynamic_bias
    
    def forward(self, hidden_states: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Forward pass for liquid layer normalization.
        
        Args:
            hidden_states: Input hidden states
            step: Current computation step
            
        Returns:
            Output tensor after layer normalization
        """
        # Get dynamic parameters
        weight, bias = self.get_dynamic_params(step)
        
        # Apply layer normalization
        mean = hidden_states.mean(-1, keepdim=True)
        var = hidden_states.var(-1, unbiased=False, keepdim=True)
        normalized = (hidden_states - mean) / torch.sqrt(var + self.eps)
        
        return weight * normalized + bias
