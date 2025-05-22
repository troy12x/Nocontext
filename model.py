"""
Quasar Model Implementation

This module implements the Quasar model architecture, which extends GPT-2 with
Parameter Memory Bank and Liquid Neural Network capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import os

try:
    from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
except ImportError:
    raise ImportError(
        "You need to install the transformers library to use the Quasar model. "
        "Run: pip install transformers"
    )

from .memory_bank import ParameterMemoryBank
from .liquid_layers import LiquidAttention, LiquidMLP


class QuasarConfig:
    """Configuration class for Quasar model."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_memory_blocks: int = 32,
        block_capacity: int = 1024,
        key_dim: int = 256,
        value_dim: int = 256,
        liquid_update_interval: int = 100,
    ):
        """Initialize QuasarConfig."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Parameter Memory Bank configuration
        self.num_memory_blocks = num_memory_blocks
        self.block_capacity = block_capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Liquid Neural Network configuration
        self.liquid_update_interval = liquid_update_interval


class QuasarEmbeddings(nn.Module):
    """Embeddings for the Quasar model."""
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position IDs buffer (for caching position IDs)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward pass for embeddings."""
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class QuasarLayer(nn.Module):
    """A single layer of the Quasar model."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = LiquidAttention(config)
        self.intermediate = LiquidMLP(config)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.memory_bank = ParameterMemoryBank(
            num_blocks=config.num_memory_blocks,
            block_capacity=config.block_capacity,
            key_dim=config.key_dim,
            value_dim=config.value_dim,
            hidden_size=config.hidden_size
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """Forward pass for a Quasar layer."""
        # Self-attention with liquid neural network dynamics
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            step=step
        )
        
        # Query the parameter memory bank
        memory_output = self.memory_bank(hidden_states)
        
        # Combine attention output with memory output
        combined_output = attention_output + memory_output
        
        # Apply MLP with liquid neural network dynamics
        layer_output = self.intermediate(
            combined_output,
            step=step
        )
        
        # Apply layer normalization
        layer_output = self.output_layer_norm(layer_output + combined_output)
        
        return layer_output


class QuasarModel(nn.Module):
    """Quasar model with Parameter Memory Bank and Liquid Neural Network architecture."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = QuasarEmbeddings(config)
        self.layers = nn.ModuleList([QuasarLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Step counter for liquid neural network dynamics
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the Quasar model."""
        # Increment step counter for liquid neural network dynamics
        self.step_counter += 1
        step = self.step_counter.item()
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
        
        # Create attention mask if not provided
        if attention_mask is None and input_ids is not None:
            attention_mask = (input_ids != self.config.pad_token_id).float()
        
        # Extend attention mask for multi-head attention
        if attention_mask is not None:
            # Expand attention mask to [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert attention mask to additive mask
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        hidden_states = inputs_embeds
        
        # Process through each layer
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                step=step
            )
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states


class QuasarForCausalLM(nn.Module):
    """Quasar model for causal language modeling."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = QuasarModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights between input embeddings and output embeddings
        self.lm_head.weight = self.model.embeddings.word_embeddings.weight
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for causal language modeling."""
        # Get hidden states from the model
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        
        # Project hidden states to vocabulary
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss using cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if loss is not None:
            return {"loss": loss, "logits": lm_logits}
        else:
            return lm_logits
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.LongTensor:
        """Generate text using the model."""
        batch_size = input_ids.shape[0]
        
        # Set effective batch size and effective batch multiplier
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
        
        if effective_batch_size != batch_size:
            input_ids = input_ids.repeat_interleave(effective_batch_mult, dim=0)
        
        cur_len = input_ids.shape[1]
        
        if max_length < cur_len:
            raise ValueError(f"max_length ({max_length}) must be greater than current length ({cur_len})")
        
        # Keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(effective_batch_size).fill_(1)
        
        while cur_len < max_length and unfinished_sequences.max() > 0:
            # Forward pass to get logits
            with torch.no_grad():
                logits = self.forward(input_ids=input_ids)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(effective_batch_size):
                        for token_id in set(input_ids[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply top-k and top-p filtering
                if do_sample:
                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float("Inf")
                    
                    # Top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = -float("Inf")
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Update input_ids
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                cur_len = input_ids.shape[1]
                
                # Check if any sequences are finished
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != self.config.eos_token_id).long()
                )
        
        # Return generated sequences
        if num_return_sequences == 1:
            return input_ids
        else:
            return input_ids.view(batch_size, num_return_sequences, -1)
