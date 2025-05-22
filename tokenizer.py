"""
Quasar Tokenizer Implementation

This module provides a wrapper around the GPT-2 tokenizer for use with the Quasar model.
"""

from typing import List, Optional, Union, Dict
import os

try:
    from transformers import GPT2Tokenizer, GPT2TokenizerFast
except ImportError:
    raise ImportError(
        "You need to install the transformers library to use the Quasar tokenizer. "
        "Run: pip install transformers"
    )


class QuasarTokenizer:
    """
    Wrapper around the GPT-2 tokenizer for use with the Quasar model.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "gpt2",
        cache_dir: Optional[str] = None,
        use_fast: bool = True,
        **kwargs
    ):
        """
        Initialize the Quasar tokenizer.
        
        Args:
            pretrained_model_name_or_path: Name or path of the pretrained GPT-2 tokenizer
            cache_dir: Directory to cache the tokenizer
            use_fast: Whether to use the fast tokenizer implementation
            **kwargs: Additional arguments to pass to the tokenizer
        """
        # Load the GPT-2 tokenizer
        tokenizer_class = GPT2TokenizerFast if use_fast else GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            **kwargs
        )
        
        # Set properties based on the GPT-2 tokenizer
        self.vocab_size = len(self.tokenizer)
        self.pad_token = self.tokenizer.pad_token
        self.eos_token = self.tokenizer.eos_token
        self.bos_token = self.tokenizer.bos_token
        
        # Set max length
        self.max_len = self.tokenizer.model_max_length
    
    @property
    def vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        return self.tokenizer.get_vocab()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences
            padding: Whether to pad sequences
            
        Returns:
            List of token IDs
        """
        if max_length is None:
            max_length = self.max_len
        
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[str], List[List[str]]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        """
        Encode a batch of texts.
        
        Args:
            batch_text_or_text_pairs: Batch of texts or text pairs
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences
            padding: Whether to pad sequences
            return_tensors: Type of tensors to return
            
        Returns:
            Dictionary containing the encoded sequences
        """
        if max_length is None:
            max_length = self.max_len
        
        return self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the tokenizer to a directory.
        
        Args:
            save_directory: Directory to save the tokenizer to
        """
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Load a tokenizer from a pretrained model.
        
        Args:
            pretrained_model_name_or_path: Name or path of the pretrained model
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            QuasarTokenizer instance
        """
        return cls(pretrained_model_name_or_path, *args, **kwargs)
