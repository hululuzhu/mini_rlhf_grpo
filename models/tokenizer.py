"""Character-level tokenizer for transformer LLM."""

import pickle
from typing import List, Optional


class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab: Optional[List[str]] = None):
        """Initialize tokenizer with vocabulary.
        
        Args:
            vocab: List of characters in vocabulary. If None, will be built from data.
        """
        if vocab is not None:
            self.vocab = vocab
            self.char_to_idx = {ch: i for i, ch in enumerate(vocab)}
            self.idx_to_char = {i: ch for i, ch in enumerate(vocab)}
        else:
            self.vocab = []
            self.char_to_idx = {}
            self.idx_to_char = {}
        
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
    
    def build_vocab(self, text: str):
        """Build vocabulary from text.
        
        Args:
            text: Training text to extract characters from.
        """
        # Get unique characters
        chars = sorted(list(set(text)))
        
        # Add special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.vocab = special_tokens + chars
        
        # Build mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids.
        
        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            
        Returns:
            List of token ids.
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.char_to_idx[self.bos_token])
        
        for ch in text:
            tokens.append(self.char_to_idx.get(ch, self.char_to_idx[self.unk_token]))
        
        if add_special_tokens:
            tokens.append(self.char_to_idx[self.eos_token])
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text.
        
        Args:
            token_ids: List of token ids to decode.
            skip_special_tokens: Whether to skip special tokens in output.
            
        Returns:
            Decoded text.
        """
        special_token_ids = [
            self.char_to_idx[self.pad_token],
            self.char_to_idx[self.bos_token],
            self.char_to_idx[self.eos_token],
        ]
        
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue
            chars.append(self.idx_to_char.get(token_id, self.unk_token))
        
        return ''.join(chars)
    
    def save(self, path: str):
        """Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer.
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """Load tokenizer from file.
        
        Args:
            path: Path to load tokenizer from.
            
        Returns:
            Loaded tokenizer.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab=data['vocab'])
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = data['idx_to_char']
        return tokenizer
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        """Return pad token id."""
        return self.char_to_idx[self.pad_token]
    
    @property
    def bos_token_id(self) -> int:
        """Return BOS token id."""
        return self.char_to_idx[self.bos_token]
    
    @property
    def eos_token_id(self) -> int:
        """Return EOS token id."""
        return self.char_to_idx[self.eos_token]

