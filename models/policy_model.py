"""Policy model for language generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .transformer import TransformerLM


class PolicyModel(nn.Module):
    """Policy model for token generation with language modeling head."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """Initialize policy model.
        
        Args:
            vocab_size: Vocabulary size.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            d_ff: Feed-forward dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
            pad_token_id: Padding token id.
        """
        super().__init__()
        self.transformer = TransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights with token embedding
        self.lm_head.weight = self.transformer.token_embedding.weight
        
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            labels: Target labels for loss computation [batch_size, seq_len].
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size].
            loss: Cross-entropy loss if labels provided, else None.
        """
        # Validate inputs
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [batch, seq_len], got shape {input_ids.shape}")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError(f"labels must match input_ids shape {input_ids.shape}, got {labels.shape}")
        
        # Get hidden states from transformer (don't use cache for training)
        hidden_states, _ = self.transformer(input_ids, attention_mask, past_kvs=None, use_cache=False)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
        
        return logits, loss
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get log probabilities for each token.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Log probabilities [batch_size, seq_len, vocab_size].
        """
        logits, _ = self.forward(input_ids, attention_mask, labels=None)
        return F.log_softmax(logits, dim=-1)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text from prompt.
        
        Args:
            input_ids: Prompt token ids [batch_size, prompt_len].
            max_length: Maximum length to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            do_sample: Whether to sample or use greedy decoding.
            eos_token_id: End-of-sequence token id.
            repetition_penalty: Penalty for repeated tokens (>1.0 discourages, <1.0 encourages).
                              If None, no repetition penalty is applied.
            
        Returns:
            Generated token ids [batch_size, seq_len].
        """
        # Validate parameters
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if top_k is not None and not (1 <= top_k < self.vocab_size):
            raise ValueError(f"top_k must be in [1, {self.vocab_size-1}], got {top_k}")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if repetition_penalty is not None and repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be positive, got {repetition_penalty}")
        
        batch_size, cur_len = input_ids.shape
        device = input_ids.device
        
        # Check if prompt fits within max_seq_len
        if cur_len >= self.transformer.max_seq_len:
            raise ValueError(
                f"Prompt length {cur_len} >= max_seq_len {self.transformer.max_seq_len}. "
                "Cannot generate further without increasing max_seq_len."
            )
        
        generated = input_ids
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Limit generation to not exceed max_seq_len
        max_new_tokens = min(max_length, self.transformer.max_seq_len) - cur_len
        
        # Initialize KV cache
        past_kvs = None
        
        for step in range(max_new_tokens):
            # Content-based attention mask (not all-ones!)
            # For first step, process full prompt; afterwards, only process new token
            if step == 0:
                # First iteration: process entire prompt
                current_input = generated
                # Create mask: 1 for real tokens (non-padding), 0 for padding
                attn_mask = (generated != self.pad_token_id).long()
            else:
                # Subsequent iterations: only process last token with KV cache
                current_input = generated[:, -1:]
                # Extend attention mask for the new token (it's always real, not padding)
                new_token_mask = torch.ones(batch_size, 1, device=device, dtype=torch.long)
                attn_mask = torch.cat([attn_mask, new_token_mask], dim=1)
            
            # Forward pass with KV caching
            hidden_states, past_kvs = self.transformer(
                current_input, 
                attention_mask=attn_mask.bool(),
                past_kvs=past_kvs,
                use_cache=True
            )
            logits = self.lm_head(hidden_states)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty is not None and repetition_penalty != 1.0:
                # Get unique tokens in the generated sequence so far
                for batch_idx in range(batch_size):
                    # Get all tokens generated for this batch item
                    generated_tokens = generated[batch_idx].unique()
                    
                    # Apply penalty: if score > 0, divide; if score < 0, multiply
                    for token_id in generated_tokens:
                        if next_token_logits[batch_idx, token_id] > 0:
                            next_token_logits[batch_idx, token_id] /= repetition_penalty
                        else:
                            next_token_logits[batch_idx, token_id] *= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                kth = torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < kth, float('-inf'))
            
            # Apply top-p (nucleus) filtering with vectorized implementation and fallback
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(next_token_logits, dim=-1, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cums = torch.cumsum(probs, dim=-1)
                
                # Mask tokens beyond nucleus, but always keep at least the first token
                nucleus_mask = cums <= top_p
                nucleus_mask[..., 0] = True  # Ensure at least one token per row
                
                # Build keep-mask in original order
                keep = torch.zeros_like(next_token_logits, dtype=torch.bool)
                keep.scatter_(dim=-1, index=sorted_idx, src=nucleus_mask)
                
                # Fallback: if a row is all False (shouldn't happen with above), keep argmax
                row_all_false = ~keep.any(dim=-1, keepdim=True)
                argmax_idx = next_token_logits.argmax(dim=-1, keepdim=True)
                keep.scatter_(dim=-1, index=argmax_idx, src=row_all_false)
                
                next_token_logits = next_token_logits.masked_fill(~keep, float('-inf'))
            
            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                # Additional safety: if probs has NaN/Inf, fall back to greedy
                if not torch.isfinite(probs).all():
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Force finished sequences to stay at EOS
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token
                )
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update finished status
            if eos_token_id is not None:
                finished |= (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
        
        return generated

