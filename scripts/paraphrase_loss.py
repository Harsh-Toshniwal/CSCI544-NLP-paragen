"""
Hybrid loss functions for paraphrase generation.
Combines CLM loss with semantic similarity and lexical diversity constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, util


class HybridParaphraseLoss(nn.Module):
    """
    Hybrid loss for paraphrase generation combining:
    1. Causal Language Modeling (CLM) loss
    2. Semantic Similarity loss (preserve meaning)
    3. Lexical Overlap Penalty (encourage diversity)
    """

    def __init__(
        self,
        use_semantic_loss: bool = True,
        use_overlap_penalty: bool = True,
        semantic_weight: float = 0.2,
        overlap_weight: float = 0.1,
        overlap_alpha: float = 0.5,
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        """
        Args:
            use_semantic_loss: Whether to include semantic similarity loss
            use_overlap_penalty: Whether to include lexical overlap penalty
            semantic_weight: Weight for semantic similarity loss
            overlap_weight: Weight for overlap penalty
            overlap_alpha: Alpha for overlap penalty calculation
            semantic_model: SentenceTransformer model for embeddings
            device: Device to load models on
        """
        super().__init__()
        self.use_semantic_loss = use_semantic_loss
        self.use_overlap_penalty = use_overlap_penalty
        self.semantic_weight = semantic_weight
        self.overlap_weight = overlap_weight
        self.overlap_alpha = overlap_alpha
        self.device = device

        # Load semantic similarity model
        if use_semantic_loss:
            try:
                self.semantic_model = SentenceTransformer(semantic_model, device=device)
                self.semantic_model.eval()
            except Exception as e:
                print(f"Warning: Could not load semantic model: {e}")
                print("Semantic loss will be disabled")
                self.use_semantic_loss = False

    def compute_clm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute standard cross-entropy loss for language modeling.
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len) with -100 for padding
            
        Returns:
            Scalar loss
        """
        loss_fct = nn.CrossEntropyLoss()
        # Flatten for loss computation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Only compute loss on non-padding tokens
        active_loss = shift_labels != -100
        active_logits = shift_logits[active_loss]
        active_labels = shift_labels[active_loss]
        
        loss = loss_fct(active_logits, active_labels)
        return loss

    def compute_overlap_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer_pad_id: int = 0,
    ) -> torch.Tensor:
        """
        Penalize when model generates tokens from source text.
        Encourages lexical diversity.
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            input_ids: (batch_size, seq_len) - source tokens
            labels: (batch_size, seq_len) - target tokens
            tokenizer_pad_id: Padding token ID
            
        Returns:
            Scalar penalty
        """
        batch_size = input_ids.size(0)
        vocab_size = logits.size(-1)
        device = logits.device

        # Create source token mask (multi-hot encoding)
        src_multi_hot = torch.zeros(batch_size, vocab_size, device=device)
        src_multi_hot.scatter_(1, input_ids, 1.0)
        
        # Ignore special tokens
        src_multi_hot[:, :3] = 0.0
        src_multi_hot[:, tokenizer_pad_id] = 0.0

        # Get prediction probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        
        # Compute overlap probability: how likely to generate source tokens
        overlap_prob = (probs * src_multi_hot.unsqueeze(1)).sum(dim=-1)  # (batch_size, seq_len)
        
        # Mask out padding positions
        valid_tgt_mask = (labels != -100).float()
        
        # Average overlap penalty
        overlap_penalty = (overlap_prob * valid_tgt_mask).sum() / (valid_tgt_mask.sum() + 1e-8)
        
        return overlap_penalty

    def compute_semantic_similarity(
        self,
        source_texts: list,
        generated_ids: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        """
        Compute semantic similarity loss between source and generated text.
        Ensures meaning is preserved.
        
        Args:
            source_texts: List of source sentences
            generated_ids: (batch_size, seq_len) - generated token IDs
            tokenizer: Tokenizer to decode generated IDs
            
        Returns:
            Scalar loss (1 - similarity)
        """
        device = generated_ids.device
        
        try:
            # Decode generated text
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            # Encode both source and generated
            source_embeds = self.semantic_model.encode(
                source_texts, convert_to_tensor=True, show_progress_bar=False
            )
            generated_embeds = self.semantic_model.encode(
                generated_texts, convert_to_tensor=True, show_progress_bar=False
            )
            
            # Compute cosine similarity
            similarities = util.pytorch_cos_sim(source_embeds, generated_embeds).diag()
            
            # Loss is 1 - similarity (maximize similarity)
            loss = 1.0 - similarities.mean()
            
            return loss.to(device)
        except Exception as e:
            print(f"Warning: Could not compute semantic similarity: {e}")
            return torch.tensor(0.0, device=device)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        source_texts: Optional[list] = None,
        tokenizer=None,
        tokenizer_pad_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss.
        
        Args:
            logits: (batch_size, seq_len, vocab_size) - model logits
            labels: (batch_size, seq_len) - target token IDs
            input_ids: (batch_size, seq_len) - source token IDs
            source_texts: List of source sentences (for semantic loss)
            tokenizer: Tokenizer (for semantic loss)
            tokenizer_pad_id: Padding token ID
            
        Returns:
            Dict with 'total_loss' and individual loss components
        """
        losses = {}
        
        # 1. CLM Loss (always computed)
        clm_loss = self.compute_clm_loss(logits, labels)
        losses["clm_loss"] = clm_loss
        
        total_loss = clm_loss
        
        # 2. Overlap Penalty (lexical diversity)
        if self.use_overlap_penalty:
            overlap_penalty = self.compute_overlap_penalty(
                logits, input_ids, labels, tokenizer_pad_id
            )
            losses["overlap_penalty"] = overlap_penalty
            total_loss = total_loss + self.overlap_weight * overlap_penalty
        
        # 3. Semantic Similarity Loss (meaning preservation)
        if self.use_semantic_loss and source_texts is not None and tokenizer is not None:
            semantic_loss = self.compute_semantic_similarity(
                source_texts, labels, tokenizer
            )
            losses["semantic_loss"] = semantic_loss
            total_loss = total_loss + self.semantic_weight * semantic_loss
        
        losses["total_loss"] = total_loss
        
        return losses


class SimpleParaphraseLoss(nn.Module):
    """
    Lightweight version without semantic model dependency.
    Only uses CLM + overlap penalty.
    """

    def __init__(
        self,
        overlap_weight: float = 0.1,
        tokenizer_pad_id: int = 0,
    ):
        """
        Args:
            overlap_weight: Weight for overlap penalty
            tokenizer_pad_id: Padding token ID
        """
        super().__init__()
        self.overlap_weight = overlap_weight
        self.tokenizer_pad_id = tokenizer_pad_id

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CLM loss + overlap penalty.
        
        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len)
            input_ids: (batch_size, seq_len)
            
        Returns:
            Dict with loss components
        """
        device = logits.device
        
        # CLM Loss
        loss_fct = nn.CrossEntropyLoss()
        clm_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Overlap Penalty
        batch_size = input_ids.size(0)
        vocab_size = logits.size(-1)
        
        src_multi_hot = torch.zeros(batch_size, vocab_size, device=device)
        src_multi_hot.scatter_(1, input_ids, 1.0)
        src_multi_hot[:, :3] = 0.0  # Ignore special tokens
        src_multi_hot[:, self.tokenizer_pad_id] = 0.0
        
        probs = torch.softmax(logits, dim=-1)
        overlap_prob = (probs * src_multi_hot.unsqueeze(1)).sum(dim=-1)
        valid_tgt_mask = (labels != -100).float()
        overlap_penalty = (overlap_prob * valid_tgt_mask).sum() / (valid_tgt_mask.sum() + 1e-8)
        
        total_loss = clm_loss + self.overlap_weight * overlap_penalty
        
        return {
            "clm_loss": clm_loss,
            "overlap_penalty": overlap_penalty,
            "total_loss": total_loss,
        }
