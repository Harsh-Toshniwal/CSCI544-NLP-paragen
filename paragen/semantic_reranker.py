"""Semantic reranking module for paraphrase selection"""

import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticReranker:
    """Rerank paraphrase candidates based on semantic similarity"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        metric: str = "combined",
        lambda_weight: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_name: Sentence-BERT model name
            metric: 'cosine', 'bertscore', or 'combined'
            lambda_weight: Weight for cosine similarity in combined metric
            device: Device to load model on
        """
        self.device = device
        self.metric = metric
        self.lambda_weight = lambda_weight

        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Loaded semantic reranker: {model_name} on {device}")

    def compute_cosine_similarity(
        self, source: str, candidates: List[str]
    ) -> List[float]:
        """
        Compute cosine similarity between source and candidates

        Args:
            source: Source sentence
            candidates: List of candidate paraphrases

        Returns:
            List of similarity scores
        """
        source_emb = self.model.encode(source, convert_to_tensor=True)
        candidate_embs = self.model.encode(candidates, convert_to_tensor=True)

        # Compute cosine similarity
        similarities = util.pytorch_cos_sim(source_emb, candidate_embs)[0].cpu().tolist()

        return similarities

    def compute_bertscore(
        self, source: str, candidates: List[str]
    ) -> List[float]:
        """
        Compute BERTScore F1 between source and candidates

        Args:
            source: Source sentence
            candidates: List of candidate paraphrases

        Returns:
            List of BERTScore F1 scores
        """
        try:
            from bert_score import score as bert_score_fn

            # Prepare inputs (BERTScore expects lists)
            cands = candidates
            refs = [source] * len(candidates)

            # Compute BERTScore
            _, _, f1_scores = bert_score_fn(
                cands, refs, lang="en", verbose=False, model_type="bert-base-uncased"
            )

            return f1_scores.tolist()
        except ImportError:
            logger.warning("bert_score not installed, falling back to cosine similarity")
            return self.compute_cosine_similarity(source, candidates)

    def rerank(
        self,
        source: str,
        candidates: List[str],
        return_scores: bool = False,
    ) -> Tuple[str, Optional[List[Tuple[str, float]]]]:
        """
        Rerank candidates and return the best one

        Args:
            source: Source sentence
            candidates: List of candidate paraphrases
            return_scores: Whether to return all candidates with scores

        Returns:
            Best paraphrase and optionally all candidates with scores
        """
        if not candidates:
            raise ValueError("No candidates provided for reranking")

        if self.metric == "cosine":
            scores = self.compute_cosine_similarity(source, candidates)
        elif self.metric == "bertscore":
            scores = self.compute_bertscore(source, candidates)
        elif self.metric == "combined":
            # Combined metric: lambda * cosine + (1-lambda) * bertscore
            cosine_scores = self.compute_cosine_similarity(source, candidates)
            bert_scores = self.compute_bertscore(source, candidates)

            # Normalize BERTScore to [0, 1] if needed
            scores = [
                self.lambda_weight * cos + (1 - self.lambda_weight) * bert
                for cos, bert in zip(cosine_scores, bert_scores)
            ]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Get best candidate
        best_idx = scores.index(max(scores))
        best_candidate = candidates[best_idx]
        best_score = scores[best_idx]

        if return_scores:
            ranked = sorted(
                zip(candidates, scores), key=lambda x: x[1], reverse=True
            )
            return best_candidate, ranked
        else:
            return best_candidate, None

    def batch_rerank(
        self,
        sources: List[str],
        candidates_list: List[List[str]],
        return_scores: bool = False,
    ) -> List[Tuple[str, Optional[List[Tuple[str, float]]]]]:
        """
        Rerank candidates for multiple sources

        Args:
            sources: List of source sentences
            candidates_list: List of lists of candidates
            return_scores: Whether to return scores

        Returns:
            List of (best_candidate, ranked_list) tuples
        """
        results = []
        for source, candidates in zip(sources, candidates_list):
            best, ranked = self.rerank(source, candidates, return_scores=return_scores)
            results.append((best, ranked))

        return results
