"""Evaluation metrics for paraphrase generation"""

import numpy as np
from typing import List, Tuple, Dict
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)


class ParaphraseEvaluator:
    """Compute evaluation metrics for paraphrase generation"""

    def __init__(self):
        """Initialize evaluator"""
        self.metrics_computed = {}

    @staticmethod
    def compute_bert_score_similarity(source: str, target: str) -> float:
        """
        Compute sentence-level semantic similarity using Sentence-BERT
        (Loads from local checkpoint)
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            import os

            # Use local checkpoint if available, otherwise download
            local_model_path = "./checkpoints/sentence-transformers"
            if os.path.exists(local_model_path):
                model = SentenceTransformer(local_model_path)
            else:
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            embeddings = model.encode([source, target], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
            return float(similarity.item())
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return 0.5  # Default value

    @staticmethod
    def compute_inverse_bleu(source: str, target: str) -> float:
        """
        Compute 1-BLEU to measure diversity
        Lower value = higher diversity
        """
        from nltk.translate.bleu_score import sentence_bleu

        source_tokens = source.lower().split()
        target_tokens = target.lower().split()

        # Compute BLEU
        bleu = sentence_bleu([source_tokens], target_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        inverse_bleu = 1 - bleu

        return inverse_bleu

    @staticmethod
    def compute_self_bleu(candidates: List[str]) -> float:
        """
        Compute Self-BLEU: average BLEU between each candidate and others
        Lower = higher diversity within candidate set
        """
        from nltk.translate.bleu_score import sentence_bleu

        if len(candidates) < 2:
            return 0.0

        scores = []
        for i, candidate in enumerate(candidates):
            ref_candidates = candidates[:i] + candidates[i + 1 :]
            candidate_tokens = candidate.lower().split()

            bleu_scores = []
            for ref in ref_candidates:
                ref_tokens = ref.lower().split()
                bleu = sentence_bleu([ref_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
                bleu_scores.append(bleu)

            scores.append(np.mean(bleu_scores) if bleu_scores else 0)

        return np.mean(scores)

    @staticmethod
    def compute_lexical_diversity_ratio(source: str, target: str) -> float:
        """
        Compute lexical diversity ratio (LDR)
        LDR = 1 - (|tokens_source ∩ tokens_target| / |tokens_source ∪ tokens_target|)
        """
        source_tokens = set(source.lower().split())
        target_tokens = set(target.lower().split())

        if len(source_tokens | target_tokens) == 0:
            return 0.0

        intersection = len(source_tokens & target_tokens)
        union = len(source_tokens | target_tokens)

        ldr = 1 - (intersection / union)
        return ldr

    @staticmethod
    def compute_meteor(source: str, target: str) -> float:
        """
        Compute METEOR score (requires nltk data)
        Simplified version - full METEOR requires stemming and synonyms
        """
        try:
            from nltk.translate.meteor_score import meteor_score

            source_tokens = source.lower().split()
            target_tokens = target.lower().split()

            score = meteor_score([source_tokens], target_tokens)
            return score
        except ImportError:
            logger.warning("METEOR computation requires nltk data")
            return 0.5  # Default value

    @staticmethod
    def compute_perplexity(text: str, model_name: str = "gpt2") -> float:
        """
        Compute perplexity using a pre-trained language model
        (Placeholder - requires transformers and a language model)
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            input_ids = tokenizer.encode(text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss

            perplexity = torch.exp(loss).item()
            return perplexity
        except Exception as e:
            logger.warning(f"Could not compute perplexity: {e}")
            return float("inf")  # Default to infinity if computation fails

    def compute_para_score(
        self, semantic_similarity: float, diversity: float, weight: float = 0.6
    ) -> float:
        """
        Compute ParaScore = weight * semantic_similarity + (1-weight) * diversity
        """
        return weight * semantic_similarity + (1 - weight) * diversity

    def evaluate_pair(
        self, source: str, target: str, compute_all: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a single source-target paraphrase pair

        Args:
            source: Source sentence
            target: Target paraphrase
            compute_all: Whether to compute all metrics

        Returns:
            Dictionary of metric scores
        """
        scores = {}

        if compute_all:
            # Semantic similarity
            scores["sentence_bert_similarity"] = self.compute_bert_score_similarity(
                source, target
            )

            # Diversity metrics
            scores["inverse_bleu"] = self.compute_inverse_bleu(source, target)
            scores["lexical_diversity_ratio"] = self.compute_lexical_diversity_ratio(
                source, target
            )

            # Fluency
            scores["perplexity"] = self.compute_perplexity(target)
            scores["meteor"] = self.compute_meteor(source, target)

            # Composite score
            scores["para_score"] = self.compute_para_score(
                scores["sentence_bert_similarity"],
                scores["inverse_bleu"],
                weight=0.7,
            )

        return scores

    def evaluate_batch(
        self,
        sources: List[str],
        targets: List[str],
        compute_all: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Evaluate multiple source-target pairs

        Args:
            sources: List of source sentences
            targets: List of target paraphrases
            compute_all: Whether to compute all metrics

        Returns:
            Dictionary mapping metric names to lists of scores
        """
        all_scores = {}

        for source, target in zip(sources, targets):
            pair_scores = self.evaluate_pair(source, target, compute_all=compute_all)

            for metric, score in pair_scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)

        # Compute aggregates
        summary = {}
        for metric, scores in all_scores.items():
            summary[f"{metric}_mean"] = np.mean(scores)
            summary[f"{metric}_std"] = np.std(scores)
            summary[f"{metric}_median"] = np.median(scores)

        return all_scores, summary


def compute_inter_annotator_agreement(annotations: List[List[int]]) -> float:
    """
    Compute Krippendorff's alpha for inter-annotator agreement
    (Simplified version assuming 5-point Likert scale)

    Args:
        annotations: List of annotation lists from different annotators

    Returns:
        Krippendorff's alpha score (0-1)
    """
    try:
        import krippendorff

        # Convert to pairable data format
        data = np.array(annotations).T  # Transpose to get [item x rater]
        alpha = krippendorff.alpha(data)
        return alpha
    except ImportError:
        logger.warning("krippendorff not installed, computing simple agreement")

        # Fallback: simple agreement calculation
        if len(annotations) < 2:
            return 0.0

        agreements = 0
        total = len(annotations[0])

        for i in range(total):
            if all(annotations[j][i] == annotations[0][i] for j in range(len(annotations))):
                agreements += 1

        return agreements / total
