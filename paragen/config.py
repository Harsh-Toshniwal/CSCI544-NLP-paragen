"""Configuration settings for ParaGen"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration"""
    train_dataset: str = "qqp"  # 'qqp', 'paws', 'both'
    eval_datasets: list = None  # ['mrpc', 'qqp', 'twitter_ppdb']
    data_dir: str = "./data"
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    max_source_length: int = 128
    max_target_length: int = 128
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.eval_datasets is None:
            self.eval_datasets = ['mrpc', 'qqp', 'twitter_ppdb']


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "t5-base"
    model_type: str = "t5"  # 't5' or 'other'
    hidden_size: int = 768
    num_layers: int = 12
    num_beams: int = 5
    num_diverse_beams: int = 3
    diversity_penalty: float = 0.5
    length_penalty: float = 1.0
    early_stopping: bool = True
    max_length: int = 128
    min_length: int = 5


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    save_strategy: str = "epoch"  # 'epoch', 'steps'
    eval_strategy: str = "epoch"
    logging_steps: int = 100
    seed: int = 42


@dataclass
class RerankerConfig:
    """Semantic reranking configuration"""
    reranker_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_metric: str = "combined"  # 'cosine', 'bertscore', 'combined'
    lambda_weight: float = 0.6  # weight for cosine similarity in combined metric
    num_candidates: int = 5  # number of candidates to generate before reranking
    use_reranking: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    eval_metrics: list = None
    human_eval_samples: int = 200
    results_dir: str = "./results"
    use_llm_judge: bool = True
    llm_model: str = "gpt-4"  # for LLM-as-a-judge evaluation

    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = [
                'sentence_bert_similarity',
                'bertscore',
                'inverse_bleu',
                'self_bleu',
                'lexical_diversity',
                'perplexity',
                'meteor'
            ]


@dataclass
class AttributeConfig:
    """Attribute token configuration for controllable generation"""
    length_tokens: list = None
    diversity_tokens: list = None
    use_length_control: bool = True
    use_diversity_control: bool = True

    def __post_init__(self):
        if self.length_tokens is None:
            self.length_tokens = ['[SHORT]', '[SAME]', '[LONG]']
        if self.diversity_tokens is None:
            self.diversity_tokens = ['[CONSERVATIVE]', '[CREATIVE]']


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    reranker: RerankerConfig = None
    evaluation: EvaluationConfig = None
    attributes: AttributeConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.reranker is None:
            self.reranker = RerankerConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.attributes is None:
            self.attributes = AttributeConfig()


def get_config() -> Config:
    """Get default configuration"""
    return Config()
