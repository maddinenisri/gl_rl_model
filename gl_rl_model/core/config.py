"""
Configuration settings for the GL RL Model system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for the base language model."""
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_seq_length: int = 8192
    temperature: float = 0.3
    top_p: float = 0.95
    max_new_tokens: int = 1024

    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # SFT settings
    sft_learning_rate: float = 3e-5
    sft_batch_size: int = 4
    sft_gradient_accumulation_steps: int = 8
    sft_num_epochs: int = 5
    sft_warmup_steps: int = 100

    # GRPO settings
    grpo_learning_rate: float = 1e-5
    grpo_batch_size: int = 16
    grpo_num_generations_per_prompt: int = 8
    grpo_num_train_steps: int = 1000
    grpo_beta: float = 0.01  # KL penalty coefficient

    # General settings
    seed: int = 42
    output_dir: Path = Path("./checkpoints")
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50

@dataclass
class SchemaConfig:
    """Configuration for schema handling."""
    schema_file: Path = Path("./gl_rl_model/data/schema/ddl_schema.sql")
    entity_mappings_file: Path = Path("./gl_rl_model/data/schema/entity_mappings.json")
    max_tables_per_query: int = 6

    # Key tables from the ERD
    core_tables: List[str] = None

    def __post_init__(self):
        if self.core_tables is None:
            self.core_tables = [
                "SRM_PROJECTS",
                "SRM_COMPANIES",
                "SRM_CONTACTS",
                "PAC_MNT_PROJECTS",
                "PAC_MNT_RESOURCES",
                "CLNTSUPP",
                "CLNTRESPONS",
                "PROJCNTRTS",
                "PROJSTAFF",
                "PROJEVISION"
            ]

@dataclass
class RewardConfig:
    """Configuration for reward function weights."""
    # Reward component weights
    syntax_weight: float = 2.0
    execution_weight: float = 3.0
    schema_compliance_weight: float = 3.0
    business_logic_weight: float = 4.0
    performance_weight: float = 1.0
    reasoning_quality_weight: float = 1.0

    # Penalty thresholds
    max_query_complexity_penalty: float = -2.0
    missing_reasoning_penalty: float = -1.5
    invalid_syntax_penalty: float = -3.0

@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    # Agent-specific settings
    orchestrator_timeout: float = 30.0
    schema_analyzer_cache_ttl: int = 3600  # 1 hour
    query_generator_max_retries: int = 3
    validator_strict_mode: bool = True
    reward_evaluator_batch_size: int = 8

    # Agent communication
    inter_agent_timeout: float = 10.0
    max_concurrent_agents: int = 4

@dataclass
class SystemConfig:
    """Main system configuration."""
    model: ModelConfig = None
    training: TrainingConfig = None
    schema: SchemaConfig = None
    reward: RewardConfig = None
    agent: AgentConfig = None

    # System-wide settings
    debug: bool = False
    verbose: bool = True
    log_level: str = "INFO"
    data_dir: Path = Path("./gl_rl_model/data")
    cache_dir: Path = Path("./.cache")

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.schema is None:
            self.schema = SchemaConfig()
        if self.reward is None:
            self.reward = RewardConfig()
        if self.agent is None:
            self.agent = AgentConfig()

        # Create necessary directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.training.output_dir.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = SystemConfig()

def load_config(config_path: Optional[Path] = None) -> SystemConfig:
    """
    Load configuration from a file or use defaults.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        SystemConfig instance
    """
    if config_path and config_path.exists():
        # TODO: Implement loading from JSON/YAML
        pass

    return SystemConfig()

def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config