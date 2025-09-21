"""
Dataset loader for GL RL Model training.

This module provides data loading and preprocessing for both SFT and GRPO training.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import random
import numpy as np
from collections import defaultdict
from datetime import datetime

@dataclass
class TrainingExample:
    """Single training example."""
    query: str
    sql: str
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard
    domain: str = "general"  # general, projects, resources, reporting

@dataclass
class SFTBatch:
    """Batch for supervised fine-tuning."""
    queries: List[str]
    sqls: List[str]
    reasonings: List[str]
    metadata: List[Dict[str, Any]]

@dataclass
class GRPOBatch:
    """Batch for GRPO training."""
    prompts: List[str]
    chosen_sqls: List[str]
    rejected_sqls: List[str]
    chosen_rewards: List[float]
    rejected_rewards: List[float]
    advantages: List[float]
    metadata: List[Dict[str, Any]]

class DatasetLoader:
    """
    Loader for training datasets supporting both SFT and GRPO training.

    Features:
    - Loads data from JSONL files
    - Supports curriculum learning with difficulty levels
    - Provides balanced sampling by domain
    - Handles data augmentation
    - Supports train/val/test splits
    """

    def __init__(
        self,
        data_path: str = None,
        schema_path: str = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        augment: bool = True,
        curriculum_mode: bool = True
    ):
        """
        Initialize the dataset loader.

        Args:
            data_path: Path to training data (JSONL file)
            schema_path: Path to schema directory
            val_split: Validation split ratio
            test_split: Test split ratio
            seed: Random seed for reproducibility
            augment: Whether to augment data
            curriculum_mode: Whether to enable curriculum learning
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Set default paths
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "training" / "query_pairs.jsonl"
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "data" / "schema"

        self.data_path = Path(data_path)
        self.schema_path = Path(schema_path)

        self.val_split = val_split
        self.test_split = test_split
        self.augment = augment
        self.curriculum_mode = curriculum_mode

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize data containers
        self.train_data: List[TrainingExample] = []
        self.val_data: List[TrainingExample] = []
        self.test_data: List[TrainingExample] = []
        self.all_data: List[TrainingExample] = []

        # Domain and difficulty tracking
        self.domain_examples: Dict[str, List[TrainingExample]] = defaultdict(list)
        self.difficulty_examples: Dict[str, List[TrainingExample]] = defaultdict(list)

        # Schema information
        self.schema_info: Dict[str, Any] = {}

        # Statistics
        self.stats: Dict[str, Any] = {
            "total_examples": 0,
            "train_size": 0,
            "val_size": 0,
            "test_size": 0,
            "domains": {},
            "difficulties": {},
            "avg_query_length": 0,
            "avg_sql_length": 0
        }

        # Load data
        self._load_data()
        self._load_schema()

    def _load_data(self):
        """Load and parse training data from JSONL file."""
        if not self.data_path.exists():
            self.logger.warning(f"Data file not found: {self.data_path}")
            self._generate_synthetic_data()
            return

        self.logger.info(f"Loading data from {self.data_path}")

        with open(self.data_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Create training example
                    example = TrainingExample(
                        query=data.get("query", ""),
                        sql=data.get("sql", ""),
                        reasoning=data.get("reasoning", ""),
                        metadata=data.get("metadata", {}),
                        difficulty=data.get("difficulty", "medium"),
                        domain=data.get("domain", "general")
                    )

                    # Add to containers
                    self.all_data.append(example)
                    self.domain_examples[example.domain].append(example)
                    self.difficulty_examples[example.difficulty].append(example)

                except Exception as e:
                    self.logger.warning(f"Error parsing line {line_num}: {e}")

        # Create splits
        self._create_splits()

        # Calculate statistics
        self._calculate_statistics()

        self.logger.info(f"Loaded {len(self.all_data)} examples")
        self.logger.info(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")

    def _load_schema(self):
        """Load schema information."""
        schema_file = self.schema_path / "entity_mappings.json"
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                self.schema_info = json.load(f)
                self.logger.info("Loaded schema information")
        else:
            self.logger.warning("Schema file not found")

    def _create_splits(self):
        """Create train/val/test splits."""
        # Shuffle data
        random.shuffle(self.all_data)

        total = len(self.all_data)
        val_size = int(total * self.val_split)
        test_size = int(total * self.test_split)
        train_size = total - val_size - test_size

        self.train_data = self.all_data[:train_size]
        self.val_data = self.all_data[train_size:train_size + val_size]
        self.test_data = self.all_data[train_size + val_size:]

        # Apply augmentation if enabled
        if self.augment and self.train_data:
            self._augment_training_data()

    def _augment_training_data(self):
        """Apply data augmentation to training examples."""
        augmented = []

        for example in self.train_data[:10]:  # Augment first 10 examples
            # Create variations
            variations = self._create_variations(example)
            augmented.extend(variations)

        self.train_data.extend(augmented)
        self.logger.info(f"Added {len(augmented)} augmented examples")

    def _create_variations(self, example: TrainingExample) -> List[TrainingExample]:
        """Create variations of a training example."""
        variations = []

        # Variation 1: Paraphrased query
        if "project" in example.query.lower():
            paraphrase = example.query.replace("Show", "List").replace("Find", "Get")
            variations.append(TrainingExample(
                query=paraphrase,
                sql=example.sql,
                reasoning=example.reasoning,
                metadata={**example.metadata, "augmented": True, "type": "paraphrase"},
                difficulty=example.difficulty,
                domain=example.domain
            ))

        # Variation 2: Add/remove LIMIT
        if "LIMIT" not in example.sql:
            sql_with_limit = example.sql.rstrip(";") + " LIMIT 100;"
            variations.append(TrainingExample(
                query=example.query + " (limit to 100 results)",
                sql=sql_with_limit,
                reasoning=example.reasoning + "\nAdded LIMIT for performance.",
                metadata={**example.metadata, "augmented": True, "type": "limit_added"},
                difficulty=example.difficulty,
                domain=example.domain
            ))

        return variations

    def _calculate_statistics(self):
        """Calculate dataset statistics."""
        if not self.all_data:
            return

        # Count by domain and difficulty
        for example in self.all_data:
            self.stats["domains"][example.domain] = self.stats["domains"].get(example.domain, 0) + 1
            self.stats["difficulties"][example.difficulty] = self.stats["difficulties"].get(example.difficulty, 0) + 1

        # Calculate averages
        query_lengths = [len(ex.query) for ex in self.all_data]
        sql_lengths = [len(ex.sql) for ex in self.all_data]

        self.stats.update({
            "total_examples": len(self.all_data),
            "train_size": len(self.train_data),
            "val_size": len(self.val_data),
            "test_size": len(self.test_data),
            "avg_query_length": np.mean(query_lengths) if query_lengths else 0,
            "avg_sql_length": np.mean(sql_lengths) if sql_lengths else 0
        })

    def _generate_synthetic_data(self):
        """Generate synthetic training data if file doesn't exist."""
        self.logger.info("Generating synthetic training data")

        synthetic_examples = [
            TrainingExample(
                query="Show all active projects",
                sql="SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active';",
                reasoning="Filter projects table by active status",
                difficulty="easy",
                domain="projects"
            ),
            TrainingExample(
                query="Find projects with budget over 100000",
                sql="SELECT Project_Code, Project_Name, Budget FROM PAC_MNT_PROJECTS WHERE Budget > 100000;",
                reasoning="Filter projects by budget threshold",
                difficulty="easy",
                domain="projects"
            ),
            TrainingExample(
                query="List all companies with their project count",
                sql="""SELECT c.Company_Name, COUNT(p.Project_ID) as Project_Count
                       FROM SRM_COMPANIES c
                       LEFT JOIN SRM_PROJECTS p ON c.Company_ID = p.Company_ID
                       GROUP BY c.Company_Name;""",
                reasoning="Join companies and projects, count projects per company",
                difficulty="medium",
                domain="reporting"
            ),
            TrainingExample(
                query="Show resource allocation by project",
                sql="""SELECT p.Project_Name, s.Staff_Code, s.Allocation_Percent
                       FROM PAC_MNT_PROJECTS p
                       INNER JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code
                       WHERE s.Allocation_Percent > 0
                       ORDER BY p.Project_Name, s.Allocation_Percent DESC;""",
                reasoning="Join projects with staff allocations, filter non-zero allocations",
                difficulty="medium",
                domain="resources"
            ),
            TrainingExample(
                query="Find projects exceeding budget with contractor details",
                sql="""SELECT p.Project_Name, p.Budget, p.Actual_Cost,
                              c.Company_Name, ct.Contract_Value
                       FROM PAC_MNT_PROJECTS p
                       INNER JOIN PROJCNTRTS ct ON p.Project_Code = ct.Project_Code
                       INNER JOIN SRM_COMPANIES c ON ct.Company_Code = c.Company_Code
                       WHERE p.Actual_Cost > p.Budget
                       ORDER BY (p.Actual_Cost - p.Budget) DESC;""",
                reasoning="Complex join to find over-budget projects with contractor information",
                difficulty="hard",
                domain="reporting"
            )
        ]

        self.all_data = synthetic_examples
        self._create_splits()
        self._calculate_statistics()

    def get_sft_batch(self, batch_size: int = 8, split: str = "train") -> Optional[SFTBatch]:
        """
        Get a batch for supervised fine-tuning.

        Args:
            batch_size: Size of the batch
            split: Data split to use (train/val/test)

        Returns:
            SFTBatch or None if no data available
        """
        # Select appropriate data split
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")

        if not data:
            return None

        # Sample batch
        batch_examples = random.sample(data, min(batch_size, len(data)))

        return SFTBatch(
            queries=[ex.query for ex in batch_examples],
            sqls=[ex.sql for ex in batch_examples],
            reasonings=[ex.reasoning for ex in batch_examples],
            metadata=[ex.metadata for ex in batch_examples]
        )

    def get_grpo_batch(
        self,
        batch_size: int = 4,
        num_candidates: int = 4,
        reward_evaluator=None
    ) -> Optional[GRPOBatch]:
        """
        Get a batch for GRPO training.

        Args:
            batch_size: Number of prompts
            num_candidates: Number of candidates per prompt
            reward_evaluator: Reward evaluator for scoring candidates

        Returns:
            GRPOBatch or None if no data available
        """
        if not self.train_data or not reward_evaluator:
            return None

        # Sample prompts
        batch_examples = random.sample(self.train_data, min(batch_size, len(self.train_data)))

        prompts = []
        chosen_sqls = []
        rejected_sqls = []
        chosen_rewards = []
        rejected_rewards = []
        advantages = []
        metadata = []

        for example in batch_examples:
            # For now, use the ground truth as chosen and a corrupted version as rejected
            prompts.append(example.query)
            chosen_sqls.append(example.sql)

            # Create a rejected candidate (simplified version)
            rejected_sql = "SELECT * FROM PAC_MNT_PROJECTS;"
            rejected_sqls.append(rejected_sql)

            # Simulate rewards (in practice, use reward_evaluator)
            chosen_rewards.append(5.0 + np.random.normal(0, 0.5))
            rejected_rewards.append(-2.0 + np.random.normal(0, 0.5))

            # Calculate advantage
            advantage = chosen_rewards[-1] - rejected_rewards[-1]
            advantages.append(advantage)

            metadata.append(example.metadata)

        return GRPOBatch(
            prompts=prompts,
            chosen_sqls=chosen_sqls,
            rejected_sqls=rejected_sqls,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            advantages=advantages,
            metadata=metadata
        )

    def get_curriculum_batch(
        self,
        batch_size: int = 8,
        current_difficulty: str = "easy",
        split: str = "train"
    ) -> Optional[SFTBatch]:
        """
        Get a batch for curriculum learning.

        Args:
            batch_size: Size of the batch
            current_difficulty: Current difficulty level
            split: Data split to use

        Returns:
            SFTBatch focusing on current difficulty
        """
        if split != "train" or not self.curriculum_mode:
            return self.get_sft_batch(batch_size, split)

        # Get examples for current difficulty
        difficulty_data = self.difficulty_examples.get(current_difficulty, [])

        if not difficulty_data:
            # Fall back to regular batch
            return self.get_sft_batch(batch_size, split)

        # Sample from difficulty-specific data
        batch_examples = random.sample(difficulty_data, min(batch_size, len(difficulty_data)))

        # If not enough, add some from next difficulty
        if len(batch_examples) < batch_size:
            next_difficulty = {"easy": "medium", "medium": "hard", "hard": "hard"}[current_difficulty]
            next_data = self.difficulty_examples.get(next_difficulty, [])

            if next_data:
                additional = min(batch_size - len(batch_examples), len(next_data))
                batch_examples.extend(random.sample(next_data, additional))

        return SFTBatch(
            queries=[ex.query for ex in batch_examples],
            sqls=[ex.sql for ex in batch_examples],
            reasonings=[ex.reasoning for ex in batch_examples],
            metadata=[ex.metadata for ex in batch_examples]
        )

    def iterate_epochs(
        self,
        num_epochs: int,
        batch_size: int,
        split: str = "train",
        shuffle: bool = True
    ) -> Iterator[Tuple[int, SFTBatch]]:
        """
        Iterate over epochs yielding batches.

        Args:
            num_epochs: Number of epochs
            batch_size: Batch size
            split: Data split
            shuffle: Whether to shuffle data each epoch

        Yields:
            Tuple of (epoch_num, batch)
        """
        # Select data
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")

        for epoch in range(num_epochs):
            # Shuffle if requested
            if shuffle:
                data_copy = data.copy()
                random.shuffle(data_copy)
            else:
                data_copy = data

            # Generate batches
            for i in range(0, len(data_copy), batch_size):
                batch_examples = data_copy[i:i + batch_size]

                batch = SFTBatch(
                    queries=[ex.query for ex in batch_examples],
                    sqls=[ex.sql for ex in batch_examples],
                    reasonings=[ex.reasoning for ex in batch_examples],
                    metadata=[ex.metadata for ex in batch_examples]
                )

                yield epoch, batch

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.stats.copy()

    def save_processed_data(self, output_path: str):
        """
        Save processed data to file.

        Args:
            output_path: Path to save processed data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "train": [vars(ex) for ex in self.train_data],
            "val": [vars(ex) for ex in self.val_data],
            "test": [vars(ex) for ex in self.test_data],
            "stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)

        self.logger.info(f"Saved processed data to {output_path}")

    def load_processed_data(self, input_path: str):
        """
        Load processed data from file.

        Args:
            input_path: Path to load processed data from
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_path}")

        with open(input_path, 'r') as f:
            data = json.load(f)

        # Reconstruct examples
        self.train_data = [TrainingExample(**ex) for ex in data["train"]]
        self.val_data = [TrainingExample(**ex) for ex in data["val"]]
        self.test_data = [TrainingExample(**ex) for ex in data["test"]]

        # Reconstruct all_data and domain/difficulty mappings
        self.all_data = self.train_data + self.val_data + self.test_data

        for example in self.all_data:
            self.domain_examples[example.domain].append(example)
            self.difficulty_examples[example.difficulty].append(example)

        self.stats = data["stats"]

        self.logger.info(f"Loaded processed data from {input_path}")

    def get_balanced_batch(
        self,
        batch_size: int = 8,
        split: str = "train",
        balance_by: str = "domain"
    ) -> Optional[SFTBatch]:
        """
        Get a balanced batch by domain or difficulty.

        Args:
            batch_size: Size of the batch
            split: Data split to use
            balance_by: Balance by 'domain' or 'difficulty'

        Returns:
            Balanced SFTBatch
        """
        if balance_by == "domain":
            categories = self.domain_examples
        elif balance_by == "difficulty":
            categories = self.difficulty_examples
        else:
            return self.get_sft_batch(batch_size, split)

        # Calculate examples per category
        num_categories = len(categories)
        if num_categories == 0:
            return None

        per_category = batch_size // num_categories
        remainder = batch_size % num_categories

        batch_examples = []

        for i, (category, examples) in enumerate(categories.items()):
            # Filter by split
            if split == "train":
                filtered = [ex for ex in examples if ex in self.train_data]
            elif split == "val":
                filtered = [ex for ex in examples if ex in self.val_data]
            elif split == "test":
                filtered = [ex for ex in examples if ex in self.test_data]
            else:
                filtered = examples

            if not filtered:
                continue

            # Sample from this category
            sample_size = per_category + (1 if i < remainder else 0)
            sampled = random.sample(filtered, min(sample_size, len(filtered)))
            batch_examples.extend(sampled)

        if not batch_examples:
            return None

        # Shuffle the balanced batch
        random.shuffle(batch_examples)

        return SFTBatch(
            queries=[ex.query for ex in batch_examples],
            sqls=[ex.sql for ex in batch_examples],
            reasonings=[ex.reasoning for ex in batch_examples],
            metadata=[ex.metadata for ex in batch_examples]
        )