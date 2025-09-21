"""
Reward Evaluator Agent for orchestrating reward calculation in GRPO training.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
import numpy as np
from datetime import datetime

from ..core.base_agent import BaseAgent, AgentStatus
from ..core.config import get_config
from ..agents.validator import ValidatorAgent
from ..utils.reward_functions import RewardCalculator, RewardComponents
from ..utils.sql_validator import SQLValidator

@dataclass
class EvaluationResult:
    """Complete evaluation result for a generated SQL query."""
    query: str
    sql: str
    reasoning: str
    rewards: RewardComponents
    validation: Dict[str, Any]
    training_signal: Dict[str, Any]
    feedback: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchEvaluationResult:
    """Results for batch evaluation of multiple candidates."""
    prompt: str
    candidates: List[EvaluationResult]
    best_candidate_idx: int
    advantages: List[float]
    baseline_reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class RewardEvaluatorAgent(BaseAgent):
    """
    Agent responsible for orchestrating reward evaluation for GRPO training.

    This agent coordinates validation and reward calculation, providing
    training signals for reinforcement learning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the reward evaluator agent."""
        super().__init__("reward_evaluator", config)
        self.system_config = get_config()
        self.validator_agent: Optional[ValidatorAgent] = None
        self.reward_calculator = RewardCalculator()
        self.sql_validator = SQLValidator()
        self.evaluation_cache: Dict[str, EvaluationResult] = {}
        self.batch_size = self.system_config.agent.reward_evaluator_batch_size

    async def initialize(self) -> bool:
        """
        Initialize the reward evaluator agent.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Reward Evaluator Agent")

            # Initialize validator agent
            self.validator_agent = ValidatorAgent()
            await self.validator_agent.initialize()

            # Clear cache
            self.evaluation_cache.clear()

            self.status = AgentStatus.IDLE
            self.logger.info("Reward Evaluator Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Reward Evaluator: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the reward evaluator agent.

        Returns:
            True if shutdown successful, False otherwise
        """
        try:
            self.logger.info("Shutting down Reward Evaluator Agent")

            if self.validator_agent:
                await self.validator_agent.shutdown()

            self.evaluation_cache.clear()
            self.status = AgentStatus.IDLE
            return True
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an evaluation request.

        Args:
            input_data: Dictionary containing:
                - query: Natural language query
                - sql: Generated SQL
                - reasoning: Generated reasoning
                - expected_sql: Optional ground truth SQL
                - mode: 'single' or 'batch'
                - candidates: List of candidates (for batch mode)

        Returns:
            Dictionary containing evaluation results
        """
        try:
            mode = input_data.get("mode", "single")

            if mode == "batch":
                return await self._process_batch_evaluation(input_data)
            else:
                return await self._process_single_evaluation(input_data)

        except Exception as e:
            self.logger.error(f"Error processing evaluation: {e}")
            return {
                "error": str(e),
                "success": False
            }

    async def _process_single_evaluation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single SQL generation evaluation.

        Args:
            input_data: Evaluation request data

        Returns:
            Single evaluation result
        """
        query = input_data.get("query", "")
        sql = input_data.get("sql", "")
        reasoning = input_data.get("reasoning", "")
        expected_sql = input_data.get("expected_sql")
        schema_context = input_data.get("schema_context", {})

        # Check cache
        cache_key = self._generate_cache_key(query, sql)
        if cache_key in self.evaluation_cache:
            self.logger.info("Using cached evaluation result")
            cached = self.evaluation_cache[cache_key]
            return self._format_evaluation_result(cached)

        # Perform evaluation
        result = await self.evaluate_generation(
            query=query,
            sql=sql,
            reasoning=reasoning,
            expected_sql=expected_sql,
            schema_context=schema_context
        )

        # Cache result
        self.evaluation_cache[cache_key] = result

        return self._format_evaluation_result(result)

    async def _process_batch_evaluation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process batch evaluation for GRPO training.

        Args:
            input_data: Batch evaluation request

        Returns:
            Batch evaluation results
        """
        prompt = input_data.get("prompt", "")
        candidates = input_data.get("candidates", [])
        expected_sql = input_data.get("expected_sql")
        schema_context = input_data.get("schema_context", {})

        if not candidates:
            return {
                "error": "No candidates provided for batch evaluation",
                "success": False
            }

        # Evaluate all candidates
        batch_result = await self.evaluate_batch(
            prompt=prompt,
            candidates=candidates,
            expected_sql=expected_sql,
            schema_context=schema_context
        )

        return self._format_batch_result(batch_result)

    async def evaluate_generation(
        self,
        query: str,
        sql: str,
        reasoning: str,
        expected_sql: Optional[str] = None,
        schema_context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single SQL generation.

        Args:
            query: Natural language query
            sql: Generated SQL
            reasoning: Generated reasoning
            expected_sql: Optional ground truth SQL
            schema_context: Optional schema context

        Returns:
            Complete evaluation result
        """
        # Step 1: Validate SQL
        validation_result = await self.validator_agent.process({
            "sql": sql,
            "schema_context": schema_context or {},
            "strict_mode": False,
            "check_performance": True,
            "check_security": True
        })

        # Step 2: Calculate rewards
        rewards = self.reward_calculator.calculate_rewards(
            sql=sql,
            reasoning=reasoning,
            validation_result=validation_result,
            expected_sql=expected_sql,
            query=query
        )

        # Step 3: Generate training signal
        training_signal = self._prepare_training_signal(
            rewards=rewards,
            validation_result=validation_result
        )

        # Step 4: Generate feedback
        feedback = self._generate_comprehensive_feedback(
            rewards=rewards,
            validation_result=validation_result
        )

        # Create evaluation result
        return EvaluationResult(
            query=query,
            sql=sql,
            reasoning=reasoning,
            rewards=rewards,
            validation=validation_result,
            training_signal=training_signal,
            feedback=feedback,
            metadata={
                "has_expected_sql": expected_sql is not None,
                "total_reward": rewards.total_reward,
                "is_valid": validation_result.get("is_valid", False)
            }
        )

    async def evaluate_batch(
        self,
        prompt: str,
        candidates: List[Dict[str, str]],
        expected_sql: Optional[str] = None,
        schema_context: Optional[Dict[str, Any]] = None
    ) -> BatchEvaluationResult:
        """
        Evaluate a batch of candidates for GRPO training.

        Args:
            prompt: The original query prompt
            candidates: List of candidate SQL generations
            expected_sql: Optional ground truth SQL
            schema_context: Optional schema context

        Returns:
            Batch evaluation results with advantages
        """
        evaluations = []

        # Evaluate all candidates concurrently
        tasks = []
        for candidate in candidates:
            task = self.evaluate_generation(
                query=prompt,
                sql=candidate.get("sql", ""),
                reasoning=candidate.get("reasoning", ""),
                expected_sql=expected_sql,
                schema_context=schema_context
            )
            tasks.append(task)

        evaluations = await asyncio.gather(*tasks)

        # Calculate rewards and advantages
        rewards = [eval_result.rewards.total_reward for eval_result in evaluations]
        baseline_reward = np.mean(rewards)
        advantages = self.reward_calculator.calculate_advantage(rewards, baseline_reward)

        # Find best candidate
        best_idx = np.argmax(rewards)

        return BatchEvaluationResult(
            prompt=prompt,
            candidates=evaluations,
            best_candidate_idx=int(best_idx),
            advantages=advantages,
            baseline_reward=float(baseline_reward),
            metadata={
                "num_candidates": len(candidates),
                "reward_variance": float(np.var(rewards)),
                "max_reward": float(np.max(rewards)),
                "min_reward": float(np.min(rewards))
            }
        )

    def _prepare_training_signal(
        self,
        rewards: RewardComponents,
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare training signal for GRPO.

        Args:
            rewards: Calculated rewards
            validation_result: Validation results

        Returns:
            Training signal dictionary
        """
        # Determine if this example should be used for training
        should_train = rewards.total_reward > 0 or (
            validation_result.get("is_valid", False) and
            rewards.total_reward > -1.0
        )

        # Calculate loss weight based on reward magnitude
        loss_weight = abs(rewards.total_reward) / 10.0  # Normalize
        loss_weight = min(max(loss_weight, 0.1), 2.0)  # Clamp

        return {
            "should_train": should_train,
            "loss_weight": loss_weight,
            "reward_components": {
                "syntax": rewards.syntax_reward,
                "schema": rewards.schema_compliance_reward,
                "business": rewards.business_logic_reward,
                "performance": rewards.performance_reward,
                "reasoning": rewards.reasoning_quality_reward,
                "accuracy": rewards.accuracy_reward
            },
            "total_reward": rewards.total_reward,
            "is_valid": validation_result.get("is_valid", False),
            "confidence": validation_result.get("metadata", {}).get("confidence", 0.0)
        }

    def _generate_comprehensive_feedback(
        self,
        rewards: RewardComponents,
        validation_result: Dict[str, Any]
    ) -> List[str]:
        """
        Generate comprehensive feedback based on evaluation.

        Args:
            rewards: Calculated rewards
            validation_result: Validation results

        Returns:
            List of feedback messages
        """
        feedback = []

        # Reward-based feedback
        if rewards.syntax_reward < 0:
            feedback.append("❌ SQL syntax needs improvement")
        elif rewards.syntax_reward > 1:
            feedback.append("✅ SQL syntax is correct")

        if rewards.schema_compliance_reward < 0:
            feedback.append("❌ Schema compliance issues detected")
        elif rewards.schema_compliance_reward > 1:
            feedback.append("✅ Schema compliance is good")

        if rewards.business_logic_reward < 0:
            feedback.append("❌ Business logic violations found")
        elif rewards.business_logic_reward > 2:
            feedback.append("✅ Business logic correctly applied")

        if rewards.performance_reward < 0:
            feedback.append("⚠️ Performance could be optimized")
        elif rewards.performance_reward > 0.5:
            feedback.append("✅ Query is well-optimized")

        if rewards.reasoning_quality_reward < 0:
            feedback.append("❌ Reasoning needs more detail")
        elif rewards.reasoning_quality_reward > 0.5:
            feedback.append("✅ Clear reasoning provided")

        # Validation-based feedback
        errors = validation_result.get("errors", {})
        if errors.get("syntax"):
            feedback.append(f"Syntax errors: {', '.join(errors['syntax'][:2])}")

        if errors.get("schema"):
            feedback.append(f"Schema errors: {', '.join(errors['schema'][:2])}")

        if errors.get("business"):
            feedback.append(f"Business rule violations: {', '.join(errors['business'][:2])}")

        # Add suggestions if available
        suggestions = validation_result.get("suggestions", [])
        if suggestions:
            feedback.append(f"💡 Suggestion: {suggestions[0]}")

        # Overall assessment
        if rewards.total_reward > 5:
            feedback.insert(0, "🌟 Excellent SQL generation!")
        elif rewards.total_reward > 2:
            feedback.insert(0, "👍 Good SQL generation with minor issues")
        elif rewards.total_reward > 0:
            feedback.insert(0, "📝 Acceptable SQL with room for improvement")
        else:
            feedback.insert(0, "⚠️ SQL needs significant improvement")

        return feedback

    def _generate_cache_key(self, query: str, sql: str) -> str:
        """Generate cache key for evaluation results."""
        import hashlib
        combined = f"{query}|{sql}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _format_evaluation_result(self, result: EvaluationResult) -> Dict[str, Any]:
        """Format evaluation result for output."""
        return {
            "query": result.query,
            "sql": result.sql,
            "reasoning": result.reasoning,
            "rewards": {
                "total": result.rewards.total_reward,
                "components": {
                    "syntax": result.rewards.syntax_reward,
                    "schema": result.rewards.schema_compliance_reward,
                    "business": result.rewards.business_logic_reward,
                    "performance": result.rewards.performance_reward,
                    "reasoning": result.rewards.reasoning_quality_reward,
                    "accuracy": result.rewards.accuracy_reward
                }
            },
            "validation": result.validation,
            "training_signal": result.training_signal,
            "feedback": result.feedback,
            "metadata": result.metadata,
            "timestamp": result.timestamp.isoformat(),
            "success": True
        }

    def _format_batch_result(self, result: BatchEvaluationResult) -> Dict[str, Any]:
        """Format batch evaluation result for output."""
        return {
            "prompt": result.prompt,
            "num_candidates": len(result.candidates),
            "best_candidate": {
                "index": result.best_candidate_idx,
                "sql": result.candidates[result.best_candidate_idx].sql,
                "reward": result.candidates[result.best_candidate_idx].rewards.total_reward
            },
            "advantages": result.advantages,
            "baseline_reward": result.baseline_reward,
            "candidates": [
                {
                    "sql": candidate.sql[:100] + "..." if len(candidate.sql) > 100 else candidate.sql,
                    "total_reward": candidate.rewards.total_reward,
                    "is_valid": candidate.validation.get("is_valid", False)
                }
                for candidate in result.candidates
            ],
            "metadata": result.metadata,
            "success": True
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics.

        Returns:
            Statistics dictionary
        """
        if not self.evaluation_cache:
            return {
                "total_evaluations": 0,
                "cache_size": 0
            }

        all_rewards = [
            eval_result.rewards.total_reward
            for eval_result in self.evaluation_cache.values()
        ]

        valid_count = sum(
            1 for eval_result in self.evaluation_cache.values()
            if eval_result.validation.get("is_valid", False)
        )

        return {
            "total_evaluations": len(self.evaluation_cache),
            "cache_size": len(self.evaluation_cache),
            "valid_sql_percentage": (valid_count / len(self.evaluation_cache)) * 100,
            "average_reward": np.mean(all_rewards) if all_rewards else 0,
            "max_reward": np.max(all_rewards) if all_rewards else 0,
            "min_reward": np.min(all_rewards) if all_rewards else 0,
            "reward_std": np.std(all_rewards) if all_rewards else 0
        }

    def clear_cache(self):
        """Clear the evaluation cache."""
        self.evaluation_cache.clear()
        self.logger.info("Evaluation cache cleared")