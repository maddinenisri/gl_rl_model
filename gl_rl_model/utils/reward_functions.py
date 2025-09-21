"""
Reward functions for GRPO training of the GL RL Model.
"""

from typing import Dict, Any, Optional, List, Tuple
import re
import json
from dataclasses import dataclass
import numpy as np
from difflib import SequenceMatcher

from ..core.config import get_config

@dataclass
class RewardComponents:
    """Individual reward components."""
    syntax_reward: float = 0.0
    schema_compliance_reward: float = 0.0
    business_logic_reward: float = 0.0
    performance_reward: float = 0.0
    reasoning_quality_reward: float = 0.0
    accuracy_reward: float = 0.0  # For training with ground truth
    total_reward: float = 0.0

class RewardCalculator:
    """
    Calculate multi-dimensional rewards for SQL generation.

    Used for GRPO training to provide learning signals based on
    multiple aspects of query quality.
    """

    def __init__(self):
        """Initialize reward calculator."""
        self.config = get_config()
        self.reward_config = self.config.reward

        # Load weights from config
        self.weights = {
            "syntax": self.reward_config.syntax_weight,
            "schema": self.reward_config.schema_compliance_weight,
            "business": self.reward_config.business_logic_weight,
            "performance": self.reward_config.performance_weight,
            "reasoning": self.reward_config.reasoning_quality_weight
        }

        # Load penalties from config
        self.penalties = {
            "complexity": self.reward_config.max_query_complexity_penalty,
            "missing_reasoning": self.reward_config.missing_reasoning_penalty,
            "invalid_syntax": self.reward_config.invalid_syntax_penalty
        }

    def calculate_rewards(
        self,
        sql: str,
        reasoning: str,
        validation_result: Optional[Dict[str, Any]] = None,
        expected_sql: Optional[str] = None,
        query: Optional[str] = None
    ) -> RewardComponents:
        """
        Calculate all reward components.

        Args:
            sql: Generated SQL query
            reasoning: Generated reasoning explanation
            validation_result: Result from validator agent
            expected_sql: Ground truth SQL (for training)
            query: Original natural language query

        Returns:
            RewardComponents with all calculated rewards
        """
        components = RewardComponents()

        # Calculate individual rewards
        components.syntax_reward = self.syntax_reward(sql, validation_result)
        components.schema_compliance_reward = self.schema_compliance_reward(sql, validation_result)
        components.business_logic_reward = self.business_logic_reward(sql, validation_result)
        components.performance_reward = self.performance_reward(sql, validation_result)
        components.reasoning_quality_reward = self.reasoning_quality_reward(reasoning, query)

        # Calculate accuracy reward if ground truth is available
        if expected_sql:
            components.accuracy_reward = self.sql_accuracy_reward(sql, expected_sql)

        # Calculate total weighted reward
        components.total_reward = self._calculate_total_reward(components)

        return components

    def syntax_reward(self, sql: str, validation_result: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate syntax validity reward.

        Args:
            sql: SQL query
            validation_result: Validation result from validator

        Returns:
            Reward value (±2.0)
        """
        if not sql or len(sql.strip()) < 10:
            return self.penalties["invalid_syntax"]

        # Use validation result if available
        if validation_result:
            is_valid = validation_result.get("validation", {}).get("syntax_valid", False)
            syntax_errors = validation_result.get("errors", {}).get("syntax", [])

            if is_valid:
                return self.weights["syntax"]
            else:
                # Partial penalty based on error count
                penalty = min(len(syntax_errors) * 0.5, 2.0)
                return -penalty
        else:
            # Basic syntax checks
            sql_upper = sql.upper().strip()

            # Check for basic SQL structure
            if not sql_upper.startswith(("SELECT", "INSERT", "UPDATE", "DELETE")):
                return -1.5

            # Check for balanced parentheses
            if sql.count('(') != sql.count(')'):
                return -1.0

            # Check for required keywords
            if sql_upper.startswith("SELECT"):
                if "FROM" not in sql_upper:
                    # Allow simple value selections
                    if not any(x in sql_upper for x in ["CURRENT", "DUAL", "NOW()"]):
                        return -0.5

            # Check for common syntax patterns
            if ";;" in sql or "SELECT SELECT" in sql_upper:
                return -1.0

            # If basic checks pass
            return self.weights["syntax"] * 0.8  # Slightly reduced without full validation

    def schema_compliance_reward(self, sql: str, validation_result: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate schema compliance reward.

        Args:
            sql: SQL query
            validation_result: Validation result

        Returns:
            Reward value (±3.0)
        """
        if validation_result:
            is_compliant = validation_result.get("validation", {}).get("schema_compliant", False)
            schema_errors = validation_result.get("errors", {}).get("schema", [])

            if is_compliant:
                return self.weights["schema"]
            else:
                # Graduated penalty based on error severity
                penalty = min(len(schema_errors) * 0.6, 3.0)
                return -penalty
        else:
            # Basic schema checks using known tables
            core_tables = self.config.schema.core_tables
            sql_upper = sql.upper()

            # Check if any valid tables are mentioned
            tables_found = sum(1 for table in core_tables if table in sql_upper)

            if tables_found == 0:
                return -2.0
            elif tables_found == 1:
                return self.weights["schema"] * 0.5
            else:
                return self.weights["schema"] * 0.8

    def business_logic_reward(self, sql: str, validation_result: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate business logic compliance reward.

        Args:
            sql: SQL query
            validation_result: Validation result

        Returns:
            Reward value (±4.0)
        """
        if validation_result:
            is_valid = validation_result.get("validation", {}).get("business_logic_valid", False)
            business_errors = validation_result.get("errors", {}).get("business", [])

            if is_valid:
                # Full reward for business logic compliance
                return self.weights["business"]
            else:
                # Penalty based on business rule violations
                penalty = min(len(business_errors) * 0.8, 4.0)
                return -penalty
        else:
            # Basic business logic checks
            reward = 0.0
            sql_upper = sql.upper()

            # Check for date filtering on time-sensitive tables
            time_sensitive_tables = ["CLNTSUPP", "PROJEVISION", "PROJSTAFF"]
            for table in time_sensitive_tables:
                if table in sql_upper:
                    if any(date_kw in sql_upper for date_kw in ["DATE", "START_DATE", "END_DATE"]):
                        reward += 0.5
                    else:
                        reward -= 0.5

            # Check for status values
            if "STATUS" in sql_upper:
                # Check for valid status patterns
                valid_statuses = ["ACTIVE", "INACTIVE", "OPEN", "CLOSED", "COMPLETED"]
                if any(status in sql_upper for status in valid_statuses):
                    reward += 0.5
                else:
                    reward -= 0.3

            # Check for proper aggregation usage
            if any(agg in sql_upper for agg in ["COUNT", "SUM", "AVG", "MIN", "MAX"]):
                if "GROUP BY" in sql_upper:
                    reward += 0.8
                else:
                    # Aggregation without GROUP BY might be intentional (total aggregation)
                    reward += 0.3

            # Check for budget constraints
            if "BUDGET" in sql_upper or "COST" in sql_upper:
                if re.search(r"(?:BUDGET|COST)\s*>\s*0", sql_upper):
                    reward += 0.3
                if re.search(r"(?:BUDGET|COST)\s*<\s*-", sql_upper):
                    reward -= 1.0  # Negative budget/cost penalty

            return max(-4.0, min(4.0, reward * 2))  # Scale and cap

    def performance_reward(self, sql: str, validation_result: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate performance optimization reward.

        Args:
            sql: SQL query
            validation_result: Validation result

        Returns:
            Reward value (±1.0)
        """
        if validation_result:
            is_acceptable = validation_result.get("validation", {}).get("performance_acceptable", True)
            perf_warnings = validation_result.get("warnings", {}).get("performance", [])

            if is_acceptable and not perf_warnings:
                return self.weights["performance"]
            elif is_acceptable:
                # Minor penalty for warnings
                return self.weights["performance"] * 0.5
            else:
                # Penalty for performance issues
                return -min(len(perf_warnings) * 0.3, 1.0)
        else:
            # Basic performance checks
            reward = 0.0
            sql_upper = sql.upper()

            # Check for LIMIT clause
            if "LIMIT" in sql_upper:
                reward += 0.3

            # Check for SELECT *
            if "SELECT *" in sql_upper:
                reward -= 0.2
            else:
                reward += 0.2

            # Check for indexed columns in WHERE (simplified)
            if "WHERE" in sql_upper:
                # Common indexed columns
                if any(col in sql_upper for col in ["_ID", "_CODE", "STATUS", "_DATE"]):
                    reward += 0.3

            # Check for excessive joins
            join_count = sql_upper.count("JOIN")
            if join_count > 5:
                reward -= 0.5
            elif join_count > 3:
                reward -= 0.2

            # Check for subqueries in WHERE
            if "WHERE" in sql_upper and "SELECT" in sql_upper[sql_upper.index("WHERE"):]:
                reward -= 0.3

            return max(-1.0, min(1.0, reward))

    def reasoning_quality_reward(self, reasoning: str, query: Optional[str] = None) -> float:
        """
        Calculate reasoning quality reward.

        Args:
            reasoning: Reasoning explanation
            query: Original natural language query

        Returns:
            Reward value (±1.0)
        """
        if not reasoning:
            return self.penalties["missing_reasoning"]

        reward = 0.0

        # Check reasoning length
        if len(reasoning) < 20:
            return -0.5
        elif len(reasoning) > 50:
            reward += 0.2

        # Check for structured reasoning
        if "Step" in reasoning or "step" in reasoning:
            reward += 0.3

        # Check for key reasoning indicators
        reasoning_keywords = [
            "because", "therefore", "since", "given", "considering",
            "first", "then", "finally", "need to", "should", "must"
        ]
        keyword_count = sum(1 for kw in reasoning_keywords if kw in reasoning.lower())
        reward += min(keyword_count * 0.1, 0.3)

        # Check for SQL-specific reasoning
        sql_reasoning_terms = [
            "table", "join", "filter", "aggregate", "group",
            "column", "select", "where", "condition"
        ]
        sql_term_count = sum(1 for term in sql_reasoning_terms if term in reasoning.lower())
        reward += min(sql_term_count * 0.1, 0.3)

        # Check if reasoning mentions the query intent
        if query and len(query) > 10:
            # Simple check for query keywords in reasoning
            query_words = set(query.lower().split())
            reasoning_words = set(reasoning.lower().split())
            overlap = len(query_words & reasoning_words)
            if overlap > 2:
                reward += 0.2

        return max(-1.0, min(1.0, reward))

    def sql_accuracy_reward(self, generated_sql: str, expected_sql: str) -> float:
        """
        Calculate accuracy reward compared to ground truth.

        Args:
            generated_sql: Generated SQL query
            expected_sql: Expected SQL query

        Returns:
            Reward value (±2.0)
        """
        if not generated_sql or not expected_sql:
            return -1.0

        # Normalize SQLs for comparison
        gen_normalized = self._normalize_sql(generated_sql)
        exp_normalized = self._normalize_sql(expected_sql)

        # Exact match
        if gen_normalized == exp_normalized:
            return 2.0

        # Calculate similarity
        similarity = SequenceMatcher(None, gen_normalized, exp_normalized).ratio()

        # Check structural similarity
        gen_structure = self._extract_sql_structure(generated_sql)
        exp_structure = self._extract_sql_structure(expected_sql)

        structure_match = 0.0
        if gen_structure["query_type"] == exp_structure["query_type"]:
            structure_match += 0.3
        if set(gen_structure["tables"]) == set(exp_structure["tables"]):
            structure_match += 0.4
        if gen_structure["has_where"] == exp_structure["has_where"]:
            structure_match += 0.3

        # Combine similarity and structure match
        combined_score = (similarity * 0.6 + structure_match * 0.4)

        if combined_score > 0.9:
            return 1.8
        elif combined_score > 0.7:
            return 1.0
        elif combined_score > 0.5:
            return 0.3
        elif combined_score > 0.3:
            return -0.5
        else:
            return -1.5

    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL for comparison.

        Args:
            sql: SQL query

        Returns:
            Normalized SQL string
        """
        # Convert to uppercase
        normalized = sql.upper()

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        # Remove trailing semicolon
        normalized = normalized.rstrip(';')

        # Standardize quotes
        normalized = normalized.replace('"', "'")

        # Remove comments
        normalized = re.sub(r'--.*$', '', normalized, flags=re.MULTILINE)
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)

        return normalized.strip()

    def _extract_sql_structure(self, sql: str) -> Dict[str, Any]:
        """
        Extract structural elements from SQL.

        Args:
            sql: SQL query

        Returns:
            Dictionary with structural elements
        """
        sql_upper = sql.upper()

        structure = {
            "query_type": "",
            "tables": [],
            "has_where": "WHERE" in sql_upper,
            "has_join": "JOIN" in sql_upper,
            "has_group_by": "GROUP BY" in sql_upper,
            "has_order_by": "ORDER BY" in sql_upper,
            "has_limit": "LIMIT" in sql_upper
        }

        # Determine query type
        if sql_upper.startswith("SELECT"):
            structure["query_type"] = "SELECT"
        elif sql_upper.startswith("INSERT"):
            structure["query_type"] = "INSERT"
        elif sql_upper.startswith("UPDATE"):
            structure["query_type"] = "UPDATE"
        elif sql_upper.startswith("DELETE"):
            structure["query_type"] = "DELETE"

        # Extract tables (simplified)
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            structure["tables"].append(from_match.group(1))

        join_matches = re.finditer(r'JOIN\s+(\w+)', sql_upper)
        for match in join_matches:
            structure["tables"].append(match.group(1))

        return structure

    def _calculate_total_reward(self, components: RewardComponents) -> float:
        """
        Calculate total weighted reward.

        Args:
            components: Individual reward components

        Returns:
            Total reward value
        """
        total = (
            components.syntax_reward +
            components.schema_compliance_reward +
            components.business_logic_reward +
            components.performance_reward +
            components.reasoning_quality_reward
        )

        # Add accuracy reward if available (weighted heavily for training)
        if components.accuracy_reward != 0.0:
            total += components.accuracy_reward * 2.0

        return total

    def calculate_advantage(
        self,
        rewards: List[float],
        baseline: Optional[float] = None
    ) -> List[float]:
        """
        Calculate advantages for GRPO training.

        Args:
            rewards: List of rewards for candidates
            baseline: Baseline reward (if None, use mean)

        Returns:
            List of advantages
        """
        if baseline is None:
            baseline = np.mean(rewards)

        advantages = [r - baseline for r in rewards]

        # Normalize advantages
        advantages_array = np.array(advantages)
        if advantages_array.std() > 0:
            advantages_array = (advantages_array - advantages_array.mean()) / advantages_array.std()

        return advantages_array.tolist()

    def create_training_reward(
        self,
        sql: str,
        reasoning: str,
        validation_result: Dict[str, Any],
        expected_sql: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Create comprehensive reward for training.

        Args:
            sql: Generated SQL
            reasoning: Generated reasoning
            validation_result: Validation result
            expected_sql: Ground truth SQL
            query: Natural language query

        Returns:
            Dictionary with reward information
        """
        components = self.calculate_rewards(
            sql=sql,
            reasoning=reasoning,
            validation_result=validation_result,
            expected_sql=expected_sql,
            query=query
        )

        return {
            "total_reward": components.total_reward,
            "components": {
                "syntax": components.syntax_reward,
                "schema": components.schema_compliance_reward,
                "business": components.business_logic_reward,
                "performance": components.performance_reward,
                "reasoning": components.reasoning_quality_reward,
                "accuracy": components.accuracy_reward
            },
            "feedback": self._generate_feedback(components),
            "should_update": components.total_reward > 0  # Update policy if positive reward
        }

    def _generate_feedback(self, components: RewardComponents) -> List[str]:
        """
        Generate feedback based on reward components.

        Args:
            components: Reward components

        Returns:
            List of feedback messages
        """
        feedback = []

        if components.syntax_reward < 0:
            feedback.append("Improve SQL syntax correctness")

        if components.schema_compliance_reward < 0:
            feedback.append("Ensure tables and columns exist in schema")

        if components.business_logic_reward < 0:
            feedback.append("Follow business rules and constraints")

        if components.performance_reward < 0:
            feedback.append("Optimize query for better performance")

        if components.reasoning_quality_reward < 0:
            feedback.append("Provide clearer step-by-step reasoning")

        if components.accuracy_reward < 0:
            feedback.append("Generated SQL differs significantly from expected")

        if not feedback:
            feedback.append("Good SQL generation with proper reasoning")

        return feedback