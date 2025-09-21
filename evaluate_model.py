#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for GL RL Model
Evaluates SQL generation accuracy, quality, and domain-specificity
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import torch
import asyncio
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
from gl_rl_model.training.schema_loader import SchemaLoader
from gl_rl_model.agents.reward_evaluator import RewardEvaluatorAgent
from gl_rl_model.utils.sql_validator import SQLValidator

@dataclass
class EvaluationMetrics:
    """Metrics for model evaluation."""
    total_queries: int = 0
    correct_sql: int = 0
    syntactically_valid: int = 0
    uses_domain_tables: int = 0
    has_reasoning: int = 0
    avg_reward_score: float = 0.0
    avg_generation_time: float = 0.0
    error_count: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct_sql / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def syntax_validity_rate(self) -> float:
        return self.syntactically_valid / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def domain_specificity_rate(self) -> float:
        return self.uses_domain_tables / self.total_queries if self.total_queries > 0 else 0.0

class ModelEvaluator:
    """Comprehensive model evaluation framework."""

    def __init__(self, model_checkpoint: str = None):
        """Initialize evaluator with optional model checkpoint."""
        self.model_checkpoint = model_checkpoint
        self.model_wrapper = None
        self.schema_loader = SchemaLoader()
        self.prompt_templates = SQLPromptTemplates()
        self.sql_validator = SQLValidator()
        self.reward_evaluator = None
        self.results = []

        # Domain-specific tables to check for
        self.domain_tables = [
            "PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJCNTRTS",
            "PROJSTAFF", "PAC_MNT_RESOURCES", "SRM_CONTACTS"
        ]

    async def initialize(self):
        """Initialize the evaluation framework."""
        print("Initializing Evaluation Framework...")

        # Check device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Using Mac GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("✅ Using CUDA GPU")
        else:
            self.device = "cpu"
            print("⚠️ Using CPU")

        # Initialize model
        print("Loading model...")
        self.model_wrapper = QwenModelWrapper(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            use_lora=True,
            load_in_8bit=False,
            device_map=None if self.device == "mps" else self.device
        )

        # Load model
        self.model_wrapper.load_model()

        # Move to device if MPS
        if self.device == "mps":
            self.model_wrapper.model = self.model_wrapper.model.to(self.device)

        # Load checkpoint if specified
        if self.model_checkpoint:
            checkpoint_path = Path(self.model_checkpoint)
            if checkpoint_path.exists():
                print(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if "lora_state_dict" in checkpoint:
                    self.model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])
                    print("✅ Checkpoint loaded")
            else:
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")

        # Initialize reward evaluator
        self.reward_evaluator = RewardEvaluatorAgent()
        await self.reward_evaluator.initialize()

        print("✅ Evaluation framework initialized")

    async def evaluate_query(self, query: str, expected_sql: str = None) -> Dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            query: Natural language query
            expected_sql: Optional expected SQL for comparison

        Returns:
            Evaluation results dictionary
        """
        import time
        start_time = time.time()

        # Get schema context
        schema_context = self.schema_loader.get_schema_context(query)

        # Create prompt
        prompt = self.prompt_templates.zero_shot_sql_generation(
            query=query,
            schema_context=schema_context,
            business_context="Use exact table names from the schema."
        )

        # Generate SQL
        gen_params = GenerationParams(
            temperature=0.1,
            max_new_tokens=256,
            do_sample=False
        )

        try:
            result = self.model_wrapper.generate(prompt, gen_params)
            sql, reasoning = self.model_wrapper.extract_sql_and_reasoning(result)

            if not sql:
                sql = self.model_wrapper.extract_sql(result)

            generation_time = time.time() - start_time

            # Validate SQL syntax
            is_valid_syntax = self.sql_validator.validate_syntax(sql) if sql else False

            # Check for domain tables
            uses_domain_tables = any(table in (sql or "") for table in self.domain_tables)

            # Calculate reward if possible
            reward_score = 0.0
            if self.reward_evaluator and sql:
                eval_result = await self.reward_evaluator.process({
                    "query": query,
                    "sql": sql or "",
                    "reasoning": reasoning or "",
                    "expected_sql": expected_sql,
                    "mode": "single"
                })

                if "rewards" in eval_result and hasattr(eval_result["rewards"], "total_reward"):
                    reward_score = eval_result["rewards"].total_reward

            # Check correctness (if expected SQL provided)
            is_correct = False
            if expected_sql and sql:
                # Simple check - could be enhanced with semantic comparison
                is_correct = self._compare_sql(sql, expected_sql)

            result_dict = {
                "query": query,
                "generated_sql": sql,
                "reasoning": reasoning,
                "expected_sql": expected_sql,
                "is_correct": is_correct,
                "is_valid_syntax": is_valid_syntax,
                "uses_domain_tables": uses_domain_tables,
                "has_reasoning": bool(reasoning),
                "reward_score": reward_score,
                "generation_time": generation_time,
                "error": None
            }

        except Exception as e:
            result_dict = {
                "query": query,
                "generated_sql": None,
                "reasoning": None,
                "expected_sql": expected_sql,
                "is_correct": False,
                "is_valid_syntax": False,
                "uses_domain_tables": False,
                "has_reasoning": False,
                "reward_score": 0.0,
                "generation_time": time.time() - start_time,
                "error": str(e)
            }

        self.results.append(result_dict)
        return result_dict

    def _compare_sql(self, generated: str, expected: str) -> bool:
        """
        Compare generated SQL with expected SQL.

        Simple comparison for now - could be enhanced with semantic analysis.
        """
        if not generated or not expected:
            return False

        # Normalize for comparison
        gen_normalized = generated.strip().upper().replace(";", "")
        exp_normalized = expected.strip().upper().replace(";", "")

        # Remove extra whitespace
        gen_normalized = " ".join(gen_normalized.split())
        exp_normalized = " ".join(exp_normalized.split())

        return gen_normalized == exp_normalized

    async def evaluate_test_set(self, test_queries: List[Dict[str, str]]) -> EvaluationMetrics:
        """
        Evaluate a test set of queries.

        Args:
            test_queries: List of {"query": ..., "sql": ...} dictionaries

        Returns:
            Evaluation metrics
        """
        metrics = EvaluationMetrics()
        metrics.total_queries = len(test_queries)

        print(f"\nEvaluating {len(test_queries)} queries...")
        print("="*50)

        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            expected_sql = test_case.get("sql", "")

            print(f"\n[{i}/{len(test_queries)}] Query: {query}")

            result = await self.evaluate_query(query, expected_sql)

            # Update metrics
            if result["is_correct"]:
                metrics.correct_sql += 1
            if result["is_valid_syntax"]:
                metrics.syntactically_valid += 1
            if result["uses_domain_tables"]:
                metrics.uses_domain_tables += 1
            if result["has_reasoning"]:
                metrics.has_reasoning += 1
            if result["error"]:
                metrics.error_count += 1

            metrics.avg_reward_score += result["reward_score"]
            metrics.avg_generation_time += result["generation_time"]

            # Show result
            if result["generated_sql"]:
                print(f"  Generated: {result['generated_sql'][:100]}...")
                print(f"  Valid syntax: {'✅' if result['is_valid_syntax'] else '❌'}")
                print(f"  Domain tables: {'✅' if result['uses_domain_tables'] else '❌'}")
                if expected_sql:
                    print(f"  Correct: {'✅' if result['is_correct'] else '❌'}")
            else:
                print(f"  Error: {result['error']}")

        # Calculate averages
        metrics.avg_reward_score /= max(metrics.total_queries, 1)
        metrics.avg_generation_time /= max(metrics.total_queries, 1)

        return metrics

    def generate_report(self, metrics: EvaluationMetrics, output_file: str = None):
        """Generate evaluation report."""
        report = []
        report.append("\n" + "="*70)
        report.append("EVALUATION REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.model_checkpoint:
            report.append(f"Model: {self.model_checkpoint}")

        report.append("\n" + "-"*70)
        report.append("METRICS SUMMARY")
        report.append("-"*70)

        report.append(f"Total Queries: {metrics.total_queries}")
        report.append(f"Accuracy: {metrics.accuracy:.2%}")
        report.append(f"Syntax Validity: {metrics.syntax_validity_rate:.2%}")
        report.append(f"Domain Specificity: {metrics.domain_specificity_rate:.2%}")
        report.append(f"Average Reward Score: {metrics.avg_reward_score:.2f}")
        report.append(f"Average Generation Time: {metrics.avg_generation_time:.2f}s")
        report.append(f"Queries with Reasoning: {metrics.has_reasoning}/{metrics.total_queries}")
        report.append(f"Errors: {metrics.error_count}")

        # Detailed results
        if self.results:
            report.append("\n" + "-"*70)
            report.append("DETAILED RESULTS")
            report.append("-"*70)

            for i, result in enumerate(self.results[:10], 1):  # Show first 10
                report.append(f"\n{i}. Query: {result['query']}")
                if result['generated_sql']:
                    report.append(f"   SQL: {result['generated_sql'][:100]}...")
                    report.append(f"   Metrics: Syntax={'✓' if result['is_valid_syntax'] else '✗'}, "
                                f"Domain={'✓' if result['uses_domain_tables'] else '✗'}, "
                                f"Reward={result['reward_score']:.1f}")
                else:
                    report.append(f"   Error: {result['error']}")

        report.append("\n" + "="*70)

        report_text = "\n".join(report)

        # Print report
        print(report_text)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

            # Also save detailed results as JSON
            json_file = output_file.replace('.txt', '.json')
            with open(json_file, 'w') as f:
                json.dump({
                    "metrics": asdict(metrics),
                    "results": self.results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

            print(f"\nReport saved to: {output_file}")
            print(f"Detailed results saved to: {json_file}")

        return report_text

async def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GL RL Model")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--test-file", type=str, help="Test queries file (JSONL)")
    parser.add_argument("--output", type=str, default="evaluation_report.txt", help="Output report file")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")

    args = parser.parse_args()

    # Default test queries if no file provided
    default_test_queries = [
        {"query": "Show all active projects", "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'"},
        {"query": "Find projects with budget over 100000", "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > 100000"},
        {"query": "List all companies", "sql": "SELECT * FROM SRM_COMPANIES"},
        {"query": "Count projects per company", "sql": "SELECT Company_Code, COUNT(*) as project_count FROM PROJCNTRTS GROUP BY Company_Code"},
        {"query": "Show staff allocations", "sql": "SELECT * FROM PROJSTAFF"},
        {"query": "Find high-budget projects with staff", "sql": "SELECT p.*, s.* FROM PAC_MNT_PROJECTS p JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code WHERE p.Budget > 500000"},
        {"query": "List company contacts", "sql": "SELECT c.Company_Name, con.Contact_Name, con.Email FROM SRM_COMPANIES c JOIN SRM_CONTACTS con ON c.Company_ID = con.Company_ID"},
        {"query": "Show project revenue by department", "sql": "SELECT Department, SUM(Revenue) as total_revenue FROM PAC_MNT_PROJECTS GROUP BY Department"},
        {"query": "Find overbudget projects", "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Actual_Cost > Budget"},
        {"query": "Get resource utilization", "sql": "SELECT Resource_Code, SUM(Allocation_Percent) as total_allocation FROM PROJSTAFF GROUP BY Resource_Code"},
    ]

    # Load test queries
    if args.test_file and Path(args.test_file).exists():
        test_queries = []
        with open(args.test_file, 'r') as f:
            for line in f:
                test_queries.append(json.loads(line))
    else:
        test_queries = default_test_queries

    if args.compare:
        # Compare multiple model versions
        print("="*70)
        print("COMPARATIVE EVALUATION")
        print("="*70)

        checkpoints = [
            ("Base Model", None),
            ("SFT Model", "./checkpoints/sft/best.pt"),
            ("GRPO Model", "./checkpoints/grpo/best.pt")
        ]

        all_metrics = {}

        for name, checkpoint in checkpoints:
            if checkpoint and not Path(checkpoint).exists():
                print(f"\n⚠️ Skipping {name} - checkpoint not found")
                continue

            print(f"\n{'='*50}")
            print(f"Evaluating: {name}")
            print('='*50)

            evaluator = ModelEvaluator(model_checkpoint=checkpoint)
            await evaluator.initialize()

            metrics = await evaluator.evaluate_test_set(test_queries)
            all_metrics[name] = metrics

            # Cleanup
            await evaluator.reward_evaluator.shutdown()

        # Generate comparison report
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)

        for name, metrics in all_metrics.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {metrics.accuracy:.2%}")
            print(f"  Syntax Valid: {metrics.syntax_validity_rate:.2%}")
            print(f"  Domain Specific: {metrics.domain_specificity_rate:.2%}")
            print(f"  Avg Reward: {metrics.avg_reward_score:.2f}")

    else:
        # Single model evaluation
        evaluator = ModelEvaluator(model_checkpoint=args.checkpoint)
        await evaluator.initialize()

        metrics = await evaluator.evaluate_test_set(test_queries)

        evaluator.generate_report(metrics, args.output)

        # Cleanup
        await evaluator.reward_evaluator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())