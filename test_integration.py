#!/usr/bin/env python3
"""
End-to-End Integration Test for GL RL Model
Tests the complete workflow with all agents working together
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import json
import time
from typing import Dict, List, Any
from datetime import datetime

from gl_rl_model.agents.orchestrator import OrchestratorAgent
from gl_rl_model.agents.schema_analyzer import SchemaAnalyzerAgent
from gl_rl_model.agents.query_generator import QueryGeneratorAgent
from gl_rl_model.agents.validator import ValidatorAgent
from gl_rl_model.agents.reward_evaluator import RewardEvaluatorAgent
from gl_rl_model.models.qwen_wrapper import QwenModelWrapper
from gl_rl_model.training.schema_loader import SchemaLoader

class IntegrationTestRunner:
    """Runs end-to-end integration tests for the GL RL Model system."""

    def __init__(self, checkpoint_path: str = None):
        """Initialize the test runner."""
        self.checkpoint_path = checkpoint_path
        self.orchestrator = None
        self.test_results = []
        self.start_time = None

    async def setup(self):
        """Set up all components for testing."""
        print("="*70)
        print("GL RL Model - End-to-End Integration Test")
        print("="*70)
        print("\nInitializing all components...")

        self.start_time = time.time()

        # Initialize orchestrator (which manages all other agents)
        self.orchestrator = OrchestratorAgent()

        # If checkpoint specified, configure the query generator to use it
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            print(f"✅ Using trained model checkpoint: {self.checkpoint_path}")
            # This would be passed to the query generator via config
            self.orchestrator.config["query_generator"] = {
                "checkpoint_path": self.checkpoint_path
            }

        # Initialize orchestrator and all sub-agents
        success = await self.orchestrator.initialize()

        if success:
            print("✅ All agents initialized successfully")
            print("\nAgents ready:")
            print("  • Orchestrator Agent - Coordinates workflow")
            print("  • Schema Analyzer Agent - Analyzes database schema")
            print("  • Query Generator Agent - Generates SQL queries")
            print("  • Validator Agent - Validates SQL syntax and semantics")
            print("  • Reward Evaluator Agent - Evaluates query quality")
        else:
            print("❌ Failed to initialize agents")
            raise RuntimeError("Agent initialization failed")

        return success

    async def test_single_query(self, query: str, expected_sql: str = None) -> Dict[str, Any]:
        """
        Test a single query through the complete pipeline.

        Args:
            query: Natural language query
            expected_sql: Optional expected SQL for validation

        Returns:
            Test result dictionary
        """
        print(f"\nQuery: {query}")
        print("-" * 50)

        start_time = time.time()

        try:
            # Process through orchestrator
            result = await self.orchestrator.process({
                "query": query,
                "expected_sql": expected_sql,
                "mode": "end_to_end"
            })

            processing_time = time.time() - start_time

            # Extract results
            generated_sql = result.get("sql", "")
            reasoning = result.get("reasoning", "")
            validation = result.get("validation", {})
            rewards = result.get("rewards", {})

            # Check for domain tables
            domain_tables = ["PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJCNTRTS", "PROJSTAFF"]
            uses_domain_tables = any(table in generated_sql for table in domain_tables)

            # Prepare test result
            test_result = {
                "query": query,
                "generated_sql": generated_sql,
                "expected_sql": expected_sql,
                "reasoning": reasoning,
                "validation": validation,
                "rewards": rewards,
                "uses_domain_tables": uses_domain_tables,
                "processing_time": processing_time,
                "success": bool(generated_sql),
                "error": None
            }

            # Display results
            if generated_sql:
                print(f"✅ Generated SQL: {generated_sql[:100]}...")
                print(f"   Domain tables: {'✅' if uses_domain_tables else '❌'}")
                print(f"   Valid syntax: {'✅' if validation.get('is_valid', False) else '❌'}")
                if rewards:
                    total_reward = rewards.get("total_reward", 0)
                    print(f"   Reward score: {total_reward:.2f}")
                print(f"   Processing time: {processing_time:.2f}s")
            else:
                print("❌ Failed to generate SQL")

        except Exception as e:
            test_result = {
                "query": query,
                "generated_sql": None,
                "expected_sql": expected_sql,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            print(f"❌ Error: {e}")

        self.test_results.append(test_result)
        return test_result

    async def test_workflow_stages(self):
        """Test individual workflow stages."""
        print("\n" + "="*70)
        print("Testing Individual Workflow Stages")
        print("="*70)

        test_query = "Show all active projects with budget over 100000"

        # Stage 1: Schema Analysis
        print("\n[Stage 1] Schema Analysis")
        print("-" * 40)
        schema_result = await self.orchestrator.agent_registry["schema_analyzer"].process({
            "query": test_query,
            "schema_path": "gl_rl_model/data/schema/ddl_schema.sql"
        })
        if schema_result.get("relevant_tables"):
            print(f"✅ Identified tables: {', '.join(schema_result['relevant_tables'])}")
        else:
            print("❌ Failed to identify tables")

        # Stage 2: Query Generation
        print("\n[Stage 2] Query Generation")
        print("-" * 40)
        generation_result = await self.orchestrator.agent_registry["query_generator"].process({
            "query": test_query,
            "schema_info": schema_result
        })
        if generation_result.get("sql"):
            print(f"✅ Generated: {generation_result['sql'][:100]}...")
        else:
            print("❌ Failed to generate SQL")

        # Stage 3: Validation
        print("\n[Stage 3] Validation")
        print("-" * 40)
        validation_result = await self.orchestrator.agent_registry["validator"].process({
            "sql": generation_result.get("sql", ""),
            "query": test_query
        })
        print(f"   Syntax valid: {'✅' if validation_result.get('is_valid') else '❌'}")
        if validation_result.get("errors"):
            print(f"   Errors: {validation_result['errors']}")

        # Stage 4: Reward Evaluation
        print("\n[Stage 4] Reward Evaluation")
        print("-" * 40)
        reward_result = await self.orchestrator.agent_registry["reward_evaluator"].process({
            "query": test_query,
            "sql": generation_result.get("sql", ""),
            "reasoning": generation_result.get("reasoning", ""),
            "mode": "single"
        })
        if "rewards" in reward_result:
            print(f"✅ Total reward: {reward_result['rewards'].get('total_reward', 0):.2f}")
        else:
            print("❌ Failed to evaluate rewards")

    async def run_comprehensive_test(self):
        """Run comprehensive integration tests."""
        print("\n" + "="*70)
        print("Running Comprehensive Integration Tests")
        print("="*70)

        # Define test cases
        test_cases = [
            {
                "category": "Simple Queries",
                "queries": [
                    ("Show all active projects", "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'"),
                    ("List all companies", "SELECT * FROM SRM_COMPANIES"),
                    ("Show project staff", "SELECT * FROM PROJSTAFF"),
                ]
            },
            {
                "category": "Filtered Queries",
                "queries": [
                    ("Find projects with budget over 100000", "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > 100000"),
                    ("Show completed projects", "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Completed'"),
                    ("List active companies", "SELECT * FROM SRM_COMPANIES WHERE Status = 'Active'"),
                ]
            },
            {
                "category": "Aggregation Queries",
                "queries": [
                    ("Count projects per company", "SELECT Company_Code, COUNT(*) FROM PROJCNTRTS GROUP BY Company_Code"),
                    ("Calculate total project budgets", "SELECT SUM(Budget) FROM PAC_MNT_PROJECTS"),
                    ("Average project budget by department", "SELECT Department, AVG(Budget) FROM PAC_MNT_PROJECTS GROUP BY Department"),
                ]
            },
            {
                "category": "Join Queries",
                "queries": [
                    ("Show projects with their staff", None),
                    ("List companies with their contacts", None),
                    ("Find projects with contracts", None),
                ]
            }
        ]

        total_tests = 0
        successful_tests = 0
        domain_specific_tests = 0

        for test_group in test_cases:
            print(f"\n{'='*50}")
            print(f"Category: {test_group['category']}")
            print('='*50)

            for query, expected_sql in test_group['queries']:
                result = await self.test_single_query(query, expected_sql)
                total_tests += 1

                if result['success']:
                    successful_tests += 1
                if result.get('uses_domain_tables'):
                    domain_specific_tests += 1

                # Small delay between tests
                await asyncio.sleep(0.5)

        # Summary
        print("\n" + "="*70)
        print("Integration Test Summary")
        print("="*70)
        print(f"Total tests run: {total_tests}")
        print(f"Successful: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"Domain-specific: {domain_specific_tests}/{total_tests} ({domain_specific_tests/total_tests*100:.1f}%)")
        print(f"Total time: {time.time() - self.start_time:.2f}s")

        # Save results
        self.save_results()

    async def test_error_handling(self):
        """Test error handling and edge cases."""
        print("\n" + "="*70)
        print("Testing Error Handling")
        print("="*70)

        edge_cases = [
            "Show me everything",  # Ambiguous
            "Delete all data",  # Dangerous operation
            "What is 2+2",  # Not SQL related
            "",  # Empty query
            "SELECT * FROM non_existent_table",  # Invalid table
        ]

        for query in edge_cases:
            print(f"\nEdge case: '{query}'")
            result = await self.test_single_query(query)
            if result.get('error'):
                print(f"   Handled gracefully: ✅")
            elif result.get('generated_sql'):
                print(f"   Generated SQL: {result['generated_sql'][:50]}...")

    async def test_performance(self):
        """Test system performance."""
        print("\n" + "="*70)
        print("Performance Testing")
        print("="*70)

        queries = [
            "Show all active projects",
            "Find high-budget projects",
            "List companies",
        ]

        print("\nMeasuring response times...")
        total_time = 0
        times = []

        for query in queries * 3:  # Test each query 3 times
            start = time.time()
            await self.test_single_query(query)
            elapsed = time.time() - start
            times.append(elapsed)
            total_time += elapsed

        avg_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nPerformance Metrics:")
        print(f"  Average response time: {avg_time:.2f}s")
        print(f"  Min response time: {min_time:.2f}s")
        print(f"  Max response time: {max_time:.2f}s")

    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integration_test_results_{timestamp}.json"

        results = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_used": self.checkpoint_path,
            "test_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "successful": sum(1 for r in self.test_results if r.get('success')),
                "failed": sum(1 for r in self.test_results if not r.get('success')),
                "domain_specific": sum(1 for r in self.test_results if r.get('uses_domain_tables')),
                "avg_processing_time": sum(r.get('processing_time', 0) for r in self.test_results) / len(self.test_results) if self.test_results else 0
            }
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {filename}")

    async def cleanup(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.shutdown()

async def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="GL RL Model Integration Test")
    parser.add_argument("--checkpoint", help="Model checkpoint to use")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")

    args = parser.parse_args()

    # Use checkpoint if provided, otherwise try to use SFT checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        sft_checkpoint = Path("./checkpoints/sft/best.pt")
        if sft_checkpoint.exists():
            checkpoint = str(sft_checkpoint)
            print(f"Using SFT checkpoint: {checkpoint}")

    # Initialize test runner
    runner = IntegrationTestRunner(checkpoint_path=checkpoint)

    try:
        # Setup
        await runner.setup()

        if args.quick:
            # Quick test - just a few queries
            print("\nRunning quick test...")
            await runner.test_single_query("Show all active projects")
            await runner.test_single_query("Find projects with budget over 100000")
            await runner.test_single_query("Count projects per company")
        elif args.performance:
            # Performance testing
            await runner.test_performance()
        else:
            # Full test suite
            await runner.test_workflow_stages()
            await runner.run_comprehensive_test()
            await runner.test_error_handling()

        # Summary
        print("\n" + "="*70)
        print("✅ Integration Testing Complete!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())