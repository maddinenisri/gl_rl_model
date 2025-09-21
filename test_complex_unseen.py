#!/usr/bin/env python3
"""
Test the trained model with complex, unseen SQL queries.
These queries test the model's ability to generalize beyond training data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
from gl_rl_model.training.schema_loader import SchemaLoader
import torch
import time

def test_complex_queries():
    """Test with complex SQL queries not in training data."""

    print("="*70)
    print("COMPLEX SQL GENERATION TEST - UNSEEN QUERIES")
    print("="*70)
    print("\nTesting model's ability to handle complex queries it hasn't seen...")
    print("="*70)

    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Mac GPU (MPS)\n")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✅ Using CUDA GPU\n")
    else:
        device = "cpu"
        print("⚠️ Using CPU\n")

    # Initialize model
    print("Loading trained model...")
    model_wrapper = QwenModelWrapper(
        model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        use_lora=True,
        load_in_8bit=False,
        device_map=None if device == "mps" else device
    )

    # Try to load the best checkpoint
    checkpoint_paths = [
        Path("./checkpoints/improved/best_domain.pt"),
        Path("./checkpoints/sft/best.pt"),
        Path("./checkpoints/improved/best.pt")
    ]

    checkpoint_loaded = False
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            model_wrapper.load_model()

            if device == "mps":
                model_wrapper.model = model_wrapper.model.to(device)

            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "lora_state_dict" in checkpoint:
                model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])
                print("✅ Checkpoint loaded successfully\n")
                checkpoint_loaded = True
                break
            else:
                print(f"⚠️ No LoRA weights in {checkpoint_path}")

    if not checkpoint_loaded:
        print("⚠️ No checkpoint found, using base model")
        model_wrapper.load_model()
        if device == "mps":
            model_wrapper.model = model_wrapper.model.to(device)

    # Initialize helpers
    schema_loader = SchemaLoader()
    prompt_templates = SQLPromptTemplates()

    # Complex test queries - NOT in training data
    complex_queries = [
        # 1. Complex JOIN with aggregation and subquery
        {
            "query": "Find the top 5 projects by budget that have more than 3 staff members assigned",
            "expected_complexity": "JOIN + GROUP BY + HAVING + ORDER BY + LIMIT",
            "difficulty": "hard"
        },

        # 2. Window functions
        {
            "query": "Rank projects by budget within each department",
            "expected_complexity": "WINDOW FUNCTION (RANK)",
            "difficulty": "hard"
        },

        # 3. Multiple JOINs with date filtering
        {
            "query": "Show all projects with their company names and contract values for projects starting after 2024",
            "expected_complexity": "MULTIPLE JOINS + DATE FILTER",
            "difficulty": "medium"
        },

        # 4. Subquery in WHERE clause
        {
            "query": "Find projects with budget higher than the average budget of completed projects",
            "expected_complexity": "SUBQUERY in WHERE",
            "difficulty": "hard"
        },

        # 5. CASE statement with aggregation
        {
            "query": "Categorize projects as small, medium, or large based on budget and count each category",
            "expected_complexity": "CASE WHEN + GROUP BY",
            "difficulty": "hard"
        },

        # 6. Self-join scenario
        {
            "query": "Find all pairs of projects that have the same project manager",
            "expected_complexity": "SELF JOIN",
            "difficulty": "hard"
        },

        # 7. Complex aggregation with multiple conditions
        {
            "query": "Calculate total budget, average cost, and project count by company for active projects only",
            "expected_complexity": "MULTIPLE AGGREGATES + JOIN + WHERE",
            "difficulty": "medium"
        },

        # 8. EXISTS clause
        {
            "query": "Find companies that have at least one project with budget over 500000",
            "expected_complexity": "EXISTS SUBQUERY",
            "difficulty": "hard"
        },

        # 9. UNION query
        {
            "query": "List all resources from both staff and contacts tables with their types",
            "expected_complexity": "UNION",
            "difficulty": "medium"
        },

        # 10. Complex date calculations
        {
            "query": "Find projects where actual duration exceeded planned duration by more than 30 days",
            "expected_complexity": "DATE ARITHMETIC",
            "difficulty": "hard"
        },

        # 11. Nested aggregations
        {
            "query": "Find the department with the highest average project budget",
            "expected_complexity": "NESTED AGGREGATION",
            "difficulty": "hard"
        },

        # 12. COALESCE and NULL handling
        {
            "query": "Show all projects with their actual cost, using budget as fallback if actual cost is null",
            "expected_complexity": "COALESCE/NULL HANDLING",
            "difficulty": "medium"
        },

        # 13. Complex filtering with OR conditions
        {
            "query": "Find projects that are either over budget by 20% or delayed by more than 2 months",
            "expected_complexity": "COMPLEX OR CONDITIONS",
            "difficulty": "hard"
        },

        # 14. Correlated subquery
        {
            "query": "For each company, find the project with the highest budget",
            "expected_complexity": "CORRELATED SUBQUERY",
            "difficulty": "very hard"
        },

        # 15. Recursive or hierarchical query
        {
            "query": "Show the reporting hierarchy of all resources under a specific manager",
            "expected_complexity": "HIERARCHICAL/RECURSIVE",
            "difficulty": "very hard"
        }
    ]

    # Generation parameters for complex queries
    gen_params = GenerationParams(
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=512,  # More tokens for complex SQL
        do_sample=False,
        repetition_penalty=1.05
    )

    # Track results
    results = {
        "total": len(complex_queries),
        "sql_generated": 0,
        "domain_tables_used": 0,
        "syntactically_valid": 0,
        "by_difficulty": {
            "medium": {"total": 0, "success": 0},
            "hard": {"total": 0, "success": 0},
            "very hard": {"total": 0, "success": 0}
        }
    }

    print("\n" + "="*70)
    print("Testing Complex Queries")
    print("="*70)

    for i, test_case in enumerate(complex_queries, 1):
        query = test_case["query"]
        complexity = test_case["expected_complexity"]
        difficulty = test_case["difficulty"]

        results["by_difficulty"][difficulty]["total"] += 1

        print(f"\n[{i}/{len(complex_queries)}] Query: {query}")
        print(f"Expected Complexity: {complexity}")
        print(f"Difficulty: {difficulty.upper()}")
        print("-" * 60)

        # Get schema context
        schema_context = schema_loader.get_schema_context(query)

        # Create prompt
        prompt = prompt_templates.zero_shot_sql_generation(
            query=query,
            schema_context=schema_context,
            business_context="Generate complex SQL using the exact table names from the schema. Use advanced SQL features when appropriate."
        )

        # Generate SQL
        start_time = time.time()
        try:
            result = model_wrapper.generate(prompt, gen_params)
            generation_time = time.time() - start_time

            # Extract SQL
            sql, reasoning = model_wrapper.extract_sql_and_reasoning(result)
            if not sql:
                sql = model_wrapper.extract_sql(result)

            if sql:
                results["sql_generated"] += 1
                results["by_difficulty"][difficulty]["success"] += 1

                # Check for domain tables
                domain_tables = [
                    "PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJSTAFF",
                    "PROJCNTRTS", "PAC_MNT_RESOURCES", "SRM_CONTACTS"
                ]
                uses_domain = any(table in sql for table in domain_tables)

                if uses_domain:
                    results["domain_tables_used"] += 1
                    print("✅ Generated SQL (using domain tables):")
                else:
                    print("⚠️ Generated SQL (not using domain tables):")

                # Display SQL - show complete SQL
                print(f"```sql\n{sql}\n```")

                # Check for expected SQL features
                sql_upper = sql.upper()
                features_found = []

                feature_checks = {
                    "JOIN": "JOIN" in sql_upper,
                    "GROUP BY": "GROUP BY" in sql_upper,
                    "ORDER BY": "ORDER BY" in sql_upper,
                    "WHERE": "WHERE" in sql_upper,
                    "HAVING": "HAVING" in sql_upper,
                    "SUBQUERY": "SELECT" in sql_upper[sql_upper.find("FROM"):] if "FROM" in sql_upper else False,
                    "AGGREGATION": any(func in sql_upper for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"]),
                    "CASE": "CASE" in sql_upper,
                    "WINDOW": any(func in sql_upper for func in ["RANK()", "ROW_NUMBER()", "DENSE_RANK()", "OVER"]),
                    "UNION": "UNION" in sql_upper,
                    "EXISTS": "EXISTS" in sql_upper
                }

                for feature, found in feature_checks.items():
                    if found:
                        features_found.append(feature)

                if features_found:
                    print(f"SQL Features Used: {', '.join(features_found)}")
                    results["syntactically_valid"] += 1

                print(f"Generation Time: {generation_time:.2f}s")

                if reasoning:
                    print(f"\nReasoning: {reasoning[:200]}...")

            else:
                print("❌ No SQL generated")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Print summary
    print("\n" + "="*70)
    print("COMPLEX QUERY TEST RESULTS")
    print("="*70)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total Queries: {results['total']}")
    print(f"  SQL Generated: {results['sql_generated']} ({results['sql_generated']/results['total']*100:.1f}%)")
    print(f"  Domain Tables Used: {results['domain_tables_used']} ({results['domain_tables_used']/results['total']*100:.1f}%)")
    print(f"  Valid SQL Features: {results['syntactically_valid']} ({results['syntactically_valid']/results['total']*100:.1f}%)")

    print(f"\n📈 Results by Difficulty:")
    for difficulty, stats in results["by_difficulty"].items():
        if stats["total"] > 0:
            success_rate = stats["success"] / stats["total"] * 100
            print(f"  {difficulty.upper()}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

    # Evaluation
    print(f"\n🎯 Model Evaluation:")
    domain_rate = results['domain_tables_used'] / results['total'] * 100
    generation_rate = results['sql_generated'] / results['total'] * 100

    if domain_rate >= 80:
        print(f"  ✅ Excellent domain understanding ({domain_rate:.1f}%)")
    elif domain_rate >= 60:
        print(f"  ⚠️ Good domain understanding ({domain_rate:.1f}%)")
    else:
        print(f"  ❌ Needs improvement in domain understanding ({domain_rate:.1f}%)")

    if generation_rate >= 80:
        print(f"  ✅ Strong SQL generation capability ({generation_rate:.1f}%)")
    elif generation_rate >= 60:
        print(f"  ⚠️ Good SQL generation capability ({generation_rate:.1f}%)")
    else:
        print(f"  ❌ Needs improvement in SQL generation ({generation_rate:.1f}%)")

    # Recommendations
    print(f"\n💡 Recommendations:")
    if domain_rate < 100:
        print("  • Consider additional training with more complex domain-specific examples")
    if results["by_difficulty"]["very hard"]["success"] == 0:
        print("  • Model struggles with very complex queries - may need specialized training")
    if results["by_difficulty"]["hard"]["total"] > 0:
        hard_rate = results["by_difficulty"]["hard"]["success"] / results["by_difficulty"]["hard"]["total"] * 100
        if hard_rate < 50:
            print("  • Focus training on complex SQL patterns (subqueries, window functions)")

    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)

if __name__ == "__main__":
    test_complex_queries()