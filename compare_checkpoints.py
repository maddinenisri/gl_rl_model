#!/usr/bin/env python3
"""
Compare performance between SFT and improved domain checkpoints
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
from gl_rl_model.training.schema_loader import SchemaLoader
import torch
import time

def extract_sql_simple(text):
    """Simple SQL extraction function"""
    if not text:
        return None

    # Look for SQL in code blocks
    if "```sql" in text.lower():
        start = text.lower().find("```sql") + 6
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # Look for SELECT statements
    if "SELECT" in text.upper():
        lines = text.split('\n')
        sql_lines = []
        in_sql = False
        for line in lines:
            if "SELECT" in line.upper() or in_sql:
                in_sql = True
                sql_lines.append(line.strip())
                if ";" in line:
                    break
        if sql_lines:
            return '\n'.join(sql_lines).strip()

    return None

def test_checkpoint(checkpoint_path, checkpoint_name):
    """Test a specific checkpoint with domain queries"""
    print(f"\n{'='*50}")
    print(f"Testing: {checkpoint_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print('='*50)

    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Initialize model
    model_wrapper = QwenModelWrapper(
        model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        use_lora=True,
        load_in_8bit=False,
        device_map=None if device == "mps" else device
    )

    try:
        # Load model
        model_wrapper.load_model()
        if device == "mps":
            model_wrapper.model = model_wrapper.model.to(device)

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "lora_state_dict" in checkpoint:
                model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])
                print("✅ Checkpoint loaded")

        # Initialize components
        schema_loader = SchemaLoader()
        prompt_templates = SQLPromptTemplates()

        # Test queries
        test_queries = [
            "Show all active projects",
            "List companies with their status",
            "Count projects per company",
            "Show project staff allocations"
        ]

        # Generation parameters
        gen_params = GenerationParams(
            temperature=0.1,
            max_new_tokens=256,
            do_sample=False
        )

        results = {
            "queries_tested": len(test_queries),
            "sql_generated": 0,
            "domain_tables_used": 0,
            "total_time": 0
        }

        domain_tables = ["PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJCNTRTS", "PROJSTAFF"]

        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}] {query}")

            # Get schema context
            schema_context = schema_loader.get_schema_context(query)

            # Create prompt
            prompt = prompt_templates.zero_shot_sql_generation(
                query=query,
                schema_context=schema_context,
                business_context="Use exact table names: PAC_MNT_PROJECTS, SRM_COMPANIES, PROJCNTRTS, PROJSTAFF."
            )

            start_time = time.time()
            try:
                result = model_wrapper.generate(prompt, gen_params)
                generation_time = time.time() - start_time
                results["total_time"] += generation_time

                # Extract SQL
                sql = extract_sql_simple(result)

                if sql:
                    results["sql_generated"] += 1
                    print(f"  ✅ SQL: {sql[:80]}...")

                    # Check domain table usage
                    if any(table in sql for table in domain_tables):
                        results["domain_tables_used"] += 1
                        print(f"  ✅ Uses domain tables")
                    else:
                        print(f"  ❌ No domain tables")
                else:
                    print(f"  ❌ No SQL extracted")
                    print(f"  Output: {result[:100]}...")

            except Exception as e:
                print(f"  ❌ Error: {e}")

        # Calculate metrics
        sql_rate = results["sql_generated"] / results["queries_tested"]
        domain_rate = results["domain_tables_used"] / results["queries_tested"]
        avg_time = results["total_time"] / results["queries_tested"]

        print(f"\n📊 {checkpoint_name} Results:")
        print(f"  SQL Generation: {results['sql_generated']}/{results['queries_tested']} ({sql_rate:.1%})")
        print(f"  Domain Tables: {results['domain_tables_used']}/{results['queries_tested']} ({domain_rate:.1%})")
        print(f"  Avg Time: {avg_time:.2f}s")

        return {
            "name": checkpoint_name,
            "sql_rate": sql_rate,
            "domain_rate": domain_rate,
            "avg_time": avg_time,
            "results": results
        }

    except Exception as e:
        print(f"❌ Failed to test {checkpoint_name}: {e}")
        return None
    finally:
        # Cleanup
        if hasattr(model_wrapper, 'model'):
            del model_wrapper.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    """Main comparison function"""
    print("="*70)
    print("CHECKPOINT COMPARISON TEST")
    print("="*70)

    # Define checkpoints to test
    checkpoints = [
        ("./checkpoints/sft/best.pt", "SFT Model"),
        ("./checkpoints/improved/best_domain.pt", "Improved Domain Model")
    ]

    comparison_results = []

    for checkpoint_path, name in checkpoints:
        if not Path(checkpoint_path).exists():
            print(f"\n⚠️ Skipping {name} - checkpoint not found at {checkpoint_path}")
            continue

        result = test_checkpoint(checkpoint_path, name)
        if result:
            comparison_results.append(result)

    # Generate comparison report
    if len(comparison_results) >= 2:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        for result in comparison_results:
            print(f"\n{result['name']}:")
            print(f"  SQL Generation: {result['sql_rate']:.1%}")
            print(f"  Domain Usage: {result['domain_rate']:.1%}")
            print(f"  Avg Time: {result['avg_time']:.2f}s")

        # Determine best performing model
        best_domain = max(comparison_results, key=lambda x: x['domain_rate'])
        best_sql = max(comparison_results, key=lambda x: x['sql_rate'])

        print(f"\n🏆 Best Domain Specificity: {best_domain['name']} ({best_domain['domain_rate']:.1%})")
        print(f"🏆 Best SQL Generation: {best_sql['name']} ({best_sql['sql_rate']:.1%})")

        # Overall recommendation
        sft_result = next((r for r in comparison_results if "SFT" in r['name']), None)
        domain_result = next((r for r in comparison_results if "Domain" in r['name']), None)

        if sft_result and domain_result:
            print(f"\n📈 Improvement Analysis:")
            domain_improvement = domain_result['domain_rate'] - sft_result['domain_rate']
            sql_improvement = domain_result['sql_rate'] - sft_result['sql_rate']

            print(f"  Domain Usage: {domain_improvement:+.1%}")
            print(f"  SQL Generation: {sql_improvement:+.1%}")

            if domain_improvement > 0:
                print(f"  ✅ Improved model shows better domain specificity")
            else:
                print(f"  ⚠️ Improved model needs more domain training")

if __name__ == "__main__":
    main()