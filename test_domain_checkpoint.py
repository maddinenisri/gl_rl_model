#!/usr/bin/env python3
"""
Test the improved domain-specific checkpoint to verify SQL generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
from gl_rl_model.training.schema_loader import SchemaLoader
import torch
import time

print("="*70)
print("Testing Improved Domain-Specific Model")
print("="*70)

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Mac GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✅ Using CUDA GPU")
else:
    device = "cpu"
    print("⚠️ Using CPU")

# Initialize model wrapper
print("\nLoading improved domain model...")
model_wrapper = QwenModelWrapper(
    model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
    use_lora=True,
    load_in_8bit=False,
    device_map=None if device == "mps" else device
)

# Check for improved domain checkpoint
checkpoint_path = Path("./checkpoints/improved/best_domain.pt")
if checkpoint_path.exists():
    print(f"✅ Found improved domain checkpoint: {checkpoint_path}")
    try:
        # Load base model
        model_wrapper.load_model()

        # Move to device if MPS
        if device == "mps":
            model_wrapper.model = model_wrapper.model.to(device)

        # Load checkpoint weights
        print("Loading improved domain checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load LoRA weights if present
        if "lora_state_dict" in checkpoint:
            print("Loading LoRA weights from improved checkpoint...")
            model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])

        print("✅ Improved domain checkpoint loaded successfully")

        # Show training info if available
        if "training_history" in checkpoint:
            history = checkpoint["training_history"]
            print(f"📊 Training History:")
            if "domain_accuracy" in history:
                print(f"   Domain Accuracy: {history['domain_accuracy']:.2%}")
            if "final_loss" in history:
                print(f"   Final Loss: {history['final_loss']:.4f}")

    except Exception as e:
        print(f"⚠️ Could not load improved checkpoint: {e}")
        print("Using base model instead")
        model_wrapper.load_model()
        if device == "mps":
            model_wrapper.model = model_wrapper.model.to(device)
else:
    print("No improved checkpoint found, using base model...")
    model_wrapper.load_model()
    if device == "mps":
        model_wrapper.model = model_wrapper.model.to(device)

# Initialize schema loader
schema_loader = SchemaLoader()
prompt_templates = SQLPromptTemplates()

# Extended test queries for comprehensive evaluation
test_queries = [
    # Basic queries
    "Show all active projects",
    "List all companies with their status",
    "Find projects with budget over 100000",

    # Complex joins
    "Count projects per company",
    "Show project staff allocations with resource names",
    "Find companies with active projects",

    # Advanced analytics
    "Calculate total revenue by department",
    "Show average project budget by company",
    "Find overbudget projects with staff assignments",
    "List top 5 companies by project count",

    # Resource management
    "Show resource utilization across all projects",
    "Find underutilized resources (less than 50% allocated)",
    "List all contacts for companies with active projects"
]

print("\n" + "="*70)
print("Generating SQL with Improved Domain Model")
print("="*70)

# Generation parameters optimized for domain-specific output
gen_params = GenerationParams(
    temperature=0.05,  # Very low for consistent domain table usage
    top_p=0.8,
    max_new_tokens=512,  # Increased for complex queries
    do_sample=False,  # Greedy for deterministic results
    repetition_penalty=1.02
)

results = []
domain_table_usage = 0
sql_generation_success = 0

for i, query in enumerate(test_queries, 1):
    print(f"\n[{i}] Query: {query}")
    print("-" * 50)

    # Get schema context for this query
    schema_context = schema_loader.get_schema_context(query)

    # Create enhanced prompt with stronger domain guidance
    prompt = prompt_templates.zero_shot_sql_generation(
        query=query,
        schema_context=schema_context,
        business_context="CRITICAL: Use ONLY these exact table names: PAC_MNT_PROJECTS, SRM_COMPANIES, PROJCNTRTS, PROJSTAFF, PAC_MNT_RESOURCES, SRM_CONTACTS. These are the ONLY valid tables in this GL/ERP system."
    )

    # Track generation time
    start_time = time.time()

    try:
        result = model_wrapper.generate(prompt, gen_params)
        generation_time = time.time() - start_time

        # Extract SQL and reasoning
        sql, reasoning = model_wrapper.extract_sql_and_reasoning(result)

        if not sql:
            sql = model_wrapper.extract_sql(result)

        # Enhanced SQL extraction for complex cases
        if not sql and result:
            if "SELECT" in result.upper():
                lines = result.split('\n')
                sql_lines = []
                in_sql = False
                for line in lines:
                    if "```sql" in line.lower():
                        in_sql = True
                        continue
                    elif "```" in line and in_sql:
                        break
                    elif in_sql or "SELECT" in line.upper():
                        in_sql = True
                        sql_lines.append(line.strip())
                        if ";" in line:
                            break
                if sql_lines:
                    sql = '\n'.join(sql_lines).strip()

        # Check domain table usage
        domain_tables = ["PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJCNTRTS", "PROJSTAFF", "PAC_MNT_RESOURCES", "SRM_CONTACTS"]
        uses_domain = any(table in (sql if sql else result) for table in domain_tables)

        # Check for generic table usage (indicates model isn't using domain tables)
        text_to_check = (sql if sql else result).upper()
        generic_tables = ["PROJECTS", "COMPANIES", "USERS", "EMPLOYEES"]
        uses_generic = any(generic in text_to_check for generic in generic_tables if not any(domain in text_to_check for domain in domain_tables))

        if sql:
            sql_generation_success += 1

        if uses_domain:
            domain_table_usage += 1

        # Store results
        result_data = {
            "query": query,
            "sql": sql,
            "reasoning": reasoning,
            "uses_domain_tables": uses_domain,
            "uses_generic_tables": uses_generic,
            "generation_time": generation_time,
            "raw_output": result
        }
        results.append(result_data)

        print(f"Generated SQL:")
        if sql:
            print(f"  {sql}")
            print(f"  ✅ SQL Generated Successfully")
        else:
            print(f"  ❌ No SQL extracted")
            print(f"  Raw output (first 200 chars): {result[:200]}...")

        print(f"Domain Tables: {'✅' if uses_domain else '❌'}")
        print(f"Generation Time: {generation_time:.2f}s")

        if uses_domain:
            tables_found = [table for table in domain_tables if table in (sql if sql else result)]
            print(f"  Tables used: {', '.join(tables_found)}")

    except Exception as e:
        print(f"❌ Error generating SQL: {e}")
        results.append({
            "query": query,
            "sql": None,
            "error": str(e),
            "uses_domain_tables": False,
            "generation_time": time.time() - start_time
        })

print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

total_queries = len(test_queries)
sql_success_rate = sql_generation_success / total_queries
domain_usage_rate = domain_table_usage / total_queries

print(f"Total Queries: {total_queries}")
print(f"SQL Generation Success: {sql_generation_success}/{total_queries} ({sql_success_rate:.1%})")
print(f"Domain Table Usage: {domain_table_usage}/{total_queries} ({domain_usage_rate:.1%})")

# Calculate average generation time
avg_time = sum(r.get("generation_time", 0) for r in results) / len(results)
print(f"Average Generation Time: {avg_time:.2f}s")

# Show examples of good domain-specific SQL
print(f"\n🎯 Examples of Domain-Specific SQL Generated:")
domain_examples = [r for r in results if r.get("uses_domain_tables", False) and r.get("sql")]
for i, example in enumerate(domain_examples[:3], 1):
    print(f"\n{i}. Query: {example['query']}")
    print(f"   SQL: {example['sql'][:100]}...")

# Performance assessment
print(f"\n📊 Performance Assessment:")
if domain_usage_rate >= 0.8:
    print("🌟 EXCELLENT: Model consistently uses domain-specific tables")
elif domain_usage_rate >= 0.6:
    print("✅ GOOD: Model frequently uses domain-specific tables")
elif domain_usage_rate >= 0.4:
    print("⚠️ MODERATE: Model sometimes uses domain-specific tables")
else:
    print("❌ POOR: Model rarely uses domain-specific tables")

if sql_success_rate >= 0.9:
    print("🌟 EXCELLENT: Model reliably generates SQL")
elif sql_success_rate >= 0.7:
    print("✅ GOOD: Model usually generates SQL")
else:
    print("⚠️ NEEDS IMPROVEMENT: SQL generation success rate low")

print("\n" + "="*70)
print("Domain Model Test Complete")
print("="*70)