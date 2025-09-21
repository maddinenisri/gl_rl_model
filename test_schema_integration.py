#!/usr/bin/env python3
"""
Test script to verify schema context is properly integrated in training prompts
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
from gl_rl_model.training.schema_loader import SchemaLoader

print("="*60)
print("Testing Schema Integration in Training Prompts")
print("="*60)

# Test 1: Verify SchemaLoader loads schema properly
print("\n[1] Testing SchemaLoader...")
schema_loader = SchemaLoader()
schema_context = schema_loader.get_schema_context("Show all active projects")
print("Schema context loaded:")
print(schema_context[:500] + "..." if len(schema_context) > 500 else schema_context)
assert "PAC_MNT_PROJECTS" in schema_context, "PAC_MNT_PROJECTS table not found in schema!"
assert "SRM_COMPANIES" in schema_context, "SRM_COMPANIES table not found in schema!"
print("✅ SchemaLoader works correctly")

# Test 2: Verify prompt template includes schema
print("\n[2] Testing prompt generation with schema...")
test_query = "Show all active projects with budget over 100000"
test_sql = "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active' AND Budget > 100000"
test_reasoning = "Step 1: Identify table - PAC_MNT_PROJECTS contains project data\nStep 2: Apply filters - Status = 'Active' and Budget > 100000"

# Generate training prompt (should automatically include schema)
prompt = SQLPromptTemplates.generate_training_prompt(
    query=test_query,
    sql=test_sql,
    reasoning=test_reasoning
)

print("Generated prompt preview:")
print(prompt[:800] + "..." if len(prompt) > 800 else prompt)

# Verify schema is in the prompt
assert "PAC_MNT_PROJECTS" in prompt, "Schema not included in training prompt!"
assert "Database Schema:" in prompt or "Schema:" in prompt, "Schema section missing from prompt!"
print("✅ Prompt includes schema context")

# Test 3: Verify multiple tables are included
print("\n[3] Testing comprehensive schema context...")
complex_query = "List all companies with their contacts and project contracts"
schema_context = schema_loader.get_schema_context(complex_query)
assert "SRM_COMPANIES" in schema_context, "Companies table missing"
assert "SRM_CONTACTS" in schema_context, "Contacts table missing"
assert "PROJCNTRTS" in schema_context, "Contracts table missing"
print("✅ Multiple relevant tables included based on query")

# Test 4: Generate a few example prompts to inspect
print("\n[4] Generating example training prompts...")
examples = [
    ("Show all active projects", "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'"),
    ("Count projects per company", "SELECT Company_Code, COUNT(*) FROM PROJCNTRTS GROUP BY Company_Code"),
    ("Find high-budget projects", "SELECT Project_Name, Budget FROM PAC_MNT_PROJECTS WHERE Budget > 500000")
]

for i, (query, sql) in enumerate(examples, 1):
    print(f"\nExample {i}: {query}")
    prompt = SQLPromptTemplates.generate_training_prompt(query=query, sql=sql, reasoning="")
    # Check that schema is included
    assert "PAC_MNT_PROJECTS" in prompt or "PROJCNTRTS" in prompt, f"Schema missing from example {i}"
    print(f"  ✅ Schema included (prompt length: {len(prompt)} chars)")

print("\n" + "="*60)
print("✅ All schema integration tests passed!")
print("The model will now see the actual database schema during training.")
print("="*60)

print("\nNext steps:")
print("1. Run SFT training: python train_sft.py")
print("2. Test the trained model: python test_model.py")
print("3. The model should now generate domain-specific SQL with actual table names")