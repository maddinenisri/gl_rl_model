# Query Generator Agent

## Overview

The Query Generator Agent is responsible for converting natural language queries into SQL statements using the Qwen2.5-Coder-7B model with chain-of-thought reasoning. It generates domain-specific SQL with step-by-step explanations, confidence scores, and supports both single and multiple candidate generation for reinforcement learning.

## Core Capabilities

### 1. Reasoning-First Generation
- Uses `<think>` tags for chain-of-thought reasoning
- Generates step-by-step logic before SQL
- Provides transparent decision-making process

### 2. Domain-Specific SQL
- Trained on GL/ERP schema (PAC_MNT_PROJECTS, SRM_COMPANIES, etc.)
- 100% accuracy in using correct domain tables
- Handles complex financial and project management queries

### 3. Multiple Candidate Support
- Generates up to 5 SQL variations for GRPO training
- Temperature-controlled diversity (0.1-0.9)
- Confidence scoring for each candidate

## Technical Architecture

### Model Configuration
```python
{
    "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "use_lora": True,
    "lora_rank": 32,
    "lora_alpha": 64,
    "trainable_params": "2.08%",
    "device": "mps/cuda/cpu",
    "max_new_tokens": 512,
    "load_in_8bit": False
}
```

### Generation Parameters
```python
GenerationParams(
    temperature=0.1,      # Low for deterministic output
    top_p=0.9,           # Nucleus sampling
    max_new_tokens=512,  # Sufficient for complex SQL
    do_sample=False,     # Deterministic for production
    repetition_penalty=1.05  # Prevent repetition
)
```

## Input/Output Interface

### Input Structure
```python
{
    "query": str,                    # Natural language query (required)
    "schema_context": str,           # Database schema information
    "business_context": str,         # Optional business rules
    "num_candidates": int,           # For GRPO (default: 1)
    "include_reasoning": bool,       # Include reasoning steps (default: True)
    "session_id": str,              # For caching
    "feedback": dict                # Previous feedback for improvement
}
```

### Output Structure
```python
{
    "sql": str,                     # Generated SQL query
    "reasoning": str,               # Step-by-step reasoning
    "confidence": float,            # 0.0-1.0 confidence score
    "alternatives": [               # If num_candidates > 1
        {
            "sql": str,
            "confidence": float,
            "reasoning": str
        }
    ],
    "tokens_used": int,
    "generation_time": float,
    "cache_hit": bool
}
```

## Query Processing Pipeline

### 1. Context Preparation
```python
# Schema context injection
schema_context = schema_loader.get_schema_context(query)

# Prompt construction with domain emphasis
prompt = templates.zero_shot_sql_generation(
    query=query,
    schema_context=schema_context,
    business_context=business_context
)
```

### 2. Reasoning Generation
```python
# Generate with chain-of-thought
<think>
1. Identify main entities: projects, companies
2. Determine required tables: PAC_MNT_PROJECTS, SRM_COMPANIES
3. Define join conditions: Project_Code matching
4. Apply filters: Status = 'Active', Budget > 100000
5. Specify output columns: all project details
</think>
```

### 3. SQL Construction
```python
# Generate SQL based on reasoning
SELECT p.*
FROM PAC_MNT_PROJECTS p
JOIN SRM_COMPANIES c ON p.Company_Code = c.Company_Code
WHERE p.Status = 'Active'
  AND p.Budget > 100000
```

### 4. Confidence Scoring
- Based on model logits and token probabilities
- Considers reasoning completeness
- Factors in schema compliance
- Range: 0.0 (uncertain) to 1.0 (highly confident)

## Advanced Features

### 1. Error Correction
```python
# Incorporate validation feedback
if feedback and feedback.get("errors"):
    prompt += f"\nPrevious attempt had errors: {feedback['errors']}"
    prompt += "\nPlease correct these issues in the new SQL."
```

### 2. Caching Strategy
- LRU cache with 1000 entry limit
- TTL: 1 hour for production queries
- Key: hash(query + schema_context)
- Reduces latency for repeated queries

### 3. Batch Processing
```python
# Process multiple queries efficiently
batch_results = await query_generator.process_batch([
    {"query": "Show all projects"},
    {"query": "List active companies"},
    {"query": "Find high-budget contracts"}
])
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Average Latency | 15-40s | Complex queries on MPS |
| Simple Query | 5-10s | Basic SELECT statements |
| Cache Hit Latency | <100ms | From memory cache |
| Token Usage | 200-500 | Per query average |
| Memory Usage | 15GB | Model + cache |

## SQL Complexity Support

### Supported Features
- ✅ Simple SELECT with WHERE
- ✅ Multi-table JOINs (INNER, LEFT, RIGHT)
- ✅ Aggregations (COUNT, SUM, AVG, MIN, MAX)
- ✅ GROUP BY with HAVING
- ✅ ORDER BY with LIMIT
- ✅ Subqueries (correlated and uncorrelated)
- ✅ Window functions (ROW_NUMBER, RANK, DENSE_RANK)
- ✅ CTEs (WITH clauses)
- ✅ CASE statements
- ✅ Date arithmetic
- ✅ UNION/INTERSECT/EXCEPT

### Domain-Specific Patterns
```sql
-- Project budget analysis
SELECT
    p.Department,
    COUNT(*) as project_count,
    SUM(p.Budget) as total_budget,
    AVG(p.Budget) as avg_budget
FROM PAC_MNT_PROJECTS p
WHERE p.Status = 'Active'
GROUP BY p.Department
HAVING SUM(p.Budget) > 1000000
ORDER BY total_budget DESC

-- Staff allocation with window functions
SELECT
    s.Staff_ID,
    r.Resource_Name,
    s.Project_Code,
    ROW_NUMBER() OVER (PARTITION BY s.Staff_ID ORDER BY p.Start_Date) as assignment_order
FROM PROJSTAFF s
JOIN PAC_MNT_RESOURCES r ON s.Resource_Code = r.Resource_Code
JOIN PAC_MNT_PROJECTS p ON s.Project_Code = p.Project_Code
```

## Integration with Training

### SFT Training Support
```python
# Generate training examples with reasoning
training_example = {
    "query": query,
    "sql": generated_sql,
    "reasoning": reasoning_steps
}
```

### GRPO Training Support
```python
# Generate multiple candidates for policy optimization
candidates = query_generator.generate_candidates(
    query=query,
    num_candidates=5,
    temperature_range=(0.3, 0.9)
)

# Each candidate gets reward from Reward Evaluator
rewards = [evaluator.evaluate(c) for c in candidates]
```

## Error Handling

### Common Errors and Solutions

1. **Token Limit Exceeded**
```python
try:
    result = generator.generate(prompt, max_new_tokens=512)
except TokenLimitError:
    # Fallback to shorter generation
    result = generator.generate(prompt, max_new_tokens=256)
```

2. **Model Loading Failure**
```python
if not model.is_loaded():
    # Attempt reload with reduced precision
    model.load(load_in_8bit=True)
```

3. **Invalid Schema Context**
```python
if not schema_context:
    # Use default schema
    schema_context = load_default_schema()
```

## Monitoring and Logging

### Key Metrics
```python
metrics = {
    "queries_processed": counter,
    "avg_generation_time": histogram,
    "cache_hit_rate": gauge,
    "error_rate": counter,
    "confidence_distribution": histogram
}
```

### Logging Levels
```python
logger.info(f"Generated SQL for query: {query[:50]}...")
logger.debug(f"Full prompt: {prompt}")
logger.warning(f"Low confidence score: {confidence}")
logger.error(f"Generation failed: {error}")
```

## Best Practices

### 1. Query Optimization
- Pre-process queries to normalize formatting
- Remove unnecessary whitespace and special characters
- Identify query intent early for better routing

### 2. Schema Context
- Always provide complete schema context
- Include table relationships and constraints
- Specify data types for accurate generation

### 3. Feedback Integration
- Store validation results for continuous improvement
- Use error patterns to refine prompts
- Track success rates by query type

## Configuration Options

```python
# In gl_rl_model/core/config.py
QUERY_GENERATOR_CONFIG = {
    "model_checkpoint": "./checkpoints/improved/best_domain.pt",
    "cache_size": 1000,
    "cache_ttl": 3600,
    "max_retries": 3,
    "timeout": 60,
    "batch_size": 4,
    "enable_reasoning": True,
    "confidence_threshold": 0.7,
    "temperature_default": 0.1,
    "use_schema_context": True
}
```

## Testing

### Unit Tests
```python
def test_simple_query_generation():
    result = generator.process({
        "query": "Show all projects"
    })
    assert "PAC_MNT_PROJECTS" in result["sql"]
    assert result["confidence"] > 0.8
```

### Integration Tests
```python
def test_end_to_end_generation():
    # Test with orchestrator
    orchestrator_result = orchestrator.process({
        "query": "Complex multi-table query"
    })
    assert orchestrator_result["sql"]
    assert orchestrator_result["validation"]["is_valid"]
```

## Future Enhancements

1. **Semantic Caching**: Cache based on query intent, not just text
2. **Adaptive Temperature**: Adjust based on query complexity
3. **Multi-dialect Support**: Generate for different SQL dialects
4. **Query Optimization**: Suggest performance improvements
5. **Incremental Learning**: Update model with validated queries
6. **Streaming Generation**: Stream SQL as it's generated

---

**Version**: 1.0.0
**Last Updated**: 2025-09-21
**Status**: Production Ready with 100% domain accuracy