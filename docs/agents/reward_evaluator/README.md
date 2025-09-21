# Reward Evaluator Agent

## Overview

The Reward Evaluator Agent calculates multi-dimensional rewards for generated SQL queries, orchestrating validation and scoring for reinforcement learning (GRPO) training. It evaluates SQL quality across syntax, schema compliance, business logic, performance, and reasoning dimensions.

## Core Responsibilities

### 1. Reward Calculation
- Multi-dimensional reward scoring (6 components)
- Weighted aggregation for total reward
- Advantage calculation for GRPO

### 2. Validation Orchestration
- Coordinates with Validator Agent
- Aggregates validation results
- Translates validation to rewards

### 3. Feedback Generation
- Provides actionable improvement suggestions
- Identifies specific issues for correction
- Tracks improvement over iterations

## Reward Components

### 1. Syntax Reward (Weight: 0.2)
```python
{
    "component": "syntax",
    "weight": 0.2,
    "scoring": {
        "valid_syntax": 1.0,
        "minor_issues": 0.7,
        "major_issues": 0.3,
        "invalid": 0.0
    },
    "factors": [
        "SQL parseability",
        "Keyword usage",
        "Clause structure",
        "Syntax completeness"
    ]
}
```

### 2. Schema Reward (Weight: 0.25)
```python
{
    "component": "schema",
    "weight": 0.25,
    "scoring": {
        "perfect_compliance": 1.0,
        "correct_tables": 0.8,
        "valid_columns": 0.6,
        "partial_compliance": 0.4,
        "non_compliant": 0.0
    },
    "factors": [
        "Table existence",
        "Column validity",
        "Join correctness",
        "Domain table usage"
    ]
}
```

### 3. Business Logic Reward (Weight: 0.2)
```python
{
    "component": "business_logic",
    "weight": 0.2,
    "scoring": {
        "all_rules_satisfied": 1.0,
        "minor_violations": 0.7,
        "major_violations": 0.3,
        "critical_violations": 0.0
    },
    "factors": [
        "Status values",
        "Date consistency",
        "Budget constraints",
        "GL/ERP rules"
    ]
}
```

### 4. Performance Reward (Weight: 0.15)
```python
{
    "component": "performance",
    "weight": 0.15,
    "scoring": {
        "optimal": 1.0,
        "acceptable": 0.7,
        "suboptimal": 0.4,
        "poor": 0.1
    },
    "factors": [
        "Query complexity",
        "Index usage",
        "Join efficiency",
        "Result set size"
    ]
}
```

### 5. Reasoning Reward (Weight: 0.1)
```python
{
    "component": "reasoning",
    "weight": 0.1,
    "scoring": {
        "complete_reasoning": 1.0,
        "partial_reasoning": 0.6,
        "minimal_reasoning": 0.3,
        "no_reasoning": 0.0
    },
    "factors": [
        "Step clarity",
        "Logic flow",
        "Completeness",
        "Accuracy"
    ]
}
```

### 6. Accuracy Reward (Weight: 0.1)
```python
{
    "component": "accuracy",
    "weight": 0.1,
    "scoring": {
        "exact_match": 1.0,
        "semantic_match": 0.8,
        "partial_match": 0.5,
        "no_match": 0.0
    },
    "factors": [
        "Result correctness",
        "Expected vs actual",
        "Query intent match"
    ]
}
```

## Input/Output Interface

### Input Structure
```python
{
    "sql": str,                      # Generated SQL query
    "reasoning": str,                # Chain-of-thought reasoning
    "query": str,                    # Original natural language query
    "expected_sql": str,             # Ground truth SQL (optional)
    "validation_result": dict,       # From Validator Agent (optional)
    "candidates": List[dict],        # For batch evaluation (GRPO)
    "config": {
        "weights": dict,             # Custom component weights
        "strict_mode": bool,         # Penalize warnings
        "bonus_factors": dict        # Additional reward factors
    }
}
```

### Output Structure
```python
{
    "total_reward": float,           # 0.0-1.0 aggregate reward
    "rewards": {
        "syntax": float,
        "schema": float,
        "business_logic": float,
        "performance": float,
        "reasoning": float,
        "accuracy": float
    },
    "feedback": [
        {
            "component": str,
            "score": float,
            "issues": List[str],
            "suggestions": List[str]
        }
    ],
    "advantages": List[float],       # For GRPO training
    "metadata": {
        "evaluation_time": float,
        "validator_called": bool,
        "cache_hit": bool
    }
}
```

## Reward Calculation Pipeline

### Step 1: Validation
```python
def get_validation(sql: str, reasoning: str) -> ValidationResult:
    """Get comprehensive validation from Validator Agent."""

    validation = validator.process({
        "sql": sql,
        "schema_context": schema_context,
        "strict_mode": True
    })

    return validation
```

### Step 2: Component Scoring
```python
def calculate_component_rewards(validation: dict, reasoning: str) -> dict:
    """Calculate individual reward components."""

    rewards = {}

    # Syntax reward
    if validation["details"]["syntax_valid"]:
        rewards["syntax"] = 1.0
    elif validation["details"]["syntax_warnings"]:
        rewards["syntax"] = 0.7
    else:
        rewards["syntax"] = 0.0

    # Schema reward
    if validation["details"]["schema_compliant"]:
        # Bonus for using domain tables
        if all(t in DOMAIN_TABLES for t in validation["details"]["tables_used"]):
            rewards["schema"] = 1.0
        else:
            rewards["schema"] = 0.8
    else:
        rewards["schema"] = 0.0

    # Business logic reward
    violations = len(validation["details"]["business_violations"])
    if violations == 0:
        rewards["business_logic"] = 1.0
    elif violations <= 2:
        rewards["business_logic"] = 0.7
    else:
        rewards["business_logic"] = 0.3

    # Performance reward
    complexity = validation["details"]["complexity_score"]
    if complexity < 30:
        rewards["performance"] = 1.0
    elif complexity < 60:
        rewards["performance"] = 0.7
    elif complexity < 90:
        rewards["performance"] = 0.4
    else:
        rewards["performance"] = 0.1

    # Reasoning reward
    if reasoning and len(reasoning) > 100:
        rewards["reasoning"] = 1.0 if "<think>" in reasoning else 0.8
    elif reasoning:
        rewards["reasoning"] = 0.5
    else:
        rewards["reasoning"] = 0.0

    return rewards
```

### Step 3: Weighted Aggregation
```python
def calculate_total_reward(component_rewards: dict, weights: dict) -> float:
    """Calculate weighted total reward."""

    total = 0.0
    for component, reward in component_rewards.items():
        weight = weights.get(component, 0.1)
        total += reward * weight

    # Apply bonuses/penalties
    if all(r >= 0.8 for r in component_rewards.values()):
        total *= 1.1  # Excellence bonus

    if any(r == 0.0 for r in component_rewards.values()):
        total *= 0.9  # Failure penalty

    return min(1.0, total)  # Cap at 1.0
```

### Step 4: Advantage Calculation (GRPO)
```python
def calculate_advantages(candidates: List[dict]) -> List[float]:
    """Calculate advantages for GRPO training."""

    # Get rewards for all candidates
    rewards = [evaluate_single(c)["total_reward"] for c in candidates]

    # Calculate baseline (mean reward)
    baseline = np.mean(rewards)

    # Calculate advantages
    advantages = [r - baseline for r in rewards]

    # Normalize advantages
    std = np.std(advantages) + 1e-8
    advantages = [a / std for a in advantages]

    return advantages
```

## GRPO Integration

### Batch Evaluation for Policy Optimization
```python
def evaluate_for_grpo(query: str, candidates: List[str]) -> GRPOResult:
    """Evaluate multiple SQL candidates for GRPO training."""

    results = []
    for sql in candidates:
        # Validate and score each candidate
        validation = validator.validate(sql)
        rewards = calculate_component_rewards(validation)
        total = calculate_total_reward(rewards)

        results.append({
            "sql": sql,
            "reward": total,
            "validation": validation
        })

    # Sort by reward
    results.sort(key=lambda x: x["reward"], reverse=True)

    # Calculate advantages
    rewards_array = [r["reward"] for r in results]
    advantages = calculate_advantages(rewards_array)

    return GRPOResult(
        best_sql=results[0]["sql"],
        best_reward=results[0]["reward"],
        all_rewards=rewards_array,
        advantages=advantages,
        candidates=results
    )
```

### Training Loop Integration
```python
# In GRPO trainer
for batch in training_data:
    # Generate multiple candidates
    candidates = query_generator.generate_candidates(
        query=batch["query"],
        num_candidates=5
    )

    # Evaluate candidates
    grpo_result = reward_evaluator.evaluate_for_grpo(
        query=batch["query"],
        candidates=candidates
    )

    # Update policy using advantages
    loss = policy_optimizer.update(
        candidates=candidates,
        advantages=grpo_result.advantages,
        old_log_probs=old_log_probs
    )
```

## Feedback Generation

### Structured Feedback
```python
def generate_feedback(rewards: dict, validation: dict) -> List[Feedback]:
    """Generate actionable feedback for improvement."""

    feedback = []

    # Syntax feedback
    if rewards["syntax"] < 1.0:
        feedback.append(Feedback(
            component="syntax",
            level="error" if rewards["syntax"] == 0 else "warning",
            message="SQL syntax issues detected",
            suggestions=validation["details"]["syntax_errors"]
        ))

    # Schema feedback
    if rewards["schema"] < 1.0:
        feedback.append(Feedback(
            component="schema",
            level="warning",
            message="Schema compliance issues",
            suggestions=[
                "Use exact table names from schema",
                "Verify column references",
                f"Tables available: {', '.join(DOMAIN_TABLES)}"
            ]
        ))

    # Performance feedback
    if rewards["performance"] < 0.7:
        feedback.append(Feedback(
            component="performance",
            level="info",
            message="Query performance can be improved",
            suggestions=validation["details"]["optimization_suggestions"]
        ))

    return feedback
```

## Caching Strategy

### Result Caching
```python
CACHE_CONFIG = {
    "evaluation_cache": {
        "type": "lru",
        "max_size": 1000,
        "ttl": 600,  # 10 minutes
        "key": "hash(sql + reasoning)"
    },
    "validation_cache": {
        "type": "shared",  # Share with Validator
        "ttl": 300
    }
}
```

## Performance Metrics

| Operation | Average Time | With Cache |
|-----------|--------------|------------|
| Single Evaluation | 100ms | 5ms |
| Batch (5 candidates) | 450ms | 20ms |
| Validation Call | 75ms | 0ms (cached) |
| Advantage Calculation | 10ms | N/A |
| Total Pipeline | 150ms | 10ms |

## Advanced Features

### 1. Dynamic Weight Adjustment
```python
def adjust_weights_by_query_type(query: str) -> dict:
    """Adjust component weights based on query complexity."""

    if "JOIN" in query.upper() or "complex" in query.lower():
        # Emphasize schema and performance for complex queries
        return {
            "syntax": 0.15,
            "schema": 0.30,
            "business_logic": 0.20,
            "performance": 0.25,
            "reasoning": 0.10
        }
    else:
        # Standard weights for simple queries
        return DEFAULT_WEIGHTS
```

### 2. Learning Curve Tracking
```python
def track_improvement(session_id: str, rewards_history: List[float]):
    """Track reward improvement over training."""

    improvement_rate = calculate_trend(rewards_history)
    convergence = check_convergence(rewards_history)

    return {
        "improvement_rate": improvement_rate,
        "converged": convergence,
        "episodes": len(rewards_history),
        "best_reward": max(rewards_history),
        "average_reward": np.mean(rewards_history[-10:])
    }
```

### 3. Reward Shaping
```python
def shape_reward(base_reward: float, context: dict) -> float:
    """Apply reward shaping for better learning."""

    shaped = base_reward

    # Bonus for using specific patterns
    if context.get("uses_cte"):
        shaped += 0.05
    if context.get("uses_window_function"):
        shaped += 0.05

    # Penalty for anti-patterns
    if context.get("has_cartesian_join"):
        shaped -= 0.1
    if context.get("missing_where_clause"):
        shaped -= 0.05

    return max(0.0, min(1.0, shaped))
```

## Configuration

```python
# In gl_rl_model/core/config.py
REWARD_EVALUATOR_CONFIG = {
    "default_weights": {
        "syntax": 0.20,
        "schema": 0.25,
        "business_logic": 0.20,
        "performance": 0.15,
        "reasoning": 0.10,
        "accuracy": 0.10
    },
    "strict_mode": False,
    "cache_enabled": True,
    "cache_ttl": 600,
    "batch_size": 5,
    "use_reward_shaping": True,
    "track_improvement": True,
    "min_reward_threshold": 0.3
}
```

## Testing

### Unit Tests
```python
def test_single_evaluation():
    result = evaluator.evaluate({
        "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
        "reasoning": "<think>Finding active projects</think>"
    })
    assert result["total_reward"] > 0.8
    assert result["rewards"]["schema"] == 1.0

def test_batch_evaluation():
    candidates = [
        "SELECT * FROM PAC_MNT_PROJECTS",
        "SELECT * FROM projects",  # Invalid table
        "SELECT Project_Name FROM PAC_MNT_PROJECTS WHERE Budget > 100000"
    ]
    result = evaluator.evaluate_batch(candidates)
    assert result["advantages"][2] > result["advantages"][1]  # Best > worst
```

## Integration Example

### Complete Workflow
```python
# 1. Generate SQL
sql, reasoning = query_generator.generate(
    "Find high-budget active projects"
)

# 2. Validate
validation = validator.validate(sql)

# 3. Calculate rewards
reward_result = reward_evaluator.evaluate({
    "sql": sql,
    "reasoning": reasoning,
    "validation_result": validation
})

# 4. Use for training
if reward_result["total_reward"] < 0.7:
    # Need improvement
    feedback = reward_result["feedback"]
    improved_sql = query_generator.regenerate(
        original_query=query,
        feedback=feedback
    )
```

## Future Enhancements

1. **Learned Reward Functions**: ML-based reward prediction
2. **Contextual Rewards**: Adjust based on user preferences
3. **Multi-Objective Optimization**: Pareto-optimal SQL generation
4. **Online Learning**: Update rewards based on user feedback
5. **Explainable Rewards**: Detailed breakdown of scoring decisions
6. **A/B Testing**: Compare different reward configurations

---

**Version**: 1.0.0
**Last Updated**: 2025-09-21
**Status**: Production Ready for GRPO Training