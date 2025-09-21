# Orchestrator Agent

## Overview

The Orchestrator Agent is the central coordinator of the GL RL Model multi-agent system. It manages the entire SQL generation workflow, routing requests between specialized agents and ensuring proper sequencing of operations.

## Responsibilities

### Primary Functions
1. **Workflow Management**: Orchestrates the complete SQL generation pipeline
2. **Agent Coordination**: Routes messages between specialized agents
3. **Session Management**: Maintains state across multi-step interactions
4. **Error Recovery**: Handles failures and retries with fallback strategies
5. **Result Aggregation**: Combines outputs from multiple agents into final response

## Workflow Stages

The Orchestrator manages six distinct workflow stages:

### 1. Initialization
- Creates new session with unique ID
- Validates input parameters
- Prepares workflow context

### 2. Schema Analysis
- Delegates to Schema Analyzer Agent
- Identifies relevant tables and relationships
- Establishes business context

### 3. Query Generation
- Delegates to Query Generator Agent
- Produces SQL with step-by-step reasoning
- Generates confidence scores

### 4. Validation
- Delegates to Validator Agent
- Checks syntax, schema compliance, and business logic
- Identifies potential issues

### 5. Reward Evaluation (Training Mode)
- Delegates to Reward Evaluator Agent
- Calculates multi-dimensional rewards
- Provides feedback for model improvement

### 6. Finalization
- Aggregates results from all stages
- Formats final response
- Records workflow history

## API Interface

### Input Structure
```python
{
    "query": "string",           # Natural language query (required)
    "context": {                 # Optional context
        "user_preferences": {},
        "historical_queries": [],
        "business_rules": {}
    },
    "session_id": "string",      # Optional session identifier
    "mode": "generate|train"     # Operation mode (default: generate)
}
```

### Output Structure
```python
{
    "session_id": "string",
    "query": "string",
    "sql": "string",
    "reasoning": "string",
    "confidence": float,
    "validation": {
        "is_valid": bool,
        "details": {
            "syntax_valid": bool,
            "schema_compliant": bool,
            "business_logic_valid": bool,
            "performance_acceptable": bool,
            "issues": [],
            "suggestions": []
        }
    },
    "timestamp": "ISO8601",
    "workflow_stage": "string",
    "training": {                # Only in training mode
        "rewards": {},
        "total_reward": float,
        "feedback": []
    }
}
```

## Configuration

The Orchestrator uses configuration from `gl_rl_model.core.config`:

```python
orchestrator_config = {
    "timeout": 30.0,              # Maximum workflow execution time
    "max_retries": 3,             # Retry attempts for failed operations
    "cache_ttl": 3600,           # Cache time-to-live in seconds
    "max_concurrent_agents": 4,   # Maximum parallel agent operations
    "inter_agent_timeout": 10.0   # Timeout for agent communication
}
```

## Session Management

### Session Lifecycle
1. **Creation**: Unique session ID generated or provided
2. **Active**: Workflow in progress
3. **Completed**: All stages successfully executed
4. **Failed**: Error occurred, partial results available
5. **Expired**: Session timed out or manually terminated

### Session Data
- Query and context information
- Intermediate results from each stage
- Timing and performance metrics
- Error logs and recovery attempts

## Error Handling

### Recovery Strategies
1. **Retry with Backoff**: Automatic retry for transient failures
2. **Fallback Agents**: Use alternative agents if primary fails
3. **Partial Results**: Return available results on partial failure
4. **Graceful Degradation**: Provide best-effort results

### Error Response
```python
{
    "error": "string",
    "error_code": "string",
    "partial_results": {},
    "session_id": "string",
    "failed_stage": "string",
    "suggestions": []
}
```

## Communication Patterns

### Agent Messaging
- **Request-Response**: Synchronous communication with agents
- **Fire-and-Forget**: Asynchronous notifications
- **Broadcast**: Multi-agent announcements
- **Pipeline**: Sequential processing chain

### Message Format
```python
AgentMessage(
    sender="orchestrator",
    receiver="target_agent",
    message_type=MessageType.REQUEST,
    content={...},
    correlation_id="unique_id",
    timestamp=datetime.now()
)
```

## Performance Optimization

### Caching Strategy
- Cache schema analysis results (1 hour TTL)
- Store frequently used SQL patterns
- Maintain session state in memory

### Parallel Processing
- Schema analysis and entity extraction in parallel
- Concurrent validation checks
- Asynchronous reward calculation

## Monitoring and Metrics

### Key Metrics
- Workflow completion rate
- Average response time per stage
- Agent failure rates
- Cache hit ratios
- Session duration distribution

### Logging
```python
# Log levels
INFO: Workflow stage transitions
DEBUG: Agent communication details
WARNING: Retry attempts and degraded performance
ERROR: Failed operations and exceptions
```

## Integration Points

### Upstream
- API Gateway or CLI interface
- User authentication service
- Request validation layer

### Downstream
- Schema Analyzer Agent
- Query Generator Agent
- Validator Agent
- Reward Evaluator Agent

## Usage Examples

### Basic Query Generation
```python
orchestrator = OrchestratorAgent()
await orchestrator.initialize()

result = await orchestrator.process({
    "query": "Show all active projects with budget over 100000",
    "mode": "generate"
})

print(result["sql"])
print(result["reasoning"])
```

### Training Mode
```python
result = await orchestrator.process({
    "query": "List top 5 companies by project count",
    "mode": "train",
    "context": {
        "expected_sql": "SELECT c.Company_Name, COUNT(p.Project_ID) as project_count ..."
    }
})

print(result["training"]["total_reward"])
print(result["training"]["feedback"])
```

## Best Practices

1. **Session Management**
   - Reuse session IDs for related queries
   - Clean up expired sessions regularly
   - Monitor session memory usage

2. **Error Handling**
   - Log all errors with context
   - Provide meaningful error messages
   - Implement circuit breakers for agent failures

3. **Performance**
   - Use caching for repeated queries
   - Batch similar requests when possible
   - Monitor and optimize slow stages

## Future Enhancements

- **Dynamic Agent Selection**: Choose agents based on query complexity
- **Learning from Feedback**: Adapt workflow based on success patterns
- **Multi-Query Optimization**: Handle batch queries efficiently
- **Real-time Monitoring Dashboard**: Visual workflow tracking
- **Advanced Routing Logic**: ML-based agent selection