# GL RL Model API Documentation

## Base URL
```
Production: https://api.example.com/v1
Development: http://localhost:8000
```

## Authentication
All API endpoints require authentication via JWT tokens or API keys.

### Bearer Token
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### API Key
```http
X-API-Key: your-api-key-here
```

## Response Format
All responses follow a consistent JSON structure:

```json
{
  "success": true,
  "data": {},
  "error": null,
  "metadata": {
    "request_id": "uuid",
    "timestamp": "2025-09-21T10:00:00Z",
    "version": "1.0.0"
  }
}
```

## Endpoints

### 1. Generate SQL

Generate SQL from natural language query.

**Endpoint:** `POST /generate`

**Request Body:**
```json
{
  "query": "Show all active projects with budget over 100000",
  "include_reasoning": true,
  "include_alternatives": false,
  "num_alternatives": 3,
  "confidence_threshold": 0.7,
  "schema_context": "auto",
  "business_context": {}
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | Natural language query |
| include_reasoning | boolean | No | true | Include chain-of-thought reasoning |
| include_alternatives | boolean | No | false | Generate alternative SQL queries |
| num_alternatives | integer | No | 3 | Number of alternatives (1-5) |
| confidence_threshold | float | No | 0.7 | Minimum confidence score (0.0-1.0) |
| schema_context | string | No | "auto" | Schema context ("auto", "minimal", "full") |
| business_context | object | No | {} | Additional business rules |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active' AND Budget > 100000",
    "reasoning": "<think>\n1. Identify main entity: projects\n2. Table: PAC_MNT_PROJECTS\n3. Conditions: Status = 'Active' AND Budget > 100000\n</think>",
    "confidence": 0.92,
    "alternatives": [],
    "execution_time": 1.23,
    "tokens_used": 245,
    "cache_hit": false
  }
}
```

**Error Response (400):**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query is too vague or ambiguous",
    "details": "Please specify what information you want to retrieve"
  }
}
```

**cURL Example:**
```bash
curl -X POST https://api.example.com/v1/generate \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find top 5 projects by budget",
    "include_reasoning": true
  }'
```

---

### 2. Batch Generate

Generate SQL for multiple queries in a single request.

**Endpoint:** `POST /generate/batch`

**Request Body:**
```json
{
  "queries": [
    "Show all active projects",
    "List companies with contracts",
    "Find resources with high utilization"
  ],
  "include_reasoning": false,
  "parallel": true
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| queries | array | Yes | - | List of natural language queries (max 10) |
| include_reasoning | boolean | No | false | Include reasoning for each query |
| parallel | boolean | No | true | Process queries in parallel |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "query": "Show all active projects",
        "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
        "confidence": 0.95,
        "success": true
      },
      {
        "query": "List companies with contracts",
        "sql": "SELECT DISTINCT c.* FROM SRM_COMPANIES c JOIN PROJCNTRTS ct ON c.Company_Code = ct.Company_Code",
        "confidence": 0.88,
        "success": true
      }
    ],
    "total_queries": 3,
    "successful": 3,
    "failed": 0,
    "execution_time": 3.45
  }
}
```

---

### 3. Validate SQL

Validate SQL query for syntax, schema compliance, and business logic.

**Endpoint:** `POST /validate`

**Request Body:**
```json
{
  "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
  "strict_mode": false,
  "validation_layers": ["syntax", "schema", "business", "performance", "security"]
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| sql | string | Yes | - | SQL query to validate |
| strict_mode | boolean | No | false | Fail on warnings |
| validation_layers | array | No | all | Layers to validate |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "is_valid": true,
    "confidence": 0.98,
    "details": {
      "syntax_valid": true,
      "schema_compliant": true,
      "business_logic_valid": true,
      "performance_acceptable": true,
      "security_passed": true
    },
    "issues": [],
    "suggestions": [
      "Consider adding an index on Status column for better performance"
    ],
    "validated_sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'"
  }
}
```

---

### 4. Get Schema Information

Retrieve database schema information.

**Endpoint:** `GET /schema/tables`

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| include_columns | boolean | No | true | Include column details |
| include_relationships | boolean | No | false | Include table relationships |
| table_name | string | No | - | Filter for specific table |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "tables": [
      {
        "table_name": "PAC_MNT_PROJECTS",
        "type": "master",
        "description": "Master table for projects",
        "columns": [
          {
            "name": "Project_ID",
            "type": "INT",
            "nullable": false,
            "primary_key": true
          },
          {
            "name": "Project_Code",
            "type": "VARCHAR(50)",
            "nullable": false,
            "unique": true
          }
        ],
        "relationships": [
          {
            "related_table": "PROJSTAFF",
            "join_column": "Project_Code",
            "relationship_type": "one_to_many"
          }
        ]
      }
    ],
    "total_tables": 6
  }
}
```

---

### 5. Query Analysis

Analyze a natural language query to identify entities and intent.

**Endpoint:** `POST /analyze`

**Request Body:**
```json
{
  "query": "Show projects managed by John Smith with budget over 100k",
  "return_entities": true,
  "return_intent": true,
  "return_suggested_tables": true
}
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "entities": [
      {"type": "project", "value": "projects"},
      {"type": "person", "value": "John Smith"},
      {"type": "numeric", "value": "100000"}
    ],
    "intent": "retrieve",
    "suggested_tables": ["PAC_MNT_PROJECTS", "PROJSTAFF", "PAC_MNT_RESOURCES"],
    "complexity": "medium",
    "estimated_sql_type": "SELECT with JOIN"
  }
}
```

---

### 6. Get Examples

Retrieve example queries for testing and reference.

**Endpoint:** `GET /examples`

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| category | string | No | all | Category filter (simple, complex, joins) |
| limit | integer | No | 10 | Number of examples (max 50) |
| include_sql | boolean | No | true | Include SQL in response |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "examples": [
      {
        "category": "simple",
        "query": "Show all active projects",
        "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
        "description": "Basic SELECT with WHERE clause"
      },
      {
        "category": "complex",
        "query": "Find top 5 projects by budget with staff count",
        "sql": "SELECT p.*, COUNT(s.Staff_ID) as staff_count FROM PAC_MNT_PROJECTS p LEFT JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code GROUP BY p.Project_ID ORDER BY p.Budget DESC LIMIT 5",
        "description": "JOIN with aggregation and sorting"
      }
    ],
    "total": 2
  }
}
```

---

### 7. Health Check

Check API and model health status.

**Endpoint:** `GET /health`

**No Authentication Required**

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "model_loaded": true,
    "cache_connected": true,
    "database_connected": true,
    "uptime_seconds": 3600,
    "version": "1.0.0",
    "last_request": "2025-09-21T10:00:00Z"
  }
}
```

---

### 8. Model Information

Get information about the loaded model.

**Endpoint:** `GET /model/info`

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "checkpoint": "improved/best_domain.pt",
    "parameters": {
      "total": 7777097216,
      "trainable": 161480704,
      "trainable_percentage": 2.08
    },
    "device": "mps",
    "load_time": 28.5,
    "capabilities": [
      "sql_generation",
      "reasoning",
      "multi_candidate",
      "domain_specific"
    ],
    "performance": {
      "avg_generation_time": 1.5,
      "cache_hit_rate": 0.75,
      "success_rate": 0.98
    }
  }
}
```

---

### 9. Feedback Submission

Submit feedback for generated SQL to improve the model.

**Endpoint:** `POST /feedback`

**Request Body:**
```json
{
  "request_id": "uuid-from-original-request",
  "rating": 4,
  "correct": true,
  "feedback_text": "SQL was correct but could be optimized",
  "corrected_sql": "SELECT p.Project_Name FROM PAC_MNT_PROJECTS p WHERE p.Status = 'Active'"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "feedback_id": "feedback-uuid",
    "received": true,
    "message": "Thank you for your feedback"
  }
}
```

---

### 10. Query History

Retrieve query history for the authenticated user.

**Endpoint:** `GET /history`

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| limit | integer | No | 20 | Number of records (max 100) |
| offset | integer | No | 0 | Pagination offset |
| from_date | string | No | - | ISO date string |
| to_date | string | No | - | ISO date string |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "history": [
      {
        "request_id": "uuid",
        "timestamp": "2025-09-21T10:00:00Z",
        "query": "Show all active projects",
        "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
        "confidence": 0.95,
        "execution_time": 1.2
      }
    ],
    "total": 150,
    "limit": 20,
    "offset": 0
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_REQUEST | 400 | Malformed request or missing required fields |
| INVALID_QUERY | 400 | Query is invalid or unsupported |
| AUTHENTICATION_REQUIRED | 401 | Missing or invalid authentication |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| MODEL_ERROR | 500 | Model generation failed |
| INTERNAL_ERROR | 500 | Internal server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

## Rate Limiting

API implements rate limiting based on authentication level:

| Authentication Type | Requests per Minute | Requests per Day |
|--------------------|-------------------|------------------|
| Free Tier | 10 | 100 |
| Basic | 60 | 5,000 |
| Pro | 300 | 50,000 |
| Enterprise | Unlimited | Unlimited |

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1632150000
```

## Webhooks

Configure webhooks to receive notifications for async operations.

### Webhook Configuration
```json
POST /webhooks/configure
{
  "url": "https://your-app.com/webhook",
  "events": ["query.completed", "batch.completed", "error.occurred"],
  "secret": "your-webhook-secret"
}
```

### Webhook Payload
```json
{
  "event": "query.completed",
  "timestamp": "2025-09-21T10:00:00Z",
  "data": {
    "request_id": "uuid",
    "query": "Show all projects",
    "sql": "SELECT * FROM PAC_MNT_PROJECTS",
    "success": true
  },
  "signature": "hmac-sha256-signature"
}
```

## SDKs and Client Libraries

### Python Client
```python
from gl_rl_client import GLRLClient

client = GLRLClient(api_key="your-api-key")

# Generate SQL
result = client.generate("Show all active projects")
print(result.sql)
print(result.confidence)

# Batch generate
results = client.batch_generate([
    "Query 1",
    "Query 2"
])
```

### JavaScript/TypeScript Client
```javascript
import { GLRLClient } from '@gl-rl/client';

const client = new GLRLClient({ apiKey: 'your-api-key' });

// Generate SQL
const result = await client.generate('Show all active projects');
console.log(result.sql);

// With options
const result = await client.generate('Complex query', {
  includeReasoning: true,
  confidenceThreshold: 0.8
});
```

### cURL Examples
```bash
# Generate SQL
curl -X POST https://api.example.com/v1/generate \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show all active projects"}'

# Validate SQL
curl -X POST https://api.example.com/v1/validate \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT * FROM PAC_MNT_PROJECTS"}'

# Get schema
curl -X GET "https://api.example.com/v1/schema/tables?include_columns=true" \
  -H "X-API-Key: your-api-key"
```

## Best Practices

1. **Caching**: Cache frequently used queries to reduce latency
2. **Batch Requests**: Use batch endpoints for multiple queries
3. **Error Handling**: Implement exponential backoff for retries
4. **Query Clarity**: Provide clear, specific queries for best results
5. **Schema Context**: Include business context for domain-specific queries
6. **Rate Limiting**: Monitor rate limit headers to avoid throttling
7. **Webhooks**: Use webhooks for async operations instead of polling

## Changelog

### Version 1.0.0 (2025-09-21)
- Initial API release
- SQL generation with reasoning
- Multi-layer validation
- Batch processing support
- Schema information endpoints

### Version 1.1.0 (Planned)
- Streaming SQL generation
- Query optimization suggestions
- Multi-dialect SQL support
- Real-time collaboration features

---

**API Version**: 1.0.0
**Last Updated**: 2025-09-21
**Status**: Production Ready

For support, contact: api-support@example.com