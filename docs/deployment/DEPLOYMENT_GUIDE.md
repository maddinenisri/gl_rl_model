# GL RL Model Deployment Guide

## Table of Contents
1. [Deployment Options](#deployment-options)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Configuration](#production-configuration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Scaling Strategies](#scaling-strategies)
9. [Security Considerations](#security-considerations)
10. [Maintenance & Updates](#maintenance--updates)

## Deployment Options

### Overview of Deployment Strategies

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| Local Server | Development/Testing | Simple setup, Full control | Limited scalability |
| Docker | Single server production | Portable, Consistent | Single point of failure |
| Kubernetes | High-availability production | Auto-scaling, Resilient | Complex setup |
| Cloud Managed | Enterprise production | Managed infrastructure | Vendor lock-in |

## Local Deployment

### Prerequisites
- Python 3.10+
- 32GB RAM (64GB recommended)
- GPU with 24GB+ VRAM or Apple Silicon with 16GB+ unified memory
- 50GB disk space

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/yourorg/gl_rl_model.git
cd gl_rl_model

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
uv pip install gunicorn uvicorn[standard]
```

### Step 2: Model Preparation
```bash
# Download base model (if not already present)
python download_model.py

# Verify checkpoint exists
ls -la checkpoints/improved/best_domain.pt
# Should show ~650MB file
```

### Step 3: Configuration
```bash
# Create production config
cat > config/production.yaml << EOF
model:
  checkpoint: "./checkpoints/improved/best_domain.pt"
  device: "mps"  # or "cuda" or "cpu"
  load_in_8bit: false
  max_batch_size: 4

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 60
  max_requests: 1000
  max_requests_jitter: 50

cache:
  enabled: true
  type: "redis"
  ttl: 3600
  max_size: 10000

logging:
  level: "INFO"
  file: "/var/log/gl_rl_model/api.log"
  max_size: "100MB"
  backup_count: 10
EOF
```

### Step 4: Start API Server
```bash
# Development mode
python api_server.py --checkpoint ./checkpoints/improved/best_domain.pt

# Production mode with Gunicorn
gunicorn api_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 60 \
  --access-logfile /var/log/gl_rl_model/access.log \
  --error-logfile /var/log/gl_rl_model/error.log
```

### Step 5: Verify Deployment
```bash
# Health check
curl http://localhost:8000/health

# Test SQL generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Show all active projects"}'
```

## Docker Deployment

### Dockerfile
```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY gl_rl_model/ ./gl_rl_model/
COPY api_server.py .
COPY checkpoints/ ./checkpoints/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  gl-rl-model:
    build: .
    container_name: gl-rl-model
    ports:
      - "8000:8000"
    environment:
      - MODEL_CHECKPOINT=/app/checkpoints/improved/best_domain.pt
      - DEVICE=cpu  # Change to cuda if GPU available
      - LOG_LEVEL=INFO
      - CACHE_TYPE=redis
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 16G

  redis:
    image: redis:7-alpine
    container_name: gl-rl-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: gl-rl-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - gl-rl-model
    restart: unless-stopped

volumes:
  redis-data:
```

### Build and Run
```bash
# Build Docker image
docker build -t gl-rl-model:latest .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f gl-rl-model

# Scale API servers
docker-compose up -d --scale gl-rl-model=3
```

## Kubernetes Deployment

### Deployment Manifest
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-rl-model
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gl-rl-model
  template:
    metadata:
      labels:
        app: gl-rl-model
    spec:
      containers:
      - name: api
        image: gl-rl-model:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_CHECKPOINT
          value: "/app/checkpoints/improved/best_domain.pt"
        - name: DEVICE
          value: "cuda"
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"  # If using GPU
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        volumeMounts:
        - name: model-storage
          mountPath: /app/checkpoints
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

### Service Configuration
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gl-rl-service
  namespace: production
spec:
  selector:
    app: gl-rl-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-rl-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-rl-model
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace production

# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n production
kubectl get svc -n production

# View logs
kubectl logs -f deployment/gl-rl-model -n production

# Scale manually if needed
kubectl scale deployment gl-rl-model --replicas=5 -n production
```

## Cloud Deployment

### AWS Deployment

#### Using Amazon SageMaker
```python
# sagemaker_deploy.py
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Configure SageMaker session
role = 'arn:aws:iam::ACCOUNT:role/SageMakerRole'
sess = sagemaker.Session()

# Create model
pytorch_model = PyTorchModel(
    model_data='s3://my-bucket/gl-rl-model/model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='2.0.0',
    py_version='py310',
    instance_type='ml.g4dn.xlarge'  # GPU instance
)

# Deploy endpoint
predictor = pytorch_model.deploy(
    initial_instance_count=2,
    instance_type='ml.g4dn.xlarge',
    endpoint_name='gl-rl-model-endpoint'
)
```

#### Using EC2 with Auto Scaling
```bash
# Install on EC2 instance
sudo yum update -y
sudo yum install python3 git -y

# Clone and setup
git clone https://github.com/yourorg/gl_rl_model.git
cd gl_rl_model
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Setup as systemd service
sudo tee /etc/systemd/system/gl-rl-model.service << EOF
[Unit]
Description=GL RL Model API Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/gl_rl_model
Environment="PATH=/home/ec2-user/gl_rl_model/venv/bin"
ExecStart=/home/ec2-user/gl_rl_model/venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable gl-rl-model
sudo systemctl start gl-rl-model
```

### Google Cloud Deployment

#### Using Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/gl-rl-model

# Deploy to Cloud Run
gcloud run deploy gl-rl-model \
  --image gcr.io/PROJECT_ID/gl-rl-model \
  --platform managed \
  --region us-central1 \
  --memory 32Gi \
  --cpu 8 \
  --timeout 60 \
  --concurrency 10 \
  --min-instances 1 \
  --max-instances 10
```

### Azure Deployment

#### Using Azure Container Instances
```bash
# Create container instance
az container create \
  --resource-group myResourceGroup \
  --name gl-rl-model \
  --image myregistry.azurecr.io/gl-rl-model:latest \
  --cpu 8 \
  --memory 32 \
  --ports 8000 \
  --dns-name-label gl-rl-model \
  --environment-variables \
    MODEL_CHECKPOINT=/app/checkpoints/improved/best_domain.pt \
    DEVICE=cpu
```

## Production Configuration

### Environment Variables
```bash
# .env.production
MODEL_CHECKPOINT=/app/checkpoints/improved/best_domain.pt
DEVICE=cuda
LOAD_IN_8BIT=false
MAX_BATCH_SIZE=8
CACHE_ENABLED=true
CACHE_TYPE=redis
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
API_KEY_REQUIRED=true
RATE_LIMIT=100
RATE_LIMIT_WINDOW=60
CORS_ORIGINS=["https://app.example.com"]
```

### NGINX Configuration
```nginx
# nginx.conf
upstream gl_rl_backend {
    least_conn;
    server gl-rl-model-1:8000 weight=1;
    server gl-rl-model-2:8000 weight=1;
    server gl-rl-model-3:8000 weight=1;
}

server {
    listen 80;
    listen [::]:80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://gl_rl_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    location /health {
        access_log off;
        proxy_pass http://gl_rl_backend/health;
    }
}
```

## Monitoring & Logging

### Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
request_count = Counter('gl_rl_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('gl_rl_request_duration_seconds', 'Request duration')
active_requests = Gauge('gl_rl_active_requests', 'Active requests')
model_load_time = Histogram('gl_rl_model_load_seconds', 'Model load time')
cache_hits = Counter('gl_rl_cache_hits_total', 'Cache hits')
cache_misses = Counter('gl_rl_cache_misses_total', 'Cache misses')

# Usage in API
@app.post("/generate")
async def generate_sql(request: GenerateRequest):
    request_count.labels(method='POST', endpoint='/generate').inc()
    active_requests.inc()

    with request_duration.time():
        try:
            result = await process_request(request)
            return result
        finally:
            active_requests.dec()
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "GL RL Model Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(gl_rl_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, gl_rl_request_duration_seconds)"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "gl_rl_cache_hits_total / (gl_rl_cache_hits_total + gl_rl_cache_misses_total)"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration
```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

# Configure logging
def setup_logging():
    logger = logging.getLogger('gl_rl_model')
    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        '/var/log/gl_rl_model/api.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger
```

## Scaling Strategies

### Horizontal Scaling
```yaml
# Auto-scaling configuration
scaling:
  min_replicas: 2
  max_replicas: 20
  metrics:
    - type: cpu
      target: 70
    - type: memory
      target: 80
    - type: custom
      metric: request_rate
      target: 100  # requests per second per replica
```

### Load Balancing
```python
# Load balancer health check
@app.get("/health")
async def health_check():
    try:
        # Check model is loaded
        if not model_wrapper.is_loaded():
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "Model not loaded"}
            )

        # Check cache connection
        if not await cache.ping():
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "Cache unavailable"}
            )

        return {"status": "healthy", "timestamp": time.time()}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )
```

### Caching Strategy
```python
# Multi-tier caching
class CacheManager:
    def __init__(self):
        self.l1_cache = {}  # In-memory
        self.l2_cache = Redis()  # Redis
        self.l3_cache = S3()  # S3 for large results

    async def get(self, key):
        # Check L1
        if key in self.l1_cache:
            return self.l1_cache[key]

        # Check L2
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value

        # Check L3
        value = await self.l3_cache.get(key)
        if value:
            await self.l2_cache.set(key, value)
            self.l1_cache[key] = value
            return value

        return None
```

## Security Considerations

### API Authentication
```python
# JWT authentication
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

@app.post("/generate", dependencies=[Depends(verify_token)])
async def generate_sql(request: GenerateRequest):
    # Protected endpoint
    pass
```

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/generate")
@limiter.limit("10 per minute")
async def generate_sql(request: Request, data: GenerateRequest):
    pass
```

### Input Validation
```python
from pydantic import BaseModel, validator

class GenerateRequest(BaseModel):
    query: str
    include_reasoning: bool = True

    @validator('query')
    def validate_query(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        if any(forbidden in v.lower() for forbidden in ['drop', 'delete', 'truncate']):
            raise ValueError('Forbidden keywords in query')
        return v
```

## Maintenance & Updates

### Model Updates
```bash
# Blue-green deployment for model updates

# 1. Deploy new version to green environment
kubectl apply -f k8s/deployment-green.yaml

# 2. Test green environment
curl http://green.api.example.com/health

# 3. Switch traffic to green
kubectl patch service gl-rl-service -p '{"spec":{"selector":{"version":"green"}}}'

# 4. Monitor for issues
kubectl logs -f deployment/gl-rl-model-green

# 5. Remove old blue environment
kubectl delete deployment gl-rl-model-blue
```

### Database Migrations
```python
# Alembic for schema migrations
alembic upgrade head

# Backup before migrations
pg_dump -h localhost -U postgres gl_rl_db > backup_$(date +%Y%m%d).sql
```

### Monitoring Checklist
- [ ] API response times < 2 seconds
- [ ] Error rate < 1%
- [ ] Memory usage < 80%
- [ ] Cache hit rate > 70%
- [ ] Model load time < 30 seconds
- [ ] Daily backup successful
- [ ] SSL certificates valid

---

**Version**: 1.0.0
**Last Updated**: 2025-09-21
**Status**: Production Ready