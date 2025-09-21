#!/usr/bin/env python3
"""
Production API Server for GL RL Model
FastAPI-based REST API for SQL generation service
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import torch
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from gl_rl_model.models.qwen_wrapper import QwenModelWrapper, GenerationParams
from gl_rl_model.utils.prompt_templates import SQLPromptTemplates
from gl_rl_model.training.schema_loader import SchemaLoader
from gl_rl_model.utils.sql_validator import SQLValidator
from gl_rl_model.agents.reward_evaluator import RewardEvaluatorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model_manager = None

# ============================================================================
# Request/Response Models
# ============================================================================

class SQLGenerationRequest(BaseModel):
    """Request model for SQL generation."""
    query: str = Field(..., description="Natural language query")
    include_reasoning: bool = Field(default=True, description="Include reasoning in response")
    temperature: float = Field(default=0.1, min=0.0, max=1.0, description="Generation temperature")
    max_tokens: int = Field(default=256, min=50, max=1024, description="Maximum tokens to generate")
    use_schema_context: bool = Field(default=True, description="Include database schema context")

class SQLGenerationResponse(BaseModel):
    """Response model for SQL generation."""
    query: str
    sql: str
    reasoning: Optional[str] = None
    confidence_score: Optional[float] = None
    uses_domain_tables: bool
    generation_time: float
    timestamp: str
    model_version: str

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    model_version: str
    checkpoint_loaded: str
    device: str
    trainable_parameters: Optional[int] = None
    total_parameters: Optional[int] = None
    status: str

class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    last_request: Optional[str] = None
    total_requests: int

class BatchSQLRequest(BaseModel):
    """Request model for batch SQL generation."""
    queries: List[str] = Field(..., description="List of natural language queries")
    include_reasoning: bool = Field(default=False)
    temperature: float = Field(default=0.1)

class BatchSQLResponse(BaseModel):
    """Response model for batch SQL generation."""
    results: List[SQLGenerationResponse]
    total_queries: int
    successful: int
    failed: int
    batch_generation_time: float

# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages the ML model lifecycle and inference."""

    def __init__(self, checkpoint_path: str = None):
        """Initialize the model manager."""
        self.model_wrapper = None
        self.schema_loader = SchemaLoader()
        self.prompt_templates = SQLPromptTemplates()
        self.sql_validator = SQLValidator()
        self.reward_evaluator = None
        self.checkpoint_path = checkpoint_path
        self.device = None
        self.start_time = time.time()
        self.request_count = 0
        self.last_request_time = None
        self.model_version = "1.0.0"

        # Domain tables for validation
        self.domain_tables = [
            "PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJCNTRTS",
            "PROJSTAFF", "PAC_MNT_RESOURCES", "SRM_CONTACTS"
        ]

    async def initialize(self):
        """Initialize the model and related components."""
        logger.info("Initializing Model Manager...")

        # Determine device
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Mac GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("Using CPU")

        # Initialize model
        self.model_wrapper = QwenModelWrapper(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
            use_lora=True,
            load_in_8bit=False,
            device_map=None if self.device == "mps" else self.device
        )

        # Load model
        logger.info("Loading model...")
        self.model_wrapper.load_model()

        # Move to device if needed
        if self.device == "mps":
            self.model_wrapper.model = self.model_wrapper.model.to(self.device)

        # Load checkpoint if specified
        if self.checkpoint_path:
            checkpoint_path = Path(self.checkpoint_path)
            if checkpoint_path.exists():
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if "lora_state_dict" in checkpoint:
                    self.model_wrapper.load_lora_state_dict(checkpoint["lora_state_dict"])
                    logger.info("Checkpoint loaded successfully")
                    self.model_version = f"1.0.0-{checkpoint_path.stem}"
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")

        # Initialize reward evaluator for confidence scoring
        self.reward_evaluator = RewardEvaluatorAgent()
        await self.reward_evaluator.initialize()

        logger.info("Model Manager initialized successfully")

    async def generate_sql(self, request: SQLGenerationRequest) -> SQLGenerationResponse:
        """Generate SQL from natural language query."""
        start_time = time.time()
        self.request_count += 1
        self.last_request_time = datetime.now()

        try:
            # Get schema context if requested
            schema_context = ""
            if request.use_schema_context:
                schema_context = self.schema_loader.get_schema_context(request.query)

            # Create prompt
            prompt = self.prompt_templates.zero_shot_sql_generation(
                query=request.query,
                schema_context=schema_context,
                business_context="Use exact table names from the schema provided."
            )

            # Set generation parameters
            gen_params = GenerationParams(
                temperature=request.temperature,
                max_new_tokens=request.max_tokens,
                do_sample=request.temperature > 0,
                top_p=0.9 if request.temperature > 0 else 1.0
            )

            # Generate SQL
            result = self.model_wrapper.generate(prompt, gen_params)
            sql, reasoning = self.model_wrapper.extract_sql_and_reasoning(result)

            if not sql:
                sql = self.model_wrapper.extract_sql(result)

            # Validate and score
            uses_domain_tables = any(table in (sql or "") for table in self.domain_tables)

            # Calculate confidence score using reward evaluator
            confidence_score = None
            if self.reward_evaluator and sql:
                try:
                    eval_result = await self.reward_evaluator.process({
                        "query": request.query,
                        "sql": sql or "",
                        "reasoning": reasoning or "",
                        "mode": "single"
                    })

                    if "rewards" in eval_result and hasattr(eval_result["rewards"], "total_reward"):
                        # Normalize reward to 0-1 confidence score
                        confidence_score = min(max(eval_result["rewards"].total_reward / 10.0, 0.0), 1.0)
                except Exception as e:
                    logger.warning(f"Could not calculate confidence score: {e}")

            generation_time = time.time() - start_time

            return SQLGenerationResponse(
                query=request.query,
                sql=sql or "-- Unable to generate SQL",
                reasoning=reasoning if request.include_reasoning else None,
                confidence_score=confidence_score,
                uses_domain_tables=uses_domain_tables,
                generation_time=generation_time,
                timestamp=datetime.now().isoformat(),
                model_version=self.model_version
            )

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def generate_batch(self, request: BatchSQLRequest) -> BatchSQLResponse:
        """Generate SQL for multiple queries."""
        start_time = time.time()
        results = []
        successful = 0
        failed = 0

        for query in request.queries:
            try:
                sql_request = SQLGenerationRequest(
                    query=query,
                    include_reasoning=request.include_reasoning,
                    temperature=request.temperature
                )
                result = await self.generate_sql(sql_request)
                results.append(result)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to generate SQL for query '{query}': {e}")
                failed += 1
                # Add error result
                results.append(SQLGenerationResponse(
                    query=query,
                    sql=f"-- Error: {str(e)}",
                    reasoning=None,
                    confidence_score=0.0,
                    uses_domain_tables=False,
                    generation_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    model_version=self.model_version
                ))

        return BatchSQLResponse(
            results=results,
            total_queries=len(request.queries),
            successful=successful,
            failed=failed,
            batch_generation_time=time.time() - start_time
        )

    async def shutdown(self):
        """Cleanup resources."""
        if self.reward_evaluator:
            await self.reward_evaluator.shutdown()

    def get_model_info(self) -> ModelInfoResponse:
        """Get model information."""
        model_info = self.model_wrapper.get_model_info() if self.model_wrapper else {}

        return ModelInfoResponse(
            model_name=model_info.get("model_name", "Unknown"),
            model_version=self.model_version,
            checkpoint_loaded=self.checkpoint_path or "None",
            device=self.device or "Unknown",
            trainable_parameters=model_info.get("trainable_parameters"),
            total_parameters=model_info.get("total_parameters"),
            status=model_info.get("status", "not_loaded")
        )

    def get_health(self) -> HealthCheckResponse:
        """Get health status."""
        return HealthCheckResponse(
            status="healthy" if self.model_wrapper else "unhealthy",
            model_loaded=self.model_wrapper is not None,
            uptime_seconds=time.time() - self.start_time,
            last_request=self.last_request_time.isoformat() if self.last_request_time else None,
            total_requests=self.request_count
        )

# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global model_manager

    # Startup
    logger.info("Starting GL RL Model API Server...")

    # Get checkpoint from environment or use default
    import os
    checkpoint_path = os.getenv("MODEL_CHECKPOINT", "./checkpoints/sft/best.pt")

    model_manager = ModelManager(checkpoint_path=checkpoint_path)
    await model_manager.initialize()

    logger.info("API Server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down API Server...")
    if model_manager:
        await model_manager.shutdown()
    logger.info("API Server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="GL RL Model API",
    description="SQL Generation API for General Ledger domain using Reinforcement Learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "GL RL Model API",
        "version": "1.0.0",
        "description": "SQL Generation for General Ledger domain"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return model_manager.get_health()

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return model_manager.get_model_info()

@app.post("/generate", response_model=SQLGenerationResponse)
async def generate_sql(request: SQLGenerationRequest):
    """Generate SQL from natural language query."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return await model_manager.generate_sql(request)

@app.post("/generate/batch", response_model=BatchSQLResponse)
async def generate_batch(request: BatchSQLRequest):
    """Generate SQL for multiple queries."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return await model_manager.generate_batch(request)

@app.get("/schema/tables", response_model=List[str])
async def get_tables():
    """Get list of available database tables."""
    return [
        "PAC_MNT_PROJECTS",
        "SRM_COMPANIES",
        "PROJCNTRTS",
        "PROJSTAFF",
        "PAC_MNT_RESOURCES",
        "SRM_CONTACTS"
    ]

@app.get("/examples", response_model=List[Dict[str, str]])
async def get_examples():
    """Get example queries."""
    return [
        {"query": "Show all active projects", "expected_sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'"},
        {"query": "Find projects with budget over 100000", "expected_sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > 100000"},
        {"query": "List companies and their contacts", "expected_sql": "SELECT c.*, con.* FROM SRM_COMPANIES c JOIN SRM_CONTACTS con ON c.Company_ID = con.Company_ID"},
        {"query": "Count projects per company", "expected_sql": "SELECT Company_Code, COUNT(*) FROM PROJCNTRTS GROUP BY Company_Code"},
    ]

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GL RL Model API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--checkpoint", help="Model checkpoint path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.checkpoint:
        import os
        os.environ["MODEL_CHECKPOINT"] = args.checkpoint

    # Run server
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )