"""
Orchestrator Agent for coordinating multi-agent SQL generation workflow.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from enum import Enum

from ..core.base_agent import BaseAgent, AgentMessage, MessageType, AgentStatus
from ..core.config import get_config
from .schema_analyzer import SchemaAnalyzerAgent
from .query_generator import QueryGeneratorAgent
from .validator import ValidatorAgent
from .reward_evaluator import RewardEvaluatorAgent

class WorkflowStage(Enum):
    """Workflow stages for SQL generation."""
    INITIALIZATION = "initialization"
    SCHEMA_ANALYSIS = "schema_analysis"
    QUERY_GENERATION = "query_generation"
    VALIDATION = "validation"
    REWARD_EVALUATION = "reward_evaluation"
    FINALIZATION = "finalization"

class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent responsible for coordinating the entire SQL generation workflow.

    This agent manages the flow of data between specialized agents and ensures
    proper sequencing of operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator agent."""
        super().__init__("orchestrator", config)
        self.system_config = get_config()
        self.workflow_stage = WorkflowStage.INITIALIZATION
        self.agent_registry: Dict[str, BaseAgent] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.max_history_size: int = 100  # Maximum number of workflows to keep in history

    async def initialize(self) -> bool:
        """
        Initialize the orchestrator and all sub-agents.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Orchestrator Agent")

            # Initialize all sub-agents
            self.logger.info("Initializing sub-agents...")

            # Schema Analyzer
            self.agent_registry["schema_analyzer"] = SchemaAnalyzerAgent()
            if not await self.agent_registry["schema_analyzer"].initialize():
                raise Exception("Failed to initialize Schema Analyzer")
            self.logger.info("Schema Analyzer initialized")

            # Query Generator (skip model loading if it fails)
            self.agent_registry["query_generator"] = QueryGeneratorAgent()
            try:
                if not await self.agent_registry["query_generator"].initialize():
                    self.logger.warning("Query Generator initialization incomplete (model loading may be skipped)")
            except Exception as e:
                self.logger.warning(f"Query Generator initialization warning: {e}")
                # Continue anyway as model loading might not be available

            # Validator
            self.agent_registry["validator"] = ValidatorAgent()
            if not await self.agent_registry["validator"].initialize():
                raise Exception("Failed to initialize Validator")
            self.logger.info("Validator initialized")

            # Reward Evaluator
            self.agent_registry["reward_evaluator"] = RewardEvaluatorAgent()
            if not await self.agent_registry["reward_evaluator"].initialize():
                raise Exception("Failed to initialize Reward Evaluator")
            self.logger.info("Reward Evaluator initialized")

            self.logger.info("All sub-agents initialized successfully")
            self.status = AgentStatus.IDLE
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Orchestrator: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the orchestrator and all registered agents.

        Returns:
            True if shutdown successful, False otherwise
        """
        try:
            self.logger.info("Shutting down Orchestrator Agent")

            # Shutdown all registered agents
            for agent_name, agent in self.agent_registry.items():
                if agent:
                    self.logger.info(f"Shutting down {agent_name}")
                    try:
                        await agent.shutdown()
                    except Exception as e:
                        self.logger.warning(f"Error shutting down {agent_name}: {e}")

            # Clear active sessions
            self.active_sessions.clear()

            # Clear agent registry
            self.agent_registry.clear()

            self.status = AgentStatus.IDLE
            self.logger.info("Orchestrator Agent shut down successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a SQL generation request through the complete workflow.

        Args:
            input_data: Dictionary containing:
                - query: Natural language query
                - context: Optional context information
                - session_id: Optional session identifier
                - mode: 'generate' or 'train'

        Returns:
            Dictionary containing:
                - sql: Generated SQL query
                - reasoning: Step-by-step reasoning
                - validation: Validation results
                - confidence: Confidence score
                - session_id: Session identifier
        """
        try:
            # Extract input parameters
            query = input_data.get("query")
            context = input_data.get("context", {})
            session_id = input_data.get("session_id", self._generate_session_id())
            mode = input_data.get("mode", "generate")

            if not query:
                return {"error": "No query provided"}

            # Create session
            session = self._create_session(session_id, query, context, mode)

            # Execute workflow
            result = await self._execute_workflow(session)

            # Record workflow history
            self._record_workflow(session_id, result)

            return result

        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return {
                "error": str(e),
                "session_id": session_id if 'session_id' in locals() else None
            }

    async def _execute_workflow(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete SQL generation workflow.

        Args:
            session: Session information

        Returns:
            Workflow results
        """
        results = {}
        session_id = session["session_id"]

        try:
            # Stage 1: Schema Analysis
            self.workflow_stage = WorkflowStage.SCHEMA_ANALYSIS
            schema_result = await self._execute_schema_analysis(session)
            results["schema"] = schema_result
            session["schema_context"] = schema_result

            # Stage 2: Query Generation
            self.workflow_stage = WorkflowStage.QUERY_GENERATION
            generation_result = await self._execute_query_generation(session)
            results["generation"] = generation_result

            # Stage 3: Validation
            self.workflow_stage = WorkflowStage.VALIDATION
            validation_result = await self._execute_validation(
                generation_result.get("sql", ""),
                session
            )
            results["validation"] = validation_result

            # Stage 4: Reward Evaluation (for training mode)
            if session["mode"] == "train":
                self.workflow_stage = WorkflowStage.REWARD_EVALUATION
                reward_result = await self._execute_reward_evaluation(
                    session,
                    generation_result,
                    validation_result
                )
                results["reward"] = reward_result

            # Stage 5: Finalization
            self.workflow_stage = WorkflowStage.FINALIZATION
            final_result = self._finalize_results(results, session)

            return final_result

        except Exception as e:
            self.logger.error(f"Workflow execution failed at stage {self.workflow_stage.value}: {e}")
            return {
                "error": f"Workflow failed at {self.workflow_stage.value}: {str(e)}",
                "partial_results": results,
                "session_id": session_id
            }

    async def _execute_schema_analysis(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute schema analysis stage using SchemaAnalyzerAgent.

        Args:
            session: Session information

        Returns:
            Schema analysis results
        """
        self.logger.info(f"Executing schema analysis for session {session['session_id']}")

        # Get the schema analyzer agent
        schema_analyzer = self.agent_registry.get("schema_analyzer")
        if not schema_analyzer:
            raise Exception("Schema Analyzer not available")

        # Call the actual schema analyzer agent
        result = await schema_analyzer.process({
            "query": session["query"],
            "context": session.get("context", {})
        })

        if result.get("error"):
            raise Exception(f"Schema analysis failed: {result['error']}")

        # Extract schema context from result
        schema_context = {
            "relevant_tables": result.get("relevant_tables", []),
            "relationships": result.get("relationships", {}),
            "business_context": result.get("business_context", {}),
            "entity_mappings": result.get("entity_mappings", {}),
            "suggested_columns": result.get("suggested_columns", [])
        }

        return schema_context

    async def _execute_query_generation(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute query generation stage using QueryGeneratorAgent.

        Args:
            session: Session information

        Returns:
            Query generation results
        """
        self.logger.info(f"Executing query generation for session {session['session_id']}")

        # Get the query generator agent
        query_generator = self.agent_registry.get("query_generator")
        if not query_generator:
            raise Exception("Query Generator not available")

        # Call the actual query generator agent
        result = await query_generator.process({
            "query": session["query"],
            "schema_context": session.get("schema_context", {}),
            "generate_alternatives": session.get("generate_alternatives", False),
            "num_alternatives": session.get("num_alternatives", 1)
        })

        if result.get("error"):
            # Query generator might fail if model is not loaded, provide fallback
            self.logger.warning(f"Query generation failed: {result['error']}")
            # Use a simple fallback
            return {
                "sql": f"-- Query generation unavailable: {result.get('error')}\nSELECT * FROM PAC_MNT_PROJECTS WHERE 1=1;",
                "reasoning": "Model-based generation not available, using fallback",
                "confidence": 0.1,
                "alternatives": [],
                "metadata": {"fallback": True}
            }

        # Extract generation results
        generation_result = {
            "sql": result.get("sql", ""),
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", 0.0),
            "alternatives": result.get("alternatives", []),
            "metadata": result.get("metadata", {})
        }

        return generation_result

    async def _execute_validation(self, sql: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute validation stage.

        Args:
            sql: Generated SQL query
            session: Session information

        Returns:
            Validation results
        """
        self.logger.info(f"Executing validation for session {session['session_id']}")

        # Get the validator agent
        validator = self.agent_registry.get("validator")
        if not validator:
            raise Exception("Validator not available")

        # Call the actual validator agent
        validation_result = await validator.process({
            "sql": sql,
            "schema_context": session.get("schema_context", {}),
            "strict_mode": session.get("strict_mode", False),
            "check_performance": True,
            "check_security": True
        })

        if validation_result.get("error"):
            self.logger.warning(f"Validation encountered error: {validation_result['error']}")

        return validation_result

    async def _execute_reward_evaluation(
        self,
        session: Dict[str, Any],
        generation_result: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute reward evaluation for training mode.

        Args:
            session: Session information
            generation_result: Query generation results
            validation_result: Validation results

        Returns:
            Reward evaluation results
        """
        self.logger.info(f"Executing reward evaluation for session {session['session_id']}")

        # Get the reward evaluator agent
        reward_evaluator = self.agent_registry.get("reward_evaluator")
        if not reward_evaluator:
            raise Exception("Reward Evaluator not available")

        # Call the actual reward evaluator agent
        reward_result = await reward_evaluator.process({
            "query": session["query"],
            "sql": generation_result.get("sql", ""),
            "reasoning": generation_result.get("reasoning", ""),
            "expected_sql": session.get("expected_sql"),
            "mode": "single"
        })

        if reward_result.get("error"):
            self.logger.warning(f"Reward evaluation failed: {reward_result['error']}")
            # Provide fallback rewards
            return {
                "rewards": {
                    "syntax_score": 0.0,
                    "schema_compliance_score": 0.0,
                    "business_logic_score": 0.0,
                    "performance_score": 0.0,
                    "reasoning_quality_score": 0.0
                },
                "total_reward": 0.0,
                "feedback": ["Reward evaluation unavailable"]
            }

        return reward_result

    def _finalize_results(self, results: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize and format the workflow results.

        Args:
            results: Raw results from all stages
            session: Session information

        Returns:
            Formatted final results
        """
        generation = results.get("generation", {})
        validation = results.get("validation", {})

        final_result = {
            "session_id": session["session_id"],
            "query": session["query"],
            "sql": generation.get("sql", ""),
            "reasoning": generation.get("reasoning", ""),
            "confidence": generation.get("confidence", 0.0),
            "validation": {
                "is_valid": all([
                    validation.get("syntax_valid", False),
                    validation.get("schema_compliant", False),
                    validation.get("business_logic_valid", False)
                ]),
                "details": validation
            },
            "timestamp": datetime.now().isoformat(),
            "workflow_stage": self.workflow_stage.value
        }

        # Add training-specific results if in training mode
        if session["mode"] == "train" and "reward" in results:
            reward_data = results["reward"]
            # Handle different reward result structures
            if "rewards" in reward_data:
                rewards = reward_data["rewards"]
                # Check if total is nested under rewards or at the top level
                total_reward = rewards.get("total", 0.0) if isinstance(rewards, dict) else 0.0
            else:
                rewards = {}
                total_reward = 0.0

            # Check for total_reward in metadata if not found
            if total_reward == 0.0 and "metadata" in reward_data:
                total_reward = reward_data["metadata"].get("total_reward", 0.0)

            final_result["rewards"] = rewards
            final_result["total_reward"] = total_reward
            final_result["feedback"] = reward_data.get("feedback", [])

        return final_result

    def _create_session(self, session_id: str, query: str, context: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Create a new workflow session."""
        session = {
            "session_id": session_id,
            "query": query,
            "context": context,
            "mode": mode,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        self.active_sessions[session_id] = session
        return session

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return f"session_{uuid.uuid4().hex[:8]}"

    def _identify_relevant_tables(self, query: str) -> List[str]:
        """Identify tables relevant to the query."""
        # Simple keyword-based identification (to be replaced with actual logic)
        tables = []
        keywords = {
            "project": ["SRM_PROJECTS", "PAC_MNT_PROJECTS", "PROJCNTRTS"],
            "company": ["SRM_COMPANIES"],
            "contact": ["SRM_CONTACTS"],
            "resource": ["PAC_MNT_RESOURCES"],
            "staff": ["PROJSTAFF"],
            "client": ["CLNTSUPP", "CLNTRESPONS"]
        }

        query_lower = query.lower()
        for keyword, table_list in keywords.items():
            if keyword in query_lower:
                tables.extend(table_list)

        return list(set(tables)) if tables else ["SRM_PROJECTS"]  # Default

    def _get_table_relationships(self) -> Dict[str, List[str]]:
        """Get relationships between tables."""
        return {
            "SRM_PROJECTS": ["SRM_COMPANIES", "PROJCNTRTS", "PROJSTAFF"],
            "SRM_COMPANIES": ["SRM_PROJECTS", "SRM_CONTACTS", "CLNTSUPP"],
            "PAC_MNT_PROJECTS": ["PAC_MNT_RESOURCES", "PROJCNTRTS"]
        }

    def _get_business_context(self, query: str) -> Dict[str, Any]:
        """Extract business context from the query."""
        return {
            "domain": "project_management",
            "operation_type": self._identify_operation_type(query),
            "time_context": self._identify_time_context(query)
        }

    def _identify_operation_type(self, query: str) -> str:
        """Identify the type of SQL operation needed."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["list", "show", "get", "find"]):
            return "SELECT"
        elif any(word in query_lower for word in ["count", "total", "sum"]):
            return "AGGREGATE"
        elif any(word in query_lower for word in ["create", "add", "insert"]):
            return "INSERT"
        elif any(word in query_lower for word in ["update", "change", "modify"]):
            return "UPDATE"
        else:
            return "SELECT"  # Default

    def _identify_time_context(self, query: str) -> Optional[str]:
        """Identify temporal context in the query."""
        # Simple pattern matching (to be enhanced)
        if "today" in query.lower():
            return "current_day"
        elif "this month" in query.lower():
            return "current_month"
        elif "this year" in query.lower():
            return "current_year"
        return None

    def _generate_sample_sql(self, session: Dict[str, Any]) -> str:
        """Generate a sample SQL query (placeholder)."""
        tables = session.get("schema_context", {}).get("relevant_tables", ["SRM_PROJECTS"])
        return f"SELECT * FROM {tables[0]} WHERE 1=1"

    def _generate_reasoning(self, session: Dict[str, Any]) -> str:
        """Generate reasoning for the SQL query."""
        return """<think>
Step 1: Identified the main entity as projects based on the query.
Step 2: Determined relevant tables: SRM_PROJECTS as the primary table.
Step 3: Applied appropriate filters based on business context.
Step 4: Constructed SELECT statement with necessary joins.
</think>"""

    def _evaluate_reasoning_quality(self, reasoning: str) -> float:
        """Evaluate the quality of reasoning."""
        if not reasoning:
            return -1.0
        if "<think>" in reasoning and "</think>" in reasoning:
            if len(reasoning) > 100:
                return 1.0
            return 0.5
        return 0.0

    def _generate_feedback(self, rewards: Dict[str, float]) -> List[str]:
        """Generate feedback based on rewards."""
        feedback = []
        for component, score in rewards.items():
            if score < 0:
                feedback.append(f"Improvement needed in {component.replace('_', ' ')}")
        return feedback

    def _record_workflow(self, session_id: str, result: Dict[str, Any]):
        """Record workflow history."""
        self.workflow_history.append({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })

        # Keep only last 100 entries
        if len(self.workflow_history) > self.max_history_size:
            self.workflow_history = self.workflow_history[-self.max_history_size:]