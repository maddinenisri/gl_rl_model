"""
Unit tests for OrchestratorAgent.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gl_rl_model.agents.orchestrator import OrchestratorAgent, WorkflowStage
from gl_rl_model.core.base_agent import AgentStatus


class TestOrchestratorAgent:
    """Test suite for the Orchestrator Agent."""

    @pytest.fixture
    async def orchestrator(self):
        """Create an orchestrator instance for testing."""
        orch = OrchestratorAgent()
        yield orch
        # Cleanup
        if orch.status != AgentStatus.IDLE:
            await orch.shutdown()

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly with all sub-agents."""
        orchestrator = OrchestratorAgent()

        # Initialize orchestrator
        success = await orchestrator.initialize()
        assert success, "Orchestrator should initialize successfully"

        # Check that all agents are registered
        assert "schema_analyzer" in orchestrator.agent_registry
        assert "query_generator" in orchestrator.agent_registry
        assert "validator" in orchestrator.agent_registry
        assert "reward_evaluator" in orchestrator.agent_registry

        # Check that agents are not None
        assert orchestrator.agent_registry["schema_analyzer"] is not None
        assert orchestrator.agent_registry["validator"] is not None
        assert orchestrator.agent_registry["reward_evaluator"] is not None

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self):
        """Test that orchestrator shuts down all agents properly."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Shutdown
        success = await orchestrator.shutdown()
        assert success, "Orchestrator should shutdown successfully"

        # Check that agent registry is cleared
        assert len(orchestrator.agent_registry) == 0

    @pytest.mark.asyncio
    async def test_process_generation_mode(self):
        """Test the complete workflow in generation mode."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Test input
        input_data = {
            "query": "Show all active projects",
            "mode": "generate"
        }

        # Process request
        result = await orchestrator.process(input_data)

        # Verify result structure
        assert "sql" in result
        assert "reasoning" in result
        assert "validation" in result
        assert "session_id" in result

        # The SQL might be a fallback if model is not loaded
        assert result["sql"] is not None

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_process_training_mode(self):
        """Test the complete workflow in training mode."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Test input with expected SQL for training
        input_data = {
            "query": "Show all active projects",
            "mode": "train",
            "expected_sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'"
        }

        # Process request
        result = await orchestrator.process(input_data)

        # Verify result structure (training mode includes rewards)
        assert "sql" in result
        assert "reasoning" in result
        assert "validation" in result
        assert "rewards" in result
        assert "session_id" in result

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling when no query is provided."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Test with missing query
        input_data = {
            "mode": "generate"
        }

        # Process request
        result = await orchestrator.process(input_data)

        # Should return error
        assert "error" in result
        assert result["error"] == "No query provided"

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_workflow_stages(self):
        """Test that workflow progresses through correct stages."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Track stages during execution
        stages_executed = []

        original_execute = orchestrator._execute_workflow
        async def track_stages(session):
            # Record initial stage
            stages_executed.append(orchestrator.workflow_stage)
            result = await original_execute(session)
            # Record final stage
            stages_executed.append(orchestrator.workflow_stage)
            return result

        orchestrator._execute_workflow = track_stages

        # Process a request
        input_data = {
            "query": "Show all projects",
            "mode": "generate"
        }

        result = await orchestrator.process(input_data)

        # Verify stages were executed
        assert WorkflowStage.INITIALIZATION in stages_executed
        assert WorkflowStage.FINALIZATION in stages_executed

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_schema_analysis_integration(self):
        """Test schema analysis stage with real agent."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Create a test session
        session = {
            "session_id": "test-123",
            "query": "Show projects with budget over 100000",
            "context": {}
        }

        # Execute schema analysis
        schema_result = await orchestrator._execute_schema_analysis(session)

        # Verify schema result structure
        assert isinstance(schema_result, dict)
        assert "relevant_tables" in schema_result
        assert "relationships" in schema_result
        assert "business_context" in schema_result

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_validation_integration(self):
        """Test validation stage with real agent."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Create a test session
        session = {
            "session_id": "test-456",
            "query": "Show all projects",
            "schema_context": {}
        }

        # Test SQL
        test_sql = "SELECT * FROM PAC_MNT_PROJECTS;"

        # Execute validation
        validation_result = await orchestrator._execute_validation(test_sql, session)

        # Verify validation result structure
        assert isinstance(validation_result, dict)
        assert "is_valid" in validation_result

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session creation and management."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Process multiple requests
        queries = [
            "Show all active projects",
            "List companies",
            "Find resources"
        ]

        session_ids = []
        for query in queries:
            result = await orchestrator.process({
                "query": query,
                "mode": "generate"
            })
            session_ids.append(result.get("session_id"))

        # All session IDs should be unique
        assert len(session_ids) == len(set(session_ids))

        # Check workflow history
        assert len(orchestrator.workflow_history) <= orchestrator.max_history_size

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        orchestrator = OrchestratorAgent()
        await orchestrator.initialize()

        # Create multiple concurrent requests
        requests = [
            {"query": f"Query {i}", "mode": "generate"}
            for i in range(5)
        ]

        # Process concurrently
        tasks = [orchestrator.process(req) for req in requests]
        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 5
        for result in results:
            assert "session_id" in result

        # Cleanup
        await orchestrator.shutdown()


# Fixtures for mocking when model is not available
@pytest.fixture
def mock_model_loading():
    """Mock model loading for tests without GPU."""
    with patch('gl_rl_model.agents.query_generator.QueryGeneratorAgent.initialize') as mock_init:
        mock_init.return_value = True
        yield mock_init


# Test runner
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])