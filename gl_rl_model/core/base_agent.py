"""
Base agent class for the GL RL Model multi-agent system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
import asyncio
from enum import Enum

class AgentStatus(Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"

class MessageType(Enum):
    """Message type enumeration for inter-agent communication."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    INFO = "info"
    RESULT = "result"

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the GL RL Model system.

    This class provides common functionality for agent communication,
    lifecycle management, and error handling.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.

        Args:
            name: Unique identifier for the agent
            config: Agent-specific configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.logger = self._setup_logger()
        self.message_queue: List[AgentMessage] = []
        self.response_cache: Dict[str, Any] = {}
        self._running = False

    def _setup_logger(self) -> logging.Logger:
        """Set up agent-specific logger."""
        logger = logging.getLogger(f"agent.{self.name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the agent.

        Args:
            input_data: Input data for processing

        Returns:
            Processing result as a dictionary

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process method")

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize agent resources.

        Returns:
            True if initialization successful, False otherwise

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement initialize method")

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Clean up agent resources.

        Returns:
            True if shutdown successful, False otherwise

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement shutdown method")

    async def send_message(self, receiver: str, message_type: MessageType,
                          content: Dict[str, Any], correlation_id: Optional[str] = None) -> bool:
        """
        Send a message to another agent.

        Args:
            receiver: Name of the receiving agent
            message_type: Type of message
            content: Message content
            correlation_id: Optional correlation ID for request-response tracking

        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            message = AgentMessage(
                sender=self.name,
                receiver=receiver,
                message_type=message_type,
                content=content,
                correlation_id=correlation_id
            )

            self.logger.debug(f"Sending message to {receiver}: {message_type.value}")
            # In a real implementation, this would use a message broker
            # For now, we'll add to a queue that the orchestrator manages
            self.message_queue.append(message)
            return True

        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False

    async def receive_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """
        Process a received message.

        Args:
            message: The received message

        Returns:
            Response data if applicable, None otherwise
        """
        try:
            self.logger.debug(f"Received message from {message.sender}: {message.message_type.value}")

            if message.message_type == MessageType.REQUEST:
                return await self.handle_request(message.content)
            elif message.message_type == MessageType.ERROR:
                return await self.handle_error(message.content)
            else:
                self.logger.info(f"Received {message.message_type.value} message")
                return None

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {"error": str(e)}

    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming request.

        Args:
            request_data: Request data

        Returns:
            Response data
        """
        self.status = AgentStatus.PROCESSING
        try:
            result = await self.process(request_data)
            self.status = AgentStatus.COMPLETED
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Error handling request: {e}")
            return {"error": str(e)}

    async def handle_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle error messages.

        Args:
            error_data: Error information

        Returns:
            Error handling response
        """
        self.logger.error(f"Received error: {error_data}")
        self.status = AgentStatus.ERROR
        return {"acknowledged": True, "error": error_data}

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        return self.status

    def set_status(self, status: AgentStatus):
        """Set agent status."""
        self.status = status
        self.logger.debug(f"Status changed to: {status.value}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the agent.

        Returns:
            Health status information
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "timestamp": datetime.now().isoformat(),
            "message_queue_size": len(self.message_queue),
            "cache_size": len(self.response_cache)
        }

    def clear_cache(self):
        """Clear the response cache."""
        self.response_cache.clear()
        self.logger.info("Response cache cleared")

    async def run(self):
        """Main agent loop for processing messages."""
        self._running = True
        self.logger.info(f"Agent {self.name} started")

        try:
            await self.initialize()

            while self._running:
                # Process messages from queue
                if self.message_queue:
                    message = self.message_queue.pop(0)
                    await self.receive_message(message)

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Agent error: {e}")
            self.status = AgentStatus.ERROR
        finally:
            await self.shutdown()
            self.logger.info(f"Agent {self.name} stopped")

    def stop(self):
        """Stop the agent."""
        self._running = False
        self.logger.info(f"Stopping agent {self.name}")

class ReasoningAgent(BaseAgent):
    """
    Extended base class for agents that perform reasoning tasks.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize reasoning agent with additional reasoning capabilities."""
        super().__init__(name, config)
        self.reasoning_history: List[Tuple[str, str]] = []  # (input, reasoning) pairs

    def add_reasoning_step(self, input_text: str, reasoning: str):
        """
        Add a reasoning step to history.

        Args:
            input_text: The input that triggered reasoning
            reasoning: The reasoning process/output
        """
        self.reasoning_history.append((input_text, reasoning))
        self.logger.debug(f"Added reasoning step: {len(self.reasoning_history)}")

    def get_reasoning_context(self, max_history: int = 5) -> List[Tuple[str, str]]:
        """
        Get recent reasoning context.

        Args:
            max_history: Maximum number of historical items to return

        Returns:
            List of recent reasoning steps
        """
        return self.reasoning_history[-max_history:]

    def format_reasoning(self, steps: List[str]) -> str:
        """
        Format reasoning steps into a structured output.

        Args:
            steps: List of reasoning steps

        Returns:
            Formatted reasoning string
        """
        formatted = "<think>\n"
        for i, step in enumerate(steps, 1):
            formatted += f"Step {i}: {step}\n"
        formatted += "</think>"
        return formatted

    async def validate_reasoning(self, reasoning: str, context: Dict[str, Any]) -> bool:
        """
        Validate that reasoning is logical and complete.

        Args:
            reasoning: The reasoning to validate
            context: Context for validation

        Returns:
            True if reasoning is valid, False otherwise
        """
        # Basic validation - can be extended
        if not reasoning or len(reasoning) < 10:
            return False

        # Check for common reasoning markers
        reasoning_markers = ["because", "therefore", "since", "given", "considering"]
        has_reasoning = any(marker in reasoning.lower() for marker in reasoning_markers)

        return has_reasoning