"""
Query Generator Agent for SQL generation with reasoning using Qwen model.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from ..core.base_agent import ReasoningAgent, AgentStatus
from ..core.config import get_config
from ..models.qwen_wrapper import QwenModelWrapper, GenerationParams
from ..utils.prompt_templates import SQLPromptTemplates

@dataclass
class SQLGenerationResult:
    """Result of SQL generation."""
    sql: str
    reasoning: str
    confidence: float
    alternatives: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class QueryGeneratorAgent(ReasoningAgent):
    """
    Agent responsible for generating SQL queries with reasoning using Qwen model.

    This agent uses the Qwen2.5-Coder model to generate SQL queries from natural
    language, providing step-by-step reasoning for transparency.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the query generator agent."""
        super().__init__("query_generator", config)
        self.system_config = get_config()
        self.model_wrapper: Optional[QwenModelWrapper] = None
        self.prompt_templates = SQLPromptTemplates()
        self.generation_cache: Dict[str, SQLGenerationResult] = {}
        self.few_shot_examples: List[Dict[str, str]] = []
        self.model_loaded = False

    async def initialize(self) -> bool:
        """
        Initialize the query generator and load the model.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Query Generator Agent")

            # Initialize model wrapper
            self.model_wrapper = QwenModelWrapper(
                model_name_or_path=self.system_config.model.model_name,
                use_lora=True,
                load_in_8bit=False  # Can be changed based on hardware
            )

            # Load model (this will be async in production)
            await self._load_model_async()

            # Load few-shot examples
            await self._load_few_shot_examples()

            self.status = AgentStatus.IDLE
            self.model_loaded = True
            self.logger.info("Query Generator Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Query Generator: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the query generator agent.

        Returns:
            True if shutdown successful, False otherwise
        """
        try:
            self.logger.info("Shutting down Query Generator Agent")
            self.generation_cache.clear()
            self.few_shot_examples.clear()
            self.model_wrapper = None
            self.model_loaded = False
            self.status = AgentStatus.IDLE
            return True
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query generation request.

        Args:
            input_data: Dictionary containing:
                - query: Natural language query
                - schema_context: Schema information
                - mode: 'single' or 'multiple' (for GRPO)
                - num_candidates: Number of SQL candidates to generate
                - use_cache: Whether to use cached results
                - examples: Optional few-shot examples

        Returns:
            Dictionary containing:
                - sql: Generated SQL query
                - reasoning: Step-by-step reasoning
                - confidence: Confidence score
                - alternatives: Alternative SQL queries (if mode='multiple')
                - metadata: Additional information
        """
        try:
            if not self.model_loaded:
                return {
                    "error": "Model not loaded",
                    "sql": "",
                    "reasoning": "",
                    "confidence": 0.0
                }

            query = input_data.get("query", "")
            schema_context = input_data.get("schema_context", {})
            mode = input_data.get("mode", "single")
            num_candidates = input_data.get("num_candidates", 1)
            use_cache = input_data.get("use_cache", True)
            custom_examples = input_data.get("examples", [])

            # Check cache
            cache_key = self._generate_cache_key(query, schema_context)
            if use_cache and cache_key in self.generation_cache:
                self.logger.info("Using cached result")
                cached_result = self.generation_cache[cache_key]
                return self._format_result(cached_result)

            # Generate SQL based on mode
            if mode == "multiple":
                result = await self._generate_multiple_candidates(
                    query, schema_context, num_candidates, custom_examples
                )
            else:
                result = await self._generate_single_query(
                    query, schema_context, custom_examples
                )

            # Cache result
            if use_cache:
                self.generation_cache[cache_key] = result

            # Add to reasoning history
            self.add_reasoning_step(query, result.reasoning)

            return self._format_result(result)

        except Exception as e:
            self.logger.error(f"Error processing query generation: {e}")
            return {
                "error": str(e),
                "sql": "",
                "reasoning": "",
                "confidence": 0.0,
                "alternatives": []
            }

    async def _load_model_async(self) -> None:
        """Load the model asynchronously."""
        # In production, this would be truly async
        # For now, we'll run it in an executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.model_wrapper.load_model)

    async def _load_few_shot_examples(self) -> None:
        """Load few-shot examples from training data."""
        try:
            training_file = Path(self.system_config.data_dir) / "training" / "query_pairs.jsonl"
            if training_file.exists():
                with open(training_file, 'r') as f:
                    # Load first 5 examples as few-shot
                    for i, line in enumerate(f):
                        if i >= 5:
                            break
                        example = json.loads(line)
                        self.few_shot_examples.append(example)
                self.logger.info(f"Loaded {len(self.few_shot_examples)} few-shot examples")
        except Exception as e:
            self.logger.warning(f"Could not load few-shot examples: {e}")

    async def _generate_single_query(
        self,
        query: str,
        schema_context: Dict[str, Any],
        custom_examples: List[Dict[str, str]]
    ) -> SQLGenerationResult:
        """
        Generate a single SQL query with reasoning.

        Args:
            query: Natural language query
            schema_context: Schema information
            custom_examples: Custom few-shot examples

        Returns:
            SQLGenerationResult object
        """
        # Format schema context
        schema_str = self._format_schema_for_prompt(schema_context)

        # Determine which examples to use
        examples = custom_examples if custom_examples else self.few_shot_examples

        # Create prompt based on availability of examples
        if examples:
            prompt = self.prompt_templates.few_shot_sql_generation(
                query=query,
                schema_context=schema_str,
                examples=examples,
                business_context=schema_context.get("business_entities")
            )
        else:
            prompt = self.prompt_templates.zero_shot_sql_generation(
                query=query,
                schema_context=schema_str,
                business_context=schema_context.get("business_entities")
            )

        # Generate SQL
        generation_params = GenerationParams(
            temperature=self.system_config.model.temperature,
            top_p=self.system_config.model.top_p,
            max_new_tokens=self.system_config.model.max_new_tokens
        )

        generated_text = self.model_wrapper.generate(prompt, generation_params)

        # Extract SQL and reasoning
        sql, reasoning = self.model_wrapper.extract_sql_and_reasoning(generated_text)

        # Calculate confidence
        confidence = self._calculate_confidence(sql, reasoning, schema_context)

        # Build reasoning steps
        reasoning_steps = self._parse_reasoning_steps(reasoning)

        return SQLGenerationResult(
            sql=sql,
            reasoning=self.format_reasoning(reasoning_steps),
            confidence=confidence,
            alternatives=[],
            metadata={
                "prompt_type": "few_shot" if examples else "zero_shot",
                "model_name": self.system_config.model.model_name,
                "temperature": generation_params.temperature
            }
        )

    async def _generate_multiple_candidates(
        self,
        query: str,
        schema_context: Dict[str, Any],
        num_candidates: int,
        custom_examples: List[Dict[str, str]]
    ) -> SQLGenerationResult:
        """
        Generate multiple SQL query candidates for GRPO training.

        Args:
            query: Natural language query
            schema_context: Schema information
            num_candidates: Number of candidates to generate
            custom_examples: Custom few-shot examples

        Returns:
            SQLGenerationResult with alternatives
        """
        # Format schema context
        schema_str = self._format_schema_for_prompt(schema_context)

        # Use reasoning-first template for better quality
        prompt = self.prompt_templates.reasoning_first_generation(
            query=query,
            schema_context=schema_str,
            force_reasoning=True
        )

        # Generate multiple candidates with different temperatures
        candidates = []
        temperatures = [0.3, 0.5, 0.7, 0.9]  # Vary temperature for diversity

        for i in range(num_candidates):
            temp = temperatures[i % len(temperatures)]

            generation_params = GenerationParams(
                temperature=temp,
                top_p=0.95,
                max_new_tokens=self.system_config.model.max_new_tokens,
                do_sample=True
            )

            generated_text = self.model_wrapper.generate(prompt, generation_params)
            sql, reasoning = self.model_wrapper.extract_sql_and_reasoning(generated_text)

            confidence = self._calculate_confidence(sql, reasoning, schema_context)

            candidates.append({
                "sql": sql,
                "reasoning": reasoning,
                "confidence": confidence,
                "temperature": temp
            })

        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)

        # Best candidate
        best = candidates[0]

        return SQLGenerationResult(
            sql=best['sql'],
            reasoning=best['reasoning'],
            confidence=best['confidence'],
            alternatives=candidates[1:],  # Rest as alternatives
            metadata={
                "num_candidates": num_candidates,
                "generation_mode": "multiple",
                "best_temperature": best['temperature']
            }
        )

    def _format_schema_for_prompt(self, schema_context: Dict[str, Any]) -> str:
        """
        Format schema context for inclusion in prompt.

        Args:
            schema_context: Schema information dictionary

        Returns:
            Formatted schema string
        """
        if isinstance(schema_context, str):
            return schema_context

        # Extract relevant tables
        relevant_tables = schema_context.get("relevant_tables", [])

        if not relevant_tables:
            return "No schema information available"

        return self.prompt_templates.format_schema_context(
            tables=relevant_tables,
            include_descriptions=True,
            include_relationships=True,
            max_columns_per_table=10  # Limit for context window
        )

    def _calculate_confidence(
        self,
        sql: str,
        reasoning: str,
        schema_context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for generated SQL.

        Args:
            sql: Generated SQL query
            reasoning: Reasoning explanation
            schema_context: Schema information

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Check if SQL is not empty
        if sql and len(sql.strip()) > 10:
            confidence += 0.1

        # Check if reasoning is provided
        if reasoning and len(reasoning) > 20:
            confidence += 0.15

        # Check for SELECT statement
        if sql.strip().upper().startswith("SELECT"):
            confidence += 0.1

        # Check if tables mentioned in schema are used
        relevant_tables = schema_context.get("relevant_tables", [])
        if relevant_tables:
            table_names = [t.get("name", "") for t in relevant_tables]
            tables_used = sum(1 for table in table_names if table in sql)
            if tables_used > 0:
                confidence += min(0.15, tables_used * 0.05)

        # Check for proper SQL keywords
        sql_keywords = ["FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]
        keywords_present = sum(1 for kw in sql_keywords if kw in sql.upper())
        confidence += min(0.1, keywords_present * 0.02)

        return min(confidence, 1.0)

    def _parse_reasoning_steps(self, reasoning: str) -> List[str]:
        """
        Parse reasoning text into structured steps.

        Args:
            reasoning: Raw reasoning text

        Returns:
            List of reasoning steps
        """
        if not reasoning:
            return ["No reasoning provided"]

        steps = []

        # Try to parse structured steps
        if "Step" in reasoning:
            import re
            pattern = r'Step \d+: (.+?)(?=Step \d+:|$)'
            matches = re.findall(pattern, reasoning, re.DOTALL)
            if matches:
                steps = [match.strip() for match in matches]

        # If no structured steps, split by sentences
        if not steps:
            sentences = reasoning.split('.')
            steps = [s.strip() for s in sentences if s.strip()]

        return steps[:5]  # Limit to 5 steps

    def _generate_cache_key(self, query: str, schema_context: Any) -> str:
        """
        Generate cache key for query results.

        Args:
            query: Natural language query
            schema_context: Schema information

        Returns:
            Cache key string
        """
        import hashlib
        context_str = json.dumps(schema_context, sort_keys=True) if isinstance(schema_context, dict) else str(schema_context)
        combined = f"{query}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _format_result(self, result: SQLGenerationResult) -> Dict[str, Any]:
        """
        Format SQLGenerationResult for output.

        Args:
            result: SQLGenerationResult object

        Returns:
            Formatted dictionary
        """
        return {
            "sql": result.sql,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "alternatives": result.alternatives,
            "metadata": result.metadata
        }

    async def correct_sql(
        self,
        query: str,
        incorrect_sql: str,
        error_message: str,
        schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attempt to correct an incorrect SQL query.

        Args:
            query: Original natural language query
            incorrect_sql: The SQL that had errors
            error_message: Error message from validation
            schema_context: Schema information

        Returns:
            Corrected SQL generation result
        """
        schema_str = self._format_schema_for_prompt(schema_context)

        prompt = self.prompt_templates.correction_prompt(
            query=query,
            incorrect_sql=incorrect_sql,
            error_message=error_message,
            schema_context=schema_str
        )

        generation_params = GenerationParams(
            temperature=0.2,  # Lower temperature for corrections
            top_p=0.9,
            max_new_tokens=self.system_config.model.max_new_tokens
        )

        generated_text = self.model_wrapper.generate(prompt, generation_params)
        sql, reasoning = self.model_wrapper.extract_sql_and_reasoning(generated_text)

        return {
            "sql": sql,
            "reasoning": reasoning,
            "confidence": self._calculate_confidence(sql, reasoning, schema_context),
            "correction_attempt": True,
            "original_error": error_message
        }

    async def generate_with_feedback(
        self,
        query: str,
        feedback: str,
        previous_sql: str,
        schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate SQL incorporating user feedback.

        Args:
            query: Original query
            feedback: User feedback on previous attempt
            previous_sql: Previously generated SQL
            schema_context: Schema information

        Returns:
            New SQL generation result
        """
        prompt = self.prompt_templates.create_feedback_prompt(
            query=query,
            generated_sql=previous_sql,
            feedback=feedback,
            correct_sql=None  # User didn't provide correct SQL
        )

        generation_params = GenerationParams(
            temperature=0.3,
            top_p=0.95,
            max_new_tokens=self.system_config.model.max_new_tokens
        )

        generated_text = self.model_wrapper.generate(prompt, generation_params)

        # Generate new SQL based on feedback
        new_prompt = self.prompt_templates.zero_shot_sql_generation(
            query=f"{query} (Note: {feedback})",
            schema_context=self._format_schema_for_prompt(schema_context)
        )

        new_sql_text = self.model_wrapper.generate(new_prompt, generation_params)
        sql, reasoning = self.model_wrapper.extract_sql_and_reasoning(new_sql_text)

        return {
            "sql": sql,
            "reasoning": reasoning,
            "confidence": self._calculate_confidence(sql, reasoning, schema_context),
            "incorporated_feedback": feedback
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        if self.model_wrapper:
            return self.model_wrapper.get_model_info()
        return {"status": "not_initialized"}