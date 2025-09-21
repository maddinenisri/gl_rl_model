"""
Prompt templates for SQL generation with reasoning.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

@dataclass
class PromptTemplate:
    """Base class for prompt templates."""
    name: str
    description: str
    template: str

class SQLPromptTemplates:
    """
    Collection of prompt templates for SQL generation with reasoning.

    Templates follow the Qwen chat format and include support for
    chain-of-thought reasoning.
    """

    @staticmethod
    def zero_shot_sql_generation(
        query: str,
        schema_context: str,
        business_context: Optional[str] = None
    ) -> str:
        """
        Zero-shot SQL generation prompt.

        Args:
            query: Natural language query
            schema_context: Schema information
            business_context: Optional business rules/context

        Returns:
            Formatted prompt
        """
        prompt = "<|im_start|>system\n"
        prompt += "You are an expert SQL analyst for a GL/ERP database system.\n"
        prompt += "CRITICAL: You MUST use the EXACT table names from the schema provided.\n\n"
        prompt += "Available Tables (USE THESE EXACT NAMES):\n"
        prompt += "- PAC_MNT_PROJECTS (for all project-related queries)\n"
        prompt += "- SRM_COMPANIES (for all company-related queries)\n"
        prompt += "- PROJSTAFF (for staff/resource assignments)\n"
        prompt += "- PROJCNTRTS (for contracts)\n"
        prompt += "- PAC_MNT_RESOURCES (for resources)\n"
        prompt += "- SRM_CONTACTS (for contacts)\n\n"
        prompt += "NEVER use generic names like 'projects', 'companies', 'staff', etc.\n"
        prompt += "ALWAYS use the full table names as listed above.\n\n"
        prompt += "Instructions:\n"
        prompt += "1. Identify which of the above tables are needed\n"
        prompt += "2. Use ONLY the table names listed above\n"
        prompt += "3. Generate syntactically correct SQL\n"

        if business_context:
            prompt += f"\nBusiness Context:\n{business_context}\n"

        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += f"Schema Details:\n{schema_context}\n\n"
        prompt += f"Query: {query}\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    @staticmethod
    def few_shot_sql_generation(
        query: str,
        schema_context: str,
        examples: List[Dict[str, str]],
        business_context: Optional[str] = None
    ) -> str:
        """
        Few-shot SQL generation prompt with examples.

        Args:
            query: Natural language query
            schema_context: Schema information
            examples: List of example query-SQL pairs
            business_context: Optional business rules/context

        Returns:
            Formatted prompt
        """
        prompt = "<|im_start|>system\n"
        prompt += "You are an expert SQL analyst specializing in financial ERP systems. "
        prompt += "Generate accurate SQL queries following the examples provided.\n"
        prompt += "Always provide step-by-step reasoning using <think> tags.\n"

        if business_context:
            prompt += f"\nBusiness Context:\n{business_context}\n"

        prompt += "<|im_end|>\n"

        # Add examples
        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
            prompt += "<|im_start|>user\n"
            prompt += f"Example {i}:\n"
            if 'schema' in example:
                prompt += f"Schema: {example['schema']}\n"
            prompt += f"Query: {example['query']}\n"
            prompt += "<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            # Add reasoning if available
            if 'reasoning' in example:
                prompt += "<think>\n"
                prompt += f"{example['reasoning']}\n"
                prompt += "</think>\n\n"

            prompt += f"{example['sql']}\n"
            prompt += "<|im_end|>\n"

        # Add actual query
        prompt += "<|im_start|>user\n"
        prompt += f"Now generate SQL for this query:\n"
        prompt += f"Schema:\n{schema_context}\n\n"
        prompt += f"Query: {query}\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    @staticmethod
    def reasoning_first_generation(
        query: str,
        schema_context: str,
        force_reasoning: bool = True
    ) -> str:
        """
        Prompt that enforces reasoning before SQL generation.

        Args:
            query: Natural language query
            schema_context: Schema information
            force_reasoning: Whether to explicitly require reasoning

        Returns:
            Formatted prompt
        """
        prompt = "<|im_start|>system\n"
        prompt += "You are an expert SQL analyst. Your task is to:\n"
        prompt += "1. First, think through the problem step-by-step\n"
        prompt += "2. Explain your reasoning clearly\n"
        prompt += "3. Then generate the SQL query\n\n"

        if force_reasoning:
            prompt += "IMPORTANT: You MUST provide reasoning in <think> tags before the SQL.\n"
            prompt += "The reasoning should include:\n"
            prompt += "- Understanding of what the query asks for\n"
            prompt += "- Identification of relevant tables\n"
            prompt += "- Explanation of joins needed\n"
            prompt += "- Justification for filters and conditions\n"

        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += f"Schema:\n{schema_context}\n\n"
        prompt += f"Query: {query}\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        prompt += "<think>\n"  # Start reasoning section

        return prompt

    @staticmethod
    def schema_aware_generation(
        query: str,
        relevant_tables: List[Dict[str, Any]],
        relationships: Dict[str, List[str]],
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Schema-aware prompt with detailed table information.

        Args:
            query: Natural language query
            relevant_tables: List of relevant table definitions
            relationships: Table relationship information
            examples: Optional examples

        Returns:
            Formatted prompt
        """
        prompt = "<|im_start|>system\n"
        prompt += "You are an SQL expert. Generate queries using the provided schema.\n"
        prompt += "Focus on the relevant tables and their relationships.\n"
        prompt += "<|im_end|>\n"

        # Add examples if provided
        if examples:
            for example in examples[:2]:
                prompt += "<|im_start|>user\n"
                prompt += f"Query: {example['query']}\n"
                prompt += "<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"
                if 'reasoning' in example:
                    prompt += f"<think>\n{example['reasoning']}\n</think>\n\n"
                prompt += f"{example['sql']}\n"
                prompt += "<|im_end|>\n"

        prompt += "<|im_start|>user\n"
        prompt += "Relevant Tables:\n"

        # Format table information
        for table in relevant_tables:
            prompt += f"\nTable: {table['name']}\n"
            if 'description' in table:
                prompt += f"Description: {table['description']}\n"
            prompt += "Columns:\n"
            for col in table.get('columns', []):
                prompt += f"  - {col['name']} ({col['type']})"
                if col.get('nullable') == False:
                    prompt += " NOT NULL"
                prompt += "\n"
            if 'primary_keys' in table:
                prompt += f"Primary Keys: {', '.join(table['primary_keys'])}\n"
            if 'foreign_keys' in table and table['foreign_keys']:
                prompt += "Foreign Keys:\n"
                for fk_col, ref in table['foreign_keys'].items():
                    prompt += f"  - {fk_col} -> {ref}\n"

        # Add relationships
        if relationships:
            prompt += "\nTable Relationships:\n"
            for table, related in relationships.items():
                if related:
                    prompt += f"  - {table} relates to: {', '.join(related)}\n"

        prompt += f"\nQuery: {query}\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    @staticmethod
    def correction_prompt(
        query: str,
        incorrect_sql: str,
        error_message: str,
        schema_context: str
    ) -> str:
        """
        Prompt for correcting an incorrect SQL query.

        Args:
            query: Original natural language query
            incorrect_sql: The SQL that had errors
            error_message: Error message from validation
            schema_context: Schema information

        Returns:
            Formatted prompt
        """
        prompt = "<|im_start|>system\n"
        prompt += "You are an SQL expert. A previous SQL query had errors.\n"
        prompt += "Analyze the error and generate a corrected version.\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += f"Original Query: {query}\n\n"
        prompt += f"Incorrect SQL:\n{incorrect_sql}\n\n"
        prompt += f"Error: {error_message}\n\n"
        prompt += f"Schema:\n{schema_context}\n\n"
        prompt += "Please provide the corrected SQL with reasoning.\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        prompt += "<think>\n"
        prompt += f"The error '{error_message}' suggests that "  # Model will complete

        return prompt

    @staticmethod
    def training_prompt(
        query: str,
        schema_context: str,
        expected_sql: str,
        reasoning: Optional[str] = None
    ) -> str:
        """
        Training prompt format with expected output.

        Args:
            query: Natural language query
            schema_context: Schema information
            expected_sql: The correct SQL query
            reasoning: Optional reasoning explanation

        Returns:
            Formatted training prompt
        """
        prompt = "<|im_start|>system\n"
        prompt += "You are an expert SQL analyst for a GL/ERP database.\n"
        prompt += "ALWAYS use these exact table names:\n"
        prompt += "PAC_MNT_PROJECTS, SRM_COMPANIES, PROJSTAFF, PROJCNTRTS, PAC_MNT_RESOURCES, SRM_CONTACTS\n"
        prompt += "NEVER use generic table names.\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += f"Schema:\n{schema_context}\n\n"
        prompt += f"Query: {query}\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        if reasoning:
            prompt += "<think>\n"
            prompt += f"{reasoning}\n"
            prompt += "</think>\n\n"

        prompt += f"{expected_sql}\n"
        prompt += "<|im_end|>"

        return prompt

    @staticmethod
    def generate_training_prompt(
        query: str,
        sql: str,
        reasoning: str = "",
        schema_context: str = None
    ) -> str:
        """
        Generate training prompt with schema context.

        Args:
            query: Natural language query
            sql: Expected SQL output
            reasoning: Reasoning explanation
            schema_context: Optional schema context (will be loaded if not provided)

        Returns:
            Formatted training prompt
        """
        # Load schema context if not provided
        if schema_context is None:
            from ..training.schema_loader import SchemaLoader
            schema_loader = SchemaLoader()
            schema_context = schema_loader.get_schema_context(query)

        return SQLPromptTemplates.training_prompt(
            query=query,
            schema_context=schema_context,
            expected_sql=sql,
            reasoning=reasoning if reasoning else None
        )

    @staticmethod
    def format_schema_context(
        tables: List[Dict[str, Any]],
        include_descriptions: bool = True,
        include_relationships: bool = True,
        max_columns_per_table: Optional[int] = None
    ) -> str:
        """
        Format schema information for inclusion in prompts.

        Args:
            tables: List of table definitions
            include_descriptions: Whether to include table descriptions
            include_relationships: Whether to include foreign keys
            max_columns_per_table: Limit columns shown per table

        Returns:
            Formatted schema string
        """
        schema_str = ""

        for table in tables:
            schema_str += f"Table: {table['name']}"

            if include_descriptions and 'description' in table:
                schema_str += f" -- {table['description']}"
            schema_str += "\n"

            # Add columns
            columns = table.get('columns', [])
            if max_columns_per_table:
                columns = columns[:max_columns_per_table]

            for col in columns:
                schema_str += f"  {col['name']} {col['type']}"
                if col.get('nullable') == False:
                    schema_str += " NOT NULL"
                if 'description' in col:
                    schema_str += f" -- {col['description']}"
                schema_str += "\n"

            # Add keys
            if 'primary_keys' in table and table['primary_keys']:
                schema_str += f"  PRIMARY KEY: {', '.join(table['primary_keys'])}\n"

            if include_relationships and 'foreign_keys' in table and table['foreign_keys']:
                for fk_col, ref in table['foreign_keys'].items():
                    schema_str += f"  FOREIGN KEY: {fk_col} -> {ref}\n"

            schema_str += "\n"

        return schema_str.strip()

    @staticmethod
    def format_reasoning_steps(steps: List[str]) -> str:
        """
        Format reasoning steps for inclusion in output.

        Args:
            steps: List of reasoning steps

        Returns:
            Formatted reasoning string
        """
        reasoning = "<think>\n"
        for i, step in enumerate(steps, 1):
            reasoning += f"Step {i}: {step}\n"
        reasoning += "</think>"
        return reasoning

    @staticmethod
    def extract_sql_from_response(response: str) -> tuple[str, str]:
        """
        Extract SQL and reasoning from model response.

        Args:
            response: Model generated response

        Returns:
            Tuple of (sql_query, reasoning)
        """
        reasoning = ""
        sql = response

        # Extract reasoning from <think> tags
        if "<think>" in response and "</think>" in response:
            start = response.index("<think>") + 7
            end = response.index("</think>")
            reasoning = response[start:end].strip()
            # Get SQL after reasoning
            sql = response[end + 8:].strip()

        # Clean up SQL
        sql = sql.strip()

        # Remove any system tokens
        sql = sql.replace("<|im_end|>", "").strip()
        sql = sql.replace("<|im_start|>", "").strip()

        # Remove SELECT prefix if duplicated
        if sql.upper().startswith("SQL:"):
            sql = sql[4:].strip()

        return sql, reasoning

    @staticmethod
    def create_feedback_prompt(
        query: str,
        generated_sql: str,
        feedback: str,
        correct_sql: Optional[str] = None
    ) -> str:
        """
        Create a prompt for learning from feedback.

        Args:
            query: Original query
            generated_sql: SQL that was generated
            feedback: Feedback on the generated SQL
            correct_sql: Optional correct SQL

        Returns:
            Formatted feedback prompt
        """
        prompt = "<|im_start|>system\n"
        prompt += "Learn from the feedback to improve SQL generation.\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += f"Query: {query}\n\n"
        prompt += f"Generated SQL:\n{generated_sql}\n\n"
        prompt += f"Feedback: {feedback}\n"

        if correct_sql:
            prompt += f"\nCorrect SQL:\n{correct_sql}\n"

        prompt += "\nExplain what was wrong and how to fix it.\n"
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt