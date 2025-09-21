"""
Validator Agent for SQL query validation and compliance checking.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import re

from ..core.base_agent import BaseAgent, AgentStatus
from ..core.config import get_config
from ..utils.sql_validator import SQLValidator, SQLParseResult

@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    is_valid: bool
    syntax_valid: bool
    schema_compliant: bool
    business_logic_valid: bool
    performance_acceptable: bool
    security_passed: bool

    # Detailed results
    syntax_errors: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)
    business_errors: List[str] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    security_warnings: List[str] = field(default_factory=list)

    # Suggestions
    suggestions: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)

    # Metadata
    parse_result: Optional[SQLParseResult] = None
    complexity_score: float = 0.0
    confidence: float = 0.0

class ValidatorAgent(BaseAgent):
    """
    Agent responsible for comprehensive SQL validation.

    Performs multi-layer validation including syntax, schema compliance,
    business logic, performance analysis, and security checks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the validator agent."""
        super().__init__("validator", config)
        self.system_config = get_config()
        self.sql_validator = SQLValidator()
        self.schema_loaded = False
        self.business_rules = {}
        self.performance_thresholds = {}
        self._load_validation_rules()

    def _load_validation_rules(self):
        """Load business rules and performance thresholds."""
        # Business rules for the ERP system
        self.business_rules = {
            "date_ranges": {
                "max_days": 365,  # Maximum date range for queries
                "require_date_filter": ["CLNTSUPP", "PROJEVISION"],  # Tables requiring date filters
            },
            "status_values": {
                "SRM_PROJECTS": ["Active", "Inactive", "Planning", "Completed"],
                "PAC_MNT_PROJECTS": ["Active", "Inactive", "On Hold"],
                "CLNTSUPP": ["Open", "In Progress", "Resolved", "Closed"],
                "PAC_MNT_RESOURCES": ["Available", "Busy", "Unavailable"]
            },
            "budget_constraints": {
                "min_budget": 0,
                "max_budget": 999999999.99,
                "currency_required": ["Contract_Value", "Budget", "Actual_Cost"]
            },
            "project_hierarchy": {
                "max_depth": 5,  # Maximum project hierarchy depth
                "require_parent_check": True
            },
            "required_filters": {
                # Tables that should always have WHERE clauses
                "large_tables": ["CLNTSUPP", "PROJEVISION", "PROJSTAFF"],
                "sensitive_tables": ["SRM_CONTACTS", "PROJCNTRTS"]
            }
        }

        # Performance thresholds
        self.performance_thresholds = {
            "max_tables_without_limit": 3,
            "max_join_count": 5,
            "max_subquery_depth": 2,
            "require_index_hint": ["CLNTSUPP", "PROJEVISION"],  # Large tables
            "max_complexity_score": 7.0,
            "warn_cartesian_product": True
        }

    async def initialize(self) -> bool:
        """
        Initialize the validator agent.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Validator Agent")

            # Load schema information
            await self._load_schema()

            self.status = AgentStatus.IDLE
            self.schema_loaded = True
            self.logger.info("Validator Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Validator: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the validator agent.

        Returns:
            True if shutdown successful, False otherwise
        """
        try:
            self.logger.info("Shutting down Validator Agent")
            self.business_rules.clear()
            self.performance_thresholds.clear()
            self.status = AgentStatus.IDLE
            return True
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a validation request.

        Args:
            input_data: Dictionary containing:
                - sql: SQL query to validate
                - schema_context: Schema information
                - strict_mode: Whether to enforce strict validation
                - check_performance: Whether to check performance
                - check_security: Whether to check security

        Returns:
            Dictionary containing validation results
        """
        try:
            sql = input_data.get("sql", "")
            schema_context = input_data.get("schema_context", {})
            strict_mode = input_data.get("strict_mode", self.system_config.agent.validator_strict_mode)
            check_performance = input_data.get("check_performance", True)
            check_security = input_data.get("check_security", True)

            if not sql:
                return {
                    "is_valid": False,
                    "error": "No SQL provided for validation"
                }

            # Perform comprehensive validation
            result = await self._perform_validation(
                sql, schema_context, strict_mode, check_performance, check_security
            )

            return self._format_validation_result(result)

        except Exception as e:
            self.logger.error(f"Error processing validation: {e}")
            return {
                "is_valid": False,
                "error": str(e)
            }

    async def _perform_validation(
        self,
        sql: str,
        schema_context: Dict[str, Any],
        strict_mode: bool,
        check_performance: bool,
        check_security: bool
    ) -> ValidationResult:
        """
        Perform comprehensive SQL validation.

        Args:
            sql: SQL query
            schema_context: Schema information
            strict_mode: Whether to enforce strict validation
            check_performance: Whether to check performance
            check_security: Whether to check security

        Returns:
            ValidationResult with all validation details
        """
        result = ValidationResult(
            is_valid=True,
            syntax_valid=True,
            schema_compliant=True,
            business_logic_valid=True,
            performance_acceptable=True,
            security_passed=True
        )

        # Layer 1: Syntax Validation
        syntax_valid, syntax_errors = self._validate_syntax(sql)
        result.syntax_valid = syntax_valid
        result.syntax_errors = syntax_errors
        if not syntax_valid and strict_mode:
            result.is_valid = False
            return result

        # Parse SQL for further validation
        parse_result = self.sql_validator.parse_sql(sql)
        result.parse_result = parse_result
        result.complexity_score = parse_result.complexity_score

        if not parse_result.is_valid:
            result.syntax_valid = False
            result.syntax_errors.extend(parse_result.errors)
            result.is_valid = False
            return result

        # Layer 2: Schema Compliance
        schema_valid, schema_errors = self._validate_schema_compliance(
            parse_result, schema_context
        )
        result.schema_compliant = schema_valid
        result.schema_errors = schema_errors
        if not schema_valid and strict_mode:
            result.is_valid = False

        # Layer 3: Business Logic
        business_valid, business_errors = self._validate_business_logic(
            parse_result, sql
        )
        result.business_logic_valid = business_valid
        result.business_errors = business_errors
        if not business_valid and strict_mode:
            result.is_valid = False

        # Layer 4: Performance Analysis
        if check_performance:
            perf_acceptable, perf_warnings, optimizations = self._analyze_performance(
                parse_result, sql
            )
            result.performance_acceptable = perf_acceptable
            result.performance_warnings = perf_warnings
            result.optimizations = optimizations

        # Layer 5: Security Checks
        if check_security:
            security_passed, security_warnings = self._check_security(sql)
            result.security_passed = security_passed
            result.security_warnings = security_warnings
            if not security_passed:
                result.is_valid = False

        # Generate suggestions
        result.suggestions = self._generate_suggestions(result)

        # Calculate confidence
        result.confidence = self._calculate_confidence(result)

        # Final validation status
        if strict_mode:
            result.is_valid = all([
                result.syntax_valid,
                result.schema_compliant,
                result.business_logic_valid,
                result.security_passed
            ])
        else:
            result.is_valid = result.syntax_valid and result.security_passed

        return result

    def _validate_syntax(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate SQL syntax.

        Args:
            sql: SQL query

        Returns:
            Tuple of (is_valid, errors)
        """
        return self.sql_validator.validate_syntax(sql)

    def _validate_schema_compliance(
        self,
        parse_result: SQLParseResult,
        schema_context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate schema compliance.

        Args:
            parse_result: Parsed SQL result
            schema_context: Schema information

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        # Get valid tables and columns from schema context
        valid_tables = set()
        valid_columns = {}

        if "relevant_tables" in schema_context:
            for table in schema_context["relevant_tables"]:
                table_name = table.get("name", "")
                valid_tables.add(table_name)
                valid_columns[table_name] = [
                    col.get("name", "") for col in table.get("columns", [])
                ]

        # Add core tables from config
        valid_tables.update(self.system_config.schema.core_tables)

        # Validate tables
        for table in parse_result.tables:
            if table not in valid_tables:
                errors.append(f"Table '{table}' not found in schema")

        # Validate columns (when table context is available)
        for column in parse_result.columns:
            if '.' in column:
                table, col = column.split('.', 1)
                if table in valid_columns and col not in valid_columns[table]:
                    if col != '*':
                        errors.append(f"Column '{col}' not found in table '{table}'")

        # Validate joins
        for join in parse_result.joins:
            join_table = join.get("table", "")
            if join_table and join_table not in valid_tables:
                errors.append(f"Join table '{join_table}' not found in schema")

            # Check join conditions reference valid columns
            condition = join.get("condition", "")
            if condition and "=" in condition:
                # Simple check for column existence in join condition
                for part in condition.split("="):
                    part = part.strip()
                    if "." in part:
                        table, col = part.split(".", 1)
                        table = table.strip()
                        col = col.strip()
                        if table in valid_columns and col not in valid_columns[table]:
                            errors.append(f"Join condition references invalid column: {table}.{col}")

        return len(errors) == 0, errors

    def _validate_business_logic(
        self,
        parse_result: SQLParseResult,
        sql: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate business logic rules.

        Args:
            parse_result: Parsed SQL result
            sql: Original SQL query

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        sql_upper = sql.upper()

        # Check status values
        for table, valid_statuses in self.business_rules["status_values"].items():
            if table in parse_result.tables:
                # Look for status conditions
                status_pattern = rf"STATUS\s*=\s*['\"](\w+)['\"]"
                matches = re.finditer(status_pattern, sql, re.IGNORECASE)
                for match in matches:
                    status = match.group(1)
                    if status not in valid_statuses:
                        errors.append(
                            f"Invalid status '{status}' for table {table}. "
                            f"Valid values: {', '.join(valid_statuses)}"
                        )

        # Check date ranges
        if any("DATE" in cond.upper() for cond in parse_result.where_conditions):
            # Simplified date range check
            date_pattern = r"(\w+DATE\w*)\s*(?:BETWEEN|>=?|<=?)"
            if re.search(date_pattern, sql_upper):
                # Check if tables requiring date filters have them
                for table in self.business_rules["date_ranges"]["require_date_filter"]:
                    if table in parse_result.tables and not parse_result.where_conditions:
                        errors.append(f"Table {table} requires date filter in WHERE clause")

        # Check required filters for large/sensitive tables
        for table in parse_result.tables:
            if table in self.business_rules["required_filters"]["large_tables"]:
                if not parse_result.where_conditions:
                    errors.append(f"Large table {table} requires WHERE clause for performance")
            if table in self.business_rules["required_filters"]["sensitive_tables"]:
                if not parse_result.where_conditions:
                    errors.append(f"Sensitive table {table} requires WHERE clause for security")

        # Check budget constraints
        if "BUDGET" in sql_upper or "COST" in sql_upper or "VALUE" in sql_upper:
            # Check for negative values
            negative_pattern = r"(?:BUDGET|COST|VALUE)\s*<?=?\s*-\d+"
            if re.search(negative_pattern, sql_upper):
                errors.append("Budget/Cost/Value cannot be negative")

        # Check project hierarchy
        if "PARENT_ID" in sql_upper and "SRM_PROJECTS" in parse_result.tables:
            # Ensure hierarchical queries have depth limit
            if "CONNECT BY" in sql_upper or "WITH RECURSIVE" in sql_upper:
                if "LEVEL" not in sql_upper and "depth" not in sql.lower():
                    errors.append("Hierarchical query should have depth limit")

        # Check for GROUP BY consistency with aggregations
        if parse_result.aggregations and not parse_result.group_by:
            # Check if all columns are aggregated
            non_agg_cols = [
                col for col in parse_result.columns
                if not any(agg in col.upper() for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX'])
            ]
            if non_agg_cols and '*' not in non_agg_cols:
                errors.append("Non-aggregated columns must be in GROUP BY clause")

        return len(errors) == 0, errors

    def _analyze_performance(
        self,
        parse_result: SQLParseResult,
        sql: str
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Analyze query performance.

        Args:
            parse_result: Parsed SQL result
            sql: SQL query

        Returns:
            Tuple of (is_acceptable, warnings, optimizations)
        """
        warnings = []
        optimizations = []
        is_acceptable = True

        # Check complexity score
        if parse_result.complexity_score > self.performance_thresholds["max_complexity_score"]:
            warnings.append(
                f"Query complexity ({parse_result.complexity_score:.1f}) exceeds threshold "
                f"({self.performance_thresholds['max_complexity_score']})"
            )
            is_acceptable = False

        # Check join count
        if len(parse_result.joins) > self.performance_thresholds["max_join_count"]:
            warnings.append(
                f"Too many joins ({len(parse_result.joins)}). "
                f"Maximum recommended: {self.performance_thresholds['max_join_count']}"
            )
            optimizations.append("Consider breaking query into smaller parts or using CTEs")

        # Check for LIMIT on large tables
        if (len(parse_result.tables) > self.performance_thresholds["max_tables_without_limit"]
            and not parse_result.limit):
            warnings.append("Query on multiple tables without LIMIT clause")
            optimizations.append("Add LIMIT clause to restrict result set size")

        # Check for cartesian products
        if (self.performance_thresholds["warn_cartesian_product"]
            and len(parse_result.tables) > 1
            and len(parse_result.joins) < len(parse_result.tables) - 1):
            warnings.append("Potential cartesian product detected (missing joins)")
            is_acceptable = False

        # Check for missing WHERE on large tables
        if not parse_result.where_conditions:
            for table in parse_result.tables:
                if table in self.performance_thresholds.get("require_index_hint", []):
                    warnings.append(f"Large table {table} accessed without WHERE clause")
                    optimizations.append(f"Add WHERE clause to filter {table} records")

        # Check for SELECT *
        if '*' in parse_result.columns:
            optimizations.append("Avoid SELECT *, specify required columns explicitly")

        # Check for subqueries in WHERE
        if any("SELECT" in cond.upper() for cond in parse_result.where_conditions):
            optimizations.append("Consider replacing subqueries with JOINs for better performance")

        # Check for OR conditions
        if any("OR" in cond.upper() for cond in parse_result.where_conditions):
            optimizations.append("Multiple OR conditions may prevent index usage, consider UNION")

        # Check for functions in WHERE
        function_pattern = r"\w+\([^)]*\)\s*(?:=|<|>|LIKE)"
        if any(re.search(function_pattern, cond, re.IGNORECASE)
              for cond in parse_result.where_conditions):
            optimizations.append("Functions in WHERE clause may prevent index usage")

        return is_acceptable, warnings, optimizations

    def _check_security(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Check for security issues.

        Args:
            sql: SQL query

        Returns:
            Tuple of (is_safe, warnings)
        """
        return self.sql_validator.check_sql_injection(sql)

    def _generate_suggestions(self, result: ValidationResult) -> List[str]:
        """
        Generate suggestions based on validation results.

        Args:
            result: ValidationResult

        Returns:
            List of suggestions
        """
        suggestions = []

        # Syntax suggestions
        if result.syntax_errors:
            suggestions.append("Fix syntax errors before proceeding")

        # Schema suggestions
        if result.schema_errors:
            suggestions.append("Verify table and column names against schema")

        # Business logic suggestions
        if result.business_errors:
            for error in result.business_errors:
                if "date filter" in error.lower():
                    suggestions.append("Add date range filter to improve performance and accuracy")
                elif "status" in error.lower():
                    suggestions.append("Use valid status values from the business rules")

        # Performance suggestions
        if result.performance_warnings:
            suggestions.append("Consider performance optimizations for production use")

        # Security suggestions
        if result.security_warnings:
            suggestions.append("Review query for potential security vulnerabilities")

        # General suggestions based on parse result
        if result.parse_result:
            if not result.parse_result.limit and result.parse_result.query_type == "SELECT":
                suggestions.append("Consider adding LIMIT for testing")

            if len(result.parse_result.tables) == 1 and not result.parse_result.where_conditions:
                suggestions.append("Add WHERE clause to filter results")

        return suggestions

    def _calculate_confidence(self, result: ValidationResult) -> float:
        """
        Calculate confidence score for validation result.

        Args:
            result: ValidationResult

        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0

        # Reduce confidence for errors
        confidence -= len(result.syntax_errors) * 0.2
        confidence -= len(result.schema_errors) * 0.15
        confidence -= len(result.business_errors) * 0.1
        confidence -= len(result.performance_warnings) * 0.05
        confidence -= len(result.security_warnings) * 0.25

        # Reduce confidence for high complexity
        if result.complexity_score > 7:
            confidence -= 0.1
        elif result.complexity_score > 5:
            confidence -= 0.05

        return max(0.0, min(1.0, confidence))

    async def _load_schema(self):
        """Load schema information."""
        # In production, this would load from the schema files
        # For now, we use the configuration
        self.sql_validator.schema = {
            "tables": self.system_config.schema.core_tables,
            "columns": []  # Would be loaded from schema files
        }

    def _format_validation_result(self, result: ValidationResult) -> Dict[str, Any]:
        """
        Format validation result for output.

        Args:
            result: ValidationResult

        Returns:
            Formatted dictionary
        """
        return {
            "is_valid": result.is_valid,
            "validation": {
                "syntax_valid": result.syntax_valid,
                "schema_compliant": result.schema_compliant,
                "business_logic_valid": result.business_logic_valid,
                "performance_acceptable": result.performance_acceptable,
                "security_passed": result.security_passed
            },
            "errors": {
                "syntax": result.syntax_errors,
                "schema": result.schema_errors,
                "business": result.business_errors
            },
            "warnings": {
                "performance": result.performance_warnings,
                "security": result.security_warnings
            },
            "suggestions": result.suggestions,
            "optimizations": result.optimizations,
            "metadata": {
                "complexity_score": result.complexity_score,
                "confidence": result.confidence,
                "tables_count": len(result.parse_result.tables) if result.parse_result else 0,
                "joins_count": len(result.parse_result.joins) if result.parse_result else 0
            }
        }

    async def validate_batch(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate multiple queries in batch.

        Args:
            queries: List of query dictionaries

        Returns:
            List of validation results
        """
        results = []
        for query_data in queries:
            result = await self.process(query_data)
            results.append(result)
        return results