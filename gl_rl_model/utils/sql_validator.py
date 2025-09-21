"""
SQL parsing and validation utilities for the GL RL Model.
"""

import sqlparse
from sqlparse.sql import Token, TokenList, Identifier, IdentifierList, Function, Where, Comparison
from sqlparse.tokens import Keyword, DML, DDL, Punctuation, Name, Whitespace
from typing import List, Dict, Set, Tuple, Optional, Any
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SQLParseResult:
    """Result of SQL parsing."""
    is_valid: bool
    query_type: str  # SELECT, INSERT, UPDATE, DELETE
    tables: List[str]
    columns: List[str]
    joins: List[Dict[str, Any]]
    where_conditions: List[str]
    aggregations: List[str]
    group_by: List[str]
    order_by: List[str]
    limit: Optional[int]
    errors: List[str]
    warnings: List[str]
    complexity_score: float

class SQLValidator:
    """
    SQL parsing and validation utility class.

    Provides methods for parsing SQL queries, extracting components,
    validating syntax, and analyzing query complexity.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize SQL validator.

        Args:
            schema: Optional schema dictionary for validation
        """
        self.schema = schema or {}
        self.sql_injection_patterns = [
            r"(;|\||--)",  # Command separators and comments
            r"(union|select|insert|delete|update|drop|create|alter|exec|execute|script|javascript)",
            r"(xp_|sp_|0x|ascii|char|concat|cast|convert)",  # SQL Server specific
            r"(\\x[0-9a-fA-F]+|\\[0-9]+)",  # Hex and octal literals
        ]

    def parse_sql(self, sql: str) -> SQLParseResult:
        """
        Parse SQL query and extract components.

        Args:
            sql: SQL query string

        Returns:
            SQLParseResult with extracted components
        """
        result = SQLParseResult(
            is_valid=True,
            query_type="",
            tables=[],
            columns=[],
            joins=[],
            where_conditions=[],
            aggregations=[],
            group_by=[],
            order_by=[],
            limit=None,
            errors=[],
            warnings=[],
            complexity_score=0.0
        )

        try:
            # Parse SQL
            parsed = sqlparse.parse(sql)
            if not parsed:
                result.is_valid = False
                result.errors.append("Failed to parse SQL")
                return result

            statement = parsed[0]

            # Get query type
            result.query_type = self._get_query_type(statement)

            # Extract components based on query type
            if result.query_type == "SELECT":
                self._parse_select_statement(statement, result)
            elif result.query_type in ["INSERT", "UPDATE", "DELETE"]:
                result.errors.append(f"{result.query_type} statements not supported")
                result.is_valid = False
            else:
                result.errors.append(f"Unknown query type: {result.query_type}")
                result.is_valid = False

            # Calculate complexity
            result.complexity_score = self._calculate_complexity(result)

            # Validate if schema is available
            if self.schema:
                self._validate_against_schema(result)

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Parse error: {str(e)}")

        return result

    def _get_query_type(self, statement) -> str:
        """Get the type of SQL query."""
        for token in statement.tokens:
            if token.ttype is DML:
                return token.value.upper()
        return "UNKNOWN"

    def _parse_select_statement(self, statement, result: SQLParseResult):
        """Parse SELECT statement components."""
        # Track parsing state
        in_select = False
        in_from = False
        in_where = False
        in_group = False
        in_order = False
        in_limit = False

        for token in statement.tokens:
            if token.is_whitespace:
                continue

            # Check for keywords
            if token.ttype is Keyword:
                keyword = token.value.upper()
                if keyword == "SELECT":
                    in_select = True
                    in_from = in_where = in_group = in_order = in_limit = False
                elif keyword == "FROM":
                    in_from = True
                    in_select = in_where = in_group = in_order = in_limit = False
                elif keyword == "WHERE":
                    in_where = True
                    in_select = in_from = in_group = in_order = in_limit = False
                elif keyword in ["GROUP", "ORDER"]:
                    if "GROUP" in keyword:
                        in_group = True
                    if "ORDER" in keyword:
                        in_order = True
                    in_select = in_from = in_where = in_limit = False
                elif keyword == "LIMIT":
                    in_limit = True
                    in_select = in_from = in_where = in_group = in_order = False
                elif keyword in ["INNER", "LEFT", "RIGHT", "FULL", "CROSS", "JOIN"]:
                    # Handle joins
                    self._extract_joins(statement, result)

            # Process based on current section
            elif in_select:
                self._extract_select_columns(token, result)
            elif in_from:
                self._extract_tables(token, result)
            elif in_where:
                self._extract_where_conditions(token, result)
            elif in_group:
                self._extract_group_by(token, result)
            elif in_order:
                self._extract_order_by(token, result)
            elif in_limit:
                self._extract_limit(token, result)

    def _extract_select_columns(self, token, result: SQLParseResult):
        """Extract columns from SELECT clause."""
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                col = self._get_name(identifier)
                if col and col != '*':
                    result.columns.append(col)
                # Check for aggregation functions
                if self._is_aggregate_function(identifier):
                    result.aggregations.append(str(identifier))
        elif isinstance(token, Identifier):
            col = self._get_name(token)
            if col:
                result.columns.append(col)
        elif isinstance(token, Function):
            result.aggregations.append(str(token))

    def _extract_tables(self, token, result: SQLParseResult):
        """Extract table names from FROM clause."""
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                table = self._get_table_name(identifier)
                if table:
                    result.tables.append(table)
        elif isinstance(token, Identifier):
            table = self._get_table_name(token)
            if table:
                result.tables.append(table)

    def _extract_where_conditions(self, token, result: SQLParseResult):
        """Extract WHERE conditions."""
        if isinstance(token, Where):
            conditions = str(token).replace("WHERE", "").strip()
            result.where_conditions.append(conditions)
        elif isinstance(token, Comparison):
            result.where_conditions.append(str(token))

    def _extract_joins(self, statement, result: SQLParseResult):
        """Extract JOIN information."""
        sql_text = str(statement)
        join_pattern = r'(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+(\w+)\s+(?:AS\s+)?(\w+)?\s*ON\s+([^;]+?)(?=\s+(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN|WHERE|GROUP|ORDER|LIMIT|$)'

        matches = re.finditer(join_pattern, sql_text, re.IGNORECASE)
        for match in matches:
            join_type = match.group(1) or "INNER"
            table = match.group(2)
            alias = match.group(3)
            condition = match.group(4).strip()

            result.joins.append({
                "type": join_type,
                "table": table,
                "alias": alias,
                "condition": condition
            })

            if table not in result.tables:
                result.tables.append(table)

    def _extract_group_by(self, token, result: SQLParseResult):
        """Extract GROUP BY columns."""
        if not token.is_keyword:
            col = self._get_name(token)
            if col:
                result.group_by.append(col)

    def _extract_order_by(self, token, result: SQLParseResult):
        """Extract ORDER BY columns."""
        if not token.is_keyword:
            col = self._get_name(token)
            if col:
                result.order_by.append(col)

    def _extract_limit(self, token, result: SQLParseResult):
        """Extract LIMIT value."""
        try:
            if token.ttype is None and token.value.isdigit():
                result.limit = int(token.value)
        except:
            pass

    def _get_name(self, token) -> Optional[str]:
        """Get the name from a token."""
        if isinstance(token, Identifier):
            return token.get_name()
        elif hasattr(token, 'value'):
            # Clean up the value
            value = str(token.value).strip()
            if value and not value.upper() in ['SELECT', 'FROM', 'WHERE', 'AND', 'OR']:
                return value
        return None

    def _get_table_name(self, token) -> Optional[str]:
        """Extract table name from token."""
        if isinstance(token, Identifier):
            # Handle aliased tables
            real_name = token.get_real_name()
            return real_name if real_name else token.get_name()
        elif hasattr(token, 'value'):
            value = str(token.value).strip()
            if value and not value.upper() in ['FROM', 'JOIN', 'WHERE']:
                # Handle table.column references
                if '.' in value:
                    return value.split('.')[0]
                return value
        return None

    def _is_aggregate_function(self, token) -> bool:
        """Check if token is an aggregate function."""
        aggregate_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT']
        if isinstance(token, Function):
            return True
        token_str = str(token).upper()
        return any(func in token_str for func in aggregate_functions)

    def _calculate_complexity(self, result: SQLParseResult) -> float:
        """
        Calculate query complexity score.

        Returns:
            Complexity score (0-10)
        """
        score = 0.0

        # Base score for SELECT
        if result.query_type == "SELECT":
            score = 1.0

        # Add complexity for joins
        score += len(result.joins) * 1.5

        # Add for WHERE conditions
        score += len(result.where_conditions) * 0.5

        # Add for aggregations
        score += len(result.aggregations) * 0.8

        # Add for GROUP BY
        score += len(result.group_by) * 0.6

        # Add for subqueries (simplified detection)
        for condition in result.where_conditions:
            if "SELECT" in condition.upper():
                score += 2.0

        return min(score, 10.0)  # Cap at 10

    def _validate_against_schema(self, result: SQLParseResult):
        """Validate parsed SQL against schema."""
        if not self.schema:
            return

        # Validate tables
        valid_tables = self.schema.get('tables', [])
        for table in result.tables:
            if table not in valid_tables:
                result.warnings.append(f"Table '{table}' not found in schema")

        # Validate columns (simplified - would need table context)
        valid_columns = self.schema.get('columns', [])
        for column in result.columns:
            # Remove table prefix if present
            col_name = column.split('.')[-1] if '.' in column else column
            if col_name != '*' and col_name not in valid_columns:
                result.warnings.append(f"Column '{col_name}' might not exist in schema")

    def validate_syntax(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate SQL syntax.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Use sqlparse to check basic syntax
            parsed = sqlparse.parse(sql)
            if not parsed:
                errors.append("Unable to parse SQL")
                return False, errors

            # Format and re-parse to check for issues
            formatted = sqlparse.format(sql, reindent=True, keyword_case='upper')
            reparsed = sqlparse.parse(formatted)

            if not reparsed:
                errors.append("SQL formatting failed")
                return False, errors

            # Check for common syntax errors
            sql_upper = sql.upper()

            # Check for unmatched parentheses
            if sql.count('(') != sql.count(')'):
                errors.append("Unmatched parentheses")

            # Check for SELECT without FROM (unless it's a simple value selection)
            if 'SELECT' in sql_upper and 'FROM' not in sql_upper:
                if not any(func in sql_upper for func in ['CURRENT_DATE', 'CURRENT_TIME', 'DUAL']):
                    errors.append("SELECT statement missing FROM clause")

            # Check for GROUP BY without aggregation
            if 'GROUP BY' in sql_upper and not any(
                func in sql_upper for func in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
            ):
                errors.append("GROUP BY without aggregation function")

            return len(errors) == 0, errors

        except Exception as e:
            errors.append(f"Syntax validation error: {str(e)}")
            return False, errors

    def check_sql_injection(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Check for potential SQL injection patterns.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_safe, warnings)
        """
        warnings = []
        sql_lower = sql.lower()

        for pattern in self.sql_injection_patterns:
            if re.search(pattern, sql_lower):
                warnings.append(f"Potential SQL injection pattern detected: {pattern}")

        # Check for multiple statements
        if ';' in sql and sql.strip()[-1] != ';':
            warnings.append("Multiple SQL statements detected")

        # Check for system tables
        system_tables = ['information_schema', 'mysql', 'sys', 'pg_catalog']
        for table in system_tables:
            if table in sql_lower:
                warnings.append(f"Access to system table '{table}' detected")

        return len(warnings) == 0, warnings

    def extract_table_references(self, sql: str) -> List[str]:
        """
        Extract all table references from SQL.

        Args:
            sql: SQL query string

        Returns:
            List of table names
        """
        result = self.parse_sql(sql)
        return result.tables

    def extract_column_references(self, sql: str) -> List[str]:
        """
        Extract all column references from SQL.

        Args:
            sql: SQL query string

        Returns:
            List of column names
        """
        result = self.parse_sql(sql)
        return result.columns

    def analyze_join_conditions(self, sql: str) -> List[Dict[str, Any]]:
        """
        Analyze JOIN conditions in the query.

        Args:
            sql: SQL query string

        Returns:
            List of join information dictionaries
        """
        result = self.parse_sql(sql)
        return result.joins

    def estimate_performance(self, sql: str) -> Dict[str, Any]:
        """
        Estimate query performance characteristics.

        Args:
            sql: SQL query string

        Returns:
            Performance estimation dictionary
        """
        result = self.parse_sql(sql)

        performance = {
            "complexity_score": result.complexity_score,
            "estimated_cost": "low",
            "optimization_suggestions": []
        }

        # Estimate cost based on complexity
        if result.complexity_score < 3:
            performance["estimated_cost"] = "low"
        elif result.complexity_score < 6:
            performance["estimated_cost"] = "medium"
        else:
            performance["estimated_cost"] = "high"

        # Optimization suggestions
        if len(result.joins) > 3:
            performance["optimization_suggestions"].append(
                "Consider breaking complex joins into smaller queries or using CTEs"
            )

        if not result.limit and result.query_type == "SELECT":
            performance["optimization_suggestions"].append(
                "Consider adding LIMIT clause for large result sets"
            )

        if result.where_conditions == []:
            performance["optimization_suggestions"].append(
                "No WHERE clause detected - ensure this is intentional"
            )

        if len(result.tables) > 5:
            performance["optimization_suggestions"].append(
                "Many tables involved - verify all joins are necessary"
            )

        return performance

    def validate_aggregation_consistency(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate that aggregations are used correctly with GROUP BY.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_valid, error_messages)
        """
        result = self.parse_sql(sql)
        errors = []

        # If there are aggregations, check GROUP BY
        if result.aggregations:
            # Get non-aggregated columns in SELECT
            non_agg_columns = [
                col for col in result.columns
                if not any(agg in col for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX'])
            ]

            # Check if non-aggregated columns are in GROUP BY
            for col in non_agg_columns:
                if col != '*' and col not in result.group_by:
                    errors.append(
                        f"Column '{col}' must be in GROUP BY clause or used in aggregation"
                    )

        # If there's GROUP BY but no aggregation
        if result.group_by and not result.aggregations:
            errors.append("GROUP BY used without aggregation functions")

        return len(errors) == 0, errors

    def format_sql(self, sql: str, style: str = 'standard') -> str:
        """
        Format SQL query for readability.

        Args:
            sql: SQL query string
            style: Formatting style ('standard', 'compact')

        Returns:
            Formatted SQL string
        """
        if style == 'compact':
            return sqlparse.format(
                sql,
                reindent=False,
                keyword_case='upper',
                strip_comments=True,
                compact=True
            )
        else:
            return sqlparse.format(
                sql,
                reindent=True,
                keyword_case='upper',
                strip_comments=True,
                indent_width=2,
                wrap_after=80
            )