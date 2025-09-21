"""
SQL Post-processor for Domain Table Enforcement

Automatically corrects generic table names to domain-specific ones.
"""

import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SQLPostProcessor:
    """Post-process generated SQL to ensure domain-specific table names."""

    def __init__(self):
        """Initialize the post-processor with table mappings."""

        # Mapping of common generic names to domain tables
        self.table_mappings = {
            # Project-related mappings
            "projects": "PAC_MNT_PROJECTS",
            "project": "PAC_MNT_PROJECTS",
            "proj": "PAC_MNT_PROJECTS",
            "project_table": "PAC_MNT_PROJECTS",
            "projects_table": "PAC_MNT_PROJECTS",
            "pac_projects": "PAC_MNT_PROJECTS",

            # Company-related mappings
            "companies": "SRM_COMPANIES",
            "company": "SRM_COMPANIES",
            "comp": "SRM_COMPANIES",
            "company_table": "SRM_COMPANIES",
            "companies_table": "SRM_COMPANIES",
            "organizations": "SRM_COMPANIES",
            "organisation": "SRM_COMPANIES",
            "organizations": "SRM_COMPANIES",
            "srm_company": "SRM_COMPANIES",

            # Staff-related mappings
            "staff": "PROJSTAFF",
            "project_staff": "PROJSTAFF",
            "staff_table": "PROJSTAFF",
            "assignments": "PROJSTAFF",
            "staff_assignments": "PROJSTAFF",
            "project_assignments": "PROJSTAFF",
            "allocations": "PROJSTAFF",

            # Contract-related mappings
            "contracts": "PROJCNTRTS",
            "contract": "PROJCNTRTS",
            "project_contracts": "PROJCNTRTS",
            "contract_table": "PROJCNTRTS",
            "agreements": "PROJCNTRTS",

            # Resource-related mappings
            "resources": "PAC_MNT_RESOURCES",
            "resource": "PAC_MNT_RESOURCES",
            "resource_table": "PAC_MNT_RESOURCES",
            "pac_resources": "PAC_MNT_RESOURCES",
            "personnel": "PAC_MNT_RESOURCES",
            "employees": "PAC_MNT_RESOURCES",

            # Contact-related mappings
            "contacts": "SRM_CONTACTS",
            "contact": "SRM_CONTACTS",
            "contact_table": "SRM_CONTACTS",
            "srm_contact": "SRM_CONTACTS",
            "company_contacts": "SRM_CONTACTS"
        }

        # Column mappings for common mismatches
        self.column_mappings = {
            # Common column name variations
            "project_id": "Project_ID",
            "project_code": "Project_Code",
            "project_name": "Project_Name",
            "status": "Status",
            "budget": "Budget",
            "actual_cost": "Actual_Cost",
            "start_date": "Start_Date",
            "end_date": "End_Date",
            "department": "Department",
            "revenue": "Revenue",

            "company_id": "Company_ID",
            "company_code": "Company_Code",
            "company_name": "Company_Name",

            "staff_id": "Staff_ID",
            "resource_code": "Resource_Code",
            "allocation_percent": "Allocation_Percent",
            "bill_rate": "Bill_Rate",
            "cost_rate": "Cost_Rate",
            "role": "Role",

            "contract_number": "Contract_Number",
            "contract_value": "Contract_Value",
            "contract_type": "Contract_Type",

            "resource_name": "Resource_Name",
            "resource_type": "Resource_Type",
            "email": "Email",
            "capacity": "Capacity",
            "availability": "Availability",

            "contact_id": "Contact_ID",
            "contact_name": "Contact_Name",
            "phone": "Phone"
        }

        # Domain table list for validation
        self.domain_tables = [
            "PAC_MNT_PROJECTS",
            "SRM_COMPANIES",
            "PROJSTAFF",
            "PROJCNTRTS",
            "PAC_MNT_RESOURCES",
            "SRM_CONTACTS"
        ]

    def process(self, sql: str, query: Optional[str] = None) -> Tuple[str, bool, List[str]]:
        """
        Process SQL to ensure domain-specific table names.

        Args:
            sql: Generated SQL query
            query: Optional natural language query for context

        Returns:
            Tuple of (processed_sql, was_corrected, corrections_made)
        """
        if not sql:
            return sql, False, []

        original_sql = sql
        processed_sql = sql
        corrections = []

        # Apply table name corrections
        processed_sql, table_corrections = self._correct_table_names(processed_sql, query)
        corrections.extend(table_corrections)

        # Apply column name corrections
        processed_sql, column_corrections = self._correct_column_names(processed_sql)
        corrections.extend(column_corrections)

        # Validate final SQL has domain tables
        has_domain = self._validate_domain_tables(processed_sql)

        if not has_domain and query:
            # Try context-based correction
            processed_sql, context_corrections = self._context_based_correction(processed_sql, query)
            corrections.extend(context_corrections)

        was_corrected = processed_sql != original_sql

        if was_corrected:
            logger.info(f"SQL corrected: {len(corrections)} changes made")
            for correction in corrections:
                logger.debug(f"  - {correction}")

        return processed_sql, was_corrected, corrections

    def _correct_table_names(self, sql: str, query: Optional[str] = None) -> Tuple[str, List[str]]:
        """Correct generic table names to domain-specific ones."""
        corrected_sql = sql
        corrections = []

        # Sort mappings by length (longest first) to avoid partial replacements
        sorted_mappings = sorted(self.table_mappings.items(), key=lambda x: len(x[0]), reverse=True)

        for generic, domain in sorted_mappings:
            # Create regex patterns for table name replacement
            patterns = [
                (r'\bFROM\s+' + re.escape(generic) + r'\b', f'FROM {domain}'),
                (r'\bfrom\s+' + re.escape(generic) + r'\b', f'FROM {domain}'),
                (r'\bJOIN\s+' + re.escape(generic) + r'\b', f'JOIN {domain}'),
                (r'\bjoin\s+' + re.escape(generic) + r'\b', f'JOIN {domain}'),
                (r'\bINTO\s+' + re.escape(generic) + r'\b', f'INTO {domain}'),
                (r'\binto\s+' + re.escape(generic) + r'\b', f'INTO {domain}'),
                (r'\bUPDATE\s+' + re.escape(generic) + r'\b', f'UPDATE {domain}'),
                (r'\bupdate\s+' + re.escape(generic) + r'\b', f'UPDATE {domain}'),
                # Handle table aliases
                (r'\b' + re.escape(generic) + r'\s+(?=[a-zA-Z])', domain + ' '),
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, corrected_sql, re.IGNORECASE):
                    corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                    corrections.append(f"Replaced '{generic}' with '{domain}'")

        return corrected_sql, corrections

    def _correct_column_names(self, sql: str) -> Tuple[str, List[str]]:
        """Correct column names to match schema conventions."""
        corrected_sql = sql
        corrections = []

        for generic, proper in self.column_mappings.items():
            # Only replace if not already properly cased
            if proper not in corrected_sql and generic in corrected_sql.lower():
                # Create patterns for column replacement
                pattern = r'\b' + re.escape(generic) + r'\b'
                if re.search(pattern, corrected_sql, re.IGNORECASE):
                    corrected_sql = re.sub(pattern, proper, corrected_sql, flags=re.IGNORECASE)
                    corrections.append(f"Corrected column '{generic}' to '{proper}'")

        return corrected_sql, corrections

    def _validate_domain_tables(self, sql: str) -> bool:
        """Check if SQL uses at least one domain table."""
        sql_upper = sql.upper()
        return any(table in sql_upper for table in self.domain_tables)

    def _context_based_correction(self, sql: str, query: str) -> Tuple[str, List[str]]:
        """Apply corrections based on query context."""
        corrected_sql = sql
        corrections = []
        query_lower = query.lower()

        # Determine likely table based on query keywords
        table_hints = {
            "project": "PAC_MNT_PROJECTS",
            "company": "SRM_COMPANIES",
            "companies": "SRM_COMPANIES",
            "staff": "PROJSTAFF",
            "resource": "PAC_MNT_RESOURCES",
            "contract": "PROJCNTRTS",
            "contact": "SRM_CONTACTS"
        }

        for keyword, table in table_hints.items():
            if keyword in query_lower and table not in corrected_sql:
                # Try to identify generic SELECT statements
                generic_pattern = r'SELECT\s+\*\s+FROM\s+(\w+)'
                match = re.search(generic_pattern, corrected_sql, re.IGNORECASE)
                if match:
                    generic_table = match.group(1)
                    if generic_table.upper() not in self.domain_tables:
                        corrected_sql = corrected_sql.replace(
                            f"FROM {generic_table}",
                            f"FROM {table}",
                            1
                        )
                        corrections.append(f"Context-based: replaced '{generic_table}' with '{table}'")
                        break

        return corrected_sql, corrections

    def validate_and_fix(self, sql: str, query: Optional[str] = None) -> Dict[str, any]:
        """
        Comprehensive validation and fixing of SQL.

        Returns:
            Dictionary with processed SQL and metadata
        """
        processed_sql, was_corrected, corrections = self.process(sql, query)

        # Check if processed SQL uses domain tables
        uses_domain = self._validate_domain_tables(processed_sql)

        # Calculate confidence based on corrections needed
        confidence = 1.0
        if was_corrected:
            confidence -= 0.1 * len(corrections)
        if not uses_domain:
            confidence -= 0.5

        confidence = max(0.1, confidence)  # Minimum confidence

        return {
            "original_sql": sql,
            "processed_sql": processed_sql,
            "was_corrected": was_corrected,
            "corrections": corrections,
            "uses_domain_tables": uses_domain,
            "confidence": confidence
        }

    def extract_tables_used(self, sql: str) -> List[str]:
        """Extract list of tables used in SQL."""
        tables = []
        sql_upper = sql.upper()

        for table in self.domain_tables:
            if table in sql_upper:
                tables.append(table)

        return tables

    def suggest_improvements(self, sql: str, query: str) -> List[str]:
        """Suggest improvements for the SQL based on query intent."""
        suggestions = []

        # Check if using domain tables
        if not self._validate_domain_tables(sql):
            suggestions.append("Use domain-specific table names (PAC_MNT_PROJECTS, SRM_COMPANIES, etc.)")

        # Check for common patterns
        query_lower = query.lower()

        if "active" in query_lower and "Status = 'Active'" not in sql:
            suggestions.append("Consider adding Status = 'Active' filter")

        if "budget" in query_lower and "Budget" not in sql:
            suggestions.append("Include Budget column or filter")

        if "count" in query_lower and "COUNT" not in sql.upper():
            suggestions.append("Use COUNT() aggregate function")

        if "group" in query_lower or "per" in query_lower:
            if "GROUP BY" not in sql.upper():
                suggestions.append("Consider using GROUP BY clause")

        if "join" in query_lower or "with their" in query_lower:
            if "JOIN" not in sql.upper():
                suggestions.append("Consider using JOIN to combine tables")

        return suggestions


# Singleton instance for easy import
sql_postprocessor = SQLPostProcessor()

def postprocess_sql(sql: str, query: Optional[str] = None) -> str:
    """
    Convenience function to postprocess SQL.

    Args:
        sql: Generated SQL
        query: Optional natural language query

    Returns:
        Processed SQL with domain tables
    """
    processed_sql, _, _ = sql_postprocessor.process(sql, query)
    return processed_sql