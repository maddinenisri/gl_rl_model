"""
Schema loader for providing context during training.

This module loads and formats schema information for training prompts.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

class SchemaLoader:
    """Loads and provides schema context for training."""

    def __init__(self, schema_dir: str = None):
        """Initialize schema loader."""
        if schema_dir is None:
            schema_dir = Path(__file__).parent.parent / "data" / "schema"

        self.schema_dir = Path(schema_dir)
        self.entity_mappings = {}
        self.table_definitions = {}
        self.relationships = {}

        self._load_schema()

    def _load_schema(self):
        """Load schema files."""
        # Load entity mappings
        entity_file = self.schema_dir / "entity_mappings.json"
        if entity_file.exists():
            with open(entity_file, 'r') as f:
                self.entity_mappings = json.load(f)

        # Parse DDL for table definitions (simplified)
        self._parse_ddl()

    def _parse_ddl(self):
        """Parse DDL file to extract table definitions."""
        ddl_file = self.schema_dir / "ddl_schema.sql"
        if not ddl_file.exists():
            return

        # Simple parsing of key tables
        self.table_definitions = {
            "PAC_MNT_PROJECTS": {
                "columns": [
                    "Project_ID", "Project_Code", "Project_Name", "Status",
                    "Budget", "Actual_Cost", "Start_Date", "End_Date",
                    "Department", "Revenue"
                ],
                "description": "Main projects table"
            },
            "SRM_COMPANIES": {
                "columns": [
                    "Company_ID", "Company_Code", "Company_Name", "Status"
                ],
                "description": "Companies and organizations"
            },
            "PROJSTAFF": {
                "columns": [
                    "Staff_ID", "Project_Code", "Resource_Code", "Role",
                    "Allocation_Percent", "Bill_Rate", "Cost_Rate", "Status"
                ],
                "description": "Project staff assignments"
            },
            "PAC_MNT_RESOURCES": {
                "columns": [
                    "Resource_Code", "Resource_Name", "Resource_Type",
                    "Email", "Capacity", "Cost_Rate", "Availability"
                ],
                "description": "Resources and personnel"
            },
            "PROJCNTRTS": {
                "columns": [
                    "Contract_Number", "Project_Code", "Company_Code",
                    "Contract_Value", "Contract_Type", "Start_Date", "End_Date", "Status"
                ],
                "description": "Project contracts"
            },
            "SRM_CONTACTS": {
                "columns": [
                    "Contact_ID", "Company_ID", "Contact_Name", "Email", "Phone"
                ],
                "description": "Company contacts"
            }
        }

        # Define key relationships
        self.relationships = {
            "PAC_MNT_PROJECTS": ["PROJSTAFF", "PROJCNTRTS", "PROJEVISION"],
            "SRM_COMPANIES": ["SRM_CONTACTS", "PROJCNTRTS"],
            "PAC_MNT_RESOURCES": ["PROJSTAFF"]
        }

    def get_schema_context(self, query: str = "") -> str:
        """
        Get formatted schema context for a query.

        Args:
            query: The natural language query (to determine relevant tables)

        Returns:
            Formatted schema context string
        """
        # Determine relevant tables based on query keywords
        relevant_tables = self._identify_relevant_tables(query)

        # Format schema context
        context = "Database Schema:\n\n"

        for table_name in relevant_tables:
            if table_name in self.table_definitions:
                table_def = self.table_definitions[table_name]
                context += f"Table: {table_name}\n"
                context += f"Description: {table_def['description']}\n"
                context += f"Columns: {', '.join(table_def['columns'])}\n"

                # Add relationships
                if table_name in self.relationships:
                    context += f"Related tables: {', '.join(self.relationships[table_name])}\n"

                context += "\n"

        # Add common join patterns
        context += "Common Join Patterns:\n"
        context += "- Projects ↔ Staff: PAC_MNT_PROJECTS.Project_Code = PROJSTAFF.Project_Code\n"
        context += "- Projects ↔ Contracts: PAC_MNT_PROJECTS.Project_Code = PROJCNTRTS.Project_Code\n"
        context += "- Staff ↔ Resources: PROJSTAFF.Resource_Code = PAC_MNT_RESOURCES.Resource_Code\n"
        context += "- Companies ↔ Contacts: SRM_COMPANIES.Company_ID = SRM_CONTACTS.Company_ID\n"
        context += "- Companies ↔ Contracts: SRM_COMPANIES.Company_Code = PROJCNTRTS.Company_Code\n"

        return context

    def _identify_relevant_tables(self, query: str) -> List[str]:
        """Identify relevant tables based on query keywords."""
        query_lower = query.lower()
        tables = []

        # Always include main tables for context
        tables.extend(["PAC_MNT_PROJECTS", "SRM_COMPANIES"])

        # Add tables based on keywords
        if any(word in query_lower for word in ["resource", "staff", "allocation", "team"]):
            tables.extend(["PAC_MNT_RESOURCES", "PROJSTAFF"])

        if any(word in query_lower for word in ["contact", "email", "phone"]):
            tables.append("SRM_CONTACTS")

        if any(word in query_lower for word in ["contract", "agreement"]):
            tables.append("PROJCNTRTS")

        if any(word in query_lower for word in ["budget", "cost", "financial"]):
            if "PAC_MNT_PROJECTS" not in tables:
                tables.append("PAC_MNT_PROJECTS")

        # Remove duplicates while preserving order
        seen = set()
        unique_tables = []
        for table in tables:
            if table not in seen:
                seen.add(table)
                unique_tables.append(table)

        return unique_tables

    def get_training_context(self) -> str:
        """Get comprehensive schema context for training."""
        context = "Complete Database Schema:\n\n"

        # Include all tables for training
        for table_name, table_def in self.table_definitions.items():
            context += f"Table: {table_name}\n"
            context += f"Description: {table_def['description']}\n"
            context += f"Columns: {', '.join(table_def['columns'])}\n"

            if table_name in self.relationships:
                context += f"Related tables: {', '.join(self.relationships[table_name])}\n"

            context += "\n"

        return context