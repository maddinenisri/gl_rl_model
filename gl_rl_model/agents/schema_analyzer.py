"""
Schema Analyzer Agent for understanding and mapping database schemas.
"""

import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import re

from ..core.base_agent import ReasoningAgent, AgentStatus
from ..core.config import get_config

@dataclass
class Table:
    """Represents a database table."""
    name: str
    columns: List[Dict[str, Any]] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, str] = field(default_factory=dict)  # column -> referenced_table.column
    description: Optional[str] = None

@dataclass
class SchemaContext:
    """Context information about the schema relevant to a query."""
    relevant_tables: List[Table]
    relationships: Dict[str, List[str]]  # table -> list of related tables
    join_paths: List[List[str]]  # possible join paths between tables
    business_entities: Dict[str, str]  # entity name -> primary table

class SchemaAnalyzerAgent(ReasoningAgent):
    """
    Agent responsible for analyzing database schemas and providing context for SQL generation.

    This agent understands the ERD structure, identifies relevant tables for queries,
    and provides schema context to other agents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the schema analyzer agent."""
        super().__init__("schema_analyzer", config)
        self.system_config = get_config()
        self.schema: Dict[str, Table] = {}
        self.entity_mappings: Dict[str, str] = {}
        self.relationship_graph: Dict[str, Set[str]] = {}
        self.schema_loaded = False

    async def initialize(self) -> bool:
        """
        Initialize the schema analyzer and load schema information.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Schema Analyzer Agent")

            # Load schema from configuration
            await self._load_schema()

            # Build relationship graph
            self._build_relationship_graph()

            # Load entity mappings
            await self._load_entity_mappings()

            self.status = AgentStatus.IDLE
            self.schema_loaded = True
            self.logger.info("Schema Analyzer Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Schema Analyzer: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the schema analyzer agent.

        Returns:
            True if shutdown successful, False otherwise
        """
        try:
            self.logger.info("Shutting down Schema Analyzer Agent")
            self.schema.clear()
            self.entity_mappings.clear()
            self.relationship_graph.clear()
            self.status = AgentStatus.IDLE
            return True
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a schema analysis request.

        Args:
            input_data: Dictionary containing:
                - query: Natural language query
                - tables: Optional list of specific tables to analyze
                - depth: Optional depth for relationship traversal

        Returns:
            Dictionary containing:
                - schema_context: Relevant schema information
                - reasoning: Explanation of schema analysis
                - suggestions: Suggestions for query construction
        """
        try:
            query = input_data.get("query", "")
            specific_tables = input_data.get("tables", [])
            depth = input_data.get("depth", 2)

            # Generate reasoning for schema analysis
            reasoning_steps = []

            # Step 1: Identify entities in the query
            entities = self._identify_entities(query)
            reasoning_steps.append(f"Identified entities in query: {', '.join(entities)}")

            # Step 2: Map entities to tables
            relevant_tables = self._map_entities_to_tables(entities, specific_tables)
            reasoning_steps.append(f"Mapped to tables: {', '.join([t.name for t in relevant_tables])}")

            # Step 3: Find relationships
            relationships = self._find_relationships(relevant_tables, depth)
            reasoning_steps.append(f"Found {len(relationships)} table relationships")

            # Step 4: Determine join paths
            join_paths = self._find_join_paths(relevant_tables)
            reasoning_steps.append(f"Identified {len(join_paths)} possible join paths")

            # Step 5: Extract business context
            business_context = self._extract_business_context(query, relevant_tables)
            reasoning_steps.append("Extracted business context from query")

            # Format reasoning
            reasoning = self.format_reasoning(reasoning_steps)
            self.add_reasoning_step(query, reasoning)

            # Build schema context
            schema_context = SchemaContext(
                relevant_tables=relevant_tables,
                relationships=relationships,
                join_paths=join_paths,
                business_entities=business_context
            )

            # Generate suggestions
            suggestions = self._generate_suggestions(schema_context, query)

            return {
                "schema_context": self._serialize_schema_context(schema_context),
                "reasoning": reasoning,
                "suggestions": suggestions,
                "confidence": self._calculate_confidence(schema_context)
            }

        except Exception as e:
            self.logger.error(f"Error processing schema analysis: {e}")
            return {
                "error": str(e),
                "schema_context": {},
                "reasoning": "",
                "suggestions": []
            }

    async def _load_schema(self):
        """Load schema information from the ERD."""
        # Based on the ERD image, define the schema structure
        self.schema = {
            "SRM_PROJECTS": Table(
                name="SRM_PROJECTS",
                columns=[
                    {"name": "Project_ID", "type": "INTEGER", "nullable": False},
                    {"name": "Project_Name", "type": "VARCHAR(255)"},
                    {"name": "Company_ID", "type": "INTEGER"},
                    {"name": "Start_Date", "type": "DATE"},
                    {"name": "End_Date", "type": "DATE"},
                    {"name": "Status", "type": "VARCHAR(50)"}
                ],
                primary_keys=["Project_ID"],
                foreign_keys={"Company_ID": "SRM_COMPANIES.Company_ID"},
                description="Project master data"
            ),
            "SRM_COMPANIES": Table(
                name="SRM_COMPANIES",
                columns=[
                    {"name": "Company_ID", "type": "INTEGER", "nullable": False},
                    {"name": "Company_Name", "type": "VARCHAR(255)"},
                    {"name": "Company_Code", "type": "VARCHAR(50)"},
                    {"name": "Principal_ID", "type": "INTEGER"},
                    {"name": "Industry", "type": "VARCHAR(100)"}
                ],
                primary_keys=["Company_ID"],
                foreign_keys={},
                description="Company master data"
            ),
            "SRM_CONTACTS": Table(
                name="SRM_CONTACTS",
                columns=[
                    {"name": "Contact_ID", "type": "INTEGER", "nullable": False},
                    {"name": "Company_ID", "type": "INTEGER"},
                    {"name": "Contact_Name", "type": "VARCHAR(255)"},
                    {"name": "Email", "type": "VARCHAR(255)"},
                    {"name": "Phone", "type": "VARCHAR(50)"}
                ],
                primary_keys=["Contact_ID"],
                foreign_keys={"Company_ID": "SRM_COMPANIES.Company_ID"},
                description="Contact information"
            ),
            "PAC_MNT_PROJECTS": Table(
                name="PAC_MNT_PROJECTS",
                columns=[
                    {"name": "Project_Code", "type": "VARCHAR(50)", "nullable": False},
                    {"name": "Project_Name", "type": "VARCHAR(255)"},
                    {"name": "Budget", "type": "DECIMAL(15,2)"},
                    {"name": "Actual_Cost", "type": "DECIMAL(15,2)"},
                    {"name": "Status", "type": "VARCHAR(50)"}
                ],
                primary_keys=["Project_Code"],
                foreign_keys={},
                description="Project accounting and maintenance"
            ),
            "PAC_MNT_RESOURCES": Table(
                name="PAC_MNT_RESOURCES",
                columns=[
                    {"name": "Resource_Code", "type": "VARCHAR(50)", "nullable": False},
                    {"name": "Resource_Name", "type": "VARCHAR(255)"},
                    {"name": "Resource_Type", "type": "VARCHAR(50)"},
                    {"name": "Cost_Rate", "type": "DECIMAL(10,2)"},
                    {"name": "Availability", "type": "VARCHAR(50)"}
                ],
                primary_keys=["Resource_Code"],
                foreign_keys={},
                description="Resource management"
            ),
            "CLNTSUPP": Table(
                name="CLNTSUPP",
                columns=[
                    {"name": "Support_ID", "type": "INTEGER", "nullable": False},
                    {"name": "Company_Code", "type": "VARCHAR(50)"},
                    {"name": "Support_Type", "type": "VARCHAR(100)"},
                    {"name": "Status", "type": "VARCHAR(50)"},
                    {"name": "Priority", "type": "VARCHAR(20)"}
                ],
                primary_keys=["Support_ID"],
                foreign_keys={"Company_Code": "SRM_COMPANIES.Company_Code"},
                description="Client support tickets"
            ),
            "CLNTRESPONS": Table(
                name="CLNTRESPONS",
                columns=[
                    {"name": "Response_ID", "type": "INTEGER", "nullable": False},
                    {"name": "Project_Code", "type": "VARCHAR(50)"},
                    {"name": "Response_Date", "type": "DATE"},
                    {"name": "Response_Type", "type": "VARCHAR(100)"},
                    {"name": "Status", "type": "VARCHAR(50)"}
                ],
                primary_keys=["Response_ID"],
                foreign_keys={"Project_Code": "PAC_MNT_PROJECTS.Project_Code"},
                description="Client responses"
            ),
            "PROJCNTRTS": Table(
                name="PROJCNTRTS",
                columns=[
                    {"name": "Contract_ID", "type": "INTEGER", "nullable": False},
                    {"name": "Project_Code", "type": "VARCHAR(50)"},
                    {"name": "Company_Code", "type": "VARCHAR(50)"},
                    {"name": "Contract_Value", "type": "DECIMAL(15,2)"},
                    {"name": "Start_Date", "type": "DATE"},
                    {"name": "End_Date", "type": "DATE"}
                ],
                primary_keys=["Contract_ID"],
                foreign_keys={
                    "Project_Code": "PAC_MNT_PROJECTS.Project_Code",
                    "Company_Code": "SRM_COMPANIES.Company_Code"
                },
                description="Project contracts"
            ),
            "PROJSTAFF": Table(
                name="PROJSTAFF",
                columns=[
                    {"name": "Staff_ID", "type": "INTEGER", "nullable": False},
                    {"name": "Project_Code", "type": "VARCHAR(50)"},
                    {"name": "Resource_Code", "type": "VARCHAR(50)"},
                    {"name": "Role", "type": "VARCHAR(100)"},
                    {"name": "Allocation_Percent", "type": "DECIMAL(5,2)"}
                ],
                primary_keys=["Staff_ID"],
                foreign_keys={
                    "Project_Code": "PAC_MNT_PROJECTS.Project_Code",
                    "Resource_Code": "PAC_MNT_RESOURCES.Resource_Code"
                },
                description="Project staffing"
            ),
            "PROJEVISION": Table(
                name="PROJEVISION",
                columns=[
                    {"name": "Revision_ID", "type": "INTEGER", "nullable": False},
                    {"name": "Project_Code", "type": "VARCHAR(50)"},
                    {"name": "Revision_Date", "type": "DATE"},
                    {"name": "Revision_Type", "type": "VARCHAR(100)"},
                    {"name": "Description", "type": "TEXT"}
                ],
                primary_keys=["Revision_ID"],
                foreign_keys={"Project_Code": "PAC_MNT_PROJECTS.Project_Code"},
                description="Project revisions"
            )
        }

        self.logger.info(f"Loaded schema with {len(self.schema)} tables")

    def _build_relationship_graph(self):
        """Build a graph of table relationships."""
        self.relationship_graph = {}

        for table_name, table in self.schema.items():
            if table_name not in self.relationship_graph:
                self.relationship_graph[table_name] = set()

            # Add relationships based on foreign keys
            for fk_column, ref_table_column in table.foreign_keys.items():
                ref_table = ref_table_column.split('.')[0]
                self.relationship_graph[table_name].add(ref_table)

                # Add reverse relationship
                if ref_table not in self.relationship_graph:
                    self.relationship_graph[ref_table] = set()
                self.relationship_graph[ref_table].add(table_name)

    async def _load_entity_mappings(self):
        """Load business entity to table mappings."""
        self.entity_mappings = {
            "project": ["SRM_PROJECTS", "PAC_MNT_PROJECTS"],
            "company": ["SRM_COMPANIES"],
            "contact": ["SRM_CONTACTS"],
            "resource": ["PAC_MNT_RESOURCES"],
            "staff": ["PROJSTAFF"],
            "contract": ["PROJCNTRTS"],
            "support": ["CLNTSUPP"],
            "client": ["CLNTSUPP", "CLNTRESPONS"],
            "revision": ["PROJEVISION"],
            "budget": ["PAC_MNT_PROJECTS"],
            "cost": ["PAC_MNT_PROJECTS", "PAC_MNT_RESOURCES"]
        }

    def _identify_entities(self, query: str) -> List[str]:
        """Identify business entities mentioned in the query."""
        entities = []
        query_lower = query.lower()

        for entity, _ in self.entity_mappings.items():
            if entity in query_lower or entity + 's' in query_lower:
                entities.append(entity)

        # If no entities found, default to project
        if not entities:
            entities = ["project"]

        return entities

    def _map_entities_to_tables(self, entities: List[str], specific_tables: List[str]) -> List[Table]:
        """Map identified entities to database tables."""
        tables = []
        table_names = set()

        # If specific tables are provided, use them
        if specific_tables:
            for table_name in specific_tables:
                if table_name in self.schema:
                    tables.append(self.schema[table_name])
                    table_names.add(table_name)
        else:
            # Map entities to tables
            for entity in entities:
                if entity in self.entity_mappings:
                    for table_name in self.entity_mappings[entity]:
                        if table_name in self.schema and table_name not in table_names:
                            tables.append(self.schema[table_name])
                            table_names.add(table_name)

        return tables

    def _find_relationships(self, tables: List[Table], depth: int) -> Dict[str, List[str]]:
        """Find relationships between tables up to specified depth."""
        relationships = {}
        table_names = {t.name for t in tables}

        for table in tables:
            related = []
            visited = set()
            self._traverse_relationships(table.name, depth, visited, related, table_names)
            if related:
                relationships[table.name] = related

        return relationships

    def _traverse_relationships(self, table: str, depth: int, visited: Set[str],
                               related: List[str], target_tables: Set[str]):
        """Recursively traverse table relationships."""
        if depth <= 0 or table in visited:
            return

        visited.add(table)

        if table in self.relationship_graph:
            for related_table in self.relationship_graph[table]:
                if related_table in target_tables and related_table not in related:
                    related.append(related_table)
                self._traverse_relationships(related_table, depth - 1, visited, related, target_tables)

    def _find_join_paths(self, tables: List[Table]) -> List[List[str]]:
        """Find possible join paths between tables."""
        join_paths = []

        if len(tables) < 2:
            return join_paths

        # For each pair of tables, find join paths
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                path = self._find_path(tables[i].name, tables[j].name)
                if path:
                    join_paths.append(path)

        return join_paths

    def _find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find a path between two tables using BFS."""
        if start == end:
            return [start]

        visited = set()
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current in visited:
                continue

            visited.add(current)

            if current in self.relationship_graph:
                for neighbor in self.relationship_graph[current]:
                    if neighbor == end:
                        return path + [neighbor]
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        return None

    def _extract_business_context(self, query: str, tables: List[Table]) -> Dict[str, str]:
        """Extract business context from query and tables."""
        context = {}

        for table in tables:
            if "project" in table.name.lower():
                context["project_management"] = table.name
            elif "company" in table.name.lower():
                context["company_data"] = table.name
            elif "resource" in table.name.lower():
                context["resource_allocation"] = table.name
            elif "contract" in table.name.lower():
                context["contract_management"] = table.name

        return context

    def _generate_suggestions(self, schema_context: SchemaContext, query: str) -> List[str]:
        """Generate suggestions for query construction."""
        suggestions = []

        # Suggest joins if multiple tables
        if len(schema_context.relevant_tables) > 1:
            suggestions.append(f"Consider joining {len(schema_context.relevant_tables)} tables")

        # Suggest filters based on query keywords
        if "recent" in query.lower() or "latest" in query.lower():
            suggestions.append("Add date filters for recent data")

        if "active" in query.lower():
            suggestions.append("Filter by status = 'Active'")

        # Suggest aggregations
        if any(word in query.lower() for word in ["count", "total", "sum", "average"]):
            suggestions.append("Use appropriate aggregation functions")

        return suggestions

    def _serialize_schema_context(self, context: SchemaContext) -> Dict[str, Any]:
        """Serialize schema context for transmission."""
        return {
            "relevant_tables": [
                {
                    "name": t.name,
                    "columns": t.columns,
                    "primary_keys": t.primary_keys,
                    "foreign_keys": t.foreign_keys,
                    "description": t.description
                }
                for t in context.relevant_tables
            ],
            "relationships": context.relationships,
            "join_paths": context.join_paths,
            "business_entities": context.business_entities
        }

    def _calculate_confidence(self, context: SchemaContext) -> float:
        """Calculate confidence score for schema analysis."""
        score = 0.5  # Base score

        # Increase confidence based on number of relevant tables found
        if context.relevant_tables:
            score += min(len(context.relevant_tables) * 0.1, 0.3)

        # Increase confidence if join paths are found
        if context.join_paths:
            score += 0.1

        # Increase confidence if business entities are identified
        if context.business_entities:
            score += 0.1

        return min(score, 1.0)