"""
Example usage of the GL RL Model components.

This script demonstrates how the implemented components work together
for SQL generation, validation, and reward calculation.
"""

import asyncio
import json
from pathlib import Path

# Import our components
from gl_rl_model.agents.schema_analyzer import SchemaAnalyzerAgent
from gl_rl_model.agents.validator import ValidatorAgent
from gl_rl_model.utils.sql_validator import SQLValidator
from gl_rl_model.utils.reward_functions import RewardCalculator

async def demonstrate_sql_validation():
    """Demonstrate SQL validation capabilities."""
    print("\n" + "="*60)
    print("SQL VALIDATION DEMONSTRATION")
    print("="*60)

    # Initialize validator
    validator_agent = ValidatorAgent()
    await validator_agent.initialize()

    # Test queries
    test_queries = [
        {
            "name": "Valid Query",
            "sql": """
                SELECT p.Project_Name, c.Company_Name, p.Budget
                FROM PAC_MNT_PROJECTS p
                INNER JOIN PROJCNTRTS ct ON p.Project_Code = ct.Project_Code
                INNER JOIN SRM_COMPANIES c ON ct.Company_Code = c.Company_Code
                WHERE p.Status = 'Active'
                AND p.Budget > 100000
                LIMIT 10
            """
        },
        {
            "name": "Query with Issues",
            "sql": """
                SELECT *
                FROM INVALID_TABLE
                WHERE Budget < -1000
            """
        },
        {
            "name": "Performance Warning Query",
            "sql": """
                SELECT *
                FROM PAC_MNT_PROJECTS p1
                CROSS JOIN PAC_MNT_PROJECTS p2
                CROSS JOIN PAC_MNT_PROJECTS p3
            """
        }
    ]

    for test in test_queries:
        print(f"\n📝 Testing: {test['name']}")
        print(f"SQL: {test['sql'][:100]}...")

        # Validate
        result = await validator_agent.process({
            "sql": test['sql'],
            "strict_mode": False,
            "check_performance": True,
            "check_security": True
        })

        # Display results
        print(f"✓ Valid: {result['is_valid']}")
        if result.get('errors'):
            for error_type, errors in result['errors'].items():
                if errors:
                    print(f"  ⚠️ {error_type}: {errors}")
        if result.get('suggestions'):
            print(f"  💡 Suggestions: {result['suggestions']}")

    await validator_agent.shutdown()

def demonstrate_reward_calculation():
    """Demonstrate reward calculation for GRPO training."""
    print("\n" + "="*60)
    print("REWARD CALCULATION DEMONSTRATION")
    print("="*60)

    calculator = RewardCalculator()

    # Test scenarios
    test_cases = [
        {
            "name": "Good SQL with Reasoning",
            "sql": """
                SELECT p.Project_Name, SUM(s.Allocation_Percent) as Total_Allocation
                FROM PAC_MNT_PROJECTS p
                INNER JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code
                WHERE p.Status = 'Active'
                GROUP BY p.Project_Name
                LIMIT 20
            """,
            "reasoning": """
                Step 1: Identify that we need project names and staff allocation.
                Step 2: Join PAC_MNT_PROJECTS with PROJSTAFF on Project_Code.
                Step 3: Filter for active projects only.
                Step 4: Group by project name to aggregate allocations.
                Step 5: Add LIMIT for performance.
            """,
            "expected_sql": """
                SELECT p.Project_Name, SUM(s.Allocation_Percent) as Total_Allocation
                FROM PAC_MNT_PROJECTS p
                INNER JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code
                WHERE p.Status = 'Active'
                GROUP BY p.Project_Name
                LIMIT 20
            """
        },
        {
            "name": "Poor SQL without Reasoning",
            "sql": "SELECT * FROM PROJECTS",
            "reasoning": "",
            "expected_sql": "SELECT Project_Name FROM PAC_MNT_PROJECTS WHERE Status = 'Active'"
        }
    ]

    for test in test_cases:
        print(f"\n📊 Testing: {test['name']}")

        # Calculate rewards
        rewards = calculator.calculate_rewards(
            sql=test['sql'],
            reasoning=test['reasoning'],
            expected_sql=test['expected_sql'],
            query="Show project allocations"
        )

        # Display rewards
        print(f"  Total Reward: {rewards.total_reward:.2f}")
        print(f"  Components:")
        print(f"    - Syntax: {rewards.syntax_reward:.2f}")
        print(f"    - Schema: {rewards.schema_compliance_reward:.2f}")
        print(f"    - Business: {rewards.business_logic_reward:.2f}")
        print(f"    - Performance: {rewards.performance_reward:.2f}")
        print(f"    - Reasoning: {rewards.reasoning_quality_reward:.2f}")
        print(f"    - Accuracy: {rewards.accuracy_reward:.2f}")

def demonstrate_sql_parsing():
    """Demonstrate SQL parsing capabilities."""
    print("\n" + "="*60)
    print("SQL PARSING DEMONSTRATION")
    print("="*60)

    parser = SQLValidator()

    sql = """
        SELECT
            c.Company_Name,
            COUNT(p.Project_ID) as Project_Count,
            SUM(pm.Budget) as Total_Budget
        FROM SRM_COMPANIES c
        LEFT JOIN SRM_PROJECTS p ON c.Company_ID = p.Company_ID
        LEFT JOIN PAC_MNT_PROJECTS pm ON p.Project_Code = pm.Project_Code
        WHERE c.Status = 'Active'
        AND pm.Budget > 50000
        GROUP BY c.Company_Name
        HAVING COUNT(p.Project_ID) > 2
        ORDER BY Total_Budget DESC
        LIMIT 10
    """

    print("Parsing SQL Query:")
    print(sql[:150] + "...")

    # Parse SQL
    result = parser.parse_sql(sql)

    print(f"\n📋 Parse Results:")
    print(f"  ✓ Valid: {result.is_valid}")
    print(f"  Query Type: {result.query_type}")
    print(f"  Tables: {result.tables}")
    print(f"  Columns: {result.columns[:3]}...")
    print(f"  Joins: {len(result.joins)} joins detected")
    print(f"  Aggregations: {result.aggregations}")
    print(f"  Group By: {result.group_by}")
    print(f"  Complexity Score: {result.complexity_score:.1f}/10")

    # Check for SQL injection
    is_safe, warnings = parser.check_sql_injection(sql)
    print(f"\n🔒 Security Check:")
    print(f"  Safe: {is_safe}")
    if warnings:
        print(f"  Warnings: {warnings}")

    # Performance estimation
    perf = parser.estimate_performance(sql)
    print(f"\n⚡ Performance Analysis:")
    print(f"  Estimated Cost: {perf['estimated_cost']}")
    if perf['optimization_suggestions']:
        print(f"  Suggestions:")
        for suggestion in perf['optimization_suggestions']:
            print(f"    - {suggestion}")

async def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("GL RL MODEL - COMPONENT DEMONSTRATION")
    print("="*60)
    print("\nThis demonstrates the key components implemented in Phase 2:")
    print("1. SQL Parsing and Analysis")
    print("2. Comprehensive Validation")
    print("3. Multi-dimensional Reward Calculation")

    # Run demonstrations
    demonstrate_sql_parsing()
    demonstrate_reward_calculation()
    await demonstrate_sql_validation()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\n✅ All components are working together successfully!")
    print("🚀 Ready for the next phase: Training Pipeline Implementation")

if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(main())