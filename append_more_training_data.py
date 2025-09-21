#!/usr/bin/env python3
"""
Append more training examples to reach 150+ total, focusing on missing patterns
"""
import json
from pathlib import Path

def generate_additional_examples():
    """Generate additional training examples focusing on PAC_MNT_RESOURCES and more complex patterns."""

    examples = []

    # ==========================================================================
    # PAC_MNT_RESOURCES QUERIES (Missing from first batch)
    # ==========================================================================

    examples.extend([
        {
            "query": "Show all resources",
            "sql": "SELECT * FROM PAC_MNT_RESOURCES",
            "reasoning": "Retrieve all records from PAC_MNT_RESOURCES table"
        },
        {
            "query": "List available resources",
            "sql": "SELECT * FROM PAC_MNT_RESOURCES WHERE Availability = 'Available'",
            "reasoning": "Filter PAC_MNT_RESOURCES by Availability status"
        },
        {
            "query": "Find resources with high cost rate",
            "sql": "SELECT * FROM PAC_MNT_RESOURCES WHERE Cost_Rate > 150",
            "reasoning": "Query PAC_MNT_RESOURCES where Cost_Rate exceeds 150"
        },
        {
            "query": "Show human resources",
            "sql": "SELECT * FROM PAC_MNT_RESOURCES WHERE Resource_Type = 'Human'",
            "reasoning": "Filter PAC_MNT_RESOURCES by Resource_Type"
        },
        {
            "query": "Get equipment resources",
            "sql": "SELECT Resource_Code, Resource_Name FROM PAC_MNT_RESOURCES WHERE Resource_Type = 'Equipment'",
            "reasoning": "Query PAC_MNT_RESOURCES for equipment type"
        },
        {
            "query": "List resources by capacity",
            "sql": "SELECT * FROM PAC_MNT_RESOURCES ORDER BY Capacity DESC",
            "reasoning": "Order PAC_MNT_RESOURCES by Capacity descending"
        },
        {
            "query": "Find resources with email",
            "sql": "SELECT Resource_Name, Email FROM PAC_MNT_RESOURCES WHERE Email IS NOT NULL",
            "reasoning": "Filter PAC_MNT_RESOURCES for non-null Email"
        },
        {
            "query": "Show resources with capacity over 40 hours",
            "sql": "SELECT * FROM PAC_MNT_RESOURCES WHERE Capacity > 40",
            "reasoning": "Filter PAC_MNT_RESOURCES by Capacity"
        },
        {
            "query": "Get resource allocation details",
            "sql": "SELECT r.Resource_Name, s.Project_Code, s.Allocation_Percent FROM PAC_MNT_RESOURCES r JOIN PROJSTAFF s ON r.Resource_Code = s.Resource_Code",
            "reasoning": "Join PAC_MNT_RESOURCES with PROJSTAFF"
        },
        {
            "query": "Find unallocated resources",
            "sql": "SELECT * FROM PAC_MNT_RESOURCES WHERE Resource_Code NOT IN (SELECT DISTINCT Resource_Code FROM PROJSTAFF)",
            "reasoning": "Subquery to find resources not in PROJSTAFF"
        }
    ])

    # ==========================================================================
    # SRM_CONTACTS QUERIES (Also important)
    # ==========================================================================

    examples.extend([
        {
            "query": "Show all contacts",
            "sql": "SELECT * FROM SRM_CONTACTS",
            "reasoning": "Retrieve all records from SRM_CONTACTS table"
        },
        {
            "query": "List contacts for a company",
            "sql": "SELECT * FROM SRM_CONTACTS WHERE Company_ID = 1",
            "reasoning": "Filter SRM_CONTACTS by Company_ID"
        },
        {
            "query": "Find contacts with email",
            "sql": "SELECT Contact_Name, Email FROM SRM_CONTACTS WHERE Email IS NOT NULL",
            "reasoning": "Query SRM_CONTACTS for non-null emails"
        },
        {
            "query": "Get company contacts with phone",
            "sql": "SELECT c.Company_Name, ct.Contact_Name, ct.Phone FROM SRM_COMPANIES c JOIN SRM_CONTACTS ct ON c.Company_ID = ct.Company_ID",
            "reasoning": "Join SRM_COMPANIES with SRM_CONTACTS"
        },
        {
            "query": "Count contacts per company",
            "sql": "SELECT Company_ID, COUNT(*) as contact_count FROM SRM_CONTACTS GROUP BY Company_ID",
            "reasoning": "Group SRM_CONTACTS by Company_ID and count"
        }
    ])

    # ==========================================================================
    # MORE COMPLEX MULTI-TABLE JOINS
    # ==========================================================================

    examples.extend([
        {
            "query": "Show projects with resources and their allocations",
            "sql": "SELECT p.Project_Name, r.Resource_Name, s.Allocation_Percent FROM PAC_MNT_PROJECTS p JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code JOIN PAC_MNT_RESOURCES r ON s.Resource_Code = r.Resource_Code",
            "reasoning": "Three-table join: PAC_MNT_PROJECTS, PROJSTAFF, PAC_MNT_RESOURCES"
        },
        {
            "query": "Get complete project team information",
            "sql": "SELECT p.Project_Name, r.Resource_Name, s.Role, s.Allocation_Percent, r.Cost_Rate FROM PAC_MNT_PROJECTS p INNER JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code INNER JOIN PAC_MNT_RESOURCES r ON s.Resource_Code = r.Resource_Code",
            "reasoning": "Multi-join for complete team view"
        },
        {
            "query": "Show projects with companies and contracts",
            "sql": "SELECT p.Project_Name, c.Company_Name, ct.Contract_Number, ct.Contract_Value FROM PAC_MNT_PROJECTS p JOIN PROJCNTRTS ct ON p.Project_Code = ct.Project_Code JOIN SRM_COMPANIES c ON ct.Company_Code = c.Company_Code",
            "reasoning": "Join projects, contracts, and companies"
        },
        {
            "query": "List active projects with active companies",
            "sql": "SELECT p.Project_Name, c.Company_Name FROM PAC_MNT_PROJECTS p JOIN PROJCNTRTS ct ON p.Project_Code = ct.Project_Code JOIN SRM_COMPANIES c ON ct.Company_Code = c.Company_Code WHERE p.Status = 'Active' AND c.Status = 'Active'",
            "reasoning": "Multi-table join with filters"
        }
    ])

    # ==========================================================================
    # MORE AGGREGATION VARIATIONS
    # ==========================================================================

    examples.extend([
        {
            "query": "Calculate total cost per project",
            "sql": "SELECT Project_Code, SUM(Bill_Rate * Allocation_Percent / 100) as total_cost FROM PROJSTAFF GROUP BY Project_Code",
            "reasoning": "Calculate costs in PROJSTAFF"
        },
        {
            "query": "Average cost rate by resource type",
            "sql": "SELECT Resource_Type, AVG(Cost_Rate) as avg_rate FROM PAC_MNT_RESOURCES GROUP BY Resource_Type",
            "reasoning": "Group PAC_MNT_RESOURCES and average"
        },
        {
            "query": "Count resources per project",
            "sql": "SELECT Project_Code, COUNT(DISTINCT Resource_Code) as resource_count FROM PROJSTAFF GROUP BY Project_Code",
            "reasoning": "Count distinct resources in PROJSTAFF"
        },
        {
            "query": "Total allocation per resource",
            "sql": "SELECT Resource_Code, SUM(Allocation_Percent) as total_allocation FROM PROJSTAFF GROUP BY Resource_Code",
            "reasoning": "Sum allocations in PROJSTAFF"
        },
        {
            "query": "Maximum contract value per company",
            "sql": "SELECT Company_Code, MAX(Contract_Value) as max_contract FROM PROJCNTRTS GROUP BY Company_Code",
            "reasoning": "Find max contract in PROJCNTRTS"
        },
        {
            "query": "Project count by year",
            "sql": "SELECT YEAR(Start_Date) as year, COUNT(*) as projects FROM PAC_MNT_PROJECTS GROUP BY YEAR(Start_Date)",
            "reasoning": "Group PAC_MNT_PROJECTS by year"
        }
    ])

    # ==========================================================================
    # MORE BUSINESS-SPECIFIC QUERIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Find overallocated resources",
            "sql": "SELECT Resource_Code, SUM(Allocation_Percent) as total FROM PROJSTAFF GROUP BY Resource_Code HAVING SUM(Allocation_Percent) > 100",
            "reasoning": "Find overallocated resources in PROJSTAFF"
        },
        {
            "query": "Show underutilized resources",
            "sql": "SELECT r.Resource_Name, COALESCE(SUM(s.Allocation_Percent), 0) as utilization FROM PAC_MNT_RESOURCES r LEFT JOIN PROJSTAFF s ON r.Resource_Code = s.Resource_Code GROUP BY r.Resource_Code, r.Resource_Name HAVING COALESCE(SUM(s.Allocation_Percent), 0) < 50",
            "reasoning": "Find resources with low utilization"
        },
        {
            "query": "Calculate project ROI",
            "sql": "SELECT Project_Name, ((Revenue - Actual_Cost) / Actual_Cost * 100) as ROI FROM PAC_MNT_PROJECTS WHERE Actual_Cost > 0",
            "reasoning": "Calculate ROI in PAC_MNT_PROJECTS"
        },
        {
            "query": "Find delayed projects",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE End_Date < CURRENT_DATE AND Status = 'Active'",
            "reasoning": "Find overdue active projects"
        },
        {
            "query": "Show budget variance",
            "sql": "SELECT Project_Name, (Budget - Actual_Cost) as variance FROM PAC_MNT_PROJECTS",
            "reasoning": "Calculate budget variance"
        },
        {
            "query": "List high-value customers",
            "sql": "SELECT c.Company_Name, SUM(ct.Contract_Value) as total_value FROM SRM_COMPANIES c JOIN PROJCNTRTS ct ON c.Company_Code = ct.Company_Code GROUP BY c.Company_Code, c.Company_Name ORDER BY total_value DESC",
            "reasoning": "Aggregate contract values by company"
        }
    ])

    # ==========================================================================
    # VARIATIONS OF COMMON QUERIES (Different phrasings)
    # ==========================================================================

    examples.extend([
        {
            "query": "What are all the active projects?",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Query PAC_MNT_PROJECTS for active status"
        },
        {
            "query": "Can you show me projects that are active?",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Filter PAC_MNT_PROJECTS by active status"
        },
        {
            "query": "I need to see all active project data",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Retrieve active projects from PAC_MNT_PROJECTS"
        },
        {
            "query": "Display projects where budget exceeds 100k",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > 100000",
            "reasoning": "Filter PAC_MNT_PROJECTS by budget threshold"
        },
        {
            "query": "Which projects have budgets larger than 100000?",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > 100000",
            "reasoning": "Query PAC_MNT_PROJECTS for high budget"
        },
        {
            "query": "Give me a list of all companies",
            "sql": "SELECT * FROM SRM_COMPANIES",
            "reasoning": "Retrieve all SRM_COMPANIES records"
        },
        {
            "query": "I want to see company information",
            "sql": "SELECT * FROM SRM_COMPANIES",
            "reasoning": "Query SRM_COMPANIES table"
        },
        {
            "query": "How many projects do we have?",
            "sql": "SELECT COUNT(*) as total_projects FROM PAC_MNT_PROJECTS",
            "reasoning": "Count all PAC_MNT_PROJECTS"
        },
        {
            "query": "What's the total number of projects?",
            "sql": "SELECT COUNT(*) as project_count FROM PAC_MNT_PROJECTS",
            "reasoning": "Count records in PAC_MNT_PROJECTS"
        }
    ])

    # ==========================================================================
    # MORE UPDATE/INSERT/DELETE PATTERNS (For completeness)
    # ==========================================================================

    examples.extend([
        {
            "query": "Update project status to completed",
            "sql": "UPDATE PAC_MNT_PROJECTS SET Status = 'Completed' WHERE Project_Code = 'PRJ001'",
            "reasoning": "Update Status in PAC_MNT_PROJECTS"
        },
        {
            "query": "Insert a new company",
            "sql": "INSERT INTO SRM_COMPANIES (Company_Code, Company_Name, Status) VALUES ('COMP001', 'New Company', 'Active')",
            "reasoning": "Insert new record into SRM_COMPANIES"
        },
        {
            "query": "Delete inactive projects",
            "sql": "DELETE FROM PAC_MNT_PROJECTS WHERE Status = 'Inactive'",
            "reasoning": "Remove inactive records from PAC_MNT_PROJECTS"
        }
    ])

    # ==========================================================================
    # WINDOW FUNCTIONS (If supported)
    # ==========================================================================

    examples.extend([
        {
            "query": "Rank projects by budget",
            "sql": "SELECT Project_Name, Budget, RANK() OVER (ORDER BY Budget DESC) as rank FROM PAC_MNT_PROJECTS",
            "reasoning": "Use window function to rank PAC_MNT_PROJECTS"
        },
        {
            "query": "Show running total of project budgets",
            "sql": "SELECT Project_Name, Budget, SUM(Budget) OVER (ORDER BY Start_Date) as running_total FROM PAC_MNT_PROJECTS",
            "reasoning": "Running sum using window function"
        }
    ])

    # ==========================================================================
    # EXISTS QUERIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Find projects that have staff assigned",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS p WHERE EXISTS (SELECT 1 FROM PROJSTAFF s WHERE s.Project_Code = p.Project_Code)",
            "reasoning": "Use EXISTS to check PROJSTAFF"
        },
        {
            "query": "Companies without contracts",
            "sql": "SELECT * FROM SRM_COMPANIES c WHERE NOT EXISTS (SELECT 1 FROM PROJCNTRTS ct WHERE ct.Company_Code = c.Company_Code)",
            "reasoning": "Use NOT EXISTS with PROJCNTRTS"
        }
    ])

    # ==========================================================================
    # COALESCE AND NULL HANDLING
    # ==========================================================================

    examples.extend([
        {
            "query": "Show projects with budget or zero",
            "sql": "SELECT Project_Name, COALESCE(Budget, 0) as budget FROM PAC_MNT_PROJECTS",
            "reasoning": "Use COALESCE for null handling in PAC_MNT_PROJECTS"
        },
        {
            "query": "Get resources with email or 'No Email'",
            "sql": "SELECT Resource_Name, COALESCE(Email, 'No Email') as email FROM PAC_MNT_RESOURCES",
            "reasoning": "Handle nulls in PAC_MNT_RESOURCES.Email"
        }
    ])

    return examples

def main():
    # Load existing data
    existing_file = Path("gl_rl_model/data/training/query_pairs_expanded.jsonl")
    existing_examples = []

    if existing_file.exists():
        with open(existing_file, 'r') as f:
            for line in f:
                existing_examples.append(json.loads(line))

    print(f"📚 Loaded {len(existing_examples)} existing examples")

    # Generate additional examples
    new_examples = generate_additional_examples()
    print(f"✨ Generated {len(new_examples)} new examples")

    # Combine all examples
    all_examples = existing_examples + new_examples
    print(f"📊 Total examples: {len(all_examples)}")

    # Save combined dataset
    with open(existing_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Saved expanded dataset to: {existing_file}")

    # Print statistics
    tables = ["PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJSTAFF", "PROJCNTRTS", "PAC_MNT_RESOURCES", "SRM_CONTACTS"]
    print("\n📈 Table Coverage:")
    for table in tables:
        count = sum(1 for ex in all_examples if table in ex['sql'])
        print(f"  {table}: {count} queries")

    # Ensure every example uses domain tables
    generic_count = sum(1 for ex in all_examples if not any(table in ex['sql'] for table in tables))
    print(f"\n⚠️ Generic queries (no domain tables): {generic_count}")

    if generic_count > 0:
        print("Warning: Some queries don't use domain tables!")

if __name__ == "__main__":
    main()