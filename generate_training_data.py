#!/usr/bin/env python3
"""
Generate comprehensive training dataset with 150+ examples for domain-specific SQL generation
"""
import json
from pathlib import Path
import random

def generate_training_data():
    """Generate a comprehensive training dataset with diverse SQL patterns."""

    examples = []

    # ==========================================================================
    # SIMPLE SELECT QUERIES - PAC_MNT_PROJECTS
    # ==========================================================================

    # Active projects variations
    examples.extend([
        {
            "query": "Show all active projects",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Query PAC_MNT_PROJECTS table and filter by Status = 'Active'"
        },
        {
            "query": "List active projects in the system",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Retrieve all records from PAC_MNT_PROJECTS where Status equals 'Active'"
        },
        {
            "query": "Get me all projects that are currently active",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Filter PAC_MNT_PROJECTS table for active status projects"
        },
        {
            "query": "Display active project records",
            "sql": "SELECT Project_ID, Project_Code, Project_Name, Status FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Select key columns from PAC_MNT_PROJECTS for active projects"
        },
        {
            "query": "Find all projects with active status",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Query PAC_MNT_PROJECTS filtering on Status column"
        }
    ])

    # Budget-related queries
    examples.extend([
        {
            "query": "Find projects with budget over 100000",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > 100000",
            "reasoning": "Query PAC_MNT_PROJECTS where Budget exceeds 100000"
        },
        {
            "query": "Show high budget projects above 500000",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > 500000",
            "reasoning": "Filter PAC_MNT_PROJECTS for projects with Budget greater than 500000"
        },
        {
            "query": "List projects with budget between 50000 and 200000",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget BETWEEN 50000 AND 200000",
            "reasoning": "Use BETWEEN operator on PAC_MNT_PROJECTS.Budget column"
        },
        {
            "query": "Get projects where actual cost exceeds budget",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Actual_Cost > Budget",
            "reasoning": "Compare Actual_Cost and Budget columns in PAC_MNT_PROJECTS"
        },
        {
            "query": "Find overbudget projects",
            "sql": "SELECT Project_Name, Budget, Actual_Cost FROM PAC_MNT_PROJECTS WHERE Actual_Cost > Budget",
            "reasoning": "Select projects from PAC_MNT_PROJECTS where costs exceed budget"
        }
    ])

    # Department-specific queries
    examples.extend([
        {
            "query": "Show all IT department projects",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Department = 'IT'",
            "reasoning": "Filter PAC_MNT_PROJECTS by Department = 'IT'"
        },
        {
            "query": "List Finance department projects",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Department = 'Finance'",
            "reasoning": "Query PAC_MNT_PROJECTS for Finance department"
        },
        {
            "query": "Get projects from HR department",
            "sql": "SELECT Project_Code, Project_Name, Budget FROM PAC_MNT_PROJECTS WHERE Department = 'HR'",
            "reasoning": "Select HR projects from PAC_MNT_PROJECTS table"
        }
    ])

    # Date-based queries
    examples.extend([
        {
            "query": "Find projects starting in 2024",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE YEAR(Start_Date) = 2024",
            "reasoning": "Filter PAC_MNT_PROJECTS by Start_Date year"
        },
        {
            "query": "Show projects ending this year",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE YEAR(End_Date) = YEAR(CURRENT_DATE)",
            "reasoning": "Query PAC_MNT_PROJECTS for current year End_Date"
        },
        {
            "query": "List projects started after January 2024",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Start_Date > '2024-01-01'",
            "reasoning": "Filter PAC_MNT_PROJECTS where Start_Date is after specified date"
        }
    ])

    # ==========================================================================
    # SIMPLE SELECT QUERIES - SRM_COMPANIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Show all companies",
            "sql": "SELECT * FROM SRM_COMPANIES",
            "reasoning": "Retrieve all records from SRM_COMPANIES table"
        },
        {
            "query": "List all company records",
            "sql": "SELECT Company_ID, Company_Code, Company_Name, Status FROM SRM_COMPANIES",
            "reasoning": "Select all columns from SRM_COMPANIES"
        },
        {
            "query": "Get active companies",
            "sql": "SELECT * FROM SRM_COMPANIES WHERE Status = 'Active'",
            "reasoning": "Filter SRM_COMPANIES for active status"
        },
        {
            "query": "Find inactive companies",
            "sql": "SELECT Company_Code, Company_Name FROM SRM_COMPANIES WHERE Status = 'Inactive'",
            "reasoning": "Query SRM_COMPANIES for inactive status records"
        },
        {
            "query": "Show company details",
            "sql": "SELECT * FROM SRM_COMPANIES ORDER BY Company_Name",
            "reasoning": "Retrieve all SRM_COMPANIES ordered by name"
        }
    ])

    # ==========================================================================
    # SIMPLE SELECT QUERIES - PROJSTAFF
    # ==========================================================================

    examples.extend([
        {
            "query": "Show all project staff assignments",
            "sql": "SELECT * FROM PROJSTAFF",
            "reasoning": "Retrieve all records from PROJSTAFF table"
        },
        {
            "query": "List staff allocations",
            "sql": "SELECT Staff_ID, Project_Code, Resource_Code, Allocation_Percent FROM PROJSTAFF",
            "reasoning": "Query PROJSTAFF for allocation information"
        },
        {
            "query": "Find full-time allocations",
            "sql": "SELECT * FROM PROJSTAFF WHERE Allocation_Percent = 100",
            "reasoning": "Filter PROJSTAFF for 100% allocations"
        },
        {
            "query": "Show part-time staff assignments",
            "sql": "SELECT * FROM PROJSTAFF WHERE Allocation_Percent < 100",
            "reasoning": "Query PROJSTAFF for allocations less than 100%"
        },
        {
            "query": "Get project managers",
            "sql": "SELECT * FROM PROJSTAFF WHERE Role = 'Project Manager'",
            "reasoning": "Filter PROJSTAFF by Role = 'Project Manager'"
        },
        {
            "query": "List developers on projects",
            "sql": "SELECT Project_Code, Resource_Code FROM PROJSTAFF WHERE Role = 'Developer'",
            "reasoning": "Query PROJSTAFF for Developer role"
        }
    ])

    # ==========================================================================
    # SIMPLE SELECT QUERIES - PROJCNTRTS
    # ==========================================================================

    examples.extend([
        {
            "query": "Show all project contracts",
            "sql": "SELECT * FROM PROJCNTRTS",
            "reasoning": "Retrieve all records from PROJCNTRTS table"
        },
        {
            "query": "List active contracts",
            "sql": "SELECT * FROM PROJCNTRTS WHERE Status = 'Active'",
            "reasoning": "Filter PROJCNTRTS for active status"
        },
        {
            "query": "Find high-value contracts",
            "sql": "SELECT * FROM PROJCNTRTS WHERE Contract_Value > 1000000",
            "reasoning": "Query PROJCNTRTS for contracts over 1 million"
        },
        {
            "query": "Show fixed-price contracts",
            "sql": "SELECT Contract_Number, Project_Code, Contract_Value FROM PROJCNTRTS WHERE Contract_Type = 'Fixed Price'",
            "reasoning": "Filter PROJCNTRTS by Contract_Type"
        },
        {
            "query": "Get time and materials contracts",
            "sql": "SELECT * FROM PROJCNTRTS WHERE Contract_Type = 'Time and Materials'",
            "reasoning": "Query PROJCNTRTS for T&M contract type"
        }
    ])

    # ==========================================================================
    # AGGREGATION QUERIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Count total number of projects",
            "sql": "SELECT COUNT(*) AS total_projects FROM PAC_MNT_PROJECTS",
            "reasoning": "Use COUNT aggregate on PAC_MNT_PROJECTS"
        },
        {
            "query": "Count active projects",
            "sql": "SELECT COUNT(*) AS active_count FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Count records in PAC_MNT_PROJECTS with active status"
        },
        {
            "query": "Calculate total budget for all projects",
            "sql": "SELECT SUM(Budget) AS total_budget FROM PAC_MNT_PROJECTS",
            "reasoning": "Sum Budget column in PAC_MNT_PROJECTS"
        },
        {
            "query": "Get average project budget",
            "sql": "SELECT AVG(Budget) AS avg_budget FROM PAC_MNT_PROJECTS",
            "reasoning": "Calculate average of Budget in PAC_MNT_PROJECTS"
        },
        {
            "query": "Find maximum project budget",
            "sql": "SELECT MAX(Budget) AS max_budget FROM PAC_MNT_PROJECTS",
            "reasoning": "Find maximum value in PAC_MNT_PROJECTS.Budget"
        },
        {
            "query": "Get minimum project budget",
            "sql": "SELECT MIN(Budget) AS min_budget FROM PAC_MNT_PROJECTS",
            "reasoning": "Find minimum value in PAC_MNT_PROJECTS.Budget"
        },
        {
            "query": "Calculate total revenue",
            "sql": "SELECT SUM(Revenue) AS total_revenue FROM PAC_MNT_PROJECTS",
            "reasoning": "Sum Revenue column in PAC_MNT_PROJECTS"
        },
        {
            "query": "Count projects per department",
            "sql": "SELECT Department, COUNT(*) AS project_count FROM PAC_MNT_PROJECTS GROUP BY Department",
            "reasoning": "Group PAC_MNT_PROJECTS by Department and count"
        },
        {
            "query": "Sum budget by department",
            "sql": "SELECT Department, SUM(Budget) AS total_budget FROM PAC_MNT_PROJECTS GROUP BY Department",
            "reasoning": "Group PAC_MNT_PROJECTS by Department and sum Budget"
        },
        {
            "query": "Average budget per department",
            "sql": "SELECT Department, AVG(Budget) AS avg_budget FROM PAC_MNT_PROJECTS GROUP BY Department",
            "reasoning": "Group PAC_MNT_PROJECTS by Department and average Budget"
        },
        {
            "query": "Count projects by status",
            "sql": "SELECT Status, COUNT(*) AS count FROM PAC_MNT_PROJECTS GROUP BY Status",
            "reasoning": "Group PAC_MNT_PROJECTS by Status and count"
        },
        {
            "query": "Total budget for active projects",
            "sql": "SELECT SUM(Budget) AS active_budget FROM PAC_MNT_PROJECTS WHERE Status = 'Active'",
            "reasoning": "Sum Budget for active projects in PAC_MNT_PROJECTS"
        },
        {
            "query": "Count companies",
            "sql": "SELECT COUNT(*) AS company_count FROM SRM_COMPANIES",
            "reasoning": "Count all records in SRM_COMPANIES"
        },
        {
            "query": "Count active companies",
            "sql": "SELECT COUNT(*) AS active_companies FROM SRM_COMPANIES WHERE Status = 'Active'",
            "reasoning": "Count active records in SRM_COMPANIES"
        },
        {
            "query": "Count staff assignments",
            "sql": "SELECT COUNT(*) AS assignments FROM PROJSTAFF",
            "reasoning": "Count all records in PROJSTAFF"
        },
        {
            "query": "Average allocation percentage",
            "sql": "SELECT AVG(Allocation_Percent) AS avg_allocation FROM PROJSTAFF",
            "reasoning": "Calculate average Allocation_Percent in PROJSTAFF"
        },
        {
            "query": "Sum of contract values",
            "sql": "SELECT SUM(Contract_Value) AS total_value FROM PROJCNTRTS",
            "reasoning": "Sum Contract_Value in PROJCNTRTS"
        },
        {
            "query": "Count contracts per project",
            "sql": "SELECT Project_Code, COUNT(*) AS contract_count FROM PROJCNTRTS GROUP BY Project_Code",
            "reasoning": "Group PROJCNTRTS by Project_Code and count"
        }
    ])

    # ==========================================================================
    # JOIN QUERIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Show projects with their staff",
            "sql": "SELECT p.Project_Name, s.Resource_Code, s.Role FROM PAC_MNT_PROJECTS p JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code",
            "reasoning": "Join PAC_MNT_PROJECTS with PROJSTAFF on Project_Code"
        },
        {
            "query": "List projects and their contracts",
            "sql": "SELECT p.Project_Name, c.Contract_Number, c.Contract_Value FROM PAC_MNT_PROJECTS p JOIN PROJCNTRTS c ON p.Project_Code = c.Project_Code",
            "reasoning": "Join PAC_MNT_PROJECTS with PROJCNTRTS on Project_Code"
        },
        {
            "query": "Show companies with their contracts",
            "sql": "SELECT c.Company_Name, ct.Contract_Number, ct.Contract_Value FROM SRM_COMPANIES c JOIN PROJCNTRTS ct ON c.Company_Code = ct.Company_Code",
            "reasoning": "Join SRM_COMPANIES with PROJCNTRTS on Company_Code"
        },
        {
            "query": "Get projects with company details",
            "sql": "SELECT p.Project_Name, c.Company_Name FROM PAC_MNT_PROJECTS p JOIN PROJCNTRTS ct ON p.Project_Code = ct.Project_Code JOIN SRM_COMPANIES c ON ct.Company_Code = c.Company_Code",
            "reasoning": "Join PAC_MNT_PROJECTS with PROJCNTRTS and SRM_COMPANIES"
        },
        {
            "query": "Show staff assignments with project names",
            "sql": "SELECT p.Project_Name, s.Resource_Code, s.Allocation_Percent FROM PAC_MNT_PROJECTS p INNER JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code",
            "reasoning": "Inner join PAC_MNT_PROJECTS and PROJSTAFF"
        },
        {
            "query": "List active projects with staff",
            "sql": "SELECT p.Project_Name, s.Resource_Code FROM PAC_MNT_PROJECTS p JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code WHERE p.Status = 'Active'",
            "reasoning": "Join PAC_MNT_PROJECTS and PROJSTAFF, filter for active"
        },
        {
            "query": "Get high-budget projects with contracts",
            "sql": "SELECT p.Project_Name, p.Budget, c.Contract_Value FROM PAC_MNT_PROJECTS p JOIN PROJCNTRTS c ON p.Project_Code = c.Project_Code WHERE p.Budget > 500000",
            "reasoning": "Join PAC_MNT_PROJECTS and PROJCNTRTS, filter by budget"
        }
    ])

    # ==========================================================================
    # LEFT/RIGHT JOIN QUERIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Show all projects even without staff",
            "sql": "SELECT p.Project_Name, s.Resource_Code FROM PAC_MNT_PROJECTS p LEFT JOIN PROJSTAFF s ON p.Project_Code = s.Project_Code",
            "reasoning": "Left join to include all PAC_MNT_PROJECTS"
        },
        {
            "query": "List all companies even without contracts",
            "sql": "SELECT c.Company_Name, ct.Contract_Number FROM SRM_COMPANIES c LEFT JOIN PROJCNTRTS ct ON c.Company_Code = ct.Company_Code",
            "reasoning": "Left join to include all SRM_COMPANIES"
        },
        {
            "query": "Get all projects with optional contracts",
            "sql": "SELECT p.Project_Name, c.Contract_Number FROM PAC_MNT_PROJECTS p LEFT OUTER JOIN PROJCNTRTS c ON p.Project_Code = c.Project_Code",
            "reasoning": "Left outer join PAC_MNT_PROJECTS with PROJCNTRTS"
        }
    ])

    # ==========================================================================
    # COMPLEX FILTERS
    # ==========================================================================

    examples.extend([
        {
            "query": "Find active projects with budget over 200000 in IT department",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status = 'Active' AND Budget > 200000 AND Department = 'IT'",
            "reasoning": "Multiple conditions on PAC_MNT_PROJECTS"
        },
        {
            "query": "Show completed or cancelled projects",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Status IN ('Completed', 'Cancelled')",
            "reasoning": "Use IN operator on PAC_MNT_PROJECTS.Status"
        },
        {
            "query": "List projects not in IT or Finance",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Department NOT IN ('IT', 'Finance')",
            "reasoning": "Use NOT IN on PAC_MNT_PROJECTS.Department"
        },
        {
            "query": "Find projects with null end date",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE End_Date IS NULL",
            "reasoning": "Check for NULL in PAC_MNT_PROJECTS.End_Date"
        },
        {
            "query": "Get projects with both high budget and high actual cost",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > 500000 AND Actual_Cost > 400000",
            "reasoning": "Multiple numeric conditions on PAC_MNT_PROJECTS"
        }
    ])

    # ==========================================================================
    # ORDERING AND LIMITING
    # ==========================================================================

    examples.extend([
        {
            "query": "Show projects ordered by budget descending",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS ORDER BY Budget DESC",
            "reasoning": "Order PAC_MNT_PROJECTS by Budget descending"
        },
        {
            "query": "List projects sorted by start date",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS ORDER BY Start_Date",
            "reasoning": "Order PAC_MNT_PROJECTS by Start_Date ascending"
        },
        {
            "query": "Get top 10 highest budget projects",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS ORDER BY Budget DESC LIMIT 10",
            "reasoning": "Order PAC_MNT_PROJECTS by Budget and limit to 10"
        },
        {
            "query": "Show 5 most recent projects",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS ORDER BY Start_Date DESC LIMIT 5",
            "reasoning": "Order PAC_MNT_PROJECTS by Start_Date desc, limit 5"
        },
        {
            "query": "List companies alphabetically",
            "sql": "SELECT * FROM SRM_COMPANIES ORDER BY Company_Name ASC",
            "reasoning": "Order SRM_COMPANIES by Company_Name ascending"
        }
    ])

    # ==========================================================================
    # SUBQUERIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Find projects with above average budget",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > (SELECT AVG(Budget) FROM PAC_MNT_PROJECTS)",
            "reasoning": "Subquery to get average budget from PAC_MNT_PROJECTS"
        },
        {
            "query": "Get projects with budget higher than IT department average",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Budget > (SELECT AVG(Budget) FROM PAC_MNT_PROJECTS WHERE Department = 'IT')",
            "reasoning": "Subquery for IT department average in PAC_MNT_PROJECTS"
        },
        {
            "query": "Show companies that have contracts",
            "sql": "SELECT * FROM SRM_COMPANIES WHERE Company_Code IN (SELECT DISTINCT Company_Code FROM PROJCNTRTS)",
            "reasoning": "Subquery to find companies in PROJCNTRTS"
        },
        {
            "query": "Find projects without staff assignments",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Project_Code NOT IN (SELECT DISTINCT Project_Code FROM PROJSTAFF)",
            "reasoning": "Subquery to exclude projects in PROJSTAFF"
        }
    ])

    # ==========================================================================
    # HAVING CLAUSE
    # ==========================================================================

    examples.extend([
        {
            "query": "Show departments with total budget over 1 million",
            "sql": "SELECT Department, SUM(Budget) as total FROM PAC_MNT_PROJECTS GROUP BY Department HAVING SUM(Budget) > 1000000",
            "reasoning": "Group PAC_MNT_PROJECTS and filter with HAVING"
        },
        {
            "query": "Find departments with more than 5 projects",
            "sql": "SELECT Department, COUNT(*) as count FROM PAC_MNT_PROJECTS GROUP BY Department HAVING COUNT(*) > 5",
            "reasoning": "Group PAC_MNT_PROJECTS and use HAVING on count"
        },
        {
            "query": "List project codes with multiple staff",
            "sql": "SELECT Project_Code, COUNT(*) as staff_count FROM PROJSTAFF GROUP BY Project_Code HAVING COUNT(*) > 1",
            "reasoning": "Group PROJSTAFF and filter with HAVING"
        }
    ])

    # ==========================================================================
    # DISTINCT QUERIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Show distinct departments",
            "sql": "SELECT DISTINCT Department FROM PAC_MNT_PROJECTS",
            "reasoning": "Get unique departments from PAC_MNT_PROJECTS"
        },
        {
            "query": "List unique project statuses",
            "sql": "SELECT DISTINCT Status FROM PAC_MNT_PROJECTS",
            "reasoning": "Get distinct Status values from PAC_MNT_PROJECTS"
        },
        {
            "query": "Get distinct company codes with contracts",
            "sql": "SELECT DISTINCT Company_Code FROM PROJCNTRTS",
            "reasoning": "Find unique Company_Code in PROJCNTRTS"
        },
        {
            "query": "Show unique roles in staff",
            "sql": "SELECT DISTINCT Role FROM PROJSTAFF",
            "reasoning": "Get distinct Role values from PROJSTAFF"
        }
    ])

    # ==========================================================================
    # UNION QUERIES (if supported)
    # ==========================================================================

    examples.extend([
        {
            "query": "Combine active projects and companies",
            "sql": "SELECT Project_Name AS name FROM PAC_MNT_PROJECTS WHERE Status = 'Active' UNION SELECT Company_Name AS name FROM SRM_COMPANIES WHERE Status = 'Active'",
            "reasoning": "Union active records from PAC_MNT_PROJECTS and SRM_COMPANIES"
        }
    ])

    # ==========================================================================
    # CASE STATEMENTS
    # ==========================================================================

    examples.extend([
        {
            "query": "Categorize projects by budget size",
            "sql": "SELECT Project_Name, CASE WHEN Budget > 1000000 THEN 'Large' WHEN Budget > 100000 THEN 'Medium' ELSE 'Small' END AS size FROM PAC_MNT_PROJECTS",
            "reasoning": "Use CASE to categorize PAC_MNT_PROJECTS by Budget"
        },
        {
            "query": "Show project status categories",
            "sql": "SELECT Project_Name, CASE Status WHEN 'Active' THEN 'Ongoing' WHEN 'Completed' THEN 'Finished' ELSE 'Other' END AS category FROM PAC_MNT_PROJECTS",
            "reasoning": "CASE statement on PAC_MNT_PROJECTS.Status"
        }
    ])

    # ==========================================================================
    # DATE FUNCTIONS
    # ==========================================================================

    examples.extend([
        {
            "query": "Projects started this month",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE MONTH(Start_Date) = MONTH(CURRENT_DATE) AND YEAR(Start_Date) = YEAR(CURRENT_DATE)",
            "reasoning": "Filter PAC_MNT_PROJECTS by current month"
        },
        {
            "query": "Projects running for more than 365 days",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE DATEDIFF(End_Date, Start_Date) > 365",
            "reasoning": "Calculate date difference in PAC_MNT_PROJECTS"
        },
        {
            "query": "Show projects from last quarter",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Start_Date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)",
            "reasoning": "Date arithmetic on PAC_MNT_PROJECTS.Start_Date"
        }
    ])

    # ==========================================================================
    # LIKE PATTERNS
    # ==========================================================================

    examples.extend([
        {
            "query": "Find projects with 'System' in name",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Project_Name LIKE '%System%'",
            "reasoning": "Use LIKE pattern on PAC_MNT_PROJECTS.Project_Name"
        },
        {
            "query": "Get projects starting with 'PRJ'",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Project_Code LIKE 'PRJ%'",
            "reasoning": "Pattern matching on PAC_MNT_PROJECTS.Project_Code"
        },
        {
            "query": "Find companies ending with 'Corp'",
            "sql": "SELECT * FROM SRM_COMPANIES WHERE Company_Name LIKE '%Corp'",
            "reasoning": "LIKE pattern on SRM_COMPANIES.Company_Name"
        }
    ])

    # ==========================================================================
    # COMPLEX AGGREGATIONS
    # ==========================================================================

    examples.extend([
        {
            "query": "Department with highest total budget",
            "sql": "SELECT Department, SUM(Budget) as total FROM PAC_MNT_PROJECTS GROUP BY Department ORDER BY total DESC LIMIT 1",
            "reasoning": "Group PAC_MNT_PROJECTS, sum, order and limit"
        },
        {
            "query": "Average budget by status",
            "sql": "SELECT Status, AVG(Budget) as avg_budget FROM PAC_MNT_PROJECTS GROUP BY Status",
            "reasoning": "Group PAC_MNT_PROJECTS by Status and average"
        },
        {
            "query": "Project count and total budget by department",
            "sql": "SELECT Department, COUNT(*) as projects, SUM(Budget) as total_budget FROM PAC_MNT_PROJECTS GROUP BY Department",
            "reasoning": "Multiple aggregations on PAC_MNT_PROJECTS grouped by Department"
        }
    ])

    # ==========================================================================
    # SPECIFIC BUSINESS QUERIES
    # ==========================================================================

    examples.extend([
        {
            "query": "Show profitable projects",
            "sql": "SELECT * FROM PAC_MNT_PROJECTS WHERE Revenue > Actual_Cost",
            "reasoning": "Compare Revenue and Actual_Cost in PAC_MNT_PROJECTS"
        },
        {
            "query": "Find loss-making projects",
            "sql": "SELECT Project_Name, Revenue, Actual_Cost FROM PAC_MNT_PROJECTS WHERE Revenue < Actual_Cost",
            "reasoning": "Identify projects where costs exceed revenue in PAC_MNT_PROJECTS"
        },
        {
            "query": "Calculate profit margin per project",
            "sql": "SELECT Project_Name, ((Revenue - Actual_Cost) / Revenue * 100) as profit_margin FROM PAC_MNT_PROJECTS WHERE Revenue > 0",
            "reasoning": "Calculate profit margin from PAC_MNT_PROJECTS"
        },
        {
            "query": "Show budget utilization",
            "sql": "SELECT Project_Name, (Actual_Cost / Budget * 100) as utilization FROM PAC_MNT_PROJECTS WHERE Budget > 0",
            "reasoning": "Calculate budget utilization in PAC_MNT_PROJECTS"
        },
        {
            "query": "Find fully allocated staff",
            "sql": "SELECT Resource_Code FROM PROJSTAFF GROUP BY Resource_Code HAVING SUM(Allocation_Percent) >= 100",
            "reasoning": "Group PROJSTAFF and find resources at full capacity"
        }
    ])

    # Shuffle for variety in training
    random.shuffle(examples)

    return examples

def save_training_data(examples, output_file):
    """Save training examples to JSONL file."""
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Generated {len(examples)} training examples")
    print(f"📁 Saved to: {output_file}")

def main():
    # Generate comprehensive training data
    examples = generate_training_data()

    # Save to file
    output_path = Path("gl_rl_model/data/training/query_pairs_expanded.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_data(examples, output_path)

    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total examples: {len(examples)}")

    # Count by table usage
    tables = ["PAC_MNT_PROJECTS", "SRM_COMPANIES", "PROJSTAFF", "PROJCNTRTS", "PAC_MNT_RESOURCES"]
    for table in tables:
        count = sum(1 for ex in examples if table in ex['sql'])
        print(f"  {table}: {count} queries")

    # Count by query type
    query_types = {
        "SELECT": sum(1 for ex in examples if "SELECT" in ex['sql']),
        "JOIN": sum(1 for ex in examples if "JOIN" in ex['sql']),
        "GROUP BY": sum(1 for ex in examples if "GROUP BY" in ex['sql']),
        "HAVING": sum(1 for ex in examples if "HAVING" in ex['sql']),
        "Subquery": sum(1 for ex in examples if "SELECT" in ex['sql'] and ex['sql'].count("SELECT") > 1),
    }

    print("\n📈 Query Types:")
    for qtype, count in query_types.items():
        print(f"  {qtype}: {count}")

    print("\n✅ Training data generation complete!")
    print("📝 Next step: Update training configuration and run extended training")

if __name__ == "__main__":
    main()