-- GL RL Model Database Schema
-- Based on Project/Resource Management ERD
-- Generated from ERD analysis

-- =============================================
-- Core Business Tables
-- =============================================

-- Company master data
CREATE TABLE SRM_COMPANIES (
    Company_ID INTEGER PRIMARY KEY,
    Company_Name VARCHAR(255) NOT NULL,
    Company_Code VARCHAR(50) UNIQUE NOT NULL,
    Principal_ID INTEGER,
    Industry VARCHAR(100),
    Status VARCHAR(20) DEFAULT 'Active',
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Project master data
CREATE TABLE SRM_PROJECTS (
    Project_ID INTEGER PRIMARY KEY,
    Project_Name VARCHAR(255) NOT NULL,
    Company_ID INTEGER,
    Parent_ID INTEGER,  -- For project hierarchy
    Start_Date DATE,
    End_Date DATE,
    Status VARCHAR(50) DEFAULT 'Planning',
    Description TEXT,
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Company_ID) REFERENCES SRM_COMPANIES(Company_ID),
    FOREIGN KEY (Parent_ID) REFERENCES SRM_PROJECTS(Project_ID)
);

-- Contact information
CREATE TABLE SRM_CONTACTS (
    Contact_ID INTEGER PRIMARY KEY,
    Company_ID INTEGER,
    Contact_Name VARCHAR(255) NOT NULL,
    Email VARCHAR(255),
    Phone VARCHAR(50),
    Role VARCHAR(100),
    Is_Primary BOOLEAN DEFAULT FALSE,
    Status VARCHAR(20) DEFAULT 'Active',
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Company_ID) REFERENCES SRM_COMPANIES(Company_ID)
);

-- =============================================
-- Project Accounting and Management
-- =============================================

-- Project accounting and maintenance
CREATE TABLE PAC_MNT_PROJECTS (
    Project_Code VARCHAR(50) PRIMARY KEY,
    Project_Name VARCHAR(255) NOT NULL,
    Budget DECIMAL(15,2),
    Actual_Cost DECIMAL(15,2) DEFAULT 0,
    Revenue DECIMAL(15,2) DEFAULT 0,
    Status VARCHAR(50) DEFAULT 'Active',
    Cost_Center VARCHAR(50),
    Department VARCHAR(100),
    Start_Date DATE,
    End_Date DATE,
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resource management
CREATE TABLE PAC_MNT_RESOURCES (
    Resource_Code VARCHAR(50) PRIMARY KEY,
    Resource_Name VARCHAR(255) NOT NULL,
    Resource_Type VARCHAR(50), -- 'Human', 'Material', 'Equipment'
    Cost_Rate DECIMAL(10,2),
    Billing_Rate DECIMAL(10,2),
    Currency VARCHAR(3) DEFAULT 'USD',
    Availability VARCHAR(50) DEFAULT 'Available',
    Capacity DECIMAL(5,2) DEFAULT 100.00, -- Percentage
    Department VARCHAR(100),
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- Client and Support Tables
-- =============================================

-- Client classifications
CREATE TABLE CLNTCLASS (
    ClntClass_ID INTEGER PRIMARY KEY,
    Class_Code VARCHAR(50) UNIQUE NOT NULL,
    Class_Name VARCHAR(255) NOT NULL,
    Description TEXT,
    Priority INTEGER DEFAULT 5,
    Status VARCHAR(20) DEFAULT 'Active',
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Client support tickets
CREATE TABLE CLNTSUPP (
    Support_ID INTEGER PRIMARY KEY,
    Company_Code VARCHAR(50),
    ClntClass VARCHAR(50),
    Support_Type VARCHAR(100),
    Priority VARCHAR(20) DEFAULT 'Medium',
    Status VARCHAR(50) DEFAULT 'Open',
    Reported_Date DATE,
    Resolved_Date DATE,
    Description TEXT,
    Resolution TEXT,
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Company_Code) REFERENCES SRM_COMPANIES(Company_Code),
    FOREIGN KEY (ClntClass) REFERENCES CLNTCLASS(Class_Code)
);

-- Client responses
CREATE TABLE CLNTRESPONS (
    Response_ID INTEGER PRIMARY KEY,
    Project_Code VARCHAR(50),
    Unique_Name VARCHAR(100),
    Response_Date DATE,
    Response_Type VARCHAR(100),
    Status VARCHAR(50) DEFAULT 'Pending',
    Response_Content TEXT,
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Project_Code) REFERENCES PAC_MNT_PROJECTS(Project_Code)
);

-- =============================================
-- Project Management Tables
-- =============================================

-- Project contracts
CREATE TABLE PROJCNTRTS (
    Contract_ID INTEGER PRIMARY KEY,
    Project_Code VARCHAR(50),
    Company_Code VARCHAR(50),
    Contract_Number VARCHAR(100) UNIQUE,
    Contract_Type VARCHAR(50), -- 'Fixed Price', 'Time & Material', 'Retainer'
    Contract_Value DECIMAL(15,2),
    Start_Date DATE,
    End_Date DATE,
    Status VARCHAR(50) DEFAULT 'Draft',
    Terms TEXT,
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Project_Code) REFERENCES PAC_MNT_PROJECTS(Project_Code),
    FOREIGN KEY (Company_Code) REFERENCES SRM_COMPANIES(Company_Code)
);

-- Project staffing
CREATE TABLE PROJSTAFF (
    Staff_ID INTEGER PRIMARY KEY,
    Project_Code VARCHAR(50),
    Resource_Code VARCHAR(50),
    Role VARCHAR(100),
    Allocation_Percent DECIMAL(5,2) DEFAULT 100.00,
    Start_Date DATE,
    End_Date DATE,
    Status VARCHAR(50) DEFAULT 'Assigned',
    Bill_Rate DECIMAL(10,2),
    Cost_Rate DECIMAL(10,2),
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Project_Code) REFERENCES PAC_MNT_PROJECTS(Project_Code),
    FOREIGN KEY (Resource_Code) REFERENCES PAC_MNT_RESOURCES(Resource_Code)
);

-- Project revisions/versions
CREATE TABLE PROJEVISION (
    Revision_ID INTEGER PRIMARY KEY,
    Project_Code VARCHAR(50),
    Revision_Number VARCHAR(20),
    Revision_Date DATE,
    Revision_Type VARCHAR(100), -- 'Scope Change', 'Budget Update', 'Timeline Adjustment'
    Previous_Value TEXT,
    New_Value TEXT,
    Description TEXT,
    Approved_By VARCHAR(255),
    Approval_Date DATE,
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Project_Code) REFERENCES PAC_MNT_PROJECTS(Project_Code)
);

-- =============================================
-- Additional Business Tables
-- =============================================

-- Business opportunities
CREATE TABLE BIZ_OPP_GEN_PROPERTIES (
    Opportunity_ID INTEGER PRIMARY KEY,
    Company_ID INTEGER,
    Opportunity_Name VARCHAR(255),
    Estimated_Value DECIMAL(15,2),
    Probability DECIMAL(5,2),
    Stage VARCHAR(50), -- 'Lead', 'Qualified', 'Proposal', 'Negotiation', 'Closed'
    Expected_Close_Date DATE,
    Status VARCHAR(50) DEFAULT 'Active',
    Owner VARCHAR(255),
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Company_ID) REFERENCES SRM_COMPANIES(Company_ID)
);

-- Company supplier properties
CREATE TABLE BIZ_COM_SUP_PROPERTIES (
    Supplier_ID INTEGER PRIMARY KEY,
    Company_ID INTEGER,
    Supplier_Type VARCHAR(100),
    Payment_Terms VARCHAR(100),
    Credit_Limit DECIMAL(15,2),
    Rating VARCHAR(20),
    Status VARCHAR(50) DEFAULT 'Active',
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Modified_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Company_ID) REFERENCES SRM_COMPANIES(Company_ID)
);

-- Financial purge list (for archiving)
CREATE TABLE PAC_PURGE_FINANCIAL_LIST (
    Purge_ID INTEGER PRIMARY KEY,
    Table_Name VARCHAR(100),
    Record_Type VARCHAR(50),
    Cutoff_Date DATE,
    Records_Count INTEGER,
    Purge_Status VARCHAR(50) DEFAULT 'Pending',
    Purge_Date DATE,
    Archived_Location VARCHAR(500),
    Created_Date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- Indexes for Performance
-- =============================================

-- Company indexes
CREATE INDEX idx_company_code ON SRM_COMPANIES(Company_Code);
CREATE INDEX idx_company_status ON SRM_COMPANIES(Status);

-- Project indexes
CREATE INDEX idx_project_company ON SRM_PROJECTS(Company_ID);
CREATE INDEX idx_project_status ON SRM_PROJECTS(Status);
CREATE INDEX idx_project_dates ON SRM_PROJECTS(Start_Date, End_Date);

-- PAC Project indexes
CREATE INDEX idx_pac_project_status ON PAC_MNT_PROJECTS(Status);
CREATE INDEX idx_pac_project_dates ON PAC_MNT_PROJECTS(Start_Date, End_Date);

-- Resource indexes
CREATE INDEX idx_resource_type ON PAC_MNT_RESOURCES(Resource_Type);
CREATE INDEX idx_resource_availability ON PAC_MNT_RESOURCES(Availability);

-- Contract indexes
CREATE INDEX idx_contract_project ON PROJCNTRTS(Project_Code);
CREATE INDEX idx_contract_company ON PROJCNTRTS(Company_Code);
CREATE INDEX idx_contract_dates ON PROJCNTRTS(Start_Date, End_Date);

-- Staff indexes
CREATE INDEX idx_staff_project ON PROJSTAFF(Project_Code);
CREATE INDEX idx_staff_resource ON PROJSTAFF(Resource_Code);
CREATE INDEX idx_staff_dates ON PROJSTAFF(Start_Date, End_Date);

-- Support indexes
CREATE INDEX idx_support_company ON CLNTSUPP(Company_Code);
CREATE INDEX idx_support_status ON CLNTSUPP(Status);
CREATE INDEX idx_support_priority ON CLNTSUPP(Priority);

-- =============================================
-- Views for Common Queries
-- =============================================

-- Active projects with company details
CREATE VIEW v_active_projects AS
SELECT
    p.Project_Code,
    p.Project_Name,
    c.Company_Name,
    c.Company_Code,
    p.Budget,
    p.Actual_Cost,
    p.Revenue,
    (p.Revenue - p.Actual_Cost) as Profit,
    p.Start_Date,
    p.End_Date,
    p.Status
FROM PAC_MNT_PROJECTS p
LEFT JOIN PROJCNTRTS ct ON p.Project_Code = ct.Project_Code
LEFT JOIN SRM_COMPANIES c ON ct.Company_Code = c.Company_Code
WHERE p.Status = 'Active';

-- Resource allocation summary
CREATE VIEW v_resource_allocation AS
SELECT
    r.Resource_Code,
    r.Resource_Name,
    r.Resource_Type,
    COUNT(DISTINCT ps.Project_Code) as Projects_Count,
    SUM(ps.Allocation_Percent) as Total_Allocation,
    r.Capacity,
    (r.Capacity - COALESCE(SUM(ps.Allocation_Percent), 0)) as Available_Capacity
FROM PAC_MNT_RESOURCES r
LEFT JOIN PROJSTAFF ps ON r.Resource_Code = ps.Resource_Code
    AND ps.Status = 'Assigned'
    AND (ps.End_Date IS NULL OR ps.End_Date >= CURRENT_DATE)
GROUP BY r.Resource_Code, r.Resource_Name, r.Resource_Type, r.Capacity;

-- Project financial summary
CREATE VIEW v_project_financial_summary AS
SELECT
    p.Project_Code,
    p.Project_Name,
    p.Budget,
    p.Actual_Cost,
    p.Revenue,
    ct.Contract_Value,
    COUNT(DISTINCT ps.Staff_ID) as Staff_Count,
    SUM(ps.Allocation_Percent * ps.Cost_Rate) as Total_Staff_Cost,
    (p.Budget - p.Actual_Cost) as Budget_Variance,
    CASE
        WHEN p.Budget > 0 THEN ((p.Actual_Cost / p.Budget) * 100)
        ELSE 0
    END as Budget_Utilization_Percent
FROM PAC_MNT_PROJECTS p
LEFT JOIN PROJCNTRTS ct ON p.Project_Code = ct.Project_Code
LEFT JOIN PROJSTAFF ps ON p.Project_Code = ps.Project_Code
GROUP BY p.Project_Code, p.Project_Name, p.Budget, p.Actual_Cost, p.Revenue, ct.Contract_Value;