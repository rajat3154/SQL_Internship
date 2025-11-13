-- Employee Table
CREATE TABLE Employee (
    EmpID INT PRIMARY KEY,
    EmpName VARCHAR(50),
    DeptID INT,
    Salary DECIMAL(10,2),
    JoinDate DATE
);

-- Department Table
CREATE TABLE Department (
    DeptID INT PRIMARY KEY,
    DeptName VARCHAR(50),
    Location VARCHAR(50)
);

-- Projects Table
CREATE TABLE Project (
    ProjectID INT PRIMARY KEY,
    ProjectName VARCHAR(50),
    DeptID INT,
    Budget DECIMAL(10,2)
);

INSERT INTO Department VALUES
(1, 'HR', 'New York'),
(2, 'IT', 'San Francisco'),
(3, 'Finance', 'Chicago');

INSERT INTO Employee VALUES
(101, 'Alice', 2, 70000, '2022-01-15'),
(102, 'Bob', 2, 80000, '2021-05-23'),
(103, 'Charlie', 1, 60000, '2023-02-10'),
(104, 'David', 3, 90000, '2020-09-12'),
(105, 'Eva', 2, 75000, '2022-06-30');

INSERT INTO Project VALUES
(201, 'Project A', 2, 100000),
(202, 'Project B', 2, 150000),
(203, 'Project C', 1, 50000),
(204, 'Project D', 3, 200000);

CREATE VIEW EmpDeptView AS
SELECT e.EmpID, e.EmpName, e.Salary, d.DeptName
FROM Employee e
JOIN Department d ON e.DeptID = d.DeptID;

CREATE VIEW HighSalaryEmp AS
SELECT EmpName, Salary
FROM Employee
WHERE Salary > 75000;

CREATE VIEW RecentJoinees AS
SELECT EmpName, JoinDate
FROM Employee
WHERE JoinDate > '2022-01-01';

CREATE VIEW DeptProjectBudget AS
SELECT d.DeptName, SUM(p.Budget) AS TotalBudget
FROM Project p
JOIN Department d ON p.DeptID = d.DeptID
GROUP BY d.DeptName;

CREATE VIEW EmpCountDept AS
SELECT d.DeptName, COUNT(e.EmpID) AS NumEmployees
FROM Employee e
JOIN Department d ON e.DeptID = d.DeptID
GROUP BY d.DeptName;

CREATE VIEW EmpProjectCount AS
SELECT e.EmpName, COUNT(p.ProjectID) AS ProjectsAssigned
FROM Employee e
LEFT JOIN Project p ON e.DeptID = p.DeptID
GROUP BY e.EmpName;

CREATE VIEW DeptMaxSalary AS
SELECT d.DeptName, MAX(e.Salary) AS MaxSalary
FROM Employee e
JOIN Department d ON e.DeptID = d.DeptID
GROUP BY d.DeptName;

CREATE VIEW EmpSalaryPercent AS
SELECT e.EmpName, e.Salary, 
       ROUND((e.Salary / SUM(e.Salary) OVER(PARTITION BY e.DeptID) * 100), 2) AS SalaryPercent
FROM Employee e;


SELECT * FROM EmpDeptView;

SELECT * FROM HighSalaryEmp;

SELECT * FROM RecentJoinees;

SELECT * FROM DeptProjectBudget;

SELECT * FROM EmpCountDept;

SELECT * FROM EmpProjectCount;

SELECT * FROM DeptMaxSalary;

SELECT * FROM EmpSalaryPercent;
