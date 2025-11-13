CREATE TABLE Employees (
    emp_id INT AUTO_INCREMENT PRIMARY KEY,
    emp_name VARCHAR(50),
    department VARCHAR(30),
    salary DECIMAL(10,2),
    joining_date DATE
);

INSERT INTO Employees (emp_name, department, salary, joining_date) VALUES
('Rajat', 'IT', 60000, '2022-01-15'),
('Akash', 'HR', 45000, '2023-03-20'),
('Rohan', 'Finance', 55000, '2021-07-10'),
('Meera', 'IT', 70000, '2020-12-05');

DELIMITER $$
CREATE PROCEDURE GiveSalaryHike(IN dept_name VARCHAR(30), IN hike_percent INT)
BEGIN
    UPDATE Employees
    SET salary = salary + (salary * hike_percent / 100)
    WHERE department = dept_name;

    SELECT emp_name, department, salary
    FROM Employees
    WHERE department = dept_name;
END$$
DELIMITER ;

CALL GiveSalaryHike('IT', 10);

DELIMITER $$
CREATE FUNCTION CalculateBonus(empSalary DECIMAL(10,2))
RETURNS DECIMAL(10,2)
DETERMINISTIC
BEGIN
    DECLARE bonus DECIMAL(10,2);

    IF empSalary < 50000 THEN
        SET bonus = empSalary * 0.05; 
    ELSEIF empSalary BETWEEN 50000 AND 70000 THEN
        SET bonus = empSalary * 0.10;
    ELSE
        SET bonus = empSalary * 0.15; 
    END IF;
    RETURN bonus;
END$$
DELIMITER ;


SELECT emp_name, salary, CalculateBonus(salary) AS Bonus
FROM Employees;


DELIMITER $$
CREATE PROCEDURE ShowSalaryWithBonus(IN dept_name VARCHAR(30))
BEGIN
    SELECT emp_name, salary, CalculateBonus(salary) AS Bonus
    FROM Employees
    WHERE department = dept_name;
END$$
DELIMITER ;

CALL ShowSalaryWithBonus('IT');


DELIMITER $$
CREATE TRIGGER BeforeInsertEmployee
BEFORE INSERT ON Employees
FOR EACH ROW
BEGIN
    IF NEW.salary < 30000 THEN
        SET NEW.salary = 30000;
    END IF;
END$$
DELIMITER ;


INSERT INTO Employees (emp_name, department, salary, joining_date)
VALUES ('Rahul', 'IT', 20000, '2023-10-01');

SELECT * FROM Employees;

INSERT INTO Employees (emp_name, department, salary, joining_date)
VALUES ('Anjali', 'HR', 50000, '2023-09-20');

SELECT * FROM Employees;