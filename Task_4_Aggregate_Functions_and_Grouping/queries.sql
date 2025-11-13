SELECT department,COUNT(*) AS tot_employees FROM employees
GROUP BY department;

SELECT department ,ROUND(AVG(salary),2) AS avg_salary
FROM employees
GROUP BY department;

SELECT department ,SUM(salary) AS total_salary
FROM employees
GROUP BY department;

SELECT department,COUNT(*) AS tot_employees
FROM employees
GROUP BY department
HAVING COUNT(*) > 1;

SELECT department,MAX(salary) AS highest_salary,MIN(salary) AS minimum_salary
FROM employees
GROUP BY department;

SELECT COUNT(DISTINCT department) AS distinct_departments FROM employees;


SELECT department,ROUND(AVG(salary),2) AS avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary DESC
LIMIT 1;


SELECT department,COUNT(*) AS cnt
FROM employees
WHERE salary>50000
GROUP BY department;

SELECT department,COUNT(*) AS cnt,AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary)>65000;