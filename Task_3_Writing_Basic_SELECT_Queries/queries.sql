SELECT * FROM students;

SELECT * FROM students WHERE age > 20;

SELECT * FROM students ORDER BY age DESC LIMIT 3;

SELECT DISTINCT grade FROM students;

SELECT * FROM students WHERE grade='A' OR age=18;

SELECT * FROM students WHERE name LIKE "A%";

SELECT * FROM students WHERE age BETWEEN 19 AND 22;

SELECT * FROM students WHERE grade IN ('A','B');

SELECT grade, COUNT(*) AS total_students FROM students GROUP BY grade;