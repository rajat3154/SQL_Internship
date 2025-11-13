CREATE DATABASE college;
USE college;
CREATE TABLE students(
      id INT PRIMARY KEY,
      name VARCHAR(15) NOT NULL,
      age INT(3) DEFAULT 18,
      email VARCHAR(20) UNIQUE,
      grade VARCHAR(5)
);

INSERT INTO students(id,name,age,email,grade) VALUES (1,"aditya",21,"aditya@gmail.com","A");

INSERT INTO students(id,name,age,email,grade) VALUES (2,"arjun",NULL,"arjun@gmail.com","B");

INSERT INTO students(id,name,email,grade) VALUES (3,"rajat","rajat@gmail.com","A");

INSERT INTO students(id,name,age,email,grade) VALUES (4,"divya",20,"divya@gmail.com","B");

INSERT INTO students(id,name,age,email,grade) VALUES (5,"swati",21,"divya@gmail.com","A");

UPDATE students SET age=20 WHERE name="aditya";

UPDATE students SET grade="A+" WHERE grade="A";

DELETE FROM students WHERE id=2;

SELECT * FROM students;

CREATE TABLE alumnis AS SELECT * FROM students WHERE age=20;

SELECT * FROM alumnis;