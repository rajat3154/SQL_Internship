CREATE DATABASE internship;

USE internship;

CREATE TABLE Customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(50),
    city VARCHAR(50)
);

INSERT INTO Customers (customer_id, name, city) VALUES
(1, 'Alice', 'Delhi'),
(2, 'Bob', 'Mumbai'),
(3, 'Charlie', 'Delhi'),
(4, 'David', 'Bangalore'),
(5, 'Eva', 'Chennai');

CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);

INSERT INTO Orders (order_id, customer_id, amount) VALUES
(101, 1, 500),
(102, 1, 700),
(103, 2, 300),
(104, 3, 1200),
(105, 4, 400),
(106, 5, 1500);

SELECT 
    name,
    (SELECT SUM(amount) 
     FROM Orders o 
     WHERE o.customer_id = c.customer_id) AS total_spent
FROM Customers c;

SELECT name 
FROM Customers
WHERE customer_id IN (SELECT customer_id FROM Orders);

SELECT name 
FROM Customers c
WHERE EXISTS (
    SELECT 1 
    FROM Orders o 
    WHERE o.customer_id = c.customer_id AND o.amount > 600
);


SELECT name
FROM Customers c
WHERE EXISTS (
    SELECT 1 FROM Orders o WHERE o.customer_id = c.customer_id
);

SELECT customer_id, AVG(total_amount) AS avg_spent
FROM (
    SELECT customer_id, SUM(amount) AS total_amount
    FROM Orders
    GROUP BY customer_id
) AS derived
GROUP BY customer_id;

SELECT name 
FROM Customers
WHERE customer_id IN (
    SELECT customer_id 
    FROM Orders
    WHERE amount > (SELECT AVG(amount) FROM Orders)
);