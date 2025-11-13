
-- 1. INNER JOIN
SELECT Customers.name, Orders.product, Orders.amount
FROM Customers
INNER JOIN Orders
ON Customers.customer_id = Orders.customer_id;

-- 2. LEFT JOIN
SELECT Customers.name, Orders.product
FROM Customers
LEFT JOIN Orders
ON Customers.customer_id = Orders.customer_id;

-- 3. RIGHT JOIN
SELECT Customers.name, Orders.product
FROM Customers
RIGHT JOIN Orders
ON Customers.customer_id = Orders.customer_id;

-- 4. FULL OUTER JOIN (workaround for SQLite/MySQL)
SELECT Customers.name, Orders.product
FROM Customers
LEFT JOIN Orders
ON Customers.customer_id = Orders.customer_id
UNION
SELECT Customers.name, Orders.product
FROM Customers
RIGHT JOIN Orders
ON Customers.customer_id = Orders.customer_id;
