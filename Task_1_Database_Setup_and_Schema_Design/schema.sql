CREATE DATABASE bank;

USE bank;

CREATE TABLE customers(
      cust_id INT AUTO_INCREMENT PRIMARY KEY,
      cust_name VARCHAR(20) NOT NULL,
      cust_email VARCHAR(20) UNIQUE,
      cust_phone INT(15),
      address VARCHAR(50)
);

CREATE TABLE branches(
      branch_id INT AUTO_INCREMENT PRIMARY KEY.
      branch_name VARCHAR(20) NOT NULL,
      branch_location VARCHAR(20) NOT NULL
);

CREATE TABLE accounts(
      account_id INT AUTO_INCREMENT PRIMARY KEY,
      cust_id INT,
      branch_id INT,
      account_type ENUM('savings','current','fixed deposit') NOT NULL,
      balance DECIMAL(15,2) DEFAULT 0,
      FOREIGN KEY (cust_id) REFERENCES customers(cust_id)
      FOREIGN KEY  (branch_id) REFERENCES branches(branch_id)
);

CREATE TABLE transactions(
      transaction_id INT AUTO_INCREMENT PRIMARY KEY,
      account_id INT
      amount DECIMAL(15,2) NOT NULL,
      transaction_type ENUM('deposit','withdrawal','transfer'),
      transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(account_id) REFERENCES accounts(account_id)
);

CREATE TABLE loans(
      loan_id INT AUTO_INCREMENT PRIMARY KEY,
      cust_id INT,
      amount DECIMAL(15,2) NOT NULL,
      loan_type ENUM('home','car','personal','education'),
      issue_date DATE,
      FOREIGN KEY (cust_id) REFERENCES customers(cust_id)
)