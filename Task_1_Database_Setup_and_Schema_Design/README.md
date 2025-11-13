## Bank Database Schema
 This project contains the schema for a simple banking system. It includes tables for customers, branches, accounts, transactions, and loans. The database is designed with normalization principles to avoid redundancy and ensure data integrity.

 ### Database Setup: bank

 ```sql
CREATE DATABASE bank;
USE bank;
```

## Tables

### 1.Customers Table

```sql
CREATE TABLE customers(
    cust_id INT AUTO_INCREMENT PRIMARY KEY,
    cust_name VARCHAR(20) NOT NULL,
    cust_email VARCHAR(50) UNIQUE,
    cust_phone VARCHAR(15),
    address VARCHAR(50)
);
```
**Primary Key (cust_id)**: Uniquely identifies each customer

**Unique Key (cust_email)**: Ensures no duplicate emails

**Stores** customer information such as name, email, phone, and address

**Normalization**: 1NF applied (atomic columns, unique identifiers)

### 2.Branches Table

```sql
CREATE TABLE branches(
    branch_id INT AUTO_INCREMENT PRIMARY KEY,
    branch_name VARCHAR(20) NOT NULL,
    branch_location VARCHAR(20) NOT NULL
);
```


**Primary Key (branch_id)**: Uniquely identifies each branch

**Stores** branch name and location

**Normalization**: 1NF and 2NF applied as data depends solely on branch ID

### 3.Accounts Table
```sql
CREATE TABLE accounts(
    account_id INT AUTO_INCREMENT PRIMARY KEY,
    cust_id INT,
    branch_id INT,
    account_type ENUM('savings','current','fixed deposit') NOT NULL,
    balance DECIMAL(15,2) DEFAULT 0,
    FOREIGN KEY (cust_id) REFERENCES customers(cust_id),
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
);
```


**Primary Key (account_id)**: Uniquely identifies each account

**Foreign Keys**:

cust_id → references customers(cust_id)

branch_id → references branches(branch_id)

**Stores** account details including type and balance

**Normalization**: 1NF, 2NF, 3NF applied. Account depends on customer and branch IDs

### 4.Transactions Table
```sql
CREATE TABLE transactions(
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    account_id INT,
    amount DECIMAL(15,2) NOT NULL,
    transaction_type ENUM('deposit','withdrawal','transfer'),
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);
```

**Primary Key (transaction_id)**: Unique transaction identifier

**Foreign Key (account_id)**: Links transaction to an account

**Stores** transaction details including type, amount, and date

**Normalization**: 1NF, 2NF, 3NF applied. Transactions depend on account ID only

### 5.Loans Table
```sql
CREATE TABLE loans(
    loan_id INT AUTO_INCREMENT PRIMARY KEY,
    cust_id INT,
    amount DECIMAL(15,2) NOT NULL,
    loan_type ENUM('home','car','personal','education'),
    issue_date DATE,
    FOREIGN KEY (cust_id) REFERENCES customers(cust_id)
);
```
**Primary Key (loan_id)**: Unique loan identifier

**Foreign Key (cust_id)**: Links loan to a customer

**Stores** loan details including amount, type, and issue date

**Normalization**n: 1NF, 2NF, 3NF applied. Each loan record depends solely on loan_id


## Entity-Relationship Diagram (ERD)

![Alt Text](images/erdiagram.png)
### Explanation of Relationships:

**customers ↔ accounts**: One-to-many (a customer can have multiple accounts)

**branches ↔ accounts**: One-to-many (a branch can have multiple accounts)

**accounts ↔ transactions**: One-to-many (an account can have multiple transactions)

**customers ↔ loans**: One-to-many (a customer can have **multiple loans)

## Normalization Summary
**1NF**: All tables have atomic columns and unique primary keys

**2NF**: All non-key columns fully depend on the primary key

**3NF**: No transitive dependencies; foreign keys handle relationships