---
layout: post
title: Using SQL to extract key insights for a Diner
image: "/posts/sql-title.jpg"
tags: [SQL]
---


Our client, Danny's Diner a Japanese restaurant, hired an analytics consultancy to get insights into their sales, customer preference and membership offers

## Project Overview

Danny is a Japanese food connoisseur and always dreamed of opening a Japanese restaurant someday. His dreams were realised in begining of 2021 when he opened his own Japanese diner with limited menu of his favorite Sushi, Ramen and Curry. A month into business Danny is wondering if his business is doing okay? Is there is a way if he could attract more customer? He wants to find out if there was an impact of the existing customer loyalty program and if he should expand it further.

By understanding customer behavior and pattern, he will be able to increase his sales and provide better customer service. Discerning frequently ordered food will help him expand his menu and introduce different versions of a dish. 

As analytics consultants, we would acquire available data from him and use SQL queries to help him find insights!
<br>

## Data Overview

As it has been only little over a month that Danny has opened doors of his restaurant, there is limited data which is available. Also, due to privacy issues, he could only provide us with a sample of the data. 

Danny has provided us with 3 datasets for analysis. This includes:


* Sales
* Menu
* Membership

The sales table contains all the sales data and relative customer and order information.

customer_id - unique id for individual customers
<br>
order_date - date on which the customer ordered the food
<br>
product_id - unique id for the food/product on the menu

The menu table has the all the information pertaining to the food.

product_id - unique id for the food/product on the menu
<br>
product_name - actual name of the food item corresponding to the product_id
<br>
price - cost of the food item

The members table is associated loyalty program introducted by Danny which gives additional points to the members.

customer_id - unique id for individual customers
join_date - date on which the customers were enlisted in the program


A relational database is used to store our data. In relational databases the data is stored in what is called a **table**. 
A table is a collection of related data entities which is stored in the form of columns and rows. A table may have multiple columns and rows and a database has/ may have multiple such tables. The 3 provided data sets will be stored in 3 tables.

To create a table we use DDL - Data Definition Language which consists of SQL commands. This consists of queries which are used to define or alter structure of a table. As these queries can lead to a irreversible change in contents of the table, not all the users in the organization will have acsess to implement them.  

##### Create Sales table

```sql
CREATE TABLE sales (

  "customer_id" VARCHAR(1),
  "order_date" DATE,
  "product_id" INTEGER
);
```
<br>

Now, create a sales table with columns: customer_id, order_date and product_id. While creating a table it is essential to specify the data type of the column along with its name. The data type is a specification for SQL as to what type of data is expected in each column, and it also identifies how SQL will interact with the stored data.

* For customer_id, **VARCHAR**, which is used tostore a variable lenght string. The string could consists if letters, characters or numbers. The lenght of the string is set in the parenthesis which indicates that the total lenght of data in the specified column cannot exceed this set value. As we have information of only 3 customers, set the lenght to 1. This may be altered in the future.

* For order_date,  **DATE**, which is used to store date. It is in the format of YYYY-MM-DD.

* For product_id,  **INTEGER**, which is used to store integer values.

Apart from these data types other most commonly used data types include:

* FLOAT - It is used to store float values in a table
* DATETIME - It is used to store a combination of date and time. For eg, in this case, it could be used to store the date and time at which the customer ordered a food item. It is in the format of YYYY-MM-DD hh:mm:ss.
<br>

##### Insert data into sales table

After the table structure is ready, Let's insert the data into these tables. DML or Data Manipulation language is used for this purpose. As the name suggests it deals with manipulation of data and perform CRUD (Create, Read, Update, Delete) operations. 

```sql
INSERT INTO sales
  ("customer_id", "order_date", "product_id")
VALUES
  ('A', '2021-01-01', '1'),
  ('A', '2021-01-01', '2'),
  ('A', '2021-01-07', '2'),
  ('A', '2021-01-10', '3'),
  ('A', '2021-01-11', '3'),
  ('A', '2021-01-11', '3'),
  ('B', '2021-01-01', '2'),
  ('B', '2021-01-02', '2'),
  ('B', '2021-01-04', '1'),
  ('B', '2021-01-11', '1'),
  ('B', '2021-01-16', '3'),
  ('B', '2021-02-01', '3'),
  ('C', '2021-01-01', '3'),
  ('C', '2021-01-01', '3'),
  ('C', '2021-01-07', '3');
  ```

This is how our sales table looks:

![alt text](/img/posts/sales_table.png)

Similary, create menu and members table and insert data in them

##### Create Menu table

```sql
CREATE TABLE menu (

  "product_id" INTEGER,
  "product_name" VARCHAR(5),
  "price" INTEGER
);
```

##### Insert data into menu table

```sql
INSERT INTO menu
  ("product_id", "product_name", "price")
VALUES
  ('1', 'sushi', '10'),
  ('2', 'curry', '15'),
  ('3', 'ramen', '12');
```

This is how our menu table looks:

![alt text](/img/posts/menu_table.png)
<br>

##### Create members table

```sql
CREATE TABLE members (
  "customer_id" VARCHAR(1),
  "join_date" DATE
);
```

##### Insert data into members table

```sql
INSERT INTO members
  ("customer_id", "join_date")
VALUES
  ('A', '2021-01-07'),
  ('B', '2021-01-09');
```

This is how our members table looks:

![alt text](/img/posts/members_table.png)
<br>

### Danny's Questions

Data is now ready to answer Danny's questions, that will help him find insights into his business and will determine his future course of actions.

##### Q1) What is the total amount each customer spent at the restaurant?

We use DQL - Data Query Language to query data from the given tables. Also, use the **JOIN** function to join the tables of sales and menu, as the customer data would come from the sales table while to determine the customer spend, we would have to refer the menu table for price of each item. A **JOIN** clause is used to combine rows from two or more tables, based on a related column between them.

In addition, we also implement Aggregate functions like SUM and GROUP BY. 

**Aggregate function** performs operations on collection of values to return a single value.

**SUM** returns the total sum of a numeric column.

**GROUP BY** groups rows that have the same values into summary rows. Here we group our individual customers 
to determine the total amount they spent at the restaurant

<br>
![alt text](/img/posts/q1.png)

Customer A spent \$76.
<br>
Customer B spent \$74.
<br>
Customer C spent \$36.
<br>
<br>

##### Q2) How many days has each customer visited the restaurant?

Here COUNT function is used to count the number of days. **COUNT** function is used to count the number of rows returned in a SELECT statement. Alongside, **DISTINCT** function is used to return only distinct values. 

If DISTINCT is not used a repeated value may be counted twice. For eg, from sales table we observe that customer A visited the restaurant twice on ‘2021–01–07’, then number of visits may be counted as 2 instead of 1 if we don't use DISTINCT function.

<br>
![alt text](/img/posts/q2.png)

Customer A visited 4 times.
<br>
Customer B visited 6 times.  
Customer C visited 2 times.
<br>
<br>

##### Q3) What was the first item from the menu purchased by each customer?

As this question, cannot be solved using a simple Select Query, we take help of CTE and Windows function.

A **CTE** (Common Table Expression) is a temporary result set that can be used in a subsequent SELECT statement. Each SQL CTE is like a named query, whose result is stored in a virtual table (a CTE) to be referenced later in the main query. The biggest difference between temporary table and CTE is that the scope of CTE is limited to a query only.

Window functions applies aggregate and ranking functions over a set of rows. GROUP BY function collapses rows into groups and hence we cannot access individual records. Windows function helps overcome this.

<br>
![alt text](/img/posts/q3.png)

All the 3 customers purchased food for the first time on 1-1-2021. Customer A purchased sushi and curry. As we do not have timestamp of purchase, we cannot determine which of these items was purchased first.
Customer B and Customer C bought curry and ramen respectively.
<br>
<br>

##### Q4) What is the most purchased item, how many times was it purchased by all customers?

<br>
![alt text](/img/posts/q4.png)

Ramen was the most popular item and was bought a total of 8 times. Hence, Danny could introduce some offers on ramen or provide combo meal of ramen with other menu items to boost sales. 
<br>
<br>

##### Q5) Which item was the most popular for each customer?

<br>
![alt text](/img/posts/q5.png)

Ramen was the most bought item for customer A and C. Customer B enjoyed ordering all the 3 items on the menu equal number(2) of times. 
<br>
<br>

##### Q6) Which item was purchased first by the customer after they became a member?

<br>
![alt text](/img/posts/q6.png)

From members table we observe that only customer A and B have opted to be a member of Danny's Diner loyalty program. Customer A joined the program on 7th January and purchased curry on the same day. Customer B became a part of the program on 9th January and ordered sushi on 11th January. 

We would require more data to determine if the loyalty program has any impact on customer food choice.
<br>
<br>

##### Q7) Which item was purchased just before the customer became a member?

<br>
![alt text](/img/posts/q7.png)

This is reverse of question 6. Before becoming a member customer A ordered sushi and curry, while customer B ordered curry.
<br>
<br>

##### Q8) What is the total items and amount spent for each member before they became a member?

<br>
![alt text](/img/posts/q8.png)

Before becoming a member, customer A bought 2 items worth \$25 and customer B bought 3 items worth \$40.
<br>
<br>

##### Q9) If each  $1 spent equates to 10 points and sushi has a 2x points multiplier - how many points would each customer have?


To find the total points we implement CASE-WHEN statement. It is used for handling logic statements in SQL. It is similar to If-Then-Else statement in other programming languages. The CASE statement goes through list/s of condition/s followed by THEN  and returns a value when the condition is met. If no condition is met, it returns the value in ELSE statement.

We implement CASE-WHEN to create conditional statements:

$1 spent = 10 points 

But, sushi  gets 2x points, i.e for product_id = 1:
<br>
$1 = 20 points.

<br>
![alt text](/img/posts/q9.png)

Customer A, B and C have 860, 940 and 360 total points respectively.
<br>
<br>

##### Q10) In the first week after a customer joins the program (including their join date) they earn 2x points on all items, not just sushi - how many points do customer A and B have at the end of January?

<br>
![alt text](/img/posts/q10.png)

At the end of January, customer A and B have 1370 and 940 points respectively.

## Conclusion

We as analytics consultant have provided Danny with all the data he requires to grow his business. We also, informed him that it would be challenging to form any insights with limited data and suggested him to collect data further. After a few months, we could revisit these queries and explore any potential trends or insights. Depending on the data collected and reports required, we could also write stored procedures for routine queries.
