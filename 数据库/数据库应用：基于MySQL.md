# 数据库应用：基于MySQL

## 第一章 结构化查询语言SQL

​		结构化查询语言（Structured Query Language，SQL）是一种介于关系代数与关系演算之间的语言，其功能包括查询、操纵、定义和控制四个方面。SQL语言已经被定义为关系数据库系统（Relational Data Base System，RDBS）的国际标准（International Standard，IS）。

### 一、SQL概述

​	SQL数据库的体系结构为三级模式。具有以下特征：

- 一个SQL模式(*Schema*)是表和约束的集合。
- 一个表(*Table*)是行(*Row*)的集合。每行是列(*Column*)的序列，每列对应一个数据项。
- 一个表可以是一个基本表，也可以是一个视图。基本表是实际存储在数据库中的表。视图是从基本表或其他视图中导出的表，本身不独立存储在数据库中。*视图是一个虚表。*



​	SQL主要分为以下四个部分：

- 数据定义（SQL DDL）：定义SQL模式、基本表、视图和索引。
- 数据操纵（SQL DML）：数据查询，数据更新。数据更新又分为插入、删除、修改。
- 数据控制（SQL DCL）：对基本表和视图的授权、完整性规则的描述、事务控制语句。
- 嵌入式SQL。



### 二、SQL的数据定义

​		SQL的数据定义部分包括对SQL模式(*Schema*)，基本表(*Table*)，视图(*View*)，索引(*Index*)的创建和撤销操作。

#### 1、SQL模式的创建和撤销

一个SQL模式被定义为基本表的集合。创建了一个SQL模式就是定义了一个存储空间。

SQL模式的创建可以使用`CREATE`语句实现：

```sql
CREATE SCHEMA <模式名> AUTHORIZATION <用户名>
```

使用MySQL创建SQL模式：

```sql
CREATE DATABASE [IF NOT EXISTS] <数据库名> [DEFAULT CHARSET utf8 COLLATE utf8_general_ci];
```

执行结果为：创建指定名称的数据库，如果数据库不存在则创建，存在则不创建；并设定编码集为utf8。

当一个SQL模式及其所属的基本表、视图都不需要时，使用`DROP`语句撤销SQL模式。

```sql
DROP SCHEMA <模式名> [CASCADE|RESTRICT]
```

> CASCADE模式下，把SQL模式及其下属的基本表、视图、索引等元素全部删除
>
> RESTRICT模式下，只有SQL模式没有任何下属元素时才允许撤销SQL模式

使用MySQL删除SQL模式：

```sql
DROP DATABASE <数据库名>;
```

#### 2、SQL的基本数据类型

数值型

| 数据类型 | 解释 |
| - | - |
| INTEGER | 长整数 |
| SMALLINT | 短整数 |
| REAl | 浮点数 |
| DOUBLE PRECISION | 双精度浮点数 |
| FLOAT(n) | 精度至少为n位数字的浮点数 |
| NUMERIC(p,d) | 由p位数字，d位小数组成的浮点数 |

字符串型

| 数据类型 | 解释 |
| - | - |
| CHAR(n) | 长度为n的定长字符串 |
| VARCHAR(n) | 最大长度为n的变长字符串 |

位串型

| 数据类型 | 解释 |
| - | - |
| BIT(n) | 长度为n的定长二进制位串 |
| BIT VARYING(n) | 最大长度为n的变长二进制位串 |

时间型

| 数据类型 | 解释 |
| - | - |
| DATE | YYYY-MM-DD |
| TIME | HH:MM:SS |



#### 3、基本表的创建、修改和撤销

**基本表的创建**

创建基本表，就是定义基本表的结构。可以用`CREATE`语句实现。

```sql
CREATE TABLE [<模式名>] <表名> (<列名> <类型>, ..., [完整性约束])
```

完整性约束包括

- 主键子句`PRIMARY KEY`
- 检查子句`CHECK`
- 外键子句`FOREIGN KEY`

```sql
CREATE TABLE [<模式名>] <表名> (<列名> <类型>, ..., [PRIMARY KEY <列名>], [[CONSTRAINT <约束名>]CHECK <属性> BETWEEN val AND val], [FOREIGN KEY <列名> REFERENCES <表名>(<列名>)])
```

**基本表的修改**

增加新的属性

```sql
ALTER TABLE <表名> ADD <新属性名> [新属性完整性约束]
```

删除原有的属性

```sql
ALTER TABLE <表名> DROP <属性名> [CASCADE|RESTRICT]
```

增加约束

```sql
ALTER TABLE <表名> ADD CONSTRAINT <约束名> CHECK(<属性> BETWEEN val AND val)
```

删除约束

```sql
ALTER TABLE <表名> DROP <约束名>
```

**删除基本表**

```sql
DROP TABLE <表名> [CASCADE|RESTRICT]
```



#### 4、视图的创建和撤销

todo

#### 5、索引的创建和撤销

创建索引

```sql
CREATE [UNIQUE] INDEX <索引名> ON <表名> (<列名> [ASC|DESC])
```

撤销索引

```sql
DROP INDEX <索引名> ON <表名>
```



### 三、SQL的数据查询

#### 1、SELECT语句

SELECT语句的组成：

1. `SELECT`子句
2. `FROM`子句
3. `WHERE`子句（行条件语句）
4. `GROUP BY`子句（分组子句），`HAVING`子句（组条件子句）
5. `ORDER BY`子句（排序子句）

SELECT语句的句法和执行过程：

1. 读取`FROM`子句中基本表、视图的数据，执行笛卡尔积操作
2. 选取满足`WHERE`子句中给出条件的元组
3. 按`GROUP`子句中指定列的值分组，同时提取满足`HAVING`子句中组条件表达式的组
4. 按`SELECT`子句中给出的列名或列表达式求值输出
5. 按`ORDER`子句对输出的目标进行排序

#### 2、单表查询

**查询指定列**

```sql
SELECT <列名> FROM <表名>
```

**查询所有列**

```sql
SELECT * FROM <表名>
```

**查询不同值**

```sql
SELECT DISTINCT <列名> FROM <表名>
```

**比较**（`>, <, =, >=, <=, <>`）

```sql
SELECT <列名> FROM <表名> WHERE <属性> = <条件>
```

**条件**（`NOT, AND, OR`）

```sql
SELECT <列名> FROM <表名> WHERE <属性1> = <value1> AND <属性2> = <value2>
```

**确定范围**

```sql
SELECT <列名> FROM <表名> WHERE <属性> [NOT] BETWEEN <value1> AND <value2>
```

**确定集合**

```sql
SELECT <列名> FROM <表名> WHERE <属性> [NOT] IN <(集合)>
```

**涉及空值**

```sql
SELECT <列名> FROM <表名> WHERE <属性> IS [NOT] NULL
```

**字符查询、通配符，模糊匹配**（`%`任意匹配，`_`匹配一次）

```sql
SELECT <列名> FROM <表名> WHERE <属性> LIKE <%Pattern_>
```

**聚合函数**（用于`SELECT`子句）

```sql
COUNT([DISTINCT] *)           统计元组个数
COUNT([DISTINCT] <列名>)       统计某一列值的个数
SUM([DISTINCT] <列名>)         统计某一列总和
AVG([DISTINCT] <列名>)         统计某一列平均值
MAX([DISTINCT] <列名>)         统计某一列最大值
MIN([DISTINCT] <列名>)         统计某一列最小值
```

```sql
SELECT SUM([DISTINCT] <列名>) FROM <表名> WHERE <属性> = <条件>
```

**分组**

```sql
SELECT <列名>, <属性> FROM <表名> GROUP BY <属性>
```

> SQL规定，凡是分组`GROUP BY`使用的类名必须在`SELECT`子句中出现。
>
> 对于*SQL Server*，`SELECT`中出现的属性必须在`GROUP BY`中出现，但不代表一定要按此分组；对*MySQL*则无此规定。

```sql
SELECT <列名>， <属性1> FROM <表名> GROUP BY <属性1> HAVING <属性2> = <值>
```

**排序**（升序`ASC`，降序`DESC`）

```sql
SELECT <列名> FROM <表名> ORDER BY <属性>
SELECT <列名>，MAX([DISTINCT] <列名>) FROM <表名> ORDER BY 2 DESC
```

> 使用聚合函数作为排序依据时，可以使用其在`SELECT`子句中的位置作为参数。

#### 3、多表查询

联接操作，多表合并（内联接，外联接，交叉联接）

**自身联接**，查询一张表相同列的两个属性

```sql
SELECT <列名> FROM <表名> AS <别名1>，<表名> AS <别名2> WHERE <别名1>.主键 = <别名2>.主键 AND <别名1.属性> = <别名2.属性>
```

**复合条件联接**（有相同属性）

```sql
SELECT <表名.列名> FROM <表名1>，<表名2> WHERE <表名1.键> = <表名2.键> AND <表名1.属性> = 'value' AND <表名2.属性> = 'value'
```

**嵌套查询**  IN子查询（值 `IN` 集合，IN相当于$\in$）

```sql
SELECT <列名> FROM <表名> WHERE <属性> [NOT] IN (SELECT <列名> FROM <表名> WHERE <属性> = 'value')
```

**ANY,ALL比较子查询**

```sql
SELECT <列名> FROM <表名> WHERE <属性>=<条件> GROUP BY <属性> HAVING AVG(<属性>) > ALL (SELECT <列名> FROM <表名> WHERE <属性>=<条件>)
等同于
SELECT <列名> FROM <表名> WHERE <属性>=<条件> GROUP BY <属性> HAVING AVG(<属性>) > (SELECT MAX(<列名>) FROM <表名> WHERE <属性>=<条件>)
```

**EXISTS存在子查询**

```sql
SELECT <列名> FROM <表名1> WHERE <属性> [NOT] EXISTS (SELECT * FROM <表名2> WHERE <属性> = 'value' AND <表名1>.<属性>=<表名2>.<属性>)
等同于
SELECT <列名> FROM <表名1> WHERE <属性> [NOT] IN (SELECT <列名> FROM <表名2> WHERE <属性> = 'value')
```

> SQL语言不存在全称量词，使用全程需要转换为不存在不满足（双重否定）。
> $$
> (\forall x) P \equiv \neg (\exists x) \neg P
> $$

```sql
SELECT cno,cname FROM c WHERE NOT EXISTS (SELECT sno FROM s WHERE NOT EXISTS (SELECT * FROM sc WHERE sno=s.sno AND cno=c.cno))
```

> SQL不支持逻辑蕴含，使用需要转换。
> $$
> p \rightarrow q \equiv \neg p \lor q
> $$

求：所有选修 学号为S3的学生选修的课程 的学生的学号。
设p：S3选修课程，q：Sx选修课程。
$$
(\forall c_y)(p \rightarrow q) \\ \equiv \neg(\exists c_y)\neg(p \rightarrow q) \\ \equiv \neg(\exists c_y)\neg(\neg p \lor q) \\ \equiv \neg(\exists c_y)( p \land \neg q)
$$

```sql
SELECT sno FROM sc x WHERE NOT EXiSTS (SELECT * FROM sc y WHERE sno='s3' AND NOT EXISTS (SELECT FROM sc z WHERE sno=x.sno and cno=y.cno))
```



### 四、SQL的数据操纵

#### 1、数据插入

一次插入一行

```sql
INSERT INTO <表名> [<属性名>] VALUES (<元组值>)
```

一次插入多行

```sql
INSERT INTO <表名> [<属性名>] VALUES (<元组值>), (<元组值>), (<元组值>)...
```

子查询插入

```sql
INSERT INTO <表名> [<属性名>] (子查询)
```

#### 2、数据删除

删除满足条件的行

```sql
DELETE FROM <唯一表名> [WHERE <带子查询的条件表达式(IN)>]
DELETE FROM s WHERE sno NOT IN (SELECT sno FROM sc)
```

#### 3、数据修改

```sql
UPDATE <表名> SET <列名>=<表达式> [WHERE <带子查询的条件表达式>]
```

