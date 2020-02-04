# Transform the Column with Key-value Pairs String to Wide Table
This document describes the design to transform the column with key-value pairs string to a wide ODPS table where
the values of each key is in a each column.

## Motivation
Generally, the data developers parse logs to features and the number of features may vary. It is convenient to save features using
key-value pair string as a column in a table. However, it is difficult to analyze features in the table using SQL syntax. So we need to transform
the column to a wide table. Then we can analyze each features like calculate the mean and variance for the numeric feature. For example, the key-value column is

| features| label|
|---------|------|
| age:19,income:1200,education:Master | 1 |
| age:23,income:4500,education:Doctor | 0 |

And we need to transform it to 

| age | income | education | label |
|-----|--------|-----------|-------|
| 19  | 1200   | Master    | 1 |
| 23  | 4500   | Doctor    | 0 |


## Design Components

### ODPS UDF to Transform Key-value Pairs String to Multiple Values
ODPS provides [UDF](https://help.aliyun.com/document_detail/73359.html?spm=5176.10695662.1996646101.searchclickresult.759c46d7SD5Hmr#title-mty-z7z-s1j) for users to define transformation using Python. We choose the [UDTF](https://help.aliyun.com/document_detail/73359.html?spm=5176.10695662.1996646101.searchclickresult.759c46d7SD5Hmr#title-d80-lvi-uc7) in UDF to implement the transformation because it can 
match any input parameters in SQL. The UDTF demo is

```python
#coding:utf-8
from odps.udf import annotate
from odps.udf import BaseUDTF

class Explode(BaseUDTF):
   def process(self, arg):
       ...
```

For the UDTF to transform key-value pairs, we can implement a UDTF with multiple arguments. The arguments in the UDTF likes 
```
udtf_func(kv_column, append_columns, feature_names, inter_kv_separator, intra_kv_separator)
```
The "kv_column" is the column name with key-value pairs strings. \
The "append_columns" is column names which are directly exported without any transformation. \
The "feature_names" is the a string with feature names separated by ",". \
The "inter_kv_separator" is the inter key-value pairs separator. \
The "intra_kv_separator" is the intra key-value pairs separator.

For the table showed in the motivation section, The SQL with the UDTF is 
```sql
SELECT udtf_func(features, label, "age, income, education", ",", ":") as (age, income, education, label)  FROM input_table
```

The implementation of the UDF is
```python
#coding:utf-8
from odps.udf import annotate
from odps.udf import BaseUDTF

class Explode(BaseUDTF):
   def process(self, *args):
       feature_names = args[-3].split(",")
        inter_kv_sep = args[-2]
        intra_kv_sep = args[-1]
        feature_values = parse_kv_string_to_dict(
            args[0],
            feature_names,
            inter_kv_sep,
            intra_kv_sep
        )
        for value in args[1:-3]:
            feature_values.append(str(value))
        self.forward(*feature_values)
```

### Infer the Feature Names and Generate the SQL to Transform.
If the number of features in key-value pairs string is very large, it is tedious for users to write the feature names in the transformation SQL.
Using Pyodps, we can infer all feature names from the records in the table and then generate the SQL. 

By default, we download 100 records from the ODPS table by pyodps tunnel and then parse feature names from those records.
```python
feature_names = set()
for record in table.head(100):
    kv_dict = parse_kv_string_to_dict()
    feature_names.update(kv_dict.keys())
```

After inferring the feature names, we can generate the SQL by the template:
```sql
TRANSFORM_SQL_TEMPLATE = "CREATE TABLE IF NOT EXISTS {output_table} LIFECYCLE 7 AS \n\
    SELECT \n\
        {udf} \n\
    FROM {input_table}"
```
The template of "udf" in the SQL template is:
```python
"""{udf_func}({input_col_str},
    "{features_str}", "{inter_sep}", "{intra_sep}")
    as ({output_col_str})
"""
```

### Execute the Generated SQL by PyODPS.
The steps to execute the generated SQL by PyODPS are:
1. Create an ODPS resource using the UDTF python file. 
2. Create an ODPS function using the resource. 
3. Using PyODPS to submit the SQL to ODPS cluster.
