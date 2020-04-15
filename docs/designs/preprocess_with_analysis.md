# Preprocess with Analysis Result Design
This document describes the design about how to use analysis result while preprocessing feature inputs.

## Motivation
Before preprocessing the feature inputs, we need to analyze the training dataset to collect the feature statistical results. For example, we need the mean and standard deviation to normalize a numeric value, `vocabulary` to lookup a string value to an integer id and `boundary` to discretize a numeric value. Using SQLFlow, the training dataset is usually a table saved in MySQL or MaxCompute and other databases, so we can use SQL to analyze the training table. During [data transformation pipeline](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/data_transform.md), we may launch a pod to analyze the training table and then submit the ElasticDL training job. So, the design is to solve how to pass the analysis results into the pods of an ElasticDL training job.

## Define preprocess layers with analysis result

### 1. Persist the analysis result collected in the analysis pod.
For MySQL or MaxCompute table, we can use SQL to analyze each column. For example, the table is

|  age | education | marital |
| ---- | --- | --- |
|  34  | Master | Divorced |
|  54  | Doctor | Never-married |
|  42  | Bachelor | Never-married |
|  49  | Bachelor | Divorced |

For numeric column, we can get the mean, standard deviation and bucket boundaries using
```sql
SELECT 
    AVG(age) AS age_avg,
    STDDEV(age) AS age_std,
    PERCENTILE(age, ARRAY(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) AS age_bucket_boundaries 
FROM ${training_table}
```
The SQL expression use MaxCompute SQL syntax and `PERCENTILE` is a function in MaxCompute SQL

For feature to hash with bucket size, we can get the count of distinct values by 
```sql
SELECT count(distinct(marital)) AS martial_count FROM ${training_table}
```

For feature to lookup with vocabulary, we can get the vocabulary by
```sql
SELECT value FROM (
    SELECT education AS value, count(education) AS _count
    FROM {training_table}
    GROUP BY education
    ORDER BY _count DESC
)
WHERE _count >= {threshold};

SELECT value FROM (
    SELECT martial AS value, count(martial) AS _count
    FROM {training_table}
    GROUP BY martial
    ORDER BY _count DESC
)
WHERE _count >= {threshold};
```
The `WHERE _count >= {threshold}` will filter the values whose count is less than threshold to avoid overfitting.

Besides vocabulary, other analysis results are a number or a list of number like bucket boundaries. So we can save them into a table like:

|  feature_stats | value | 
| ---- | --- | 
|  age_mean  | 44.75 |
|  age_std_dev  | 56.6875 | 
|  age_bucket_boundries  | 30,40,50 | 
| martial-status-count  | 2 | 

Because the vocabulary size may be huge, we cannot save it into a record like:

|  feature_stats | value | 
| ---- | --- |  
| education_vocab  | Master,Doctor,Bachelor |
| martial_vocab  | Divorced,Never-married |

So, we save the vocabulary into a column and each record has an element, like

| education | martial | 
| ---- | --- |  
| Master  | Divorced |
| Doctor  | Never-married |
| Bachelor|  |

After analysis, we get two tables with the analysis results. One is the statistics table which saves the mean, standard deviation, bucket boundaries and distinct count. And another is vocabulary table which saves the vocabulary.

### Pass analysis results to build a model in training pods.
For the values in the statistics table, we can write them into environment variables for the training pod to build model. For example:
```shell
envs='age_mean=44.75,age_std=56.6875,age_bucket_boundaries="30,40,50"'
```
In preprocessing layers, we can get the statistics from environment variables like:
```python
import os
from elasticdl_preprocessing.layers import Discretization
age_boundaries = list(
    map(float, os.getenv("age_bucket_boundaries", "30,50").split(","))
)
layer = Discretization(bins=age_boundaries)
```
Further, we can provide an `analyzer_utils` in `elasticdl_preprocessing` to get the statistics from environment variables like:
```python
import os
def get_mean(feature_name, default_value):
    env_name = feature_name + "_mean"
    mean = os.getenv(env_name, None)
    if mean is None:
        return default_value
    else:
        return float(mean)

def get_stddev(feature_name, default_value):
    env_name = feature_name + "_stddev"
    std_dev = os.getenv(env_name, None)
    if std_dev is None:
        return default_value
    else:
        return float(std_dev)

def get_bucket_boundaries(feature_name, default_value):
    env_name = feature_name + "_bkt_boundaries"
    boundaries = os.getenv(env_name, None)
    if boundaries is None:
        return default_value
    else:
        return list(map(float, boundaries.split(",")))

def get_distinct_count(feature_name, default_value):
    env_name = feature_name + "_count"
    count = os.getenv(env_name, None)
    if count is None:
        return default_value
    else:
        return int(count)
```
Using the default values in `analyzer_utils`, users can debug the model without analysis.

So, we can define the preprocessing layers like:
```python
import os
from elasticdl_preprocessing.layers import Discretization, Hashing
from elasticdl_preprocessing import analyzer_utils

discretize_layer = Discretization(
    bins=analyzer_utils.get_bucket_boundaries(
        feature_name="age", default_value=[10, 30])
)
hash_layer = Hashing(
    num_bins=analyzer_utils.get_distinct_count(
        feature_name="martial", default_value=100
    )
)
```

For the values in the vocabulary table, we cannot save the vocabulary into environment variables because the vocabulary size may be huge. However, we can save the vocabulary into the shared storage like glusterfs and write the path into the environment variables of a training pod. After the training job completes, we can clear the vocabulary files in the storage.
```shell
envs="education_vocab_path=/testdata/elasticdl/vocabulary/education.txt"
```

```python
def get_vocabulary(feature_name, default_value):
    env_name = feature_name + "_vocab_path"
    vocabulary_path = os.getenv(env_name, None)
    if vocabulary_path is None:
        return default_value
    else:
        return read_file(vocabulary_path)

lookup_layer = IndexLookup(
    vocabulary=get_vocabulary("education", ["Master"])
)
```
