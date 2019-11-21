# Data Tranform Design Doc

## Motivation

Data transform is an important part in an end-to-end machine learning pipeline. It processes the raw data using operations such as standardization, bucketization and so on. The target is to make sure the data is in the right format and ready for the model training and inference. [SQLFlow](https://github.com/sql-machine-learning/sqlflow) is a bridge that connects a SQL engine and machine learning toolkits. It extends the SQL syntax to define a ML pipeline. Naturally SQLFlow syntax should be able to describe the data transform process. In this doc, we are focusing on how to do data transform using SQLFlow.  

Data transform contains two key parts: analyzer and transformer. Analyzer scans the entire data set and calculates the statistical values such as mean, min, variance and so on. Transformer combines the statistical value and the transform function to construct the concrete transform logic. And then it transforms the data records one by one.  
[TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started) is the open source solution for data transform in [TensorFlow Extended](https://www.tensorflow.org/tfx/guide). Users need write a python function 'preprocess_fn' to define the preprocess logic. SQLFlow users prefer to write SQL instead of python. It's user unfriendly to SQLFlow users if we integrate TF Transform with SQLFlow directly.  
From another point of view, SQL can naturally support statistical work just like the analyzer. [Feature column API](https://tensorflow.google.cn/api_docs/python/tf/feature_column) can take charge of transform logic such as transformer. For dense column, we can use numeric_column and pass a user defined function *normalizer_fn* to convert the column value. For sparse column, we can use embedding_column to map the sparse value to embedding vector or use cross_column to do the feature crossing. We plan to use SQL and feature column together to do the data transform work.  

Consistency between offline and online is the key point of data transform. Users write the transform code only once. And then the same logic can run in batch mode for training and in real time mode for serving. In this way, we can prevent the training/serving skew. Both TF Transform and feature column can keep the consistency. The data transform logic in the training stage is built into the inference graph as the SavedModel.  

## Transform Expression in SQLFlow

We can extend the SQLFlow syntax and add **TO TRANSFORM** keyword to describe the transform process. Let's take the following SQL expression for example. It trains a model to classify someone's income level using the [census income dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income). The transform expression is **TO TRANSFORM STANDARDIZE(age) as age_std, NORMALIZE(capital_gain) as capital_gain_norm, BUCKETIZED(hours_per_week, bucket_num=10) as hours_per_week_bkt**. It will standardize the column *age* to the column *age_std*, normalize the column *capital_gain* to *capital_gain_norm*, bucketize the column *hours_per_week* to 10 buckets to the column *hours_per_week_bkt*. The output of transform will be passed to the **COLUMN** expression.  
We add some built-in transform API and users can use them directly in the TRANSFORM expression. The API set contains NORMALIZE, STANDARDIZE, BUCKETIZED, LOG and more to be added in the future.  

```SQL
SELECT *
FROM census_income
TO TRANSFORM STANDARDIZE(age) as age_std, NORMALIZE(capital_gain) as capital_gain_norm, BUCKETIZED(hours_per_week, bucket_num=10) as hours_per_week_bkt
TO TRAIN DNNClassifier
WITH model.hidden_units = [10, 20]
COLUMNS NUMERIC(age_std), NUMERIC(capital_gain_norm), EMBEDDING(hours_per_week_bkt, dim=128)
LABEL label
```

## Design

Data transform contains two stages: analyze and transform. In our design, we will do the analysis using SQL as the first step, and generate the feature column definition as the second step. The feature column contains the transform logic and executes along with the model training process.  
We choose to convert the **TRANSFORM** expression into two steps of the work flow described by [Couler](https://github.com/sql-machine-learning/sqlflow/blob/develop/python/couler/README.md): analyze and feature column generation. Couler is a programming language for describing workflows. Its compiler translates a workflow represented by a Python program into an [Argo](https://argoproj.github.io/) YAML file. The output of feature column generation will be passed to the next model training step.  
![data_transform_pipeline](../images/data_transform_pipeline.png)

Let's take STANDARDIZE(age) for example, the following figure describes how the data transform pipeline works in detail.  

![transform_steps](../images/transform_steps.png)

A transform API contains two members: analyzers and feature column template. Analyzer is the statistical operation which needs run at first to complement the whole transform logic. Feature column template is used to build the concrete feature column definition.  

The **Analyze Step** and **Feature Column Generation Step** are two couler steps. Analyze Result and Generated Feature Column Definition Result are the output of these two couler steps.  
In the Analyze step, we will parse the TRANSFORM expression and collect the statistics requirements. It's a dictionary of {statistic_variable_name} -> tuple({analyze_operation_name}, {column_name_in_source_table}). The SQL generator will generate the analyze SQL expression containing built-in aggregate functions from this dictionary for different data sources such as [Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF), [MaxCompute](https://help.aliyun.com/document_detail/48975.html) and so on. After executing the SQL, the statistical result will be writen to the standard output of the container.  
In the feature column generation step, we will format the feature column template with the variable name and the statistical values to get the integral feature column definition for the transform logic.  
The generated feature column definitions will be passed to the next couler step: model training. We combine them with the COLUMN expression to generated the final feature column definitions and then pass to the model. Let's take **COLUMNS NUMERIC(age_std)** for example, the final definition will be **numeric_column('age', normalizer_fn=lambda x: x - 18.0 / 6.0)**  

We plan to implement the following common used transform APIs at the first step. And we will add more according to further requirements.  
|            Name             |                      Feature Column Template                                   |     Analyzers      |
|:---------------------------:|:------------------------------------------------------------------------------:|:------------------:|
|       STANDARDIZE(x)        | numeric_column({var_name}, normalizer_fn=lambda x : x - {mean} / {std})        |    MEAN, STDDEV    |
|        NORMALIZE(x)         | numeric_column({var_name}, normalizer_fn=lambda x : x - {min} / {max} - {min}) |      MAX, MIN      |
|           LOG(x)            | numeric_column({var_name}, normalizer_fn=lambda x : tf.math.log(x))            |         N/A        |
| BUCKETIZE(x, bucket_num=10) | bucketized_column({var_name}, boundaries={percentile})                         |     PERCENTILE     |
