# Data Tranform Design Doc

## Motivation

Data transform is an important part in an end-to-end machine learning pipeline. It processes the raw data using operations such as standardization, bucketization and so on. The target is to make sure the data is in the right format and ready for the model training and inference. [SQLFlow](https://github.com/sql-machine-learning/sqlflow) is a bridge that connects a SQL engine and machine learning toolkits. It extends the SQL syntax to define a ML pipeline. Naturally SQLFlow syntax should be able to describe the data transform process. In this doc, we are focusing on how to do data transform using SQLFlow.  

Data transform contains two key parts: analyzer and transformer. Analyzer scans the entire data set and calculates the statistical values such as mean, min, variance and so on. Transformer combines the statistical value and the transform function to construct the concrete transform logic. And then it transforms the data records one by one.  
[TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started) is the open source solution for data transform in [TensorFlow Extended](https://www.tensorflow.org/tfx/guide). Users need write a python function 'preprocess_fn' to define the preprocess logic. SQLFlow users prefer to write SQL instead of python. It's user unfriendly to SQLFlow users if we integrate TF Transform with SQLFlow directly.  
From another point of view, SQL can naturally supports statistical work just like the analyzer. [Feature column API](https://tensorflow.google.cn/api_docs/python/tf/feature_column) can take charge of transform logic as transformer. For dense column, we can use numeric_column and pass a user defined function *normalizer_fn* to convert the column value. For sparse column, we can use embedding_column to map the sparse value to embedding vector or use cross_column to do the feature crossing. We plan to use SQL and feature column together to do the data transform work.  

Consistency between offline and online is the key point of data transform. Users write the transform code only once. And then the same logic can run in batch mode for training and in real time mode for serving. In this way, we can prevent the training/serving skew. Both TF Transform and feature column can keep the consistency. The data transform logic in the training stage is built into the inference graph as the SavedModel.  

## Transform Expression in SQLFlow

We can extend the SQLFlow syntax and enrich the COLUMN expression. We can add the built-in transform API call in it to describe the transform process. Let's take the following SQL expression for example. It trains a model to classify someone's income level using the [census income dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income). The transform expression is **COLUMNS NUMERIC(STANDARDIZE(age)), NUMERIC(NORMALIZE(capital_gain)), EMBEDDING(BUCKETIZED(hours_per_week, bucket_num=10), dim=128)**. It will standardize the column *age*, normalize the column *capital_gain*, bucketize the column *hours_per_week* to 10 buckets and then map it to embedding value.  
We will implement some built-in transform API. The API set contains NORMALIZE, STANDARDIZE, BUCKETIZED, LOG and more to be added in the future.  

```SQL
SELECT *
FROM census_income
TO TRAIN DNNClassifier
WITH model.hidden_units = [10, 20]
COLUMNS NUMERIC(STANDARDIZE(age)), NUMERIC(NORMALIZE(capital_gain)), EMBEDDING(BUCKETIZED(hours_per_week, bucket_num=10), dim=128)
LABEL label
```

## Design

Data transform contains two stages: analyze and transform. In our design, we will do the analysis using SQL as the first step, and generate the feature column definition as the second step. The feature column contains the transform logic and executes along with the model training process.  
We choose to convert the transform expression into two steps of the work flow described by [Couler](https://github.com/sql-machine-learning/sqlflow/blob/develop/python/couler/README.md): analyze and feature column generation. Couler is a programming language for describing workflows. Its compiler translates a workflow represented by a Python program into an [Argo](https://argoproj.github.io/) YAML file. The output of feature column generation will be passed to the next model training step.  
![data_transform_pipeline](../images/data_transform_pipeline.png)

Let's take STANDARDIZE(age) for example, the following figure describes how the data transform pipeline works in detail.  

![transform_steps](../images/transform_steps.png)

A transform API contains two members: analyzers and feature column template. Analyzer is the statistical operation which needs run at first to complement the whole transform logic. Feature column template is used to build the concrete feature column definition.  

The **Analyze Step** and **Feature Column Generation Step** are two couler steps. Analyze Result and Generated Feature Column Definition Result are the output of these two couler steps.  
In the Analyze step, we will parse the TRANSFORM expression and collect the statistics requirements. It's a dictionary of {statistic_variable_name} -> tuple({analyze_operation_name}, {column_name_in_source_table}). The SQL generator will generate the analyze SQL expression containing built-in aggregate functions from this dictionary for different data sources such as [Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF), [MaxCompute](https://help.aliyun.com/document_detail/48975.html) and so on. After executing the SQL, the statistical result will be writen to the standard output of the container.  
In the feature column generation step, we will format the feature column template with the variable name and the statistical values to get the integral feature column definition for the transform logic.  
The generated feature column definitions will be passed to the next couler step: model training. We combine them with the COLUMN expression to generated the final feature column definitions and then pass to the model. Let's take **COLUMNS NUMERIC(STANDARDIZE(age))** for example, the final definition will be **numeric_column('age', normalizer_fn=lambda x: x - 18.0 / 6.0)**  

We plan to implement the following common used transform APIs at the first step. And we will add more according to further requirements.  
|            Name             |                      Feature Column Template                                   |      Analyzer      |
|:---------------------------:|:------------------------------------------------------------------------------:|:------------------:|
|       STANDARDIZE(x)        | numeric_column({var_name}, normalizer_fn=lambda x : x - {mean} / {std})        |    MEAN, STDDEV    |
|        NORMALIZE(x)         | numeric_column({var_name}, normalizer_fn=lambda x : x - {min} / {max} - {min}) |      MAX, MIN      |
|           LOG(x)            | numeric_column({var_name}, normalizer_fn=lambda x : tf.math.log(x))            |         N/A        |
| BUCKETIZED(x, bucket_num=y) | bucketized_column({var_name}, boundaries={percentiles})                        |     PERCENTILE     |

## Further Consideration

In the design above, we generated the concrete feature column definition for data transformation in the Transform stage. The actual transform logic on the raw data executes along with the model training process. Based on this design, we can further consider transforming the raw data and writing the transformed result into a new table in the stage.  
After analyzing the data, we construct the TF graph for transform instead of feature column definition and export it to SavedModel. And then we submit a data processing job to transform the raw data by executing UDF with the SavedModel. The whole process is also matched with the TFX pipeline.  
This solution can bring the following benifits:

1. We can reuse the transformed data in the temporary table to execute multipe model training run for different hyperparameter combinations and all the epochs. Data transformation is only executed once.
2. We can support more flexible transform logic such as inter column calculation. Feature column has some limit on the inter column calculation. Please check the [Wiki](https://github.com/sql-machine-learning/elasticdl/wiki/ElasticDL-TF-Transform-Explore#inter-columns-calculation) for more details.

We need figure out the following points for this further solution:

1. Model Export: Upgrade keras API to support exporting the transform logic and the model definition together to SavedModel for inference. [Issue](https://github.com/tensorflow/transform/issues/150)
2. Transform Execution: We will transform the data records one by one using the transform logic in the SavedModel format and then write to a new table. We also need write a Jar, it packages the TensorFlow library, loads the SavedModel into memory and processes the input data. And then we register it as UDF in Hive or MaxCompute and use it to transform the data.  
