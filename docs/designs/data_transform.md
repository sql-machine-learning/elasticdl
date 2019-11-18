# Data Tranform Design Doc

## Motivation

Data transform is an important part in an end-to-end machine learning pipeline. It processes the raw data using some operations such as standardize, bucketize and so on. The target is to make sure the data is in the right format and ready for the model training. SQLFlow extends the SQL syntax to define a ML pipeline. Naturally SQLFlow syntax should be able to describe the data transform process. In this doc, we are focusing on how to do data transform using SQLFlow.  

Data transform contains two key parts: analyzer and transformer. Analyzer scans the entire data set and calculates the statistical values such as mean, min, variance and so on. Transformer combines the statistical value and the transform function to construct the concrete transform logic. And then it transforms the data records one by one.  
[TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started) is the open source solution for data transform in TFX. Users need write a python function preprocess_fn to define the preprocess logic. SQLFlow users prefer to write SQL instead of python. It's user unfriendly to SQLFlow users if we integrate TF Transform with SQLFlow directly.  
From another point of view, SQL can naturally support statistical work just like analyzer. Feature column api can take charge of transform logic such as transformer. For dense column, we can use numeric_column and pass a user defined function *normalizer_fn* to convert the column value. For sparse column, we can use embedding_column to map the sparse value to embedding vector or use cross_column to do the feature crossing. We plan to use SQL and feature column together to do the data transform work.  

Consistency between offline and online is the key point of data transform. Users write the transform code only once. And then the same logic can run in batch mode offline and in real time mode online. In this way, we can prevent the training/serving skew. Both TF Transform and feature column can keep the consistency. The data transform logic in the training stage is built into the inference graph as the SavedModel.  

## Transform Expression in SQLFlow

We can extend the SQLFlow syntax and add **TO TRANSFORM** keyword to describe the transform process. Let's take the following SQL expression for example: **TO TRANSFORM STANDARDIZE(age) as age_std, NORMALIZE(capital_gain) as capital_gain_norm, BUCKETIZED(hours_per_week, bucket_num=10) as hours_per_week_bkt**. Standardize the column *age* to the column *age_std*, normalize the column *capital_gain* to *capital_gain_norm*, bucketize the column *hours_per_week* to 10 buckets to the column *hours_per_week_bkt*. The output of transform will be passed to the **COLUMN** expression.  
We add some built-in transform API and users can use them directly in the TRANSFORM expression. The Api set contains NORMALIZE, STANDARDIZE, BUCKETIZED, LOG and more to be added in the future.  

```SQL
SELECT *
FROM census
TO TRANSFORM STANDARDIZE(age) as age_std, NORMALIZE(capital_gain) as capital_gain_norm, BUCKETIZED(hours_per_week, bucket_num=10) as hours_per_week_bkt
TO TRAIN DNNClassifier
WITH model.hidden_units = [10, 20]
COLUMNS NUMERIC(age_std), NUMERIC(capital_gain_norm), EMBEDDING(hours_per_week_bkt, dim=128)
LABEL label
```

## Design

Data transform contains two stages: analyze and transform. In our design, we will do the analyze using SQL as the first step, and generate the feature column definition as the second step. The feature column contains the transform logic and execute along with the model training process.  
We choose to convert the **TRANSFORM** expression into two steps in work flow described by [Couler](https://github.com/sql-machine-learning/sqlflow/blob/develop/python/couler/README.md): analyze and feature column generation. Couler is a programming language for describing workflows. Its compiler translates a workflow represented by a Python program into an [Argo](https://argoproj.github.io/) YAML file. Couler The output of feature column generation will be passed to the next model training step.  
![data_transform_pipeline](../images/data_transform_pipeline.png)

Let's take STANDARDIZE(age) for example, the following figure describe how the data transform pipeline works in detail.  

![transform_steps](../images/transform_steps.png)

The **Analyze Step** and **Feature Column Generation Step** are two couler steps. Analyze Result and Generated Feature Column Definition Result are the output of these two couler steps.  

A transform api contains two members: analyzers and feature column template. Analyzer is the statistical operation which needs run at first to complement the whole transform logic. After completing all the statistical operation, we will format the feature column template with variable name and statistical values to get the integral feature column definition for the transform logic.  
The generated feature column definitions will be passed to the next couler step: model training. We combine them with the COLUMN expression to generated the final feature column definitions and then pass to the model. Let's take **COLUMNS NUMERIC(age_std)** for example, the final definition will be **numeric_column('age', normalizer_fn=lambda x: x - 18.0 / 6.0)**  
