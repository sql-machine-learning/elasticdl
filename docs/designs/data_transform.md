# Data Tranform Design Doc

## Motivation

Data transform is an important part in an end-to-end machine learning pipeline. It processes the raw data using standardize, bucketize and so on. The target is to make sure the data is in the right format and ready for the model training. SQLFlow extends the SQL syntax to define a ML pipeline. Naturally SQLFlow syntax should be able to describe the data transformation process. In this doc, we are focusing on how to do data transformation using SQLFlow.  

Data transformation contains two key parts: analyzer and transformer. Analyzer scans the entire data set and calculates the statistcal values such as mean, min, variance and so on. Transformer combine the statistical value and transform function to construct the concrete transform logic. And then it transform the data records one by one.  
[TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started) is the open source solution for data transformation in TFX. Users need write a python function preprocess_fn to define the preprocess logic. SQLFlow users prefer to write SQL instead of python. It's user unfriendly to SQLFlow users if integrate TF Transform directly.  
SQL can naturally support statistical work just like analyzer. Feature column api can take charge of transform logic such as transformer. For dense column, we can use numeric_column and pass a user defined function *normalizer_fn* to convert the column value. For sparse column, we can use embedding_column to map the sparse value to embedding vector or use cross_column to do the feature crossing. We plan to use SQL and feature column api to complete the data transformation.  

Consistency between offline and online is the key point of data transformation. Users write the transform code only once. And then the same logic can run in batch mode offline and in real time mode online. In this way, we can prevent the training/serving skew. Both TF Transform and feature column can keep the consistency. The data transform logic on the training stage are built into the inference graph as the SavedModel.  

## Transform Expression in SQLFlow

We can extend the SQLFlow syntax and add **TO TRANSFORM** keyword to describe the transform process. Let's take the following SQL expression for example: **TO TRANSFORM NORMALIZE(age) as age_norm, STANDARDIZE(capital_gain) as capital_gain_std, BUCKETIZED(hours_per_week, bucket_num=10) as hours_per_week_bkt**. Normalize the column 'age' to the column 'age_norm', standardize the column 'capital_gain' to 'capital_gain_std', bucketize the column 'hours_per_week' to 10 buckets to the column 'hours_per_week_bkt'. The output of transform will be passed to the **COLUMN** expression.

```SQL
SELECT *
FROM census
TO TRANSFORM NORMALIZE(age) as age_norm, STANDARDIZE(capital_gain) as capital_gain_std, BUCKETIZED(hours_per_week, bucket_num=10) as hours_per_week_bkt
TO TRAIN DNNClassifier
WITH model.hidden_units = [10, 20]
COLUMNS NUMERIC(age_norm), NUMERIC(capital_gain_std), EMBEDDING(hours_per_week_bkt, dim=128)
LABEL label
```

## Design


