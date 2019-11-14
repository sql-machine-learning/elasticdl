# Data Tranform Design Doc

## Motivation

Why data transform?
How to keep consistency between online and offline?
Why transform in SQLFlow?

SQLFlow extends the SQL syntax to define a end-to-end machine learning pipeline. Naturally SQLFlow syntax should be able to describe the feature transformation process.

Consistency between offline and online is the key point of data transformation. Users write the transform code once. The same logic can run in batch mode offline and in real time mode online. In this way, we can prevent the training/serving skew.  

## Transform in SQLFlow

```SQL
SELECT *
FROM census
TO TRAIN DNNClassifier
WITH model.hidden_units = [10, 20]
COLUMNS NORMALIZE(NUMERIC(age)), STANDARDIZE(NUMERIC(capital-gain)), BUCKETIZE(NUMERIC(hours-per-week), bucket_num=10)
LABEL label
```

## Design
