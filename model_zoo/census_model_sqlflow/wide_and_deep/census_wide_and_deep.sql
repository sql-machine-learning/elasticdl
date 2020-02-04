SELECT *
FROM census_income
TO TRAIN WideAndDeepClassifier
COLUMN (
    SET GROUP(APPLY_VOCAB(workclass), BUCKETIZE(capital_gain, bucket_num=5), BUCKETIZE(capital_loss, bucket_num=5), BUCKTIZE(hours_per_week, bucket_num=6)) AS group_1,
    SET GROUP(HASH(education), HASH(occupation), APPLY_VOCAB(martial_status), APPLY_VOCAB(relationship)) AS group_2,
    SET GROUP(BUCKETIZE(age, bucket_num=5), HASH(native_country), APPLY_VOCAB(race), APPLY_VOCAB(sex)) AS group_3,

    [EMBEDDING(group1, 1), EMBEDDING(group2, 1)] AS wide_embeddings
    [EMBEDDING(group1, 8), EMBEDDING(group2, 8), EMBEDDING(group3, 8)] AS deep_embeddings
)
LABEL label