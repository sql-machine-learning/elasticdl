SELECT *
FROM census_income
TO TRAIN WideAndDeepClassifier
COLUMN
    EMBEDDING(CONCAT(VOCABULARIZE(workclass), BUCKETIZE(capital_gain, num_buckets=5), BUCKETIZE(capital_loss, num_buckets=5), BUCKTIZE(hours_per_week, num_buckets=6)) AS group_1, 8),
    EMBEDDING(CONCAT(HASH(education), HASH(occupation), VOCABULARIZE(martial_status), VOCABULARIZE(relationship)) AS group_2, 8),
    EMBEDDING(CONCAT(BUCKETIZE(age, num_buckets=5), HASH(native_country), VOCABULARIZE(race), VOCABULARIZE(sex)) AS group_3, 8)
    FOR deep_embeddings
COLUMN
    EMBEDDING(group1, 1),
    EMBEDDING(group2, 1)
    FOR wide_embeddings
LABEL label
