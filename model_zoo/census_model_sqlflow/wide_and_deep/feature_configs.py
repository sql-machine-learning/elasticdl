import tensorflow as tf

from model_zoo.census_model_sqlflow.wide_and_deep.feature_info_utils import (
    FeatureTransformInfo,
    SchemaInfo,
    TransformOp,
)

# The following objects can be generated from the meta parsed from
# the `COLUMN` clause in SQLFlow statement and the analysis result.

WORK_CLASS_VOCABULARY = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
]

MARITAL_STATUS_VOCABULARY = [
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]

RELATION_SHIP_VOCABULARY = [
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]

RACE_VOCABULARY = [
    "White",
    "Asian-Pac-Islander",
    "Amer-Indian-Eskimo",
    "Other",
    "Black",
]

SEX_VOCABULARY = ["Female", "Male"]

AGE_BOUNDARIES = [0, 20, 40, 60, 80]
CAPITAL_GAIN_BOUNDARIES = [6000, 6500, 7000, 7500, 8000]
CAPITAL_LOSS_BOUNDARIES = [2000, 2500, 3000, 3500, 4000]
HOURS_BOUNDARIES = [10, 20, 30, 40, 50, 60]

education_hash = FeatureTransformInfo(
    "education_hash",
    "education",
    "education_hash",
    TransformOp.HASH,
    tf.string,
    30,
)
occupation_hash = FeatureTransformInfo(
    "occupation_hash",
    "occupation",
    "occupation_hash",
    TransformOp.HASH,
    tf.string,
    30,
)
native_country_hash = FeatureTransformInfo(
    "native_country_hash",
    "native_country",
    "native_country_hash",
    TransformOp.HASH,
    tf.string,
    100,
)

workclass_lookup = FeatureTransformInfo(
    "workclass_lookup",
    "workclass",
    "workclass_lookup",
    TransformOp.LOOKUP,
    tf.string,
    WORK_CLASS_VOCABULARY,
)
marital_status_lookup = FeatureTransformInfo(
    "marital_status_lookup",
    "marital_status",
    "marital_status_lookup",
    TransformOp.LOOKUP,
    tf.string,
    MARITAL_STATUS_VOCABULARY,
)
relationship_lookup = FeatureTransformInfo(
    "relationship_lookup",
    "relationship",
    "relationship_lookup",
    TransformOp.LOOKUP,
    tf.string,
    RELATION_SHIP_VOCABULARY,
)
race_lookup = FeatureTransformInfo(
    "race_lookup",
    "race",
    "race_lookup",
    TransformOp.LOOKUP,
    tf.string,
    RACE_VOCABULARY,
)
sex_lookup = FeatureTransformInfo(
    "sex_lookup",
    "sex",
    "sex_lookup",
    TransformOp.LOOKUP,
    tf.string,
    SEX_VOCABULARY,
)

age_bucketize = FeatureTransformInfo(
    "age_bucketize",
    "age",
    "age_bucketize",
    TransformOp.BUCKETIZE,
    tf.float32,
    AGE_BOUNDARIES,
)
capital_gain_bucketize = FeatureTransformInfo(
    "capital_gain_bucketize",
    "capital_gain",
    "capital_gain_bucketize",
    TransformOp.BUCKETIZE,
    tf.float32,
    CAPITAL_GAIN_BOUNDARIES,
)
capital_loss_bucketize = FeatureTransformInfo(
    "capital_loss_bucketize",
    "capital_loss",
    "capital_loss_bucketize",
    TransformOp.BUCKETIZE,
    tf.float32,
    CAPITAL_LOSS_BOUNDARIES,
)
hours_per_week_bucketize = FeatureTransformInfo(
    "hours_per_week_bucketize",
    "hours_per_week",
    "hours_per_week_bucketize",
    TransformOp.BUCKETIZE,
    tf.float32,
    HOURS_BOUNDARIES,
)

group1 = FeatureTransformInfo(
    "group1",
    [
        "workclass_lookup",
        "hours_per_week_bucketize",
        "capital_gain_bucketize",
        "capital_loss_bucketize",
    ],
    "group1",
    TransformOp.GROUP,
    None,
    [8, 7, 6, 5],
)
group2 = FeatureTransformInfo(
    "group2",
    [
        "education_hash",
        "marital_status_lookup",
        "relationship_lookup",
        "occupation_hash",
    ],
    "group2",
    TransformOp.GROUP,
    None,
    [30, 7, 6, 30],
)
group3 = FeatureTransformInfo(
    "group3",
    ["age_bucketize", "sex_lookup", "race_lookup", "native_country_hash"],
    "group3",
    TransformOp.GROUP,
    None,
    [6, 2, 5, 100],
)

group1_embedding_wide = FeatureTransformInfo(
    "group1_embedding_wide",
    "group1",
    "group1_embedding_wide",
    TransformOp.EMBEDDING,
    tf.int32,
    (300, 1),
)
group2_embedding_wide = FeatureTransformInfo(
    "group2_embedding_wide",
    "group2",
    "group2_embedding_wide",
    TransformOp.EMBEDDING,
    tf.int32,
    (1000, 1),
)

group1_embedding_deep = FeatureTransformInfo(
    "group1_embedding_deep",
    "group1",
    "group1_embedding_deep",
    TransformOp.EMBEDDING,
    tf.int32,
    (300, 8),
)
group2_embedding_deep = FeatureTransformInfo(
    "group2_embedding_deep",
    "group2",
    "group2_embedding_deep",
    TransformOp.EMBEDDING,
    tf.int32,
    (1000, 8),
)
group3_embedding_deep = FeatureTransformInfo(
    "group3_embedding_deep",
    "group3",
    "group3_embedding_deep",
    TransformOp.EMBEDDING,
    tf.int32,
    (512, 8),
)

wide_embeddings = FeatureTransformInfo(
    "wide_embeddings",
    ["group1_embedding_wide", "group2_embedding_wide"],
    "wide_embeddings",
    TransformOp.ARRAY,
    None,
    None,
)

deep_embeddings = FeatureTransformInfo(
    "deep_embeddings",
    [
        "group1_embedding_deep",
        "group2_embedding_deep",
        "group3_embedding_deep",
    ],
    "deep_embeddings",
    TransformOp.ARRAY,
    None,
    None,
)

TRANSFORM_OUTPUTS = ["wide_embeddings", "deep_embeddings"]

# Get this execution order by topological sort
FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY = [
    education_hash,
    occupation_hash,
    native_country_hash,
    workclass_lookup,
    marital_status_lookup,
    relationship_lookup,
    race_lookup,
    sex_lookup,
    age_bucketize,
    capital_gain_bucketize,
    capital_loss_bucketize,
    hours_per_week_bucketize,
    group1,
    group2,
    group3,
    group1_embedding_wide,
    group2_embedding_wide,
    group1_embedding_deep,
    group2_embedding_deep,
    group3_embedding_deep,
    wide_embeddings,
    deep_embeddings,
]

# The schema information can be gotten from the
# source table in SQLFlow statement
# For example, {table_name} from
# SELECT * FROM {table_name}
INPUT_SCHEMAS = [
    SchemaInfo("education", tf.string),
    SchemaInfo("occupation", tf.string),
    SchemaInfo("native_country", tf.string),
    SchemaInfo("workclass", tf.string),
    SchemaInfo("marital_status", tf.string),
    SchemaInfo("relationship", tf.string),
    SchemaInfo("race", tf.string),
    SchemaInfo("sex", tf.string),
    SchemaInfo("age", tf.float32),
    SchemaInfo("capital_gain", tf.float32),
    SchemaInfo("capital_loss", tf.float32),
    SchemaInfo("hours_per_week", tf.float32),
]
