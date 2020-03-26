import itertools

import tensorflow as tf

from model_zoo.census_model_sqlflow.wide_and_deep.transform_ops import (
    Array,
    Bucketize,
    Concat,
    Embedding,
    Hash,
    SchemaInfo,
    Vocabularize,
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

education_hash = Hash(
    "education_hash", "education", "education_hash", hash_bucket_size=30,
)
occupation_hash = Hash(
    "occupation_hash", "occupation", "occupation_hash", hash_bucket_size=30,
)
native_country_hash = Hash(
    "native_country_hash",
    "native_country",
    "native_country_hash",
    hash_bucket_size=100,
)

workclass_lookup = Vocabularize(
    "workclass_lookup",
    "workclass",
    "workclass_lookup",
    vocabulary_list=WORK_CLASS_VOCABULARY,
)
marital_status_lookup = Vocabularize(
    "marital_status_lookup",
    "marital_status",
    "marital_status_lookup",
    vocabulary_list=MARITAL_STATUS_VOCABULARY,
)
relationship_lookup = Vocabularize(
    "relationship_lookup",
    "relationship",
    "relationship_lookup",
    vocabulary_list=RELATION_SHIP_VOCABULARY,
)
race_lookup = Vocabularize(
    "race_lookup", "race", "race_lookup", vocabulary_list=RACE_VOCABULARY,
)
sex_lookup = Vocabularize(
    "sex_lookup", "sex", "sex_lookup", vocabulary_list=SEX_VOCABULARY,
)

age_bucketize = Bucketize(
    "age_bucketize", "age", "age_bucketize", boundaries=AGE_BOUNDARIES,
)
capital_gain_bucketize = Bucketize(
    "capital_gain_bucketize",
    "capital_gain",
    "capital_gain_bucketize",
    boundaries=CAPITAL_GAIN_BOUNDARIES,
)
capital_loss_bucketize = Bucketize(
    "capital_loss_bucketize",
    "capital_loss",
    "capital_loss_bucketize",
    boundaries=CAPITAL_LOSS_BOUNDARIES,
)
hours_per_week_bucketize = Bucketize(
    "hours_per_week_bucketize",
    "hours_per_week",
    "hours_per_week_bucketize",
    boundaries=HOURS_BOUNDARIES,
)


def _get_id_offsets_from_dependency_bucket_num(num_buckets):
    return list(itertools.accumulate([0] + num_buckets[:-1]))


group1 = Concat(
    "group1",
    [
        "workclass_lookup",
        "hours_per_week_bucketize",
        "capital_gain_bucketize",
        "capital_loss_bucketize",
    ],
    "group1",
    id_offsets=_get_id_offsets_from_dependency_bucket_num([8, 7, 6, 6]),
)
group2 = Concat(
    "group2",
    [
        "education_hash",
        "marital_status_lookup",
        "relationship_lookup",
        "occupation_hash",
    ],
    "group2",
    id_offsets=_get_id_offsets_from_dependency_bucket_num([30, 7, 6, 30]),
)
group3 = Concat(
    "group3",
    ["age_bucketize", "sex_lookup", "race_lookup", "native_country_hash"],
    "group3",
    id_offsets=_get_id_offsets_from_dependency_bucket_num([6, 2, 5, 100]),
)

group1_embedding_wide = Embedding(
    "group1_embedding_wide",
    "group1",
    "group1_embedding_wide",
    input_dim=sum([8, 7, 6, 6]),
    output_dim=1,
)
group2_embedding_wide = Embedding(
    "group2_embedding_wide",
    "group2",
    "group2_embedding_wide",
    input_dim=sum([30, 7, 6, 30]),
    output_dim=1,
)

group1_embedding_deep = Embedding(
    "group1_embedding_deep",
    "group1",
    "group1_embedding_deep",
    input_dim=sum([8, 7, 6, 6]),
    output_dim=8,
)
group2_embedding_deep = Embedding(
    "group2_embedding_deep",
    "group2",
    "group2_embedding_deep",
    input_dim=sum([30, 7, 6, 30]),
    output_dim=8,
)
group3_embedding_deep = Embedding(
    "group3_embedding_deep",
    "group3",
    "group3_embedding_deep",
    input_dim=sum([6, 2, 5, 100]),
    output_dim=8,
)

wide_embeddings = Array(
    "wide_embeddings",
    ["group1_embedding_wide", "group2_embedding_wide"],
    "wide_embeddings",
)

deep_embeddings = Array(
    "deep_embeddings",
    [
        "group1_embedding_deep",
        "group2_embedding_deep",
        "group3_embedding_deep",
    ],
    "deep_embeddings",
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
