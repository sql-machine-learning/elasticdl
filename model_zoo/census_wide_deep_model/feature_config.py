import tensorflow as tf

from model_zoo.census_wide_deep_model.feature_info_util import (
    FeatureInfo,
    TransformOp,
)

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

education = FeatureInfo("education", TransformOp.HASH, tf.string, 30)
occupation = FeatureInfo("occupation", TransformOp.HASH, tf.string, 30)
native_country = FeatureInfo(
    "native-country", TransformOp.HASH, tf.string, 100
)

workclass = FeatureInfo(
    "workclass", TransformOp.LOOKUP, tf.string, WORK_CLASS_VOCABULARY
)
marital_status = FeatureInfo(
    "marital-status", TransformOp.LOOKUP, tf.string, MARITAL_STATUS_VOCABULARY
)
relationship = FeatureInfo(
    "relationship", TransformOp.LOOKUP, tf.string, RELATION_SHIP_VOCABULARY
)
race = FeatureInfo("race", TransformOp.LOOKUP, tf.string, RACE_VOCABULARY)
sex = FeatureInfo("sex", TransformOp.LOOKUP, tf.string, SEX_VOCABULARY)

age = FeatureInfo("age", TransformOp.BUCKETIZE, tf.float32, AGE_BOUNDARIES)
capital_gain = FeatureInfo(
    "capital-gain", TransformOp.BUCKETIZE, tf.float32, CAPITAL_GAIN_BOUNDARIES
)
capital_loss = FeatureInfo(
    "capital-loss", TransformOp.BUCKETIZE, tf.float32, CAPITAL_LOSS_BOUNDARIES
)
hours_per_week = FeatureInfo(
    "hours-per-week", TransformOp.BUCKETIZE, tf.float32, HOURS_BOUNDARIES
)

FEATURE_GROUPS = {
    "group1": [workclass, hours_per_week, capital_gain, capital_loss],
    "group2": [education, marital_status, relationship, occupation],
    "group3": [age, sex, race, native_country],
}

MODEL_INPUTS = {
    "wide": ["group1", "group2"],
    "deep": ["group1", "group2", "group3"],
}

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
