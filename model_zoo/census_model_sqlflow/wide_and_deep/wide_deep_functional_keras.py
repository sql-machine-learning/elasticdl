# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from elasticdl.python.elasticdl.callbacks import LearningRateScheduler
from elasticdl_preprocessing.layers import SparseEmbedding
from elasticdl_preprocessing.layers.concatenate_with_offset import (
    ConcatenateWithOffset,
)
from elasticdl_preprocessing.layers.discretization import Discretization
from elasticdl_preprocessing.layers.hashing import Hashing
from elasticdl_preprocessing.layers.index_lookup import IndexLookup
from elasticdl_preprocessing.layers.to_sparse import ToSparse
from model_zoo.census_model_sqlflow.wide_and_deep.feature_configs import (
    FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY,
    INPUT_SCHEMAS,
    TRANSFORM_OUTPUTS,
    age_bucketize,
    capital_gain_bucketize,
    capital_loss_bucketize,
    education_hash,
    group1,
    group1_embedding_deep,
    group1_embedding_wide,
    group2,
    group2_embedding_deep,
    group2_embedding_wide,
    group3,
    group3_embedding_deep,
    hours_per_week_bucketize,
    marital_status_lookup,
    native_country_hash,
    occupation_hash,
    race_lookup,
    relationship_lookup,
    sex_lookup,
    workclass_lookup,
)
from model_zoo.census_model_sqlflow.wide_and_deep.transform_ops import (
    TransformOpType,
)


# The model definition from model zoo. It's functional style.
# Input Params:
#   input_layers: The input layers dict of feature inputs
#   wide_embeddings: The embedding list for the wide part
#   deep_embeddings: The embedding list for the deep part
def wide_and_deep_classifier(input_layers, wide_embeddings, deep_embeddings):
    # Wide Part
    wide = tf.keras.layers.Concatenate()(wide_embeddings)  # shape = (None, 3)

    # Deep Part
    dnn_input = tf.reshape(deep_embeddings, shape=(-1, 3 * 8))
    for i in [16, 8, 4]:
        dnn_input = tf.keras.layers.Dense(i)(dnn_input)

    # Output Part
    concat_input = tf.concat([wide, dnn_input], 1)

    logits = tf.reduce_sum(concat_input, 1, keepdims=True)
    probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))

    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="wide_deep",
    )


# Build the input layers from the schema of the input features
def get_input_layers(input_schemas):
    input_layers = {}

    for schema_info in input_schemas:
        input_layers[schema_info.name] = tf.keras.layers.Input(
            name=schema_info.name, shape=(1,), dtype=schema_info.dtype
        )

    return input_layers


# Build the transform logic from the metadata in feature_configs.py.
def transform(inputs):
    transformed = inputs.copy()

    for feature_transform_info in FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY:
        if feature_transform_info.op_type == TransformOpType.HASH:
            transformed[feature_transform_info.input] = ToSparse()(
                transformed[feature_transform_info.input]
            )
            transformed[feature_transform_info.output] = Hashing(
                feature_transform_info.hash_bucket_size
            )(transformed[feature_transform_info.input])
        elif feature_transform_info.op_type == TransformOpType.BUCKETIZE:
            transformed[feature_transform_info.input] = ToSparse()(
                transformed[feature_transform_info.input]
            )
            transformed[feature_transform_info.output] = Discretization(
                feature_transform_info.boundaries
            )(transformed[feature_transform_info.input])
        elif feature_transform_info.op_type == TransformOpType.LOOKUP:
            transformed[feature_transform_info.input] = ToSparse()(
                transformed[feature_transform_info.input]
            )
            transformed[feature_transform_info.output] = IndexLookup(
                feature_transform_info.vocabulary_list
            )(transformed[feature_transform_info.input])
        elif feature_transform_info.op_type == TransformOpType.CONCAT:
            inputs_to_concat = [
                transformed[name] for name in feature_transform_info.input
            ]
            transformed[feature_transform_info.output] = ConcatenateWithOffset(
                feature_transform_info.id_offsets
            )(inputs_to_concat)
        elif feature_transform_info.op_type == TransformOpType.EMBEDDING:
            transformed[feature_transform_info.output] = SparseEmbedding(
                input_dim=feature_transform_info.input_dim,
                output_dim=feature_transform_info.output_dim,
            )(transformed[feature_transform_info.input])
        elif feature_transform_info.op_type == TransformOpType.ARRAY:
            transformed[feature_transform_info.output] = [
                transformed[name] for name in feature_transform_info.input
            ]

    return tuple([transformed[name] for name in TRANSFORM_OUTPUTS])


# The following code has the same logic with the `transform` function above.
# It can be generated from the parsed meta in feature_configs using code_gen.
def transform_from_code_gen(source_inputs):
    inputs = source_inputs.copy()

    education_hash_out = Hashing(education_hash.hash_bucket_size)(
        ToSparse()(inputs["education"])
    )
    occupation_hash_out = Hashing(occupation_hash.hash_bucket_size)(
        ToSparse()(inputs["occupation"])
    )
    native_country_hash_out = Hashing(native_country_hash.hash_bucket_size)(
        ToSparse()(inputs["native_country"])
    )
    workclass_lookup_out = IndexLookup(workclass_lookup.vocabulary_list)(
        ToSparse()(inputs["workclass"])
    )
    marital_status_lookup_out = IndexLookup(
        marital_status_lookup.vocabulary_list
    )(ToSparse()(inputs["marital_status"]))
    relationship_lookup_out = IndexLookup(relationship_lookup.vocabulary_list)(
        ToSparse()(inputs["relationship"])
    )
    race_lookup_out = IndexLookup(race_lookup.vocabulary_list)(
        ToSparse()(inputs["race"])
    )
    sex_lookup_out = IndexLookup(sex_lookup.vocabulary_list)(
        ToSparse()(inputs["sex"])
    )
    age_bucketize_out = Discretization(age_bucketize.boundaries)(
        ToSparse()(inputs["age"])
    )
    capital_gain_bucketize_out = Discretization(
        capital_gain_bucketize.boundaries
    )(ToSparse()(inputs["capital_gain"]))
    capital_loss_bucketize_out = Discretization(
        capital_loss_bucketize.boundaries
    )(ToSparse()(inputs["capital_loss"]))
    hours_per_week_bucketize_out = Discretization(
        hours_per_week_bucketize.boundaries
    )(ToSparse()(inputs["hours_per_week"]))

    group1_out = ConcatenateWithOffset(group1.id_offsets)(
        [
            workclass_lookup_out,
            hours_per_week_bucketize_out,
            capital_gain_bucketize_out,
            capital_loss_bucketize_out,
        ]
    )
    group2_out = ConcatenateWithOffset(group2.id_offsets)(
        [
            education_hash_out,
            marital_status_lookup_out,
            relationship_lookup_out,
            occupation_hash_out,
        ]
    )
    group3_out = ConcatenateWithOffset(group3.id_offsets)(
        [
            age_bucketize_out,
            sex_lookup_out,
            race_lookup_out,
            native_country_hash_out,
        ]
    )

    group1_embedding_wide_out = SparseEmbedding(
        input_dim=group1_embedding_wide.input_dim,
        output_dim=group1_embedding_wide.output_dim,
    )(group1_out)
    group2_embedding_wide_out = SparseEmbedding(
        input_dim=group2_embedding_wide.input_dim,
        output_dim=group2_embedding_wide.output_dim,
    )(group2_out)

    group1_embedding_deep_out = SparseEmbedding(
        input_dim=group1_embedding_deep.input_dim,
        output_dim=group1_embedding_deep.output_dim,
    )(group1_out)
    group2_embedding_deep_out = SparseEmbedding(
        input_dim=group2_embedding_deep.input_dim,
        output_dim=group2_embedding_deep.output_dim,
    )(group2_out)
    group3_embedding_deep_out = SparseEmbedding(
        input_dim=group3_embedding_deep.input_dim,
        output_dim=group3_embedding_deep.output_dim,
    )(group3_out)

    wide_embeddings_out = [
        group1_embedding_wide_out,
        group2_embedding_wide_out,
    ]

    deep_embeddings_out = [
        group1_embedding_deep_out,
        group2_embedding_deep_out,
        group3_embedding_deep_out,
    ]

    return wide_embeddings_out, deep_embeddings_out


# The entry point of the submitter program
def custom_model():
    input_layers = get_input_layers(input_schemas=INPUT_SCHEMAS)
    wide_embeddings, deep_embeddings = transform_from_code_gen(input_layers)

    return wide_and_deep_classifier(
        input_layers, wide_embeddings, deep_embeddings
    )


def loss(labels, predictions):
    logits = predictions["logits"]
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(tf.reshape(labels, (-1, 1)), tf.float32),
            logits=logits,
        )
    )


def optimizer(lr=0.001):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def eval_metrics_fn():
    return {
        "logits": {
            "accuracy": lambda labels, predictions: tf.equal(
                tf.cast(tf.reshape(predictions, [-1]) > 0.5, tf.int32),
                tf.cast(tf.reshape(labels, [-1]), tf.int32),
            )
        },
        "probs": {"auc": tf.keras.metrics.AUC()},
    }


def callbacks():
    def _schedule(model_version, world_size):
        if model_version < 5000:
            return 0.0003
        elif model_version < 12000:
            return 0.0002
        else:
            return 0.0001

    return [LearningRateScheduler(_schedule)]


if __name__ == "__main__":
    model = custom_model()
    print(model.summary())

    output = model.call(
        {
            "education": tf.constant([["Bachelors"]], tf.string),
            "occupation": tf.constant([["Tech-support"]], tf.string),
            "native_country": tf.constant([["United-States"]], tf.string),
            "workclass": tf.constant([["Private"]], tf.string),
            "marital_status": tf.constant([["Separated"]], tf.string),
            "relationship": tf.constant([["Husband"]], tf.string),
            "race": tf.constant([["White"]], tf.string),
            "sex": tf.constant([["Female"]], tf.string),
            "age": tf.constant([[18]], tf.float32),
            "capital_gain": tf.constant([[100.0]], tf.float32),
            "capital_loss": tf.constant([[1.0]], tf.float32),
            "hours_per_week": tf.constant([[40]], tf.float32),
        }
    )

    print(output)
