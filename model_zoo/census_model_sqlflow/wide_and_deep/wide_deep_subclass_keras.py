import tensorflow as tf

from elasticdl_preprocessing.layers import SparseEmbedding
from elasticdl_preprocessing.layers.concatenate_with_offset import (
    ConcatenateWithOffset,
)
from elasticdl_preprocessing.layers.discretization import Discretization
from elasticdl_preprocessing.layers.hashing import Hashing
from elasticdl_preprocessing.layers.index_lookup import IndexLookup
from elasticdl_preprocessing.layers.to_sparse import ToSparse
from elasticdl_preprocessing.utils.decorators import model_input_name
from model_zoo.census_model_sqlflow.wide_and_deep.feature_configs import (
    INPUT_SCHEMAS,
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


# The model definition in model zoo
# Add this annotation `@model_input_name` to indicate that this model
# need two input tensors: `wide_embeddings` and `deep_embeddings`.
@model_input_name(["wide_embeddings", "deep_embeddings"])
class WideAndDeepClassifier(tf.keras.Model):
    def __init__(self, hidden_units=[16, 8, 4]):
        super(WideAndDeepClassifier, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(i) for i in hidden_units]

    # TODO: Add a decorator here to describe the InputSpec
    # of `inputs`. It contains two tensors in this model.
    # The decorator should be able to inject some metadata of the InputSpec
    # and model zoo can extract this information from it.
    def call(self, inputs):
        wide_input, dnn_input = inputs

        for dense_layer in self.dense_layers:
            dnn_input = dense_layer(dnn_input)

        # Output Part
        concat_input = tf.concat([wide_input, dnn_input], 1)

        logits = tf.reduce_sum(concat_input, 1, keepdims=True)
        probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))

        return {"logits": logits, "probs": probs}


# Build the input layers from the schema of the input features
def get_input_layers(input_schemas):
    input_layers = {}

    for schema_info in input_schemas:
        input_layers[schema_info.name] = tf.keras.layers.Input(
            name=schema_info.name, shape=(1,), dtype=schema_info.dtype
        )

    return input_layers


# The following code has the same logic with the `transform` function above.
# It can be generated from the parsed meta in feature_configs using code_gen.
def transform_from_code_gen(inputs):
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

    wide_embeddings_out = ConcatenateWithOffset(offsets=None)(
        [group1_embedding_wide_out, group2_embedding_wide_out]
    )

    deep_embeddings_out = ConcatenateWithOffset(offsets=None)(
        [
            group1_embedding_deep_out,
            group2_embedding_deep_out,
            group3_embedding_deep_out,
        ]
    )

    return wide_embeddings_out, deep_embeddings_out


# The entry point of the submitter program.
# This should be generated by SQLFlow.
def custom_model():
    input_layers = get_input_layers(input_schemas=INPUT_SCHEMAS)
    transformed = transform_from_code_gen(input_layers)
    main_model = WideAndDeepClassifier()
    outputs = main_model(transformed)
    return tf.keras.Model(inputs=input_layers, outputs=outputs)


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


if __name__ == "__main__":
    print(WideAndDeepClassifier._model_input_names)

    model = custom_model()
    print(model.summary())

    inputs = {
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

    output = model.call(inputs)
    print(output)
