# Preprocess Inputs using ElasticDL Preprocessing Layers

This document is a tutorial for ElasticDL preprocessing layers.

## ElasticDL Preprocessing Layers
ElasticDL preprocessing layers is a library based TensorFlow for preprocessing input data for TensorFlow. ElasticDL Preprocessing layers allow users to include data preprocessing directly in their Keras model.
For example, using ElasticDL preprocessing layers you could:

* Normalize an input value by using the mean and standard deviation.
* Convert strings to integers by lookuping a vocabulary or hashing.
* Convert floats to integers by assigning them to buckets and rounding.


### Transform Numeric Inputs
For numeric inputs, ElasticDL provides [Normalizer](#normalizer-layer) to scale the numeric data to a range, [Discretization](#discretization-layer), [LogRound](#loground-layer) and [RoundIdentity](#roundidentity-layer) to map the numeric data to integer values.

#### Normalizer Layer
The `Normalizer` layer is to normalize numeric values by (x-subtractor)/divisor. For example, we can set the subtractor to the minimum and divisor to the range size to 
implement normalization.
```python
minimum = 3.0
maximum = 7.0
layer = Normalizer(subtractor=minimum, divisor=(maximum - minimum))
input_data = tf.constant([[3.0], [5.0], [7.0]])
result = layer(input_data)
```
If we want to implement standardization, we can set the subtractor and divisor to the mean and standard deviation.

#### Discretization Layer
The `Discretization` layer is to bucketize numeric data into discrete ranges according to boundaries and return integer values. For example, if the numeric
data is `[19, 42, 55]` and boundaries are `[30, 45]`, then outputs are `[0, 1, 2]`.

```python
age_values = tf.constant([[34, 45, 23, 67], [15, 37, 52, 47]])
bins = [20, 30, 40, 50]
layer = Discretization(bins=bins)
result = layer(age_values)
```
The outputs are `[[2, 3, 1, 4], [0, 2, 4, 3]]`

#### LogRound Layer
The `LogRound` layer is a special case of `Discretization` with fixed boundaries. It casts a numeric value into a discrete integer value by `round(log(x))`.
The `base` of `LogRound` is the base of the `log` operator and the `num_bins` is the maximum output value. If the input value is bigger than `2^min_bins`, the output is also
the `num_bins`. 
```python
    layer = LogRound(num_bins=16, base=2)
    input_data = np.asarray([[1.2], [1.6], [0.2], [3.1], [100]])
    result = layer(input_data)
```
The output is `[[0], [1], [0], [2], [7]]`


#### RoundIdentity Layer
The `RoundIdentity` layer is to cast a float value to a integer value using `round(x)`. Then we can feed integer values into `tf.keras.layer.Embedding`. If the input is bigger than
the `max_value`, the output will be the `max_value`.
 ```python
    layer = RoundIdentity(max_value=5)
    input_data = np.asarray([[1.2], [1.6], [0.2], [3.1], [4.9]])
    result = layer(input_data)
    
```
The output is `[[1], [2], [0], [3], [5]]`

### Transform String Inputs

ElasticDL provides [Hashing](#hashing-layer) and [IndexLookup](#indexlookup-layer) layers to map strings to numeric values..

#### Hashing Layer
The `Hashing` layer is to distribute the string value into a finite number of buckets by `hash(x) % num_bins`. 

```python
layer = Hashing(num_bins=3)
input_data = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
result = layer(input_data)
```
The output is `[[1], [0], [1], [1], [2]]`

#### IndexLookup Layer
The `IndexLookup` layer is to map strings to integer indices by looking up vocabulary.

```python
layer = IndexLookup(vocabulary=['A', 'B', 'C'])
input_data = np.array([['A'], ['B'], ['C'], ['D'], ['E']])
result = layer(inputs)
```
The output is `[[0], [1], [2], [3], [3]]`

## Embedding for Preprocessing Results

After preprocessing layers, we get numeric tensors. These numeric tensors can fed into NN layers. Here, we provide some examples of using preprocessing layers to provide inputs for embedding layers.

### Embedding for Features Group 
Sometimes, we may divide input features into groups and use the same embedding layer for one group. Firstly, we may convert inputs to zero-based integer values using the above
preprocessing layers. Then, we can concatenate those outputs into a big tensor. For example, the data set is

| education | marital-status |
| --- | --- |
| Master | Divorced |
| Doctor | Never-married |
| Bachelor | Never-married |

Then, we use preprocessing layers to convert the input data to zero-based integer values.
```python
education = tf.keras.layers.Input(shape=(1, ), dtype=tf.string, name="education")
marital_status = tf.keras.layers.Input(shape=(1, ), dtype=tf.string, name="marital_status")
education_lookup = IndexLookup(vocabulary=['Master', 'Doctor', 'Bachelor'])
education_result = education_lookup(education)
marital_status_lookup = IndexLookup(vocabulary=['Divorced', 'Never-married', 'Never-married'])
marital_status_result = martial_status_lookup (marital_status)
```
Outputs are

```python
education_result = [[0], [1], [2]]
marital_status_result = [[0], [1], [1]]
```
Then, we may want to lookup embedding to map those integer values to different embedding vectors for different features. What's more, we want to set "education" and "martial-status" into a group and lookup embedding with the "education" and "marital_status" results using the same embedding table. If we directly
concatenate two results into a tensor `[[0, 0], [1, 1], [2, 1]]` and lookup an embedding table, embedding results are the same for the same integer values of those features. 
It will make information loss. So, we need to cast integer results of different features into different ranges. 
For example, we can add the vocabulary size of `education_lookup` to `martial_status_result` and concatenate them into a tensor to lookup embedding. In the example, the vocabulary size of `education_lookup is 3, 
the `martial_status_result` is `[[3], [4], [4]]` and the concatenated result is `[[0, 3], [1, 4], [2, 4]]`. So we can map the feature values to different embedding vectors using an embedding table.

ElasticDL provides [ConcatenateWithOffset](https://github.com/sql-machine-learning/elasticdl/blob/develop/elasticdl_preprocessing/layers/concatenate_with_offset.py) layer to concatenate features in a group and cast integer values to different ranges. 

```python
offsets = [0, education_lookup.vocab_size()]
concat_result = ConcatenateWithOffset(offsets=offsets, axis=1)([education_result, martial_status_result])
```

After concatenate features in a group into a tensor, we can feed the tensor into `tf.keras.layer.Embedding`. However, we need to set the `input_dim` for `Embedding` layer and the `input_dim`
should be bigger than the max integer value in the tensor. We can get the maximum by the preprocessing layers like:
```
max_value = education_lookup.vocab_size() + marital_status_lookup.vocab_size()
embedding_result = tf.keras.layer.Embedding(max_value, 1)(concat_result)
embedding_sum = tf.keras.backend.sum(embedding_result, axis=1)
```


### Embedding for Feature Group with Missing Values
Generally, there are missing values in the data set and the missing value may be an empty string for string feature and -1 for the numeric feature. For example, there are
missing values for "education" and "marital-status". 

| education | marital-status |
| --- | --- |
| Master | Divorced |
|  | Never-married |
| Bachelor |  |

We may not want to lookup embedding for those missing values. We need to filter those missing values before converting those values to zero-based
integer values using preprocessing layers. After filtering missing values, we need to use `tf.SparseTensor` to contain the result. ElasticDL provides the `ToSparse` layer to filter
missing values and return a `tf.SparseTensor`. 

```python
education = tf.keras.layers.Input(shape=(1, ), dtype=tf.string, name="education")
marital_status = tf.keras.layers.Input(shape=(1, ), dtype=tf.string, name="marital_status")
to_sparse = ToSparse(ignore_value='')
education_sparse = to_sparse(education)
marital_status_sparse = to_sparse(marital_status)
```
Then, we can use `IndexLookup` layer to convert the sparse tensors to sparse integer tensor and concatenate them into a tensor like the above example. However, `tf.keras.layers.Embedding`
cannot support `tf.SparseTenor` and we need use `elasticdl_preprocessing.layer.Embedding` to lookup embedding with `tf.SparseTenor`.
```
from elasticdl_preprocessing.layers import Embedding

education_lookup = IndexLookup(vocabulary=['Master', 'Doctor', 'Bachelor'])
education_result = education_lookup(education_sparse)
marital_status_lookup = IndexLookup(vocabulary=['Divorced', 'Never-married', 'Never-married'])
marital_status_result = martial_status_lookup (marital_status_sparse)

offsets = [0, education_lookup.vocab_size()]
concat_result = ConcatenateWithOffset(offsets=offsets, axis=1)([education_result, martial_status_result])
max_value = education_lookup.vocab_size() + marital_status_lookup.vocab_size()
embedding_result = Embedding(max_value, 1, combiner="sum")(concat_result)
```

There is another solution to fill the missing value with a default value and convert the default value to a fixed integer value. But the solution has some negative effects and we
have discussed it in the [issue](https://github.com/sql-machine-learning/elasticdl/issues/1844)
 