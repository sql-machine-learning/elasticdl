# Serving Design

## Background

Model serving is an essential part in an end-to-end machine learning lifecycle. Publishing the trained model as a service in production can make it valuable in the real world.
Current situation: [To be Added]

[SavedModel](https://www.tensorflow.org/guide/saved_model?hl=zh_cn) is the universal serialization format for tensorflow models. It's language neutral and can be loaded by multiple frameworks (such as TFServing, TFLite, TensorFlow.js and so on). We choose to store the ElaticDL model into SavedModel format. In this way, we can leverage various mature solutions to serving our model in different scenarios.

The model size can vary from several kilobytes to several terabytes. We divide the model size into two categories: *Small or medium size* and *large size*. The small or medium size model can be loaded by a process, and the latter can not fit in a single process. Training and serving strategies are different between these two cases. Please check the following table:

|                            | Master Central Storage | AllReduce |            Parameter Server              |
|----------------------------|:----------------------:|:---------:|------------------------------------------|
| Small or Medium Size Model |       SavedModel       | SavedModel|               SavedModel                 |
| Large Size Model           |          N/A           |    N/A    | Distributed Parameter Server for Serving |

Small or medium size model\
A single serving process can load the entire model into memory. No matter which training strategy to choose, we will serialzied the model into SavedModel format. And then we can deploy it using the existed serving frameworks like TFServing. **We focus on this case in this article.**

Large size model\
A single serving process will run out of memory while loading the model. We partition the model variables into multiple shards, store them in distributed parameter server for serving. The inference engine will execute the serving graph, query the variable values from the distributed parameter server as needed and finish the calculation. It's necessary to upgrade the serving framework to support this. **We will discuss this case in a separate doc in the next step.**

## Feature Columns
ElasticDL is a distributed deep learning framework based TensorFlow 2.0 eager execution. In ElasticDL, we use dataset to create input pipeline for training and the recommended way to preprocess data from dataset is to use [feature columns](https://www.tensorflow.org/tutorials/structured_data/feature_columns) in TensorFlow. What's more, tf.saved_model.save will save the defined feature columns with the model. So, tf-serving will use the same preprocessing as training to make inference.
```
import time
import numpy as np
import tensorflow as tf 

def get_feature_columns():
    age = tf.feature_column.numeric_column("age", dtype=tf.int64)
    education = tf.feature_column.categorical_column_with_hash_bucket(
        'education', hash_bucket_size=4)
    education_one_hot = tf.feature_column.indicator_column(education)
    
    return [age, education_one_hot]

def get_input_layer():
    input_layers = {}
    input_layers['age'] = tf.keras.layers.Input(name='age', shape=(1,), dtype=tf.int64)
    input_layers['education'] = tf.keras.layers.Input(name='education', shape=(1,), dtype=tf.string)
    return input_layers

def custom_model(feature_columns):
    input_layers = get_input_layer()
    dense_feature = tf.keras.layers.DenseFeatures(feature_columns=feat_cols)(input_layers)
    dense = tf.keras.layers.Dense(10, activation='relu')(dense_feature)
    dense = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    return tf.keras.models.Model(inputs=input_layers, outputs=dense)

feat_cols = get_feature_columns()
model = custom_model(feat_cols)  
output = model.call({'age':tf.constant([[10],[16]]),
                     'education':tf.constant([['Bachelors'],['Master']])})
print(output)

export_dir = './saved_models/feature_columns/{}'.format(int(time.time()))
tf.saved_model.save(model, export_dir=export_dir)  
```
the outputs of model.call is 
```
tf.Tensor(
[[0.6658912 ]
 [0.73223007]], shape=(2, 1), dtype=float32)
```

Then we test the outputs of tf-serving with the model.
```
>> curl -d '{"instances": [{"age":[10],"education":["Bachelors"]}]}' -X POST http://localhost:8501/v1/models/model:predict

{
    "predictions": [[0.665891171]
    ]
}
```

Although all feature columns in TensorFlow can be used in ElasticDL, the tf.feature_column.embedding_column is not recommended in ElasticDL. Because the embedding_column has a large trainable embedding parameters. In eager execution the model must get all embedding parameters to train which will cost large inter-process communication overhead.

## Save model using SavedModel
Using native Keras layers to define a model is more user-friendly than using custom layers in ElasticDL. However, it is inefficient to train a model with tf.keras.layers.Embedding. When the model executes the forward-pass computation for each mini-batch, it must get all embedding parameters from the parameter server (PS) even if the mini-batch only contains several embedding ids. So, the elastic.layers.Embedding is designed to improve the training efficiency in ElasticDL. Considering user-friendliness and training efficiency, we need to define a model with tf.keras.layers.Embedding and execute training with elastic.layers.Embedding. There is a feasible method that we replace the tf.keras.layers.Embedding layer with elastic.layers.Embedding layer to create a new model instance. For the Sequential model and the Model class used with the functional API, we can do it by tf.keras.models.clone_model.

```
import tensorflow as tf
from tensorflow import keras
from elasticdl.python.elasticdl.layers.embedding import Embedding as edl_Embedding
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense

def clone_function(layer):
    if type(layer) == keras.layers.Embedding:
        edl_layer = edl_Embedding(layer.output_dim)
        return edl_layer
    return layer

inputs = Input(shape=(10,))
embedding = Embedding(10,4)(inputs)
flatten = Flatten()(embedding)
output = Dense(1, activation='sigmoid')(flatten)
model = tf.keras.Model(inputs=[inputs], outputs=[output])
new_model = keras.models.clone_model(model, clone_function=clone_function)
```

For subclass model, we can replace the tf.keras.layers.Embedding attribute with elastic.layers.Embedding.
```
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from elasticdl.python.elasticdl.layers.embedding import Embedding as edl_Embedding

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = Embedding(10,4)
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    def call(self, inputs):
        embedding = self.embedding(inputs)
        x = self.dense1(embedding)
        return self.dense2(x)
model = MyModel()

for attr_name, attr_value in model.__dict__.items():
    if type(attr_value) == keras.layers.Embedding:
        setattr(model, attr_name, edl_Embedding(attr_value.output_dim))
```

However, tf.saved_model.save can not save the replaced model using SavedModel, because elasticDL.layers.Embedding is not the native layer in tf.keras.layers. There are two methods to save the model by tf.saved_model.save. One is that we add the elasticDL.layers.Embedding to tensorflow.keras.layers and compile TensorFlow with the custom layer to a custom version. It may be incompatible with a new TensorFlow version. In this case, we may need to adjust the elasticDL.layers.Embedding implementation when every new version of TensorFlow is released. Another method is that we can save the origin model and replace the embedding parameters with the trained parameters of elasticDL.layers.embedding layer. 

To verify the feasibility for tf-serving, we make an experiment to save a Keras model and replace the embedding parameters with values in a CSV file. The CSV file mocks the trained parameters of elasticDL.layers.embedding layer. Firstly, we define a model with 4x4 tf.keras.layers.Embedding. Then we view the outputs of the embedding layer and the last dense layer.
```
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten

inputs = Input(shape=(1,))
embedding = Embedding(4,4)(inputs)
flatten = Flatten()(embedding)
output = Dense(1, activation='sigmoid')(flatten)
model = tf.keras.Model(inputs=[inputs], outputs=[output, embedding])

input_array = tf.constant([[1]])
output = model.call(input_array)
print('training output : ', output)
```

```
training output :  
[<tf.Tensor: id=67, shape=(1, 1), dtype=float32, numpy=array([[0.4970102]], dtype=float32)>, 
<tf.Tensor: id=60, shape=(1, 1, 4), dtype=float32, numpy=
array([[[-0.04582943,  0.00981156,  0.03840413,  0.03487637]]], dtype=float32)>]
```

Now, we save the model with embedding parameters in a CSV file, the values in the CSV file is:
```
0,  1,  2,  3
4,  5,  6,  7
8,  9, 10, 11
12, 13, 14, 15
```

```
def replace_embedding_params_with_edl(layer):
    import pandas as pd
    var_values = pd.read_csv('variable.csv')
    custom_param = var_values.values
    for var in layer.trainable_variables:
        var.assign(custom_param)
    
def replace_model_embedding_layer(model):
    for layer in model.layers:
        if type(layer) == tf.keras.layers.Embedding:
            replace_embedding_params_with_edl(layer)

replace_model_embedding_layer(model)
output = model.call(input_array)
print('predict output : ',output)
```
After the parameters are replaced, the outputs are:
```
predict output :  
[<tf.Tensor: id=442, shape=(1, 1), dtype=float32, numpy=array([[0.9999994]], dtype=float32)>, 
<tf.Tensor: id=435, shape=(1, 1, 4), dtype=float32, numpy=array([[[4., 5., 6., 7.]]], dtype=float32)>]
```

Finally, we save the model with replaced parameters using SavedModel and test the outputs of the SavedModel by [tf-serving](https://www.tensorflow.org/tfx/serving/docker). 
```
>> curl -d '{"instances": [1]}' -X POST http://localhost:8501/v1/models/model:predict

{
    "predictions": [
        {
            "dense": [0.9999994],
            "embedding": [4.0, 5.0, 6.0, 7.0]
        }
    ]
}
```

## Model Saving Process
