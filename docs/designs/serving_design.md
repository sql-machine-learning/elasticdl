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
A single serving process will run out of memory while loading the model. We partition the model variables into multiple shards, store them in distributed parameter server for serving. The inference engine will execute the serving graph, query the variable values from the distributed parameter server as needed and finish the calculation. It's necessary to upgrade the serving framework to support this. **We will discuss this case in a separate design in the next step.**

## Feature Columns

During model development, we can use feature columns to describe the feature engineering process. And then we wrap feature_column array with tf.keras.layer.DenseFeatures and pass it to keras Model. While exporting the trained model to SavedModel, the logic and variable weights of feature columns will be saved together.

## Model Saving Process
