# Serving Design

## Backgroud

Model serving is an essential part an the end-to-end machine learning lifecycle. Publishing the trained model as a service in production can make it valuable in the real world.

[SavedModel](https://www.tensorflow.org/guide/saved_model?hl=zh_cn) is the universal serialization format for tensorflow models. It's language neutral and can be loaded by multiple frameworks (such as TFServing, TFLite, TensorFlow.js and so on). We choose to store the ElaticDL model into SavedModel format. In this way, we can leverage various mature solutions to serving our model in different scenarios.

The model size varies from several kilobytes to several terabytes. We divide the model size into two categories: *Small or medium size* and *large size*. The small or medium size model can be loaded by a process, and the latter can not fit in a single process. Training and serving strategies will be different between these two cases. Please check the following table:

|                            | Master Central Storage | AllReduce |            Parameter Server              |
|----------------------------|:----------------------:|:---------:| -----------------------------------------|
| Small or Medium Size Model |        SavedModel      | SavedModel|               SavedModel                 |
| Large Size Model           |           N/A          |    N/A    | Distributed Parameter Server for Serving |

## Feature Engineering

## Model Saving Process
