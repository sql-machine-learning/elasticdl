# ElasticDL Preprocessing

This is a feature preprocessing library provided by ElasticDL.  
It provides API in the following forms:

- Keras layer
- Feature column Api

This is an extension of the native [Keras Preprocessing Layers](https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/preprocessing) and [Feature Column API](https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/feature_column) from TensorFlow. We can develop our model using the native API and our extension and train this model using both Native TensorFlow and ElasticDL.  

*Note: Some native [Keras Preprocess layers](https://github.com/tensorflow/community/pull/188) will be released in TF2.2. For the TF version < 2.1, we will provide our implementation of the same functionality.*
