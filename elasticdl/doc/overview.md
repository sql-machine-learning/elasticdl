ElasticDL is a framework implements the swamp optimization meta-algorithm. It is like Apache Hadoop is a framework that implements the MapReduce parallel programming paradigm.

To program the ElasticDL framework, programmers need to provide at least one `nn.Module`-derived class that describes the specification of a model. It is like programmers of Hadoop need to provide a class that implements the methods of Map and Reduce.

To train a model, ElasticDL needs (1) hyperparameter values, and (2) the data.  Each ElasticDL job uses the same data to train one or more models, where each model needs a set of hyperparameter values. A model could have more than one sets of hyperparameter values.  In such a case, they are considered multiple models.

- A job is associated with a dataset.
- A job is associated with one or more model specifications, each model specification is a Python classed derived from torch.nn.Module.
- A model specification is associated with one or more sets of hyperparameter values.
- The pair of a model specification and a set of its hyperparameter values is a model.
- A job includes a coordinator process and one or more bee processes.
- A bee process trains one or more models.
- The coordinator dispatches models to bees.

The following example command line starts an ElasticDL job.

```bash
elasticdl start \
-model='MyModuleClass,a_param=[0.1:0.001:10:logscale],another_one="a string value"' \
-model='AnotherModuleClass,yet_a_param=[1:10:5]'
```

This example train the following models six models:

1. MyModuleClass(a_param=0.1, another_one="a string value")
1. MyModuleClass(a_param=0.01, another_one="a string value")
1. MyModuleClass(a_param=0.001, another_one="a string value")
1. AnotherModuleClass(yet_a_param=1)
1. AnotherModuleClass(yet_a_param=5)
1. AnotherModuleClass(yet_a_param=10)
