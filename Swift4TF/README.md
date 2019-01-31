# Swift For TensorFlow

[Swift for TensorFlow(S4TF)](https://www.tensorflow.org/swift/) is an early stage project that aims to describing TensorFlow training using programming model.

To help evaluating S4TF, we have built a docker image with latest pre-built packages. The docker image is at `reg.docker.alibaba-inc.com/elasticdl/swift4tf`. Note that, the Docker image has to be run in privileged mode. For example:

```
docker run -it --privileged reg.docker.alibaba-inc.com/elasticdl/swift4tf /bin/bash
```

You can follow the instructions [here](https://github.com/tensorflow/swift/blob/master/Usage.md) to try it out.

Currently (1/30/2019), the REPL way works. However, the interpreter and compiler way won't work. Here are some related bugs: 

* Interpreter `swift -O` won't work. [Issue](https://bugs.swift.org/browse/SR-8610)
* Compiler `swiftc` won't work. [Issue](https://github.com/tensorflow/swift/issues/10#issuecomment-451068821)

When there are new pre-built package available [here](https://github.com/tensorflow/swift/blob/master/Installation.md), we can rebuild the Docker image by updating the dockerfile in `dockerfile/Swift4TF` directory.
