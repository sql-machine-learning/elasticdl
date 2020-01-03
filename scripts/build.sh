#! /bin/bash
set -e

#Build the project
bazel build -s --verbose_failures  //...

#Run the tests
bazel test  //...