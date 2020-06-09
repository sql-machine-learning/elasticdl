#!/bin/bash

PYTHON_SOURCE_FILES=$(find elasticdl/python \
                           elasticdl_preprocessing \
                           model_zoo \
                           tools \
                           setup.py \
                           scripts \
                           -name '*.py' -print0 | tr '\0' ' ')

GO_SOURCE_FILES=$(find elasticdl/pkg -name '*.go' -print0 | tr '\0' ' ')

pre-commit run --files "$PYTHON_SOURCE_FILES $GO_SOURCE_FILES"
