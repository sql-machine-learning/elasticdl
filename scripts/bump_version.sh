#!/bin/bash

sed -i "" "s/version=.*/version=\"$1\",/" setup.py
