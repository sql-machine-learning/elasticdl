# Install elasticdl package by setup.py
# Local installation:
#     pip3 install .
# Make package:
#     python3 setup.py bdist

from setuptools import setup, find_packages

packages = find_packages(where='.', exclude=['*.pyc'])

install_requires = ['numpy']

setup(name='elasticdl',
      version='0.0.1',
      description='distributed programming framework for deep learning',
      url='https://github.com/wangkuiyi/elasticdl',
      author='Ant',
      author_email='XXX@antfin.com',
      license='TBD',
      packages=packages,
      install_requires=install_requires)
