# Install elasticdl package by setup.py
# Local installation:
#     pip3 install .
# Make package:
#     python3 setup.py sdist bdist_wheel

from setuptools import setup, find_packages

setup(name='elasticdl',
      version='0.0.1',
      description='elasticdl master, worker and client',
      url='https://github.com/wangkuiyi/elasticdl',
      author='Ant',
      author_email='XXX@antfin.com',
      license='TBD',
      packages=['elasticdl', 'elasticdl.common', 'elasticdl.proto', 
          'elasticdl.client', 'elasticdl.master', 'elasticdl.worker',
          'elasticdl.recordio_ds_gen', 'elasticdl.recordio_ds_gen.cifar10',
          'elasticdl.recordio_ds_gen.mnist'],
      include_package_data=True)
