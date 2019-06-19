from setuptools import setup, find_packages


setup(
    name="ElasticDL",
    version='0.0.1',
    description="A Kubernetes-naitve Elastic Deep Learning Framework",
    author="Ant Financial",
    url="https://github.com/wangkuiyi/elasticdl",
    packages=find_packages(exclude=['*test*']),
    package_data={'': ['proto/elasticdl.proto', 'docker/*', 'Makefile']},
    entry_points={
        'console_scripts':
            ['elasticdl=elasticdl.python.elasticdl.client.client:main'],
    },
)
