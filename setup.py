from setuptools import find_packages, setup

with open("elasticdl/requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="elasticdl",
    version="0.1.0rc0",
    description="A Kubernetes-native Deep Learning Framework",
    long_description="ElasticDL is a Kubernetes-native deep learning framework built on top of TensorFlow 2.0 that supports fault-tolerance and elastic scheduling.",
    long_description_content_type="text/markdown",
    author="Ant Financial",
    url="https://elasticdl.org",
    install_requires=requirements,
    packages=find_packages(exclude=["*test*"]),
    package_data={"": ["proto/elasticdl.proto", "docker/*", "Makefile"]},
    entry_points={
        "console_scripts": ["elasticdl=elasticdl.python.elasticdl.client:main"]
    },
)
