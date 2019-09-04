from setuptools import find_packages, setup

with open("elasticdl/requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="elasticdl",
    version="0.0.1",
    description="A Kubernetes-native Deep Learning Framework",
    author="Ant Financial",
    url="https://github.com/sql-machine-learning/elasticdl",
    install_requires=requirements,
    packages=find_packages(exclude=["*test*"]),
    package_data={"": ["proto/elasticdl.proto", "docker/*", "Makefile"]},
    entry_points={
        "console_scripts": ["elasticdl=elasticdl.python.elasticdl.client:main"]
    },
)
