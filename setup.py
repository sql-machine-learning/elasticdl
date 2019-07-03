from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="elasticdl",
    version="0.0.1",
    description="A Kubernetes-native Elastic Deep Learning Framework",
    author="Ant Financial",
    url="https://github.com/wangkuiyi/elasticdl",
    install_requires=requirements,
    packages=find_packages(exclude=["*test*"]),
    package_data={"": ["proto/elasticdl.proto", "docker/*", "Makefile"]},
    entry_points={
        "console_scripts": ["elasticdl=elasticdl.python.client.client:main"]
    },
)
