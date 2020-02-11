from setuptools import find_packages, setup

with open("elasticdl/requirements.txt") as f:
    required_deps = f.read().splitlines()

extras = {}
with open("elasticdl/requirements-dev.txt") as f:
    extras["develop"] = f.read().splitlines()

setup(
    name="elasticdl",
    version="v0.2.0rc0",
    description="A Kubernetes-native Deep Learning Framework",
    long_description="ElasticDL is a Kubernetes-native deep learning framework"
    " built on top of TensorFlow 2.0 that supports"
    " fault-tolerance and elastic scheduling.",
    long_description_content_type="text/markdown",
    author="Ant Financial",
    url="https://elasticdl.org",
    install_requires=required_deps,
    extras_require=extras,
    python_requires=">=3.5",
    packages=find_packages(exclude=["*test*"]),
    package_data={
        "": ["proto/*.proto", "docker/*", "Makefile", "requirements.txt"]
    },
    entry_points={
        "console_scripts": ["elasticdl=elasticdl.python.elasticdl.client:main"]
    },
)
