from setuptools import setup

with open("elasticdl_preprocessing/requirements.txt") as f:
    required_deps = f.read().splitlines()

extras = {}
with open("elasticdl_preprocessing/requirements-dev.txt") as f:
    extras["develop"] = f.read().splitlines()

setup(
    name="elasticdl_preprocessing",
    version="0.1.0",
    description="A Kubernetes-native Deep Learning Framework",
    long_description="This is an extension of the native Keras Preprocessing"
    " Layers and Feature Column API from TensorFlow. We can develop our model"
    " using the native high-level API from TensorFlow and our library."
    "We can train this model using native TensorFlow or ElasticDL.",
    long_description_content_type="text/markdown",
    author="Ant Financial",
    url="https://elasticdl.org",
    install_requires=required_deps,
    extras_require=extras,
    python_requires=">=3.5",
    packages=["elasticdl_preprocessing"],
    package_data={
        "": [
            "requirements.txt",
        ]
    },
)
