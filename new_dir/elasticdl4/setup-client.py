from setuptools import find_packages, setup

setup(
    name="elasticdl4_client",
    packages=find_packages(
        include=["elasticdl4_client.*"]
    ),
    entry_points={
        "console_scripts": ["elasticdl4_client=elasticdl4_client.python.client:main"]
    }
)