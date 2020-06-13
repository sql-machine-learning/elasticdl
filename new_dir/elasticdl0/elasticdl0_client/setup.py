from setuptools import setup

setup(
    name="elasticdl0_client",
    packages=['elasticdl0_client.python'],
    entry_points={
        "console_scripts": ["elasticdl0_client=elasticdl0_client.python.client:main"]
    }
)