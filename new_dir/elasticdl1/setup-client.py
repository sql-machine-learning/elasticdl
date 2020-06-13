from setuptools import setup

setup(
    name="elasticdl1_client",
    packages=['elasticdl1_client.python', 'elasticdl1_common.python'],
    entry_points={
        "console_scripts": ["elasticdl1_client=elasticdl1_client.python.client:main"]
    }
)