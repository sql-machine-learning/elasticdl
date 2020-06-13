from setuptools import setup

setup(
    name="elasticdl3_client",
    packages=['elasticdl3_client.python'],
    install_requires=['elasticdl3_common'],
    entry_points={
        "console_scripts": ["elasticdl3_client=elasticdl3_client.python.client:main"]
    }
)