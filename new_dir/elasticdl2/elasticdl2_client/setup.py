from setuptools import setup

setup(
    name="elasticdl2_client",
    packages=['elasticdl2_client.python'],
    install_requires=['elasticdl2_common'],
    entry_points={
        "console_scripts": ["elasticdl2_client=elasticdl2_client.python.client:main"]
    }
)