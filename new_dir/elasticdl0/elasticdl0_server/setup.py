from setuptools import setup

setup(
    name="elasticdl0_server",
    packages=['elasticdl0_server.python'],
    entry_points={
        "console_scripts": ["elasticdl0_server=elasticdl0_server.python.server:main"]
    }
)