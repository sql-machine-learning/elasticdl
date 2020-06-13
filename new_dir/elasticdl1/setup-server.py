from setuptools import setup

setup(
    name="elasticdl1_server",
    packages=['elasticdl1_server.python', 'elasticdl1_common.python'],
    entry_points={
        "console_scripts": ["elasticdl1_server=elasticdl1_server.python.server:main"]
    }
)