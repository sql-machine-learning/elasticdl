from setuptools import setup

setup(
    name="elasticdl4_server",
    packages=['elasticdl4_server.python'],
    install_requires=['elasticdl4_client'],
    entry_points={
        "console_scripts": ["elasticdl4_server=elasticdl4_server.python.server:main"]
    }
)