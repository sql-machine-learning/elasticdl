from setuptools import setup

setup(
    name="elasticdl3_server",
    packages=['elasticdl3_server.python'],
    install_requires=['elasticdl3_common'],
    entry_points={
        "console_scripts": ["elasticdl3_server=elasticdl3_server.python.server:main"]
    }
)