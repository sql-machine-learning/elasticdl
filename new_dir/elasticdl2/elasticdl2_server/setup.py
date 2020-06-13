from setuptools import setup

setup(
    name="elasticdl2_server",
    packages=['elasticdl2_server.python'],
    install_requires=['elasticdl2_common'],
    entry_points={
        "console_scripts": ["elasticdl2_server=elasticdl2_server.python.server:main"]
    }
)