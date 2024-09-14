from setuptools import setup, find_packages
import os

# 读取版本信息
version_file = os.path.join(os.path.dirname(__file__), 'yunchang', '__version__.py')
with open(version_file, 'r') as f:
    exec(f.read())

setup(
    name="yunchang",
    version=__version__,
    author="fangjiarui123@gmail.com",
    url="https://github.com/feifeibear/long-context-attention",
    packages=find_packages(exclude=['test', 'benchmark']),
    install_requires=[
        'flash-attn>=2.6.0', # default install cuda version,
    ]
)
