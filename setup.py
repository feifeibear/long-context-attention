from setuptools import setup, find_packages

setup(
    name="yunchang",
    version="0.3",
    author="Jiarui Fang, Zilin Zhu, Yang Yu",
    url="https://github.com/feifeibear/long-context-attention",
    packages=find_packages(exclude=['test', 'benchmark']),
    install_requires=[
        'flash-attn>=2.6.0',
    ],
)
