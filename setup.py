from setuptools import setup, find_packages

setup(
    name="long_context_attn",
    version="0.1",
    author="Jiarui Fang, Zilin Zhu, Yang Yu",
    url="https://github.com/feifeibear/long-context-attention",
    packages=find_packages(),
    install_requires=[
        'flash-attn',
    ],
)
