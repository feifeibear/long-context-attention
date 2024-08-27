from setuptools import setup, find_packages


setup(
    name="yunchang",
    version="0.3",
    author="Jiarui Fang, Zilin Zhu, Yang Yu",
    url="https://github.com/feifeibear/long-context-attention",
    packages=find_packages(exclude=['test', 'benchmark']),
    install_requires=[
        'ninja',
        'flash-attn>=2.6.0', # default install cuda version,
    ],
    extras_require={ 
        'amd': [
            'ninja',
            'flash_attn @ git+https://git@github.com/Dao-AILab/flash-attention.git' # flash_attn-2.6.3+rocm62 for pytorch 2.3.0a0
            ],
    },
)
