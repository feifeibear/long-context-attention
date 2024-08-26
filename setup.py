from setuptools import setup, find_packages


setup(
    name="yunchang",
    version="0.3",
    author="Jiarui Fang, Zilin Zhu, Yang Yu",
    url="https://github.com/feifeibear/long-context-attention",
    packages=find_packages(exclude=['test', 'benchmark']),
    install_requires=[
        'ninja',
        # ROCM support has been added to main branch since Jun [PR#1010](https://github.com/Dao-AILab/flash-attention/pull/1010);
        # flash_attn-2.6.3+rocm62 for pytorch 2.3a
        'flash_attn @ git+https://git@github.com/Dao-AILab/flash-attention.git',
    ],
)
