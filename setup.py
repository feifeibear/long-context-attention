from setuptools import setup, find_packages

setup(
    name="yunchang",
    version="0.3.2",
    author="Jiarui Fang",
    author_email="fangjiarui123@gmail.com",
    description="A package for long context attention",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/feifeibear/long-context-attention",
    project_urls={
        "Bug Tracker": "https://github.com/feifeibear/long-context-attention/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: APL 2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=["yunchang"],
    python_requires=">=3.7",
    install_requires=[
        "flash-attn>=2.6.0",
    ],
)