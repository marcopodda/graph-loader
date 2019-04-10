import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graph-loader",
    version="0.0.1",
    author="Marco Podda",
    author_email="marcopodda1985@gmail.com",
    description="Loads graphs from repository",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcopodda/graph-loader",
    packages=setuptools.find_packages(),
    install_requires=['numpy',
                      'scikit-learn',
                      'requests',
                      'pyyaml',
                      'networkx'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Lincense :: OSI Approved :: MIT License",
        "Operating System :: OS Linux"
    ]
)
