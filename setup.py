from setuptools import setup, find_packages


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


long_description = read_file("README.md")
requirements = read_requirements("requirements.txt")

setup(
    name='optimalportfolios',
    version='3.3.2',
    author='Artur Sepp',
    author_email='artursepp@gmail.com',
    url='https://github.com/ArturSepp/OptimalPortfolios',
    description='Implementation of optimisation analytics for constructing and backtesting optimal portfolios',
    long_description_content_type="text/x-rst",  # If this causes a warning, upgrade your setuptools package
    long_description=long_description,
    license="MIT license",
    packages=find_packages(exclude=["optimalportfolios/examples/figures", "optimalportfolios/examples/crypto_allocation/data"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)