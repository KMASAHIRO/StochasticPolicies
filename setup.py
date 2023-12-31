from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='StochasticPolicies',
    version='0.0.8',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    url='https://github.com/KMASAHIRO/StochasticPolicies',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description='Rethinking Stochastic Policy Gradient Methods Using Traffic Simulator Benchmark',
    python_requires=">=3.6.9"
)
