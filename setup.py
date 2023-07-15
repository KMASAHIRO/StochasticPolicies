from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='TrafficFlowControl_with_StochasticPolicies',
    version='0.0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    url='https://github.com/KMASAHIRO/RESCO',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description='Rethinking the Stochastic Policy Gradient Methods Using a Traffic Simulator.',
    python_requires=">=3.6.9"
)