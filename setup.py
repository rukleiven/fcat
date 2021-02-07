from setuptools import setup, find_packages

setup(
    name="fcat",
    author=["Ruben Kleiven"],
    author_email="kleiven.rk@gmail.com",
    long_description="Flight Control Analysis Toolbox",
    url="https://github.com/rukleiven/fcat",
    packages=find_packages(),
    keywords=["Flight", "Control Systems", "Adaptive Control"],
    install_requires=['numpy', 'matplotlib', 'scipy', 'control', 'click', 'pyyaml'],
    scripts=['scripts/cli.py']
)