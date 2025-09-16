from setuptools import setup, find_packages

setup(
    name="wilson_suite",
    version="0.1.0",
    packages=find_packages(include=["wilson_suite", "wilson_suite.*"]),
    install_requires=[],
    description="Wilson suite core meta utilities",
    author="Your Name",
    author_email="your.email@example.com",
)
