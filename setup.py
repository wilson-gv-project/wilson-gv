from setuptools import setup, find_packages
setup(
    name="wilson_suite",  # The overall package name
    version="0.1.0",
    description="A suite of tools for Wilson project.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repo",  # Replace with your repo URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        # Add your dependencies here
    ],
    extras_require={
        "dev": [
            # Add development dependencies here
        ]
    },
    python_requires=">=3.7",
)