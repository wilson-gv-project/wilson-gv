from setuptools import setup, find_packages

setup(
    name='Wilson',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='wilson'),
    package_dir={'': 'wilson'},
    include_package_data=True,
    install_requires=[],
    license='LICENSE',
    description='A short description of your project',
    long_description=open('README.md').read(),
    url='https://github.com/yourusername/your_package_name',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
