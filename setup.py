from setuptools import setup, find_packages

setup(
    # Basic info
    name='Wilson',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',

    # Packages to include
    packages=find_packages(),

    # Include additional files into the package
    include_package_data=True,

    # Dependencies
    install_requires=[
        # List your project dependencies here.
        # For example: 'requests>=2.23.0',
    ],

    # Metadata
    license='LICENSE',
    description='A short description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # If your README is in markdown
    url='https://github.com/yourusername/your_package_name',  # Optional project URL

    # Classifiers help users find your project by categorizing it.
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3, or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    # Could also include keywords, download_url, etc.
)