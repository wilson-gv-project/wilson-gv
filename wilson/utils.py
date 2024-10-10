import os

# Get the root directory of the package dynamically
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_package_root():
    """Returns the absolute path to the package root."""
    return PACKAGE_ROOT
