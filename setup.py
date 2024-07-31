"""
Document:
    - Why you shouldn't invoke setup.py directly:
    https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html

    - Packaging and distributing projects:
    https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#create-an-account

    -
"""

import os
import subprocess
import sys
from sys import platform

from setuptools import setup, find_packages, Command

CWD: str = os.path.dirname(__file__)


def read(file_name: str) -> str:
    """Read README.md file."""
    with open(os.path.join(CWD, file_name), 'r') as f:
        return f.read()


def list_all(file_name: str) -> list[str]:
    """List all lines"""
    with open(os.path.join(CWD, file_name), 'r') as f:
        return f.read().splitlines()


PY_CMD: str = 'python3' if platform != 'win32' else 'py'
meta: dict = {}

with open(os.path.join('florentino', '__version__.py')) as file:
    exec(file.read(), meta)


class Publish(Command):
    """Publish to PyPI with twine."""
    description = ('\033[1;92mBuild and publish the package to PyPI\n->'
                   f'\033[1;93mflorentino VERSION=\033[1;92m{meta['__version__']}\033[0m')
    user_options = []

    def initialize_options(self):
        print(self.description)

    def finalize_options(self):
        pass

    def run(self):
        result = subprocess.run([PY_CMD, '-m', 'pytest'], check=False)
        if result.returncode != 0:
            print("Tests failed. Aborting publish.", file=sys.stderr)
            sys.exit(result.returncode)
        print(f'florentino VERSION=\033[1;92m{meta['__version__']}\033[0m')
        subprocess.run([PY_CMD, '-m', 'pip', 'install', '--upgrade', 'pip', 'build', 'twine'], check=True)
        subprocess.run([PY_CMD, 'setup.py', 'sdist', 'bdist_wheel'], check=True)
        print(self.description)
        subprocess.run([PY_CMD, '-m', 'twine', 'upload','--repository', 'pypi', 'dist/*'], check=True)
        print(self.description)


class TestPublish(Command):
    """Test publish to PyPI with twine."""
    description = ('\033[1;92mBuild and publish the package to Test PyPI\n->'
                   f'\033[1;93mflorentino VERSION=\033[1;92m{meta['__version__']}\033[0m')
    user_options = []

    def initialize_options(self):
        print(self.description)

    def finalize_options(self):
        pass

    def run(self):
        result = subprocess.run([PY_CMD, '-m', 'pytest'], check=False)
        if result.returncode != 0:
            print("Tests failed. Aborting publish.", file=sys.stderr)
            sys.exit(result.returncode)
        subprocess.run([PY_CMD, '-m', 'pip', 'install', '--upgrade', 'pip', 'build', 'twine'], check=True)
        subprocess.run([PY_CMD, 'setup.py', 'sdist', 'bdist_wheel'], check=True)
        print(self.description)
        subprocess.run([PY_CMD, '-m', 'twine', 'upload', '--repository', 'testpypi', 'dist/*'], check=True)
        print(self.description)


setup(
    name=meta['__name__'],
    version=meta['__version__'],
    author=meta['__author__'],
    maintainer=meta['__maintainer__'],
    maintainer_email=meta['__maintainer_email__'],
    license=meta['__license__'],
    description=meta['__description__'],
    url=meta['__url__'],
    packages=find_packages(),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=list_all('requirements.txt'),
    python_requires=meta['__python_requires__'],
    project_urls={
        'Source Code': meta['__src__'],
    },
    keywords='deeplearning machinelearning AI florentino hcmut bku',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    cmdclass={
        'publish': Publish,
        'test_publish': TestPublish,
    },
)
