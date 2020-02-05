#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import json

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = list(requirements_file.readlines())

    # problenm with tox
    requirements.append('tensorflow==1.15.2')

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Skander Kamoun",
    author_email='skander.kam2@gmail.com',
    python_requires='>=3.5, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Generic Interpretability package",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='trelawney',
    name='trelawney',
    packages=find_packages(include=['trelawney', 'trelawney.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/skanderkam/trelawney',
    version='0.3.1',
    zip_safe=False,
)
