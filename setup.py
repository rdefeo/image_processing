from distutils.core import setup
import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='image_processing',
    version='0.1dev',
    packages=[
      'improc',
      'improc.features',
    ],
    install_requires=required
    # long_description=open('README.txt').read(),
)
