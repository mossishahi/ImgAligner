#!/usr/bin/env python

# This is a shim to hopefully allow Github to detect the package, build is done with poetry

from setuptools import setup

setup(name='ImgAligner',
      version='0.1.0',
      description='ImgAligner',
      author='Mostafa Shahhoseini',
      author_email='shahhosseini.94@gmail.com',
      url='https://github.com/mossishahi/ImgAligner',
      packages=['ImgAligner'],
     )