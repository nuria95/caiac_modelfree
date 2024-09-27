#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    # 'M3L @ git+https://github.com/sukhijab/M3L.git',
    #'dm_control',
    'stable-baselines3[extra]'
    # 'minitouch @ git+https://github.com/sukhijab/MiniTouch.git',
    # 'tactile_envs @ git+https://github.com/sukhijab/tactile_envs.git',
]

extras = {}
setup(
    name='multimexmf',
    version='0.0.1',
    license="MIT",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=required,
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)