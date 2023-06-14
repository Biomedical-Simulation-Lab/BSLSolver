#!/usr/bin/env python

from setuptools import setup

setup(
    name = 'BSLSolver',
    version = '0.1',
    packages = ['BSLSolver',
                'BSLSolver.common',
                'BSLSolver.problems',
                'BSLSolver.Post',
                ],
   entry_points = {
              'console_scripts': [
                  'bslsolver = BSLSolver.run_BSLSolver:main',                  
              ],
              },
   data_files = [('BSLSolver/common/data', ['BSLSolver/common/data/FC_case1']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/FC_case2']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/FC_CCA']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/FC_CONST']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/FC_ECA']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/FC_ICA']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/FC_MCA']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/FC_MCA_10']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/ICA_values']),
                 ('BSLSolver/common/data', ['BSLSolver/common/data/PC_SQUARE']),
                ],
)
