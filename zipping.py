# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 22:03:45 2022

@author: artur
"""

import shutil
import os

os.chdir('/home-net/axesparraguera/data/test_predictions')

shutil.make_archive('Results', 'zip', '/home-net/axesparraguera/data/test_predictions')

