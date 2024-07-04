#!/usr/bin/env python
from scriptsHPC.utils import calcsCFOUR
from input_parameters import config_fja

calcsCFOUR.process_fja(config_fja)

print('The last step was submitted. The final results will be in "save" directory')
