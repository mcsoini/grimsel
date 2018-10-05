#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

try:
    import grimsel.config_local as conf_local
    PATH_CSV = conf_local.PATH_CSV
except:
    PATH_CSV = os.path.abspath('input_data')
    print('Using default csv path %s'%PATH_CSV)

try:
    import grimsel.config_local as conf_local
    FN_XLSX = conf_local.FN_XLSX
    DATABASE = conf_local.DATABASE
    SCHEMA = conf_local.SCHEMA

except:
    raise RuntimeError('Please set configuration parameters in '
                       'grimsel/config_local.py, e.g. \n'
                       'FN_XLSX = os.path.abspath(\'../DATA/input.xlsx\')\n'
                       'DATABASE = \'database_name\'\n'
                       'SCHEMA = \'model_input_schema_name\'\n'
                       )



