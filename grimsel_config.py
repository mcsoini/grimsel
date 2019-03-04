#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import grimsel

logger = grimsel._get_logger(__name__)


try:
    import grimsel.config_local as conf_local
    PATH_CSV = conf_local.PATH_CSV
except Exception as e:
    print(e)

    PATH_CSV = os.path.join(grimsel.__path__[0], 'input_data')
    print('Using default csv path %s'%PATH_CSV)

try:
    import grimsel.config_local as conf_local
    FN_XLSX = conf_local.FN_XLSX
    DATABASE = conf_local.DATABASE
    SCHEMA = conf_local.SCHEMA

    PSQL_USER = conf_local.PSQL_USER
    PSQL_PASSWORD = conf_local.PSQL_PASSWORD
    PSQL_HOST = conf_local.PSQL_HOST
    PSQL_PORT = conf_local.PSQL_PORT

except Exception as e:
    print(e)
    logger.error('''
Please set configuration parameters in
grimsel/config_local.py, e.g.

import os
FN_XLSX = os.path.abspath(\'../DATA/input.xlsx\')
DATABASE = \'database_name\'
SCHEMA = 'model_input_schema_name'

PSQL_USER = 'user'
PSQL_PASSWORD = 'password'
PSQL_PORT = 5432
PSQL_HOST = 'localhost'
'''
)

