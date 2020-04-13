#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import grimsel

logger = grimsel._get_logger(__name__)


logger.info(f'grimsel_config file {__file__}')

try:
    import config_local as conf_local
    PATH_CSV = conf_local.PATH_CSV
except Exception as e:
    print(e)

    PATH_CSV = os.path.join(grimsel.__path__[0], 'input_data')
    print('Using default csv path %s'%PATH_CSV)

try:
    import config_local as conf_local
    BASE_DIR = conf_local.BASE_DIR
except Exception as e:
    print(e)
    logger.warning('Could not import BASE_DIR path.')

try:
    import config_local as conf_local
    DATABASE = conf_local.DATABASE
    SCHEMA = conf_local.SCHEMA

    PSQL_USER = conf_local.PSQL_USER
    PSQL_PASSWORD = conf_local.PSQL_PASSWORD
    PSQL_HOST = conf_local.PSQL_HOST
    PSQL_PORT = conf_local.PSQL_PORT

except Exception as e:
    print(e)
    logger.error('''
Could not read PSQL configuration parameters from
config_local.
Please set configuration parameters in
a new project-specific config_local.py file, e.g.

import os
DATABASE = \'database_name\'
SCHEMA = 'model_input_schema_name'

PSQL_USER = 'user'
PSQL_PASSWORD = 'password'
PSQL_PORT = 5432
PSQL_HOST = 'localhost'
'''
)

