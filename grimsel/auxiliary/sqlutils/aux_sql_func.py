'''
Auxiliary functions for database manipulation.
TODO: Make this a class with a single db connection.
'''

import psycopg2 as pg
import time
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy
from collections import OrderedDict


import subprocess
from os import walk
try:
    # Linux only
    from sh import pg_dump
except:
    pass
import os

import grimsel.grimsel_config as config

# %%



def close_con(conn):
    conn.close()


class SqlConnector():
    '''
    Manages connection strings for psycopg2 and sqlalchemy, open and closes
    connections.

    TODO: Could pass config module to __init__ for config_dict initialization.

    '''

    def __init__(self, db, **kwargs):

        self.db = db

        config_dict = {
        'user': config.PSQL_USER,
        'password': config.PSQL_PASSWORD,
        'host': config.PSQL_HOST,
        'port': config.PSQL_PORT,
        }
        for kw, val in config_dict.items():
            setattr(self, kw, val)
        self.__dict__.update(**kwargs)

        self.pg_str = ('dbname={db} user={user} password={password} '
                      'host={host}').format(**self.__dict__)

        self.sqlal_str = ('postgresql://{user}:{password}'
                          '@{host}:{port}/{db}').format(**self.__dict__)

        self._sqlalchemy_engine = None
        self._conn = None
        self._cur = None

    def get_sqlalchemy_engine(self):

        if not self._sqlalchemy_engine:
            self._sqlalchemy_engine = create_engine(self.sqlal_str)

        return self._sqlalchemy_engine

    def get_pg_con_cur(self):

        if not self._conn:
            self._conn = pg.connect(self.pg_str)
            self._cur = self._conn.cursor()

        return self._conn, self._cur

    def __repr__(self):
        strg = '%s\n%s'%(self.pg_str, self.sqlal_str)
        return(strg)


def exec_sql(exec_str, ret_res=True, time_msg=False, db=None, con_cur=None):
    t = time.time()
    ''' Pass sql query to the server. '''

    conn, cur = (SqlConnector(db).get_pg_con_cur()
                 if not con_cur else con_cur)

    cur.execute(exec_str)
    conn.commit()
    result = None
    try:
        result = cur.fetchall()
    except:
        pass
#        print("No results")

    if con_cur is None:
        conn.close()

    if time_msg:
        print(time_msg, time.time() - t, 'seconds')

    if ret_res:
        return result


# %%

def get_column_string(cols, kind):
    '''
    Generates strings for the multiple manipulation of columns.

    Parameters:
    cols -- columns to be included in the string;
            for kind in (drop, set): list; for kind in (add): dictionary
            of shape {name: datatype}
    kind -- string, one of ('drop', 'add', 'set')
    '''

    if kind == 'drop':
        return ', '.join(['DROP COLUMN IF EXISTS {}'.format(col)
                         for col in cols])
    elif kind == 'add':
        return ', '.join(['ADD COLUMN {} {}'.format(col, dt)
                         for col, dt in cols.items()])
    elif kind == 'set':
        return 'SET ' + ','.join(['{col} = temp.{col}'.format(col=col)
                                  for col in cols])


# %%

def get_sql_tables(sc, db=None, con_cur=None):
    '''
    Returns list of tables in schema
    '''

    exec_str = '''
               SELECT table_name FROM
               information_schema.tables
               WHERE table_schema = \'{sc}\'
               AND table_type = \'BASE TABLE\'
               '''.format(sc=sc)
    return [itb[0] for itb in exec_sql(exec_str, db=db, con_cur=con_cur)]


# %%

def get_sql_cols(tb, sc='public', db=None, con_cur=None):
    '''
    Returns the names and data types of the selected table as a dictionary
    {'col_name': 'col_type', ...}
    '''

    exec_str = '''
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = \'{sc}\'
                AND table_name = \'{tb}\'
                '''.format(sc=sc, tb=tb)

    exec_sql_kwargs = {'exec_str': exec_str}
    exec_sql_kwargs.update({'con_cur': con_cur} if con_cur else {'db': db})

    dict_col = OrderedDict(exec_sql(**exec_sql_kwargs))
    return dict_col


# %%

def reset_schema(sc, db, warn=True):

    sc_exist = len(exec_sql('''
                            SELECT schema_name
                            FROM information_schema.schemata
                            WHERE schema_name = \'{sc}\';
                            '''.format(sc=sc), db=db)) > 0

    if sc_exist:
        try:
            max_run_id = exec_sql('''
                                  SELECT MAX(run_id)
                                  FROM {sc}.def_loop
                                  '''.format(sc=sc), db=db)[0][0]
        except:
            # no def_loop
            max_run_id = 'None'


        if warn:
            input(
'''
~~~~~~~~~~~~~~~   WARNING:  ~~~~~~~~~~~~~~~~
You are about to delete existing schema {sc}.
The maximum run_id is {max_run_id}.

Hit enter to proceed.
'''.format(sc=sc, max_run_id=max_run_id)
)


    print('Dropping output schema: ', sc)
    # This could be a call to a dedicated function init_schema
    # in a dedicated module (like aux_sql_func)
    exec_sql('''
             DROP SCHEMA IF EXISTS {sc_out} CASCADE;
             CREATE SCHEMA IF NOT EXISTS {sc_out}
             '''.format(sc_out=sc), db=db)



# %%

def write_sql(df, db=None, sc=None, tb=None, if_exists=None,
              engine=None, chunksize=None, con_cur=None):


    if not engine:
        sqlc = SqlConnector(db)
        _engine = sqlc.get_sqlalchemy_engine()
    else:
        _engine = engine

    if con_cur:
        exec_sql_kwargs = dict(con_cur=con_cur)
    else:
        exec_sql_kwargs = dict(db=db)



    if if_exists == 'replace':
        exec_str = ('''DROP TABLE IF EXISTS {sc}.{tb} CASCADE;
                    ''').format(sc=sc, tb=tb)
        exec_sql(**dict(exec_str=exec_str, **exec_sql_kwargs))
    else:
        # add columns which exist in the source table but not in
        # the database table;
        # using some internal pandas methods to map the pandas
        # datatypes to SQL datatypes

        pandas_sqltable = pd.io.sql.SQLTable('_', _engine, df, schema='_',
                                             index=False)

        dtype_mapper = pandas_sqltable._sqlalchemy_type

        new_cols = pandas_sqltable._get_column_names_and_types(dtype_mapper)

        old_cols = get_sql_cols(tb, sc,
                                **({'con_cur': con_cur}
                                   if con_cur else {'db': db})).keys()

        VisitableType = sqlalchemy.sql.visitors.VisitableType

        new_cols = {'"{}"'.format(c[0]):
                    c[1]().compile()
                        if isinstance(c[1], VisitableType)
                        else c[1].compile()
                    for c in new_cols if not c[0] in old_cols}
        add_str = get_column_string(new_cols, 'add')

        if not add_str == '':
            exec_strg = '''
                        CREATE TABLE IF NOT EXISTS {sc}.{tb}();
                        ALTER TABLE {sc}.{tb}
                        {add_str};
                        '''.format(sc=sc, tb=tb, add_str=add_str)
            exec_sql(**dict(exec_str=exec_strg, **exec_sql_kwargs))

    df.to_sql(name=tb, con=_engine, schema=sc, if_exists=if_exists,
              index=False, chunksize=chunksize)

    if not engine:
        _engine.dispose()

# %%


def write_sql_grouped(df, db, sc, tb, if_exists, by, verbose=True):
    '''
    Writes DataFrames by slices to avoid the overload due to conversion
    of the DataFrame to an SQL string.

    Parameters:
    df -- DataFrame to be written
    db -- Target database
    sc -- Target schema
    tb -- Target schema
    if_exists -- string ('append', 'replace')
    by -- the by parameter of pandas.DataFrame.groupby
    '''

    dfgrpd = df.groupby(by)

    for ngroup, group in enumerate(dfgrpd.groups):
        if verbose:
            print(('Writing group {} to table {}, number {} of {}')
                  .format(group, sc + '.' + tb, ngroup + 1,
                          len(dfgrpd.groups)))

        dfg = dfgrpd.get_group(group)

        _if_exists = if_exists if ngroup is 0 else 'append'

        write_sql(dfg, db, sc, tb, if_exists=_if_exists)





# %%
def assemble_filt_sql(filt, func):
    '''
    Takes an appropriately shaped filt list and returns
    the argument for the SQL call to WHERE.
    args:
    filt -- list like [('columns', ['value1', 'value2'], binary_operator,
                                 secondary_binary_operator), (next col, ...)]
    func -- dict like {'columns': 'ABS'}
    Example:
    filt=[('run_id', [1, 3]),
          ('pp', ['%WIN%', '%SOL%', '%HCO%'], ' NOT LIKE ', ' AND '),
          ('nd', ['DE0', 'AT0'], ' <> ')]
    returns:
        (nd <> 'DE0' OR nd <> 'AT0')
            AND (run_id = 1 OR run_id = 3)
            AND (pp NOT LIKE '%WIN%' AND pp NOT LIKE '%SOL%'
                 AND pp NOT LIKE '%HCO%')
    Note how ('nd', ['DE0', 'AT0'], ' <> ') has no effect due to the
    default 'OR', hence the necessity for the explicit
    secondary_binary_operator.
    Note: empty value sublists mean all (filter ignored).
    '''

    filt = [list(f) if len(f) >= 3 else list(f) + [' = '] for f in filt]
    filt = [list(f) if len(f) >= 4 else list(f) + [' OR '] for f in filt]
    filt = [f for f in filt if not len(f[1]) is 0]

    rel = {f[0]: f[2] for f in filt}
    opr = {f[0]: f[3] for f in filt}
    fnc = {f[0]: func[f[0]] if f[0] in func.keys() else '' for f in filt}

    filt_dict = {f[0]: [('\'' + if1 + '\'') for if1 in f[1]]
                 if type(f[1][0]) == str else f[1] for f in filt}
    filt_str = ' AND '.join(['(' + opr[ky].join([fnc[ky] + '(' + ky + ')' + rel[ky] + str(ivl)
                                                for ivl in vl]) + ')'
                             for ky, vl in filt_dict.items()])
    return filt_str
#
#if __name__ == '__main__':
#    filt=[('run_id', [1, 3]),
#          ('pp', ['%WIN%', '%SOL%', '%HCO%'], ' NOT LIKE ', ' AND '),
#          ('nd', ['DE0', 'AT0'], ' <> '),
#          ('erg_yr_sy', [1e-6], '<')]
#
#    print(assemble_filt_sql(filt, func={'erg_yr_sy': 'ABS'}))

# %%

def copy_table_structure(sc, tb, sc0, db, verbose=False):
    '''
    Performs the CREATE TABLE LIKE operation.

    Arguments:
    sc -- schema new table
    sc0 -- schema existing table
    tb -- table name
    '''

    exec_str = ('''
                DROP TABLE IF EXISTS {sc}.{tb} CASCADE;
                CREATE TABLE {sc}.{tb} (LIKE {sc0}.{tb} INCLUDING ALL);
                ''').format(sc=sc, tb=tb, sc0=sc0)
    if verbose:
        print(exec_str)
    exec_sql(exec_str, db=db)



# %%

# OBSOLETE? read_sql with intermediate temporary table

#def read_sql(db, sc, tb, filt=False, filt_func=False, drop=False, keep=False,
#             copy=False, tweezer=False, distinct=False, verbose=False):
#    '''
#    Keyword arguments:
#    tweezer -- list; filter (exclude or include) single combinations of values
#                tweezer = [' AND ', ({'nd': 'IT0', 'bool_out': True}, ' NOT '),
#                                    ({'nd': 'FR0', 'bool_out': False}, ' NOT ')]
#    '''
#
#    sqlc = sql_connector(db)
#    engine = sqlc.get_sqlalchemy_engine()
#
#    if verbose:
#        print('Reading ' + sc + '.' + tb +  ' from database ' + db)
#        print('filt=', filt)
#    tweez_str = ''
#    if tweezer:
#        if type(tweezer[0]) is tuple:
#            tweezer.insert(0, ' AND ')
#
#        tweez_str = ' %s (' %tweezer[0]
#        tweez_str += ' AND '.join([tw[1] + '(' + ' AND '.join([str(kk) + ' = ' + '\'' + str(vv) + '\'' for kk, vv in tw[0].items()]) + ')' for tw in tweezer[1:]])
#        tweez_str += ')'
#        if verbose:
#            print('tweez_str', tweez_str)
#
#    # filtering through sql
#    if filt:
#
#        temp_tb = ('temp_read_sql_filtered_'
#                   + ''.join(random.choice(string.ascii_lowercase)
#                             for _ in range(10)))
#        if not filt_func:
#            filt_func = {}
#
#        distinct_str = 'DISTINCT ' if distinct else ''
#        keep_str = ', '.join(keep) if keep else '*'
#        filt_str = assemble_filt_sql(filt, filt_func)
#        exec_str = ('''
#                    DROP TABLE IF EXISTS {temp_tb} CASCADE;
#                    SELECT {distinct_str} {keep_str}
#                    INTO {temp_tb}
#                    FROM {sc}.{tb}
#                    WHERE {filt_str}
#                          {tweez_str};
#                    ''').format(keep_str=keep_str, sc=sc, tb=tb,
#                                filt_str=filt_str, tweez_str=tweez_str,
#                                distinct_str=distinct_str, temp_tb=temp_tb)
#        if verbose:
#            print(exec_str)
#        exec_sql(exec_str, db=db)
#        df = pd.read_sql_table(temp_tb, engine, schema='public')
#
#        exec_sql('DROP TABLE {};'.format(temp_tb), db=db)
#
#    else:
#        df = pd.read_sql_table(tb, engine, schema=sc)
#        if keep:
#            df = df.loc[:, keep]
#
#        if distinct:
#            df = df.drop_duplicates()
#
#    if copy:
#        copy_table_structure(sc=copy, tb=tb, sc0=sc, db=db, verbose=verbose)
##        exec_str = ('''
##                    DROP TABLE IF EXISTS {sc}.{tb} CASCADE;
##                    CREATE TABLE {sc}.{tb} (LIKE {sc0}.{tb} INCLUDING ALL);
##                    ''').format(sc=copy, tb=tb, sc0=sc)
##        if verbose:
##            print(exec_str)
##        exec_sql(exec_str, db=db)
#        write_sql(df, db, copy, tb, 'append')
#
#    if drop:
#        df = df.drop(drop, axis=1)
#
#    engine.dispose()
#
#    return df



def read_sql(db=None, sc=None, tb=None, filt=False, filt_func=False, drop=False,
             keep=False, copy=False, tweezer=False, distinct=False, verbose=False,
             engine=None, limit=None):
    '''
    Keyword arguments:
    tweezer -- list; filter (exclude or include) single combinations of values
                tweezer = [' AND ', ({'nd': 'IT0', 'bool_out': True}, ' NOT '),
                                    ({'nd': 'FR0', 'bool_out': False}, ' NOT ')]
    '''

    if not engine:
        sqlc = SqlConnector(db)
        _engine = sqlc.get_sqlalchemy_engine()
    else:
        _engine = engine


    limit_str = 'LIMIT %d'%limit if isinstance(limit, int) else ''

    if verbose:
        print('Reading ' + sc + '.' + tb +  ' from database ' + db)
        print('filt=', filt)
    tweez_str = ''
    if tweezer:
        if type(tweezer[0]) is tuple:
            tweezer.insert(0, ' AND ')

        tweez_str = ' %s (' %tweezer[0]
        tweez_str += ' AND '.join([tw[1] + '(' + ' AND '.join([str(kk) + ' = ' + '\'' + str(vv) + '\'' for kk, vv in tw[0].items()]) + ')' for tw in tweezer[1:]])
        tweez_str += ')'
        if verbose:
            print('tweez_str', tweez_str)

    cols = keep if keep else get_sql_cols(tb, sc, db)

    # filtering through sql
    if filt and not any(not list(ff[1]) for ff in filt):

        if not filt_func:
            filt_func = {}

        distinct_str = 'DISTINCT ' if distinct else ''
        keep_str = ', '.join(keep) if keep else '*'
        filt_str = assemble_filt_sql(filt, filt_func)
        exec_str = ('''
                    SELECT {distinct_str} {keep_str}
                    FROM {sc}.{tb}
                    WHERE {filt_str}
                          {tweez_str}
                    {limit_str};
                    ''').format(keep_str=keep_str, sc=sc, tb=tb,
                                filt_str=filt_str, tweez_str=tweez_str,
                                distinct_str=distinct_str,
                                limit_str=limit_str)
        if verbose:
            print(exec_str)

        df = pd.DataFrame(exec_sql(exec_str, db=db), columns=cols)

    elif filt:
        # one of the value lists is empty... return empty DataFrame
        df = pd.DataFrame(columns=cols)

    else:
        df = pd.read_sql_table(tb, _engine, schema=sc)
        if keep:
            df = df.loc[:, keep]

        if distinct:
            df = df.drop_duplicates()

    if copy:
        copy_table_structure(sc=copy, tb=tb, sc0=sc, db=db, verbose=verbose)
        write_sql(df, db, copy, tb, 'append')

    if drop:
        df = df.drop(drop, axis=1)

    if not engine:
        _engine.dispose()

    return df



# %%

def on_str(lst, pl='', pr=''):
    return ' AND '.join(['(' + pl + '.' + c + ' = '
                             + pr + '.'  + c + ')'
                         for c in lst])
# %%


def slct_str(lst, exclude=[], pl='', pr='', sl='', sr='', commas=['','']):
    if lst == []:
        if commas == [',',',']:
            commas = [',','']
        else:
            commas = ['','']
    lst = [l for l in lst if not l in exclude]

    lst_1 = []
    for ilst in lst:

        if type(ilst) == list: # defined AS
            lst_1_add = ((pl if pl else '') + ilst[0] + (sl if sl else ''),
                         (pr if pr else '') + ilst[1] + (sr if sr else ''))
            lst_1.append(' AS '.join(lst_1_add))
        elif type(ilst) == str: # only left
            lst_1.append((pl if pl else '') + ilst + (sl if sl else ''))

    lst_1 = ', '.join(lst_1)


    return commas[0] + lst_1 + commas[1]


# %%

# define dictionary containing SQL data type and foreign key for
# model sets
coldict = {
           # FORMAT:
           # 'set_name': ('DATATYPE', )
           'sy': ('INTEGER', '{sc}.tm_soy(sy)'),
           'hy': ('FLOAT', '{sc}.hoy_soy(hy)'),
           'pp_id': ('SMALLINT', '{sc}.def_plant(pp_id)'),
           'pp': ('VARCHAR', '{sc}.def_plant(pp)'),
           'pt_id': ('SMALLINT', '{sc}.def_pp_type(pt_id)'),
           'pt': ('VARCHAR', '{sc}.def_pp_type(pt)'),
           'ca_id': ('SMALLINT', '{sc}.def_encar(ca_id)'),
           'wk_id': ('SMALLINT', '{sc}.def_week(wk_id)'),
           'value': ('DOUBLE PRECISION',),
           'mt_id': ('SMALLINT', '{sc}.def_month(mt_id)'),
           'month': ('VARCHAR',),
           'fl_id': ('SMALLINT', '{sc}.def_fuel(fl_id)'),
           'fl': ('VARCHAR', '{sc}.def_fuel(fl)'),
           'nd_id': ('SMALLINT', '{sc}.def_node(nd_id)'),
           'nd': ('VARCHAR', '{sc}.def_node(nd)'),
           'nd_2_id': ('SMALLINT', '{sc}.def_node(nd_id)'),
           'bool_out': ('BOOLEAN',),
           'weight': ('FLOAT',),
           'min_hoy': ('SMALLINT',), 'year': ('SMALLINT',),
           'day': ('SMALLINT',),
           'dow': ('SMALLINT',), 'dow_name': ('VARCHAR',),
           'doy': ('SMALLINT',), 'hom': ('SMALLINT',),
           'hour': ('SMALLINT',), 'how': ('SMALLINT',),
           'wk': ('SMALLINT',), 'wk_weight': ('SMALLINT',),
           'dow_type': ('VARCHAR',),
           '"DateTime"': ('TIMESTAMP',),
           'mt': ('VARCHAR',),
           'ndays': ('SMALLINT',),
           'season': ('VARCHAR',),
           'wom': ('SMALLINT',),
           'pwrerg_cat': ('VARCHAR',),
           'tm_id': ('SMALLINT',),
           'run_id': ('SMALLINT', '{sc}.def_loop(run_id)'),
#           'supply_pf_id': ('SMALLINT', '{sc}.def_profile(pf_id)'),
#           'dmnd_pf_id': ('SMALLINT', '{sc}.def_profile(pf_id)'),
#           'pricesll_pf_id': ('SMALLINT', '{sc}.def_profile(pf_id)'),
#           'pricebuy_pf_id': ('SMALLINT', '{sc}.def_profile(pf_id)'),
           'pf_id': ('SMALLINT', '{sc}.def_profile(pf_id)'),
          }

def get_coldict(sc=None, db=None, fk_include_missing=False, con_cur=None):
    ''' Adding SQL schema name to foreign keys in coldict. '''


    if sc:
        _coldict = {}
        for kk, vv in coldict.items():
            _val = tuple(ivv.format(sc=sc) for ivv in vv)
            if len(_val) > 1:
                if (not _val[1].split('.')[1].split('(')[0]
                         in get_sql_tables(sc, db, con_cur=con_cur)):
                    _val = (_val[0],)
            _coldict[kk] = [iv.format(sc) for iv in _val]
    else:
        _coldict = {col: [coltype[0]] for col, coltype in coldict.items()}


    return _coldict


def init_table(tb_name, cols, schema='public', ref_schema=None,
               pk=[], unique=[], appendix='', bool_auto_fk=False,
               bool_return=False, db=None, con_cur=None,
               skip_if_exists=False, warn_if_exists=False):
    '''
    Drop and re-initialize indexed table.
    Keyword arguments:
    - tb_name -- table name
    - cols -- list of table columns plus definition of foreign keys
    - schema -- database schema
    - pk -- list of primary key column names
    - unique -- list of unique column names
    - appendix --
    - bool_auto_fk -- complete foreign keys based on the coldict
    - bool_return -- function returns sql string if True
    '''

    if ref_schema is None:
        ref_schema = schema


    _skip = (True if (skip_if_exists
                      and tb_name in get_sql_tables('log', 'fnnc'))
                 else False)


    def warn_exists():

        if warn_if_exists:

            tb_exists = len(exec_sql('''
                            SELECT table_name
                            FROM information_schema.tables
                            WHERE table_schema = \'{sc}\'
                            AND table_name = \'{tb}\';
                            '''.format(sc=schema, tb=tb_name), db=db)) > 0

            if tb_exists:
                exec_count = 'SELECT COUNT(*) FROM %s.%s'%(schema, tb_name)
                len_tb = exec_sql(exec_count, db=db)[0][0]
                input(
'''
~~~~~~~~~~~~~~~   WARNING:  ~~~~~~~~~~~~~~~~
You are about to delete existing table %s.%s.
The length is %d.

Hit enter to proceed.
'''%(schema, tb_name, len_tb))




    if _skip:
        print('Table {} in schema {} exists: skipping init.'.format(tb_name,
                                                                    schema))
    else:

        warn_exists()

        # apply default data type and foreign key to the columns if not provided
        _coldict = get_coldict(ref_schema, db, con_cur=con_cur)

        if not bool_auto_fk:
            _coldict = {col: (typeref[0],)
                        for col, typeref in _coldict.items()}

        _cols = []
        for icol in cols:
            if not (type(icol) is tuple or type(icol) is list):
                icol = (icol,)

            if len(icol) is 1: # column must be in _coldict, otherwise ill-defined
                icol_new = (*icol, *_coldict[icol[0]])
            elif len(icol) is 2:
                icol_new = icol
            else:
                icol_new = icol # all there
            _cols.append(icol_new)



        exec_str = 'DROP TABLE IF EXISTS ' + schema + '.' + tb_name + ' CASCADE;\n'
        exec_str += ('CREATE TABLE ' + schema + '.' + tb_name + ' (')
        exec_str += ',\n '.join([icol[0] + ' ' + icol[1] +
                               (' UNIQUE' if icol[0] in unique else '') +
                               (' REFERENCES ' + icol[2] if len(icol) > 2 else '')
                               for icol in _cols])
        exec_str += ', PRIMARY KEY (' + ', '.join(pk) + ')' if len(pk) > 0 else ''
        exec_str += ', ' + appendix if appendix != '' else ''
        exec_str += ')'

        exec_sql(exec_str, db=db, con_cur=con_cur)

        return ', '.join([c[0] for c in cols])




# %%

def filtered_aggregates(nbin=20,
                        value_table=['out_st_lp', 'duals_supply'],
                        mask_table=['out_st_lp', 'variabs_soy_mod'],
                        out_table=['out_st_lp_analysis', 'filtered_mc_respto_pp'],
                        value_slct={'encar': ['EL'],
                                    'constr': ['supply']},
                        mask_slct={'encar': ['EL'],
                                   'variable': ['power'],
                                   'fuel': ['pumped_hydro_charg',
                                                'pumped_hydro']},
                        merge_cols=['soy', 'node', 'year'],
                        weight_type='filtered_histogram'
                        ):

    nbin=20
    value_table=[out_sc, 'duals_supply']
    mask_table=[out_sc, 'var_sy_pwr']
    out_table=['public', 'filtered_mc_respto_pp']
    value_slct={}
    mask_slct={'pp_type': ['CAS_STO',
                           'LIO_STO']}
    merge_cols=['soy', 'node']
    weight_type='filtered_histogram'

    """
    Obtain filtered histograms or filtered/weighted averages from PSQL DB

    Keyword arguments:
    nbin -- number of bins between min and max of the respective data.
    value_table -- ['schema', 'table_name']
    mask_table -- ['schema', 'table_name']
    out_table -- ['schema', 'table_name']
    value_slct -- dictionary {'column': ['value1', 'value2']} for value table
    mask_slct -- dictionary {'column': ['value1', 'value2']} for mask table
    merge_cols -- list of columns to merge the two tables on
    weight_type -- ['filtered_histogram', ...]

    """

#
    t = time.time()

    # derived
    val_cat = [c for c in value_slct]# if not len(value_slct[c]) == 1]
    cols_slct_val = (['value']
                     + val_cat
                     + merge_cols)
    mask_cat = [c for c in mask_slct]# if not len(mask_slct[c]) == 1]
    cols_slct_mask = (['value']
                      + mask_cat
                      + merge_cols)

    def slct_str(lst, exclude=[], prefix=False, suffix_as=False, suffix=False,
                 commas=['','']):
        if lst == []:
            if commas == [',',',']:
                commas = [',','']
            else:
                commas = ['','']
        main = ', '.join([(prefix if prefix else '')
                           + l + (suffix if suffix else '')
                           + (' AS ' + l + suffix_as if suffix_as else '')
                           for l in lst if not l in exclude])
        return commas[0] + main + commas[1]

    def dict_to_where(dct):
        '''
        Takes input dictionary {column: [list of values]} and transforms it into
        an SQL WHERE statement
        '''
        return (' AND '.join(['(' + ' OR '.join([ivr + ' = ' + '\'' + ivl + '\''
                                                 for ivl in dct[ivr]]) + ')'
                              for ivr in dct]))
    def on_str(lst):
        return ' AND '.join(['tm.' + c + '_mask = ' + 'tv.' + c + '_val' for c in lst])


    exec_str = (
    'DROP TABLE IF EXISTS temp_mrg; SELECT '
    + ', '.join(['tv.' + c + '_val AS ' + c for c in cols_slct_val]) + ', '
    + slct_str([c + '_mask' for c in cols_slct_mask if not c in  merge_cols],
                prefix='tm.') +
    ' INTO temp_mrg' +
    ' FROM ( SELECT ' + slct_str(cols_slct_val, suffix_as='_val') +
    ' FROM ' + '.'.join(value_table) +
    ' WHERE ' + dict_to_where(value_slct) + ' ) tv ' +
    ' LEFT JOIN ( SELECT ' +
    slct_str(cols_slct_mask, suffix_as='_mask', prefix='tm.') +
    ' FROM ( SELECT ' + slct_str(cols_slct_mask) + ' FROM ' +
    '.'.join(mask_table) +
    ' WHERE ' + dict_to_where(mask_slct) +
    ') tm ' +
    ') tm ' +
    ' ON ' + on_str(merge_cols) +
    ' WHERE NOT value_mask IS NULL; ' +
    ' DROP TABLE IF EXISTS ' + '.'.join(out_table) + ';'
    )

    #
    #if weight_type == 'bin':
    #    exec_str += (
    #    '''
    #    /* BINARY MASK */
    #    SELECT AVG(value)
    #    '''
    #     + ('' if len(merge_cols) == 0 else ', ') + ', '.join(merge_cols) +
    #    '''
    #    INTO
    #    '''
    #    + '.'.join(out_table) +
    #    '''
    #    FROM temp_mrg
    #    	WHERE mask <> 0
    #        GROUP BY
    #    '''
    #    + ', '.join(merge_cols) + ';'
    #    )
    #
    #
    #
    #if weight_type == 'weighted':
    #    exec_str += (
    #    '''
    #    /* WEIGHTED AVERAGE */
    #    SELECT SUM(mask * value) / SUM(mask) AS weighted_sum
    #    '''
    #    + ('' if len(merge_cols) == 0 else ', ') + ', '.join(merge_cols) +
    #    '''
    #    INTO
    #    '''
    #    + '.'.join(out_table) +
    #    '''
    #    FROM temp_mrg
    #        GROUP BY
    #    '''
    #    + ', '.join(merge_cols) + ';'
    #    )


# %%

#def multijoin(tables, col_select, merge_cols, keep_cols, output_where):
#
#    '''
#    dict tables --- {0: ['schema', 'name', 'alias'], 1: ...,
#                     'out': ['schema', 'name']}
#    dict col_select = {0: ['column_1', ['column_2', 'AS alias']], 1: ...,
#                       'out': ['col_from_input']}
#    dict merge_cols = {0: [], # not relevant by definition
#                       1: ['col_1'], ...}
#    dict keep_cols = {0: ['variable', 'soy', 'node', 'node_2', 'encar',
#                     'year','power_transmission'],
#                     1: ['mc_node',],
#                     2: ['mc_node_2']}
#    list output_where = ['power_transmission <> 0']
#    '''
#
#    ntb = len(tables) - 1
#    exec_str = 'DROP TABLE IF EXISTS ' + '.'.join(tables['out']) + ';'
#    exec_str += ' WITH '
#    for itb in range(ntb):
#        exec_str += (tables[itb][2] +
#                     ' AS ( SELECT ' + slct_str(col_select[itb]) +
#                     ' FROM ' + '.'.join(tables[itb][:2]) + '), '
#                     )
#    exec_str += 'temp_mrg AS (SELECT '
#    for itb in range(0, ntb):
#        exec_str += slct_str(keep_cols[itb], pl=tables[itb][2] + '.')
#        exec_str += ', ' if (itb < ntb - 1) else ' '
#    exec_str += 'FROM ' + tables[0][2] + ' '
#    for itb in range(1, ntb):
#        exec_str += 'LEFT JOIN ' + tables[itb][2] + ' ON '
#
#        exec_str += on_str(merge_cols[itb], pl=tables[0][2],
#                                            pr=tables[itb][2])
#    exec_str += ') SELECT ' + slct_str(col_select['out'])
#    exec_str += ' INTO ' + '.'.join(tables['out'])
#    exec_str += ' FROM temp_mrg '
#    exec_str += (' WHERE ' + ' AND '.join(['(' + o + ')'
#                                           for o in output_where])
#                  if output_where != [] else '')
#
#    conn = pg.connect(get_config('sql_connect')['psycopg2']
#                      .format(db=get_config('sql_connect')['db']))
#    cur = conn.cursor()
#    cur.execute(exec_str)
#    conn.commit()
#    try:
#        cur.fetchall()
#    except:
#        print("No results")
#    conn.close()
#

# %%



def add_column(df_src, tb_tgt, col_new, on_cols, db,
               data_type='DOUBLE PRECISION'):
    '''
    Add column from dataframe to existing table in database.
    Keyword arguments:
    df_src -- dataframe containing the column to be added
    tb_tgt -- target table as list [schema, table]
    col_new -- name of column to be added
    on_cols -- list of column names for table alignment
    '''
#new_cols
#df_src = dfstep
#tb_tgt = ['out_disagg', 'step_evts_all']
#col_new = new_cols
#on_cols
#db = 'storage2'
#data_type = 'VARCHAR'

    if type(col_new) is dict:
        data_type = col_new.copy()
        col_new = list(col_new.keys())
    else:
        if not type(col_new) is list:
            col_new = [col_new]
        if not type(data_type) is dict:
            data_type = {col: data_type for col in col_new}

    # get data into database, then copy within sql
    write_sql(df_src[on_cols + col_new], db, 'public',
              'temp_add_column', 'replace', chunksize=10000)

    where_str = ' AND '.join(['tgt.' + c + ' = temp.' + c for c in on_cols])

    drop_str = get_column_string(col_new, 'drop')
    add_str = get_column_string(data_type, 'add')
    set_str = get_column_string(col_new, 'set')

    exec_str = ('''
                ALTER TABLE {tb_tgt}
                {drop_str};

                ALTER TABLE {tb_tgt}
                {add_str};

                UPDATE {tb_tgt} AS tgt
                {set_str}
                FROM temp_add_column AS temp
                WHERE {where};
                ''').format(drop_str=drop_str, add_str=add_str,
                            set_str=set_str,
                            tb_tgt='.'.join(tb_tgt),
                            col_new=col_new, where=where_str,
                            data_type=data_type)
    print(exec_str)
    exec_sql(exec_str, db=db)

# %%
def joinon(db, new_col, on_col, tb_target, tb_source,
           new_columns=True, verbose=False):
    '''
    Join columns from one PostgreSQL table to another table.

    Parameters:
    db         : database name
    new_col    : list or dict, if dict, then
                 {col1_name_in_source: new_col1_name_target, ...}
    on_col     : list or dict, if dict, then
                 {col1_name_in_source: col1_name_in_target, ...}
    tb_target  : list or str, specifying target table
                 as ['schema', 'table'] or 'schema.table'
    tb_source  : list or str, specifying source table
                 as ['schema', 'table'] or 'schema.table'
    new_columns: boolean; True: generates new columns
    '''

    if not type(new_col) is dict:
        new_col = {c: c for c in new_col}
    if not type(on_col) is dict:
        on_col = {c: c for c in on_col}

    if type(tb_target) is str:
        tb_target = tb_target.split('.')
    if type(tb_source) is str:
        tb_source = tb_source.split('.')

    exec_str = '''
               SELECT column_name, data_type
               FROM information_schema.columns
               WHERE table_name = \'{tb}\'
               AND table_schema = \'{sc}\';
               '''.format(tb=tb_source[1], sc=tb_source[0])
    type_list = exec_sql(exec_str, db=db)

    if False in [c in [c[0] for c in type_list] for c in list(new_col.keys()) + list(on_col.keys())]:
        raise ValueError('joinon: Either new_col or on_col are not in tb_source.')


    new_col = [tuple([vv]) + [t for t in type_list if kk in t][0]
               for kk, vv in new_col.items()]

    if new_columns:
        alt_str = ('ALTER TABLE ' + '.'.join(tb_target))
        add_str = ' ADD COLUMN IF NOT EXISTS '
        alt_str += ', '.join([add_str + ic + ' SMALLINT'
                              if type(ic) == str
                              else add_str + ' '.join([ic[i] for i in [0, 2]])
                              for ic in new_col]) + '; '
    else:
        alt_str = ''

    upd_str = 'UPDATE  ' + '.'.join(tb_target) + ' AS tg SET '
    new_cn = [ic if type(ic) == str else (ic[0], ic[1]) for ic in new_col]
    upd_str += ', '.join([ic[0] + ' = src.' + ic[1] for ic in new_cn])
    upd_str += ' FROM ' + '.'.join(tb_source) + ' AS src'
    upd_str += ' WHERE ' + ' AND '.join(['tg.' + ic[1] + ' = ' + 'src.' + ic[0]
                                      for ic in on_col.items()])
    upd_str += ';'

    exec_str = alt_str + upd_str

    if verbose:
        print(exec_str)

    time_msg = 'Join columns {} from {} to {}'.format(', '.join([c[0] for c in new_col]),
                                                      '.'.join(tb_source),
                                                      '.'.join(tb_target))
    exec_sql(exec_str, db=db, time_msg=time_msg)
    return exec_str

# %%
# CONSIDERED OBSOLETE DUE TO MISSING DB ARGUMENT

#def filtered_histogram(nbin=50,
#                       input_table = ['public', 'temp_mrg'],
#                       input_select = ['value', 'constr', 'encar', 'node',
#                                       'year', 'value_mask', 'fuel_mask',
#                                       'encar_mask', 'variable_mask'],
#                       input_filter = ['value_mask <> 0'],
#                       column_histogram = 'value',
#                       histogram_sum_cols = ['value', 'value_mask'],
#                       out_table=['out_st_lp_analysis', 'filtered_tr_prices']):
#
#    nbin=50
#    input_table = ['public', 'temp_mrg']
#    input_select = ['value', 'constr', 'encar', 'node',
#                    'year', 'value_mask', 'fuel_mask',
#                    'encar_mask', 'variable_mask']
#    input_filter = ['value_mask <> 0']
#    column_histogram = 'value'
#    histogram_sum_cols = ['value', 'value_mask']
#    out_table=['out_st_lp_analysis', 'filtered_tr_prices']
#
#    exec_str = ('DROP TABLE IF EXISTS ' + '.'.join(out_table) + '; ')
#    exec_str += (' WITH ' + input_table[1] + '_filt AS (' +
#                 ' SELECT ' + slct_str(input_select) +
#                 ' FROM ' + '.'.join(input_table) +
#                 ' ), ')
#    excl_list = [column_histogram] + histogram_sum_cols
#    exec_str += ('val_min_max AS ( SELECT *, ' +
#                 ' MIN(' + column_histogram + ') OVER (PARTITION BY ' +
#                 slct_str(input_select, exclude=excl_list) +
#                 ') as min, ' +
#                 ' MAX(' + column_histogram + ') OVER (PARTITION BY ' +
#                 slct_str(input_select, exclude=excl_list) +
#                 ') as max, ' +
#                 ' COUNT(' + column_histogram + ') OVER (PARTITION BY ' +
#                 slct_str(input_select, exclude=excl_list) +
#                 ') as cnt ' +
#                 ' FROM ' + input_table[1] + '_filt' +
#                 '), ')
#    exec_str += ('val_min_max_filt AS ( SELECT * FROM val_min_max' +
#                 ((' WHERE ' + ' AND '.join(input_filter))
#                   if input_filter != [] else '') +
#                 ')')
#    exec_str += (' SELECT ' +
#                 slct_str(input_select, exclude=excl_list) +
#                 ', MIN(min), MAX(max), ' +
#                 ' width_bucket(' + column_histogram +
#                                ', min, max * 1.0000000001, ' +
#                                str(nbin) + ') as bucket,' +
#                 ' COUNT(*) AS freq, ' +
#                 ', '.join(['SUM(' + c + ') AS ' + c + '_sum'
#                            for c in histogram_sum_cols]))
#    exec_str += (' INTO ' + '.'.join(out_table) +
#                 ' FROM val_min_max_filt ' +
#                 ' GROUP BY bucket '
#                 + slct_str(input_select, exclude=excl_list, commas=[',',''])
#                 )
#
#    conn = pg.connect(get_config('sql_connect')['psycopg2']
#                      .format(db=get_config('sql_connect')['db']))
#    cur = conn.cursor()
#    cur.execute(exec_str)
#    conn.commit()
#    try:
#        cur.fetchall()
#    except:
#        print("No results")
#    conn.close()
#
#    return dt.read_sql('storage1', *out_table), exec_str


# %% compare schemas

def compare_tables(db, list_sc, tb):
#
#    common_filt = [('swvr_vl', ['5.0%']),
#            ('swws_vl', ['All']),
#            ('swst_vl', ['0.0%']),
#            ('swtc_vl', ['CAS'])]
#
#    tb = 'par_fc_om'
#    list_sc = ['out_onlywinsol_new', 'out_nucspreadvr']

    flag_tb_exist = True

    list_tb_exist = []
    for nsc, isc in enumerate(list_sc):
        tb_exist = tb in get_sql_tables(isc, db)
        list_tb_exist.append((tb, isc, tb_exist))

        if not tb_exist:
            flag_tb_exist = False

    df_tot = pd.DataFrame()

    if flag_tb_exist:


        nsc, isc = list(enumerate(list_sc))[0]
        for nsc, isc in enumerate(list_sc):

        #    dict_ids =\
        #    {
        #     'nd_id': read_sql(db, isc, 'def_node').set_index('nd_id')['nd'].to_dict(),
        #     'pp_id': read_sql(db, isc, 'def_plant').set_index('pp_id')['pp'].to_dict(),
        #     }

#            loop_cols = get_sql_cols('def_loop', sc=isc, db=db)
#            slct_filt =  [f for f in common_filt if f[0] in loop_cols]
#            slct_filt = []

#            slct_run_id = read_sql(db, isc, 'def_loop', slct_filt).run_id.tolist()

#            df = read_sql(db, isc, tb, filt=[('run_id', slct_run_id)])
            df = read_sql(db, isc, tb)

            cols_index = ['pp_id', 'pp',
                          'nd_id', 'nd',
                          'fl_id', 'fl',
                          'pt_id', 'pt', 'pp_broad_cat',
                          'ca_id', 'ca',
                          'wk_id', 'wk',
                          'bool_out',
                          'hy', 'sy', 'run_id',
                          'mt_id', 'set_1_name', 'set_2_name', 'set_3_name',
                          'set_1_id', 'set_2_id', 'set_3_id', 'parameter',
                          'swhy_vl', 'color', 'mt']

#            cols = get_sql_cols(tb, sc=isc, db=db)
            slct_cols_index = [c for c in df.columns if c in cols_index]
            df = df.set_index(slct_cols_index).stack().rename(isc)

#            df = df.drop(['run_id'], axis=1).rename(columns={'value': 'value_' + str(nsc)})
#            df = df.rename(columns={'value': 'value_' + str(nsc)})

#            print(df)

            df_tot = pd.concat([df_tot, df], axis=1)


        df_tot = df_tot.fillna(0)

        df_tot['diff'] = (df_tot[list_sc[0]] - df_tot[list_sc[1]]).abs()
        df_tot['reldiff'] = df_tot['diff'] / df_tot[list_sc[0]]
        return df_tot.reset_index().sort_values(['reldiff'], ascending=False)

    else:
        print(list_tb_exist)
        return pd.DataFrame()


if __name__ == '__main__':

    list_sc = ['out_cal_1', 'out_cal_test_lin']
    db = 'storage2'
    tb = 'profdmnd'

    excl = ['profprice', 'profprice_soy', 'def_pp_type',
            'tm_soy_full', 'var_sy_pwr']

    excl += [tb for tb in get_sql_tables(list_sc[0], db) if 'analysis' in tb]
    excl += [tb for tb in get_sql_tables(list_sc[0], db) if 'var_sy' in tb]
    excl += [tb for tb in get_sql_tables(list_sc[0], db) if 'prof' in tb]

    for tb in [c for c in get_sql_tables(list_sc[0], db) if not c in excl]:
        print('*' * 30, tb, '*' * 30)
        dfcomp = compare_tables(db, list_sc, tb)
        dfcomp = dfcomp.loc[dfcomp['reldiff'] > 1e-5]#, ['nd_id', 'value_0', 'value_1', 'diff', 'reldiff']]


        if len(dfcomp) > 0:
            print(dfcomp)


# %%

def dump_by_table_sh(sc, db, target_dir):

    if __name__ == '__main__':
        sc='out_replace_basesmall'
        db='storage2'
        source_base='/run/user/1000/gvfs/dav:host=drive.switch.ch,ssl=true,prefix=%2Fremote.php%2Fdav/files/martin.soini@unige.ch'
        source_dir='SQL_DUMPS/out_replace_basesmall/'
        target_dir=os.path.join(source_base, source_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    db_format = dict(user=None,
                     pw=None,
                     host=None,
                     port=None,
                     db=db)
    dbname = 'postgresql://{user}:{pw}@{host}:{port}/{db}'.format(**db_format)



    for itb in get_sql_tables(sc, db=db):
        tb = sc + '.' + itb
        fn = os.path.join(target_dir, itb + '.sql')
        print('Dumping table ', itb, ' to file ', fn)
        with open(fn, 'w+') as f:
            pg_dump('--dbname', dbname, '--table', tb, _out=f)

dbname = 'postgresql://postgres:postgres@localhost:5432/{db}'

def dump_by_table(sc, db, target_dir='C:\\Users\\ashreeta\\Documents\\Martin\\SWITCHdrive\\SQL_DUMPS\\out_disagg_new\\'):

#    if __name__ == '__main__':
#        sc='out_nucspreadvr_ext_it'
#        db='storage1'
#        target_dir='C:\\Users\\ashreeta\\Documents\\Martin\\SWITCHdrive\\SQL_DUMPS\\out_nucspreadvr_ext_it\\'

    exe = '\"C:\\Program Files\\PostgreSQL\\9.6\\bin\\pg_dump.exe\"'

    for itb in get_sql_tables(sc, db=db):
        tb = sc + '.' + itb
        fn = os.path.join(target_dir, tb + '.sql')
        print('Dumping table ', itb, ' to ', fn)
        run_str = ('{exe} --table {tb} --dbname={dbname} > {fn}'
                       .format(exe=exe, tb=tb, dbname=dbname.format(db=db), fn=fn))

        subprocess.run(run_str, shell=True, check=True)


def read_by_table(db, sc,
                  source_base='/run/user/1000/gvfs/dav:host=drive.switch.ch,ssl=true,prefix=%2Fremote.php%2Fdav/files/martin.soini@unige.ch',
                  source_dir='', warn_reset_schema=True,
                  patterns_only=False):

#    if __name__ == '__main__':
#        print('main')
#        db='storage2'
#        source_base='/run/user/1000/gvfs/dav:host=drive.switch.ch,ssl=true,prefix=%2Fremote.php%2Fdav/files/martin.soini@unige.ch'
#        source_dir='SQL_DUMPS/out_replace_vreseries/'


    source_dir = os.path.join(source_base, source_dir)


    f = []
    for (dirpath, dirnames, filenames) in walk(source_dir):
        f.extend(filenames)
        break

    if patterns_only:
        f = [fn for fn in f if any(pat in fn for pat in patterns_only)]


    print('Reading from %s, %d files found.' % (source_dir, len(f)))

    exe = 'psql'

    reset_schema(sc, db, warn=warn_reset_schema)

    db_format = dict(user=None,
                     pw=None,
                     host=None,
                     port=None,
                     db=db)
    dbname = 'postgresql://{user}:{pw}@{host}:{port}/{db}'.format(**db_format)

    for file in f:
        print('Reading file ', file)
        fn = source_dir + file

        run_str = '{exe} --dbname={dbname} < {fn}'.format(exe=exe, dbname=dbname, fn=fn)
        subprocess.run(run_str, shell=True, check=True)

if __name__ == '__main__':
    pass

#    source_dir = 'SQL_DUMPS/out_replace_emission/'
#    sc = 'out_replace_emission'
#    read_by_table(db, sc, source_dir=source_dir, warn_reset_schema=False,
#                  patterns_only=['analysis_', 'def_', 'encar'])

#    dump_by_table('out_marg_store', 'storage2', target_dir='C:\\Users\\ashreeta\\Documents\\Martin\\SWITCHdrive\\SQL_DUMPS\\out_marg_store_new\\')

# %%



