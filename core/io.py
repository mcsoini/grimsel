#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:03:42 2019

@author: user
"""
import time
import itertools
import os

import pandas as pd
from pyomo.core.base.param import _ParamData # required if params are mutable

import grimsel.auxiliary.sqlutils.aux_sql_func as aql
import grimsel

import grimsel.core.autocomplete as ac


import grimsel.core.table_struct as table_struct

dict_tables = {lst: {tb[0]: tuple(tbb for tbb in tb[1:])
                     for tb in getattr(table_struct, lst)}
               for lst in table_struct.list_collect}


chg_dict = table_struct.chg_dict

def get_table_dicts():
    '''
    Get the dictionaries describing all tables.

    Also used in post_process_index, therefore classmethod.
    '''

    # construct table name group name and component name
    dict_idx = {comp: spec[0]
                for grp, tbs in dict_tables.items()
                for comp, spec in tbs.items()}
    dict_table = {comp: (grp + '_' + spec[1]
                         if len(spec) > 1
                         else grp + '_' + comp)
                  for grp, tbs in dict_tables.items()
                  for comp, spec in tbs.items()}
    dict_group = {comp: grp
                  for grp, tbs in dict_tables.items()
                  for comp, spec in tbs.items()}

    return dict_idx, dict_table, dict_group

# expose as module variable for easier access
DICT_IDX, DICT_TABLE, DICT_GROUP = get_table_dicts()


# %%


if __name__ == '__main__':

    DICT_IDX = io.DICT_IDX
    DICT_TABLE = io.DICT_TABLE
    DICT_GROUP = io.DICT_GROUP


class CompIO():
    '''
    A CompIO instance takes care of extracting a single variable/parameter from
    the model and of writing a single table to the database.
    '''

    def __init__(self, tb, sc, comp_obj, idx, connect, model=None,
                 coldict=None):

        self.tb = tb
        self.sc = sc
        self.comp_obj = comp_obj
        self.connect = connect
        self.model = model

        self.columns = None # set in index setter
        self.run_id = None  # set in call to self.write_run

        self.index = idx

        self.coldict = aql.get_coldict()

    def post_processing(self, df):
        ''' Child-specific method called prior to writing. '''
        return df

    def to_df(self):
        '''
        Calls classmethods _to_df.

        Is overwritten in DualIO, where _to_df is not used as classmethod.
        '''

        return self._to_df(self.comp_obj,
                           [c for c in self.index if not c == 'bool_out'])


    def init_output_table(self):
        '''
        Initialization of output table.

        Note: Keys need to be added in post-processing due to table
        writing performance.
        '''

        print('Initializing output table ', self.tb)
        col_names = self.index + ('value',)
        cols = [(c,) + (self.coldict[c][0],) for c in col_names]
        cols += [('run_id', 'SMALLINT')]
        pk = [] # pk added later for writing/appending performance
        unique = []

        aql.init_table(tb_name=self.tb, cols=cols,
                       schema=self.sc,
                       ref_schema=self.sc, pk=pk,
                       unique=unique, bool_auto_fk=False, db=self.connect.db,
                       con_cur=self.connect.get_pg_con_cur())

    def add_bool_out_col(self, df):

        if 'bool_out' in self.index:
            df['bool_out'] = chg_dict[self.tb]

        return df

    def _finalize(self, df, tb=None):
        ''' Add run_id column and write to database table '''

        tb = self.tb if not tb else tb
        print('Writing %s to'%self.comp_obj.name, self.sc + '.' + tb, end='... ')

        # value generally positive, directionalities expressed through bool_out
        df['value'] = df['value'].abs()

        df['run_id'] = self.run_id

        t = time.time()
        df.to_sql(tb, self.connect.get_sqlalchemy_engine(),
                  schema=self.sc, if_exists='append', index=False)
        print('done in %.3f sec'%(time.time() - t))

        if 'bool_out' in df.columns:
            print('Unique bool: ', df.bool_out.unique())
        print(df.head())

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        ''' Makes sure idx is tuple and updates columns attribute. '''
        self._index = (value,) if not isinstance(value, tuple) else value
        self.columns = list(self.index + ('value',))
        self.columns = [c for c in self.columns if not c == 'bool_out']

    def write(self, run_id):

        self.run_id = run_id

        df = self.to_df()

        df = self.post_processing(df)

        self._finalize(df)

    def _node_to_plant(self, pt):
        '''
        TODO: THIS SHOULD BE IN MAPS!!!!
        Method for translation of node_id to respective plant_id
        in the cases of demand and inter-node transmission. This is used to
        append demand/inter-nodal transmission to pwr table.
        Returns a dictionary node -> plant
        Keyword arguments:

        * pt -- string, selected plant type for translation
        '''
        df_np = self.model.df_def_plant[['nd_id', 'pp_id', 'pp', 'pt_id']]
        df_pt = self.model.df_def_pp_type[['pt_id', 'pt']]
        mask_pt = df_pt.pt.apply(lambda x: x.replace('_ST', '')) == pt
        slct_pp_type = df_pt.loc[mask_pt, 'pt_id'].astype(int).iloc[0]

        mask_tech = df_np['pt_id'] == slct_pp_type
        df_np_slct = df_np.loc[mask_tech]
        dict_node_plant_slct = df_np_slct.set_index('nd_id')
        dict_node_plant_slct = dict_node_plant_slct['pp_id'].to_dict()
        return dict_node_plant_slct

    def __repr__(self):

        return 'Comp_obj: ' + str(self.comp_obj)

class DualIO(CompIO):
    '''
    Base class for dual values. Performs the data extraction of constraint
    shadow prices.
    '''

    def to_df(self):

        dat = [ico + (self.model.dual[self.comp_obj[ico]],)
               for ico in self.comp_obj]
        return pd.DataFrame(dat, columns=self.columns)



class VariabIO(CompIO):
    '''
    Base class for variables. Performs the data extraction of variable
    objects.
    Also: Special methods related to the handling of negative plant
    variables defined by setlst[sll] + setlst[curt]
    as well as storage charging.

    '''

    @classmethod
    def _to_df(cls, obj, cols):
        ''' Converts pyomo variable to DataFrame.'''

#        # replace none with zeros
#        for v in obj.iteritems():
#            if v[1].value is None:
#                v[1].value = 0
#
#        dat = [v[0] + (v[1].value,) for v in obj.iteritems()]

        df = pd.Series(obj.extract_values()).fillna(0).reset_index()
        df.columns = list(cols) + ['value']

        return df#pd.DataFrame(dat, columns=list(cols) + ['value'])

    def post_processing(self, df):
        '''
        Calls _set_bool_out prior to writing.
        Input arguments:
        * df -- DataFrame; primary dataframe
        '''

        return self._set_bool_out(df) if 'bool_out' in self.index else df

    def _set_bool_out(self, df):
        '''
        Set the bool_out values according to the pp_id.

        pp_ids corresponding to curtailment, charging, and sales have
        bool_out=True.
        * df -- DataFrame; primary dataframe
        '''

        if self.comp_obj.name in ['pwr_st_ch', 'erg_ch_yr']:
            df['bool_out'] = True
        else:
            df['bool_out'] = False

            pp_true = self.model.setlst['curt'] + self.model.setlst['sll']
            df.loc[df.pp_id.isin(pp_true), 'bool_out'] = True

        return df


class ParamIO(CompIO):
    '''
    Base class for parameters. Performs the data extraction of parameter
    objects.
    '''

    @classmethod
    def _to_df(cls, obj, cols):
        ''' Converts pyomo parameter to DataFrame. '''

        df = pd.Series(obj.extract_values()).fillna(0).reset_index()
        if df.empty:
            df = pd.DataFrame(columns=list(cols) + ['value'])
        else:
            if not cols and len(df) is 1:
                df = df[[0]].rename(columns={0: 'value'})
            else:
                df.columns = list(cols) + ['value']
##
#        dat = []
#        for v in obj.iteritems():
#            v = list(v)
#            if not v[0] == None:
#                if isinstance(v[1], _ParamData):
#                    # if parameter is mutable v[1] is a _ParamData;
#                    # requires manual extraction of value
#                    v[1] = v[1].value
#                if v[1] == None:
#                    v[1] = 0
#                if not type(v[0]) == tuple:
#                    v_sets = [v[0]]
#                else:
#                    v_sets = [iv for iv in v[0]]
#                dat += [v_sets + [v[1]]]
#            else:
#                dat = [v[1].extract_values()[None]]

        return df#pd.DataFrame(dat, columns=list(cols) + ['value'])


class TransmIO(VariabIO):
    '''
    Special methods related to the translation of nodes to plant names and
    the simplified representation after aggregating secondary nodes.
    '''

    def post_processing(self, df):
        ''' Write aggregated transmission table to pwr. '''


        dfagg = self.aggregate_nd2(df)
        self._translate_trm(dfagg)

        self._finalize(dfagg, 'var_sy_pwr')

        return self.add_bool_out_col(df)

    def aggregate_nd2(self, dfall):
        '''
        Aggregates trm table over all secondary nodes for simplification and
        to append to the pwr table.
        '''

        # mirror table to get both directions
        dfall = pd.concat([dfall,
                           dfall.assign(nd_2_id = dfall.nd_id,
                                        nd_id = dfall.nd_2_id,
                                        value = -dfall.value)])

        dfexp = dfall.loc[dfall.value > 0]
        dfexp = dfexp.groupby(['sy', 'nd_id', 'ca_id'])['value'].sum()
        dfexp = dfexp.reset_index()
        dfexp['bool_out'] = True

        dfimp = dfall.loc[dfall.value < 0]
        dfimp = dfimp.groupby(['sy', 'nd_id', 'ca_id'])['value'].sum()
        dfimp = dfimp.reset_index()
        dfimp['bool_out'] = False

        dfagg = pd.concat([dfexp, dfimp], axis=0)


        return dfagg

    def _translate_trm(self, df):

        df['pp_id'] = df.nd_id.replace(self._node_to_plant('TRNS'))
        df.drop('nd_id', axis=1, inplace=True)

        return df

    def add_bool_out_col(self, df):
        ''' The bool out column value depends on the sign of the data. '''

        df['bool_out'] = False
        df.loc[df.value < 0, 'bool_out'] = True

        return df


class DmndIO(ParamIO):
    ''' Demand is appended to the pwr table after translation '''

    def post_processing(self, df):

        dfpp = self._translate_dmnd(df.copy())
        dfpp['bool_out'] = True

        self._finalize(dfpp, 'var_sy_pwr')

        return df

    def _translate_dmnd(self, df):

        ''''''

        dict_ndpp = self._node_to_plant('DMND')

        dict_pfpp = {val: dict_ndpp[key[0]] for key, val
                     in self.model.dict_dmnd_pf.items()}

        df['pp_id'] = df.dmnd_pf_id.replace(dict_pfpp)

        df.drop('dmnd_pf_id', axis=1, inplace=True)

        return df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def skip_if_resume_loop(f):
    def wrapper(self, *args, **kwargs):
        if self.resume_loop:
            pass
        else:
            f(self, *args, **kwargs)
    return wrapper

def skip_if_no_output(f):
    def wrapper(self, *args, **kwargs):
        if self.no_output:
            pass
        else:
            f(self, *args, **kwargs)
    return wrapper


class ModelWriter():
    '''
    The IO singleton class manages the TableIO instances and communicates with
    other classes. Manages database connection.
    '''

    IO_CLASS_DICT = {'var': VariabIO,
                     'var_tr': TransmIO,
                     'par_dmnd': DmndIO,
                     'par': ParamIO,
                     'dual': DualIO}

    _default_init = {'sc_warmstart': False,
                     'resume_loop': False,
                     'replace_runs_if_exist': False,
                     'model': None,
                     'sql_connector': None,
                     'no_output': False,
                     'dev_mode': False,
                     'sc_out': None,
                     'db': None}


    def __init__(self, **kwargs):

        ''''''

        self.run_id = None  # set in call to self.write_run
        self.dict_comp_obj = {}

        # define instance attributes and update with kwargs
        for key, val in self._default_init.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)


        print('Output schema: ', self.sc_out,
              '; resume loop=', self.resume_loop)
        self.reset_schema()


    @skip_if_resume_loop
    def reset_schema(self):

        aql.reset_schema(self.sc_out, self.sql_connector.db, not self.dev_mode)

    def init_output_schema(self):

        aql.exec_sql('CREATE SCHEMA IF NOT EXISTS ' + self.sc_out,
                     db=self.db, )

    @skip_if_no_output
    def init_compio_objs(self):
        '''
        Initialize all output table IO objects.
        '''

        comp, idx = 'pwr_st_ch', DICT_IDX['pwr_st_ch']
        for comp, idx in DICT_IDX.items():
            if not hasattr(self.model, comp):
                print('Component ' + comp + ' does not exist... skipping '
                      'init CompIO.')
            else:
                comp_obj = getattr(self.model, comp)

                grp = DICT_GROUP[comp]
                if DICT_TABLE[comp] in self.IO_CLASS_DICT:
                    io_class = self.IO_CLASS_DICT[DICT_TABLE[comp]]
                elif grp in self.IO_CLASS_DICT:
                    io_class = self.IO_CLASS_DICT[DICT_GROUP[comp]]
                else:
                    io_class = self.IO_CLASS_DICT[DICT_GROUP[comp].split('_')[0]]

                io_class_args = (DICT_TABLE[comp], self.sc_out, comp_obj,
                                   idx, self.sql_connector)
                io_class_args += (self.model,)

                self.dict_comp_obj[comp] = io_class(*io_class_args)

    @skip_if_no_output
    def write_all(self):

        ''' Calls the write methods of all CompIO objects. '''

        for comp, io_obj in self.dict_comp_obj.items():

            io_obj.write(self.run_id)

    @skip_if_no_output
    def init_all(self):
        '''
        Initializes all SQL tables.

        Calls the init_output_table methods of all CompIO instances.
        '''

        coldict = aql.get_coldict(self.sc_out, self.sql_connector.db)

        for comp, io_obj in self.dict_comp_obj.items():

            io_obj.coldict = coldict
            io_obj.init_output_table()

    def delete_run_id(self, run_id=False, operator='>='):
        '''
        In output tables delete all rows with run_id >=/== the selected value.

        Used in :
            1. in ModelLoop.perform_model_run if replace_runs_if_exist == True
                with operator '=' to remove current run_id
                from all tables prior to writing

        TODO: The SQL part would be better fit with the aux_sql_func module.
        '''
        # Get overview of all tables
        list_all_tb_0 = [list(itb_list + '_' + itb[0] for itb
                              in getattr(self, itb_list)
                              if not len(itb) == 3)
                         for itb_list in self.list_collect]
        self.list_all_tb = list(itertools.chain(*list_all_tb_0))
        self.list_all_tb += ['def_loop']

        if run_id:
            for itb in self.list_all_tb:

                print('Deleting from ' + self.sc_out + '.' + itb
                      + ' where run_id %s %s'%(operator, str(run_id)))
                exec_strg = '''
                            DELETE FROM {sc_out}.{tb}
                            WHERE run_id {op} {run_id};
                            '''.format(sc_out=self.sc_out, tb=itb,
                                       run_id=run_id, op=operator)
                try:
                    aql.exec_sql(exec_strg, db=self.db)
                except pg.ProgrammingError as e:
                    print(e)
                    sys.exit()




    @classmethod
    def post_process_index(cls, sc, db, drop=False):

        coldict = aql.get_coldict(sc, db)

        dict_idx, dict_table, _ = ModelWriter.get_table_dicts()

        list_tables = aql.get_sql_tables(sc, db)

        for comp, index in dict_idx.items():

            if not dict_table[comp] in list_tables:
                print('Table ' + comp + ' does not exist... skipping '
                      'index generation.')
            else:

                tb_name = dict_table[comp]

                print('tb_name:', tb_name)

                pk_list = index + ('run_id',)

                fk_dict = {}
                for c in pk_list:
                    if len(coldict[c]) > 1:
                        fk_dict[c] = coldict[c][1]


                pk_kws = {'pk_list': ', '.join(pk_list),
                          'tb': tb_name, 'sc_out': sc}
                exec_str = ('''
                            ALTER TABLE {sc_out}.{tb}
                            DROP CONSTRAINT IF EXISTS {tb}_pkey;
                            ''').format(**pk_kws)
                if not drop:
                    exec_str += ('''
                                 ALTER TABLE {sc_out}.{tb}
                                 ADD CONSTRAINT {tb}_pkey
                                 PRIMARY KEY ({pk_list})
                                 ''').format(**pk_kws)
                print(exec_str)
                aql.exec_sql(exec_str, db=db)

                for fk_keys, fk_vals in fk_dict.items():
                    fk_kws = {'sc_out': sc, 'tb': tb_name,
                              'fk': fk_keys, 'ref': fk_vals}

                    exec_str = ('''
                                ALTER TABLE {sc_out}.{tb}
                                DROP CONSTRAINT IF EXISTS fk_{tb}_{fk};
                                ''').format(**fk_kws)

                    if not drop:
                        exec_str += ('''
                                     ALTER TABLE {sc_out}.{tb}
                                     ADD CONSTRAINT fk_{tb}_{fk}
                                     FOREIGN KEY ({fk})
                                     REFERENCES {ref}
                                     ''').format(**fk_kws)
                    print(exec_str)
                    aql.exec_sql(exec_str, db=db)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TableReader():
    '''
    Reads tables from input data sources and makes them attributes of the
    model attribute.
    '''

    def __init__(self, sql_connector, sc_inp, data_path, model):

        self.sqlc = sql_connector
        self.sc_inp = sc_inp
        self.data_path = data_path
        self.model = model

        if not self.sc_inp and not self.data_path:
            print('Falling back to grimsel default csv tables.')
            self.data_path = os.path.join(grimsel.__path__[0], 'input_data')


        self.table_list = self._get_table_list()

    def _get_table_list(self):
        ''' Obtain list of tables in the relevant data source. '''

        if self.sc_inp:
            return aql.get_sql_tables(self.sc_inp, self.sqlc.db)
        elif self.data_path:
            return [fn.replace('.csv', '')
                    for fn in next(os.walk(self.data_path))[-1]]

    def _expand_table_families(self, dct):
        '''
        Searches for tables with identical name + suffix in the same data source.
        Updates and returns the dct.
        '''

        dct_add = {}
        for table, filt in dct.items():

            tbs_other = [tb for tb in self.table_list
                         if table in tb and not tb == table]

            if tbs_other:
                dct_add.update({tb: filt for tb in tbs_other})

        dct.update(dct_add)

    def df_from_dict(self, dct):
        ''' Reads filtered input tables and assigns them to instance
            attributes. (Copies filtered tables to output schema.
            OBSOLETE DUE TO AUTO-COMPLETE) '''

        self._expand_table_families(dct)

        for kk, vv in dct.items():

            df, tb_exists, source_str = self.get_input_table(kk, vv)

            setattr(self.model, 'df_' + kk, df)

            if not tb_exists:
                print('Input table {tb} does not exist. '
                      + 'Setting model attribute df_{tb} '
                      + 'to None.'.format(tb=kk))
            else:
                filt = ('filtered by ' if len(vv) > 0 else '') +\
                   ', '.join([vvv[0] + ' in ' + str(vvv[1]) for vvv in vv
                              if not len(vvv[1]) is 0])
                print(('Reading input table {tb} {flt} from '
                       '{source_str}').format(tb=kk, flt=filt,
                                              source_str=source_str))

    def get_input_table(self, table, filt):

        if self.sc_inp:
            tb_exists = table in aql.get_sql_tables(self.sc_inp, self.sqlc.db)
            if tb_exists:
                df = aql.read_sql(self.sqlc.db, self.sc_inp, table, filt)
            source = '%s %s.%s'%(self.sqlc.db, self.sc_inp, table)
        else:
            if self.data_path:
                path = self.data_path
            else:
                path = os.path.join(grimsel.__path__[0], 'input_data')
            fn = os.path.join(path, '%s.csv'%table)

            source = fn
            tb_exists = os.path.exists(fn)

            if tb_exists:
                df = pd.read_csv(fn)

                for col, vals in filt:
                    df = df.loc[df[col].isin(vals)]

        return (df if tb_exists else None), tb_exists, (' from %s'%source
                                                        if source else '')


class DataReader():

    def __init__(self, **kwargs):

        defaults = {'resume_loop': False,
                    'replace_runs_if_exist': False,
                    'model': None,
                    'autocomplete_curtailment': False,
                    'autocompletion': True,
                    'no_output': False,
                    'dev_mode': False,
                    'data_path': None,
                    'sql_connector': None,
                    'sc_inp': None,
                    'sc_out': None,
                    'db': None,
                    }

        defaults.update(kwargs)
        for key, val in defaults.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)


        self._coldict = aql.get_coldict()



    def read_model_data(self):
        '''
        Read all input data to instance attributes.
        '''

        tbrd = TableReader(self.sql_connector, self.sc_inp,
                           self.data_path, self.model)

        # unfiltered input
        dict_tb_2 = {'def_month': [], 'def_week': [],
                     'parameter_month': []}
        tbrd.df_from_dict(dict_tb_2)

        # read input data filtered by node and energy carrier
        _flt_nd = [('nd', self.model.slct_node)]
        _flt_ca = [('ca', self.model.slct_encar)]
        _flt_pt = ([('pt', self.model.slct_pp_type)]
                   if self.model.slct_pp_type else [])
        dict_tb_3 = {'def_node': _flt_nd,
                     'def_pp_type': _flt_pt,
                     'def_encar': _flt_ca}
        tbrd.df_from_dict(dict_tb_3)

        self.model.slct_node_id = self.model.df_def_node.nd_id.tolist()
        self.model.slct_encar_id = self.model.df_def_encar.ca_id.tolist()
        self.model.slct_pp_type_id = self.model.df_def_pp_type.pt_id.tolist()

        # update filters in case the keyword argument slct_node_id holds more
        # nodes than present in the table
        _flt_nd = [('nd_id', self.model.slct_node_id)]
        _flt_ca = [('ca_id', self.model.slct_encar_id)]
        _flt_nd_2 = [('nd_2_id', self.model.df_def_node.nd_id.tolist())]
        _flt_pt = [('pt_id', self.model.df_def_pp_type.pt_id.tolist())]

        # read input data filtered by node, energy carrier, and fuel
        dict_tb_0 = {'def_plant': _flt_nd + _flt_pt,
                     'profchp': _flt_nd,
                     'node_encar': _flt_nd + _flt_ca,
                     'node_connect': _flt_nd + _flt_ca + _flt_nd_2}
        tbrd.df_from_dict(dict_tb_0)

        # secondary filtering by plant
        _flt_pp = [('pp_id', self.model.df_def_plant['pp_id'].tolist())]
        _flt_fl = [('fl_id', self.model.df_def_plant.fl_id.unique().tolist())]
        dict_tb_1 = {'profinflow': _flt_pp,
                     'plant_encar': _flt_pp + _flt_ca,
                     'hydro': _flt_pp,
                     'def_fuel': _flt_fl,
                     'plant_month': _flt_pp,
                     'plant_week': _flt_pp,
                     'fuel_node_encar': _flt_fl + _flt_nd + _flt_ca}
        tbrd.df_from_dict(dict_tb_1)

        # initialize profile index dicts
        self.model._init_pf_dicts()

        _flt_pf_supply = [('supply_pf_id', list(self.model.dict_supply_pf.values()))]
        _flt_pf_dmnd = [('dmnd_pf_id', list(self.model.dict_dmnd_pf.values()))]
        _flt_pf_price = [('price_pf_id', list(self.model.dict_price_pf.values()))]
        dict_pf_0 = {'profsupply': _flt_pf_supply,
                     'profdmnd': _flt_pf_dmnd,
                     'profprice': _flt_pf_price,
                     }
        tbrd.df_from_dict(dict_pf_0)

        _flt_pf = [('pf_id', _flt_pf_price[-1][-1] + _flt_pf_dmnd[-1][-1] + _flt_pf_supply[-1][-1])]
        dict_pf_1 = {'def_profile': _flt_pf}
        tbrd.df_from_dict(dict_pf_1)

        # filter plants requiring input from non-existing ca
        # e.g. if a fuel-cell is in the input table but no hydrogen is
        # included in the model, the plant's H2 demand wouldn't be accounted
        # for;
        fl_id_ca = self.model.df_def_encar.fl_id.tolist()
        mask_del = (self.model.df_def_fuel.is_ca.isin([1])
                & - self.model.df_def_fuel.fl_id.isin(fl_id_ca))

        self.model.df_def_fuel = self.model.df_def_fuel.loc[-mask_del]

        # filter table by special index name/id columns
        self.model.df_parameter_month = self.filter_by_name_id_cols(
                                            'df_parameter_month',
                                            _flt_fl + _flt_nd + _flt_pp + _flt_ca)

        # autocomplete input tables
        self.data_autocompletion()

        self.fix_df_node_connect()


        input_table_list = (list(dict_tb_1) + list(dict_tb_2)
                            + list(dict_tb_0)+ list(dict_tb_3))




        self.write_input_tables_to_output_schema(input_table_list)


    def data_autocompletion(self):

        if self.autocompletion:
            print('#' * 60)

            ac.AutoCompletePpType(self.model, self.autocomplete_curtailment)
            ac.AutoCompleteFuelTrns(self.model)
            ac.AutoCompleteFuelDmnd(self.model, self.autocomplete_curtailment)
            ac.AutoCompletePlantTrns(self.model)
            ac.AutoCompletePlantDmnd(self.model, self.autocomplete_curtailment)
            ac.AutoCompletePlantCons(self.model)
            ac.AutoCompletePpCaFlex(self.model, self.autocomplete_curtailment)
            print('#' * 60)


    def filter_by_name_id_cols(self, name_df, filt):
        '''
        Filter a pandas DataFrame with index names in columns.

        This operates on pandas DataFrames where the indices are not provided
        as column names but as row entries in special columns.
        E.g., instead of

              | nd_id | fl_id | value |
              |-------|-------|-------|
              | 1     | 2     |   1.2 |

        we have

            | set_1_name | set_2_name | set_1_id | set_2_id | value |
            |------------|------------|----------|----------|-------|
            | nd_id      | fl_id      | 1        | 2        |   1.2 |

        This allows to combine structurally different tables.

        Filtering is implemented as an iteration over the set_n_name/set_n_id
        column pairs, each of which is filtered with respect to all elements
        in the filt parameter.

        -----------------------------------------------------------------------
        Parameters:

        df -- DataFrame as described above
        filt -- filtering list of the same format as for
                aql.read_sql(..., filt)

        -----------------------------------------------------------------------
        Returns:

        df -- filtered DataFrame

        '''

        df = getattr(self.model, name_df)

        if df is not None:

            # perform iterative filtering for each name/id column pair
            for name_col, id_col in [(cc, cc.replace('name', 'id'))
                                     for cc in df.columns if 'name' in cc]:
                # init mask
                mask = False

                # loop over all filter elements
                iflt = filt[0]
                for iflt in filt:
                    mask |= ( # 1. select value for current set_n_name column
                             ((df[name_col] == iflt[0])
                              # 2. select corresponding values for set_n_id col
                               & (df[id_col].isin(iflt[1])))
                              # 3. skip if this name_col is not relevant
                             | (df[name_col] == 'na'))

            # reporting
            report_df = df[[c for c in df.columns if 'set_' in c]]
            report_df = report_df.drop_duplicates()

            # get stackable columns
            newcols = [tuple(c.split('_')[1:]) if '_' in c else (None, c)
                       for c in report_df.columns]
            report_df.columns = pd.MultiIndex.from_tuples(newcols,
                                                          names=['col_n',
                                                                 'col_type'])
            report_df = report_df.stack(level='col_n')

            # get all name values from all name columns
            names = df[[c for c in df.columns if 'name' in c]]
            names = names.stack().unique().tolist()
            names = [nn for nn in names if not nn == 'na']

            # get dictionary {set names: set values}
            names_dict = {nn: list(set(report_df.loc[report_df.name == nn, 'id']))
                          for nn in names}

            print('Ex-post filtering of DataFrame {}:'.format(name_df))
            for kk, vv in names_dict.items():
                print('\tSet {} is in ({})'.format(kk, ', '.join(map(str, vv))))


        return df




    def fix_df_node_connect(self):
        '''
        Makes sure the table df_node_connect corresponds to the new style.

        New style: The transmission capacities are expressed as
        * cap_trme_leg for exports and
        * cap_trmi_leg for imports
        for single directions, i.e. non-redundant. The input table has columns
        (nd_id, nd_2_id, ca_id, mt_id, cap_trme_leg, cap_trmi_leg).

        Old style: Single transmission capacity for both directions; columns:
        (nd_id, nd_2_id, ca_id, mt_id, eff, cap_trm_leg)
        '''

        if 'cap_trm_leg' in self.model.df_node_connect.columns:

            df = self.model.df_node_connect

            df['dir'] = df.nd_id < df.nd_2_id

            df_e = df.loc[df.dir].assign(nd_id = df.nd_2_id,
                                         nd_2_id = df.nd_id,
                                         cap_trmi_leg = df.cap_trm_leg)
            df = df.loc[-df.dir].assign(cap_trme_leg = df.cap_trm_leg)
            dfn = pd.concat([df_e, df], sort=False)
            dfn = dfn.drop('cap_trm_leg', axis=1).fillna(0)
            idx = ['nd_id', 'nd_2_id', 'ca_id', 'mt_id']
            print('Aggregation count in fix_df_node_connect:\n',
                  dfn.pivot_table(index=idx, aggfunc=[min, max, len],
                                  values=['cap_trme_leg', 'cap_trmi_leg']))
            dfn = dfn.pivot_table(index=idx, aggfunc=sum,
                                  values=['cap_trme_leg', 'cap_trmi_leg'])


            self.model.df_node_connect = dfn.reset_index()

    @skip_if_resume_loop
    def write_input_tables_to_output_schema(self, tb_list):
        '''
        TODO: Input tables in output schema are required even if no_output.
              Check where + fix. Ideally this would have decorator
              skip_if_no_output.
        '''

        for itb in tb_list:
            df = getattr(self.model, 'df_' + itb)
            if (df is not None
                and not ((not 'def_' in itb) and self.no_output)
                and not 'prof' in itb):
                print('Writing table {} to output schema.'.format(itb))
                engine = self.sql_connector.get_sqlalchemy_engine()
                db = self.sql_connector.db
                aql.write_sql(df, db, self.sc_out, itb,
                              if_exists='replace', engine=engine)

    @skip_if_resume_loop
    @skip_if_no_output
    def write_runtime_tables(self):
        '''
        Some input tables depend on model parameters (time resolution).
        Write these to output database schema.
        Also, table def_node altered due to addition of column dmnd_max.
        '''
        skip_fks = [('tm_soy', 'sy'),  # defines sy
                    ('hoy_soy', 'hy')]  # defines hy

        engine = self.sql_connector.get_sqlalchemy_engine()
        con_cur = self.sql_connector.get_pg_con_cur()

        tb_name, pk = ('tm_soy', ['sy'])
        for tb_name, pk in [('tm_soy', ['sy']),
                            ('hoy_soy', ['hy']),
                            ('tm_soy_full', ['sy']),
                            ]:

            if hasattr(self.model, 'df_' + tb_name):
                df = getattr(self.model, 'df_' + tb_name)

                print('Writing runtime table ' + tb_name)

                cols = []
                c = 'DateTime'#df.columns[0]
                for c in df.columns:
                    col_add = [c]

                    if c not in self._coldict: # same as "value"
                        self._coldict[c] = self._coldict['value']

                    col_add += (list(self._coldict[c])
                                 if (tb_name, c) not in skip_fks
                                 else list(self._coldict[c][:1]))
                    cols.append(tuple(col_add))

                aql.init_table(tb_name=tb_name, cols=cols, schema=self.sc_out,
                               ref_schema=self.sc_out, pk=pk, unique=[],
                               db=self.sql_connector.db, con_cur=con_cur)

                aql.write_sql(df, sc=self.sc_out, tb=tb_name,
                              if_exists='append', engine=engine,
                              con_cur=con_cur)

class IO:
    '''

    '''

    def __init__(self, **kwargs):



        defaults = {'sc_warmstart': False,
                    'resume_loop': False,
                    'replace_runs_if_exist': False,
                    'model': None,
                    'autocomplete_curtailment': False,
                    'sql_connector': None,
                    'autocompletion': True,
                    'no_output': False,
                    'dev_mode': False,
                    'data_path': None,
                    'sc_inp': None,
                    'sc_out': None,
                    'db': None,
                    }

        defaults.update(kwargs)

        self.datrd = DataReader(**defaults)
        self.modwr = ModelWriter(**defaults)

        self.resume_loop = defaults['resume_loop']
        self.sql_connector = defaults['sql_connector']
        self.replace_runs_if_exist = defaults['replace_runs_if_exist']
        self.db = self.sql_connector.db

    @classmethod
    def variab_to_df(cls, py_obj, sets):
        ''' Wrapper for backward compatibility. '''

        return VariabIO._to_df(py_obj, sets)

    @classmethod
    def param_to_df(cls, py_obj, sets):
        ''' Wrapper for backward compatibility. '''

        return ParamIO._to_df(py_obj, sets)

    def read_model_data(self):

        self.datrd.read_model_data()

    def write_runtime_tables(self):

        self.datrd.write_runtime_tables()

    def init_output_tables(self):

        self.modwr.init_compio_objs()
        self.modwr.init_all()

    def write_run(self, run_id):

        self.modwr.run_id = run_id
        self.modwr.write_all()
# %%
if __name__ == '__main__':


    pass
