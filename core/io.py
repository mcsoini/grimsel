'''
Contains the class IO which handles all input/output.
Note: It would be better if all interaction with SQL where concentrated in
a dedicated method/class
'''

import sys, os

import pandas as pd
from pandas.io.sql import SQLTable
import pyomo.environ as po
from pyomo.core.base.param import _ParamData # required if params are mutable
import psycopg2 as pg
from sqlalchemy import create_engine
import itertools
from importlib import reload
import time

import grimsel.auxiliary.aux_sql_func as aql
from grimsel.auxiliary.aux_general import get_config, print_full
import grimsel.core.autocomplete as ac
import grimsel


reload(ac)
reload(aql)

#def _execute_insert(self, conn, keys, data_iter):
##    print("Using monkey-patched _execute_insert")
#    data = [dict((k, v) for k, v in zip(keys, row)) for row in data_iter]
#    conn.execute(self.insert_statement().values(data))
#SQLTable._execute_insert = _execute_insert

class IO():
    '''
    Input/Output functionalities.
    An instance of IO is owned by the ModelLoop class.

    Note: bool_out columns are added here. Description:
    True if energy outflow from the node's perspective,
    i.e. charging, exports, demand etc. False otherwise.
    '''

    # definition of output tables with column names. These mirror
    # the model components.
    var_sy = [
              ('pwr', ('sy', 'pp_id', 'ca_id', 'bool_out')),
              ('dmnd_flex', ('sy', 'nd_id', 'ca_id', 'bool_out')),
              ('pwr_st_ch', ('sy', 'pp_id', 'ca_id', 'bool_out'), 'pwr'),
              ('erg_st', ('sy', 'pp_id', 'ca_id'))
             ]
    var_mt = [
        ('erg_mt', ('mt_id', 'pp_id', 'ca_id'))]
    var_yr = [
        ('erg_yr', ('pp_id', 'ca_id', 'bool_out')),
        ('erg_fl_yr', ('pp_id', 'nd_id', 'ca_id', 'fl_id')),
        ('pwr_ramp_yr', ('pp_id', 'ca_id')),
        ('vc_fl_pp_yr', ('pp_id', 'ca_id', 'fl_id')),
        ('vc_ramp_yr', ('pp_id', 'ca_id')),
        ('vc_co2_pp_yr', ('pp_id', 'ca_id')),
        ('vc_om_pp_yr', ('pp_id', 'ca_id')),
        ('fc_om_pp_yr', ('pp_id', 'ca_id')),
        ('fc_cp_pp_yr', ('pp_id', 'ca_id')),
        ('fc_dc_pp_yr', ('pp_id', 'ca_id')),
        ('vc_dmnd_flex_yr', ('nd_id', 'ca_id')),
        ('cap_pwr_rem', ('pp_id', 'ca_id')),
        ('cap_pwr_tot', ('pp_id', 'ca_id')),
        ('cap_erg_tot', ('pp_id', 'ca_id')),
        ('cap_pwr_new', ('pp_id', 'ca_id')),
        ('erg_ch_yr', ('pp_id', 'ca_id', 'bool_out'), 'erg_yr')]
    var_tr = [
        ('trm_sd', ('sy', 'nd_id', 'nd_2_id', 'ca_id', 'bool_out')),
        ('trm_rv', ('sy', 'nd_id', 'nd_2_id', 'ca_id', 'bool_out')),
        ('erg_trm_rv_yr', ('nd_id', 'nd_2_id', 'ca_id', 'bool_out')),
        ('erg_trm_sd_yr', ('nd_id', 'nd_2_id', 'ca_id', 'bool_out'))]
    par = [
        ('share_ws_set', ('nd_id',)),
        ('price_co2', ('nd_id',)),
        ('co2_int', ('fl_id',)),
        ('cap_pwr_leg', ('pp_id', 'ca_id')),
        ('cap_avlb', ('pp_id', 'ca_id')),
        ('cap_trm_leg', ('mt_id', 'nd_id', 'nd_2_id', 'ca_id')),
        ('weight', ('sy',)),
        ('cf_max', ('pp_id', 'ca_id')),
        ('grid_losses', ('nd_id', 'ca_id')),
        ('erg_max', ('nd_id', 'ca_id', 'fl_id')),
        ('hyd_pwr_in_mt_max', ('pp_id',)),
        ('hyd_pwr_out_mt_min', ('pp_id',)),
        ('vc_dmnd_flex', ('nd_id', 'ca_id')),
        ('vc_fl', ('fl_id', 'nd_id')),
        ('vc_om', ('pp_id', 'ca_id')),
        ('fc_om', ('pp_id', 'ca_id')),
        ('fc_dc', ('pp_id', 'ca_id')),
        ('fc_cp_ann', ('pp_id', 'ca_id')),
        ('ca_share_min', ('pp_id', 'ca_id')),
        ('ca_share_max', ('pp_id', 'ca_id')),
        ('pp_eff', ('pp_id', 'ca_id')),
        ('vc_ramp', ('pp_id', 'ca_id')),
        ('st_lss_hr', ('pp_id', 'ca_id')),
        ('st_lss_rt', ('pp_id', 'ca_id')),
        ('hyd_erg_bc', ('sy', 'pp_id')),
        ('hyd_erg_min', ('pp_id',)),
        ('inflowprof', ('sy', 'pp_id', 'ca_id')),
        ('chpprof', ('sy', 'nd_id', 'ca_id')),
        ('supprof', ('sy', 'pp_id', 'ca_id')),
        ('priceprof', ('sy', 'nd_id', 'fl_id')),
        ('week_ror_output', ('wk', 'pp_id')),
        ('dmnd', ('sy', 'nd_id', 'ca_id')),
        ('erg_inp', ('nd_id', 'ca_id', 'fl_id')),
        ('erg_chp', ('nd_id', 'ca_id', 'fl_id')),
        ('capchnge_max', tuple()),
#        ('objective', tuple()),
#        ('objective_lin', tuple()),
#        ('objective_quad', tuple())
        ]
    dual = [('supply', ('sy', 'nd_id', 'ca_id'))]

    list_collect = ['var_sy', 'var_mt', 'var_yr', 'par', 'dual']

    loop_pk = ['run_id']

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

    def __init__(self, **kwargs):

        self.run_id = None  # set in call to self.write_run

        # define instance attributes and update with kwargs
        defaults = {'sc_warmstart': False,
                    'resume_loop': False,
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
        for key, val in defaults.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)

        self.loop_cols = [('run_id', 'SMALLINT')]

#        # read database connection params from config file
        self.conn = pg.connect(self.sql_connector.pg_str)
        self.engine = create_engine(self.sql_connector.sqlal_str)
        self.cur = self.conn.cursor()

        print('Output schema: ', self.sc_out,
              '; resume loop=', self.resume_loop)
        if not self.resume_loop:
            aql.reset_schema(self.sc_out, self.db, not self.dev_mode)

        # copying since we need to modify
        self._coldict = aql.get_coldict(self.sc_out, self.db)

        # values for chg_bool columns
        self.chg_dict = {'var_sy_pwr': False,
                         'var_sy_pwr_st_ch': True,
                         'var_yr_erg_yr': False,
                         'var_yr_erg_ch_yr': True,
                         'var_tr_trm_sd': True,
                         'var_tr_trm_rv': False,
                         'var_tr_erg_trm_rv_yr': False,
                         'var_tr_erg_trm_sd_yr': True,
                         'dmnd': True,
                         'var_sy_dmnd_flex': True}

        # List of all tables; generated in method delete_run_id
        self.list_all_tb = []

        # SOME SPECIAL TREATMENT FOR CROSS-BORDER TRANSMISSION AND DEMAND
        self.dict_cross_table = {
                                 'erg_trm_rv_yr': 'var_yr_erg_yr',
                                 'erg_trm_sd_yr': 'var_yr_erg_yr',
                                 'trm_rv': 'var_sy_pwr',
                                 'trm_sd': 'var_sy_pwr',
                                 'dmnd': 'var_sy_pwr',
#                                 'dmnd_flex': 'var_sy_pwr'
                                 }

        self.dict_tcname = {'trm_sd': 'TRNS_ST',
                            'trm_rv': 'TRNS_RV',
                            'erg_trm_rv_yr': 'TRNS_RV',
                            'erg_trm_sd_yr': 'TRNS_ST',
                            'dmnd': 'DMND',
                            'dmnd_flex': 'DMND_FLEX'}

        if self.model.skip_runs:
            print('model.skip_runs is True; modifying IO lists.')
            # only write parameter output if the model runs are skipped
            for lst in [c for c in self.list_collect if not c == 'par']:
                setattr(self, lst, [])

            _obj = [c for c in self.par if 'objective' in c][0]
            self.par.remove(_obj)


    def sanitize_component_lists(self):

        '''
        TODO: Make sure the second elements are all tuples.
        '''


    @skip_if_no_output
    def init_output_tables(self):
        '''
        Call initialization method in output writing class.
        This is done here to avoid resume_loop in the latter.
        '''
        if not self.resume_loop:
            self.init_output_database()
        else:
            # delete run_ids equal or greater the resume_loop run_id
            self.delete_run_id(self.resume_loop)


    def get_input_table(self, table, filt):

        if self.sc_inp:
            tb_exists = table in aql.get_sql_tables(self.sc_inp, self.db)
            if tb_exists:
                df = aql.read_sql(self.db, self.sc_inp, table, filt)
        else:
            if self.data_path:
                path = self.data_path
            else:
                path = os.path.join(grimsel.__path__[0], 'input_data')
            fn = os.path.join(path, '%s.csv'%table)

            tb_exists = os.path.exists(fn)

            if tb_exists:
                df = pd.read_csv(fn)

                for col, vals in filt:

                    df = df.loc[df[col].isin(vals)]

        return (df if tb_exists else None), tb_exists


    def read_model_data(self):
        '''
        Read all input data to instance attributes.
        '''

        def df_from_dict(dct):
            ''' Reads filtered input tables and assigns them to instance
                attributes. (Copies filtered tables to output schema.
                OBSOLETE DUE TO AUTO-COMPLETE) '''

            for kk, vv in dct.items():

                df, tb_exists = self.get_input_table(kk, vv)

                setattr(self.model, 'df_' + kk, df)

                if not tb_exists:
                    print('Input table {tb} does not exist. '
                          + 'Setting model attribute df_{tb} '
                          + 'to None.'.format(tb=kk))
                else:
                    filt = ('filtered by ' if len(vv) > 0 else '') +\
                       ', '.join([vvv[0] + ' in ' + str(vvv[1]) for vvv in vv
                                  if not len(vvv[1]) is 0])
                    print('Reading input table {tb} {flt}'.format(tb=kk,
                                                                  flt=filt))


        # unfiltered input
        dict_tb_2 = {'def_month': [], 'def_week': [],
                     'parameter_month': [], 'def_plant': []}
        df_from_dict(dict_tb_2)

        # read input data filtered by node and energy carrier
        _flt_nd = [('nd', self.model.slct_node)]
        _flt_ca = [('ca', self.model.slct_encar)]
        _flt_pt = ([('pt', self.model.slct_pp_type)]
                   if self.model.slct_pp_type else [])
        dict_tb_3 = {'def_node': _flt_nd,
                     'def_pp_type': _flt_pt,
                     'def_encar': _flt_ca}
        df_from_dict(dict_tb_3)

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
                     'profdmnd': _flt_nd + _flt_ca,
                     'profchp': _flt_nd,
                     'profprice': _flt_nd,
                     'node_encar': _flt_nd + _flt_ca,
                     'node_connect': _flt_nd + _flt_ca + _flt_nd_2}
        df_from_dict(dict_tb_0)



        # secondary filtering by plant
        _flt_pp = [('pp_id', self.model.df_def_plant['pp_id'].tolist())]
        _flt_fl = [('fl_id', self.model.df_def_plant.fl_id.unique().tolist())]
        dict_tb_1 = {'profinflow': _flt_pp,
                     'profsupply': _flt_pp,
                     'plant_encar': _flt_pp + _flt_ca,
                     'hydro': _flt_pp,
                     'def_fuel': [],
                     'plant_month': _flt_pp,
                     'plant_week': _flt_pp,
                     'fuel_node_encar': _flt_fl + _flt_nd + _flt_ca}
        df_from_dict(dict_tb_1)

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


        input_table_list = (list(dict_tb_1.keys()) + list(dict_tb_2.keys())
                            + list(dict_tb_0.keys())+ list(dict_tb_3.keys()))

        self.write_input_tables_to_output_schema(input_table_list)

    def write_input_tables_to_output_schema(self, tb_list):

        for itb in tb_list:

            df = getattr(self.model, 'df_' + itb)
            if df is not None:
                print('Writing table {} to output schema.'.format(itb))
#                aql.copy_table_structure(sc=self.sc_out, tb=itb,
#                                         sc0=self.sc_inp, db=self.db)
                aql.write_sql(df, self.db, self.sc_out, itb,
                              if_exists='replace')

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


    def data_autocompletion(self):

        if self.autocompletion:
            print('#' * 60)

            ac.AutoCompletePpType(self.model)
            ac.AutoCompleteFuelTrns(self.model)
            ac.AutoCompleteFuelDmnd(self.model)
            ac.AutoCompletePlantTrns(self.model)
            ac.AutoCompletePlantDmnd(self.model)
            ac.AutoCompletePlantCons(self.model)
            if self.autocomplete_curtailment:
                ac.AutoCompletePpCaFlex(self.model)
            print('#' * 60)

    @skip_if_no_output
    def write_run(self, run_id):
        '''
        Main call of methods in output writing class.
        '''

        if self.model.skip_runs:
            # write only parameters

            for write_slct in self.list_collect:
                for comp in getattr(self, write_slct):
                    if 'var' in comp[0]:
                        comp


        self.run_id = run_id
        self.write_prim_data('var_sy')
        self.write_prim_data('var_yr')
        self.write_prim_data('var_mt')
        self.write_parameters('par')
        self.write_duals('dual')
        if len(self.model.slct_node) > 1: # or better based on pyomo attribute?
            self.write_prim_data('var_tr')

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

        tb_name, pk = ('tm_soy', ['sy'])
        for tb_name, pk in [('tm_soy', ['sy']),
                            ('hoy_soy', ['hy']),
#                            ('profsupply_soy', ['sy', 'pp_id', 'ca_id']),
#                            ('profinflow_soy', ['sy', 'pp_id', 'ca_id']),
#                            ('profchp_soy', ['sy', 'nd_id', 'ca_id']),
#                            ('profdmnd_soy', ['sy', 'nd_id', 'ca_id']),
#                            ('profprice_soy', ['sy', 'nd_id', 'fl_id']),
                            ('tm_soy_full', ['sy']),
                            ]:

            if 'df_' + tb_name in self.model.__dict__:
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
                               db=self.db)

                aql.write_sql(df, self.db, self.sc_out, tb_name, 'append')



#        # update table in output schema
#        if 'df_profdmnd_soy' in self.model.__dict__:
#            aql.add_column(df_src=self.model.df_def_node,
#                            tb_tgt=[self.model.sc_out, 'def_node'],
#                            col_new='dmnd_max', on_cols=['nd_id'],
#                            db=self.db)

    @skip_if_no_output
    def init_output_database(self):
        '''
        Creation of output schema and initialization of tables.
        Loops over all entries in the dictionaries defined by list_collect,
        which again contain all table names for parameters, variables, etc.

        Note: Keys need to be added in post-processing due to table
        writing performance.

        Makes use of init_table function in aux_sql_func.
        '''
        aql.exec_sql('CREATE SCHEMA IF NOT EXISTS ' + self.sc_out,
                     db=self.db)

        conn, cur = self.sql_connector.get_pg_con_cur()

        for ilist in self.list_collect:

            # elements of length 3 don't get their own table as they are
            # appended to a different one
            table_list = [tt for tt in getattr(self, ilist)
                          if not len(tt) == 3]

            for iv in table_list:
                tb_name = ilist + '_' + iv[0]
                print('Initializing output table ', tb_name)
                col_names = [s for s in iv[1]] + ['value']
                cols = [(c,) + (self._coldict[c][0],) for c in col_names]
                cols += self.loop_cols
                # pk now added later for writing/appending performance
                pk = []#list(iv[1]) + self.loop_pk
                unique = []

                aql.init_table(tb_name=tb_name, cols=cols,
                               schema=self.sc_out,
                               ref_schema=self.sc_out, pk=pk,
                               unique=unique, bool_auto_fk=False, db=self.db,
                               con_cur=(conn, cur))
        aql.close_con(conn)


    def _finalize(self, _df, sql_tb):
        ''' Add run_id column and write to database table '''
        _df['run_id'] = self.run_id
        print('Writing to db: ', self.sc_out + '.' + sql_tb, end='... ')

        t = time.time()
        _df.to_sql(sql_tb, self.engine, schema=self.sc_out,
                   if_exists='append', index=False)
        print('done in %.3f sec'%(time.time() - t))


#        t = time.time()
#        db_uri = ('postgresql://postgres:postgres'
#                   '@localhost:5432/storage2::%s'%sql_tb)
#        odo.odo(_df, db_uri, schema=self.sc_out)
#        print(time.time() - t)
#



    def _node_to_plant(self, tbname):
        '''
        TODO: THIS SHOULD BE IN MAPS!!!!
        Method for translation of node_id to respective plant_id
        in the cases of demand and inter-node transmission. This is used to
        append demand/inter-nodal transmission to pwr table.
        Returns a dictionary node -> plant
        Keyword arguments:
        tbname -- table name, serves as key to the dict_tcname
        '''
        df_np = self.model.df_def_plant[['nd_id', 'pp_id', 'pp', 'pt_id']]
        df_pt = self.model.df_def_pp_type[['pt_id', 'pt']]
        slct_pp_type = df_pt.loc[df_pt['pt'] == self.dict_tcname[tbname],
                                 'pt_id'].iloc[0]
        mask_tech = df_np['pt_id'] == slct_pp_type
        df_np_slct = df_np.loc[mask_tech]
        dict_node_plant_slct = df_np_slct.set_index('nd_id')
        dict_node_plant_slct = dict_node_plant_slct['pp_id'].to_dict()
        return dict_node_plant_slct


    @classmethod
    def variab_to_df(cls, py_obj, sets):

        # replace none with zeros
        for v in py_obj.iteritems():
            if v[1].value is None:
                v[1].value = 0

        dat = [v[0] + (v[1].value,) for v in py_obj.iteritems()]

        # make dataframe
        cols = list(sets + ('value',))
        _df = pd.DataFrame(dat, columns=[c for c in cols
                                         if not c == 'bool_out'])

        return _df, cols


    def write_prim_data(self, write_slct='var_tr'):
        '''
        Write
        '''


        write_list = getattr(self, write_slct)
        variabs = ('pwr', ('sy', 'pp_id', 'ca_id', 'bool_out'))
        for variabs in write_list:

            if not variabs[0] in [c.name for c in self.model.component_objects()]:
                print('Component ' + variabs[0] + ' does not exist... skipping.')
            else:
                py_obj = getattr(self.model, variabs[0])

                _df, cols = self.variab_to_df(py_obj, variabs[1])

                # table name as used in the database
                tb_name = write_slct + '_' + variabs[0]

                # add bool_out column according to corresponding dictionary
                if 'bool_out' in cols:
                    _df['bool_out'] = self.chg_dict[tb_name]


                # for plants in sell-set 'sll' input and output are reversed;
                # therefore, bool_out is True, i.e. encar is consumed and fuel
                # sold
                lst_true = (self.model.setlst['sll']
                            + self.model.setlst['curt'])
                if (len(lst_true) > 0
                    and 'pp_id' in _df.columns and 'bool_out' in _df.columns):
                    _df.loc[_df.pp_id.isin(lst_true), 'bool_out'] = True

                # some variables don't get their own tables; this is defined
                # through the third element in the variabs tuple
                # (e.g. pwr_st_ch)
                if len(variabs) == 3: # third item is target table name
                    tb_name = write_slct + '_' + variabs[2]

                self._finalize(_df[cols], tb_name)

                # inter-node transmission is aggregated and added to power
                # production table
                if variabs[0] in self.dict_cross_table:

                    # aggregate all node_2
                    idx = [c for c in _df.columns
                           if not c in ['value', 'nd_2_id']]
                    _df = _df.pivot_table(values='value', aggfunc=sum, index=idx)
                    _df = _df.reset_index()

                    dict_node_plant_rvsd = self._node_to_plant(variabs[0])

                    _df['nd_id'] = _df['nd_id'].replace(dict_node_plant_rvsd)
                    _df = _df.rename(columns={'nd_id': 'pp_id'})

                    # select table
                    tb_name = self.dict_cross_table[variabs[0]]

                    self._finalize(_df, tb_name)



    @classmethod
    def param_to_df(cls, py_obj, sets):

        dat = []
        for v in py_obj.iteritems():
            v = list(v)
            if not v[0] == None:
                if type(v[1]) == _ParamData:
                    # if parameter is mutable v[1] is a _ParamData;
                    # requires manual extraction of value
                    v[1] = v[1].value
                if v[1] == None:
                    v[1] = 0
                if not type(v[0]) == tuple:
                    v_sets = [v[0]]
                else:
                    v_sets = [iv for iv in v[0]]
                dat += [v_sets + [v[1]]]
            else:
#                if type(py_obj) == SimpleObjective:
#                    # not exactly a parameter but hey, it fits
#                    objective_value = po.value(py_obj)
#                    dat = [objective_value]
#                else:
                dat = [v[1].extract_values()[None]]

        return pd.DataFrame(dat, columns=sets + ('value',))



    def write_parameters(self, write_slct):
        '''
        Write all model parameters as defined in the self.par list.
        '''
        write_list = getattr(self, write_slct)
        for variabs in write_list:

            if not variabs[0] in [c.name for c
                                  in self.model.component_objects()]:
                print('Component {} does not exist... skipping.'
                      .format(variabs[0]))
            else:
                py_obj = getattr(self.model, variabs[0])
                _df = self.param_to_df(py_obj, variabs[1])

                self._finalize(_df, write_slct + '_' + variabs[0])

                # demand data is aggregated and added to power production table
                if variabs[0] in self.dict_cross_table:

                    # add bool_out column
                    _df['bool_out'] = self.chg_dict[variabs[0]]

                    dict_node_plant_dmnd = self._node_to_plant(variabs[0])

                    _df['nd_id'] = _df['nd_id'].replace(dict_node_plant_dmnd)
                    _df = _df.rename(columns={'nd_id': 'pp_id'})

                    # select table
                    tb_name = self.dict_cross_table[variabs[0]]

                    self._finalize(_df, tb_name)


    def write_duals(self, write_slct):
        '''
        Write dual variable values.
        '''
        for idu in self.dual:
            py_obj = getattr(self.model, idu[0])
            dat = [ico + (self.model.dual[py_obj[ico]],) for ico in py_obj]
            _df = pd.DataFrame(dat, columns=idu[1] + ('value',))
            self._finalize(_df, write_slct + '_' + idu[0])

    def delete_run_id(self, _run_id=False):
        '''
        In output tables delete all rows with run_id >= the selected value.
        '''
        # Get overview of all tables
        list_all_tb_0 = [list(itb_list + '_' + itb[0] for itb
                              in getattr(self, itb_list)
                              if not len(itb) == 3)
                         for itb_list in self.list_collect]
        self.list_all_tb = list(itertools.chain(*list_all_tb_0))
        self.list_all_tb += ['def_loop']

        if _run_id:
            for itb in self.list_all_tb:

                print('Deleting from ', self.sc_out + '.' + itb
                      + ' where run_id >= ' + str(_run_id))
                exec_str = ('''
                            DELETE FROM {sc_out}.{tb}
                            WHERE run_id >= {run_id};
                            ''').format(sc_out=self.sc_out, tb=itb,
                                        run_id=_run_id)
                try:
                    aql.exec_sql(exec_str, db=self.db)
                except pg.ProgrammingError as e:
                    print(e)
                    sys.exit()

    @skip_if_no_output
    def write_df_to_sql(self, df, tb, if_exists):

        # write to database
        aql.write_sql(df, self.db, self.sc_out, tb, if_exists)


    @classmethod
    def post_process_index(cls, sc, db, drop=False):
        ''' Add indices to all tables defined by the IO class. '''

        print('Adding indices to all output tables. This might take a while.')
        _coldict = aql.get_coldict(sc, db)

        # add primary and foreign keys to all tables
        ilist = cls.list_collect[0]
        for ilist in cls.list_collect:
            table_list = getattr(cls, ilist)

            iv = table_list[0]
            for iv in table_list:

                tb_name = ilist + '_' + iv[0]

                print('tb_name:', tb_name)

                pk_list = list(iv[1]) + cls.loop_pk

                fk_dict = {}
                for c in pk_list:
                    if len(_coldict[c]) > 1:
                        fk_dict[c] = _coldict[c][1]


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


    @classmethod
    def post_process_storage_mismatch(cls, sc, drop=False):
        '''
        By default the combinations
        swst_vl=LIO/pt_id=CAS_STO
        swst_vl=CAS/pt_id=LIO_STO
        which are always zero by definition are still contained in the tables.
        Here we loop over all tables, search for column combinations of
        pp_id and run_id, and delete the corresponding rows by moving them
        into tables with suffix _del if drop is False.
        Keyword arguments:
        drop -- boolean; if True, selected rows get deleted instead of being
                moved to new tables
        '''

        # add primary and foreign keys to all tables
        ilist = cls.list_collect[0]
        for ilist in cls.list_collect:
            table_list = getattr(cls, ilist)

            for table in table_list:
                cols = aql.get_sql_cols(ilist +'_' + table[0], sc=sc)

                format_kw = {'tb': ilist +'_' + table[0],
                             'sc': sc}

                if 'run_id' in cols and 'pp_id' in cols:


                    list_pt = aql.exec_sql('''
                                           SELECT DISTINCT pt
                                           FROM {sc}.{tb} AS tb
                                           LEFT JOIN {sc}.def_plant AS dfpp
                                               ON dfpp.pp_id = tb.pp_id
                                           LEFT JOIN {sc}.def_pp_type AS dfpt
                                               ON dfpt.pt_id = dfpp.pt_id
                                           ;
                                           '''.format(**format_kw))
                    print(list_pt)
                    if ('CAS_STO', ) in list_pt or ('LIO_STO',) in list_pt:



                        exec_str = '''
                                   DELETE FROM {sc}.{tb} AS tb
                                   USING
                                       {sc}.def_loop AS dflp,
                                       {sc}.def_plant AS dfpp,
                                       {sc}.def_pp_type AS dfpt
                                   WHERE dflp.run_id = tb.run_id
                                   AND dfpp.pp_id = tb.pp_id
                                   AND dfpt.pt_id = dfpp.pt_id
                                   AND (pt IN ('CAS_STO', 'LIO_STO')
                                        AND NOT swtc_vl::VARCHAR || '_STO' = pt)
                                   '''.format(**format_kw)

                        if not drop:
                            aql.exec_sql('''
                                         DROP TABLE IF EXISTS
                                         {sc}.{tb}_del CASCADE;
                                         '''.format(**format_kw))

                            exec_str = '''
                                       WITH del_tb AS (
                                       {exec_str}
                                       RETURNING tb.*
                                       )
                                       SELECT *
                                       INTO {sc}.{tb}_del
                                       FROM del_tb;
                                       '''.format(**format_kw, exec_str=exec_str)
                        print(exec_str)
                        aql.exec_sql(exec_str)





# %%
if __name__ == '__main__':
    pass
#    IO.post_process_index(ml.m.sc_out, ml.io.db)

#    IO.post_process_index('out_1_1823')





