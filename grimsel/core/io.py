#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:03:42 2019

@author: user
"""
import time
import itertools
import os
import tables
import shutil
from glob import glob

import fastparquet as pq
import numpy as np
import pandas as pd
import psycopg2 as pg

import grimsel
import grimsel.auxiliary.sqlutils.aux_sql_func as aql
import grimsel.core.autocomplete as ac
import grimsel.core.table_struct as table_struct
from grimsel import _get_logger

logger = _get_logger(__name__)


FORMAT_RUN_ID = '{:04d}'  # modify for > 9999 model runs


class _HDFWriter:
    ''' Mixing class for :class:`CompIO` and :class:`DataReader`. '''

    def write_hdf(self, tb, df, put_append):
        '''
        Opens connection to HDF file and writes output.

        Parameters
        ----------
        put_append: str, one of `('append', 'put')`
            Write one-time table to the output file (`'put'`) or append
            to existing table (`'append'`).

        '''

        with pd.HDFStore(self.cl_out, mode='a') as store:

            method_put_append = getattr(store, put_append)
            method_put_append(tb, df, data_columns=True, format='table',
                              complevel=9, complib='blosc:blosclz')

class _ParqWriter:
    ''' Mixing class for :class:`CompIO` and :class:`DataReader`. '''

    def write_parquet(self, fn, df, engine):
        '''
        Opens connection to HDF file and writes output.

        Parameters
        ----------
        fn: str
            filename for table writing
        df: pandas DataFrame
            table to be written
        engine: str
            engine name as in the pandas DataFrame.to_parquet parameter
        '''

        if self.output_target == 'fastparquet':
            df.to_parquet(fn, engine='fastparquet',
                          compression='gzip',)

#            if 'run_id' in df.columns:
#                df.to_parquet(fn, #append=os.path.isfile(fn),
#                              engine='fastparquet',
#                              compression='gzip',
#                              partition_cols=['run_id'])
#            else:
#                df.to_parquet(fn, #append=os.path.isfile(fn),
#                              engine='fastparquet',
#                              compression='gzip'
#                              )


        else:
            raise RuntimeError('Writing using parquet engine %s '
                               'not implemented.'%self.output_target)



class CompIO(_HDFWriter, _ParqWriter):
    '''
    A CompIO instance takes care of extracting a single variable/parameter from
    the model and of writing a single table to the database.
    '''

    def __init__(self, tb, cl_out, comp_obj, idx, connect, output_target,
                 model=None):

        self.tb = tb
        self.cl_out = cl_out
        self.comp_obj = comp_obj
        self.output_target = output_target
        self.connect = connect
        self.model = model

        self.columns = None  # set in index setter
        self.run_id = None  # set in call to self.write_run

        self.index = tuple(idx) if not isinstance(idx, tuple) else idx

        self.coldict = aql.get_coldict()

    def post_processing(self, df):
        ''' Child-specific method called after reading. '''
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

        Calls the :func:`aux_sql_func` method with appropriate parameters.

        .. note:
           Keys need to be added in post-processing due to table
           writing performance.

        '''

        logger.info('Generating output table {}'.format(self.tb))
        col_names = self.index + ('value',)
        cols = [(c,) + (self.coldict[c][0],) for c in col_names]
        cols += [('run_id', 'SMALLINT')]
        pk = []  # pk added later for writing/appending performance
        unique = []

        aql.init_table(tb_name=self.tb, cols=cols,
                       schema=self.cl_out,
                       ref_schema=self.cl_out, pk=pk,
                       unique=unique, bool_auto_fk=False, db=self.connect.db,
                       con_cur=self.connect.get_pg_con_cur())


    def _to_file(self, df, tb):
        '''
        Casts the data types of the output table and writes the
        table to the output HDF file.

        '''

        dtype_dict = {'value': np.dtype('float64'),
                      'bool_out': np.dtype('bool')}
        dtype_dict.update({col: np.dtype('int32') for col in df.columns
                           if not col in ('value', 'bool_out')})

        df = df.astype({col: dtype for col, dtype in dtype_dict.items()
                        if col in df.columns})


        if self.output_target == 'hdf5':
            self.write_hdf(tb, df, 'append')

        elif self.output_target in ['fastparquet']:

            fn = os.path.join(self.cl_out,
                              tb + ('_%s'%FORMAT_RUN_ID).format(self.run_id) + '.parq')

            self.write_parquet(fn, df, engine=self.output_target)

        else:
            raise RuntimeError('_to_file: no '
                               'output_target applicable')


    def _to_sql(self, df, tb):

        df.to_sql(tb, self.connect.get_sqlalchemy_engine(),
                  schema=self.cl_out, if_exists='append', index=False)

    def _finalize(self, df, tb=None):
        ''' Add run_id column and write to database table '''

        tb = self.tb if not tb else tb
        logger.info('Writing {} to {}.{}'.format(self.comp_obj.name,
                                                 self.cl_out, tb))

        # value always positive, directionalities expressed through bool_out
        df['value'] = df['value'].abs()

        df['run_id'] = self.run_id

        t = time.time()

        if self.output_target in ['hdf5', 'fastparquet']:
            self._to_file(df, tb)
        elif self.output_target == 'psql':
            self._to_sql(df, tb)
        else:
            raise RuntimeError('_finalize: no '
                               'output_target applicable')

        logger.info(' ... done in %.3f sec'%(time.time() - t))

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        ''' Makes sure idx is tuple and updates columns attribute. '''
        self._index = (value,) if not isinstance(value, tuple) else value
        self.columns = list(self.index + ('value',))
        self.columns = [c for c in self.columns if not c == 'bool_out']

    def get_df(self):

        df = self.to_df()
        df = self.post_processing(df)

        return df

    def write(self, run_id):

        self.run_id = run_id

        df = self.get_df()

        self._finalize(df)

    def _node_to_plant(self, pt):
        '''
        TODO: THIS SHOULD BE IN MAPS!!!!
        TODO: THIS SHOULD ALSO INCLUDE ca_id FOR DMND!
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
               for ico in self.comp_obj
               if self.comp_obj[ico].active]
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

        df = pd.Series(obj.extract_values()).fillna(0).reset_index()
        df.columns = list(cols) + ['value']

        return df

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
    Base class for parameters.

    Is inherited by :class:`DmndIO`.

    Only contains the parameter ``_to_df`` classmethod.

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

        return df


class TransmIO(VariabIO):
    """
    Special methods related to the translation of nodes to
    transmission plant names and
    to the simplified representation after aggregating secondary nodes.
    """

    def post_processing(self, df):
        ''' Write aggregated transmission table to pwr. '''

        dfagg = self.aggregate_nd2(df)
        dfagg = self._translate_trm(dfagg)

        self._finalize(dfagg, 'var_sy_pwr')

        return self.add_bool_out_col(df)

    def aggregate_nd2(self, dfall):
        '''
        Aggregates trm table over all secondary nodes for simplification and
        to append to the pwr table.
        '''
        # mirror table to get both directions
        dfall = pd.concat([dfall,
                           dfall.assign(nd_2_id=dfall.nd_id,
                                        nd_id=dfall.nd_2_id,
                                        value=-dfall.value)])

        dict_nhours = {nd_id:
                       self.model._tm_objs[self.model.dict_nd_tm_id[nd_id]].nhours
                       for nd_id in dfall.nd_id.unique()}

        def avg_to_nhours(x):

            if self.model.is_min_node[x.name]:
                return x.reset_index()
            else:  # reduce time resolution
                nhours = dict_nhours[x.name[0]]
                nhours_2 = dict_nhours[x.nd_2_id.iloc[0]]

                x['sy'] = np.repeat(np.arange(
                                    np.ceil(len(x) / (nhours / nhours_2))),
                                    nhours / nhours_2)

                idx = [c for c in x.columns if not c == 'value']
                x = x.pivot_table(index=idx, values='value', aggfunc=np.mean)
                return x.reset_index()

        dfall = (dfall.groupby(['nd_id', 'nd_2_id'], as_index=True)
                      .apply(avg_to_nhours)
                      .reset_index(drop=True))

        dfexp = dfall.loc[dfall.value > 0]
        dfexp = dfexp.groupby(['sy', 'nd_id', 'ca_id'])['value'].sum()
        dfexp = dfexp.reset_index()
        dfexp['bool_out'] = True

        dfimp = dfall.loc[dfall.value < 0]
        dfimp = dfimp.groupby(['sy', 'nd_id', 'ca_id'])['value'].sum()
        dfimp = dfimp.reset_index()
        dfimp['bool_out'] = False

        dfagg = pd.concat([dfexp, dfimp], axis=0)

        dict_nd_weight = {key: self.model.nd_weight[key].value
                          for key in self.model.nd_weight}
        dfagg['value'] /= dfagg.nd_id.replace(dict_nd_weight)

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
    '''
    Demand is appended to the *pwr* table after translating the nd_id to
    the corresponding "power plant" pp_id.

    '''

    def post_processing(self, df):

        dfpp = self._translate_dmnd(df.copy())
        dfpp['bool_out'] = True
        dfpp = dfpp[['sy', 'pp_id', 'ca_id', 'value', 'bool_out']]

        self._finalize(dfpp, 'var_sy_pwr')

        return df

    def _translate_dmnd(self, df):
        '''
        Translate the demand ``pf_id`` to the corresponding ``pp_id``s

        This is based on a mapping ``pf_id`` |rarr| ``nd_id`` |rarr| ``pp_id``.
        The ``pp_id`` definition for demand is retrieved from the
        ``ModelBase.df_def_plant`` table.
        '''

        dict_ndpp = self._node_to_plant('DMND')
        df['pp_id'] = df.nd_id.replace(dict_ndpp)

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

    io_class_dict = {'var': VariabIO,
                     'var_tr': TransmIO,
                     'par_dmnd': DmndIO,
                     'par': ParamIO,
                     'dual': DualIO}

    _default_init = {'sc_warmstart': False,
                     'resume_loop': False,
                     'replace_runs_if_exist': False,
                     'model': None,
                     'output_target': 'hdf5',
                     'sql_connector': None,
                     'no_output': False,
                     'dev_mode': False,
                     'coll_out': None,
                     'keep': None,
                     'drop': None,
                     'db': None}

    def __init__(self, **kwargs):
        """
        """

        self.run_id = None  # set in call to self.write_run
        self.dict_comp_obj = {}


        # define instance attributes and update with kwargs
        for key, val in self._default_init.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)

        self.dict_comp_idx = None
        self.dict_comp_table = None
        self.dict_comp_group = None

        ls = 'Output collection: {}; resume loop={}'
        logger.info(ls.format(self.cl_out, self.resume_loop))

        self.reset_tablecollection()


    def _make_table_dicts(self, keep=None, drop=None):
        '''
        Get the dictionaries describing all tables.

        Also used in the class method ``post_process_index``,
        therefore module function.
        '''

        keep = list(table_struct.DICT_COMP_IDX) if not keep else keep
        keep = set(keep) - set(drop if drop else [])

        options = list(table_struct.DICT_COMP_IDX)
        unknowns = [tb for tb in keep if not tb in options]
        if unknowns:
            raise RuntimeError(('Unknown table selection %s. Possible options '
                               'are %s')%(str(unknowns), str(options)))

        filter_dict = lambda d: {k: v for k, v in d.items() if k in keep}

        self.dict_comp_idx = filter_dict(table_struct.DICT_COMP_IDX)
        self.dict_comp_table = filter_dict(table_struct.DICT_COMP_TABLE)
        self.dict_comp_group = filter_dict(table_struct.DICT_COMP_GROUP)

    def reset_tablecollection(self):
        '''
        Reset the SQL schema or hdf file for model output writing.

        '''

        if self.output_target == 'psql':
            self._reset_schema()
        elif self.output_target == 'hdf5':
            self._reset_hdf_file()
        elif self.output_target in ['fastparquet']:
            self._reset_parquet_file()

    @skip_if_resume_loop
    def _reset_hdf_file(self):

        ModelWriter.reset_hdf_file(self.cl_out, not self.dev_mode)

    def _reset_parquet_file(self):

        ModelWriter.reset_parquet_file(self.cl_out, not self.dev_mode,
                                       self.resume_loop)

    @staticmethod
    def reset_hdf_file(fn, warn):
        '''
        Deletes existing hdf5 file and creates empty one.

        Parameters
        ----------
        fn: str
            filename
        warn: bool
            prompt user input if the file exists

        '''
#        pass

        if os.path.isfile(fn):

            try:
                max_run_id = pd.read_hdf(fn, 'def_run',
                                         columns=['run_id']).run_id.max()
            except Exception as e:
                logger.error(e)
                logger.warn('reset_hdf_file: Could not determine max_run_id '
                            '... setting to None.')
                max_run_id = None

            if warn:
                input(
'''
~~~~~~~~~~~~~~~   WARNING:  ~~~~~~~~~~~~~~~~
You are about to delete existing file {fn}.
The maximum run_id is {max_run_id}.

Hit enter to proceed.
'''.format(fn=fn, max_run_id=max_run_id)
)

            logger.info('Dropping output file {}'.format(fn))
            os.remove(fn)

    def reset_parquet_file(dirc, warn, resume_loop, ):
        '''
        Deletes existing parquet file folder and creates empty one.

        Parameters
        ----------
        dirc: str
            parquet directory name
        warn: bool
            prompt user input if the file exists

        '''

        if os.path.isdir(dirc) and not resume_loop:

            try:
                max_run_id = pd.read_parquet(os.path.join(dirc, 'def_run.parq'),
                                         columns=['run_id']).run_id.max()

            except Exception as e:
                logger.error(e)
                logger.warn('reset_parquet_file: Could not determine max_run_id '
                            '... setting to None.')
                max_run_id = None

            if warn:
                input(
'''
~~~~~~~~~~~~~~~   WARNING:  ~~~~~~~~~~~~~~~~
You are about to delete existing directory {dirc}.
The maximum run_id is {max_run_id}.

Hit enter to proceed.
'''.format(dirc=dirc, max_run_id=max_run_id)
)

            logger.info('Dropping parquet output directory {}'.format(dirc))
            shutil.rmtree(dirc)

        elif os.path.isdir(dirc) and resume_loop:

            logger.info('Deleting run_ids >= resume_loop = '
                        '{:d}'.format(resume_loop))

            del_fn = [fn for fn in glob(os.path.join(dirc, '*[0-9].parq')) if
                      int(fn.split('_')[-1].replace('.parq', ''))
                      >= resume_loop]

            if del_fn:
                for fn in del_fn:
                    logger.info('... deleting file {}'.format(fn))
                    os.remove(fn)

                fn_run = os.path.join(dirc, 'def_run.parq')
                df_def_run = pd.read_parquet(fn_run)
                df_def_run = df_def_run.query('run_id < %d'%resume_loop)
                df_def_run = df_def_run.reset_index(drop=True)
                pq.write(fn_run, df_def_run, append=False, compression='GZIP')

            else:
                logger.info('... nothing to delete.')

        if not os.path.isdir(dirc):

            os.mkdir(dirc)

    @skip_if_resume_loop
    def _reset_schema(self):

        aql.reset_schema(self.cl_out, self.sql_connector.db,
                         not self.dev_mode)

    def init_output_schema(self):

        aql.exec_sql('CREATE SCHEMA IF NOT EXISTS ' + self.cl_out,
                     db=self.db, )

    @skip_if_no_output
    def init_compio_objs(self):
        '''
        Initialize all output table IO objects.
        '''

        self._make_table_dicts(keep=self.keep, drop=self.drop)

        if __name__ == '__main__':
            comp, idx = 'pwr_st_ch', self.dict_comp_idx['pwr_st_ch']
        for comp, idx in self.dict_comp_idx.items():
            if not hasattr(self.model, comp):
                logger.warning(('Component {} does not exist... '
                               'skipping init CompIO.').format(comp))
            else:
                logger.debug('Adding component %s to dict_comp_obj'%comp)
                comp_obj = getattr(self.model, comp)

                grp = self.dict_comp_group[comp]
                if self.dict_comp_table[comp] in self.io_class_dict:
                    io_class = self.io_class_dict[self.dict_comp_table[comp]]
                elif grp in self.io_class_dict:
                    io_class = self.io_class_dict[self.dict_comp_group[comp]]
                else:
                    io_class = self.io_class_dict[self.dict_comp_group[comp].split('_')[0]]

                io_class_kwars = dict(tb=self.dict_comp_table[comp],
                                      cl_out=self.cl_out,
                                      comp_obj=comp_obj,
                                      idx=idx,
                                      connect=self.sql_connector,
                                      output_target=self.output_target,
                                      model=self.model)

                self.dict_comp_obj[comp] = io_class(**io_class_kwars)

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

        if self.output_target == 'psql':

            coldict = aql.get_coldict(self.cl_out, self.sql_connector.db)

            for comp, io_obj in self.dict_comp_obj.items():

                io_obj.coldict = coldict
                io_obj.init_output_table()

        elif self.output_target in ['hdf5', 'fastparquet']:

            pass

    def delete_run_id(self, run_id=False, operator='>='):
        '''
        In output tables delete all rows with run_id >=/== the selected value.

        Used in :
            1. in ModelLoop.perform_model_run if replace_runs_if_exist == True
                with operator '=' to remove current run_id
                from all tables prior to writing

        TODO: The SQL part would be better fit with the aux_sql_func module.
        '''

        if run_id:



            # Get overview of all tables
            list_all_tb_0 = [list(itb_list + '_' + itb[0] for itb
                                  in getattr(table_struct, itb_list)
                                  if not len(itb) == 3)
                             for itb_list in table_struct.list_collect]
            self.list_all_tb = list(itertools.chain(*list_all_tb_0))
            self.list_all_tb += ['def_run']

            for itb in self.list_all_tb:

                if self.output_target == 'fastparquet':

                    self._delete_run_id_parquet(tb=itb, run_id=run_id)

                elif self.output_target == 'psql':

                    logger.info('Deleting from ' + self.cl_out + '.' + itb
                                + ' where run_id {} {}'.format(operator,
                                                               str(run_id)))
                    exec_strg = '''
                                DELETE FROM {cl_out}.{tb}
                                WHERE run_id {op} {run_id};
                                '''.format(cl_out=self.cl_out, tb=itb,
                                           run_id=run_id, op=operator)
                    try:
                        aql.exec_sql(exec_strg, db=self.db)
                    except pg.ProgrammingError as e:
                        logger.error(e)
                        raise(e)

    def _delete_run_id_parquet(self, tb, run_id):

        pat = os.path.join(self.cl_out, ('{}_%s.*'%FORMAT_RUN_ID).format(tb, run_id))
        fn_del = glob(pat)

        try:
            assert fn_del, 'Pattern not found: %s.'%pat
            assert not len(fn_del) > 1, \
                        'Found more than one table to delete: %s.'%fn_del

            os.remove(fn_del[0])

            logger.info('Successfully deleted table '
                        '{} of model run {:d}'.format(tb, run_id))
        except Exception as e:
            logger.error(e)


#    @classmethod
#    def post_process_index(cls, sc, db, drop=False):
#
#        coldict = aql.get_coldict(sc, db)
#
#        dict_idx, dict_table, _ = ModelWriter.get_table_dicts()
#
#        list_tables = aql.get_sql_tables(sc, db)
#
#        for comp, index in dict_idx.items():
#
#            if not dict_table[comp] in list_tables:
#                logger.warning('Table ' + comp + ' does not exist... skipping '
#                      'index generation.')
#            else:
#
#                tb_name = dict_table[comp]
#
#                logger.info('tb_name:', tb_name)
#
#                pk_list = index + ('run_id',)
#
#                fk_dict = {}
#                for c in pk_list:
#                    if len(coldict[c]) > 1:
#                        fk_dict[c] = coldict[c][1]
#
#
#                pk_kws = {'pk_list': ', '.join(pk_list),
#                          'tb': tb_name, 'cl_out': sc}
#                exec_str = ('''
#                            ALTER TABLE {cl_out}.{tb}
#                            DROP CONSTRAINT IF EXISTS {tb}_pkey;
#                            ''').format(**pk_kws)
#                if not drop:
#                    exec_str += ('''
#                                 ALTER TABLE {cl_out}.{tb}
#                                 ADD CONSTRAINT {tb}_pkey
#                                 PRIMARY KEY ({pk_list})
#                                 ''').format(**pk_kws)
#                logger.debug(exec_str)
#                aql.exec_sql(exec_str, db=db)
#
#                for fk_keys, fk_vals in fk_dict.items():
#                    fk_kws = {'cl_out': sc, 'tb': tb_name,
#                              'fk': fk_keys, 'ref': fk_vals}
#
#                    exec_str = ('''
#                                ALTER TABLE {cl_out}.{tb}
#                                DROP CONSTRAINT IF EXISTS fk_{tb}_{fk};
#                                ''').format(**fk_kws)
#
#                    if not drop:
#                        exec_str += ('''
#                                     ALTER TABLE {cl_out}.{tb}
#                                     ADD CONSTRAINT fk_{tb}_{fk}
#                                     FOREIGN KEY ({fk})
#                                     REFERENCES {ref}
#                                     ''').format(**fk_kws)
#                    logger.debug(exec_str)
#                    aql.exec_sql(exec_str, db=db)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TableReader():
    '''
    Reads tables from input data sources and makes them attributes of the
    model attribute.
    '''

    def __init__(self, sql_connector, sc_inp, data_path, model):

        self.sqlc = sql_connector
        self.sc_inp = sc_inp
        self.data_path = (data_path if
                          isinstance(data_path, (tuple, list))
                          else [data_path])
        self.model = model

        if not self.sc_inp and not self.data_path:
            logger.warning('Falling back to grimsel default csv tables.')
            self.data_path = os.path.abspath(os.path.join(grimsel.__path__[0],
                                                          '..', 'input_data'))


        self.table_set, self.dict_tb_path = self._get_table_dict()

    def _get_table_dict(self):
        '''
        Obtain list of tables in the relevant data source.

        TODO: Update PSQL

        '''

        if self.sc_inp:
            return aql.get_sql_tables(self.sc_inp, self.sqlc.db)

        elif self.data_path:

            # path -> tables list
            dict_pt_tb = {path: [fn.replace('.csv', '')
                          for fn in next(os.walk(path))[-1]]
                          for path in self.data_path}

            table_set = set(itertools.chain.from_iterable(
                                                        dict_pt_tb.values()))

            # table -> under which paths
            dict_tb_path = {tb: [pt for pt, tb_list in dict_pt_tb.items() if
                            tb in tb_list] for tb in table_set}

            return table_set, dict_tb_path

    def _expand_table_families(self, dct):
        '''
        Searches for tables with identical name + suffix.
        Updates the dct.
        '''

        dct_add = {}
        for table, filt in dct.items():

            tbs_other = [tb for tb in self.table_set
                         if table in tb and not tb == table]

            if tbs_other:
                dct_add.update({tb: filt for tb in tbs_other})

        dct.update(dct_add)

    def df_from_dict(self, dct):
        '''
        Reads filtered input tables and assigns them to instance
        attributes.
        '''

        self._expand_table_families(dct)

        for table, filt in dct.items():

            list_df, tb_exists, source_str = self.get_input_table(table, filt)

            df = pd.concat(list_df, axis=0, sort=False) if tb_exists else None

            setattr(self.model, 'df_' + table, df)

            if not tb_exists:
                warn_str = ('Input table {tb} does not exist. Setting model '
                            'attribute df_{tb} to None.')
                logger.warning(warn_str.format(tb=table))
            else:
                filt = ('filtered by ' if len(filt) > 0 else '') +\
                   ', '.join([str(vvv[0]) + ' in ' + str(vvv[1]) for vvv in filt
                              if not len(vvv[1]) == 0])
                logger.info(('Reading input table {tb} {flt} from '
                       '{source_str}').format(tb=table, flt=filt,
                                              source_str=source_str))

    def get_input_table(self, table, filt):
        '''
        Returns list of tables.
        '''

        if self.sc_inp:

            tb_exists = table in aql.get_sql_tables(self.sc_inp, self.sqlc.db)

            if tb_exists:
                list_df = [aql.read_sql(self.sqlc.db, self.sc_inp,
                                        table, filt)]
            source = '%s %s.%s'%(self.sqlc.db, self.sc_inp, table)

        else:
            list_df = []

            tb_exists = table in self.dict_tb_path

            paths = self.dict_tb_path[table] if tb_exists else []

            source = []

            for path in paths:

                fn = os.path.join(path, '{}.csv'.format(table))
                source.append(fn)

                df = pd.read_csv(fn)

                logger.debug('Done reading, filtering according to {}'.format(filt))

                for col, vals in filt:
                    if isinstance(col, str):  # single column filtering
                        mask = df[col].isin(vals)
                    elif isinstance(col, (list, tuple)):  # multiple columns
                        mask = df[list(col)].apply(tuple, axis=1).isin(vals)
                    df = df.loc[mask]

                list_df.append(df)

        return ((list_df if tb_exists else None), tb_exists,
                (' from {}'.format(' and '.join(source)) if tb_exists else ''))

class DataReader(_HDFWriter, _ParqWriter):
    '''

    '''

    runtime_tables = [('tm_soy', ['sy', 'tm_id']),
                      ('hoy_soy', ['hy', 'tm_id']),
                      ('tm_soy_full', ['sy', 'tm_id']),
                      ('sy_min_all', ['sy_min', 'tm_id']),
                      ]

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
                    'cl_out': None,
                    'db': None,
                    }

        defaults.update(kwargs)
        for key, val in defaults.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)

        self._coldict = aql.get_coldict()

    def read_model_data(self):
        '''
        Read all input data and generate :class:`ModelBase` instance
        attributes.

        '''

        tbrd = TableReader(self.sql_connector, self.sc_inp,
                           self.data_path, self.model)

        # unfiltered input
        dict_tb_2 = {'def_month': [], 'def_week': [],
                     'parameter_month': [], 'tm_soy': []}
        tbrd.df_from_dict(dict_tb_2)

        # read input data filtered by node and energy carrier
        _flt_nd = ([('nd', self.model.slct_node)]
                   if self.model.slct_node else [])
        _flt_ca = ([('ca', self.model.slct_encar)]
                   if self.model.slct_encar else [])
        _flt_pt = ([('pt', self.model.slct_pp_type)]
                   if self.model.slct_pp_type else [])
        dict_tb_3 = {'def_node': _flt_nd,
                     'def_pp_type': _flt_pt,
                     'def_encar': _flt_ca}
        tbrd.df_from_dict(dict_tb_3)

        # translate slct_node_connect to nd_ids
        if self.model.slct_node_connect:
            dict_nd = self.model.df_def_node.set_index('nd').nd_id.to_dict()
            slct_node_connect_id = [(dict_nd[nd1], dict_nd[nd2])
                                    for nd1, nd2 in self.model.slct_node_connect
                                    if nd1 in dict_nd and nd2 in dict_nd]
            _flt_ndcnn = [(('nd_id', 'nd_2_id'), slct_node_connect_id)]
        else:
            _flt_ndcnn = []


        # update filters in case the keyword argument slct_node_id holds more
        # nodes than present in the table
        self.model.slct_node_id = self.model.df_def_node.nd_id.tolist()
        self.model.slct_encar_id = self.model.df_def_encar.ca_id.tolist()
        self.model.slct_pp_type_id = self.model.df_def_pp_type.pt_id.tolist()
        _flt_nd = [('nd_id', self.model.slct_node_id)]
        _flt_ca = [('ca_id', self.model.slct_encar_id)]
        _flt_nd_2 = [('nd_2_id', self.model.df_def_node.nd_id.tolist())]
        _flt_pt = [('pt_id', self.model.df_def_pp_type.pt_id.tolist())]

        # read input data filtered by node, energy carrier, and fuel
        dict_tb_0 = {'def_plant': _flt_nd + _flt_pt,
                     'profchp': _flt_nd,
                     'node_encar': _flt_nd + _flt_ca,
                     'node_connect': _flt_nd + _flt_ca + _flt_nd_2 + _flt_ndcnn}
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

        _flt_pf_supply = [('supply_pf_id',
                          list(self.model.dict_supply_pf.values()))]
        _flt_pf_dmnd = [('dmnd_pf_id',
                        list(self.model.dict_dmnd_pf.values()))]
        _flt_pf_price = [('price_pf_id',
                         list(self.model.dict_pricebuy_pf.values())
                         + list(self.model.dict_pricesll_pf.values()))]
        dict_pf_0 = {'profsupply': _flt_pf_supply,
                     'profdmnd': _flt_pf_dmnd,
                     'profprice': _flt_pf_price,
                     }
        tbrd.df_from_dict(dict_pf_0)

        _flt_pf = [('pf_id', (_flt_pf_price[-1][-1] + _flt_pf_dmnd[-1][-1]
                              + _flt_pf_supply[-1][-1]))]
        dict_pf_1 = {'def_profile': _flt_pf}
        tbrd.df_from_dict(dict_pf_1)

        # filter plants requiring input from non-existing ca
        # e.g. if a fuel-cell is in the input table but no hydrogen is
        # included in the model, the plant's H2 demand wouldn't be accounted
        # for;
        if 'fl_id' in self.model.df_def_encar.columns:

            fl_id_ca = self.model.df_def_encar.fl_id.tolist()
            mask_del = (self.model.df_def_fuel.is_ca.isin([1])
                        & - self.model.df_def_fuel.fl_id.isin(fl_id_ca))

            self.model.df_def_fuel = self.model.df_def_fuel.loc[-mask_del]

        # filter table by special index name/id columns
        self.model.df_parameter_month = \
            self.filter_by_name_id_cols('df_parameter_month',
                                        _flt_fl + _flt_nd + _flt_pp + _flt_ca)

        self._split_profprice()

        # autocomplete input tables
        self.data_autocompletion()

        if isinstance(self.model.df_node_connect, pd.DataFrame):
            self._fix_df_node_connect()

        self.input_table_list = (list(dict_tb_1) + list(dict_tb_2)
                                 + list(dict_tb_0) + list(dict_tb_3)
                                 + list(dict_pf_1))

#        self.model._update_slct_lists()


    def _split_profprice(self):
        '''
        Make two separate DataFrames for profprice buying and selling.

        Having both in the same table gets too complicated down the road.

        '''

        bool_exists = (hasattr(self.model, 'df_profprice')
                       and self.model.df_profprice is not None)

        for bs in ['buy', 'sll']:

            tb_name = 'df_profprice%s' % bs

            if bool_exists:
                mask = self.model.df_def_profile.pf.str.contains('price' + bs)
                list_pd_id = self.model.df_def_profile.loc[mask].pf_id.tolist()
                mask_prf = self.model.df_profprice.price_pf_id.isin(list_pd_id)
                df_split = self.model.df_profprice.loc[mask_prf]
                setattr(self.model, tb_name, df_split)
            else:
                setattr(self.model, tb_name, None)

    def data_autocompletion(self):

        if self.autocompletion:
            logger.info('#' * 60)

            ac.AutoCompletePpType(self.model, self.autocomplete_curtailment)
            ac.AutoCompleteFuelTrns(self.model)
            ac.AutoCompleteFuelDmnd(self.model, self.autocomplete_curtailment)
            ac.AutoCompletePlantTrns(self.model)
            ac.AutoCompletePlantDmnd(self.model, self.autocomplete_curtailment)
            if 'fl_id' in self.model.df_def_encar:
                ac.AutoCompletePlantCons(self.model)
            ac.AutoCompletePpCaFlex(self.model, self.autocomplete_curtailment)
            logger.info('#' * 60)


    def filter_by_name_id_cols(self, name_df, filt):
        """
        Filter a pandas DataFrame with index names in columns.

        This operates on pandas DataFrames where the indices are not provided
        as column names but as row entries in special columns.
        E.g., instead of

        ===== ===== =====
        nd_id fl_id value
        ===== ===== =====
        1     2     1.2
        ===== ===== =====

        we have

        ========== ========== ======== ======== =======
        set_1_name set_2_name set_1_id set_2_id value
        ========== ========== ======== ======== =======
        nd_id      fl_id      1        2        1.2
        ========== ========== ======== ======== =======

        This allows to combine structurally different tables.

        Filtering is implemented as an iteration over the set_n_name/set_n_id
        column pairs, each of which is filtered with respect to all elements
        in the filt parameter.

        Parameters
        ==========

        df : pandas DataFrame
            as described above
        filt : list
            filtering list of the same format as the
            :func:`grimsel.auxiliary.sqlutils.aux_sql_func.read_sql`
            parameter

        Returns
        =======
            filtered DataFrame

        """

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
                    mask |= (  # 1. select value for current set_n_name column
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

            logger.info('Ex-post filtering of DataFrame {}:'.format(name_df))
            for kk, vv in names_dict.items():
                logger.info('\tSet {} is in ({})'.format(kk, ', '.join(map(str, vv))))

        return df

    def _fix_df_node_connect(self):
        '''
        Makes sure the table df_node_connect corresponds to the new style.

        New style: The transmission capacities are expressed as

        * ``cap_trme_leg`` for exports and
        * ``cap_trmi_leg`` for imports

        for single directions, i.e. non-redundant. The input table has columns
        (nd_id, nd_2_id, ca_id, mt_id, cap_trme_leg, cap_trmi_leg).

        Old style: Single transmission capacity for both directions; columns:
        (nd_id, nd_2_id, ca_id, mt_id, eff, cap_trm_leg)
        '''

        if 'cap_trm_leg' in self.model.df_node_connect.columns:

            df = self.model.df_node_connect

            df['dir'] = df.nd_id < df.nd_2_id

            df_e = df.loc[df.dir].assign(nd_id=df.nd_2_id,
                                         nd_2_id=df.nd_id,
                                         cap_trmi_leg=df.cap_trm_leg)
            df = df.loc[-df.dir].assign(cap_trme_leg=df.cap_trm_leg)
            dfn = pd.concat([df_e, df], sort=False)
            dfn = dfn.drop('cap_trm_leg', axis=1).fillna(0)
            idx = ['nd_id', 'nd_2_id', 'ca_id', 'mt_id']
            logger.info('Aggregation count in fix_df_node_connect:\n',
                        dfn.pivot_table(index=idx, aggfunc=[min, max, len],
                                        values=['cap_trme_leg',
                                                'cap_trmi_leg']))
            dfn = dfn.pivot_table(index=idx, aggfunc=sum,
                                  values=['cap_trme_leg', 'cap_trmi_leg'])

            self.model.df_node_connect = dfn.reset_index()

    @skip_if_resume_loop
    @skip_if_no_output
    def _write_input_tables_to_output_schema(self, tb_list):
        '''
        Gathers relevant input tables and writes them to the output collection.

        Note: As of now profile input tables are excluded. All relevant data
        is included in the parameter output anyway.

        '''

        for itb in set(tb_list) - set(list(zip(*self.runtime_tables))[0]):

            df = getattr(self.model, 'df_' + itb)

            if (df is not None and ('def_' in itb or 'prof' not in itb)):

                log_str = 'Writing input table {} to {} output: {}.'
                logger.info(log_str.format(itb, self.output_target,
                                           self.cl_out))

                if self.output_target == 'psql':

                    logger.info('Writing table {} to output.'.format(itb))
                    engine = self.sql_connector.get_sqlalchemy_engine()
                    db = self.sql_connector.db
                    aql.write_sql(df, db, self.cl_out, itb,
                                  if_exists='replace', engine=engine)

                elif self.output_target == 'hdf5':

                    self.write_hdf(itb, df, 'put')

                elif self.output_target in ['fastparquet']:

                    fn = os.path.join(self.cl_out, itb + '.parq')
                    self.write_parquet(fn, df, self.output_target)
                else:
                    raise RuntimeError('_write_input_tables_to_output_schema: '
                                       'no output_target applicable')

    @skip_if_resume_loop
    @skip_if_no_output
    def write_runtime_tables(self):
        '''
        Some input tables depend on model parameters (time resolution).
        Write these to output database schema.
        Also, table def_node altered due to addition of column dmnd_max.

        '''

        self._write_input_tables_to_output_schema(self.input_table_list)

        skip_fks = [('tm_soy', 'sy'),  # defines sy
                    ('hoy_soy', 'hy')]  # defines hy

        tb_name, pk = ('hoy_soy', ['hy', 'tm_id'])
        for tb_name, pk in self.runtime_tables:

            if getattr(self.model, 'df_' + tb_name, None) is not None:
                df = getattr(self.model, 'df_' + tb_name)

                logger.info('Writing runtime table ' + tb_name)

                cols = []
                for c in df.columns:
                    col_add = [c]

                    if c not in self._coldict:  # same as "value"
                        self._coldict[c] = self._coldict['value']

                    col_add += (list(self._coldict[c])
                                if (tb_name, c) not in skip_fks
                                else list(self._coldict[c][:1]))
                    cols.append(tuple(col_add))

                if self.output_target == 'psql':

                    engine = self.sql_connector.get_sqlalchemy_engine()
                    con_cur = self.sql_connector.get_pg_con_cur()

                    aql.init_table(tb_name=tb_name, cols=cols,
                                   schema=self.cl_out,
                                   ref_schema=self.cl_out, pk=pk, unique=[],
                                   db=self.sql_connector.db, con_cur=con_cur)

                    aql.write_sql(df, sc=self.cl_out, tb=tb_name,
                                  if_exists='append', engine=engine,
                                  con_cur=con_cur)

                elif self.output_target == 'hdf5':

                    self.write_hdf(tb_name, df, 'put')

                elif self.output_target in ['fastparquet']:

                    fn = os.path.join(self.cl_out, tb_name + '.parq')

                    self.write_parquet(fn, df, engine=self.output_target)

                else:
                    raise RuntimeError('write_runtime_tables: no '
                                       'output_target applicable')

    @staticmethod
    def _get_max_run_id(cl_out):

        run_fn = os.path.join(cl_out, 'def_run.parq')

        if os.path.isfile(run_fn):
            logger.debug('_get_max_run_id: Reading file %s.'%run_fn)
            max_run = pd.read_parquet(run_fn, columns=['run_id']).run_id.max()
        else:
            logger.warning('_get_max_run_id: %s not found.'%run_fn)
            max_run = False

        return max_run

class IO:
    '''
    Primary IO class exposing the :module:`io` module.

    :ivar datrd: :class:`DataReader` instance
    :ivar modwr: :class:`ModelWriter` instance
    '''

    def __init__(self, **kwargs):

        self._close_all_hdf_connections()

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
                    'cl_out': None,
                    'db': 'postgres',
                    'output_target': 'psql'
                    }

        defaults.update(kwargs)

        self.sql_connector = defaults['sql_connector']
        self.replace_runs_if_exist = defaults['replace_runs_if_exist']
        self.db = self.sql_connector.db if self.sql_connector else None
        self.cl_out = defaults['cl_out']

        if defaults['resume_loop'] == 'auto':
            defaults['resume_loop'] = \
                    self._get_auto_resume_loop(defaults['output_target'])

        self.resume_loop = defaults['resume_loop']

        self.datrd = DataReader(**defaults)
        self.modwr = ModelWriter(**defaults)

    def _get_auto_resume_loop(self, output_target):

        if not output_target == 'fastparquet':
            raise RuntimeError ('resume_loop="auto" not implemented for '
                                '%s output.')%output_target

        resloop = DataReader._get_max_run_id(self.cl_out)
        resloop = (resloop + 1) if not isinstance(resloop, bool) else resloop

        logger.info('Setting "auto" resume_loop to %s'%resloop)

        return resloop

    @classmethod
    def variab_to_df(cls, py_obj, sets=None):
        ''' Wrapper for backward compatibility. '''

        if not sets:
            sets = table_struct.DICT_COMP_IDX[py_obj.name]
            sets = [st for st in sets if not st == 'bool_out']

        return VariabIO._to_df(py_obj, sets)

    @classmethod
    def param_to_df(cls, py_obj, sets=None):
        ''' Wrapper for backward compatibility. '''

        if not sets:
            sets = table_struct.DICT_COMP_IDX[py_obj.name]

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

    def _init_loop_table(self, cols_id, cols_step, cols_val):

        tb_name = 'def_run'
        cols = ([('tdiff_solve', 'DOUBLE PRECISION'),
                 ('tdiff_write', 'DOUBLE PRECISION'),
                 ('run_id', 'SMALLINT'),
                 ]
                + [(s, 'SMALLINT') for s in cols_id]
                + [(s, 'DOUBLE PRECISION') for s in cols_step]
                + [(s, 'VARCHAR(30)') for s in cols_val]
                + [('info', 'VARCHAR'), ('objective', 'DOUBLE PRECISION')])

        if self.modwr.output_target == 'psql':

            aql.init_table(tb_name, cols, self.cl_out,
                           pk=cols_id, unique=['run_id'], db=self.db)

        elif self.modwr.output_target == 'hdf5':

            df = pd.DataFrame(columns=list(zip(*cols))[0])

            if self.modwr.output_target == 'hdf5':
                df.to_hdf(self.cl_out, tb_name, format='table')

        elif self.modwr.output_target == 'fastparquet':
            pass  # parquet table is not initialized

        else:
            raise RuntimeError('_init_loop_table: no '
                               'output_target applicable')

    @staticmethod
    def _close_all_hdf_connections():

        tables.file._open_files.close_all()


# %%
if __name__ == '__main__':

    pass
