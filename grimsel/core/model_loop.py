'''
Module doc
'''
import os
from multiprocessing import Lock, Pool, current_process
import numpy as np
import pandas as pd
import itertools
from glob import glob
import time
from importlib import reload
import fastparquet as pq

import grimsel.core.model_base as model_base
import grimsel.core.io as io
import grimsel.core.model_loop_modifier as model_loop_modifier
import grimsel.auxiliary.sqlutils.aux_sql_func as aql
import grimsel.auxiliary.maps as maps
from grimsel import _get_logger

logger = _get_logger(__name__)
logger_parallel = _get_logger(__name__ + '_parallel')


reload(model_base)
reload(io)
reload(model_loop_modifier)

class ModelLoop():
    '''
    Defines the model loop framework. Main attribute is the dataframe
    df_def_run.

     -> variables:
         - dicts
         - def_run
     -> functions:
         - generate run name
         - _print_run_title
         - init loop table
         - reset_run_id
         - generate_loop_row <- from run_id and current values of dictionaries
                                (option for zero row)
    '''

    @property
    def df_def_run(self):
        return self._df_def_run

    @df_def_run.setter
    def df_def_run(self, df_def_run):
        self._df_def_run = df_def_run
        self._df_def_run = self.restore_run_id(self._df_def_run)

    def __init__(self, **kwargs):
        '''
        Keyword arguments:
        nsteps -- list of model loop dimensions and steps; format:
                  (name::str, number_of_steps::int, type_of_steps::function)
        '''

        defaults = {
                    'nsteps': [],
                    'mkwargs': {},
                    'iokwargs': {},
                    'full_setup': True
                    }

        for key, val in defaults.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)

        self.run_id = None  # set later
        self.__runlevel_state = -1

        self.m = model_base.ModelBase(**self.mkwargs)

        self.iokwargs.update({'model': self.m})
        self.io = io.IO(**self.iokwargs)

        self.init_run_table()
        self.select_run(0)


    @property
    def _runlevel_state(self):
        return self.__runlevel_state

    @_runlevel_state.setter
    def _runlevel_state(self, _runlevel_state):

        assert _runlevel_state == self.__runlevel_state + 1, (
                'Must increment runlevel state by one. Caught attempt to jump '
                f'from {self.__runlevel_state} '
                f'({self._dict_runlevels[self.__runlevel_state]})'
                f' to {_runlevel_state} '
                f'({self._dict_runlevels[_runlevel_state]})')

        self.__runlevel_state = _runlevel_state


    _dict_runlevels = {
            0: 'io.read_model_data',
            1: 'm.init_maps',
            2: 'm.map_to_time_res',
            3: 'io.write_runtime_tables',
            4: 'm.get_setlst',
            5: 'm.define_sets',
            6: 'm.add_parameters',
            7: 'm.define_variables',
            8: 'm.add_all_constraints',
            9: 'm.init_solver',
            10: 'io.init_output_tables'}


    def build_model(self, to_runlevel='full'):
        '''
        Run sequence of methods to read the data and initialize the Pyomo
        components.

        Parameters
        ----------
        to_runlevel : str, one of `['full', 'input_data']`
            `'input_data'`: stop after reading all input data; allows to
            make modifications to the input dataframes
            `'full'`: complete construction of the model; allows to
            make modications to the Pyomo components
        '''

        dict_to_runlevel = {'input_data': 2, 'full': 10}

        assert to_runlevel in dict_to_runlevel, (
                f'Unknown to_runlevel level \'{to_runlevel}\'. '
                f'Expecting one of {list(dict_to_runlevel)}.')

        # filter runlevel dictionary by selected runlevels
        _dict_runlevel_slct = {lvl: meth for lvl, meth
                               in self._dict_runlevels.items()
                               if lvl > self._runlevel_state
                               and lvl <= dict_to_runlevel[to_runlevel]}

        for runlevel, method in _dict_runlevel_slct.items():

            attr, method = method.split('.')

            func = getattr(getattr(self, attr), method)

            logger.info('%' * 60)
            logger.info(f'ModelLoop.build_model: Runlevel {runlevel}: '
                        f'Calling method {method}')
            logger.info('%' * 60)

            func()
            self._runlevel_state = runlevel


    def init_run_table(self):
        '''
        Initializes the ``df_def_run`` table by expanding the ``nsteps`` list.

        Expands the ``nsteps`` parameter to the corresponding DataFrame.
        The resulting attribute ``df_def_run`` contains all relevant
        combinations of the model change indices as defined in ``nsteps``.

        Also initializes the output ``def_run`` table, if required.
        '''

        for istep in self.nsteps:
            assert (isinstance(istep[0], str) and isinstance(istep[1], int)
                    and (len(istep) == 2
                         or istep[2] in (np.linspace, np.arange))), (
                         'Items of nlist must be tuples like '
                         ' (str, int, np.arange or np.linspace)'
                         f' or (str, int); found {istep}')

        _nsteps = [list(ist) + ([0, 1] if ist[-1] == np.linspace else [])
                  for ist in self.nsteps]

        def getfunc(istep): return istep[2] if len(istep) > 2 else np.arange

        list_steps = [list(getfunc(istep)(*istep[3:], istep[1]))
                      for istep in _nsteps]

        list_steps = [list(map(float, lst)) for lst in list_steps]
        full_steps = np.array([tuple(reversed(lst)) for lst in
                      list(itertools.product(*list(reversed(list_steps))))])
        list_index = [list(range(len(i))) for i in list_steps]
        full_index = np.array([tuple(reversed(lst)) for lst in
                      list(itertools.product(*list(reversed(list_index))))])
        full_all = np.concatenate([full_index, full_steps,
                                   float('nan')*full_index], axis=1)
        self.cols_step = [ist[0] for ist in _nsteps]
        self.cols_id = [c + '_id' for c in self.cols_step]
        self.cols_val = [c + '_vl' for c in self.cols_step]
        cols_all = self.cols_id + self.cols_step + self.cols_val

        self.df_def_run = pd.DataFrame(full_all, columns=cols_all)

        if not self.io.resume_loop:
            self.init_loop_table()

    def select_run(self, slct_run_id):
        '''
        Get all relevant indices and parameters for a certain slct_run_id
        '''

        self.run_id = slct_run_id

        lpsrs = self.df_def_run.loc[self.df_def_run.run_id
                                     == slct_run_id].iloc[0]

        self.dct_vl = lpsrs.loc[lpsrs.index.str.contains('_vl')].to_dict()
        self.dct_id = lpsrs.loc[lpsrs.index.str.contains('_id')
                               & ~lpsrs.index.str.contains('run')].to_dict()
        self.dct_step = lpsrs.loc[~lpsrs.index.str.contains('_vl')
                                 & ~lpsrs.index.str.contains('_id')].to_dict()
        self.dct_id = {k: int(v) for (k, v) in self.dct_id.items()}

        self.loop_series = lpsrs


    def init_loop_table(self):

        self.io._init_loop_table(self.cols_id, self.cols_step, self.cols_val)


    def _get_row_df_run(self, tdiff_solve=0, tdiff_write=0, info=''):
        '''
        Generate new row for the def_run table.

        This contains the parameter variation indices as well as information
        on the run (time, objective function, solver status).
        '''

        dtypes = {int: ['run_id'] + list(self.dct_id),
                  float: (['tdiff_solve', 'tdiff_write', 'objective']
                          + list(self.dct_step.keys())),
                  str: ['info'] + list(self.dct_vl)}
        dtypes = {col: dtp  for dtp, cols in dtypes.items() for col in cols}

        vals = [[tdiff_solve, tdiff_write] + [self.run_id] + [info]
                + list(self.dct_id.values())
                + list(self.dct_step.values())
                + list(self.dct_vl.values())]
        cols = (['tdiff_solve', 'tdiff_write', 'run_id', 'info']
                + list(self.dct_id)
                + list(self.dct_step)
                + list(self.dct_vl))

        df_add = pd.DataFrame(vals, columns=cols)

        df_add['objective'] = (self.m.objective_value
                               if hasattr(self.m, 'objective_value')
                               else 0)

        return df_add.astype(dtypes)

    def get_def_run_name(self):

        # if multiprocessing, locked writing to common parquet file has
        # too much overhead for small models. Therefore writing to files
        # by worker + later merge
        if current_process().name == 'MainProcess':
            suffix = ''
            fn = os.path.join(self.io.cl_out, 'def_run%s.parq'%suffix)
        elif current_process().name.startswith('ForkPoolWorker'):
            suffix = '_' + current_process().name
            fn = os.path.join(self.io.cl_out, 'def_run%s.csv'%suffix)
        else:
            raise ValueError('Unexpected current_process name'
                             ' %s'%current_process().name)

        csv_def_run = suffix !=''

        return fn, csv_def_run


    def append_row(self, **kwargs):
        '''
        Generate single-line pandas.DataFrame to be appended to the
        output def_run table. Options:
        - zero_row == True: Row filled with zeros for calibration run
        - zero_row == False: Loop params copied to row
        '''

        df_add = self._get_row_df_run(**kwargs)

        # can't use io method here if we want this to happen when no_output
        if self.io.modwr.output_target == 'psql':
            aql.write_sql(df_add, self.io.sql_connector.db,
                          self.io.cl_out, 'def_run', 'append')
        elif self.io.modwr.output_target == 'hdf5':
            with pd.HDFStore(self.io.cl_out, mode='a') as store:
                store.append('def_run', df_add, data_columns=True,
                             min_itemsize=150 # set string length!
                             )
        elif self.io.modwr.output_target == 'fastparquet':

            fn, csv_def_run = self.get_def_run_name()

            if not csv_def_run:
                pq.write(fn, df_add, append=os.path.isfile(fn))

            else:
                # row-wise appending to parquet is slow for larger amounts of model runs
                # therefore using csv. Merged to parquet in self._merge_df_run_files
                if os.path.isfile(fn):
                    df_add.to_csv(fn, mode='a', header=False, index=False)
                else:
                    df_add.to_csv(fn, header='column_names', index=False)

        else:
            raise ValueError('Unknown output_target '
                             '%s'%self.io.modwr.output_target)


    def _merge_df_run_files(self):
        '''
        Merge all files with name out_dir/def_run_ForkPoolWorker-%d into single
        def_run.
        '''

        list_fn = glob(os.path.join(self.io.cl_out,
                                    'def_run_ForkPoolWorker-[0-9]*.csv'))

        df_def_run = pd.concat(pd.read_csv(fn) for fn in list_fn)
        df_def_run = df_def_run.sort_values('run_id').reset_index(drop=True)

        fn = os.path.join(self.io.cl_out, 'def_run.parq')
        pq.write(fn, df_def_run, append=False)


    def _print_run_title(self, warmstartfile, solutionfile):

        sep = '*' * 60
        run_id_str = 'run_id = %d of %d'%(self.run_id,
                                          self.df_def_run['run_id'].max())
        sw_strs = [str(c[0]) + ' = ' + str(c[1])
                  for c in self.dct_vl.items()]

        logger.info(sep)
        logger.info(run_id_str)
        for strg in sw_strs:
            logger.info(strg)
        logger.info(sep)

        logger.info(run_id_str + ' on ' + current_process().name)

    @staticmethod
    def restore_run_id(df):
        '''
        Reset run_id index after external manipulation of the df_def_run
        '''

        cols_not_run_id = [c for c in df.columns
                           if not c == 'run_id']
        df = df[cols_not_run_id]
        df = df.reset_index(drop=True)
        df = df.reset_index()
        df = df.rename(columns={'index': 'run_id'})

        return df


    def get_list_run_id(self):

        return list(range(self.io.resume_loop,
                          len(self.df_def_run.run_id.tolist())))


    def perform_model_run(self, warmstart=False):
        """
        TODO: This is a mess.

        Calls model_base run methods, io writing methods, and appends to
        def_run. Also takes care of time measurement for reporting in
        the corresponding def_run columns.

        """

        t = time.time()

        with self.m.temp_files() as (tmp_dir, logf, warmf, solnf):

            self._print_run_title(self.m.warmstartfile, self.m.solutionfile)
            self.m.run(warmstart=warmstart, tmp_dir=tmp_dir,
                       logf=logf, warmf=warmf, solnf=solnf)
            tdiff_solve = time.time() - t
            stat = ('Solver: ' + str(self.m.results.Solver[0]['Termination condition']))

            if self.io.replace_runs_if_exist and self.io.resume_loop:

                self.io.delete_run_id(self.run_id, operator='=')

            # append to output tables
            t = time.time()
            self.io.write_run(run_id=self.run_id)
            tdiff_write = time.time() - t

            # append to def_run table
            self.append_row(info=stat,
                            tdiff_solve=tdiff_solve, tdiff_write=tdiff_write)




