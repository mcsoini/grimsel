'''
Module doc
'''
import os
import numpy as np
import pandas as pd
import itertools
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

    # define loop steps and indices
    # linspace: float control parameter scaled between 0 and 1
    # arange: integer control parameter between 0 and n-1
    nsteps_default = [
          ('swtc', 2, np.arange),   # storage technology
          ('swst', 6, np.linspace), # capacity share of storage
          ('swvr', 3, np.linspace), # energy share of wind solar
          ('swyr', 5, np.arange),   # meteorological year
          ('swco', 3, np.arange),   # cost CO2 emissions
          ('swcd', 2, np.arange),   # charging only wind/solar yes/no
          ('swrc', 3, np.arange),   # net ramping cost
          ('swdr', 3, np.arange),   # discount rate
          ('swct', 1, np.arange),   # country
          ]
    @property
    def df_def_run(self):
        return self._df_def_run

    @df_def_run.setter
    def df_def_run(self, df_def_run):
        self._df_def_run = df_def_run
        self.restore_run_id()

#    def init_output_schema(self):
#
#        if self.sc_out: # output schema name is provided
#            self.unq_code = self.sc_out.replace('out_', '')
#        else: # generate output schema name
#            self.unq_code = datetime.datetime.now().strftime("%H%M")
#            self.sc_out = 'out_{n}_{uq}'.format(n=self.mkwargs['nhours'],
#                                                uq=self.unq_code)

    def __init__(self, **kwargs):
        '''
        Keyword arguments:
        nsteps -- list of model loop dimensions and steps; format:
                  (name::str, number_of_steps::int, type_of_steps::function)
        '''

        defaults = {
                    'nsteps': ModelLoop.nsteps_default,
                    'mkwargs': {},
                    'iokwargs': {}
                    }

        for key, val in defaults.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)

        self.run_id = None  # set later

        self.m = model_base.ModelBase(**self.mkwargs)

        self.iokwargs.update({'model': self.m})

        self.io = io.IO(**self.iokwargs)

        return

#        self.io.read_model_data()
#
#        self.m.map_to_time_res()
#
#        # Write tables which are generated in dependence on the time
#        # resolution (profiles, boundary conditions).
#        self.m.get_maximum_demand()
#        self.io.write_runtime_tables()
#
#        self.m.mps = maps.Maps(self._out, db=self.db)
#
#        self.m.build_model()
#
#        self.init_run_table()
#
#        self.io.init_output_tables()
#
#        self.select_run(0)


    def init_run_table(self):
        '''
        Initializes the ``df_def_run`` table by expanding the ``nsteps`` list.

        Expands the ``nsteps`` parameter to the corresponding DataFrame.
        The resulting attribute ``df_def_run`` contains all relevant
        combinations of the model change indices as defined in ``nsteps``.

        Also initializes the output ``def_run`` table, if required.
        '''

        _nsteps = [list(ist) + ([0, 1] if ist[-1] == np.linspace else [])
                  for ist in self.nsteps]

        list_steps = [list(istep[2](*istep[3:], istep[1]))
                      for istep in _nsteps]

        list_steps = [list(map(lambda x: float(x), lst))
                      for lst in list_steps]
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

    def append_row(self, zero_row=False, tdiff_solve=0, tdiff_write=0,
                   info=''):
        '''
        Generate single-line pandas.DataFrame to be appended to the
        output def_run table. Options:
        - zero_row == True: Row filled with zeros for calibration run
        - zero_row == False: Loop params copied to row
        '''


        dtypes = {int: ['run_id'] + list(self.dct_id),
                  float: (['tdiff_solve', 'tdiff_write', 'objective']
                          + list(self.dct_step.keys())),
                  str: ['info'] + list(self.dct_vl)}
        dtypes = {col: dtp  for dtp, cols in dtypes.items() for col in cols}

        if zero_row:
            df_add = aql.read_sql(self.io.db, self.sc_out, 'def_run')
            df_add.loc[0] = 0
            df_add[self.cols_step + self.cols_id + self.cols_step] = -1
            df_add['info'] = info
            df_add['run_id'] = -1 if not self.run_id else self.run_id
            df_add['tdiff_solve'] = tdiff_solve
            df_add['tdiff_write'] = tdiff_write
        else:
            vals = [[tdiff_solve, tdiff_write]
                    + [self.run_id] + [info] + list(self.dct_id.values())
                    + list(self.dct_step.values())
                    + list(self.dct_vl.values())]
            cols = (['tdiff_solve', 'tdiff_write', 'run_id', 'info']
                    + list(self.dct_id.keys()) + list(self.dct_step.keys())
                    + list(self.dct_vl.keys()))

            df_add = pd.DataFrame(vals, columns=cols)

        df_add['objective'] = (self.m.objective_value
                               if hasattr(self.m, 'objective_value') else 0)

        df_add = df_add.astype(dtypes)

        # can't use io method here if we want this to happen when no_output
        if self.io.modwr.output_target == 'psql':
            aql.write_sql(df_add, self.io.sql_connector.db,
                          self.io.cl_out, 'def_run', 'append')
        elif self.io.modwr.output_target == 'hdf5':
            with pd.HDFStore(self.io.cl_out, mode='a') as store:
                store.append('def_run', df_add, data_columns=True,
                             min_itemsize=150 # set string length!
                             )
        elif self.io.modwr.output_target in ['fastparquet']:
            fn = os.path.join(self.io.cl_out, 'def_run.parq')
            pq.write(fn, df_add, append=os.path.isfile(fn))

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

    def restore_run_id(self):
        '''
        Reset run_id index after external manipulation of the df_def_run
        '''

        cols_not_run_id = [c for c in self._df_def_run.columns
                           if not c == 'run_id']
        self._df_def_run = self._df_def_run[cols_not_run_id]
        self._df_def_run = self._df_def_run.reset_index(drop=True)
        self._df_def_run = self._df_def_run.reset_index()
        self._df_def_run = self._df_def_run.rename(columns={'index':
                                                              'run_id'})

    def perform_model_run(self, zero_run=False, warmstart=True):
        """
        TODO: This is a mess.

        Calls model_base run methods, io writing methods, and appends to
        def_run. Also takes care of time measurement for reporting in
        the corresponding def_run columns.

        Keyword arguments:
        zero_run -- boolean; perform a zero run (method do_zero_run in model_base)
                    (default False)

        """

        t = time.time()

        if zero_run:
            self.m.do_zero_run() # zero run method in model_base
        else:
            self._print_run_title(self.m.warmstartfile, self.m.solutionfile)
            self.m.run(warmstart=warmstart)
        tdiff_solve = time.time() - t
        stat = ('Solver: ' + str(self.m.results.Solver[0]['Termination condition']))

        if zero_run:

            if not self.io.resume_loop:
                self.io.write_run(run_id=-1)
                tdiff_write = time.time() - t
                self.append_row(zero_row=True, tdiff_solve=tdiff_solve,
                                tdiff_write=tdiff_write, info=stat)
        else:

            if self.io.replace_runs_if_exist and self.io.resume_loop:

                self.io.delete_run_id(self.run_id, operator='=')

            # append to output tables
            t = time.time()
            self.io.write_run(run_id=self.run_id)
            tdiff_write = time.time() - t

            # append to def_run table
            self.append_row(False, info=stat,
                            tdiff_solve=tdiff_solve, tdiff_write=tdiff_write)




