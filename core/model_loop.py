'''
Module doc
'''

import numpy as np
import pandas as pd
import itertools
import time
from importlib import reload

import grimsel.core.model_base as model_base
import grimsel.core.io as io
import grimsel.core.model_loop_modifier as model_loop_modifier
import grimsel.core.parameter_changes as parameter_changes

import grimsel.auxiliary.aux_sql_func as aql
import grimsel.auxiliary.maps as maps
from grimsel.auxiliary.aux_general import get_config

reload(model_base)
reload(io)
reload(model_loop_modifier)
reload(parameter_changes)

class ModelLoop(parameter_changes.ParameterChanges):
    '''
    Defines the model loop framework. Main attribute is the dataframe
    df_def_loop.

     -> variables:
         - dicts
         - def_loop
     -> functions:
         - generate run name
         - print_run_title
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
    def df_def_loop(self):
        return self.__df_def_loop

    @df_def_loop.setter
    def df_def_loop(self, df_def_loop):
        self.__df_def_loop = df_def_loop
        self.restore_run_id()

    def init_output_schema(self):

        if self.sc_out: # output schema name is provided
            self.unq_code = self.sc_out.replace('out_', '')
        else: # generate output schema name
            self.unq_code = datetime.now().strftime("%H%M")
            self.sc_out = 'out_' + str(self.mkwargs['nhours'])
            self.sc_out += '_' + self.unq_code

    def __init__(self, sql_connector, **kwargs):
        '''
        Keyword arguments:
        nsteps -- list of model loop dimensions and steps; format:
                  (name::str, number_of_steps::int, type_of_steps::function)
        '''

        # define instance attributes and update with kwargs
        defaults = {
                    'nsteps': ModelLoop.nsteps_default,
#                    'ModelLoopModifier': model_loop_modifier.ModelLoopModifier,
                    'dev_mode': False,
                    'mkwargs': None,
                    'iokwargs': None,
                    'sc_inp': None,
                    'sc_out': None,
                    'db': None}

        for key, val in defaults.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)

        self.init_output_schema()

        self.mkwargs.update({'unq_code': self.unq_code})

        self.run_id = None  # set later

        # SETTING UP MODEL AND IO

        self.m = model_base.ModelBase(**self.mkwargs)

#        self.mlm = self.ModelLoopModifier(self)

        self.iokwargs.update({'model': self.m,
                              'dev_mode': self.dev_mode,
                              'sql_connector': sql_connector,
                              'db': self.db,
                              'sc_out': self.sc_out,
                              'sc_inp': self.sc_inp})

        self.io = io.IO(**self.iokwargs)

        self.io.read_model_data()

        self.m.mps = maps.Maps(self.sc_out, db=self.db)

        self.m.map_to_time_res()

        # Write tables which are generated in dependence on the time
        # resolution (profiles, boundary conditions).
        self.m.get_maximum_demand()
        self.io.write_runtime_tables()

        # Call method build_model (requires data read by io);
        self.m.build_model()

        self.io.init_output_tables()


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
        self.cols_all = self.cols_id + self.cols_step + self.cols_val

        self.__df_def_loop = pd.DataFrame(full_all, columns=self.cols_all)
        self.__df_def_loop = self.__df_def_loop.reset_index()
        col_dict = {'index': 'run_id'}
        self.__df_def_loop = self.__df_def_loop.rename(columns=col_dict)

        if not self.io.resume_loop:
            self.init_loop_table()

        self.select_run(0)

    def select_run(self, slct_run_id):
        '''
        Get all relevant indices and parameters for a certain slct_run_id
        '''

        self.run_id = slct_run_id

        lpsrs = self.df_def_loop.loc[self.df_def_loop.run_id
                                     == slct_run_id].iloc[0]

        self.dct_vl = lpsrs.loc[lpsrs.index.str.contains('_vl')].to_dict()
        self.dct_id = lpsrs.loc[lpsrs.index.str.contains('_id')
                               & ~lpsrs.index.str.contains('run')].to_dict()
        self.dct_step = lpsrs.loc[~lpsrs.index.str.contains('_vl')
                                 & ~lpsrs.index.str.contains('_id')].to_dict()
        self.dct_id = {k: int(v) for (k, v) in self.dct_id.items()}

        self.loop_series = lpsrs

    def init_loop_table(self):
        tb_name = 'def_loop'
        cols = ([('tdiff_solve', 'DOUBLE PRECISION'),
                 ('tdiff_write', 'DOUBLE PRECISION'),
                 ('run_id', 'SMALLINT'),
#                 ('run_name', 'VARCHAR(200)')
                 ]
              + [(s, 'SMALLINT') for s in self.cols_id]
              + [(s, 'DOUBLE PRECISION') for s in self.cols_step]
              + [(s, 'VARCHAR(30)') for s in self.cols_val]
              + [('info', 'VARCHAR'), ('objective', 'DOUBLE PRECISION')])
        aql.init_table(tb_name, cols, self.sc_out,
                       pk=self.cols_id, unique=['run_id'],
                       db=self.io.db)



    def append_row(self, zero_row=False, tdiff_solve=0, tdiff_write=0,
                   info=''):
        '''
        Generate single-line pandas.DataFrame to be appended to the
        output def_loop table. Options:
        - zero_row == True: Row filled with zeros for calibration run
        - zero_row == False: Loop params copied to row
        '''

        if zero_row:
            df_add = aql.read_sql(self.io.db, self.sc_out, 'def_loop')
            df_add.loc[0] = 0
            df_add[self.cols_step + self.cols_id + self.cols_step] = -1
            df_add['info'] = info
            df_add['run_id'] = -1 if not self.run_id else self.run_id
            df_add['tdiff_solve'] = tdiff_solve
            df_add['tdiff_write'] = tdiff_write
        else:
            df_add = pd.DataFrame(np.array([tdiff_solve, tdiff_write]
                                  + [self.run_id] + [info]
                                  + list(self.dct_id.values())
                                  + list(self.dct_step.values())
                                  + list(self.dct_vl.values()))).T
            df_add.columns = (['tdiff_solve', 'tdiff_write', 'run_id', 'info']
                             + list(self.dct_id.keys())
                             + list(self.dct_step.keys())
                             + list(self.dct_vl.keys()))

        df_add['objective'] = (self.m.objective_value
                               if 'objective_value' in self.m.__dict__.keys()
                               else 0)

        # update instance variable and add run_name column
        self.df_add = df_add
#        self.add_run_name()

        # can't use io method here if we want this to happen when no_output
        aql.write_sql(self.df_add, self.db, self.sc_out, 'def_loop', 'append')


        # write total def_loop to csv file
        aql.read_sql(self.io.db, self.sc_out, 'def_loop')\
          .to_csv('def_loop_' + self.sc_out + '.csv')

#    def add_run_name(self, exclude=['swct_vl']):
#        df = self.df_add
#        self.df_add['run_name'] = \
#           self.df_add[['run_id'] + \
#           [c for c in df.columns if str.find(c, '_vl') > -1 and not c in exclude]]\
#              .apply(lambda x:
#                     '; '.join([str(int(x[0])).zfill(3 if int(x[0]) >= 0 else 4)]
#                               + [str(s) for s in x]), axis=1)

    def print_run_title(self, warmstartfile, solutionfile):

        print('*'*60)
        print('run_id = ' + str(self.run_id)
              + ' of ' + str(self.df_def_loop['run_id'].max()))
        print('\n'.join([str(c[0]) + ' = ' + str(c[1])
                         for c in self.dct_vl.items()]))
        print('*'*60)
        print('Using warmstartfile: ', warmstartfile)
        print('Using solutionfile: ', solutionfile)

    def restore_run_id(self):
        '''
        Reset run_id index after external manipulation of the df_def_loop
        '''

        cols_not_run_id = [c for c in self.__df_def_loop.columns
                           if not c == 'run_id']
        self.__df_def_loop = self.__df_def_loop[cols_not_run_id]
        self.__df_def_loop = self.__df_def_loop.reset_index(drop=True)
        self.__df_def_loop = self.__df_def_loop.reset_index()
        self.__df_def_loop = self.__df_def_loop.rename(columns={'index':
                                                                'run_id'})

    def perform_model_run(self, zero_run=False, warmstart=True):
        """
        Calls model_base run methods, io writing methods, and appends to
        def_loop. Also takes care of time measurement for reporting in
        the corresponding def_loop columns.

        Keyword arguments:
        zero_run -- boolean; perform a zero run (method do_zero_run in model_base)
                    (default False)
        """

        t = time.time()
        if zero_run:
            self.m.do_zero_run() # zero run method in model_base
        else:
            self.print_run_title(self.m.warmstartfile, self.m.solutionfile)
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
            # append to output tables
            t = time.time()
            self.io.write_run(run_id=self.run_id)
            tdiff_write = time.time() - t

            self.append_row(False, info=stat,
                            tdiff_solve=tdiff_solve, tdiff_write=tdiff_write)

        # retirements or investments parameter back to inf
        self.capchnge_max = float('Inf')



