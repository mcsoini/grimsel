'''
Module docstring
'''


'''
BETTER WAY TO WARMSTART?
https://github.com/Pyomo/pyomo/blob/master/doc/GettingStarted/current/examples/ipopt_warmstart.py#L65

'''



import tempfile
import warnings
import numpy as np

import pandas as pd

import logging

logger = logging.Logger('grimsel')

import pyomo.environ as po
from pyomo.core.base.objective import SimpleObjective

from grimsel.core.io import IO as IO

def create_tempfile(self, suffix=None, prefix=None, text=False, dir=None):
    """
    Return the absolute path of a temporary filename that is
    guaranteed to be unique.  This function generates the file and returns
    the filename.
    """

    print('''
          create_tempfile is monkey patched
          ''')

    if suffix is None:
        suffix = ''
    if prefix is None:
        prefix = 'tmp'

    ans = tempfile.mkstemp(suffix=suffix, prefix=prefix, text=text, dir=dir)
    ans = list(ans)
    if not os.path.isabs(ans[1]):  #pragma:nocover
        fname = os.path.join(dir, ans[1])
    else:
        fname = ans[1]
    os.close(ans[0])

    dir = tempfile.gettempdir()

    new_fname = os.path.join('ephemeral'
                             + ''.join(np.random.choice(list('abcdefghi'), 4))
                             + suffix)
    # Delete any file having the sequential name and then
    # rename
    if os.path.exists(new_fname):
        os.remove(new_fname)
    fname = new_fname

    self._tempfiles[-1].append(fname)
    return fname

import pyutilib.component.config.tempfiles as tempfiles
tempfiles.TempfileManagerPlugin.create_tempfile = create_tempfile


import sys
import os
from importlib import reload

import numpy as np
import pandas as pd
import pyomo.environ as po
from pyomo.opt import SolverFactory
from datetime import datetime

# auxiliary modules
import grimsel.auxiliary.sqlutils.aux_sql_func as aql
import grimsel.auxiliary.maps as maps
import grimsel.auxiliary.timemap as timemap
from grimsel.auxiliary.aux_m_func import pdef, set_to_list

# model components
import grimsel.core.constraints as constraints
import grimsel.core.variables as variables
import grimsel.core.parameters as parameters
import grimsel.core.sets as sets
import grimsel.core.io as io # for class methods

TEMP_DIR = tempfile.gettempdir()

reload(constraints)
reload(variables)
reload(parameters)
reload(sets)

class ModelBase(po.ConcreteModel, constraints.Constraints,
                parameters.Parameters, variables.Variables, sets.Sets):

    # class attributes as defaults for presolve_fixed_capacities
    list_vars = [('var_yr_cap_pwr_rem', 'cap_pwr_rem'),
                 ('var_yr_cap_pwr_new', 'cap_pwr_new')]
    list_constr_deact = ['set_win_sol']

#    db = get_config('sql_connect')['db']


    def __init__(self, **kwargs):
        '''
        Keyword arguments:
        nhours -- time resolution of the model, used for profile scaling
        sc_warmstart -- input database schema for presolving
        slct_node -- limit node selection
        slct_encar -- limit to energy carrier selection
        skip_runs -- boolean; if True, solver calls are skipped, also
                     stops the IO instance from trying to write the model
                     variables.
        '''

        super(ModelBase, self).__init__() # init of po.ConcreteModel

        # define instance attributes and update with kwargs
        defaults = {'slct_node': [],
                    'slct_pp_type': [],
                    'slct_encar': ['EL'],
                    'nhours': 48,
                    'unq_code': '',
                    'mps': None,
                    'tm_filt': False,
                    'verbose_solver': True,
                    'constraint_groups': None,
                    'skip_runs': False,
                    'nthreads': False}
        for key, val in defaults.items():
            setattr(self, key, val)
        self.__dict__.update(kwargs)

        self._check_contraint_groups()

        # translate node and energy carrier selection to ids
#        self.slct_node_id = [self.mps.dict_nd_id[x] for x in self.slct_node]
#        self.slct_encar_id = [self.mps.dict_ca_id[x] for x in self.slct_encar]
#        self.slct_pp_type_id = [self.mps.dict_pt_id[x] for x in self.slct_pp_type]

        print('self.slct_encar=' + str(self.slct_encar))
        print('self.slct_pp_type=' + str(self.slct_pp_type))
        print('self.slct_node=' + str(self.slct_node))
        print('self.nhours=' + str(self.nhours))
        print('self.constraint_groups=' + str(self.constraint_groups))

        self.warmstartfile = self.solutionfile = None

        # attributes for presolve_fixed_capacities
        self.list_vars = ModelBase.list_vars
        self.list_constr_deact = ModelBase.list_constr_deact

    def build_model(self):
        '''
        Call the relevant model methods to get everything set up.

        This consists in:
        1. call self.get_setlst (in Sets mixin class) to initialize
            the self.setlst dictionary
        2. call self.define_sets (in Sets mixin class) to initialize
            Pyomo set objects
        3. call self.define_parameters (in Parameters mixin class)
        4. call self.define_variables (in Variables mixin class)
        5. call self.add_all_constraints
        6. call self.init_solver

        Note: io needs to have loaded all data, i.e. set the ModelBase
        DataFrames.
        '''

        self.get_setlst()
        self.define_sets()

        self.define_parameters()

        if not self.skip_runs:

            self.define_variables()
            self.add_all_constraints()
            self.init_solver()

    @classmethod
    def get_constraint_groups(cls, excl=None):
        '''
        Returns list names of methods defining constraint groups.

        This classmethod can also be used to define the constraint_groups
        parameter to initialize the ModelBase object by selecting certain
        groups to be excluded.

        Parameters:
        excl : exclude certain group names from the returned list
        '''

        cg_lst = [mth.replace('add_', '').replace('_rules', '')
                  for mth in dir(cls)
                  if mth.startswith('add_') and 'rule' in mth]

        if excl:
            cg_lst = [cg for cg in cg_lst if not cg in excl]

        return cg_lst


    def _check_contraint_groups(self):
        '''
        Verification and completion of the constraint group selection.

        Verifies constraint groups if the `constraint_groups` argument
        is not None. Otherwise it gathers all accordingly named
        methods from the class attributes and populates the list thusly.
        '''

        cg_options = self.get_constraint_groups()

        if self.constraint_groups is None:
            self.constraint_groups = cg_options
        else:
            # get invalid constraint groups in input
            nv = [cg for cg in self.constraint_groups
                  if not cg in cg_options]

            if nv:
                estr = ('Invalid constraint group(s): {nv}.'
                        + '\nPossible choices are:\n{cg}'
                        ).format(nv=', '.join(nv), cg=',\n'.join(cg_options))
                raise ValueError(estr)

    def add_all_constraints(self):
        '''
        Call all selected methods from the constraint mixin class.

        Loops through the `constraint_groups` list and calls the corresponding
        methods in the :class:`.Constraints` mixing class.
        '''

        for cg in self.constraint_groups:
            getattr(self, 'add_%s_rules'%cg)()


    def _limit_prof_to_cap(self):

        if len(self.chp) > 0:
            self.limit_prof_to_cap()

#    def limit_prof_to_cap(self):
#        '''
#        Make sure CHP profiles don't ask for more power than feasible.
#
#        This operates on the parameters and is called before each model run.
#        '''
#
#        df_prf = self.df_profchp_soy.copy().rename(columns={'value': 'prof'})
#        df_prf = df_prf.join(self.df_tm_soy.set_index(['sy'])['weight'], on='sy')
#        df_prf['prof'] *= df_prf.weight
#        df_erg = IO.param_to_df(self.erg_chp, ('nd_id', 'ca_id', 'fl_id')).rename(columns={'value': 'erg'})
#        df_erg = df_erg.loc[df_erg.erg > 0]
#        df_cap = IO.param_to_df(self.cap_pwr_leg, ('pp_id', 'ca_id')).rename(columns={'value': 'cap'})
#
#        cap_scale = IO.param_to_df(self.cap_avlb, ('mt_id', 'pp_id', 'ca_id')).rename(columns={'value': 'cap_scale'})
#        df_cap = cap_scale.join(df_cap.set_index(['pp_id', 'ca_id']), on=['pp_id', 'ca_id'])
#        df_cap['cap'] *= df_cap.cap_scale
#
#        df_cap = df_cap.join(self.df_def_plant.set_index('pp_id')[['fl_id', 'nd_id']], on='pp_id')
#
#        df_cap = df_cap.pivot_table(index=['nd_id', 'fl_id', 'mt_id'],
#                           values='cap', aggfunc=sum)
#
#        df_exp = pd.merge(df_prf, df_erg, on=['nd_id', 'ca_id'], how='outer')
#
#        df_exp['prof_scaled'] = df_exp.prof * df_exp.erg
#        df_exp = df_exp.join(self.df_tm_soy_full.set_index('sy')['mt_id'], on='sy')
#
#        df_exp.pivot_table(values='prof', aggfunc=sum, index=['nd_id', 'fl_id'])
#        df_exp.pivot_table(values='erg', aggfunc=[np.min, max], index=['nd_id', 'fl_id'])
#
#
#
#
#        df_exp = df_exp.join(df_cap, on=df_cap.index.names)
#
#        mask_viol = df_exp.prof_scaled > df_exp.cap
#        df_exp.loc[mask_viol, 'prof_scaled'] = df_exp.loc[mask_viol, 'cap'] * 0.999
#
#        df_exp['prof_new'] = df_exp.prof_scaled / df_exp.erg
#
#        df_exp.pivot_table(index='sy', values=['prof_new', 'prof'], columns=['fl_id', 'nd_id'], aggfunc=sum).plot()
#
#
#        df_prf_new = df_exp.pivot_table(index=['sy', 'nd_id', 'ca_id'], values='prof_new', aggfunc=min).reset_index()
#
#
#        print('New sum of profile')
#        df_print = df_prf_new.pivot_table(index='nd_id', values='prof_new', aggfunc=sum).reset_index()
#        df_print['nd_id'] = df_print.nd_id.replace(self.mps.dict_nd)
#        print(df_print)
#
#
#        dict_prf_new = df_prf_new.set_index(['sy', 'nd_id', 'ca_id'])['prof_new'].to_dict()
#
#        for key, val in dict_prf_new.items():
#            self.chpprof[key].value = val


    def limit_prof_to_cap(self, param_mod='cap_pwr_leg'):
        '''
        Make sure CHP profiles don't ask for more power than feasible.

        This operates on the parameters and is called before each model run.
        '''

        print('*'*60 + '\nModelBase: Limiting chp profiles to cap_pwr_leg', end='... ')

        # get list of plants relevant for chp from corresponding set
        pp_chp = self.setlst['chp']

        df_chpprof = io.IO.param_to_df(self.chpprof, ('sy', 'nd_id', 'ca_id'))
        df_erg_chp = io.IO.param_to_df(self.erg_chp, ('pp_id', 'ca_id'))
        df_erg_chp = df_erg_chp.loc[df_erg_chp.pp_id.isin(pp_chp)]
        df_erg_chp['nd_id'] = df_erg_chp.pp_id.replace(self.mps.dict_plant_2_node_id)

        # outer join profiles and energy to get a profile for each fuel
        df_chpprof_tot = pd.merge(df_erg_chp.rename(columns={'value': 'erg'}),
                                  df_chpprof.rename(columns={'value': 'prof'}),
                                  on=['nd_id', 'ca_id'])
        # scale profiles
        df_chpprof_tot['prof_sc'] = df_chpprof_tot['erg'] * df_chpprof_tot['prof']

        # get capacities from parameter
        df_cap_pwr_leg = io.IO.param_to_df(self.cap_pwr_leg, ('pp_id', 'ca_id'))
        # keep only chp-related fuels
        df_cap_pwr_leg = df_cap_pwr_leg.loc[df_cap_pwr_leg.pp_id.isin(self.chp)]
        # pivot_by fl_id
        df_cappv = df_cap_pwr_leg.pivot_table(values='value',
                                              index=['ca_id', 'pp_id'],
                                              aggfunc=np.sum)['value']
        # rename
        df_cappv = df_cappv.rename('cap').reset_index()

        # add capacity to profiles
        df_chpprof_tot = pd.merge(df_cappv, df_chpprof_tot, on=['ca_id', 'pp_id'])


        # find occurrences of capacity zero and chp erg non-zero
        df_slct = df_chpprof_tot[['pp_id', 'ca_id', 'cap', 'erg']].drop_duplicates().copy()
        df_slct = df_slct.loc[df_slct.cap.isin([0])
                            & -df_slct.erg.isin([0])]
        str_erg_cap = ''
        if len(df_slct > 0):
            for nrow, row in df_slct.iterrows():
                str_erg_cap += 'pp_id=%d, ca_id=%d: cap_pwr_leg=%f, erg_chp=%f\n'%tuple(row.values)
            raise ValueError ('limit_prof_to_cap: one or more cap_pwr_leg are zero '
                              'while erg_chp is greater 0: \n' + str_erg_cap)

        # find occurrences of capacity violations
        mask_viol = df_chpprof_tot.prof_sc > df_chpprof_tot.cap


        if mask_viol.sum() == 0:
            print('ok, nothing changed.\n' + '*'*60)
        else:
            # REPORTING
            df_profviol = df_chpprof_tot.loc[mask_viol]
            dict_viol = df_profviol.pivot_table(index=['pp_id', 'ca_id'],
                                                values='sy', aggfunc=len)['sy'].to_dict()

            for kk, vv in dict_viol.items():
                print('\n(pp, ca)={}: {} violations'.format(kk, vv))

            print('Modifing model parameter ' + param_mod, end=' ... ')

            if param_mod == 'chpprof':

                df_profviol['prof'] *=  0.999 * df_chpprof_tot.cap / df_chpprof_tot.prof_sc
                dict_chpprof = (df_profviol.pivot_table(index=['sy', 'nd_id', 'ca_id'],
                                                        values='prof', aggfunc=min)['prof']
                                           .to_dict())

                for kk, vv in dict_chpprof.items():
                    self.chpprof[kk] = vv

            elif param_mod == 'cap_pwr_leg':

                # calculate capacity scaling factor
                df_capsc = df_profviol.pivot_table(index=['pp_id', 'ca_id'],
                                                   values=['cap', 'prof_sc'], aggfunc=np.max)
                df_capsc['cap_sc'] = df_capsc.prof_sc / df_capsc.cap

                # merge scaling factor with capacity table
                df_cap_pwr_leg = df_cap_pwr_leg.join(df_capsc,
                                                     on=df_capsc.index.names)
                df_cap_pwr_leg = df_cap_pwr_leg.loc[-df_cap_pwr_leg.cap_sc.isnull()]

                # apply scaling factor to all capacity with the relevant fuel
                df_cap_pwr_leg['cap'] *= df_cap_pwr_leg.cap_sc * 1.0001

                # dictionary
                dict_cap_pwr_leg = df_cap_pwr_leg.set_index(['pp_id', 'ca_id'])['cap']
                dict_cap_pwr_leg = dict_cap_pwr_leg.to_dict()

                for kk, vv in dict_cap_pwr_leg.items():
                    self.cap_pwr_leg[kk] = vv

            print('done.\n' + '*'*60)

    def _init_pf_dicts(self):
        '''
        Initializes dicts mapping the profile ids to other model ids.

        This results in dictionaries which are assigned as :class:`ModelBase`
        instance attributes:

        * ``dict_price_pf``: (fl_id, nd_id, ca_id) |rarr| (price_pf_id)
        * ``dict_dmnd_pf``: (nd_id, ca_id) |rarr| (dmnd_pf_id)
        * ``dict_supply_pf``: (pp_id, ca_id) |rarr| (supply_pf_id)

        Use:
        ____

        The resulting dictionaries are used for filtering the profile tables
        in the :module:`io` module and to access the profile parameters
        in the model :class:`Constraints`.

        '''

        list_pf = [(self.df_fuel_node_encar,
                    ['fl_id', 'nd_id', 'ca_id'], 'price'),
                   (self.df_node_encar,
                    ['nd_id', 'ca_id'], 'dmnd'),
                   (self.df_plant_encar,
                    ['pp_id', 'ca_id'], 'supply')]


        df, ind, name = list_pf[-1]
        for df, ind, name in list_pf:

            col = '%s_pf_id'%name
            dct = df.loc[~df[col].isna()].set_index(ind)[col].to_dict()

            setattr(self, 'dict_%s_pf'%name, dct)


    def map_to_time_res(self):
        '''
        Generates a map between hours-of-the-year and time slots-of-the-year
        based on the fixed time resolution self.nhours, using the class
        timemap.TimeMap. Then maps relevant input data from hours to slots.
        Also generates dictionaries which contain all slot ids for each
        week/month and vice versa.
        '''

        # get time maps and dicts
        self.tm = timemap.TimeMap(self.nhours, self.tm_filt)
        self.df_tm_soy_full = self.tm.df_time_red # save to output database
        self.df_hoy_soy = self.tm.df_hoy_soy
        self.df_tm_soy = self.tm.df_time_red[['wk_id', 'mt_id', 'sy',
                                              'weight', 'wk_weight']]

        # scale fixed costs to account for less than full year
        self.adjust_cost_time()

        _df = self.df_tm_soy.copy()

        # get dictionaries month/week <-> time slots;
        # these are used in the constraint definitions
        for cl, nm in [('wk_id', 'week'), ('mt_id', 'month')]:
            d = {mm: _df.loc[_df[cl] == mm, 'sy']
                     .get_values().tolist() for mm in _df[cl].unique()}
            setattr(self, 'dict_' + nm + '_soy', d)
            setattr(self, 'dict_soy_' + nm, {s: w for w in d for s in d[w]})

        # map hydro boundary conditions (which refer to the beginning
        # of the month) to time slots
        if not self.df_plant_month is None:
            _df = self.df_def_month[['mt_id', 'month_min_hoy']].set_index('mt_id')
            self.df_plant_month = self.df_plant_month.join(_df, on=_df.index.name)
            _df = self.df_hoy_soy.rename(columns={'hy': 'month_min_hoy'})
            _df = _df.set_index('month_min_hoy')
            self.df_plant_month = self.df_plant_month.join(_df, on=_df.index.names)

            sy_null = self.df_plant_month.sy.isnull()
            self.df_plant_month = self.df_plant_month.loc[-sy_null]


        # Map profiles and bc to soy
        for itb, idx in [
                         ('dmnd', ['dmnd_pf_id', 'sy']),
                         ('price', ['price_pf_id', 'sy']),
                         ('inflow', ['pp_id', 'ca_id', 'sy']),
                         ('supply', ['supply_pf_id', 'sy']),
                         ('chp', ['nd_id', 'ca_id', 'sy']),
                        ]:

            name_df = 'df_prof' + itb
            print('Averaging table {} to time resolution {} hours.'
                  .format(name_df, self.nhours))

            df_tbsoy = getattr(self, name_df)
            if not df_tbsoy is None:
                df_tbsoy = df_tbsoy.join(self.df_hoy_soy.set_index('hy'), on='hy')
                val = [c for c in df_tbsoy.columns if not c in idx + ['hy']]
                df_tbsoy[val] = df_tbsoy[val].astype(float)
                df_tbsoy = df_tbsoy.pivot_table(values=val, index=idx,
                                                aggfunc=np.mean).reset_index()

                if df_tbsoy.empty:
                    df_tbsoy = pd.DataFrame(columns=idx+val)

                setattr(self, 'df_prof' + itb + '_soy', df_tbsoy)

    def adjust_cost_time(self):
        '''
        Scale fixed costs for incomplete years.

        If the model year doesn't have 8760 hours, fixed costs are scaled
        to keep levelized costs the same.

        This is relevant if the tm_filt ModelBase parameter is used to
        work with a simplified model version.
        '''
        tm_filt_weight = self.tm.tm_filt_weight

        lstfc = [c for c in self.df_plant_encar.columns if c.startswith('fc_')]

        self.df_plant_encar[lstfc] /= tm_filt_weight

    def get_maximum_demand(self):
        '''
        Calculation of maximum demand (in MW) with adjusted time resolution.

        Note: Database table def_node is updated in io.write_runtime_tables
        '''

        if (hasattr(self, 'df_profdmnd_soy')
            and len(self.df_profdmnd_soy.index) > 0):
            _df = self.df_profdmnd_soy
            df_dmd_params = _df.pivot_table(values=['value'],
                                            index='dmnd_pf_id', aggfunc=[max])
            df_dmd_params.columns = ['dmnd_max']


            ndca = [{val: key for key, val in self.dict_dmnd_pf.items()}[pf]
                    for pf in df_dmd_params.reset_index().dmnd_pf_id]
            df_dmd_params[['nd_id', 'ca_id']] = pd.DataFrame(ndca)
            df_dmd_params = df_dmd_params.set_index('nd_id')[['dmnd_max']]


            self.df_def_node.drop([c for c in self.df_def_node.columns
                                   if c in df_dmd_params.columns],
                                  axis=1, inplace=True)

            self.df_def_node = self.df_def_node.join(df_dmd_params, on='nd_id')

    def switch_soln_file(self, isolnfile):
        '''
        Warmstart and solutionfiles are alternated between two to
        allow to use the last solution as starting values but avoid
        excessive disk space use (which would be the case if we kept all
        solution files).

        Parameters:
        isolnfile -- binary index of last solution file
        '''
        isolnfile = 1 - isolnfile
        solnfile = os.path.join(TEMP_DIR,
                                ('manual_soln_file_{uc}_{i}.cplex.sol'
                                 .format(uc=self.unq_code, i=str(isolnfile))))
        return solnfile, isolnfile

    def init_solver(self):
        '''
        Create pyomo Solverfactory instance and adjust parameters.
        '''
        self.dual = po.Suffix(direction=po.Suffix.IMPORT)

        if sys.platform == 'win32':
            self.solver = SolverFactory("cplex")
        elif sys.platform in ['linux2', 'linux']:
            exec_str = ('/opt/ibm/ILOG/CPLEX_Studio1271/cplex/bin/'
                        +'x86-64_linux/cplex')
            self.solver = SolverFactory("cplex", executable=exec_str)
        elif sys.platform == 'darwin':
            exec_str = ('/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
            self.solver = SolverFactory("cplex", executable=exec_str)

        if self.nthreads:
            self.solver.set_options('threads=' + str(self.nthreads))

        fn = 'manual_log_file_{uc}.cplex.sol'.format(uc=self.unq_code)
        self.logfile = os.path.join(TEMP_DIR, fn)
        self.solver._problem_files = (os.path.join(TEMP_DIR, 'pyomo.lp'),)

        # init of solutionfile
        self.isolnfile = 0
        self.solutionfile, self.isolnfile = self.switch_soln_file(1)
        self.warmstartfile = None

    def check_valid_indices(self, index):
        '''
        Used in parameter and variable definitions.

        Parameters:
        index: tuple of pyomo sets
        '''


        is_empty = [pi.name for pi in index if (not pi is None) and not pi]
        is_none = [pi is None for pi in index]
        if any(is_empty) + any(is_none):
            print(('failed: set(s) {} is/are '
                   + 'empty or None.').format(str(is_empty)))
            return False
        else:
            return True

    def delete_component(self, comp_name):
        '''
        Drop a component of the pyomo model.

        A single component object is associated with various index objects.
        Because of this, some looping over self.__dict__ is required
        to catch 'em all.
        '''

        list_del = []
        for kk in self.__dict__.keys():
            if ((comp_name == kk)
                or (comp_name + '_index' in kk)
                or (comp_name + '_domain' in kk)):
                list_del.append(kk)
        print('Deleting model components ({}).'.format(', '.join(list_del)))
        for kk in list_del:
            self.__dict__.pop(kk, None)

    def run(self, warmstart=False):
        '''
        Run the model. Then switch solution/warmstartfile.

        Unless skip_runs is True. Then just create a pro-forma results object.
        '''

#        self.fix_scenario_plants() # is this too specific here??

        if self.skip_runs:
            class Result: pass # ad-hoc class mimicking the results object
            self.results = Result()
            self.results.Solver = [{'Termination condition':
                                    'Skipped due to skip_runs=True.'}]
        else:
            self.results = self.solver.solve(self, tee=self.verbose_solver,
                                             keepfiles=True,
                                             warmstart=warmstart,
                                             solnfile=self.solutionfile,
                                             logfile=self.logfile,
                                             warmstart_file=self.warmstartfile)
            self.warmstartfile = self.solutionfile
            sf, isf = self.switch_soln_file(self.isolnfile)
            self.solutionfile, self.isolnfile = [sf, isf]

            self._get_objective_value()

    def _get_objective_value(self):
        '''
        Making the objective value a ModelBase instance attribute.

        This assumes that among the objects defined by list_obj_name
        only one is
        '''

        list_name_obj = ['objective_lin', 'objective_quad', 'objective']


        if self.results.solver.termination_condition.key == 'optimal':
            for name_obj in list_name_obj:
                obj = getattr(self, name_obj, False)
                if obj and isinstance(obj, SimpleObjective) and obj.active:
                    self.objective_value = po.value(obj)
        else:
            self.objective_value = np.nan


    def print_is_fixed(self, variable='cap_pwr_new'):
        '''
        Print all fixed elements of the variable given as an input.
        TODO: This assumes pp as first index: needs to change!
        Keyword arguments:
        variable -- name string of a pyomo variable
        '''
        vv = getattr(self, variable)
        print('*'*15 + ' ' + variable + ': ' + '*'*15)
        for i in vv:
            print('vv:', vv)
            if vv[i].fixed:
                print(str(self.mps.dict_pp[i[0]]) + ': is fixed at '
                      + str(vv[i].value))
        print('*'*(33 + len(variable)) + '\n')

    def fill_peaker_plants(self, demand_factor=1.02, reset_to_zero=False,
                           list_peak=[]):
        '''
        Calculate required capacity of designated peaker plants from
        power capacity and demand profiles.
        This serves to avoid infeasibilities due to insufficient installed
        capacity in case the capacities are fixed.
        Keyword arguments:
        demand_factor -- multiplied with the peak load value, this determines
                         the required dispatchable power capacity
        from_variable -- boolean, if True, use variable 'cap_pwr_tot' instead
                         of default 'cap_pwr_leg'
        reset_to_zero -- boolean, if True: stop after the reset-to-zero stage

        '''

        list_attr = ['cap_pwr_leg']

        # reset to zero: get list of peaker plants ppca indices from
        # df_plant_encar then set both cap_pwr_tot and cap_pwr_leg to zero
        if not list_peak:
            _df = self.df_plant_encar
            _list_peak = _df.loc[_df['pp_id'].isin(self.setlst['peak']),
                                  ['pp_id', 'ca_id']].drop_duplicates()
            _list_peak = _list_peak.apply(tuple, axis=1).tolist()
        else:
            _list_peak = list_peak

        for iattr in list_attr:
            for kk in list_peak:
                getattr(self, iattr)[kk] = 0

        if (self.setlst['peak'] or list_peak) and not reset_to_zero:

            slct_attr = 'cap_pwr_leg'

            # Ids all plants considered dispatchable, not peaker plants.
            slct_pp = [pp for pp in
                       self.setlst['pp']
                       + self.setlst['st']
                       + self.setlst['hyrs']
                       if not pp in self.setlst['peak']]

            list_cap = [tuple(list(cc)
                              + [self.mps.dict_plant_2_node_id[cc[0]]]
                              + [getattr(self, slct_attr)[cc].value])
                        for cc in getattr(self, slct_attr) if cc[0] in slct_pp]

            # generate df with all capacities of dispatchable plants
            _df = pd.DataFrame(list_cap, columns=['pp_id', 'ca_id',
                                                  'nd_id', 'cap_pwr'])

            df_dmd_max = self.df_def_node.set_index('nd_id')['dmnd_max']
            df_dmd_max *= demand_factor

            df_cap_tot = pd.DataFrame(_df.pivot_table(values=['cap_pwr'],
                                          index='nd_id', aggfunc=sum),
                                          columns=['cap_pwr'],
                                          index=df_dmd_max.index).fillna(0)
            df_cap_tot = df_cap_tot['cap_pwr']

            dict_nd_pp = {nd: pp for pp, nd in
                          self.mps.dict_plant_2_node_id.items()
                          if pp in list(zip(*_list_peak))[0]}

            df_cap_peak = df_dmd_max - df_cap_tot
            df_cap_peak = df_cap_peak.apply(lambda x: max(0., x))
            df_cap_peak = df_cap_peak.reset_index()
            df_cap_peak['pp_id'] = df_cap_peak['nd_id']
            df_cap_peak = df_cap_peak.loc[df_cap_peak.nd_id.isin(list(dict_nd_pp.keys()))]
            df_cap_peak['pp_id'] = df_cap_peak['pp_id'].replace(dict_nd_pp)
            df_cap_peak['ca_id'] = 0

            df_cap_peak = df_cap_peak.set_index(['pp_id', 'ca_id'])
            dict_peak_1 = df_cap_peak[0].to_dict()

            for iattr in list_attr:
                for kk, vv in dict_peak_1.items():
                    getattr(self, iattr)[kk] = vv


#    def presolve_fixed_capacities(self, run_id,
#                                  list_vars=list_vars,
#                                  list_constr_deact=list_constr_deact):
#        '''
#        Runs the model using fixed capacities (or other variables).
#        This is to obtain starting conditions which are closer to the
#        expected solution. Presolve variable values are taken from
#        the schema self.sc_warmstart
#        Keyword arguments:
#        run_id -- which run_id to use from the sc_warmstart schema
#        list_vars -- list of tuples [(table, attribute_name)] defining the
#                     variables to be fixed
#        list_constr_deact -- constraints to be deactivated to avoid
#                             infeasibilitiess
#        '''
#        if self.sc_warmstart: # check presolve schema name is set to value
#
#            # Set variable values from init model run and run
#            # without investments to get the electricity production right
#            filt = [('run_id', [run_id])]
#
#
#
#            # load data from sc_warmstart after filtering by run_id and save in
#            # dictionary of dataframes
#            ws_var_0 = dict()
#            flag_not_empty = True
#            for ivar in list_vars:
#                ws_var_0[ivar[1]] = aql.read_sql(ModelBase.db,
#                                                 self.sc_warmstart,
#                                                 ivar[0], filt=filt)
#                if not ws_var_0[ivar[1]]:
#                    flag_not_empty = False
#
#            if flag_not_empty: # run_id exists for all input variables
#
#                # save current state of constraint activation
#                dict_constr_active = {const: getattr(self, const).active
#                                      for const in list_constr_deact}
#                # deactivate constraints
#                for const in list_constr_deact:
#                    getattr(self, const).deactivate()
#
#                # get some peaker plants to avoid infeasibilities due to
#                # fixed capacities
#                self.fill_peaker_plants(from_variable=True)
#
#                # transform dict of dataframes to dictionary of dictionaries
#                ws_dict = {}
#                for itb in ws_var_0:
#                    idx = [c for c in ws_var_0[itb].columns
#                           if not c in ['value', 'bool_chg', 'run_id']]
#                    ws_dict[itb] = ws_var_0[itb].set_index(idx)['value']
#                    ws_dict[itb] = ws_dict[itb].to_dict()
#
#                # set self self model variables and fix them
#                for itb in ws_dict:
#                    getattr(self, itb).set_values(ws_dict[itb])
#                    getattr(self, itb).fix()
#
#                # check whether they are fixed as they should be
#                for itb in ws_var_0_dict:
#                    self.print_is_fixed(itb)
#
#
#                tstr = '** Presolve using run_id={run_id} from {sc} **'
#                tstr = tstr.format(run_id=run_id, sc=self.sc_warmstart)
#                print('*'*len(tstr) + '\n' + tstr + '\n' + '*'*len(tstr))
#                print('Using solutionfile: ', self.solutionfile)
#                self.run(warmstart=True)
#
#                # Get rid of peaker plants
#                self.fill_peaker_plants(reset_to_zero=True)
#
#                # unfix all variables
#                for itb in ws_dict:
#                    getattr(self, itb).unfix()
#
#                # reset constraint activation
#                for const in list_constr_deact:
#                    if dict_constr_active[const]:
#                        getattr(self, const).activate()
#                    else:
#                        getattr(self, const).deactivate()
#            else:
#                print('''
#                      Warning in presolve_fixed_capacities: run_id={run_id}
#                      in schema {sc} doesn't seem to be available. Skipping.
#                      '''.format(run_id=run_id, sc=self.sc_warmstart))

    def fix_scenario_plants(self):
        '''
        Make sure exogenously defined capacities cannot be optimized.
        '''
        for tc in self.setlst['scen']:
            self.cap_pwr_new[(tc, 0)].fix()
            self.cap_pwr_rem[(tc, 0)].fix()


    def do_zero_run(self):
        '''
        Perform a single model run without capacity retirements/additions
        '''

        # set price of co2 to 5
        for ipp in self.price_co2:
            self.price_co2[ipp] = 5

        # no retirements or investments of capacity
        self.capchnge_max = 0

        tstr = '** First run without investments and retirements **'
        print('*' * len(tstr) + '\n' + tstr + '\n' + '*' * len(tstr))

        self.run(warmstart=False)

        # retirements or investments parameter back to inf
        self.capchnge_max = float('Inf')

    def activation(self, bool_act=False, constraint_list=False,
                   subset=False, verbose=False):
        ''' Changes activation of a list of constraints to bool_act '''

        if subset:
            if type(subset) is not dict:
                _subset = {c: subset for c in constraint_list}
            else:
                _subset = subset
        else:
            _subset = {c: [ii for ii in getattr(self, c)]
                           for c in constraint_list}

        for iconst in constraint_list:
            obj_constr = getattr(self, iconst)

            for ii in _subset[iconst]:
                if bool_act:
                    getattr(self, iconst)[ii].activate()
                else:
                    getattr(self, iconst)[ii].deactivate()
#                if verbose:
#                    print(iconst + ' ' + str(ii) + ' active? -> ' + str(getattr(self, iconst)[ii].active))

            if verbose:
                if type(verbose) == bool:
                    verbose = len(obj_constr)
                print(obj_constr)
                for ikk, kk in enumerate(obj_constr):
                    if ikk <= verbose :
                        print('{}: {}; is active: {}'.format(kk, obj_constr[kk],
                                                            obj_constr[kk].active))
                print('...\n' if verbose < len(obj_constr) else 'end\n')


    def set_variable_const(self, value=0, variable_list=False, verbose=False):

        for varname in variable_list:
            obj_var = getattr(self, varname)
            keys_var = [c for c in obj_var]
            dict_new = {kk: value for kk in keys_var}
            obj_var.set_values(dict_new)

            if verbose:
                if type(verbose) == bool:
                    verbose = len(obj_var)
                print(varname)
                for ikk, kk in enumerate(obj_var):
                    if ikk <= verbose:
                        print('{}: {}; is fixed: {}'.format(kk, obj_var[kk].value,
                                                            obj_var[kk].fixed))
                print('...\n' if verbose < len(obj_var) else 'end\n')

    def set_variable_fixed(self, bool_fix=True, variable_list=False,
                           subset=False, exclude=False, verbose=False):


        for varname in variable_list:
            obj_var = getattr(self, varname)


            _exclude = exclude if exclude else []
            _subset = [c for c in obj_var if not c in _exclude] if not subset else subset



            for ii in _subset:
                if bool_fix:
                    obj_var[ii].fix()
                else:
                    obj_var[ii].unfix()

            if verbose:
                if type(verbose) == bool:
                    verbose = len(obj_var)
                print(varname)
                for ikk, kk in enumerate(obj_var):
                    if ikk <= verbose and kk in _subset:
                        print('{}: {}; is fixed: {}'.format(kk, obj_var[kk].value,
                                                            obj_var[kk].fixed))
                print('...\n' if verbose < len(obj_var) else 'end\n')








