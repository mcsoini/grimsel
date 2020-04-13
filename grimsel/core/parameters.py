'''
Model parameters
=================


'''


import pyomo.environ as po
import pyomo.core.base.sets as poset
from pyomo.core.base.set_types import Reals
import itertools
import numpy as np
import pandas as pd
import wrapt
from collections import namedtuple

from grimsel.auxiliary.aux_m_func import set_to_list
import grimsel.core.io as io
from grimsel import _get_logger

logger = _get_logger(__name__)


# single parameter specification
_par_fields = ('parameter_name', 'parameter_index', 'source_dataframe',
               'value_col', 'filt_cols', 'filt_vals', 'mutable', 'default',
               'index_cols', 'domain')
_par_defaults = (None,) * 6 + (True, 0, None, Reals)
Par = namedtuple('Par', _par_fields)
Par.__new__.__defaults__ = _par_defaults


class ParameterAdder:
    '''
    Takes care of initializing, setting, and resetting a single parameter.

    Parameters
    ----------
    parameter_name : str
        used as model attribute, also assumed to be the
        column name in case ``value_col==False``
    parameter_index : tuple
        tuple of pyomo sets as parameter index
    source_dataframe : str or pandas.DataFrame
        input dataframe (or its attribute name)
        containing the parameter values
    value_col : str
        name of the ``source_dataframe`` parameter value column;
        optional---is set to the parameter name if no value is provided
    mutable : bool
        passed to the ``pyomo.environ.Param`` initialization
    default : numeric
        parameter default value; passed to the ``pyomo.environ.Param``
        initialization

    '''


    def __init__(self, m, par):


        self.m = m
        self.parameter_name = par.parameter_name
        self.source_dataframe = par.source_dataframe
        self.filt_cols = par.filt_cols
        self.filt_vals = par.filt_vals
        self.default = par.default
        self.param_kwargs = {'mutable': par.mutable,
                             'default': par.default,
                             'domain': par.domain}

        log_str = ('Assigning parameter '
                   '{par} ...'.format(par=self.parameter_name))
        logger.info(log_str)

        self.parameter_index = ((par.parameter_index,)
                           if not isinstance(par.parameter_index, tuple)
                           else par.parameter_index)

        self.value_col = (par.value_col
                          if par.value_col else self.parameter_name)

        self.has_monthly_factors = (
                self.m.df_parameter_month is not None and
                self.parameter_name
                in self.m.df_parameter_month.parameter.unique())

        self.index_cols = (par.index_cols
                           if par.index_cols else self._get_index_cols())

        self.df, self.flag_infeasible = self._get_param_data()

        if (not self.flag_infeasible
            and not self.m.check_valid_indices(self.parameter_index)):
            self.flag_infeasible = True

#        # check if column exists in table
#        if not self.flag_infeasible and self.value_col not in self.df.columns:
#            logger.warning(' failed (column doesn\'t exist).')
#            self.flag_infeasible = True


        if self.has_monthly_factors:
            df_mt, sets_new = self._apply_monthly_factors(self.df)

            self._df = df_mt
            self.index_cols = list(sets_new)

            # get set objects
            self.parameter_index = (self.m.mt, *self.parameter_index)

            # modify IO class attribute to get the output table indices right
            io.table_struct.DICT_COMP_IDX[self.parameter_name] = sets_new

    @property
    def df(self):
        '''
        ``df`` is the DataFrame holding the parameter data. It is implemented
        as a property so it can't be changed after initialization of the
        :class:`ParameterAdder` instance. This allows to restore the original
        values using :func:`ParameterAdder.init_update()`.

        '''
        return self._df

    @df.setter
    def df(self, val):

        if hasattr(self, 'df'):
            raise RuntimeError('Trying to set frozen attribute '
                               'ParameterAdder.df')
        else:
            self._df = val

    @wrapt.decorator
    def _if_is_feasible(f, self, args, kwargs):
        if not self.flag_infeasible:
            return f(*args, **kwargs)
        else:
            pass

    @_if_is_feasible
    def init_update(self, *args):
        '''

        Parameters
        ----------
        args
            The :func:`_get_data_dict` parameters ``(df, monthly_fact_col)``

        '''

        data = self._get_data_dict(*args)

        if not hasattr(self.m, self.parameter_name):
            # new parameter
            log_str = ' ok.'

            self.param_kwargs['initialize'] = data
            setattr(self.m, self.parameter_name,
                    po.Param(*self.parameter_index, **self.param_kwargs)
                    )
        else:
            # update parameter values
            log_str = ' parameter exists: updating.'
            for key, val in data.items():
                getattr(self.m, self.parameter_name)[key] = val

        logger.info(log_str)

    def _get_data_dict(self, df=False, monthly_fact_col=None):
        '''
        Returns a data dictionary for internal or external data.

        '''

        if isinstance(df, bool) and not df:
            # case no df input -> use internal data
            df = self.df
        else:
            # case external df cols don't have mt_id but index_cols do
            if (not 'mt_id' in df.columns) and ('mt_id' in self.index_cols):
                # apply monthly factors to external df
                df, _ = (self._apply_monthly_factors(df) if not monthly_fact_col
                         else self._apply_monthly_factors(df, monthly_fact_col))
            else:
                # use external df as is
                df = df

        return df.loc[-df[self.value_col].isna()].set_index(self.index_cols)[self.value_col].to_dict()

    def _get_param_data(self):
        '''
        Performs various checks

        Parameters
        ----------
        source_dataframe : pandas.Dataframe


        '''

        df = pd.DataFrame()
        flag_empty = False

        if type(self.source_dataframe) is str:

            if getattr(self.m, self.source_dataframe, None) is not None:
                df = getattr(self.m, self.source_dataframe).copy()
                if df is None:
                    logger.warning('... failed (source_dataframe is None).')
                    df = pd.DataFrame()
                    flag_empty = True
            else:
                logger.warning('... failed (source_dataframe does not exist).')
                df = pd.DataFrame()
                flag_empty = True

        elif type(self.source_dataframe) is pd.DataFrame:
            df = self.source_dataframe.copy()

        if (not flag_empty) and (self.filt_cols and self.filt_vals):
            try:
                df = df.set_index(self.filt_cols).loc[list(self.filt_vals)]
                df = df.reset_index()
            except Exception as e:
                logger.error(('Got error {e} when trying to filter DataFrame '
                              '{df} by values {val} of columns {cols}. This '
                              'might happen e.g. if a power plant in the pp '
                              'but not in the lin set is missing a value for '
                              'pp_eff in the plant_encar table'
                              ).format(e=e, val=self.filt_vals,
                                       cols=self.filt_cols, df=df))


        # check if column exists in table
        if not flag_empty:
            if self.value_col not in df.columns:
                if self.default is not None:
                    # make nan column which will be filled with default later
                    df[self.value_col] = np.nan
                    logger.warning(' column doesn\'t exist but default '
                                   'value provided...')
                else:
                    logger.warning(' failed (column doesn\'t exist).')
                    flag_empty = True

        if not flag_empty:
            df = df[self.index_cols + [self.value_col]]

            if self.default is not None:
                df.loc[df[self.value_col].isna(), self.value_col] = self.default

        return df, flag_empty

    def _get_index_cols(self):
        '''
        Translate the ``parameter_index`` set objects to column names.

        '''

        # dictionary sets -> column names
        dict_ind = {'sy': 'sy', 'sy_hydbc': 'sy', 'nd': 'nd_id', 'ca': 'ca_id',
                    'pr': 'pp_id', 'chp': 'pp_id',
                    'ror': 'pp_id', 'pp': 'pp_id', 'add': 'pp_id',
                    'fl': 'fl_id', 'fl_prof': 'fl_id',
                    'ppall': 'pp_id', 'hyrs': 'pp_id', 'wk': 'wk_id',
                    'mt': 'mt_id',
                    'ndcnn': ['nd_id', 'nd_2_id', 'ca_id'],
                    'st': 'pp_id', 'lin': 'pp_id',
                    'ndfl_prof': ['nd_id', 'fl_id'],
                    'ndcafl': ['nd_id', 'ca_id', 'fl_id'],
                    'sy_ndca': ['sy', 'nd_id', 'ca_id'],
                    'pricesll_pf': 'price_pf_id', 'pricebuy_pf': 'price_pf_id',
                    'dmnd_pf': 'dmnd_pf_id', 'tmsy': ['tm_id', 'sy'],
                    'supply_pf': 'supply_pf_id',
                    'sy_pr_ca': ['sy', 'pp_id', 'ca_id']}

        # get list of columns from sets
        index_cols = [dict_ind[pi.getname()]
                      if type(pi) in [poset.SimpleSet, poset.OrderedSimpleSet]
                      else 'pp_id' # Exception: Set unions are always pp
                      for pi in self.parameter_index]

        index_cols = [[c] if not type(c) == list else c for c in index_cols]
        index_cols = list(itertools.chain.from_iterable(index_cols))

        return index_cols


    def _expand_to_months(self, df0):
        '''
        Adds an additional month column to the input df.

        Parameters
        ----------
        df0 : pd.DataFrame
            Table to be expanded

        Returns
        -------
        pd.DataFrame
            table with additional month columnd

        '''

        list_mts = self.m.df_parameter_month.mt_id.unique().tolist()

        sets = [c for c in df0.columns if not c == 'value']
        old_index = df0[list(sets)].apply(tuple, axis=1).tolist()
        new_index = list(itertools.product(list_mts, old_index))
        new_index = [(cc[0],) + cc[1] for cc in new_index]

        df1 = pd.DataFrame(new_index)

        return df1



    def _get_monthly_factors(self):
        '''
        Returns monthly factors DataFrame for joining with the
        original parameter data.

        '''

        param = self.parameter_name

        param_mask = self.m.df_parameter_month.parameter == param
        dff = self.m.df_parameter_month.loc[param_mask].copy()
        name_cols = [c for c in dff.columns if '_name' in c]


        if len(dff[name_cols].drop_duplicates()) > 1:
            raise ValueError('apply_monthly_factors: Detected inconsistent '
                             + f'sets for parameter {param}. '
                             + 'Each parameter must have one set group only.')

        # rename set_id columns using the set_name values
        set_dict = dff[name_cols].iloc[0].T.to_dict()
        set_dict = {kk.replace('name', 'id'): vv
                    for kk, vv in set_dict.items()}
        dff = dff.rename(columns=set_dict)

        return dff


    def _apply_monthly_factors(self, df, val_col='mt_fact'):
        '''
        Adds a monthly index to existing parameters.

        For example, fuel prices are defined as yearly averages in the initial
        parameter definition (``df_plant_node_encar`` table). Here the
        parameter is re-initialized using the monthly factors from the
        ``ModelBase.df_parameter_month`` input table.

        Parameters
        ----------
        param : str
            name of the model parameter attribute the monthly factors are
            applied to

        Raises
        ------
        KeyError
            If the ``param`` is not included in the
            ``io.table_struct.DICT_COMP_IDX``
            table |rarr| index dictionary and therefore its index set
            definition cannot be inferred.
        ValueError
            If a parameter in the ``df_parameter_month`` is associated with
            more than one set group, e.g. ``vc_fl`` with both
            ``(nd_id, fl_id)`` and ``(fl_id,)``

        '''

        param = self.parameter_name

        logger.info('Applying monthly factors to parameter %s'%param)

        try:
            sets_io = io.table_struct.DICT_COMP_IDX[param]
        except:
            raise KeyError(('ModelBase._apply_monthly_factors: '
                              + 'Parameter {} '
                              + 'not included in the IO '
                              + 'parameter list.').format(param))

        sets = tuple([st for st in sets_io if not st == 'mt_id'])
        sets_new = ('mt_id',) + sets

        df0 = df
        df1 = self._expand_to_months(df0)
        df1.columns = sets_new + (self.value_col,)

        # join monthly factors
        df_fact = self._get_monthly_factors().set_index(list(sets_new))[val_col]

        df1 = df1.join(df_fact, on=sets_new)
        df1[val_col] = df1[val_col].fillna(1)

        # apply monthly factor
        df1[self.value_col
            ] *= df1[val_col]

        return df1[list(sets_new + (self.value_col,))], sets_new


# %%
class Parameters:
    r'''
    Mixin class for the :class:`grimsel.core.model_base.ModelBase` class
    containing all parameter definitions.

    '''


    def _get_df_supply(self):
        '''
        Returns ``df_profsupply_soy`` table with ``'pp_id', 'ca_id'`` columns
        if not empty, else empty table wih corresponding columns.
        '''

        if getattr(self, 'df_profsupply_soy', pd.DataFrame()).empty:
            return pd.DataFrame(columns=['sy', 'pp_id', 'ca_id', 'value'])
        else:
            return self.translate_pf_id(
                       self.df_profsupply_soy.rename(
                           columns={'supply_pf_id': 'pf_id'}))

    def _get_df_demand(self):
        '''
        Returns ``df_profdmnd_soy`` table with ``'nd_id', 'ca_id'`` columns
        if not empty, else empty table wih corresponding columns.
        '''

        if self.df_profdmnd_soy.empty:
            return pd.DataFrame(columns=['sy', 'nd_id', 'ca_id', 'value'])
        else:
            return self.translate_pf_id(
                       self.df_profdmnd_soy.rename(
                           columns={'dmnd_pf_id': 'pf_id'}))



    def add_parameters(self):
        '''
        Adds all parameters to the model.

        Generates :class:`ParameterAdder` instances for each of the parameters
        and calls their :meth:`ParameterAdder.init_update` method.

        '''

        if __name__ == '__main__':
            self = ml.m

        list_par = (
        Par('dmnd', self.sy_ndca, self._get_df_demand(), 'value'),
        Par('supprof', self.sy_pr_ca, self._get_df_supply(), 'value'),
        Par('chpprof', (self.sy_ndca),
            'df_profchp_soy', 'value'),
        Par('inflowprof', (self.sy_hyrs_ca | self.sy_ror_ca),
            'df_profinflow_soy', 'value',
            index_cols=['sy', 'pp_id', 'ca_id']),

        Par('pricebuyprof', (self.sy, self.pricebuy_pf),
            'df_profpricebuy_soy', 'value'),
        Par('pricesllprof', (self.sy, self.pricesll_pf),
            'df_profpricesll_soy', 'value'),

        Par('min_erg_mt_out_share', self.hyrs, 'df_hydro'),
        Par('max_erg_mt_in_share', self.hyrs, 'df_hydro', default=1),
        Par('min_erg_share', self.hyrs, 'df_hydro'),

        Par('weight', self.tmsy, 'df_tm_soy', default=1),
        Par('grid_losses', (self.nd, self.ca), 'df_node_encar'),

        Par('cap_trme_leg', (self.mt, self.ndcnn), 'df_node_connect'),
        Par('cap_trmi_leg', (self.mt, self.ndcnn), 'df_node_connect'),

        Par('vc_ramp', (self.ppall, self.ca), 'df_plant_encar', None,
            ['pp_id', 'ca_id'], set_to_list(self.rp_ca, [None, None])),

        Par('pp_eff', (self.pp - self.lin, self.ca), 'df_plant_encar', None,
            ['pp_id'], self.pp - self.lin, default=1),
        Par('cf_max', (self.pp, self.ca), 'df_plant_encar', None, ['pp_id'],
            self.pp),

        Par('erg_chp', (self.chp, self.ca), 'df_plant_encar', None, ['pp_id'],
            self.chp),

        Par('erg_inp', self.ndcafl,
            'df_fuel_node_encar'),
        Par('vc_fl', (self.fl, self.nd),
            'df_fuel_node_encar', default=0),

        Par('factor_lin_0', (self.lin, self.ca),
            'df_plant_encar', None, ['pp_id'], self.lin, default=0),
        Par('factor_lin_1', (self.lin, self.ca),
            'df_plant_encar', None, ['pp_id'], self.lin, default=0),

        Par('price_co2', self.nd, 'df_def_node'),
        Par('nd_weight', self.nd, 'df_def_node', default=1),

        Par('co2_int', self.fl, 'df_def_fuel', default=0),

        Par('cap_pwr_leg', (self.ppall_ca),
            'df_plant_encar', None, ['pp_id'], self.ppall,
            index_cols=['pp_id', 'ca_id']),
        Par('vc_om', (self.ppall, self.ca),
            'df_plant_encar', None, ['pp_id'], self.ppall),
        Par('fc_om', (self.add, self.ca),
            'df_plant_encar', None, ['pp_id'], self.add, default=0),
        Par('fc_cp_ann', (self.add, self.ca),
            'df_plant_encar', None, ['pp_id'], self.add),

        Par('pwr_pot', (self.add, self.ca),
            'df_plant_encar', None, ['pp_id'], self.add, default=1e9),
        Par('cap_avlb', (self.pp, self.ca),
            'df_plant_encar', None, ['pp_id'], self.pp, default=1),

        Par('st_lss_hr', (self.st, self.ca),
            'df_plant_encar', None, ['pp_id'], self.st),
        Par('st_lss_rt', (self.st, self.ca),
            'df_plant_encar', None, ['pp_id'], self.st),
        Par('discharge_duration', (self.st | self.hyrs, self.ca),
            'df_plant_encar', None, ['pp_id'], self.st | self.hyrs),

        Par('hyd_erg_bc', (self.sy_hydbc, self.hyrs), 'df_plant_month'),
        )


        self.dict_par = {}
        for par in list_par:
            parameter = ParameterAdder(self, par)
            self.dict_par[par.parameter_name] = parameter
            parameter.init_update()


    def reset_all_parameters(self):
        '''
        Reset all parameters to their original values.

        This can be used prior to the model parameter variations to reset
        all of the input data.

        '''

        for name, par in self.dict_par.items():

            logger.info('Resetting parameter {}'.format(name))

            par.init_update()



# %%

    @staticmethod
    def _make_model_parameters_doc():

        doc_dict = {
        'Model structure': {
                    'weight': r':math:`{w_\mathrm{\tau, t}} : \forall \mathrm{(\tau, t) \in tmsy}`: Time slot weight, indexed by time map :math:`\tau` and time slot :math:`t`.',
                    },
        'Profiles': {
                    'dmnd': r':math:`\Phi_\mathrm{dmd, t, n,c} : \forall \mathrm{(t,n,c) \in sy\_ndca}`: Demand profile',
                    'chpprof': r':math:`\Phi_\mathrm{chp, t, n,c} : \forall \mathrm{(t,n,c) \in sy\_ndca}`: CHP profile',
                    'supprof': r':math:`\Phi_\mathrm{supply, t, p,c} : \forall \mathrm{(t,p,c) \in sy\_pp\_ca}`: Supply (VRE) profile',
                    'inflowprof': r':math:`\Phi_\mathrm{inflow, t, p,c} : \forall \mathrm{(t,p,c) \in sy\_hyrs\_ca \cup sy\_ror\_ca}`: Reservoir and run-of-river inflow profile',
                    'pricebuyprof': r':math:`\Phi_\mathrm{pbuy, t, \phi} : \forall \mathrm{(t,\phi) \in sy\times pf}`: Energy carrier price profile (buying)',
                    'pricesllprof': r':math:`\Phi_\mathrm{psell, t, \phi} : \forall \mathrm{(t,\phi) \in sy\times pf}`: Energy carrier price profile (selling)',
                    },
        'Hydro parameters': {
                    'min_erg_mt_out_share': r':math:`\rho_\mathrm{min\_erg\_out, p} : \forall \mathrm{p \in hyrs}`: Minimum monthly reservoir production as share of maximum monthly inflow :math:`\rho_\mathrm{max\_erg\_in, hyrs}` :math:`\mathrm{(-)}`.',
                    'max_erg_mt_in_share': r':math:`\rho_\mathrm{max\_erg\_in, p} : \forall \mathrm{p \in hyrs}`: Maximum monthly reservoir inflow :math:`\mathrm{(-)}`.',
                    'min_erg_share': r':math:`\rho_\mathrm{min\_cap,p} : \forall \mathrm{p \in hyrs}`: Maximum monthly reservoir inflow :math:`\mathrm{(-)}`.',
                    'hyd_erg_bc': r':math:`\rho_\mathrm{hyd\_bc, t,p} : \forall \mathrm{(t,p) \in sy\_hydbc \times hyrs}`: Hydro filling level boundary conditions for specific hours as share of energy capacity :math:`\mathrm{(-)}`.',
                    },
        'Grid and transmission': {
                    'cap_trme_leg': r':math:`P_\mathrm{exp, m,n_1,n_2,c} : \forall \mathrm{(m,n_1,n_2,c) \in mt \times ndcnn}`: Internodal monthly export capacity.',
                    'cap_trmi_leg': r':math:`P_\mathrm{imp, m,n_1,n_2,c} : \forall \mathrm{(m,n_1,n_2,c) \in mt \times ndcnn}`: Internodal monthly import capacity.',
                    'grid_losses': r':math:`\mathrm{\eta_{grid, n,c}} : \forall \mathrm{(n,c) \in nd\_ca}`: Grid losses for each node and energy carrier.',
        },
        'Technical properties of assets': {
                    'cap_pwr_leg': r':math:`P_\mathrm{leg, p,c}: \forall \mathrm{(p,c) \in ppall\_ca}`: Legacy power plant capacity :math:`\mathrm{(MW)}`.',
                    'discharge_duration': r':math:`\zeta_\mathrm{p,c}: \forall\mathrm{(p,c) \in st\_ca \cup hyrs\_ca}`: Ratio of energy capacity and power capacity for energy storing assets :math:`\mathrm{(hours)}`.',
                    'pp_eff': r':math:`\eta_\mathrm{p,c}: \forall \mathrm{(p,c)\in pp\_ca\setminus lin\_ca}`: Power plant efficiency for constant supply curves :math:`\mathrm{(MWh_{c}/MWh_{fl})}`.',
                    'erg_chp': r':math:`e_\mathrm{chp,p,c} : \forall \mathrm{(p,c) \in chp\_ca}`: Heat-driven electricity generation from co-generation plants :math:`\mathrm{(MWh_{el}/yr)}`; used to scale the CHP profile :math:`\Phi_\mathrm{chp,t,n,c}`.',
                    'erg_inp': r':math:`e_\mathrm{inp,n,c,f} : \forall \mathrm{(n,c,f) \in ndcafl}`: exogenous energy production by fuel and node :math:`\mathrm{(MWh_{el}/yr)}`.',
                    'st_lss_hr': r':math:`\mathrm{\epsilon_{hr, p,c}} : \forall \mathrm{(p,c) \in st\_ca}`: Hourly storage leakage losses :math:`\mathrm{(1/hr)}`.',
                    'st_lss_rt': r':math:`\mathrm{\epsilon_{rt, p,c} }: \forall \mathrm{(p,c) \in st\_ca}`: Storage round-trip losses :math:`\mathrm{(-)}`.',
                    'cap_avlb': r':math:`\mathrm{\alpha_{mt, p,c}}: \forall \mathrm{(p,c) \in pp\_ca}`: Relative monthly capacity availability of dispatchable plants.',
                    'pwr_pot': r':math:`P_\mathrm{pot, p,c}: \add \mathrm{(p,c) \in add\_ca}`: Upper bound capacity added :math:`\mathrm{(MW)}`.',
                    },
        'Specific costs': {
                    'vc_ramp': r':math:`\mathrm{vc_{ramp, p,c}} : \forall \mathrm{(p,c) \in ppall\_ca}`: Specific variable ramping cost :math:`\mathrm{(EUR/MW)}`.',
                    'vc_fl': r':math:`\mathrm{vc_{f,n}} : \forall \mathrm{(f,n) \in fl\_nd}`: Specific fuel cost in each node :math:`\mathrm{(EUR/MWh_{fl})}`.',
                    'vc_om': r':math:`\mathrm{vc_{om, p,c}} : \forall \mathrm{(p,c) \in ppall\_ca}`: Specific variable O\&M costs :math:`\mathrm{(EUR/MWh_{el})}`.',
                    'fc_om': r':math:`\mathrm{fc_{om, p,c}} : \forall \mathrm{(p,c) \in ppadd\_ca}`: Specific fixed O\&M costs :math:`\mathrm{(EUR/MW/yr)}`.',
                    'fc_cp_ann': r':math:`\mathrm{fc_{cp, p,c}} : \forall \mathrm{(p,c) \in ppadd\_ca}`: Annualized specific capital investment costs :math:`\mathrm{(EUR/MW/yr)}`.',
                    },
        'Linear supply curve coefficients': {
                    'factor_lin_0': r':math:`f_\mathrm{0, p,c} : \forall \mathrm{(p,c) \in lin\_ca}`: Zero-order linear supply curve efficiency coefficient :math:`\mathrm{(MWh_{fl}/MWh_{el})}`.',
                    'factor_lin_1': r':math:`f_\mathrm{1, p,c} : \forall \mathrm{(p,c) \in lin\_ca}`: First-order linear supply curve efficiency coefficient :math:`\mathrm{(MWh_{fl}/MWh_{el}/MW_{el})}`.',
                    },
        'Emission parameters': {
                    'price_co2': r':math:`\pi_\mathrm{CO_2, n} : \forall \mathrm{n \in nd}`: Node-specific |CO2| price :math:`\mathrm{(EUR/t_{CO_2})}`.',
                    'co2_int': r':math:`i_\mathrm{CO_2, f} : \forall \mathrm{f \in fl}`: Fuel-specific |CO2| intensity :math:`\mathrm{(t_{CO_2}/MWh_{fl})}`.',
                    }}

        doc_dict_origin = {'cap_avlb': 'merge(df_plant_encar, df_parameter_month)',
         'cap_pwr_leg': 'df_plant_encar',
         'cap_trme_leg': 'df_node_connect',
         'cap_trmi_leg': 'df_node_connect',
         'chpprof': 'df_profchp_soy',
         'co2_int': 'df_def_fuel',
         'price_co2': 'merge(df_def_node, df_parameter_month)',
         'discharge_duration': 'df_plant_encar',
         'dmnd': 'df_profdmnd_soy',
         'erg_chp': 'df_plant_encar',
         'erg_inp': 'df_fuel_node_encar',
         'factor_lin_0': 'df_plant_encar',
         'factor_lin_1': 'df_plant_encar',
         'fc_cp_ann': 'df_plant_encar',
         'fc_om': 'df_plant_encar',
         'grid_losses': 'df_node_encar',
         'hyd_erg_bc': 'df_plant_month',
         'inflowprof': 'df_profinflow_soy',
         'max_erg_mt_in_share': 'df_hydro',
         'min_erg_mt_out_share': 'df_hydro',
         'min_erg_share': 'df_hydro',
         'pp_eff': 'df_plant_encar',
         'pricebuyprof': 'df_profpricebuy_soy',
         'pricesllprof': 'df_profpricesll_soy',
         'st_lss_hr': 'df_plant_encar',
         'st_lss_rt': 'df_plant_encar',
         'supprof': 'df_profsupply_soy',
         'vc_fl': 'merge(df_fuel_node_encar, df_parameter_month)',
         'vc_om': 'df_plant_encar',
         'vc_ramp': 'df_plant_encar',
         'weight': 'df_tm_soy',
         'pwr_pot': 'df_plant_encar',
         }

        df = pd.DataFrame(doc_dict).stack().reset_index()
        df.columns = ['par', 'cat', 'doc']


        df['tab'] = df.par.replace(doc_dict_origin).apply(lambda x: '``%s``'%x)

        df.par = df.par.apply(lambda x: '``%s``'%x)


        cols=['Parameter', 'Symbol', 'Domain', 'Doc', 'Table']
        df = pd.DataFrame(
                list(df.apply(lambda x: (x['cat'], x.par,) + tuple(x.doc.split(': ')) + (x.tab,), axis=1).values),
                columns=['cat'] + cols).set_index('cat')

        df['Symbol'] = df.Symbol.apply(lambda x: r'{}'.format(x).strip(' ')) + '`'
        df['Domain'] = ':math:`' + df.Domain.apply(lambda x: r'{}'.format(x))


        import tabulate

        doc_str = ''
        for cat in doc_dict.keys():
            table_str = tabulate.tabulate(df.loc[df.index.get_level_values('cat') == cat],
                                     tablefmt='rst', showindex=False,
                                     headers=cols)
            table_str = table_str.replace('\n', '\n    ')

            table_dir_str = '.. table:: **{}**\n\n    '.format(cat)
            table_str = table_dir_str + table_str

            doc_str += ('\n'*2)

            doc_str += table_str

        return doc_str


Parameters.__doc__ += '\n'*4 + Parameters._make_model_parameters_doc()


