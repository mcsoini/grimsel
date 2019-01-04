'''
Definition of parameters.
'''
import pyomo.environ as po
import pyomo.core.base.sets as poset
import itertools
import pandas as pd


from grimsel.auxiliary.aux_m_func import pdef
import grimsel.core.io as io

class Parameters:
    '''
    Mixin class containing all parameters.
    Methods:
    define_parameters -- contains all parameter defintions relevant
                         for the model
    padd -- helper method for compact parameter definitions
    '''

    # holds the names of all parameters which got modified by monthly factors
    parameter_month_list = []

    def define_parameters(self):
#
        mut = {'mutable': True}
        inf = {'default': float('inf')}

        print('Profile parameters:')
        self.padd('dmnd', (self.sy, self.nd, self.ca), 'df_profdmnd_soy', 'value', **mut, default=0) # Complete information on demand
        self.padd('chpprof', (self.sy, self.nd, self.ca), 'df_profchp_soy', 'value', **mut) # Relative heat demand profile.
        self.padd('supprof', (self.sy, self.pr, self.ca), 'df_profsupply_soy', 'value', **mut) # Supply from variable generators.
        self.padd('inflowprof', (self.sy, self.hyrs | self.ror, self.ca), 'df_profinflow_soy', 'value', **mut) # Hydro inflow profiles.
        self.padd('priceprof', (self.sy, self.nd, self.fl_prof), 'df_profprice_soy', 'value', **mut) # Relative heat demand profile.

        print('Hydro parameters')
        self.padd('min_erg_mt_out_share', (self.hyrs,), 'df_hydro') # minimum monthly production as share of max_erg_mt_in_share.
        self.padd('max_erg_mt_in_share', (self.hyrs,), 'df_hydro') # maximum monthly inflow as share of yearly total.
        self.padd('min_erg_share', (self.hyrs,), 'df_hydro', **mut) # minimum filling level as share of cap_erg.

        print('Defining general parameters')
        self.padd('weight', (self.sy,), self.df_tm_soy) # Weight per time slot.
        self.padd('month_weight', (self.mt,), self.df_def_month) # Hours per month.
        self.padd('dmnd_sum', (self.nd,), self.df_node_encar, default=0) # .
        self.padd('grid_losses', (self.nd, self.ca), self.df_node_encar, **mut) # Grid losses.
        self.padd('vc_dmnd_flex', (self.nd, self.ca), self.df_node_encar) # VC of flexible demand.

        self.padd('chp_cap_pwr_leg', (self.nd,), self.df_def_node, **mut) # .
        self.padd('cap_trme_leg', (self.mt, self.ndcnn,), 'df_node_connect', **mut) # Cross-node transmission capacity.
        self.padd('cap_trmi_leg', (self.mt, self.ndcnn,), 'df_node_connect', **mut) # Cross-node transmission capacity.

        print('Defining ramping parameters')
        _df = self.df_plant_encar.copy()
        _df = _df.loc[_df['pp_id'].isin(self.setlst['pp'] + self.setlst['hyrs'] + self.setlst['ror'])]
        self.padd('vc_ramp', (self.pp | self.hyrs | self.ror, self.ca), _df, **mut) # .

        print('Defining pp parameters')
        _df = self.df_plant_encar.copy()
        _df = _df.loc[_df['pp_id'].isin(self.setlst['pp'])]
        self.padd('ca_share_min', (self.pp, self.ca), _df) # .
        self.padd('ca_share_max', (self.pp, self.ca), _df) # .
        self.padd('pp_eff', (self.ppall, self.ca), _df, default=1) # .
        self.padd('cf_max', (self.pp, self.ca), _df, **mut) # .

        print('Defining fuel parameters')
        self.padd('erg_inp', (self.nd, self.ca, self.fl), 'df_fuel_node_encar', **mut)
        self.padd('erg_chp', (self.nd, self.ca, self.fl), 'df_fuel_node_encar', **mut)
        self.padd('vc_fl', (self.fl, self.nd), 'df_fuel_node_encar', default=0, **mut) # EUR/MWh_el


        _df = self.df_plant_encar.copy()
        _df = _df.loc[_df.pp_id.isin(self.setlst['lin'])]
        self.padd('factor_vc_fl_lin_0', (self.lin, self.ca), _df, default=0, **mut) # EUR/MWh_el
        self.padd('factor_vc_fl_lin_1', (self.lin, self.ca), _df, default=0, **mut) # EUR/MWh_el
        self.padd('factor_vc_co2_lin_0', (self.lin, self.ca), _df, default=0, **mut) # EUR/MWh_el
        self.padd('factor_vc_co2_lin_1', (self.lin, self.ca), _df, default=0, **mut) # EUR/MWh_el
        self.padd('price_co2', (self.nd,), 'df_def_node', **mut) # EUR/MWh_el

        print('Defining parameters for all generators')
        _df = self.df_plant_encar.copy()
        _df = _df.loc[_df['pp_id'].isin(self.setlst['ppall'])]
        sets = (self.pp | self.pr | self.ror | self.st | self.hyrs | self.curt | self.lin, self.ca)
        self.padd('cap_pwr_leg', sets, _df, **mut) # .
        self.padd('vc_om', sets, _df, **mut) # .
        self.padd('fc_om', sets, _df, **mut, default=0) # .
        self.padd('fc_dc', sets, _df, **mut) # .

        _df['cap_avlb'] = 1 # init 1, adjusted from monthly correction factors
        _df = _df.loc[_df['pp_id'].isin(self.setlst['pp'])]
        self.padd('cap_avlb', (self.pp, self.ca), _df, **mut) # .

        print('Defining parameters investment')
        _df = self.df_plant_encar.copy()
        _df = _df.loc[_df['pp_id'].isin(self.setlst['add'])]
        self.padd('fc_cp_ann', (self.add, self.ca), _df, **mut) # .

        print('Defining st + hybrid parameters')
        _df = self.df_plant_encar.copy()
        _df = _df.loc[_df['pp_id'].isin(self.setlst['st'])]
        self.padd('st_lss_hr', (self.st, self.ca), _df, **mut)
        self.padd('st_lss_rt', (self.st, self.ca), _df, **mut)

        print('Defining hy parameters')
        self.padd('hyd_erg_bc', (self.sy_hydbc, self.hyrs), self.df_plant_month)

        print('Defining st + hyrs parameters')
        _df = self.df_plant_encar
        _df = (_df.loc[_df['pp_id'].isin(self.setlst['sthyrs'])])
        self.padd('discharge_duration', (self.st | self.hyrs, self.ca), _df, **mut)

        print('Defining parameter for investment and retirement control')
        self.capchnge_max = po.Param(initialize=float('inf'), **mut)

        # applying monthly adjustment factors to some parameters as defined
        # in the self.df_parameter_month input table
        if not self.df_parameter_month is None:
            self.apply_monthly_factors_all()

    def padd(self, parameter_name, parameter_index, source_dataframe=False,
             value_col=False, filt_col=False, filt_vals=[], mutable=False,
             default=None):

        '''
        Parameter definition based on input dataframes

        Keyword arguments:
        parameter_name -- used as model attribute, also assumed to be the
                          column name in case value_col=False
        parameter_index -- tuple of pyomo sets to define the parameter index
        source_dataframe -- input dataframe containing the parameter values
        value_col -- column name in the source_dataframe, is set to the
                     parameter name if no value is provided
        filt_col -- column of source_dataframe for filtering
        filt_vals -- values of filt_col for filtering
        mutable -- like the pyomo parameter keyword argument
        default -- like the pyomo parameter keyword argument
        '''

        print('Assigning parameter {par}'.format(par=parameter_name), end='... ')


        flag_empty = False

        _df = pd.DataFrame()

        if type(source_dataframe) is str:

            if source_dataframe in self.__dict__.keys():
                _df = getattr(self, source_dataframe)
                if _df is None:
                    print('failed (source_dataframe is None).')
                    _df = pd.DataFrame()
                    flag_empty = True
#                    return None
            else:
                print('failed (source_dataframe does not exist).')
                _df = pd.DataFrame()
                flag_empty = True
#                return None

        elif type(source_dataframe) is pd.DataFrame:
            _df = source_dataframe


        # transform into tuple in case a single set is provided
        parameter_index = (tuple(parameter_index)
                           if not type(parameter_index) == tuple
                           else parameter_index)


        if not flag_empty and not self.check_valid_indices(parameter_index):
            flag_empty = True
#            return None


        # set data column to parameter name in case no other value is provided
        if not value_col:
            value_col = parameter_name

        # check if column exists in table
        if not flag_empty and value_col not in _df.columns:
            print('failed (column doesn\'t exist).')
            flag_empty = True
#            return None

        # dictionary sets -> column names
        dict_ind = {'sy': 'sy', 'sy_hydbc': 'sy', 'nd': 'nd_id', 'ca': 'ca_id',
                    'pr': 'pp_id',
                    'ror': 'pp_id', 'pp': 'pp_id', 'add': 'pp_id',
                    'fl': 'fl_id', 'fl_prof': 'fl_id',
                    'ppall': 'pp_id', 'hyrs': 'pp_id', 'wk': 'wk_id',
                    'mt': 'mt_id',
                    'ndcnn': ['nd_id', 'nd_2_id', 'ca_id'],
                    'st': 'pp_id', 'lin': 'pp_id'}

        # apply filter to dataframe
        if filt_col:
            _df = _df.loc[_df[filt_col].isin(filt_vals)]

        # get list of columns from sets
        index_cols = [dict_ind[pi.getname()]
                      if type(pi) in [poset.SimpleSet, poset.OrderedSimpleSet]
                      else 'pp_id' # Exception: Set unions are always pp
                      for pi in parameter_index]
        index_cols = [[c] if not type(c) == list else c for c in index_cols]
        index_cols = list(itertools.chain.from_iterable(index_cols))

        # set parameter
        param_kwargs = dict(mutable=mutable,
                            default=default)
        if not flag_empty:
            param_kwargs['initialize'] = pdef(_df, index_cols, value_col)
            print('ok.')

        setattr(self, parameter_name, po.Param(*parameter_index,
                                               **param_kwargs))

    def apply_monthly_factors_all(self):

        # init dictionary containing the dataframes with the reshaped
        # monthly factor dataframes
        self.dict_monthly_factors = {}
        self.parameter_month_dict = {}
        print('*' * 60)

        for param in self.df_parameter_month.parameter.unique():
            self.apply_monthly_factors(param)

        print('*' * 60)


    def apply_monthly_factors(self, param):
        '''
        '''

        print('Applying monthly factors to parameter %s'%param, end=' ... ')

        list_mts = self.df_parameter_month.mt_id.unique().tolist()

        dff = self.df_parameter_month.loc[self.df_parameter_month.parameter
                                          == param].copy()

        # get list of corresponding sets from the IO list
        try:
            sets_io = io.DICT_IDX[param]
        except:
            raise ValueError(('ModelBase.apply_monthly_factors: '
                              + 'Parameter {} '
                              + 'not included in the IO '
                              + 'parameter list.').format(param))

        sets = tuple([st for st in sets_io if not st == 'mt_id'])
        sets_new = ('mt_id',) + sets

        # get data from component
        df0 = io.IO.param_to_df(getattr(self, param), sets)

        # expand old index to months
        new_index = list(itertools.product(list_mts,
                                           df0[list(sets)].apply(tuple, axis=1)
                                                          .tolist()))
        new_index = [(cc[0],) + cc[1] for cc in new_index]

        # initialize new dataframe
        df1 = pd.DataFrame(new_index, columns=sets_new)

        # join original data
        df1 = df1.join(df0.set_index(list(sets)), on=sets)

        # get set name columns
        name_cols = [c for c in dff.columns if '_name' in c]

        # check consistency set name for this parameter
        if len(dff[name_cols]
                    .drop_duplicates()) > 1:
            raise ValueError('apply_monthly_factors: Detected inconsistent '
                             + 'sets for parameter {}. '.format(param)
                             + 'Each parameter must have one set '
                             + 'group only.')

        # rename set_id columns using the set_name values
        set_dict = dff[name_cols].iloc[0].T.to_dict()
        set_dict = {kk.replace('name', 'id'): vv
                    for kk, vv in set_dict.items()}
        dff = dff.rename(columns=set_dict)

        self.dict_monthly_factors.update({param: dff.copy()})

        val_col = 'mt_fact'

        # join monthly factors
        df1 = df1.join(dff.set_index(list(sets_new))[val_col], on=sets_new)
        df1[val_col] = df1[val_col].fillna(1)

        # apply monthly factor
        df1['value'] *= df1.mt_fact

        # delete previous parameter object
        self.delete_component(param)

        # get set objects
        param_index = tuple([getattr(self, ss.replace('_id', ''))
                             for ss in sets_new])

        # save final tables in dict
        self.parameter_month_dict[param] = df1

        # add new parameter component
        self.padd(param, param_index, df1, value_col='value', default=1,
                  mutable=True)

        # store this in the class attribute so we can make case
        # distinctions in the constraints
        self.parameter_month_list.append(param)

        # modify IO class attribute to get the output table indices right
        io.DICT_IDX[param] = sets_new

        print('done.')



##############################################################################
##### TODO: GENERALIZE THIS TO METHOD "reset_parameter"

    def set_erg_max_runs(self):
        '''
        Reset erg_max parameters, using the corresponding column from the
        fuel_encar tables.
        '''
        # reset erg_max_runs
        df = self.df_fuel_encar
        erg_max_new = pdef(df, ['nd_id', 'ca_id', 'fl_id'], 'erg_max_runs')
        for indcafl, erg_max_val in erg_max_new.items():
            print(indcafl, erg_max_val)
            self.erg_max[indcafl] = erg_max_val

    def set_cap_pwr_leg(self, slct_pp_id=None):
        '''
        Reset cap_pwr_leg parameter, using the corresponding column from the
        plant_encar table.
        '''
        if slct_pp_id is None:
            slct_pp_id = self.setlst['ppall']

        _df = self.df_plant_encar.copy()
        mask_pp = _df['pp_id'].isin(slct_pp_id)
        _df = _df.loc[mask_pp].set_index(['pp_id', 'ca_id'])['cap_pwr_leg']
        cap_dict = _df.to_dict()

        for ppca, val in cap_dict.items():
            self.cap_pwr_leg[ppca].value = val

    def set_cf_max_runs(self):
        df = self.df_plant_encar
        df = df.loc[df['pp_id'].isin(self.setlst['pp'])]
        cf_max_new = pdef(df, ['pp_id', 'ca_id'], 'cf_max_runs')
        for ippca, cf_max_val in cf_max_new.items():
            print(ippca, cf_max_val)
            self.cf_max[ippca] = cf_max_val

