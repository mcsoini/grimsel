#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:10:27 2017

@author: user
"""

import pandas as pd

import grimsel.auxiliary.sqlutils.aux_sql_func as aql
from grimsel.auxiliary.aux_m_func import pdef


class ParameterChanges():
    '''
    Mixin class for ModelLoop containing a library of possible
    model modifications
    '''

    def set_storage_flavors(self, dict_flavor=None):
        '''
        Provides different variations to the storage technology parameters
        to assess their respective impact.
        '''

        if dict_flavor is None:
            dict_flavor = {
                           0: 'default',
                           1: '0.6eff',  # CH_CAS has st_lss_rt of PHS + 0.5ppt
                           2: '0.7eff',   # CH_CAS has st_lss_rt of PHS - 0.5ppt
                           3: '0.75eff',   # CH_CAS has st_lss_rt of PHS - 0.5ppt
                           4: '0.8eff',  # CH_CAS has st_lss_rt of PHS + 0.5ppt
                           5: '0.9eff',   # CH_CAS has st_lss_rt of PHS - 0.5ppt
                           6: 'higheff',  # all storage has st_lss_rt = 0.1
                           7: 'loweff',   # all storage has st_lss_rt = 0.25
                           8: 'unltdpwr', # Storage is not constrained by power capacity
                           9: 'unltderg'  # Storage is not constrained by energy capacity
                          }

        slct_flavor = dict_flavor[self.dct_step['swfv']]

        # reset efficiency
        df = self.m.df_plant_encar.copy()
        df = df.loc[df.pp_id.isin(self.m.setlst['st']), ]
        dict_st_lss_rt = df.set_index(['pp_id', 'ca_id'])['st_lss_rt'].to_dict()
        for kk in self.m.st_lss_rt:
            self.m.st_lss_rt[kk].value = dict_st_lss_rt[kk]

        if slct_flavor == '0.745eff':
            for kk in self.m.st_lss_rt:
                if self.m.mps.dict_pp[kk[0]] in ['CH_CAS_STO', 'CH_LIO_STO']:
                    self.m.st_lss_rt[kk].value = 1 - 0.745
        if slct_flavor == '0.755eff':
            for kk in self.m.st_lss_rt:
                if self.m.mps.dict_pp[kk[0]] in ['CH_CAS_STO', 'CH_LIO_STO']:
                    self.m.st_lss_rt[kk].value = 1 - 0.755
        if slct_flavor == '0.6eff':
            for kk in self.m.st_lss_rt:
                if self.m.mps.dict_pp[kk[0]] in ['CH_CAS_STO', 'CH_LIO_STO']:
                    self.m.st_lss_rt[kk].value = 1 - 0.6
        if slct_flavor == '0.7eff':
            for kk in self.m.st_lss_rt:
                if self.m.mps.dict_pp[kk[0]] in ['CH_CAS_STO', 'CH_LIO_STO']:
                    self.m.st_lss_rt[kk].value = 1 - 0.7
        if slct_flavor == '0.75eff':
            for kk in self.m.st_lss_rt:
                if self.m.mps.dict_pp[kk[0]] in ['CH_CAS_STO', 'CH_LIO_STO']:
                    self.m.st_lss_rt[kk].value = 1 - 0.75
        if slct_flavor == '0.8eff':
            for kk in self.m.st_lss_rt:
                if self.m.mps.dict_pp[kk[0]] in ['CH_CAS_STO', 'CH_LIO_STO']:
                    self.m.st_lss_rt[kk].value = 1 - 0.8
        if slct_flavor == '0.9eff':
            for kk in self.m.st_lss_rt:
                if self.m.mps.dict_pp[kk[0]] in ['CH_CAS_STO', 'CH_LIO_STO']:
                    self.m.st_lss_rt[kk].value = 1 - 0.9
        if slct_flavor == 'higheff':
            for kk in self.m.st_lss_rt:
                self.m.st_lss_rt[kk].value = 0.1
        if slct_flavor == 'loweff':
            for kk in self.m.st_lss_rt:
                self.m.st_lss_rt[kk].value = 0.25
        if slct_flavor == 'noloss':
            for kk in self.m.st_lss_rt:
                self.m.st_lss_rt[kk].value = 0

        # activate
        self.m.PpStCapac.activate()
        self.m.CapacStPwCh.activate()
        self.m.CapacStEn.activate()

        if slct_flavor == 'unltdpwr':
            for kk in self.m.PpStCapac:
                if kk[0] in self.m.setlst['st']:
                    self.m.PpStCapac[kk].deactivate()
                    self.m.CapacStPwCh[kk].deactivate()

        if slct_flavor == 'unltderg':
            for kk in self.m.CapacStEn:
                if kk[0] in self.m.setlst['st']:
                    self.m.CapacStEn[kk].deactivate()

        self.dct_vl['swfv_vl'] = slct_flavor



    def select_connections_yesno(self, dict_cnn={0: False, 1: True}):
        '''
        '''

        slct_cnn = dict_cnn[self.dct_step['swcn']]

        # Reset parameter
        idx = ['mt_id', 'nd_id', 'nd_2_id', 'ca_id']
        dict_ndcnn = self.m.df_node_connect.set_index(idx)['cap_trm_leg']
        dict_ndcnn = dict_ndcnn.to_dict()

        if not slct_cnn:
            dict_ndcnn = {kk: 0 for kk in dict_ndcnn.keys()}

        for kk, vv in dict_ndcnn.items():
            self.m.cap_trm_leg[kk].value = vv


        dict_name = {True:  'Connect',
                     False: 'NoCnnct'}
        self.dct_vl['swcn_vl'] = dict_name[slct_cnn]

        # define peaker plant type and fill demand
        if not slct_cnn:
            pt_peak = [self.m.mps.dict_pt_id['OIL_ELC']]
            df = self.m.df_def_plant
            self.m.setlst['peak'] = df.loc[df.pt_id.isin(pt_peak)].pp_id.tolist()
            self.m.fill_peaker_plants(demand_factor=3)

    def select_storage_technology(self, dict_type, slct_nd_tec=None,
                                  keep_only=False):
        '''
        Doesn't make changes to self.model, but returns the list of pp_ids
        corresponding to the relevant storage tech.
        Arguments:
        dict_type -- dictionary which maps run indices to storage technologies
                     e.g. {0: 'CAS', 1: 'LIO'}
        '''

        if slct_nd_tec is None:
            # limitation to specific nodes
            slct_nd_tec = self.m.slct_node

        slct_type = dict_type[self.dct_step['swtc']]

        mask_tc = (self.m.df_def_plant['pp'].str.contains(slct_type)
                 & self.m.df_def_plant.nd_id.replace(self.m.mps.dict_nd).str.contains('|'.join(slct_nd_tec)))

        slct_tc = list(self.m.df_def_plant.loc[mask_tc, 'pp_id'])
        self.dct_vl['swtc_vl'] = slct_type

        if keep_only:
            # reset legacy capacity
            self.m.set_cap_pwr_leg()

            # set all storage capacity except slct_tc to zero
            for kk in self.m.cap_pwr_leg:

                if kk[0] in self.m.setlst['st'] and not kk[0] in slct_tc:
                    self.m.cap_pwr_leg[kk] = 0

        return slct_tc

    def set_storage_penetration(self, slct_tc, share_st_max):
        '''
        Arguments:
        share_st_max -- maximum storage penetration of storage
        '''

        share_st = self.dct_step['swst'] * share_st_max

        # get list of pp_id for relevant nodes
        df_tc = self.m.df_def_plant.loc[self.m.df_def_plant['nd_id'].isin(self.m.slct_node_id)
                                 & self.m.df_def_plant['pp_id'].isin(slct_tc),
                                 ['pp_id', 'nd_id']].set_index('nd_id')

        # get values for storage capacity
        df_def_cap_sto = self.m.df_def_node.set_index('nd_id')['dmnd_max'] * share_st

        # select relevant and map from country to plant
        df_def_cap_sto = (df_def_cap_sto.reset_index()
                                        .join(df_tc, on=df_tc.index.names)
                                        .dropna(axis=0, how='any')
                                        .drop('nd_id', axis=1))
        df_def_cap_sto['ca_id'] = 0

        # reset values of pyomo variable
        self.m.cap_pwr_new.set_values({i: 0 for i in self.m.cap_pwr_new
                                       if i[0] in self.m.setlst['scen']})
        # set values of fixed pyomo variable
        cap_sto_dict = (df_def_cap_sto
                        .set_index(['pp_id', 'ca_id'])['dmnd_max'].to_dict())
        self.m.cap_pwr_new.set_values(cap_sto_dict)

        self.dct_vl['swst_vl'] = str(int(share_st * 1000)/10) + '%'


    def set_variable_renewable_penetration(self, share_ws_max, slct_step=None,
                                           zero_is_status_quo=True,
                                           slct_nd_ws=None):
        '''
        Set total share of wind+solar power generation as a fraction of demand
        per node. This also actives the corresponding constraint.

        Arguments:
        share_ws_max -- maximum share of wind+solar for a parameter loop
        share_ws -- fixed value in case no parameter is used
        '''

        if slct_nd_ws is None:
            slct_nd_ws = self.m.slct_node
        slct_nd_id_ws = [self.m.mps.dict_nd_id[ind] for ind in slct_nd_ws]



        if slct_step is None:
            slct_step = self.dct_step['swvr']

        share_ws = slct_step * share_ws_max

        # reset share_ws_set
        for i in self.m.share_ws_set:
            self.m.share_ws_set[i].value = None

        # current share_ws
        df_erg_tt = aql.read_sql('storage1', self.m.sc_out, 'def_node')
        df_erg_tt = df_erg_tt[['nd_id', 'dmnd_sum']].set_index('nd_id')['dmnd_sum']
        df_erg_tt = df_erg_tt.rename('erg_tt')

        df_erg = aql.read_sql('storage1', self.m.sc_out, 'var_yr_erg_yr',
                              filt=[('run_id', [-1])] + [('ca_id', [0])])
        df_erg_ws = df_erg.loc[df_erg.pp_id.isin(self.m.setlst['winsol'])][['pp_id', 'value']]
        df_erg_ws['nd_id'] = df_erg_ws['pp_id'].replace(self.m.mps.dict_plant_2_node_id)
        df_erg_ws = df_erg_ws.set_index(['pp_id', 'nd_id'])
        df_erg_ws = df_erg_ws.sum(axis=0, level='nd_id')['value']
        df_erg_ws = df_erg_ws.rename('erg_ws')

        df_share_ws_0 = pd.concat([df_erg_ws, df_erg_tt], axis=1)
        df_share_ws_0['share_ws'] = (df_share_ws_0['erg_ws']
                                     / df_share_ws_0['erg_tt'])
        dict_share_ws_0 = df_share_ws_0['share_ws'].to_dict()

        for ict in self.m.slct_node_id:
            if ((slct_step == 0 and zero_is_status_quo)
                or not ict in slct_nd_id_ws):  # corresponds to status quo
                self.m.share_ws_set[ict].value = dict_share_ws_0[ict]
            else:  # same share of power production for all nodes
                self.m.share_ws_set[ict].value = share_ws

        # activate relevant constraint
        self.m.set_win_sol.activate()

        if 'swvr_vl' in self.dct_vl.keys():
            self.dct_vl['swvr_vl'] = str(round(share_ws * 1000)/10) + '%'


    def set_activation_cross_border_transmission(self, dict_act):

        # reset
        dict_cap = self.m.df_node_connect.set_index(['mt_id', 'nd_id', 'nd_2_id', 'ca_id'])['cap_trm_leg']
        for kk, vv in dict_cap.items():
            self.m.cap_trm_leg[kk] = vv

        if self.dct_step['swcn'] == 0:
            for kk, vv in dict_cap.items():
                self.m.cap_trm_leg[kk] = 0

        self.dct_vl['swcn_vl'] = dict_act[self.dct_step['swcn']]


    def set_only_wind_or_solar(self,
                               dict_ws = {2: 'WIN_ONS', 1: 'SOL_PHO',
                                          0: 'WIN_ONS|SOL_PHO|WIN_OFF'},
                               dict_erg_constr = {'WIN_ONS': False,
                                                  'SOL_PHO': False,
                                                  'WIN_ONS|SOL_PHO|WIN_OFF': True}):
        '''
        Note: This sets all energy limitations for wind and solar to inf.
        '''
        if __name__ == '__main__':
            dict_ws = {0: 'WIN_ONS', 1: 'SOL_PHO', 2: 'WIN_ONS|SOL_PHO|WIN_OFF'}
            dict_erg_constr = {'WIN_ONS': False, 'SOL_PHO': False,
                               'WIN_ONS|SOL_PHO|WIN_OFF': True}

        slct_ws = dict_ws[self.dct_step['swws']]

        # reset erg_max
        self.m.set_erg_max_runs()

        # reset fixed cap_pwr_tot
#        self.m.cap_pwr_tot.unfix()

        # lift all maximum energy constraints for the corresponding technology
        mask_pp_winsol = self.m.df_def_plant.pp.str.contains(slct_ws)
        ws_fl_id = self.m.df_def_plant.loc[mask_pp_winsol, 'fl_id'].unique()
        mask_pp_winsol_all = self.m.df_def_plant.pp_id.isin(self.m.setlst['winsol'])
        ws_fl_id_all = self.m.df_def_plant.loc[mask_pp_winsol_all, 'fl_id'].unique()
        ws_pp_id_all = self.m.df_def_plant.loc[mask_pp_winsol_all, 'pp_id'].unique()

        if not dict_erg_constr[slct_ws]:
            dict_dmnd = self.m.df_def_node.set_index('nd_id')['dmnd_sum'].to_dict()
            for ind in self.m.slct_node_id:
                for ifl_all in ws_fl_id_all:
                    self.m.erg_max[(ind, 0, ifl_all)] = 0
                for ifl in ws_fl_id:
                    self.m.erg_max[(ind, 0, ifl)] = 3 * dict_dmnd[ind]#float('inf')

            for ipp in ws_pp_id_all:
                self.m.cap_pwr_leg[(ipp, 0)] = 0

        self.dct_vl['swws_vl'] = {'WIN_ONS': 'Wind only',
                                  'WIN_OFF': 'Wind offshore only',
                                  'SOL_PHO': 'Solar only',
                                  'WIN_ONS|SOL_PHO|WIN_OFF': 'All'}[slct_ws]

    def set_model_simplifications(self,
                                  dict_sm = {0: ['full'],
                                             1: ['ramp'],
                                             2: ['ramp', 'chp'],
                                             3: ['ramp', 'chp', 'hyd'],
                                             4: ['ramp', 'chp', 'hyd', 'cf'],
                                             5: ['ramp', 'chp', 'hyd', 'cf', 'cap']}):

        slct_simp = dict_sm[self.dct_step['swsm']]

        # RESET EVERYTHING
        list_unfix = (['pwr_ramp_yr', 'vc_ramp_yr', 'pwr_ramp', 'pwr_ramp_abs']
                      + ['cap_pwr_new', 'cap_pwr_rem'])
        list_reset = (['calc_ramp_rate', 'ramp_rate_abs_pos',
                       'ramp_rate_abs_neg', 'calc_vc_ramp', 'yrrmp']
                      + ['chp_prof']
                      + ['hy_erg_min', 'hy_month_min',
                         'hy_reservoir_boundary_conditions']
                      + ['erg_store_level']
                      + ['ror_min_base_load', 'ror_weekly_energy']
                      + ['pp_cf', 'pp_max_fuel'])
        self.m.activation(bool_act=True, constraint_list=list_reset, verbose=3)
        self.m.set_variable_fixed(False, list_unfix, verbose=3)



        if 'ramp' in slct_simp: # NO RAMPING COST

            print('Removing ramping costs')
            # no ramping cost constraints
            list_deact = ['calc_ramp_rate', 'ramp_rate_abs_pos', 'ramp_rate_abs_neg',
                          'calc_vc_ramp', 'yrrmp']
            self.m.activation(bool_act=False, constraint_list=list_deact, verbose=3)

#            # ramping cost variables fixed at zero
#            list_zero = ['pwr_ramp_yr', 'vc_ramp_yr', 'pwr_ramp', 'pwr_ramp_abs']
#            self.m.set_variable_const(0, list_zero, verbose=3)
#            self.m.set_variable_fixed(True, list_zero, verbose=3)

        if 'chp' in slct_simp: # NO CHP PROFILE RULES

            print('Removing CHP profiles')
            list_deact = ['chp_prof']
            self.m.activation(bool_act=False, constraint_list=list_deact, verbose=3)

        if 'hyd' in slct_simp: # NO SPECIAL HYDRO AND ROR RULES

            print('Removing hydro and ror constraints')
            list_deact = ['hy_erg_min', 'hy_month_min', 'hy_reservoir_boundary_conditions']
            self.m.activation(bool_act=False, constraint_list=list_deact, verbose=3)

            list_deact = ['erg_store_level']
            subset = [ii for ii in self.m.erg_store_level
                      if self.m.mps.dict_pp[ii[0]].endswith('HYD_RES')]
            self.m.activation(bool_act=False, constraint_list=list_deact, subset=subset, verbose=False)

            list_deact = ['ror_min_base_load', 'ror_weekly_energy']
            self.m.activation(bool_act=False, constraint_list=list_deact, verbose=3)

        if 'cf' in slct_simp: # NO LIMITATION OF YEARLY ENERGY/YEARLY CAPACITY FACTOR

            print('Removing capacity and maximum energy constraints')
            list_deact = ['pp_cf', 'pp_max_fuel']
            self.m.activation(bool_act=False, constraint_list=list_deact, verbose=3)

        if 'cap' in slct_simp: # NO CAPACITY CHANGES

            # exclude wind and solar
            list_excl = [(pp, 0) for pp in self.m.df_def_plant.loc[self.m.df_def_plant.pp.str.contains('WIN|SOL'), 'pp_id']]

            print('No capacity change')
            list_zero = ['cap_pwr_new', 'cap_pwr_rem']
            self.m.set_variable_const(0, list_zero, verbose=True)
            self.m.set_variable_fixed(True, list_zero, exclude=list_excl, verbose=3)

        self.dct_vl['swsm_vl'] = '-'.join(slct_simp)


    def set_meteo_year(self):

        dict_year = {0: ('value', 'yr2016'),
                     1: ('year2015', 'yr2015'),
                     2: ('year2014', 'yr2014'),
                     3: ('year2013', 'yr2013'),
                     4: ('year2012', 'yr2012'),
                     5: ('year2011', 'yr2011'),
                     6: ('year2010', 'yr2010'),}
        iyear = self.dct_step['swyr']

        profsupply_dict = self.m.df_profsupply_soy.set_index(['sy', 'pp_id', 'ca_id'])[dict_year[iyear][0]].to_dict()

        for kk, vv in profsupply_dict.items():
            self.m.supprof[kk] = vv

        self.dct_vl['swyr_vl'] = dict_year[iyear][1]


    def set_elprice_year(self, sc, dict_year=None, dict_sta_mod=None):

        if __name__ == '__main__':
            sc = 'profiles_raw'
            dict_year=None
            dict_sta_mod=None

        if dict_year is None:
            dict_year = {0: 'yr2015',
                         1: 'yr2017',
                         2: 'yr2016',
                         3: 'yr2014',
                         4: 'yr2013',
                         5: 'yr2012',
                         6: 'yr2011',
                         7: 'yr2010',
                         8: 'yr2009',
                         9: 'yr2008',
                         10: 'yr2007'}
        if dict_sta_mod is None:
            dict_sta_mod = {0: 'stats', 1: 'model'}

        iyear = self.dct_step['swyr']
        ista_mod = self.dct_step['swmh']

        year = dict_year[iyear]
        sta_mod = dict_sta_mod[ista_mod]

        if sta_mod == 'stats':
            prof = aql.read_sql(self.io.db, sc, 'epex_price_volumes',
                                filt=[('year', [int(year.replace('yr', ''))]),
                                      ('quantity', ['price_eur_mwh']),
                                      ('nd_id', self.m.slct_node),
                                      ('hy', self.m.df_hoy_soy.hy.tolist())]
                                )
            prof['nd_id'] = prof['nd_id'].replace(self.m.mps.dict_nd_id)
            prof['sy'] = prof['hy']

        elif sta_mod == 'model':
            # do an appropriate sql query here
            prof = pd.DataFrame(aql.exec_sql('''
                                SELECT sy, nd_id, value FROM out_cal.dual_supply
                                NATURAL LEFT JOIN (SELECT run_id, swhy_vl FROM out_cal.def_loop) AS dflp
                                NATURAL LEFT JOIN (SELECT nd, nd_id FROM out_cal.def_node) AS dfnd
                                WHERE swhy_vl = '{}' AND nd IN ({})
                                '''.format(year, ','.join(['\'{}\''.format(c) for c in self.m.slct_node])), db=self.io.db), columns=['sy', 'nd_id', 'value'])

        prof['fl_id'] = 'electricity'
        prof['fl_id'] = prof['fl_id'].replace(self.m.mps.dict_fl_id)



        prof_dict = prof.set_index(['sy', 'nd_id', 'fl_id'])['value'].to_dict()

        for kk, vv in prof_dict.items():
            self.m.priceprof[kk] = vv

        self.dct_vl['swyr_vl'] = year
        self.dct_vl['swmh_vl'] = sta_mod


    def set_discount_rate(self):

        ############### SELECT DISCOUNT RATE ###############

        dict_dr = {0: 4, 1: 3, 2: 5}

        slct_dr_str = str(dict_dr[self.dct_step['swdr']])

        dct_fc_cp_ann = pdef(self.m.df_plant_encar, ['pp_id', 'ca_id'],
                             val='fc_cp_ann' + slct_dr_str)

        for ikk in self.m.fc_cp_ann:
            self.m.fc_cp_ann[ikk] = dct_fc_cp_ann[ikk]

        self.dct_vl['swdr_vl'] = slct_dr_str + '%'

    def set_ramping_cost(self):

        dict_rc = {0: 'vc_ramp_mean', 1: 'vc_ramp_low', 2: 'vc_ramp_high'}
        slct_rc = dict_rc[self.dct_step['swrc']]
        dct_cost_val = (self.m.df_plant_encar
                            .loc[self.m.df_plant_encar['pp_id'].isin(self.m.setlst['pp'])]
                            .set_index(['pp_id', 'ca_id'])[slct_rc].to_dict())
        for iky in dct_cost_val:
            self.m.vc_ramp[iky] = dct_cost_val[iky]
        self.dct_vl['swrc_vl'] = slct_rc.replace('vc_ramp_', '')


    def set_charging_only_wind_solar(self):

        dict_cd = {0: 'Chg all', 1: 'Chg ws'}
        slct_cd = dict_cd[self.dct_step['swcd']]

        # reset
        self.m.chg_only_var_ren.deactivate()

        if self.dct_step['swcd'] == 1:
            self.m.chg_only_var_ren.activate()

        self.dct_vl['swcd_vl'] = slct_cd

    def set_value_co2_price(self, dict_co2=None):

        if dict_co2 is None:
            dict_co2 = {0: 40, 1: 5, 2: 80}
        slct_co2 = dict_co2[self.dct_step['swco']]
        df = self.m.df_plant_encar.loc[self.m.df_plant_encar['pp_id'].isin(self.m.setlst['pp'])]
        dict_vc_co2 = pdef(df, ['pp_id', 'ca_id'], 'vc_co2')
        for ipp in dict_vc_co2:
            self.m.vc_co2[ipp] = dict_vc_co2[ipp] * slct_co2

        self.dct_vl['swco_vl'] = str(slct_co2) + 'EUR/t_CO2'


    def set_value_co2_price_allnodes(self, co2_price_max=140):

        slct_co2 = self.dct_step['swco'] * co2_price_max


        for ipp in self.m.price_co2:
            self.m.price_co2[ipp] =  slct_co2

        self.dct_vl['swco_vl'] = str(slct_co2) + 'EUR/t_CO2'


######################

