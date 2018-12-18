# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:23:06 2018

@author: martin-c-s
"""
import numpy as np
import pandas as pd

from grimsel.core.io import IO
from grimsel.auxiliary.aux_m_func import pdef
from grimsel.auxiliary.aux_m_func import cols2tuplelist
import grimsel.auxiliary.maps as maps

import pyomo.environ as po


class ModelLoopModifier():
    '''
    The purpose of this class is to modify the parameters of the BaseModel
    objects in dependence on the current run_id.

    This might include the modification of parameters like CO2 prices,
    capacities, but also more profound modifications like the
    redefinition of constraints.
    '''

    def __init__(self, ml):
        '''
        To initialize, the ModelBase object is made an instance attribute.
        '''

        self.ml = ml


    def set_ramping_cost(self):
        
        dict_rc = {0: 1,
                   1: 0.1,
                   2: 0.5,
                   3: 1.2,
                   4: 1.5,
                   5: 1.8,
                   6: 2,
                   7: 2.5,
                   }

        slct_rc = dict_rc[self.ml.dct_step['swrc']]
        
        # reset ramping costs
        dict_vc_ramp = self.ml.m.df_plant_encar.set_index(['pp_id', 'ca_id'])['vc_ramp'].to_dict()
        
        for key in self.ml.m.vc_ramp:
            
            self.ml.m.vc_ramp[key] = dict_vc_ramp[key] * slct_rc
        
        

        self.ml.dct_vl['swrc_vl'] = 'x%03.1f'%slct_rc



    def set_trm_cap_onoff(self):
        
        dict_trm = {0: 'on',
                    1: 'off'}

        slct_trm = dict_trm[self.ml.dct_step['swtr']]
        
        if slct_trm == 'on':
            # reset
            dict_trm = self.ml.m.df_node_connect.set_index(['mt_id', 'nd_id', 'nd_2_id',
                                                 'ca_id'])['cap_trm_leg'].to_dict()
        
            for key, val in dict_trm.items():
                self.ml.m.cap_trm_leg[key] = val
        elif slct_trm == 'off':
            
            for key in self.ml.m.cap_trm_leg:
                self.ml.m.cap_trm_leg[key] = 0
            
            
        self.ml.dct_vl['swtr_vl'] = slct_trm


    def set_co2_price(self):

        dict_co = {0: 40,
                   1: 5,
                   2: 80}

        slct_co = dict_co[self.ml.dct_step['swco']]
        str_co = '%2dEUR/tCO2'%slct_co

        for key in self.ml.m.price_co2:
            self.ml.m.price_co2[key].value = slct_co

        self.ml.dct_vl['swco_vl'] = str_co

    def set_winsol_year(self):

        dict_yr = {0: 2015,
                   1: 2014,
                   2: 2013,
                   3: 2012,
                   4: 2011,
                   5: 2010,
                   6: 2016}

        slct_yr = dict_yr[self.ml.dct_step['swyr']]
        str_yr = str(slct_yr)

        mask_winsol = self.ml.m.df_def_plant.pp.str.contains('WIN|SOL')
        list_pp_id = self.ml.m.df_def_plant.loc[mask_winsol, 'pp_id'].tolist()

        mask_winsol = self.ml.m.df_profsupply_soy.pp_id.isin(list_pp_id)
        df_prf = self.ml.m.df_profsupply_soy.loc[mask_winsol].copy()
        df_prf = df_prf.set_index(['sy', 'pp_id', 'ca_id'])
        df_prf = df_prf['value' if slct_yr == 2015
                        else 'value_yr%s'%slct_yr]
        dict_prf = (df_prf.to_dict())

        for key, val in dict_prf.items():
            self.ml.m.supprof[key] = val


        self.ml.dct_vl['swyr_vl'] = str_yr

    def select_storage_tech(self):

        dict_sto = {0: dict(name='CAS_STO', st_lss_rt=0.25, discharge_duration=20),
                   1: dict(name='LIO_STO', st_lss_rt=0.10, discharge_duration=4),
                   2: dict(name='REF_STO', st_lss_rt=0.00, discharge_duration=20)}

        dict_sto_slct = dict_sto[self.ml.dct_step['swtc']]

        slct_pp = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('NEW_STO'), 'pp_id'].tolist()

        import itertools

        for par, pp in itertools.product(['st_lss_rt', 'discharge_duration'], slct_pp):

            getattr(self.ml.m, par)[(pp, 0)].value = dict_sto_slct[par]


        self.ml.dct_vl['swtc_vl'] = dict_sto_slct['name']


    def set_storage_cap(self, dict_st):

        ''''''

#        share_st = max_share_storage * self.ml.dct_step['swst']

        share_st = dict_st[self.ml.dct_step['swst']]


        df_dmnd = IO.param_to_df(self.ml.m.dmnd, ('sy', 'nd_id', 'ca_id'))
        df_dmnd = df_dmnd.pivot_table(index='nd_id', values='value',
                                      aggfunc=np.mean)['value'].reset_index()

        pp_mask = self.ml.m.df_def_plant.pp.str.contains('NEW_STO')
        df_dmnd = df_dmnd.join(self.ml.m.df_def_plant.loc[pp_mask]
                                        .set_index('nd_id')['pp_id'],
                               on='nd_id')

        cap_st = df_dmnd.set_index('pp_id')['value'] * share_st

        for pp_id, val in cap_st.to_dict().items():
            self.ml.m.cap_pwr_leg[(pp_id, 0)].value = val

        self.ml.dct_vl['swst_vl'] = '%.2f'%(share_st * 100) + '%'


    def select_vre_pp_types(self):
        '''
        Returns list of pp
        '''

        dict_pt = {0: 'WIN_ONS|WIN_OFF|SOL_PHO',
                   1: 'SOL_PHO',
                   2: 'WIN_ONS|WIN_OFF'}

        slct_pt = dict_pt[self.ml.dct_step['swpt']]

        slct_pp = self.ml.m.df_def_plant.pp.loc[self.ml.m.df_def_plant.pp.str.contains(slct_pt)]

        self.ml.dct_vl['swpt_vl'] = slct_pt

        return slct_pp, slct_pt


    def scale_vre(self, slct_pt, dict_vre):
        '''
        Sets power capacity for selected plants
        '''

        slct_vre = dict_vre[self.ml.dct_step['swvr']]

        share_vre = slct_vre if not slct_vre == 'default' else 0

        df_vre = self.ml.m.df_def_node[['nd', 'nd_id', 'vre_share_wind',
                                        'wind_share_offshore']].copy()
        df_vre['share_vre'] = share_vre

        if 'WIN_ONS' in slct_pt and 'SOL_PHO' in slct_pt:
            df_vre['WIN_TOT'] = df_vre.share_vre * df_vre.vre_share_wind
            df_vre['SOL_PHO'] = df_vre['share_vre'] - df_vre['WIN_TOT']
        elif not 'SOL_PHO' in slct_pt:
            df_vre['WIN_TOT'] = df_vre.share_vre
            df_vre['SOL_PHO'] = 0
        elif not 'WIN_ONS' in slct_pt:
            df_vre['WIN_TOT'] = 0
            df_vre['SOL_PHO'] = df_vre.share_vre

        df_vre['WIN_ONS'] = df_vre.WIN_TOT * (1 - df_vre.wind_share_offshore)
        df_vre['WIN_OFF'] = df_vre.WIN_TOT * df_vre.wind_share_offshore

        df_dmnd = IO.param_to_df(self.ml.m.dmnd, ('sy', 'nd_id', 'ca_id'))
        df_dmnd = df_dmnd.join(self.ml.m.df_tm_soy.set_index('sy')['weight'], on='sy')
        df_dmnd['value'] *= df_dmnd.weight
        df_dmnd = df_dmnd.pivot_table(index='nd_id', values='value',
                                      aggfunc=sum)['value'].rename('dmnd_sum')

        df_vre = df_vre.join(df_dmnd, on=df_dmnd.index.names)

        list_pt = ['WIN_ONS', 'WIN_OFF', 'SOL_PHO']
        for pt in list_pt:
            df_vre[pt] *= df_vre.dmnd_sum

        df_vre = df_vre.set_index('nd')[[pt for pt in list_pt]].stack().reset_index()
        df_vre['pp'] = df_vre.nd.apply(lambda x: x.replace('0', '_')) + df_vre.level_1
        df_vre = df_vre.loc[df_vre.pp.isin(self.ml.m.mps.dict_pp.values())]
        df_vre['pp_id'] = df_vre.pp.replace(self.ml.m.mps.dict_pp_id)
        df_vre = df_vre.set_index(['pp_id'])[0].rename('vre_erg')

        df_sup = IO.param_to_df(self.ml.m.supprof, ('sy', 'pp_id', 'ca_id'))
        df_sup = df_sup.loc[df_sup.pp_id.isin(df_vre.index.get_level_values('pp_id'))]
        df_sup = df_sup.join(self.ml.m.df_tm_soy.set_index('sy')['weight'], on='sy')
        df_sup['value'] *= df_sup.weight
        df_sup = df_sup.pivot_table(index='pp_id', aggfunc=np.sum,
                                    values=['value'])['value'].rename('erg_to_cap')

        df_vre = df_vre.reset_index().join(df_sup, on=df_sup.index.names)

        df_vre['vre_cap'] = df_vre.vre_erg / df_vre.erg_to_cap



        mask_pp = self.ml.m.df_def_plant.pp_id.isin(df_vre.pp_id)

        df_vre = \
        df_vre.join(self.ml.m.df_def_plant.loc[mask_pp]
                             .join(self.ml.m.df_fuel_node_encar
                                       .set_index(['fl_id', 'nd_id'])['erg_inp'],
                                   on=['fl_id', 'nd_id'])
                             .set_index(['pp_id'])['erg_inp']
                             .rename('erg_default'), on=['pp_id'])

        df_vre['cap_default'] = df_vre.erg_default / df_vre.erg_to_cap


        # generate dict depending on whether default or not

        if slct_vre == 'default':
            col = 'cap_default'
        else:
            col = 'vre_cap'

        dict_cap = df_vre.set_index('pp_id')[col].to_dict()


        for ind, val in dict_cap.items():
            self.ml.m.cap_pwr_leg[(ind, 0)].value = val

        self.ml.dct_vl['swvr_vl'] = ('%.2f'%(slct_vre*100) + '%') if isinstance(slct_vre, float) else slct_vre

    def availability_cf_cap(self):
        '''
        Switch between availability constraints
            * capacity factor
            * capacity
        Note: requires monthly availability!
        '''

        dict_avcfcap = {0: 'cap', 1: 'cf'}

        slct_avcfcap = dict_avcfcap[self.ml.dct_step['swcfcap']]


        # reset
        self.ml.m.pp_cf.activate()
        self.ml.m.PpStCapac.activate()
        if hasattr(self.ml.m, 'ppst_cap_monthly'):
            self.ml.m.ppst_cap_monthly.deactivate()

        #define new constraint
        if not hasattr(self.ml.m, 'ppst_cap_monthly'):

            cf_max = IO.param_to_df(self.ml.m.cf_max, ('mt_id', 'pp_id', 'ca_id'))
            cap_scale = cf_max.set_index(['mt_id', 'pp_id', 'ca_id'])['value'].to_dict()

            def ppst_cap_monthly_rule(model, pp, ca, sy, mt):
                ''' Produced power less than capacity. '''

                scale = (cap_scale[(mt, pp, ca)]
                         if (mt, pp, ca) in cap_scale.keys()
                         else 1)

                return (self.ml.m.pwr[sy, pp, ca]
                        <= self.ml.m.cap_pwr_tot[pp, ca] * scale)
            self.ml.m.ppst_cap_monthly = po.Constraint((self.ml.m.pp_ca - self.ml.m.pr_ca) | self.ml.m.st_ca
                                             | self.ml.m.hyrs_ca,
                                             self.ml.m.sy_mt, rule=ppst_cap_monthly_rule)
            self.ml.m.ppst_cap_monthly.deactivate()


        if slct_avcfcap == 'cap':
            self.ml.m.PpStCapac.deactivate() # yearly capacity constraint
            self.ml.m.pp_cf.deactivate()            # monthly cf constraints
            self.ml.m.ppst_cap_monthly.activate()          # monthly capacity constraint
        elif slct_avcfcap == 'cf':
            self.ml.m.PpStCapac.activate() # yearly capacity constraint
            self.ml.m.pp_cf.activate()            # monthly cf constraints
            self.ml.m.ppst_cap_monthly.deactivate()          # monthly capacity constraint




        self.ml.dct_vl['swcfcap_vl'] = str(slct_avcfcap)




    def raise_demand(self, list_sy, slct_nd):
        '''
        Used for shadow price disaggregation.
        '''


        dict_syrs = {0: 'ref'}
        dict_syrs.update({kk + 1: val for kk, val in enumerate(list_sy)})

        slct_sy = dict_syrs[self.ml.dct_step['swsyrs']]

        slct_nd_id = self.ml.m.mps.dict_nd_id[slct_nd]

        dict_dmnd = self.ml.m.df_profdmnd_soy.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict()

        # reset
        for ind, val in dict_dmnd.items():
            self.ml.m.dmnd[ind].value = val

        if slct_sy != 'ref':

            self.ml.m.dmnd[(slct_sy, slct_nd_id, 0)].value += 1

        self.ml.dct_vl['swsyrs_vl'] = str(slct_sy)



    def chp_on_off(self, slct_nd):

        dict_chp = {0: 'chp_on', 1: 'chp_off'}

        slct_chp = self.ml.dct_step['swchp']
        str_chp = dict_chp[slct_chp]

        slct_nd_id = [self.ml.m.mps.dict_nd_id[nd] for nd in slct_nd]

        # reset
        for ind in self.ml.m.chp_prof:
            self.ml.m.chp_prof[ind].activate()

        if str_chp == 'chp_off':
            for ind in [ind for ind in self.ml.m.chp_prof_index if ind[1] in slct_nd_id]:
                self.ml.m.chp_prof[ind].deactivate()

        self.ml.dct_vl['swchp_vl'] = str_chp

    def cost_adjustment_literature(self):

        # based on BUBL2017

        dict_cadj = {0: 'orig', 1: 'adjs'}


        slct_cadj = self.ml.dct_step['swcadj']
        str_cadj = dict_cadj[slct_cadj]

        self.ml.m.vc_fl_lin_0.display()

        df_vc_fl = self.ml.m.df_plant_encar.set_index(['pp_id', 'ca_id'])[['vc_fl_lin_0', 'vc_fl_lin_1']]

        # reset
        for row in df_vc_fl.iterrows():
            if row[0] in self.ml.m.vc_fl_lin_0:
                self.ml.m.vc_fl_lin_0[row[0]].value = row[1].vc_fl_lin_0
                self.ml.m.vc_fl_lin_1[row[0]].value = row[1].vc_fl_lin_1

        if str_cadj == 'adjs':

            self.ml.m.vc_fl_lin_0[(self.ml.m.mps.dict_pp_id['DE_GAS_LIN'], 0)].value = 32
            self.ml.m.vc_fl_lin_1[(self.ml.m.mps.dict_pp_id['DE_GAS_LIN'], 0)].value = 0.0016


        self.ml.dct_vl['swcadj_vl'] = str_cadj



    def deactivate_swiss_reservoir_constraint(self):

        dict_chrs = {0: 'on', 1: 'off'}

        slct_chrs = self.ml.dct_step['swchrs']

        str_chrs = dict_chrs[slct_chrs]

        # reset
        for kk in self.ml.m.CapacStEn:
            self.ml.m.CapacStEn[kk].activate()

        if str_chrs == 'off':
            for kk in self.ml.m.CapacStEn:
                if kk[0] == self.ml.m.mps.dict_pp_id['CH_HYD_RES']:
                    self.ml.m.CapacStEn[kk].deactivate()

        self.ml.dct_vl['swchrs_vl'] = str_chrs

    def new_inflow_profile_for_ch(self):

        dict_inflchat = {0: 'original', 1: 'new'}

        slct_inflchat = self.ml.dct_step['swinflchat']

        str_inflchat = dict_inflchat[slct_inflchat]

        # reset
        df_infl_ch = self.ml.m.df_profinflow.loc[self.ml.m.df_profinflow.pp_id == self.ml.m.mps.dict_pp_id['CH_HYD_RES']]
        dict_infl_ch = df_infl_ch.set_index(['hy', 'pp_id', 'ca_id'])['value'].to_dict()
        for kk, val in dict_infl_ch.items():
            self.ml.m.inflowprof[kk].value = val

        if str_inflchat == 'new':

            dict_new_mt = {0: 0.02340,
                           1: 0.01614,
                           2: 0.02666,
                           3: 0.04655,
                           4: 0.15272,
                           5: 0.20410,
                           6: 0.17877,
                           7: 0.14128,
                           8: 0.10430,
                           9: 0.05646,
                           10: 0.02579,
                           11: 0.02383}

            df_infl_ch_new = self.ml.m.df_tm_soy[['mt_id', 'sy']].assign(pp_id=self.ml.m.mps.dict_pp_id['CH_HYD_RES'], ca_id=0)
            df_infl_ch_new['value'] = df_infl_ch_new.mt_id.replace(dict_new_mt)
            df_infl_ch_new['value'] /= df_infl_ch_new.value.sum()
            df_infl_ch_new['hy'] = df_infl_ch_new['sy']
            df_infl_new = pd.concat([df_infl_ch_new,
                                     df_infl_ch_new.assign(pp_id=self.ml.m.mps.dict_pp_id['DE_HYD_RES'])])
            dict_infl_ch_new = df_infl_new.set_index(['hy', 'pp_id', 'ca_id'])['value'].to_dict()

            for kk, val in dict_infl_ch_new.items():
                self.ml.m.inflowprof[kk].value = val

        self.ml.dct_vl['swinflchat_vl'] = str_inflchat

    def flatten_austria_hydro_inflow(self):

        dict_fh = {0: 'default', 1: 'flat'}

        slct_fh = self.ml.dct_step['swfh']
        str_fh = dict_fh[slct_fh]

        df = self.ml.io.param_to_df(self.ml.m.inflowprof, ('sy', 'pp_id', 'ca_id'))

        df = df.loc[df.pp_id.isin([kk for kk, vv in self.ml.m.mps.dict_pp.items() if 'HYD_RES' in vv and 'AT' in vv])]

        # reset:
        if str_fh == 'flat':
            df['value'] = df.groupby(['pp_id'], as_index=False)['value'].transform(lambda x: x.sum() / len(x))

        dict_df = df.set_index( ['sy', 'pp_id', 'ca_id'])['value'].to_dict()

        for kk, vv in dict_df.items():

            self.ml.m.inflowprof[kk].value = vv

        self.ml.dct_vl['swfh_vl'] = str_fh


#    def set_ramping_cost(self, slct_pp):
#
#        dict_rc = {0: 1,
#                   1: 1.1,
#                   2: 1.2,
#                   3: 1.3,
#                   4: 1.4,
#                   5: 1.5,
#                   6: 1.6,
#                   7: 1.7,
#                   8: 1.8
#                   }
#
#        slct_rc = self.ml.dct_step['swrc']
#        str_rc = dict_rc[slct_rc]
#
#        slct_pp_id = [self.ml.m.mps.dict_pp_id[pp] for pp in slct_pp]
#        dict_rc = (self.ml.m.df_plant_encar
#                            .loc[self.ml.m.df_plant_encar.pp_id
#                                                         .isin(slct_pp_id)]
#                            .set_index(['pp_id', 'ca_id'])['vc_ramp']
#                            .to_dict())
#
#
#        for kk, vv in dict_rc.items():
#            self.ml.m.vc_ramp[kk].value = vv * slct_rc
#
#
#        self.ml.dct_vl['swrc_vl'] = 'x%d'%str_rc


    def set_calibration_variations(self, dict_cl=None):


        if dict_cl is None:
            dict_cl = {4: 'double_ramping_cost',
                       1: 'triple_ramping_cost',
                       2: 'inflexible_hydro',
                       3: 'fr_nuclear_reduction',
                       0: 'default'}

        slct_cl = self.ml.dct_step['swcl']
        str_cl = dict_cl[slct_cl]

        # reset discharge duration
        for kk, vv in self.ml.m.df_plant_encar.join(self.ml.m.df_def_plant.set_index('pp_id')['pp'], on='pp_id').loc[self.ml.m.df_plant_encar.discharge_duration > 0].set_index(['pp_id', 'ca_id'])['discharge_duration'].to_dict().items():
            self.ml.m.discharge_duration[kk] = vv
        self.ml.m.hy_erg_min.activate()
        self.ml.m.hy_month_min.activate()
        self.ml.m.hy_reservoir_boundary_conditions.activate()
        self.ml.m.monthly_totals.activate()

        if str_cl == 'inflexible_hydro':
            self.ml.m.hy_erg_min.deactivate()
            self.ml.m.hy_month_min.deactivate()
            self.ml.m.hy_reservoir_boundary_conditions.deactivate()
            self.ml.m.monthly_totals.deactivate()

            for kk in self.ml.m.discharge_duration:
                self.ml.m.discharge_duration[kk] = 0
                self.ml.m.cap_pwr_leg[kk] = 100000

        # reset ramping
        for kk, vv in self.ml.m.df_plant_encar.set_index(['pp_id', 'ca_id'])['vc_ramp'].to_dict().items():
            if kk in [k for k in self.ml.m.vc_ramp]:
                self.ml.m.vc_ramp[kk].value = vv

        if str_cl == 'double_ramping_cost':
            for kk in self.ml.m.vc_ramp:
                self.ml.m.vc_ramp[kk].value = self.ml.m.vc_ramp[kk].value * 2

        if str_cl == 'triple_ramping_cost':
            for kk in self.ml.m.vc_ramp:
                self.ml.m.vc_ramp[kk].value = self.ml.m.vc_ramp[kk].value * 3

        if str_cl == 'fr_nuclear_reduction':
            # Power capacity reduced by approximate NTC of not included neighboring countries
            mps = maps.Maps(self.ml.io.sc_inp, self.ml.io.db)
            dict_cap_pwr_leg = {kk: self.ml.m.cap_pwr_leg[kk].value - 7000 for kk in self.ml.m.cap_pwr_leg.keys()
                                if kk[0] in mps.dict_pp.keys() and mps.dict_pp[kk[0]] == 'FR_NUC_ELC'}
            for kk, vv in dict_cap_pwr_leg.items():
                self.ml.m.cap_pwr_leg[kk] = vv


        self.ml.dct_vl['swcl_vl'] = str_cl


    def set_france_nuclear_vc(self, dict_fr=None):

        if dict_fr is None:
            dict_fr = {0: '1x',
                       1: '2x',
                       2: '3x',
                       3: '4x',
                       4: '5x',
                       5: '6x',
                       6: '8x',
                       7: '10x'}

        slct_fr = self.ml.dct_step['swfr']
        str_fr = dict_fr[slct_fr]


        _df = (self.ml.m.parameter_month_dict['vc_fl']
                        .set_index(['mt_id', 'fl_id', 'nd_id'])['value'])


        for kk, vv in _df.to_dict().items():
            if kk in [k for k in self.ml.m.vc_fl]:
                self.ml.m.vc_fl[kk].value = vv

        list_nuclear = self.ml.m.df_def_fuel.loc[self.ml.m.df_def_fuel.fl == 'nuclear_fuel', 'fl_id'].values
        list_france = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd == 'FR0', 'nd_id'].values

        _dfnuc = _df.loc[_df.index.get_level_values('fl_id').isin(list_nuclear)
                        &_df.index.get_level_values('nd_id').isin(list_france)]

        multiplicator = int(str_fr.replace('x', ''))

        _dfnuc *= multiplicator

        for kk, vv in _dfnuc.to_dict().items():
            if kk in [k for k in self.ml.m.vc_fl]:
                self.ml.m.vc_fl[kk].value = vv


        self.ml.dct_vl['swfr_vl'] = str_fr


    def set_chp_on_off(self, dict_chp=None):

        if dict_chp is None:
            dict_chp = {0: 'on',
                       1: 'off'}

        slct_chp = self.ml.dct_step['swchp']
        str_chp = dict_chp[slct_chp]

        self.ml.m.chp_prof.activate()

        if str_chp == 'off':
            self.ml.m.chp_prof.deactivate()

        self.ml.dct_vl['swchp_vl'] = str_chp

    def set_slope_lin(self, dict_sl=None):

        if dict_sl is None:
            dict_sl = {0: 'base_value_slope',
                       1: '2x_value_slope',
                       2: '4x_value_slope'}

        slct_sl = self.ml.dct_step['swsl']
        str_sl = dict_sl[slct_sl]

        for kk, vv in self.ml.m.df_plant_encar.set_index(['pp_id','ca_id'])['slope_lin_vc_fl'].to_dict().items():
            if kk in [k for k in self.ml.m.slope_lin_vc_fl]:
                self.ml.m.slope_lin_vc_fl[kk].value = vv

        if str_sl == '2x_value_slope':
            for kk in self.ml.m.slope_lin_vc_fl:
                self.ml.m.slope_lin_vc_fl[kk].value = self.ml.m.slope_lin_vc_fl[kk].value * 2

        if str_sl == '4x_value_slope':
            for kk in self.ml.m.slope_lin_vc_fl:
                self.ml.m.slope_lin_vc_fl[kk].value = self.ml.m.slope_lin_vc_fl[kk].value * 4

        self.ml.dct_vl['swsl_vl'] = str_sl


    def set_linear_vc(self, dict_ln=None):

        if dict_ln is None:

            dict_ln = {0: 'quadratic',
                       1: 'linear'}

        slct_ln = self.ml.dct_step['swln']
        str_ln = dict_ln[slct_ln]

        # fuels/nodes with linear power plants
        mask_lin = self.ml.m.df_def_plant.pp.str.contains('LIN')
        lin_fuel_node = (self.ml.m.df_def_plant.loc[mask_lin, ['fl_id', 'nd_id']]
                               .drop_duplicates())

        #######################################################################

        def switch_linear_quadratic(excl_pp=[]):

            slct_col = 'cap_pwr_leg' #+ (str_ln if not reset else '')

            dct_capacity = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(excl_pp)]
                             .set_index(['pp_id', 'ca_id'])[slct_col]
                             .to_dict())

            for kk, vv in dct_capacity.items():
                self.ml.m.cap_pwr_leg[kk] = 0


        if str_ln == 'linear':

            # inner join to filter (fl, nd) combinations defined above
            pp_quad = pd.merge(self.ml.m.df_def_plant,
                               lin_fuel_node, how='inner')
            pp_quad = (pp_quad.loc[pp_quad.pp.str.contains('LIN')]
                              .set_index('pp')['pp_id'].tolist())

#            excl_pp_lin=list(range(31,36))+list(range(46,50))+[55]
            switch_linear_quadratic(excl_pp=pp_quad)
            self.ml.m.objective_quad.deactivate()
#            self.ml.m.objective_lin.activate()

        if str_ln == 'quadratic':

#            switch_linear_quadratic(reset=True)
            pp_lin = pd.merge(self.ml.m.df_def_plant,
                              lin_fuel_node, how='inner')
            pp_lin = (pp_lin.loc[-pp_lin.pp.str.contains('LIN')]
                            .set_index('pp')['pp_id'].tolist())

#            excl_pp_quad=list(range(10,31))+list(range(36,46))+list(range(50,55))
            switch_linear_quadratic(excl_pp=pp_lin)
#            self.ml.m.objective_lin.deactivate()
            self.ml.m.objective_quad.activate()

#

        self.ml.dct_vl['swln_vl'] = str_ln

#    self.objective.deactivate()




    def set_historic_year(self, dict_hy=None):

        # e

        if dict_hy is None:
            dict_hy = {
                       0:  2015,
                       1:  2016,
                       2:  2014,
                       3:  2013,
                       4:  2012,
                       5:  2011,
                       6:  2010,
                       7:  2009,
                       8:  2008,
                       9:  2007,
                       10: 2006,
                       11: 2017
                      }

        slct_hy = self.ml.dct_step['swhy']

        str_hy = '_yr' + str(dict_hy[slct_hy]) if slct_hy > 0 else ''

        #######################################################################
        def set_fuel_prices(str_hy=None, reset=False):
            ''' Select fuel price values for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'vc_fl' + (str_hy if not reset else '')
            msg = ('Resetting vc_fl to base year values'
                   if reset else
                   'Setting vc_fl to values' + str_hy.replace('_', ' '))
            msg += ' from column {}'.format(slct_col)

            if 'vc_fl' in self.ml.m.dict_monthly_factors.keys():
                df_mt = self.ml.m.dict_monthly_factors['vc_fl']
                df = cols2tuplelist(self.ml.m.df_fuel_node_encar[['fl_id', 'nd_id']],
                                    df_mt['mt_id'], return_df=True)
                df = df.join(self.ml.m.df_fuel_node_encar.set_index(['fl_id', 'nd_id'])[slct_col], on=['fl_id', 'nd_id']).fillna(0)
                df = df.join(df_mt.set_index(['mt_id','fl_id', 'nd_id'])['mt_fact' + (str_hy if not reset else '')], on=['mt_id', 'fl_id', 'nd_id']).fillna(1)
                df['value'] = df[slct_col] * df['mt_fact' + (str_hy if not reset else '')]
                dct_prices = df.set_index(['mt_id', 'fl_id', 'nd_id'])['value'].to_dict()

            else:
                dct_prices = (self.ml.m.df_fuel_node_encar
                                    .set_index(['fl_id', 'nd_id'])[slct_col]
                                    .to_dict())

            print(msg)
            for kk, vv in dct_prices.items():
                self.ml.m.vc_fl[kk] = vv
        #######################################################################
        def set_cap_pwr_leg(str_hy=None, reset=False, excl_pp=[]):
            ''' Select power plant capacities for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'cap_pwr_leg' + (str_hy if not reset else '')
            msg = ('Resetting cap_pwr_leg to base year values.'
                   if reset else
                   'Setting cap_pwr_leg to values' + str_hy.replace('_', ' '))


            dct_cap = (self.ml.m.df_plant_encar.loc[-self.ml.m.df_plant_encar.pp_id.isin(excl_pp)]
                             .set_index(['pp_id', 'ca_id'])[slct_col]
                             .to_dict())

            print(msg)
            for kk, vv in dct_cap.items():
                self.ml.m.cap_pwr_leg[kk] = vv

        #######################################################################
        def set_cf_max(str_hy=None, reset=False, excl_pp=[]):
            ''' Select maximum capacity factors for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'cf_max' + (str_hy if not reset else '')
            msg = ('Resetting cf_max to base year values.'
                   if reset else
                   'Setting cf_max to values' + str_hy.replace('_', ' '))


            if 'cf_max' in self.ml.m.dict_monthly_factors.keys():
                df_mt = self.ml.m.dict_monthly_factors['cf_max']
                df = cols2tuplelist(self.ml.m.df_plant_encar[['pp_id', 'ca_id']].loc[self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.setlst['pp'])],
                                    df_mt['mt_id'], return_df=True)
                df = df.join(self.ml.m.df_plant_encar.set_index(['pp_id', 'ca_id'])[slct_col], on=['pp_id', 'ca_id']).fillna(0)
                df = df.join(df_mt.set_index(['mt_id','ca_id', 'pp_id'])['mt_fact' + (str_hy if not reset else '')], on=['mt_id', 'pp_id', 'ca_id']).fillna(1)
                df['value'] = df[slct_col] * df['mt_fact' + (str_hy if not reset else '')]
                dct_cf_max = df.set_index(['mt_id', 'pp_id', 'ca_id'])['value'].to_dict()

            else:

                dct_cf_max = (self.ml.m.df_plant_encar.loc[-self.ml.m.df_plant_encar.pp_id.isin(excl_pp)]
                                    .loc[self.ml.m.df_plant_encar.pp_id
                                               .isin(self.ml.m.setlst['pp'])]
                                    .set_index(['pp_id', 'ca_id'])[slct_col]
                                    .to_dict())

            print(msg)
            for kk, vv in dct_cf_max.items():
                self.ml.m.cf_max[kk] = vv

        #######################################################################
        def set_priceprof(str_hy=None, reset=False):
            ''' Select price profile for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'value' + (str_hy if not reset else '')
            msg = ('Resetting priceprof to base year values.'
                   if reset else
                   'Setting priceprof to values' + str_hy.replace('_', ' '))

            dct_priceprof = (self.ml.m.df_profprice_soy
                              .set_index(['sy', 'nd_id', 'fl_id'])[slct_col]
                              .to_dict())

            print(msg)
            for kk, vv in dct_priceprof.items():
                self.ml.m.priceprof[kk] = vv

        #######################################################################
        def set_dmnd(str_hy=None, reset=False):
            ''' Select demand profiles for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'value' + (str_hy if not reset else '')
            msg = ('Resetting dmnd to base year values.'
                   if reset else
                   'Setting dmnd to values' + str_hy.replace('_', ' '))





            dct_dmnd = (self.ml.m.df_profdmnd_soy
                              .set_index(['sy', 'nd_id', 'ca_id'])[slct_col]
                              .to_dict())

            print(msg)
            for kk, vv in dct_dmnd.items():
                self.ml.m.dmnd[kk] = vv

        #######################################################################
        def set_co2_price(str_hy=None, reset=False):
            ''' Select CO2 price for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'price_co2' + (str_hy if not reset else '')
            msg = ('Resetting price_co2 to base year values.'
                   if reset else
                   'Setting price_co2 to values' + str_hy.replace('_', ' '))

            if 'price_co2' in self.ml.m.dict_monthly_factors.keys():
                df_mt = self.ml.m.dict_monthly_factors['price_co2']
                df = cols2tuplelist(self.ml.m.df_def_node[['nd_id']].loc[self.ml.m.df_def_node.nd_id.isin(self.ml.m.slct_node_id)],
                                    df_mt['mt_id'], return_df=True)
                df = df.join(self.ml.m.df_def_node.set_index(['nd_id'])[slct_col], on=['nd_id']).fillna(0)
                df = df.join(df_mt.set_index(['mt_id','nd_id'])['mt_fact' + (str_hy if not reset else '')], on=['mt_id', 'nd_id']).fillna(1)
                df['value'] = df[slct_col] * df['mt_fact' + (str_hy if not reset else '')]
                dct_price_co2 = df.set_index(['mt_id', 'nd_id'])['value'].to_dict()

            else:


                dct_price_co2 = (self.ml.m.df_def_node
                                       .set_index(['nd_id'])[slct_col]
                                       .to_dict())

            print(msg)
            for kk, vv in dct_price_co2.items():
                self.ml.m.price_co2[kk] = vv
        #######################################################################

        def set_erg_inp(str_hy=None, reset=False, excl_fl=[]):
            ''' Select exogenous energy production for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'erg_inp' + (str_hy if not reset else '')
            msg = ('Resetting erg_inp to base year values.'
                   if reset else
                   'Setting erg_inp to values' + str_hy.replace('_', ' '))

            dct_erg_inp = (self.ml.m.df_fuel_node_encar.loc[-self.ml.m.df_fuel_node_encar.fl_id.isin(excl_fl)]
                               .set_index(['nd_id', 'ca_id',
                                           'fl_id'])[slct_col]
                               .to_dict())

            print(msg)
            for kk, vv in dct_erg_inp.items():
                self.ml.m.erg_inp[kk] = vv
        #######################################################################

        def set_erg_chp(str_hy=None, reset=False, excl_fl=[]):
            ''' Select exogenous chp energy production for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'erg_chp' + (str_hy if not reset else '')
            msg = ('Resetting erg_chp to base year values.'
                   if reset else
                   'Setting erg_chp to values' + str_hy.replace('_', ' '))

            dct_erg_chp = (self.ml.m.df_fuel_node_encar
                               .loc[-self.ml.m.df_fuel_node_encar.fl_id.isin(excl_fl)]
                               .set_index(['nd_id', 'ca_id',
                                           'fl_id'])[slct_col]
                               .to_dict())

            print(msg)
            for kk, vv in dct_erg_chp.items():
                self.ml.m.erg_chp[kk] = vv

        #######################################################################

        def set_cap_trm_leg(str_hy=None, reset=False):
            ''' Select exogenous energy production for selected year. '''

            if str_hy == None:
                reset=True
                str_hy = ''

            slct_col = 'cap_trm_leg' + (str_hy if not reset else '')
            msg = ('Resetting cap_trm_leg to base year values.'
                   if reset else
                   'Setting cap_trm_leg to values' + str_hy.replace('_', ' '))

            dct_cap_trm_leg = (self.ml.m.df_node_connect
                                   .set_index(['mt_id', 'nd_id',
                                               'nd_2_id', 'ca_id'])[slct_col]
                                   .to_dict())

            print(msg)
            for kk, vv in dct_cap_trm_leg.items():
                self.ml.m.cap_trm_leg[kk] = vv
        #######################################################################

        def set_supprof(str_hy=None, reset=False):
            '''
            Select supply profiles for selected year.
            (Normalized) input profiles are scaled by erg_inp
            and then divided by cap_pwr_leg to get the hourly capacity
            factor.
            '''

            if str_hy == None:
                reset=True
                str_hy = ''

            msg = ('Resetting supprof to base year values.'
                   if reset else
                   'Setting supprof to values' + str_hy.replace('_', ' '))

            df = self.ml.m.df_profsupply_soy
            df = df[['sy', 'pp_id', 'ca_id'] + ['value' + str_hy]]
            df = df.set_index(['sy', 'pp_id', 'ca_id'])
            dct_supprof = df['value' + str_hy].to_dict()

            print(msg)
            for kk, vv in dct_supprof.items():
                self.ml.m.supprof[kk] = vv
        #######################################################################

        # resetting everything to base year values
        # Note: inflow profiles are static and scaled by erg_inp in the constraint
#        set_fuel_prices(reset=True)
#        set_cap_pwr_leg(reset=True)
#        set_cf_max(reset=True)
#        set_dmnd(reset=True)
#        set_co2_price(reset=True)
#        set_erg_inp(reset=True)
#        set_erg_chp(reset=True)
#        set_cap_trm_leg(reset=True)
#        set_supprof(reset=True)
#        set_priceprof(reset=True)

#        excl_pp = []
#        excl_fl = []

#        set_fuel_prices(str_hy)
#        set_cap_pwr_leg(str_hy, excl_pp=excl_pp)
#        set_cf_max(str_hy, excl_pp=excl_pp)
#        set_dmnd(str_hy)
#        set_co2_price(str_hy)
#        set_erg_inp(str_hy, excl_fl=excl_fl)
#        set_erg_chp(str_hy, excl_fl=excl_fl)
#        set_cap_trm_leg(str_hy)
#        set_supprof(str_hy)
#        set_priceprof(str_hy)

        self.ml.dct_vl['swhy_vl'] = 'yr' + str(dict_hy[slct_hy])




