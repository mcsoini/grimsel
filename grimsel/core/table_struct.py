#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:26:27 2019

@author: user
"""

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
    ('cap_pwr_rem', ('pp_id', 'ca_id')),
    ('cap_pwr_tot', ('pp_id', 'ca_id')),
    ('cap_erg_tot', ('pp_id', 'ca_id')),
    ('cap_pwr_new', ('pp_id', 'ca_id')),
    ('erg_ch_yr', ('pp_id', 'ca_id', 'bool_out'), 'erg_yr')]
var_tr = [
    ('trm', ('sy', 'nd_id', 'nd_2_id', 'ca_id', 'bool_out')),
    ('erg_trm_rv_yr', ('nd_id', 'nd_2_id', 'ca_id', 'bool_out')),
    ('erg_trm_sd_yr', ('nd_id', 'nd_2_id', 'ca_id', 'bool_out'))]
par = [
    ('share_ws_set', ('nd_id',)),
    ('price_co2', ('nd_id',)),
    ('co2_int', ('fl_id',)),
    ('cap_pwr_leg', ('pp_id', 'ca_id')),
    ('cap_avlb', ('pp_id', 'ca_id')),
    ('cap_trme_leg', ('mt_id', 'nd_id', 'nd_2_id', 'ca_id')),
    ('cap_trmi_leg', ('mt_id', 'nd_id', 'nd_2_id', 'ca_id')),
    ('cf_max', ('pp_id', 'ca_id')),
    ('grid_losses', ('nd_id', 'ca_id')),
    ('erg_max', ('nd_id', 'ca_id', 'fl_id')),
    ('hyd_pwr_in_mt_max', ('pp_id',)),
    ('hyd_pwr_out_mt_min', ('pp_id',)),
    ('vc_fl', ('fl_id', 'nd_id')),
    ('co2_int', ('fl_id',)),
    ('nd_weight', ('nd_id',)),
    ('factor_lin_0', ('pp_id', 'ca_id')),
    ('factor_lin_1', ('pp_id', 'ca_id')),
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
    ('discharge_duration', ('pp_id', 'ca_id')),
#        ('hyd_erg_bc', ('sy', 'pp_id')), # this gets extremely large
    ('hyd_erg_min', ('pp_id',)),
    ('inflowprof', ('sy', 'pp_id', 'ca_id')),
    ('chpprof', ('sy', 'nd_id', 'ca_id')),
    ('supprof', ('sy', 'pp_id', 'ca_id')),
    ('pricesllprof', ('sy', 'pf_id')),
    ('pricebuyprof', ('sy', 'pf_id')),
    ('week_ror_output', ('wk', 'pp_id')),
    ('dmnd', ('sy', 'nd_id', 'ca_id')),
    ('erg_inp', ('nd_id', 'ca_id', 'fl_id')),
    ('erg_chp', ('pp_id', 'ca_id')),
    ('capchnge_max', tuple()),
    ]
dual = [('supply', ('sy', 'nd_id', 'ca_id'))]

list_collect = {'var_sy': var_sy,
                'var_mt': var_mt,
                'var_yr': var_yr,
                'var_tr': var_tr,
                'par': par,
                'dual': dual}

#def _make_dict(lst):
#    # can't use dict directly because of secondary table definition
#    # (value length 2)
#    return {comp[0]: comp[1] for comp in lst}
#
#
#dict_component_index = {**_make_dict(par), **_make_dict(var_mt),
#                        **_make_dict(var_sy), **_make_dict(var_tr),
#                        **_make_dict(var_yr)}
#
#dict_component_index = {key: dict_component_index[key] for key in sorted(dict_component_index)}

# table groups -> component -> index/secondary tb
DICT_TABLES = {lst: {tb[0]: tuple(tbb for tbb in tb[1:])
                     for tb in list_collect[lst]}
               for lst in list_collect}


# component -> table name
DICT_COMP_TABLE = {comp: (grp + '_' + spec[1]
                     if len(spec) > 1
                     else grp + '_' + comp)
              for grp, tbs in DICT_TABLES.items()
              for comp, spec in tbs.items()}

# component -> group
DICT_COMP_GROUP = {comp: grp
                  for grp, tbs in DICT_TABLES.items()
                  for comp, spec in tbs.items()}

# component -> index
DICT_COMP_IDX = {comp: spec[0]
            for grp, tbs in DICT_TABLES.items()
            for comp, spec in tbs.items()}

#DICT_COMP_IDX = {key: DICT_COMP_IDX[key] for key in sorted(DICT_COMP_IDX)}



