#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:48:46 2019

@author: user
"""
from importlib import reload
import os
from grimsel.core.io import IO
import numpy as np
import pandas as pd
import grimsel.core.model_loop as model_loop
from grimsel.core.model_base import ModelBase as MB
import grimsel_config as config
from grimsel import logger
logger.setLevel(1)

import grimsel


df_tm_soy = pd.DataFrame({'sy': range(4), 'weight': 8760 / 4})

df_def_node = pd.DataFrame({'nd': ['Node1'], 'nd_id':[0], 'price_co2': [0], 'nd_weight': [1]})
dict_nd = df_def_node.set_index('nd').nd_id.to_dict()

df_def_fuel = pd.DataFrame({'fl_id': range(3),
                            'fl': ['natural_gas', 'hard_coal', 'wind'],
                            'co2_int': [0.20196, 0.34596, 0]})
dict_fl = df_def_fuel.set_index('fl').fl_id.to_dict()

df_def_encar = pd.DataFrame({'ca_id': [0],
                             'ca': ['EL']})
dict_ca = df_def_encar.set_index('ca').ca_id.to_dict()

df_def_pp_type = pd.DataFrame({'pt_id': range(3), 'pt': ['GAS_LIN', 'WIND', 'HCO_ELC']})
dict_pt = df_def_pp_type.set_index('pt').pt_id.to_dict()

df_def_plant = pd.DataFrame({'pp': ['ND1_GAS_LIN', 'ND1_GAS_NEW', 'ND1_WIND', 'ND1_HCO_ELC'],
                             'pt_id': ['GAS_LIN', 'GAS_LIN', 'WIND', 'HCO_ELC'],
                             'nd_id': ['Node1'] * 4,
                             'fl_id': ['natural_gas', 'natural_gas', 'wind', 'hard_coal'],
                             'set_def_pr': [0, 0, 1, 0],
                             'set_def_pp': [1, 1, 0, 1],
                             'set_def_lin': [1, 0, 0, 0],
                             'set_def_add': [0, 1, 1, 0],
                            })
df_def_plant.index.name = 'pp_id'
df_def_plant = df_def_plant.reset_index()

# translate columns to id using the previously defined def tables
df_def_plant = df_def_plant.assign(pt_id=df_def_plant.pt_id.replace(dict_pt),
                                   nd_id=df_def_plant.nd_id.replace(dict_nd),
                                   fl_id=df_def_plant.fl_id.replace(dict_fl))
dict_pp = df_def_plant.set_index('pp').pp_id.to_dict()

df_def_profile = pd.DataFrame({'pf_id': range(2),
                               'pf': ['SUPPLY_WIND', 'DMND_NODE1']})
dict_pf = df_def_profile.set_index('pf').pf_id.to_dict()

eff_gas_min = 0.4
eff_gas_max = 0.6
cap_gas = 4000.
f0_gas = 1/eff_gas_min
f1_gas = 1/cap_gas * (f0_gas - 1/eff_gas_max)


dr, lt = 0.06, 20 # assumed discount rate 6% and life time
fact_ann = ((1+dr)**lt * dr) / ((1+dr)**lt - 1)

fc_cp_gas = 800
fc_cp_gas_ann = fact_ann * fc_cp_gas

fc_cp_wind = 1500 # assumed capital cost wind power
fc_cp_wind_ann = fact_ann * fc_cp_wind

df_plant_encar = pd.DataFrame({'pp_id': ['ND1_GAS_LIN', 'ND1_GAS_NEW', 'ND1_WIND', 'ND1_HCO_ELC'],
                               'ca_id': ['EL'] * 4,
                               'supply_pf_id': [None, None, 'SUPPLY_WIND', None],
                               'pp_eff': [None, eff_gas_max, None, 0.4],
                               'factor_lin_0': [f0_gas, None, None, None],
                               'factor_lin_1': [f1_gas, None, None, None],
                               'cap_pwr_leg': [cap_gas, 0, 0, 5000],
                               'fc_cp_ann': [None, fc_cp_gas_ann, fc_cp_wind_ann, None],
                              })

df_plant_encar = df_plant_encar.assign(supply_pf_id=df_plant_encar.supply_pf_id.replace(dict_pf),
                                       pp_id=df_plant_encar.pp_id.replace(dict_pp),
                                       ca_id=df_plant_encar.ca_id.replace(dict_ca))

df_node_encar = pd.DataFrame({'nd_id': ['Node1'], 'ca_id': ['EL'],
                              'dmnd_pf_id': ['DMND_NODE1']
                              })
df_node_encar = df_node_encar.assign(nd_id=df_node_encar.nd_id.replace(dict_nd),
                                     ca_id=df_node_encar.ca_id.replace(dict_ca),
                                     dmnd_pf_id=df_node_encar.dmnd_pf_id.replace(dict_pf))

df_fuel_node_encar = pd.DataFrame({'fl_id': ['natural_gas', 'hard_coal'],
                               'nd_id': ['Node1'] * 2,
                               'ca_id': ['EL'] * 2,
                               'vc_fl': [40, 10],
                              })
df_fuel_node_encar = df_fuel_node_encar.assign(fl_id=df_fuel_node_encar.fl_id.replace(dict_fl),
                                       nd_id=df_fuel_node_encar.nd_id.replace(dict_nd),
                                       ca_id=df_fuel_node_encar.ca_id.replace(dict_ca))

prf = [0.169, 0.122, 0.176, 0.284]

df_profsupply = pd.DataFrame({'supply_pf_id': [dict_pf['SUPPLY_WIND']] * len(prf),
                              'hy': range(len(prf)), 'value': prf})

prf = [6248.5, 6109.0, 6531.6, 6579.4]

df_profdmnd = pd.DataFrame({'dmnd_pf_id': [dict_pf['DMND_NODE1']] * len(prf),
                            'hy': range(len(prf)), 'value': prf})

for dftb, tbname in [(df_def_node, 'def_node'),
                     (df_def_plant, 'def_plant'),
                     (df_def_fuel, 'def_fuel'),
                     (df_def_encar, 'def_encar'),
                     (df_node_encar, 'node_encar'),
                     (df_def_pp_type, 'def_pp_type'),
                     (df_plant_encar, 'plant_encar'),
                     (df_tm_soy, 'tm_soy'),
                     (df_def_profile, 'def_profile'),
                     (df_fuel_node_encar, 'fuel_node_encar'),
                     (df_profsupply, 'profsupply'),
                     (df_profdmnd, 'profdmnd')]:
    dftb.to_csv('introductory_example_files/{}.csv'.format(tbname), index=False)
# %%



iokwargs = {'sc_warmstart': False,
            'cl_out': 'aaa',
            'resume_loop': False,
#            'data_path': PATH_CSV,
            'no_output': True,
            'autocomplete_curtailment': True,
            'sql_connector': None,
            'data_path': '/mnt/data/Dropbox/GRIMSEL_SOURCE/grimsel/notebooks/introductory_example_files', # config.PATH_CSV,
            'output_target': 'hdf5',
            'dev_mode': True
           }
mkwargs = {#'tm_filt': [('hy', range(4))],
           'symbolic_solver_labels': True,
                       }

ml = model_loop.ModelLoop(nsteps=[], iokwargs=iokwargs, mkwargs=mkwargs)

IO._close_all_hdf_connections()
ml.init_run_table()
ml.df_def_run


print(ml.io.datrd.data_path)


ml.io.read_model_data()


ml.m.df_def_plant

ml.m.init_maps()

ml.m.map_to_time_res()

# %
ml.io.write_runtime_tables()

ml.m.get_setlst()
ml.m.define_sets()
ml.m.add_parameters()
ml.m.define_variables()
ml.m.add_all_constraints()
ml.m.init_solver()
ml.io.init_output_tables()
ml.select_run(0)

ml.m.fc_cp_ann[(2,0)] = ml.m.fc_cp_ann[(2,0)].value * 50000

ml.perform_model_run()

ml.m.cap_pwr_new.display()

ml.m.curt.display()

ml.m.pwr.display()

## %%
#
## init ModelLoopModifier
#mlm = model_loop_modifier.ModelLoopModifier(ml)
#
#self = mlm
#
## starting row of loop
#irow_0 = ml.io.resume_loop if ml.io.resume_loop else 0
#
## loop over all rows of the resulting ml.df_def_run;
## corresponding modifications to the model a.exitt()re performed here;
## the model run method is called at the end
#irow = 0
#
#for irow in list(range(irow_0, len(ml.df_def_run))):
#    run_id = irow
#
#    ml.select_run(run_id)
#
#
#    logger.info('reset_parameters')
#    ml.m.reset_all_parameters()
#    ml.m.add_transmission_bounds_rules()
#
#
#    logger.info('select_coupling')
#    mlm.set_tariff(dict_tf=dict_tf)
#
#    logger.info('select_swiss_scenarios')
#    slct_ch = mlm.select_swiss_scenarios(dict_ch=dict_ch)
#    logger.info('set_future_year')
#    mlm.set_future_year(slct_ch=slct_ch, dict_fy=dict_fy)
#    logger.info('set hh node weight')
#    mlm.set_node_weights(dict_nw=dict_nw)
#
#    logger.info('set hh select_household_sample weight')
#    mlm.select_household_sample()
#
#    select_transmission_purchase = 'trm'
#    ml.m.delete_component('households_import_constraint')
#    ml.m.cadd('households_import_constraint', ml.m.sy_ndca, rule=households_trm_prc_constraint_rule)
#    ml.m.households_import_constraint.deactivate()
#
#    select_transmission_purchase = 'prc'
#    ml.m.delete_component('households_purchase_constraint')
#    ml.m.cadd('households_purchase_constraint', ml.m.sy_ndca, rule=households_trm_prc_constraint_rule)
#    ml.m.households_purchase_constraint.deactivate()
#
#    logger.info('select_coupling')
#    mlm.select_coupling(dict_cp=dict_cp)
#
#    logger.info('fill_peaker_plants')
#    ml.m.fill_peaker_plants(demand_factor=5,
#                            list_peak=[(ml.m.mps.dict_pp_id['CH_GAS_LIN'], 0)] if not ml.m.setlst['peak'] else []
#                            )
#
#    logger.info('_limit_prof_to_cap')
#    ml.m._limit_prof_to_cap()
#
#    ml.perform_model_run()
#
#
#
#
## %%
#




df = IO.variab_to_df(ml.m.pwr, ['sy', 'pp_id', 'ca_id'])
df['bool_out'] = False
df.loc[df.pp_id.isin(ml.m.setlst['sll']+ ml.m.setlst['curt']), 'bool_out'] = True

import grimsel.core.io as io

df_dm = IO.param_to_df(ml.m.dmnd, ['sy', 'nd_id', 'ca_id'])
df_dm = df_dm.join(ml.m.df_def_plant.loc[ml.m.df_def_plant.pp.str.endswith('_DMND')].set_index('nd_id').pp_id, on='nd_id')
df_dm['bool_out'] = True
df = pd.concat([df, df_dm.loc[:, df.columns]], axis=0, sort=False)

reload(io)
if hasattr(ml.m, 'trm'):
    iotrm = io.TransmIO('', '', ml.m.trm, ['sy', 'nd_id', 'nd_2_id', 'ca_id'], None,None, model=ml.m)
    dfall = iotrm._to_df(iotrm.comp_obj, ('sy', 'nd_id', 'nd_2_id', 'ca_id'))
    df_trm = iotrm.aggregate_nd2(dfall)
    df_trm = df_trm.join(ml.m.df_def_node.set_index('nd_id').nd_weight, on='nd_id')
    df_trm = iotrm._translate_trm(df_trm)
    df_trm.value = df_trm.value.abs() / df_trm.nd_weight
    df = pd.concat([df, df_trm], axis=0, sort=False)

df['data_cat'] = 'pwr'


incl_ch = hasattr(ml.m, 'pwr_st_ch')
if incl_ch:
    df_ch = IO.variab_to_df(ml.m.pwr_st_ch, ['sy', 'pp_id', 'ca_id'])
    df_ch['bool_out'] = True
    df_ch['data_cat'] = 'pwr'

    dferg = IO.variab_to_df(ml.m.erg_st, ['sy', 'pp_id', 'ca_id'])
    dferg['bool_out'] = False
    dferg['data_cat'] = 'erg'

    df = pd.concat([df, df_ch, dferg], sort=False)

iodual = io.DualIO('', '', ml.m.supply, ['sy', 'nd_id', 'ca_id'], None, None, model=ml.m)
df_dual = iodual.to_df()
df_dual['bool_out'] = False
df_dual['data_cat'] = 'mc'
df_dual['fl_id'] = 'mc'
df_dual = (df_dual.join(ml.m.df_def_node.set_index('nd_id').nd_weight, on='nd_id')
                  .assign(value=lambda x: x.value / x.nd_weight))


# scale by time slot weight
df_dual = (df_dual.assign(tm_id=df_dual.nd_id.replace(ml.m.dict_nd_tm_id))
                  .join(ml.m.df_tm_soy.set_index(['sy', 'tm_id']).weight, on=['sy', 'tm_id'])
                  .assign(value=lambda x: x.value / x.weight)[df_dual.columns.tolist()])


df.loc[df.bool_out, 'value'] *= -1
df = df.join(ml.m.df_def_plant.set_index('pp_id')[['fl_id', 'nd_id']], on='pp_id')
df = pd.concat([df_dual, df], sort=False)
df = df.join(ml.m.df_def_node.set_index('nd_id')[['tm_id']], on='nd_id')


if getattr(ml.m, 'df_sy_min_all', None) is not None:
    df = pd.merge(ml.m.df_sy_min_all, df, on=['tm_id', 'sy'], how='outer')


# %


df = ml.m.mps.id_to_name(df)
#df['is_hh'] = df.nd.str.contains('SFH').fillna(False)

# %
#df = df.loc[(df.fl.str.contains('exchange|mc')) |
#            (df.nd.str.contains('SFH')) |
#            (df.data_cat.str.contains('erg|mc'))]
#df = df.loc[(df.nd.str.contains('SFH'))]



df.loc[df.data_cat == 'pwr'].pivot_table(columns='fl',
                                         index='sy',
                                         values='value', aggfunc=sum).plot.bar(stacked=True)









