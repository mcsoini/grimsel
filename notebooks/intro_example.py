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

reload(model_loop)

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
mkwargs = {'tm_filt': [('hy', range(4))],
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

#for key in ml.m.weight:
#    ml.m.weight[key] = 8760 / 4
#
#for key in ml.m.price_co2:
#    ml.m.price_co2[key] = 10

ml.m.fc_cp_ann[(2,0)] = 20


ml.perform_model_run()

ml.m.cap_pwr_new.display()

ml.m.curt.display()
# %%

# init ModelLoopModifier
mlm = model_loop_modifier.ModelLoopModifier(ml)

self = mlm

# starting row of loop
irow_0 = ml.io.resume_loop if ml.io.resume_loop else 0

# loop over all rows of the resulting ml.df_def_run;
# corresponding modifications to the model a.exitt()re performed here;
# the model run method is called at the end
irow = 0

for irow in list(range(irow_0, len(ml.df_def_run))):
    run_id = irow

    ml.select_run(run_id)


    logger.info('reset_parameters')
    ml.m.reset_all_parameters()
    ml.m.add_transmission_bounds_rules()


    logger.info('select_coupling')
    mlm.set_tariff(dict_tf=dict_tf)

    logger.info('select_swiss_scenarios')
    slct_ch = mlm.select_swiss_scenarios(dict_ch=dict_ch)
    logger.info('set_future_year')
    mlm.set_future_year(slct_ch=slct_ch, dict_fy=dict_fy)
    logger.info('set hh node weight')
    mlm.set_node_weights(dict_nw=dict_nw)

    logger.info('set hh select_household_sample weight')
    mlm.select_household_sample()

    select_transmission_purchase = 'trm'
    ml.m.delete_component('households_import_constraint')
    ml.m.cadd('households_import_constraint', ml.m.sy_ndca, rule=households_trm_prc_constraint_rule)
    ml.m.households_import_constraint.deactivate()

    select_transmission_purchase = 'prc'
    ml.m.delete_component('households_purchase_constraint')
    ml.m.cadd('households_purchase_constraint', ml.m.sy_ndca, rule=households_trm_prc_constraint_rule)
    ml.m.households_purchase_constraint.deactivate()

    logger.info('select_coupling')
    mlm.select_coupling(dict_cp=dict_cp)

    logger.info('fill_peaker_plants')
    ml.m.fill_peaker_plants(demand_factor=5,
                            list_peak=[(ml.m.mps.dict_pp_id['CH_GAS_LIN'], 0)] if not ml.m.setlst['peak'] else []
                            )

    logger.info('_limit_prof_to_cap')
    ml.m._limit_prof_to_cap()

    ml.perform_model_run()




# %%





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

df = pd.merge(ml.m.df_sy_min_all, df, on=['tm_id', 'sy'],
              how='outer')


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









