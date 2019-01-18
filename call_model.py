# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:16:40 2019

@author: user
"""

import sys, os

sys.path.append('C:\\Users\\ashreeta\\Documents\\Martin\\SHARED_REPOS')
print(os.getcwd())

import numpy as np
import pandas as pd

from importlib import reload

import grimsel.core.model_loop as model_loop
from grimsel.core.model_base import ModelBase as MB

import grimsel.auxiliary.aux_sql_func as aql
import grimsel.analysis.sql_analysis as sql_analysis

import grimsel.config as config
import grimsel.model_loop_modifier as model_loop_modifier

db = config.DATABASE

sc_out = 'out_replace_all'

# %%

for fn in [fn for fn in os.listdir('.') if 'ephemeral' in fn or 'def_loop' in fn]:
    os.remove(fn)
    
# %%
    
connect_dict = dict(db=db,
                    password=config.PSQL_PASSWORD,
                    user=config.PSQL_USER,
                    port=config.PSQL_PORT,
                    host=config.PSQL_HOST)
sqlc = aql.sql_connector(**connect_dict)

# additional kwargs for the model
mkwargs = {
           'slct_encar': ['EL'],
           'slct_node': ['AT0', 'IT0', 'DE0', 'CH0', 'FR0'],
           'nhours': 1,
           'verbose_solver': False,
           'constraint_groups': MB.get_constraint_groups(excl=['chp', 'ror'])
           }

# additional kwargs for the i/o
iokwargs = {'resume_loop': 3581,
            'autocomplete_curtailment': True,
           }

nvr, nst = 30, 31
nsteps_default = [
                  ('swvr', nvr, np.arange),     # select vre share
                  ('swst', nst, np.arange),     # select storage capacity
                  ('swtc', 3, np.arange),       # select storage tech
                  ('swpt', 3, np.arange),       # select vre type
                  ('swyr', 5, np.arange),       # select meteo year
                  ('swco', 3, np.arange),       # select co2 emission price
                  ('swtr', 2, np.arange),       # cross-border transmission on/off
                  ('swrc', 26, np.arange)]      # select ramping cost

mlkwargs = {
            'sc_out': sc_out,
            'db': db,
            'nsteps': nsteps_default,
            'sql_connector': sqlc,
            }

sc_out = mlkwargs['sc_out']

ml = model_loop.ModelLoop(**mlkwargs, mkwargs=mkwargs, iokwargs=iokwargs)


swst_max = nst
dict_st = {nst: st for nst, st in enumerate(list(np.linspace(0, 0.3, swst_max)))}
swvr_max = nvr - 1
dict_vre = {nvr: vr for nvr , vr
            in enumerate(['default'] + list(np.linspace(0, 0.7, swvr_max)))}


# %%

###############################################################################
# reduce output

ml.io.var_sy = [par for par in ml.io.var_sy if 'pwr' in par[0] or 'pwr_st_ch' in par[0]]
ml.io.par = [par for par in ml.io.par if not 'prof' in par[0]]
ml.io.var_tr = [var for var in ml.io.var_tr if 'erg' in var[0]]
ml.io.dual = []

###############################################################################

ml.io.var_sy
# %%


# figure 8/9: various nuclear power indicators france 
slct_vr = [0] + list(np.arange(0, ml.df_def_loop.swvr_id.max(), 4) + 1)
slct_st = list(np.arange(0, ml.df_def_loop.swvr_id.max() + 10, 10))
mask_base = (ml.df_def_loop.swvr_id.isin(slct_vr) &
               ml.df_def_loop.swst_id.isin(slct_st) &
               ml.df_def_loop.swtc_id.isin([0, 1]) &
               ml.df_def_loop.swpt_id.isin([0]) &
               ml.df_def_loop.swyr_id.isin([0]) &
               ml.df_def_loop.swco_id.isin([0]) &
               ml.df_def_loop.swtr_id.isin([0]) &
               ml.df_def_loop.swrc_id.isin([0]))

# figure 10: ramping costs
slct_vr = list(np.arange(13, ml.df_def_loop.swvr_id.max() + 1, 4))
slct_st = list(np.arange(0, ml.df_def_loop.swvr_id.max() + 10, 10))
mask_ramping = (ml.df_def_loop.swvr_id.isin(slct_vr) &
                ml.df_def_loop.swst_id.isin(slct_st) &
                ml.df_def_loop.swtc_id.isin([0, 1]) &
                ml.df_def_loop.swpt_id.isin([0]) &
                ml.df_def_loop.swyr_id.isin([0]) &
                ml.df_def_loop.swco_id.isin([0]) &
                ml.df_def_loop.swtr_id.isin([0]) 
#               ml.df_def_loop.swrc_id.isin([0]) --> all
               )

# figure 11: qualitatively diverging storage impact
slct_vr = [{round(val * 10000) / 10000 if not val == 'default' else val: key
            for key, val in dict_vre.items()}[vr] for vr in [0.5, 0.7, 0.4]]
mask_years = (ml.df_def_loop.swvr_id.isin(slct_vr) &
#                ml.df_def_loop.swst_id.isin(slct_st) & --> all
                ml.df_def_loop.swtc_id.isin([0, 1]) &
                ml.df_def_loop.swpt_id.isin([0]) &
#                ml.df_def_loop.swyr_id.isin([0]) & --> all
                ml.df_def_loop.swco_id.isin([0]) &
                ml.df_def_loop.swtr_id.isin([0]) &
                ml.df_def_loop.swrc_id.isin([0]) 
                )

# figure 12: consecutive replacement
slct_vr = [0] + list(np.arange(0, ml.df_def_loop.swvr_id.max(), 1) + 1)
slct_st = list(np.arange(0, ml.df_def_loop.swvr_id.max() + 5, 10))
mask_consec = (#ml.df_def_loop.swvr_id.isin(slct_vr) &
                ml.df_def_loop.swst_id.isin(slct_st) &
                ml.df_def_loop.swtc_id.isin([0]) &
                ml.df_def_loop.swpt_id.isin([0, 1, 2]) &
                ml.df_def_loop.swyr_id.isin([0]) &
                ml.df_def_loop.swco_id.isin([0, 1]) &
                ml.df_def_loop.swtr_id.isin([0, 1]) &
                ml.df_def_loop.swrc_id.isin([0]) 
                )

# figure 13: emissions
slct_vr = [0] + list(np.arange(0, ml.df_def_loop.swvr_id.max(), 2) + 1)
slct_st = list(np.arange(0, ml.df_def_loop.swvr_id.max() + 100, 10))
mask_emissions = (ml.df_def_loop.swvr_id.isin(slct_vr) &
                ml.df_def_loop.swst_id.isin(slct_st) &
                ml.df_def_loop.swtc_id.isin([0]) &
                ml.df_def_loop.swpt_id.isin([0]) &
                ml.df_def_loop.swyr_id.isin([0]) &
#                ml.df_def_loop.swco_id.isin([0]) & --> all
                ml.df_def_loop.swtr_id.isin([0]) &
                ml.df_def_loop.swrc_id.isin([0]) 
                )

mask_total = mask_base | mask_ramping | mask_years | mask_consec | mask_emissions

print(mask_total.sum())

# Extra for emissions plot

slct_vr = [0] + list(np.arange(0, ml.df_def_loop.swvr_id.max(), 2) + 1)
slct_st = list(np.arange(0, ml.df_def_loop.swst_id.max() + 100, 5))
mask_emissions = (ml.df_def_loop.swvr_id.isin(slct_vr) &
                ml.df_def_loop.swst_id.isin(slct_st) &
                ml.df_def_loop.swtc_id.isin([0]) &
                ml.df_def_loop.swpt_id.isin([0]) &
                ml.df_def_loop.swyr_id.isin([0]) &
#                ml.df_def_loop.swco_id.isin([0]) & --> all
                ml.df_def_loop.swtr_id.isin([0]) &
                ml.df_def_loop.swrc_id.isin([0]) 
                )


mask_emissions = mask_emissions & -mask_total

print(mask_emissions.sum())

# Extra for emissions plot LIO_STO

slct_vr = [0] + list(np.arange(0, ml.df_def_loop.swvr_id.max(), 2) + 1)
slct_st = [0, 20] #  list(np.arange(0, ml.df_def_loop.swst_id.max() + 100, 5))
mask_emissions_LIO = (ml.df_def_loop.swvr_id.isin(slct_vr) &
                      ml.df_def_loop.swst_id.isin(slct_st) &
                      ml.df_def_loop.swtc_id.isin([1]) & # <====================
                      ml.df_def_loop.swpt_id.isin([0]) &
                ml.df_def_loop.swyr_id.isin([0]) &
#                ml.df_def_loop.swco_id.isin([0]) & --> all
                ml.df_def_loop.swtr_id.isin([0]) &
                ml.df_def_loop.swrc_id.isin([0]) 
                )


mask_emissions_LIO = mask_emissions_LIO & -mask_total& -mask_emissions

print(mask_emissions_LIO.sum())

# %%

ml.df_def_loop = pd.concat([ml.df_def_loop.loc[mask_total],
                            ml.df_def_loop.loc[mask_emissions],
                            ml.df_def_loop.loc[mask_emissions_LIO],
                           ])
ml.df_def_loop

# %%

# init ModelLoopModifier
mlm = model_loop_modifier.ModelLoopModifier(ml)

# starting row of loop
irow_0 = ml.io.resume_loop if ml.io.resume_loop else 0

irow = irow = 0

ml.m._limit_prof_to_cap()
#ml.perform_model_run(zero_run=True)
# %
for irow in list(range(irow_0, len(ml.df_def_loop))):
    run_id = irow

    print('select_run')
    ml.select_run(run_id)

    ###
    print('set_trm_cap_onoff')
    mlm.set_trm_cap_onoff()
    
    ####
    print('set_co2_price')
    mlm.set_co2_price()

    ####
    print('set_winsol_year')
    mlm.set_winsol_year()

    ####
    print('select_vre_pp_types')
    slct_pp, slct_pt = mlm.select_vre_pp_types()

    ####
    print('scale_vre')
    mlm.scale_vre(slct_pt, dict_vre)

    ####
    print('set_ramping_cost')
    mlm.set_ramping_cost()

    ####
    print('select_storage_tech')
    mlm.select_storage_tech()

    ####
    print('set_storage_cap')
    mlm.set_storage_cap(dict_st)

    ############### RUN MODEL ###############
    print('fill_peaker_plants')
    ml.m.fill_peaker_plants(demand_factor=2)

    print('_limit_prof_to_cap')
    ml.m._limit_prof_to_cap()

    print('perform_model_run')
    ml.perform_model_run()

    for fn in [fn for fn in os.listdir('.') if 'ephemeral' in fn or 'def_loop' in fn]:
        os.remove(fn)


# %%
        
sqa = sql_analysis.SqlAnalysis(sc_out=sc_out, db=db)
sqa.build_tables_plant_run_quick()
sqa.build_table_plant_run_tot_balance(from_quick=True)



# EMISSIONS LIN

slct_run_id = aql.exec_sql('''
                           SELECT run_id FROM out_replace_all.def_loop
                           WHERE swvr_vl IN (SELECT GENERATE_SERIES(0, 100, 5) || '.00%')
                           AND swst_vl IN ('0.00%', '20.00%')
                           AND swrc_vl = 'x1.0'
                           AND swyr_vl = '2015'
                           AND swpt_vl = 'WIN_ONS|WIN_OFF|SOL_PHO'
                           AND swtr_vl = 'on'
                           ''', db=db)
slct_run_id = list(np.array(slct_run_id).flatten())

sqa = sql_analysis.SqlAnalysis(sc_out=sc_out, db=db, slct_run_id=slct_run_id)
sqa.analysis_emissions_lin()


# FILTDIFF FOR FRENCH NUCLEAR
import grimsel.auxiliary.maps as maps

mps = maps.Maps(sc_out, db)

slct_nd_id = [mps.dict_nd_id[nd] for nd in ['FR0']]
slct_run_id = aql.exec_sql('''
                           SELECT run_id FROM out_replace_all.def_loop
                           WHERE swvr_vl IN (SELECT GENERATE_SERIES(0, 100, 10) || '.00%')
                           AND swst_vl IN ('0.00%', '20.00%')
                           AND swrc_vl = 'x1.0'
                           AND swyr_vl = '2015'
AND swpt_vl = 'WIN_ONS|WIN_OFF|SOL_PHO'
                           AND swtr_vl = 'on'
                           AND swco_vl = '40EUR/tCO2'
                           ''', db=db)
slct_run_id = list(np.array(slct_run_id).flatten())

reload(sql_analysis)
sqac_filt_chgdch = sql_analysis.SqlAnalysis(sc_out=sc_out, db=db,
                                            slct_pt=['NUC_ELC'],
                                            slct_run_id=slct_run_id,
                                            nd_id=slct_nd_id)
self = sqac_filt_chgdch
sqac_filt_chgdch.analysis_agg_filtdiff()
