#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Build input database schema.

All tables are copied to csv files at the very end.
'''


import pandas as pd
import sys, os
from xlrd import open_workbook
from importlib import reload
import numpy as np

from grimsel.auxiliary.aux_general import print_full
from grimsel.auxiliary.aux_general import expand_rows
from grimsel.auxiliary.aux_general import read_xlsx_table
from grimsel.auxiliary.aux_general import translate_id
from grimsel.auxiliary.aux_m_func import cols2tuplelist
import grimsel.auxiliary.aux_sql_func as aql
from grimsel.auxiliary.timemap import TimeMap
import grimsel.config as config

# change to current
os.chdir(os.path.dirname(__file__))

db = config.DATABASE
sc = config.SCHEMA
fn = config.FN_XLSX
data_path = config.PATH_CSV

wb = open_workbook(fn)

# %%

exec_str = ('''
            DROP SCHEMA IF EXISTS {sc} CASCADE;
            CREATE SCHEMA IF NOT EXISTS {sc};
            ''').format(sc=sc, )
aql.exec_sql(exec_str, db=db)

def yr_getter(par, data_type=False, rnge=range(2017, 2005 - 1, -1)):
    return [par + i if not data_type else (par + i, data_type)
            for i in [''] + ['_yr' + str(ii) for ii
            in rnge if not ii == 2015]]

# <codecell>
tb_name = 'def_pp_type'
cols = [('pt_id',' SMALLINT'),
        ('pt',' varchar(20)'),
        ('pp_broad_cat', 'varchar(100)'),
        ('color', 'VARCHAR(7)')]
pk = ['pt_id']
unique = ['pt']
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'def_fuel'
cols = [('fl_id', 'SMALLINT'), ('fl', 'varchar(20)'),
        ('co2_int', 'DOUBLE PRECISION'),
        ('is_ca', 'SMALLINT'),
        ('is_constrained', 'SMALLINT'),
        ('color', 'VARCHAR(7)')]
pk = ['fl_id']
unique = ['fl']
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'def_node'
cols = [('nd_id', 'SMALLINT'), ('nd', 'VARCHAR(3)'),
        ('discount_rate', 'DOUBLE PRECISION'),
        ('charging', 'DOUBLE PRECISION'),
        ('share_ws_set', 'DOUBLE PRECISION'),
        ('chp_cap_pwr_leg', 'DOUBLE PRECISION'),
        ('color', 'VARCHAR(7)')] + yr_getter('price_co2', 'DOUBLE PRECISION', [2015])
pk = ['nd_id']
unique = ['nd']
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'node_encar'
cols = [('nd_id', 'SMALLINT'),
        ('ca_id', 'SMALLINT'),
        ('grid_losses', 'DOUBLE PRECISION'),
        ('grid_losses_absolute', 'DOUBLE PRECISION'),
        ('vc_dmnd_flex', 'DOUBLE PRECISION')] + yr_getter('dmnd_sum', 'DOUBLE PRECISION', [2015])
pk = ['nd_id', 'ca_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'def_encar'
cols = [('ca_id', 'SMALLINT'),
        ('fl_id', 'SMALLINT'),
        ('ca', 'VARCHAR(2)')]
pk = ['ca_id']
unique = ['ca']
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'def_month'
cols = [('mt_id',' SMALLINT'),
        ('month_min_hoy',' SMALLINT'),
        ('month_weight',' SMALLINT'),
        ('mt',' VARCHAR(3)')]
pk = ['mt_id']
unique = ['name']
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'def_week'
cols = [('wk_id',' SMALLINT'),
        ('wk',' SMALLINT'),
        ('week_weight', 'SMALLINT')]
pk = ['wk_id']
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

#tb_name = 'def_soy'
#cols = [('sy',' SMALLINT'), ('weight',' SMALLINT')]
#pk = ['sy']
#unique = []
#aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
#               pk=pk, unique=unique, db=db)

tb_name = 'def_plant'
cols = [('pp_id',' SMALLINT'), ('pp',' VARCHAR(20)'),
        ('nd_id',' SMALLINT', sc + '.def_node(nd_id)'),
        ('fl_id',' SMALLINT', sc + '.def_fuel(fl_id)'),
        ('pt_id',' SMALLINT', sc + '.def_pp_type(pt_id)'),
        ('set_def_pr',' SMALLINT'),
        ('set_def_cain',' SMALLINT'),
        ('set_def_ror',' SMALLINT'),
        ('set_def_pp',' SMALLINT'), ('set_def_st',' SMALLINT'),
        ('set_def_hyrs',' SMALLINT'),
        ('set_def_chp',' SMALLINT'),
        ('set_def_add',' SMALLINT'),
        ('set_def_rem',' SMALLINT'),
        ('set_def_sll',' SMALLINT'),
        ('set_def_curt',' SMALLINT'),
        ('set_def_lin',' SMALLINT'),
        ('set_def_scen',' SMALLINT'),
        ('set_def_winsol',' SMALLINT'),
        ('set_def_tr', 'SMALLINT'),
        ('set_def_peak', 'SMALLINT')]
pk = ['pp_id']
unique = ['pp']
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'plant_month'
cols = [('mt_id',' SMALLINT', sc + '.def_month(mt_id)'),
        ('pp_id',' SMALLINT', sc + '.def_plant(pp_id)'),
        ('hyd_erg_bc','DOUBLE PRECISION')]
pk = ['mt_id', 'pp_id']
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

#tb_name = 'plant_week'
#cols = [('wk_id',' SMALLINT', sc + '.def_week(wk_id)'),
#        ('pp_id',' SMALLINT', sc + '.def_plant(pp_id)'),
#        ('week_ror_output','DOUBLE PRECISION')]
#pk = ['wk_id', 'pp_id']
#aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
#               pk=pk, unique=unique, db=db)

tb_name = 'plant_encar'
cols = [('pp_id',' SMALLINT', sc + '.def_plant(pp_id)'),
        ('ca_id',' SMALLINT', sc + '.def_encar(ca_id)'),
        ('pp_eff','DOUBLE PRECISION'),
        ('erg_max','DOUBLE PRECISION'),
        ('discharge_duration','DOUBLE PRECISION'),
        ('st_lss_rt','DOUBLE PRECISION'),
        ('st_lss_hr','DOUBLE PRECISION'),
        ('vc_fl_lin_0', 'DOUBLE PRECISION'),
        ('vc_fl_lin_1','DOUBLE PRECISION'),
        ('factor_vc_co2_lin_0', 'DOUBLE PRECISION'),
        ('factor_vc_co2_lin_1','DOUBLE PRECISION'),
        ('vc_ramp','DOUBLE PRECISION'),
        ('vc_ramp_low','DOUBLE PRECISION'),
        ('vc_ramp_high', 'DOUBLE PRECISION'),
        ('vc_om','DOUBLE PRECISION'),
       ] + (yr_getter('cap_pwr_leg', 'DOUBLE PRECISION', [2015])
         + yr_getter('cf_max', 'DOUBLE PRECISION', [2015]))

pk = ['pp_id', 'ca_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'imex_comp'
cols = [('nd_id', 'SMALLINT', sc + '.def_node(nd_id)'),
        ('nd_2_id', 'SMALLINT', sc + '.def_node(nd_id)'),
        ] + yr_getter('erg_trm', 'DOUBLE PRECISION', [2015])
pk = ['nd_id', 'nd_2_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

#tb_name = 'erg_yr_comp'
#cols = [('nd_id', 'SMALLINT', sc + '.def_node(nd_id)'),
#        ('fl_id', 'SMALLINT', sc + '.def_sub_fuel(fl_id)'),
#        ('value', 'DOUBLE PRECISION')]
#pk = ['nd_id', 'fl_id']
#unique = []
#aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
#               pk=pk, unique=unique, db=db)

tb_name = 'profdmnd'
cols = [('nd_id', 'SMALLINT', sc + '.def_node(nd_id)'),
        ('ca_id', 'SMALLINT', sc + '.def_encar(ca_id)'),
        ('hy', 'SMALLINT')] + yr_getter('value', 'DOUBLE PRECISION', [2015])
pk = ['hy', 'ca_id', 'nd_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'profchp'
cols = [('nd_id', 'SMALLINT', sc + '.def_node(nd_id)'),
        ('ca_id', 'SMALLINT', sc + '.def_encar(ca_id)'),
        ('hy', 'SMALLINT'), ('value', 'DOUBLE PRECISION')]
pk = ['hy', 'nd_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'profinflow'
cols = [('pp_id', 'SMALLINT', sc + '.def_plant(pp_id)'),
        ('ca_id', 'SMALLINT', sc + '.def_encar(ca_id)'),
        ('hy', 'SMALLINT'), ('value', 'DOUBLE PRECISION')]
pk = ['hy', 'pp_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'profprice'
cols = [('hy', 'SMALLINT'),
        ('nd_id', 'SMALLINT', sc + '.def_node(nd_id)'),
        ('fl_id', 'SMALLINT', sc + '.def_fuel(fl_id)'),
       ] + yr_getter('value', 'DOUBLE PRECISION', [2015])
pk = ['hy', 'nd_id', 'fl_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

#
tb_name = 'fuel_node_encar'
cols = ([('fl_id', 'SMALLINT', sc + '.def_fuel(fl_id)'),
         ('nd_id', 'SMALLINT', sc + '.def_node(nd_id)'),
         ('ca_id', 'SMALLINT', sc + '.def_encar(ca_id)'),
         ('has_profile', 'SMALLINT'),
         ('is_chp', 'SMALLINT'),
         ] + yr_getter('erg_inp', 'DOUBLE PRECISION', [2015])
           + yr_getter('vc_fl', 'DOUBLE PRECISION', [2015])
           + yr_getter('erg_chp', 'DOUBLE PRECISION', [2015]))
pk = ['fl_id', 'nd_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

# table with monthly parameter modifiers
tb_name = 'parameter_month'
cols = ([('set_1_name', 'VARCHAR'), # from {'nd_id', 'fl_id', 'pp_id'}
         ('set_2_name', 'VARCHAR'), # from {'nd_id', 'fl_id', 'pp_id'}
         ('set_1_id', 'SMALLINT'),
         ('set_2_id', 'SMALLINT'),
         ('mt_id',' SMALLINT', sc + '.def_month(mt_id)'),
#         ('ca_id', 'SMALLINT', sc + '.def_encar(ca_id)'),
         ('parameter', 'VARCHAR') # the parameter this applies to
         ] + yr_getter('mt_fact', 'NUMERIC(10,9)', [2015]))
pk = ['parameter', 'set_1_id', 'set_2_id', 'mt_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

tb_name = 'node_connect'
cols = [('nd_id', 'SMALLINT', sc + '.def_node (nd_id)'),
        ('nd_2_id', 'SMALLINT', sc + '.def_node (nd_id)'),
        ('ca_id', 'SMALLINT', sc + '.def_encar (ca_id)'),
        ('mt_id', 'SMALLINT', sc + '.def_month(mt_id)'),
        ('eff', 'DOUBLE PRECISION')] + yr_getter('cap_trm_leg', 'DOUBLE PRECISION', [2015])
pk = ['nd_id', 'nd_2_id', 'mt_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)


tb_name = 'hydro'
cols = [('pp_id',' SMALLINT', sc + '.def_plant(pp_id)'),
        ('min_erg_mt_out_share', 'DOUBLE PRECISION'),
        ('max_erg_mt_in_share', 'DOUBLE PRECISION'),
        ('min_erg_share', 'DOUBLE PRECISION')]
pk = ['pp_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)

# %%

ppca_cols = ['pp_id', 'ca_id', 'pp_eff', 'discharge_duration',
             'st_lss_rt',  'st_lss_hr', 'vc_ramp', 'vc_ramp_low',
             'vc_ramp_high', 'vc_om', 'vc_fl_lin_0', 'vc_fl_lin_1',
             'factor_vc_co2_lin_0', 'factor_vc_co2_lin_1',
            ] + yr_getter('cf_max') + yr_getter('cap_pwr_leg')
df_plant_encar = read_xlsx_table(wb, ['PLANT_ENCAR'], columns=ppca_cols)
df_plant_encar = df_plant_encar[[c for c in df_plant_encar.columns if not 'yr20' in c]]


lst_set = ['pr', 'cain', 'ror', 'pp', 'st', 'hyrs', 'chp', 'add',
           'rem', 'scen', 'sll', 'curt', 'lin', 'tr', 'peak']
dfpp_cols = ['pp_id', 'pp', 'nd_id', 'fl_id', 'pt_id',
             ] + ['set_def_' + i for i in lst_set]
df_def_plant = read_xlsx_table(wb, ['DEF_PLANT'], dfpp_cols)

df_def_encar = read_xlsx_table(wb, ['DEF_ENCAR'], ['ca_id', 'fl_id', 'ca'])

dfpt_cols = ['pt_id', 'pt', 'pp_broad_cat', 'color']
df_def_pp_type = read_xlsx_table(wb, ['DEF_PP_TYPE'], dfpt_cols)

nd_cols = ['nd_id', 'nd', 'discount_rate', 'charging', 'share_ws_set',
           'chp_cap_pwr_leg', 'color', 'price_co2']
df_def_node = read_xlsx_table(wb, ['DEF_NODE'], nd_cols)

ndca_cols = ['nd_id', 'ca_id', 'grid_losses',
             'grid_losses_absolute', 'vc_dmnd_flex'] + yr_getter('dmnd_sum')
df_node_encar = read_xlsx_table(wb, ['NODE_ENCAR'], ndca_cols)
df_node_encar = df_node_encar[[c for c in df_node_encar.columns if not 'yr20' in c]]

df_def_fuel = read_xlsx_table(wb, ['DEF_FUEL'],
                                  ['fl_id', 'fl', 'co2_int', 'is_ca',
                                   'is_constrained', 'color'])
df_fuel_node_encar = read_xlsx_table(wb, ['FUEL_NODE_ENCAR'],
                                     (['fl_id', 'nd_id', 'ca_id', 'is_chp',
                                       'has_profile']
                                      + yr_getter('erg_inp')
                                      + yr_getter('vc_fl')
                                      + yr_getter('erg_chp')))
df_fuel_node_encar = df_fuel_node_encar[[c for c in df_fuel_node_encar.columns
                                         if not 'yr20' in c]]


ndcn_cols = ['nd_id', 'nd_2_id', 'ca_id', 'mt_id', 'eff'
            ] + yr_getter('cap_trm_leg')
df_node_connect = read_xlsx_table(wb, ['NODE_CONNECT'], ndcn_cols)
df_node_connect = df_node_connect[[c for c in df_node_connect.columns
                                   if not 'yr20' in c]]

###############################################################################


###############################################################################

# For replacement of efficiency in df_plant_encar
#eff_cols = ['pp_id','year','nd_id','ca_id','pp_eff']
#df_eff_0 = read_xlsx_table(wb, ['EFFICIENCY'], eff_cols)

# Efficiencies exceptions for specific plants
#df_eff_plant = read_xlsx_table(wb, ['EFFICIENCY'], eff_cols,
#                               sub_table='EFF_PLANT')

# for chp profile scaling
chp_cols = ['nd_id', 'cf_profile_0', 'cap_pwr_leg_chp', 'min_prod',
            'target_prod', 'profile_scale']
df_def_chp = read_xlsx_table(wb, ['CHP'], chp_cols)

# specific hydro parameters
h_c = ['pp_id', 'min_erg_mt_out_share', 'max_erg_mt_in_share', 'min_erg_share']
df_hydro = read_xlsx_table(wb, ['HYDRO'], h_c, sub_table='HYDRO')
hm_c = ['pp_id', 'parameter', 'mt_id', 'value']
df_plant_month = read_xlsx_table(wb, ['HYDRO'], hm_c, None, 'PLANT_MONTH')

# import/export stats
imex_stats_cols = ['nd_id', 'nd_2_id'] + yr_getter('erg_trm')
df_imex_comp = read_xlsx_table(wb, ['IMEX_STATS'], imex_stats_cols)
df_imex_comp = df_imex_comp[[c for c in df_imex_comp.columns if not 'yr20' in c]]


tm = TimeMap()
tm.gen_hoy_timemap()

df_tm = tm.df_time_map

df_def_month = df_tm.pivot_table(index=['mt_id', 'mt'],
                                      values=['hy'],
                                      aggfunc=[min, len])
df_def_month.columns = (df_def_month.columns.droplevel(level=1))
new_cols = {'month': 'mt', 'min': 'month_min_hoy',
            'len': 'month_weight'}
df_def_month = (df_def_month.reset_index().rename(columns=new_cols))

df_def_week = df_tm.pivot_table(index=['wk_id', 'wk'], values='hy',
                                aggfunc=[len])
df_def_week.columns = (df_def_week.columns.droplevel(level=1))
new_cols = {'week': 'wk', 'len': 'week_weight'}
df_def_week = (df_def_week.reset_index().rename(columns=new_cols))



# %%
########### PROFINFLOW: HYD PROFILES TO HOURLY TABLE ###########

df_tm = df_tm[['hy', 'doy', 'mt']].rename(columns={'mt': 'mt_id'})

# monthly inflow reservoirs

dfres = df_plant_month.loc[df_plant_month['parameter'] == 'monthly_share_inflow']
dfres = pd.merge(df_tm, dfres, on='mt_id')
dfres = dfres[['hy', 'pp_id', 'value']]

# concatenate
df_profinflow = dfres
df_profinflow['ca_id'] = 'EL'

# normalize each profile
df_profinflow['value'] = (df_profinflow.groupby(['pp_id'])['value']
                                       .transform(lambda x: x/x.sum()))

########### REPLACE INFLOW IN FRANCE WITH EXTRA DATA ###########

exec_strg = '''
WITH tb_lvl AS (SELECT * FROM profiles_raw.hydro_level_rte_fr)
, tb_hyd AS (
  SELECT EXTRACT(week FROM "DateTime")::SMALLINT - 1 AS wk_id, *
  FROM profiles_raw.rte_production_eco2mix
  WHERE fl_id = 'reservoirs'
), tb_prd_wk AS (
    SELECT year, wk_id, sum(value) AS hydro_production_week FROM tb_hyd
    GROUP BY year, wk_id
    ORDER BY year, wk_id
), tb_wk AS (
    SELECT year, slot AS hy, EXTRACT(week FROM datetime)::SMALLINT - 1 AS wk_id
        FROM profiles_raw.timestamp_template
    WHERE year IN (SELECT DISTINCT year FROM tb_lvl)
), tb_wk_weight AS (
    SELECT year, wk_id, COUNT(*) AS wk_weight FROM tb_wk
    GROUP BY year, wk_id
)
SELECT hy, 'FR_HYD_RES'::VARCHAR AS pp_id,
    /* CALCULATING AVERAGE HOURLY WEEKLY INFLOW HERE */
    (hydro_production_week - start_level_diff) / wk_weight AS value,
    'EL'::VARCHAR AS ca_id
FROM tb_wk
LEFT JOIN tb_wk_weight USING (year, wk_id)
LEFT JOIN tb_prd_wk USING (year, wk_id)
LEFT JOIN tb_lvl USING (year, wk_id)
WHERE year = 2015
'''


df_profinflow_FR = pd.DataFrame(aql.exec_sql(exec_strg, db=db),
                                columns=['hy', 'pp_id', 'value', 'ca_id'])
df_profinflow_FR['value'] = (df_profinflow_FR.value
                             / df_profinflow_FR.value.sum())


df_profinflow = pd.concat([df_profinflow.loc[df_profinflow.pp_id != 'FR_HYD_RES'],
                           df_profinflow_FR])


df_profinflow.pivot_table(index='hy', columns='pp_id', values='value').plot()


############## PLANT_MONTH: HYDRO RESERVOIR FILLING LEVEL BCs #################

df_plant_month = (df_plant_month.loc[df_plant_month['parameter']
                                     =='energy_boundary_condition',
                                     ['mt_id', 'pp_id', 'value']]
                                .rename(columns={'value':'hyd_erg_bc'}))

################ NODE_CONNECT #################################################

# expand non-swiss node_connect to all months
df_node_connect = expand_rows(df_node_connect, ['mt_id'],
                              [df_node_connect.loc[
                                      df_node_connect.nd_id.isin(['CH0']),
                                      'mt_id'].unique().tolist()])

'''
id columns to int
'''
df_def_encar['ca_id'] = df_def_encar['ca_id'].apply(int)
df_def_plant['pp_id'] = df_def_plant['pp_id'].apply(int)
df_def_node['nd_id'] = df_def_node['nd_id'].apply(int)
df_def_fuel['fl_id'] = df_def_fuel['fl_id'].apply(int)

df_def_encar, _ = translate_id(df_def_encar, df_def_fuel, 'fl')

df_plant_encar, dict_encar_id = translate_id(df_plant_encar, df_def_encar, 'ca')
df_plant_encar, dict_plant_id = translate_id(df_plant_encar, df_def_plant, 'pp')

df_def_plant, dict_fuel_id = translate_id(df_def_plant, df_def_fuel, 'fl')
df_def_plant, dict_node_id = translate_id(df_def_plant, df_def_node, 'nd')

df_fuel_node_encar, _ = translate_id(df_fuel_node_encar, df_def_fuel, 'fl')
df_fuel_node_encar, _ = translate_id(df_fuel_node_encar, df_def_encar, 'ca')
df_fuel_node_encar, _ = translate_id(df_fuel_node_encar, df_def_node, 'nd')

df_def_plant, _ = translate_id(df_def_plant, df_def_pp_type, 'pt')

df_node_encar, _ = translate_id(df_node_encar, df_def_encar, 'ca')
df_node_encar, _ = translate_id(df_node_encar, df_def_node, 'nd')

df_node_connect, _ = translate_id(df_node_connect, df_def_node, 'nd')
df_node_connect, _ = translate_id(df_node_connect, df_def_node, ['nd', 'nd_2'])
df_node_connect, _ = translate_id(df_node_connect, df_def_encar, 'ca')

df_profinflow, _ = translate_id(df_profinflow, df_def_encar, 'ca')
df_profinflow, _ = translate_id(df_profinflow, df_def_plant, 'pp')

df_hydro, _ = translate_id(df_hydro, df_def_plant, 'pp')

df_plant_month, _ = translate_id(df_plant_month, df_def_month, 'mt')
df_plant_month, _ = translate_id(df_plant_month, df_def_plant, 'pp')

df_imex_comp, _ = translate_id(df_imex_comp, df_def_node, 'nd')
df_imex_comp, _ = translate_id(df_imex_comp, df_def_node, ['nd', 'nd_2'])





'''
add various set definition columns to df_def_plant
'''
df_def_plant['set_def_peak'] = 0
df_def_plant.loc[df_def_plant['pp'].apply(lambda x: str.find(x, 'PEAK')) > -1, 'set_def_peak'] = 1

mask_ws = (df_def_plant['pp'].apply(lambda x: str.find(x, 'WIN') > 0)
         | df_def_plant['pp'].apply(lambda x: str.find(x, 'SOL') > 0))
df_def_plant['set_def_winsol'] = 0
df_def_plant.loc[mask_ws, 'set_def_winsol'] = 1

print('Write all output')
write_dfs = [
             (df_def_node, 'def_node'),
             (df_def_encar, 'def_encar'),
             (df_def_pp_type, 'def_pp_type'),
             (df_def_fuel, 'def_fuel'),
             (df_def_plant, 'def_plant'),
             (df_def_month, 'def_month'),
             (df_def_week, 'def_week'),
             (df_plant_encar, 'plant_encar'),
#             (df_profchp, 'profchp'),
#             (df_profdmnd, 'profdmnd'),
#             (df_parmt, 'parameter_month'),
#             (df_profsupply, 'profsupply'),
             (df_profinflow, 'profinflow'),
             (df_hydro, 'hydro'),
             (df_node_encar, 'node_encar'),
             (df_plant_month, 'plant_month'),
#             (df_plant_week, 'plant_week'),
             (df_node_connect, 'node_connect'),
             (df_fuel_node_encar, 'fuel_node_encar'),
             (df_imex_comp, 'imex_comp'),
#             (df_profprice, 'profprice_comp'),
             ]
for idf in write_dfs:
    print('Writing ', idf[1])
    aql.write_sql(idf[0], db, sc, idf[1], 'append')


################ PROF ROR #####################################################

exec_strg = '''
            INSERT INTO {sc}.profinflow (pp_id, ca_id, hy, value)
            SELECT dfpp.pp_id, 0::SMALLINT AS ca_id, hy, value AS value
            FROM profiles_raw.weekly_ror_data AS ror
            LEFT JOIN (SELECT nd_id, nd FROM {sc}.def_node) AS dfnd ON dfnd.nd = ror.nd_id
            LEFT JOIN (SELECT pp_id, nd_id
                       FROM {sc}.def_plant WHERE pp LIKE '%ROR%') AS dfpp ON dfpp.nd_id = dfnd.nd_id
            '''.format(sc=sc)
aql.exec_sql(exec_strg, db=db)


################ PROFCHP ######################################################

exec_strg = '''
            INSERT INTO {sc}.profchp (hy, nd_id, ca_id, value)
            SELECT hy, dfnd.nd_id, 0::SMALLINT AS ca_id, value AS value
            FROM profiles_raw.chp_profiles AS chp
            LEFT JOIN (SELECT nd_id, nd FROM {sc}.def_node) AS dfnd
                ON dfnd.nd = chp.nd_id;
            '''.format(sc=sc)
aql.exec_sql(exec_strg, db=db)

################ PROFDMND #####################################################
# %%
df_profdmnd_0 = aql.read_sql(db, 'profiles_raw', 'load_complete',
                           keep=['nd_id', 'value', 'hy', 'year'],
                           filt=[('nd_id', ['DE0'], ' NOT LIKE '),
                                 ('year', [2015])])

df_profdmnd_de = pd.DataFrame(aql.exec_sql('''
            SELECT nd_id, value, how, wk_id, mt_id, hy, year
            FROM (SELECT how, wk_id, mt_id, slot AS hy,
                     datetime AS "DateTime"
                 FROM profiles_raw.timestamp_template
                 WHERE year = 2015) AS ts
            LEFT JOIN (SELECT * FROM profiles_raw.agora_profiles
                WHERE year = 2015 AND fl_id = 'dmnd') AS ag
            ON ts."DateTime" = ag."DateTime"
             ''', db=db), columns=['nd_id', 'value', 'how', 'wk_id', 'mt_id', 'hy', 'year'])

df_profdmnd_de_oct = df_profdmnd_de.loc[df_profdmnd_de.mt_id.isin([10, 9])].pivot_table(index='how', columns='wk_id', values='value').loc[:, 40:44]

# linear interpolation of the differences between target and source column
targt_start = df_profdmnd_de_oct.loc[117, 42]
targt_end = df_profdmnd_de_oct.loc[167, 42]
source_start = df_profdmnd_de_oct.loc[117, 41]
source_end = df_profdmnd_de_oct.loc[167, 41]

diff_start = targt_start - source_start
diff_end = targt_end - source_end

idx_nan = df_profdmnd_de_oct.loc[df_profdmnd_de_oct[42].isnull()].index.values
idx_nan = np.concatenate((np.array([idx_nan.min() - 1]),
                          idx_nan,
                          np.array([idx_nan.max() + 1])))
shifts = (diff_end - diff_start) / (idx_nan[-1] - idx_nan[0]) * (idx_nan - idx_nan[0]) + diff_start

df_profdmnd_de_oct_new = pd.DataFrame(np.array([idx_nan, shifts]).T,
                                      columns=['how', 'new'])
df_profdmnd_de_oct_new = df_profdmnd_de_oct_new.set_index('how')
df_profdmnd_de_oct = df_profdmnd_de_oct.join(df_profdmnd_de_oct_new, on='how')
df_profdmnd_de_oct['new'] += df_profdmnd_de_oct[41]

df_profdmnd_de_oct[42].fillna(df_profdmnd_de_oct['new'], inplace=True)

df_profdmnd_de_oct.columns = df_profdmnd_de_oct.columns.rename('wk_id')

df_profdmnd_de_oct = df_profdmnd_de_oct[[42]].stack().rename('new')

df_profdmnd_de = pd.merge(df_profdmnd_de, df_profdmnd_de_oct.reset_index(), on=['wk_id', 'how'], how='left')

df_profdmnd_de['value'].fillna(df_profdmnd_de.new, inplace=True)

df_profdmnd_de = df_profdmnd_de.drop(['new', 'wk_id', 'mt_id', 'how'], axis=1)


df_profdmnd_de = df_profdmnd_de.sort_values('hy').reset_index(drop=True)

df_profdmnd_de['nd_id'] = 'DE0'
df_profdmnd_de['year'] = 2015


df_profdmnd_de.set_index('hy')['value'].plot()

df_profdmnd_0 = pd.concat([df_profdmnd_0, df_profdmnd_de], axis=0)
df_profdmnd_0 = df_profdmnd_0.reset_index(drop=True)

df_profdmnd_0['ca_id'] = 'EL'


df_profdmnd_0, _ = translate_id(df_profdmnd_0, df_def_node, 'nd')
df_profdmnd_0, _ = translate_id(df_profdmnd_0, df_def_encar, 'ca')


df_profdmnd_0.pivot_table(index=['nd_id', 'ca_id', 'year', 'hy'], aggfunc=len).max()



# just scaling profile for now
# 1. normalize
df_profdmnd_0['value_norm'] = \
        (df_profdmnd_0.groupby(['nd_id', 'ca_id', 'year'])['value']
                      .apply(lambda x: x/x.sum()))

df_dmnd = df_node_encar[[c for c in df_node_encar.columns if 'dmnd_sum' in c] + ['nd_id', 'ca_id']].copy()
#df_dmnd['dmnd_sum_yr2017'].fillna(df_dmnd['dmnd_sum_yr2016'], inplace=True)
df_dmnd.columns = [(c if not 'dmnd' in c else (int(c.replace('dmnd_sum_yr', ''))) if 'yr' in c else 2015)
                   for c in df_dmnd.columns]
df_dmnd = df_dmnd.set_index(['nd_id', 'ca_id']).stack().reset_index().rename(columns={'level_2': 'year', 0: 'dmnd'})
df_dmnd = df_dmnd.set_index(['nd_id', 'ca_id', 'year'])




df_profdmnd_0['value_scaled'] = df_profdmnd_0.groupby(['nd_id', 'ca_id', 'year'])['value_norm'].apply(lambda x: x * df_dmnd.loc[x.name].values[0])

df_profdmnd = df_profdmnd_0.pivot_table(values='value_scaled',
                                        index=['nd_id', 'ca_id', 'hy'],
                                        columns=['year']).reset_index()
df_profdmnd.columns = [(c if not type(c) is int
                        else ('value_yr' + str(c)
                        if not c == 2015 else 'value'))
                       for c in df_profdmnd.columns]

dmnd_scale = {'FR0': 1.072368,
              'DE0': 1.06832,
              'IT0': 0.982416,
              'AT0': 0.91,
              'CH0': 1
             }
dmnd_scale = {dict_node_id[kk]: vv for kk, vv in dmnd_scale.items()}

df_profdmnd['scale'] = df_profdmnd.nd_id.replace(dmnd_scale)
df_profdmnd['value'] *= df_profdmnd.scale
df_profdmnd = df_profdmnd.drop('scale', axis=1)

df_profdmnd['value'] = df_profdmnd.value.apply(lambda x: int(x * 100000) / 100000)

'''
Comparison price profiles
'''

df_profprice = aql.read_sql(db, 'profiles_raw', 'epex_price_volumes',
                            filt=[('quantity', ['price_eur_mwh']),
                                  ('year', [2015], ' = ')],
                            keep=['hy', 'nd_id', 'year', 'value'])
df_profprice = df_profprice.pivot_table(index=['hy', 'nd_id'],
                                        values='value', columns='year')
df_profprice.columns = ['value' + ('_yr%d'%c if not c == 2015 else '')
                        for c in df_profprice.columns]
df_profprice = df_profprice[[c for c in yr_getter('value')
                             if c in df_profprice.columns]]
df_profprice = df_profprice.reset_index()

df_profprice['fl_id'] = 'electricity'




df_profprice, _ = translate_id(df_profprice, df_def_node, 'nd')
df_profprice, _ = translate_id(df_profprice, df_def_fuel, 'fl')

print('Write all output')
write_dfs = [
             (df_profdmnd, 'profdmnd'),
             (df_profprice, 'profprice'),
             ]
for idf in write_dfs:
    print('Writing ', idf[1])
    aql.write_sql(idf[0], db, sc, idf[1], 'append')
# %%
######## PROFSUPPLY FROM PROFILES_RAW.NINJA DATA STRAIGHT TO LP_INPUT #########

cf_data_type = 'NUMERIC(9,8)'

# Copy capacity scale adjusted base year profiles to profsupply table
exec_strg = '''
            DROP TABLE IF EXISTS {sc}.profsupply CASCADE;
            SELECT hy, pp_id AS pp, 0::SMALLINT AS ca_id, value_mod AS value
            INTO {sc}.profsupply
            FROM profiles_raw.ninja_mod
            WHERE year = 2015;
            
            ALTER TABLE {sc}.profsupply
                ALTER value TYPE {cf_data_type}
            '''.format(sc=sc, cf_data_type=cf_data_type)
aql.exec_sql(exec_strg, db=db)


# only 2015 !!
#for iyr in list(set([yr[0] for yr in
#                aql.exec_sql('SELECT DISTINCT year FROM profiles_raw.ninja_mod;',
#                db=db) if not 2015 in yr])):
#    exec_strg = '''
#                ALTER TABLE {sc}.profsupply
#                ADD COLUMN IF NOT EXISTS value_yr{yr} {cf_data_type};
#
#                UPDATE {sc}.profsupply AS prf
#                SET value_yr{yr} = rw.value_mod
#                FROM profiles_raw.ninja_mod AS rw
#                WHERE rw.year = {yr}
#                    AND prf.hy = rw.hy
#                    AND prf.pp = rw.pp_id;
#                '''.format(yr=str(iyr), sc=sc, cf_data_type=cf_data_type)
#    aql.exec_sql(exec_strg, db=db)
    
exec_strg = '''
            ALTER TABLE {sc}.profsupply
            ADD COLUMN pp_id SMALLINT;

            UPDATE {sc}.profsupply AS prfsp
            SET pp_id = dfpp.pp_id
            FROM (SELECT pp, pp_id FROM {sc}.def_plant) AS dfpp
            WHERE dfpp.pp = prfsp.pp;

            DELETE FROM {sc}.profsupply
            WHERE pp_id IS NULL;

            ALTER TABLE {sc}.profsupply
            DROP COLUMN IF EXISTS pp,
            DROP CONSTRAINT IF EXISTS pk_profsupply,
            ADD CONSTRAINT pk_profsupply
            PRIMARY KEY (pp_id, hy, ca_id),
            ADD CONSTRAINT fk_profsupply_pp_id
            FOREIGN KEY (pp_id) REFERENCES {sc}.def_plant(pp_id),
            ADD CONSTRAINT fk_profsupply_ca_id
            FOREIGN KEY (ca_id) REFERENCES {sc}.def_encar(ca_id);
            '''.format(sc=sc)
aql.exec_sql(exec_strg, db=db)

slct_pp_id = [vv for kk, vv in dict_plant_id.items() if 'FR_WIN_OFF' in kk or 'DE_WIN_OFF' in kk]
df_slct = aql.read_sql(db, sc, 'profsupply',
             filt=[('pp_id', slct_pp_id)]).set_index('hy')
df_slct.sort_index()[[c for c in df_slct.columns if 'value' in c]].plot(marker='.')

# %% ADDING BIOMASS PRODUCTION PROILE AS VRE

df = aql.read_sql(db, 'profiles_raw', 'entsoe_generation', filt=[('fl_id', ['bio_all'])])
df_CH = df.loc[df.nd_id == 'AT0']
df_CH['nd_id'] = 'CH0'
df = pd.concat([df, df_CH])
dfpv = df.loc[df.DateTime.dt.year == 2016].pivot_table(index=['DateTime'], columns='nd_id', values='value')
dfpv[dfpv == 0] = np.nan
dfpv[[c for c in dfpv.columns]] = dfpv.groupby(pd.TimeGrouper('M'))[[c for c in dfpv.columns]].transform(lambda x: x.mean())

dftm = df.loc[df.nd_id.isin(['DE0']) & df.year.isin([2016])].set_index('DateTime')['hy']
dfbio = dfpv.stack().reset_index().rename(columns={0: 'value'}).join(dftm, on='DateTime').copy()
dfbio['value'] = dfbio.groupby('nd_id')['value'].transform(lambda x: x / x.sum())
dfbio, _ = translate_id(dfbio, df_def_node, 'nd')
dfbio = dfbio.join(df_def_plant.loc[df_def_plant.pp.str.contains('BAL')].set_index('nd_id')['pp_id'], on='nd_id')
dfbio['ca_id'] = 0

'''
Copy data to additional years... doesn't do anything since profsupply is 2015
only!
'''

dfbio = dfbio.loc[:, list(aql.get_sql_cols('profsupply', sc, db).keys())]
fill_list = [c for c in dfbio.columns if 'yr' in c]
dfbio[fill_list] = value=dfbio[['value'] * len(fill_list)]

list_years = [c.replace('value', '') for c in dfbio.columns if 'value' in c]

for iyr in list_years:
    cap_col = 'cap_pwr_leg' + iyr
    erg_col = 'erg_inp' + iyr
    val_col = 'value' + iyr
    df_cap = df_plant_encar.loc[df_plant_encar.pp_id.replace({vv: kk for kk, vv in dict_plant_id.items()}).str.contains('BAL'), ['pp_id', cap_col]]
    df_erg = df_fuel_node_encar.loc[df_fuel_node_encar.fl_id.replace({vv: kk for kk, vv in dict_fuel_id.items()}).str.contains('bio'), ['fl_id', 'nd_id', erg_col]]
    df_erg = df_erg.join(df_def_plant.loc[df_def_plant.pp.str.contains('BAL')].set_index(['nd_id', 'fl_id'])['pp_id'], on=['nd_id', 'fl_id'])

    dfbio = (dfbio.join(df_erg.set_index(['pp_id'])[erg_col], on='pp_id')
                  .join(df_cap.set_index(['pp_id'])[cap_col], on='pp_id'))

    dfbio[val_col] *= dfbio[erg_col] / dfbio[cap_col]

    dfbio = dfbio.drop([cap_col, erg_col], axis=1)



aql.write_sql(dfbio, db, sc, 'profsupply', 'append')

# %%
tb_name = 'profprice_comp'
cols = [('nd_id', 'SMALLINT', sc + '.def_node(nd_id)'),
        ('ca_id', 'SMALLINT', sc + '.def_encar(ca_id)'),
        ('swhy_vl', 'VARCHAR(6)'),
#        ('month', 'SMALLINT'),
#        ('day', 'SMALLINT'),
#        ('hod', 'SMALLINT'),
        ('hy', 'SMALLINT'),
        ('price_eur_mwh', 'DOUBLE PRECISION'),
        ('volume_mwh', 'DOUBLE PRECISION')]
pk = ['hy', 'nd_id', 'swhy_vl', 'ca_id']
unique = []
aql.init_table(tb_name=tb_name, cols=cols, schema=sc, ref_schema=sc,
               pk=pk, unique=unique, db=db)


df_profprice = aql.read_sql(db, 'profiles_raw', 'epex_price_volumes',
                            filt=[('year', [2015])])
df_profprice = df_profprice.pivot_table(index=['year', 'nd_id', 'hy'],
                                        values='value',
                                        columns='quantity').reset_index()
df_profprice = df_profprice.rename(columns={'year': 'swhy_vl'})
df_profprice['swhy_vl'] = 'yr' + df_profprice['swhy_vl'].astype(str)
df_profprice['ca_id'] = 'EL'


df_profprice, _ = translate_id(df_profprice, df_def_node, 'nd')
df_profprice, _ = translate_id(df_profprice, df_def_encar, 'ca')

write_dfs = [
             (df_profprice, 'profprice_comp'),
             ]
for idf in write_dfs:
    print('Writing ', idf[1])
    aql.write_sql(idf[0], db, sc, idf[1], 'append')



# %% PARAMETER MONTHLY ADJUSTMENTS



# capacity availability for lignite and nuclear from monthly production and capacities
exec_strg = '''
WITH nhours AS (
    SELECT mt_id, COUNT(datetime) FROM profiles_raw.timestamp_template
    WHERE year = 2015
    GROUP BY mt_id
), ppca AS (
    SELECT nd, fl, pp, cap_pwr_leg FROM lp_input_replace.plant_encar
    NATURAL LEFT JOIN (SELECT pp, pp_id, fl_id, nd_id FROM lp_input_replace.def_plant) AS dfpp
    NATURAL LEFT JOIN (SELECT fl, fl_id FROM lp_input_replace.def_fuel) AS dffl
    NATURAL LEFT JOIN (SELECT nd, nd_id FROM lp_input_replace.def_node) AS dfnd
), all_cap_fl AS (
    SELECT fl, nd, SUM(cap_pwr_leg) AS cap_pwr_leg FROM ppca
    GROUP BY fl, nd
), tb_fr AS (
    SELECT fl_id AS fl, year, nd_id AS nd, mt_id, SUM(value) AS erg FROM profiles_raw.rte_production_eco2mix
    WHERE fl_id IN ('nuclear_fuel') AND year = 2015
    GROUP BY fl_id, year, nd_id, mt_id
), tb_de AS (
    SELECT fl_id AS fl, year, nd_id AS nd, mt_id, 1000 * SUM(value) AS erg
    FROM (SELECT *, EXTRACT(month FROM "DateTime") - 1 AS mt_id
          FROM profiles_raw.agora_profiles) AS tbag
    WHERE fl_id IN ('nuclear_fuel', 'lignite') AND year = 2015
    GROUP BY fl_id, year, nd_id, mt_id
), tb_ch AS (
    SELECT fl, 2015::SMALLINT AS year, nd, mt_id, erg FROM profiles_raw.monthly_production
    WHERE fl IN ('nuclear_fuel')
), tb_all AS (
    SELECT * FROM tb_fr
    UNION ALL
    SELECT * FROM tb_de
    UNION ALL
    SELECT * FROM tb_ch
), tb_final AS (
    SELECT tb_all.*, erg / count AS cap_from_erg, erg / count / cap_pwr_leg AS cap_avlb
    FROM tb_all
    LEFT JOIN nhours ON nhours.mt_id = tb_all.mt_id
    NATURAL LEFT JOIN (SELECT nd, fl, cap_pwr_leg FROM all_cap_fl) AS ppca
)
SELECT tb_final.fl, tb_final.nd, pp, year, tb_final.mt_id, 'EL'::VARCHAR AS ca,
CASE WHEN cap_avlb > 1 THEN 1 ELSE cap_avlb END AS cap_avlb
FROM tb_final
/* EXPAND TO PLANTS */
FULL OUTER JOIN (SELECT nd, fl, pp FROM ppca WHERE fl IN (SELECT fl FROM tb_final)) AS ppca
    ON ppca.fl = tb_final.fl AND ppca.nd = tb_final.nd;
'''

df_parmt_cap_avlb = pd.DataFrame(aql.exec_sql(exec_strg, db=db),
                                 columns=['fl_id', 'nd_id', 'pp_id', 'year',
                                          'mt_id', 'ca_id', 'cap_avlb'])
df_parmt_cap_avlb = df_parmt_cap_avlb.pivot_table(index=['pp_id', 'ca_id', 'mt_id'], values='cap_avlb')
df_parmt_cap_avlb = df_parmt_cap_avlb.reset_index().rename(columns={'cap_avlb': 'mt_fact',
                                                                    'pp_id': 'set_1_id',
                                                                    'ca_id': 'set_2_id'})
df_parmt_cap_avlb['set_1_name'] = 'pp_id'
df_parmt_cap_avlb['set_2_name'] = 'ca_id'
df_parmt_cap_avlb['parameter'] = 'cap_avlb'
df_parmt_cap_avlb['set_3_name'] = np.nan
df_parmt_cap_avlb['set_3_id'] = -1

###############################################################################
## expanding monthly factors for vc_fl to all countries #######################

parmt_cols = ['mt_id', 'set_1_name', 'set_2_name', 'set_3_name',
              'set_1_id', 'set_2_id', 'set_3_id', 'parameter']  + yr_getter('mt_fact')
df_parmt_fl_co2 = read_xlsx_table(wb, ['MONTHLY_FL_CO2'], columns=parmt_cols)
df_parmt_fl_co2 = df_parmt_fl_co2[[c for c in df_parmt_fl_co2.columns if not 'yr20' in c]]


mask_vc_fl = df_parmt_fl_co2.parameter.isin(['vc_fl'])
df_parmt_fl_co2 = pd.concat([df_parmt_fl_co2.loc[-mask_vc_fl],
                             expand_rows(df_parmt_fl_co2.loc[mask_vc_fl], ['set_2_id'],
                                  [df_def_node.nd.tolist()])], axis=0)
###############################################################################


df_parmt = pd.concat([df_parmt_fl_co2, df_parmt_cap_avlb], axis=0, sort=True)
df_parmt = df_parmt.drop(['set_3_id', 'set_3_name'], axis=1)

for idict in [dict_node_id, dict_encar_id, dict_fuel_id, dict_plant_id]:
    for icol in ['set_1_id', 'set_2_id']:
        df_parmt[icol] = df_parmt[icol].astype(str).replace(idict)

        df_parmt.loc[df_parmt[icol]=='-1.0', icol] = int(-1)


print('Write all output')
write_dfs = [
             (df_parmt, 'parameter_month'),
             ]
for idf in write_dfs:
    print('Writing ', idf[1])
    aql.write_sql(idf[0], db, sc, idf[1], 'append')





# %%

for tb in aql.get_sql_tables(sc, db):
    print(tb)
    df = aql.read_sql(db, sc, tb)

    if 'prof' in tb and 'value' in df.columns:
        df['value'] = df['value'].round(13)

    df.to_csv(os.path.join(data_path, '%s.csv'%tb), index=False)





sys.exit()

