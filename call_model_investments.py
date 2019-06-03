
import sys, os

import numpy as np
import pandas as pd
import itertools

import grimsel.core.model_loop as model_loop
from grimsel.core.model_base import ModelBase as MB
import grimsel.auxiliary.aux_sql_func as aql
import grimsel.config as config
import grimsel.model_loop_modifier as model_loop_modifier
import grimsel.analysis.sql_analysis as sql_analysis
import grimsel.auxiliary.maps as maps

import grimsel

db = config.DATABASE

sc_out = 'out_replace_all_investment'

#sys.exit()

# %% APPEND PLANTS WITH INVESTMENTS

data_path = os.path.join(grimsel.__path__[0], 'input_data')


df_def_pp_type = pd.read_csv(os.path.join(data_path, 'def_pp_type.csv')).query('pt != "GAS_NEW"')
df_def_plant = pd.read_csv(os.path.join(data_path, 'def_plant.csv')).query('"NGCC" not in pp')
df_def_fuel = pd.read_csv(os.path.join(data_path, 'def_fuel.csv'))
df_def_node = pd.read_csv(os.path.join(data_path, 'def_node.csv'))
df_plant_encar = pd.read_csv(os.path.join(data_path, 'plant_encar.csv'))
df_plant_encar = df_plant_encar.loc[df_plant_encar.pp_id.isin(df_def_plant.pp_id), [c for c in df_plant_encar.columns if not c == 'fc_cp_ann' and not 'Unnamed' in c]]

row_pt = {'color': ['#7F5299'], 'pp_broad_cat': ['CONVDISP'], 'pt': ['GAS_NEW'], 'pt_id': [df_def_pp_type.pt_id.max() + 1]}
df_def_pp_type_add = pd.DataFrame(row_pt)

pp_id_max = df_def_plant.pp_id.max()
set_1 = ['set_def_add', 'set_def_pp']

fl_id_ng = df_def_fuel.set_index('fl').fl_id.loc['natural_gas']

def_plant_add = pd.DataFrame(
{'fl_id': [fl_id_ng] * 5,
 'nd_id': df_def_node.nd_id,
 'pp': df_def_node.nd,
 'pp_id': range(pp_id_max + 1, pp_id_max + 6),
 'pt_id': row_pt['pt_id'] * 5
 }).assign(**{col: 1 if col in set_1 else 0 for col in df_def_plant.columns if 'set_def_' in col})
def_plant_add['pp'] = def_plant_add.pp.apply(lambda x: x.replace('0', '') + '_CCGT_NEW')


lifetime = 30
discount_rate = 0.06
fc_cp = 1e6
annuity_factor = ((1 + discount_rate) ** lifetime * discount_rate 
                  / ((1 + discount_rate) ** lifetime - 1))  # 1/year
fc_cp_ann = fc_cp * annuity_factor

ppca_row = {'ca_id': 0,
 'cap_pwr_leg': 0,
 'cf_max': np.nan,
 'discharge_duration': np.nan,
 'erg_max': np.nan,
 'factor_vc_co2_lin_0': np.nan,
 'factor_vc_co2_lin_1': np.nan,
 'factor_vc_fl_lin_0': np.nan,
 'factor_vc_fl_lin_1': np.nan,
 'fc_om': 10000,
 'pp_eff': 0.6,
 'pp_id': np.nan,
 'st_lss_hr': np.nan,
 'st_lss_rt': np.nan,
 'vc_om': 3,
 'vc_ramp': 11.425000,
 'vc_ramp_high': 15.150000,
 'vc_ramp_low': 7.700000,
 'fc_cp_ann': fc_cp_ann
}

df_plant_encar_add = pd.concat([pd.DataFrame({col: [val] for col, val in ppca_row.items()})] * 5).reset_index(drop=True).assign(pp_id=def_plant_add.pp_id)



df_def_plant = pd.concat([df_def_plant.drop_duplicates(), def_plant_add]).reset_index(drop=True)
df_def_pp_type = pd.concat([df_def_pp_type.drop_duplicates(), df_def_pp_type_add]).reset_index(drop=True)
df_plant_encar = pd.concat([df_plant_encar.drop_duplicates().assign(fc_cp_ann=np.nan), df_plant_encar_add]).reset_index(drop=True)

df_def_plant[[c for c in df_def_plant.columns if not 'Unnamed' in c]].to_csv(os.path.join(data_path, 'def_plant.csv'), index=False)
df_def_pp_type[[c for c in df_def_pp_type.columns if not 'Unnamed' in c]].to_csv(os.path.join(data_path, 'def_pp_type.csv'), index=False)
df_plant_encar[[c for c in df_plant_encar.columns if not 'Unnamed' in c]].to_csv(os.path.join(data_path, 'plant_encar.csv'), index=False)



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
           'verbose_solver': True,
           'constraint_groups': MB.get_constraint_groups(excl=['chp', 'ror'])
           }

# additional kwargs for the i/o
iokwargs = {'resume_loop': 722,
            'autocomplete_curtailment': True,
           }

nvr, nst = 30, 31
nsteps_default = [('swnw', 4, np.arange),       # new capacity
                  ('swvr', nvr, np.arange),     # select vre share
                  ('swst', nst, np.arange),     # select storage capacity
                  ('swtc', 3, np.arange),       # select storage tech
                  ('swpt', 3, np.arange),       # select vre type
                  ('swyr', 5, np.arange),       # select meteo year
                  ('swco', 3, np.arange),       # select co2 emission price
                  ('swtr', 2, np.arange),       # cross-border transmission on/off
                  ('swrc', 26, np.arange),      # select ramping cost
                  ]

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
dict_rc = {**{ii: ii/10 for ii in range(26)}, **{0: 1, 10: 0}}


# %%

ml.io.var_sy = []#[par for par in ml.io.var_sy if 'pwr' in par[0] or 'pwr_st_ch' in par[0]]
ml.io.par = [par for par in ml.io.par if not 'prof' in par[0]]
ml.io.var_tr = [var for var in ml.io.var_tr if 'erg' in var[0]]
ml.io.dual = []
print(ml.io.var_sy)


# %%

# figure 14: emissions
slct_vr = list(np.arange(0, max([vrid for vrid, vrvl in dict_vre.items() if not vrvl == 'default' and vrvl <= 0.45]), 2) + 1)
slct_st = [0, 20]
mask_emissions = (ml.df_def_loop.swvr_id.isin(slct_vr) &
                ml.df_def_loop.swst_id.isin(slct_st) &
                ml.df_def_loop.swtc_id.isin([0, 1]) &
                ml.df_def_loop.swpt_id.isin([0]) &
                ml.df_def_loop.swyr_id.isin([0]) &
#                ml.df_def_loop.swco_id.isin([0]) & --> all
                ml.df_def_loop.swtr_id.isin([0]) &
                ml.df_def_loop.swrc_id.isin([0])
#                ml.df_def_loop.swnw_id.isin([0, 1]) 
                )

# figure 13: consecutive replacement
slct_vr = list(np.arange(0, max([vrid for vrid, vrvl in dict_vre.items() if not vrvl == 'default' and vrvl <= 0.425]), 1) + 1)
slct_st = [0, 20]#list(np.arange(0, ml.df_def_loop.swvr_id.max() + 5, 10))
mask_consec = (ml.df_def_loop.swvr_id.isin(slct_vr) &
                ml.df_def_loop.swst_id.isin(slct_st) &
                ml.df_def_loop.swtc_id.isin([0]) &
                ml.df_def_loop.swpt_id.isin([0, 1, 2]) &
                ml.df_def_loop.swyr_id.isin([0]) &
                ml.df_def_loop.swco_id.isin([0, 1]) &
                ml.df_def_loop.swtr_id.isin([0]) &
                ml.df_def_loop.swrc_id.isin([0])
#                ml.df_def_loop.swnw_id.isin([0])
                )

## figure 12: qualitatively diverging storage impact
#slct_vr = [{round(val * 10000) / 10000 if not val == 'default' else val: key
#            for key, val in dict_vre.items()}[vr] for vr in [0.5, 0.7, 0.4]]
#mask_years = (ml.df_def_loop.swvr_id.isin(slct_vr) &
##                ml.df_def_loop.swst_id.isin(slct_st) & --> all
#                ml.df_def_loop.swtc_id.isin([0, 1]) &
#                ml.df_def_loop.swpt_id.isin([0]) &
##                ml.df_def_loop.swyr_id.isin([0]) & --> all
#                ml.df_def_loop.swco_id.isin([0]) &
#                ml.df_def_loop.swtr_id.isin([0]) &
#                ml.df_def_loop.swrc_id.isin([0]) &
#                ml.df_def_loop.swnw_id.isin([0]) 
#                )

## figure 11: ramping costs
#slct_vr = [{round(val * 10000) / 10000 if not val == 'default' else val: key
#            for key, val in dict_vre.items()}[vr] for vr in [0.3, 0.5, 0.7]]
#slct_st = [0, 20]#list(np.arange(0, ml.df_def_loop.swvr_id.max() + 10, 10))
#mask_ramping = (ml.df_def_loop.swvr_id.isin(slct_vr) &
#                ml.df_def_loop.swst_id.isin(slct_st) &
#                ml.df_def_loop.swtc_id.isin([0]) &
#                ml.df_def_loop.swpt_id.isin([0]) &
#                ml.df_def_loop.swyr_id.isin([0]) &
#                ml.df_def_loop.swco_id.isin([0]) &
#                ml.df_def_loop.swtr_id.isin([0]) &
#                ml.df_def_loop.swnw_id.isin([0]) 
##               ml.df_def_loop.swrc_id.isin([0]) --> all
#               )


# figures 9+10: various nuclear power indicators france 
slct_vr = [0] + list(np.arange(0, ml.df_def_loop.swvr_id.max(), 4) + 1)
slct_st = list(np.arange(0, ml.df_def_loop.swvr_id.max() + 10, 10))
mask_base = (ml.df_def_loop.swvr_id.isin(slct_vr) &
               ml.df_def_loop.swst_id.isin(slct_st) &
               ml.df_def_loop.swtc_id.isin([0, 1]) &
               ml.df_def_loop.swpt_id.isin([0]) &
               ml.df_def_loop.swyr_id.isin([0]) &
               ml.df_def_loop.swco_id.isin([0]) &
               ml.df_def_loop.swtr_id.isin([0]) &
               ml.df_def_loop.swrc_id.isin([0])
#               ml.df_def_loop.swnw_id.isin([0, 1])
               )

mask_total = mask_base | mask_consec | mask_emissions

mask_total.sum()

# %

ml.df_def_loop = pd.concat([ml.df_def_loop.loc[mask_total]])
ml.df_def_loop


# %%


from importlib import reload

reload(model_loop_modifier)

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


    #######

    print('select_run')
    ml.select_run(run_id)


    ####### new investments    
    ml.m.cap_pwr_new.unfix()
    for key in ml.m.fc_cp_ann:
        ml.m.fc_cp_ann[key] = fc_cp_ann
        
    dict_mult = {1: 1, 2: 1.2, 3: 0.8}
    if ml.dct_step['swnw'] == 0:
        for key in ml.m.cap_pwr_new:
            ml.m.cap_pwr_new[key].value = 0
            ml.m.cap_pwr_new.fix()
    else:
        ml.m.fc_cp_ann[key] = ml.m.fc_cp_ann[key].value * dict_mult[ml.dct_step['swnw']]
    ml.dct_vl['swnw_vl'] = '%.1f'%dict_mult[ml.dct_step['swnw']] if ml.dct_step['swnw'] else 'none'
        
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
    mlm.set_ramping_cost(dict_rc)

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
        try:
            os.remove(fn)
        except:
            pass


sqa = sql_analysis.SqlAnalysis(sc_out=sc_out, db=db)
print(sqa.build_tables_plant_run())
print(sqa.build_table_plant_run_tot_balance(from_quick=True))

table_list = aql.get_sql_tables(sc_out, db)
table_list = [t for t in table_list if 'analysis' in t or 'def_' in t]

aql.dump_by_table(sc_out, db, tables=table_list, target_dir='C:\\Users\\ashreeta\\Documents\\Martin\\SWITCHdrive\\SQL_DUMPS\\out_replace_all_investment\\')

