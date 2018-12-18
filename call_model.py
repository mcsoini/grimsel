import sys, os

import numpy as np
from importlib import reload

import grimsel.core.model_loop as model_loop
from grimsel.core.model_base import ModelBase as MB

import grimsel.auxiliary.aux_sql_func as aql
import grimsel.analysis.sql_analysis_comp as sql_analysis_comp

import grimsel.config as config
import grimsel.model_loop_modifier as model_loop_modifier

db = config.DATABASE

sc_out = 'out_replace_all'


# Note input data from grimsel default

# %%

reload(model_loop)
reload(model_loop_modifier)
reload(config)

connect_dict = dict(db=db, password=config.PSQL_PASSWORD,
                    user=config.PSQL_USER, port=config.PSQL_PORT,
                    host=config.PSQL_HOST)
sqlc = aql.sql_connector(**connect_dict)

# additional kwargs for the model
mkwargs = {
           'slct_encar': ['EL'],
           'slct_node': ['AT0', 'IT0', 'DE0', 'CH0', 'FR0'],
           'nhours': 168*2,
           'verbose_solver': True,
           'constraint_groups': MB.get_constraint_groups(excl=['chp', 'ror'])
           }
# additional kwargs for the i/o
iokwargs = {'sc_warmstart': False,
            'resume_loop': 1,
            'autocomplete_curtailment': True}

nvr, nst = 30, 31
nsteps_default = [
                  ('swvr', nvr, np.arange),      # vre scaling
                  ('swst', nst, np.arange),      # select storage capacity
                  ('swtc', 3, np.arange),       # select storage type
                  ('swpt', 3, np.arange),       # select vre type
                  ('swyr', 5, np.arange),
                  ('swco', 3, np.arange),
                  ('swrc', 3, np.arange)]

mlkwargs = {
            'sc_out': sc_out,
            'db': db,
            'nsteps': nsteps_default,
            'sql_connector': sqlc,
            }

sc_out = mlkwargs['sc_out']

ml = model_loop.ModelLoop(**mlkwargs, mkwargs=mkwargs, iokwargs=iokwargs)

# %%
# mask others
slct_vr = [0] + list(np.arange(0, ml.df_def_loop.swvr_id.max(), 4) + 1)
slct_st = list(np.arange(0, ml.df_def_loop.swvr_id.max() + 10, 10))
mask_others = (ml.df_def_loop.swvr_id.isin(slct_vr) &
               ml.df_def_loop.swst_id.isin(slct_st) &
               ml.df_def_loop.swtc_id.isin([0, 1]) &
               ml.df_def_loop.swpt_id.isin([0]) &
               ml.df_def_loop.swyr_id.isin([0]) &
               ml.df_def_loop.swco_id.isin([0]))
# mask yr
slct_vr = [17, 21, 29]
slct_st = list(np.arange(0, ml.df_def_loop.swvr_id.max() + 10, 10))
mask_yr = (ml.df_def_loop.swvr_id.isin(slct_vr) &
               ml.df_def_loop.swst_id.isin(slct_st) &
               ml.df_def_loop.swtc_id.isin([0, 1]) &
               ml.df_def_loop.swpt_id.isin([0]) &
               (-ml.df_def_loop.swyr_id.isin([0])) &
               ml.df_def_loop.swco_id.isin([0])
              )


# %%

ml.df_def_loop = ml.df_def_loop.loc[mask_others | mask_yr]


# init ModelLoopModifier
mlm = model_loop_modifier.ModelLoopModifier(ml)

# starting row of loop
irow_0 = ml.io.resume_loop if ml.io.resume_loop else 0



irow = irow = 0

ml.m._limit_prof_to_cap()
ml.perform_model_run(zero_run=True)
# %
for irow in list(range(irow_0, len(ml.df_def_loop))):
    run_id = irow

    print('select_run')
    ml.select_run(run_id)

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
    swvr_max = nvr - 1
    dict_vre = {nvr: vr for nvr , vr
                in enumerate(['default'] + list(np.linspace(0, 0.7, swvr_max)))}
    print('scale_vre')
    mlm.scale_vre(slct_pt, dict_vre)

    ####
    print('select_storage_tech')
    mlm.select_storage_tech()

    ####
    print('set_storage_cap')
    swst_max = nst
    dict_st = {nst: st for nst, st in enumerate(list(np.linspace(0, 0.3, swst_max)))}
    mlm.set_storage_cap(dict_st)

    #########################################
    ############### RUN MODEL ###############
    print('fill_peaker_plants')
    ml.m.fill_peaker_plants(demand_factor=2)

    print('_limit_prof_to_cap')
    ml.m._limit_prof_to_cap()

    print('perform_model_run')
    ml.perform_model_run()

    for fn in [fn for fn in os.listdir('.') if 'ephemeral' in fn or 'def_loop' in fn]:
        os.remove(fn)


for tb in ['profprice_comp', 'imex_comp']:
    aql.exec_sql(''' DROP TABLE IF EXISTS {sc_out}.{tb};
                 SELECT * INTO {sc_out}.{tb} FROM {sc_inp}.{tb}; '''
                 .format(tb=tb, sc_out=mlkwargs['sc_out'], sc_inp=sc_inp), db=db)

# %

sc_out = sc_out
sqac = sql_analysis_comp.SqlAnalysisComp(sc_out=sc_out, db=db)
self = sqac
sqac.analysis_cost_disaggregation_lin()
#sqac.analysis_production_comparison()
sqac.build_tables_plant_run_quick()
sqac.build_tables_plant_run(list_timescale=[''])
sqac.build_table_plant_run_tot_balance(from_quick=True)
#sqac.analysis_cf_comparison()
#sqac.analysis_price_comparison(valmin=-20, valmax=150, nbins=170)
#sqac.analysis_production_comparison_hourly(stats_years=['2015'])
#sqac.analysis_monthly_comparison()
sqac.analysis_chp_shares()

sys.exit()


