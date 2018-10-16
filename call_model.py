import sys

import numpy as np
from importlib import reload

import grimsel.core.model_loop as model_loop
from grimsel.core.model_base import ModelBase as MB
from grimsel.core.io import IO as IO


import grimsel.auxiliary.aux_sql_func as aql
import grimsel.core.io as io
import grimsel.analysis.sql_analysis_comp as sql_analysis_comp

import grimsel.model_loop_modifier as model_loop_modifier

import grimsel.config as config

# sc_inp currently only used to copy imex_comp and priceprof_comp to sc_out
sc_inp = config.SCHEMA
db = config.DATABASE

#sys.exit()

# %%

reload(model_loop)
reload(model_loop_modifier)
reload(config)

sqlc = aql.sql_connector(**dict(db=db,
                                password=config.PSQL_PASSWORD,
                                user=config.PSQL_USER,
                                port=config.PSQL_PORT,
                                host=config.PSQL_HOST))

# additional kwargs for the model
mkwargs = {
           'slct_encar': ['EL'],
           'slct_node': ['AT0', 'IT0', 'DE0', 'CH0', 'FR0'],
           'nhours': 1,
#           'slct_pp_type': slct_pt,
#           'skip_runs': True,
#           'tm_filt': [('wk_id', [27]), ('dow', [2])],
#           'verbose_solver': False,
           'constraint_groups': MB.get_constraint_groups(excl=['chp', 'ror'])
           }
# additional kwargs for the i/o
iokwargs = {'sc_warmstart': False,
            'resume_loop': False,
#            'autocomplete_curtailment': True
           }


nsteps_default = [
                  ('swhy', 1, np.arange),    # historic years
#                  ('swsyrs', len(list_raise_dmnd) + 1, np.arange),    # vre scaling
                  ('swcfcap', 2, np.arange),    #
                  ('swchp', 2, np.arange)
                 ]

mlkwargs = {#'sc_inp': 'lp_input_calibration_years_linonly',
            'sc_out': 'out_cal_cap_vs_cf',
            'db': db,
            'nsteps': nsteps_default,
            'sql_connector': sqlc,
            'dev_mode': False
            }

sc_out = mlkwargs['sc_out']

ml = model_loop.ModelLoop(**mlkwargs, mkwargs=mkwargs, iokwargs=iokwargs)

# init ModelLoopModifier
mlm = model_loop_modifier.ModelLoopModifier(ml)

io.IO.param_to_df(ml.m.cf_max, ('mt_id', 'pp_id', 'ca_id')).pivot_table(index='mt_id', columns='pp_id', values='value').plot()

# starting row of loop
irow_0 = ml.io.resume_loop if ml.io.resume_loop else 0

# loop over all rows of the resulting ml.df_def_loop;
# corresponding modifications to the model are performed here;
# the model run method is called at the end
irow = 0

for irow in list(range(irow_0, len(ml.df_def_loop))):
    run_id = irow

    ml.select_run(run_id)


    '''
    CAREFUL: set_historic_year doesn't apply monthly factors to cf_max etc.!
    It currently only sets the dct_vl dictionary entry.
    '''
    mlm.set_historic_year()

    mlm.availability_cf_cap()

    mlm.chp_on_off(ml.m.slct_node)

#    mlm.new_inflow_profile_for_ch()

#    mlm.chp_on_off(['DE0'])
#
#    mlm.cost_adjustment_literature()

#    mlm.raise_demand(list_raise_dmnd, 'DE0')

    #########################################
    ############### RUN MODEL ###############
    ml.m.fill_peaker_plants(demand_factor=2)

#    ml.m._limit_prof_to_cap('cap_pwr_leg')

    ml.perform_model_run()


for tb in ['profprice_comp', 'imex_comp']:
    aql.exec_sql(''' DROP TABLE IF EXISTS {sc_out}.{tb};
                 SELECT * INTO {sc_out}.{tb} FROM {sc_inp}.{tb}; '''
                 .format(tb=tb, sc_out=mlkwargs['sc_out'], sc_inp=sc_inp), db=db)

# %%

sc_out = ml.io.sc_out
sqac = sql_analysis_comp.SqlAnalysisComp(sc_out=sc_out, db=db)
sqac.analysis_production_comparison()
sqac.build_tables_plant_run()
sqac.analysis_cf_comparison()
sqac.analysis_price_comparison(valmin=-20, valmax=150, nbins=170)
sqac.build_table_plant_run_tot_balance()
sqac.analysis_production_comparison_hourly(stats_years=['2015'])
sqac.analysis_monthly_comparison()
