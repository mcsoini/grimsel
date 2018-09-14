
import time
import numpy as np
from importlib import reload

from grimsel.core.model_loop import ModelLoop

from grimsel.auxiliary.aux_m_func import pdef
from grimsel.auxiliary.maps import Maps
import grimsel.auxiliary.aux_sql_func as aql
from grimsel.auxiliary.aux_general import print_full

from grimsel.analysis.sql_analysis import SqlAnalysis
from grimsel.analysis.sql_analysis_hourly import SqlAnalysisHourly

import grimsel.plotting.plotpage as pltpg

# %% SET UP MODEL

# additional kwargs for the model
mkwargs = {'slct_encar': ['EL'],
           'slct_node': ['DE0'],#, 'AT0', 'CH0', 'IT0'],
           'sc_out': 'out_grimsel_example',
           'nhours': 12}
# additional kwargs for the i/o
iokwargs ={'sc_warmstart': False,
           'sc_inp': 'lp_input_2',
           'resume_loop': False,
           'db': 'storage1'}

nsteps = [('swvr', 6, np.linspace), # loop over variable renewable penetration
          ('swco', 3, np.arange)] # loop over CO2 emissions price
ml = ModelLoop(nsteps=nsteps, mkwargs=mkwargs, iokwargs=iokwargs)

# %%
#####################################################################
########## initial run without investments for calibration ##########
ml.perform_model_run(zero_run=True)
########## initial run without investments for calibration ##########
#####################################################################
# %%

# starting row of loop
irow_0 = 0

irow = irow_0
for irow in list(range(irow_0, len(ml.df_def_loop))):
    run_id = irow

    # sets ml attributes dct_* to the values corresponding to this run_id
    ml.select_run(run_id)

    ml.set_value_co2_price()
    ml.set_variable_renewable_penetration(0.5, zero_is_status_quo=False)

    ml.m.fill_peaker_plants()

    ml.perform_model_run()

ml.io.post_process_index(ml.m.sc_out, ml.io.db)

# %% Analysis -> produces table public.plant_run_tot

db = 'storage1'
sc_out = 'out_grimsel_example'


sqa = SqlAnalysis(sc_out=sc_out, db='storage1')
print(sqa.build_tables_plant_run(list_timescale=['']))

# Columns of resulting table
aql.read_sql(db, sc_out, 'plant_run_tot').columns.tolist()

slct_run_id = [17]
sqa = SqlAnalysis(sc_out=sc_out, db=db,
                  slct_run_id=slct_run_id)
print(sqa.generate_analysis_time_series(False))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
################    PLOTTING    ###############################################

# %% PLANT_RUN: production by model run

import matplotlib.pyplot as plt

import grimsel.plotting.plotpage as pltpg
import grimsel.auxiliary.maps as maps

mps = Maps(sc_out, db)

reload(pltpg)

ind_pltx = ['nd']
ind_plty = ['swco_vl']
ind_axx = ['swvr_vl']
values = ['erg_yr_sy']

series = ['bool_out', 'pt']
table = sc_out + '.plant_run_tot'

filt=[
#      ('pt', ['DMND', 'HYD_STO'], '<>', ' AND '),
      ('run_id', [-1], '>')
     ]
post_filt = []

data_kw = {'filt': filt, 'post_filt': post_filt, 'data_scale': {'DMND': 1},
           'data_threshold': 1e-9, 'aggfunc': np.sum}
rel_kw = {'ind_rel': False, 'relto': False, 'reltype': 'share'}

data_slct = pltpg.PlotPageData(db, ind_pltx, ind_plty, ind_axx, values, series,
                     table, **data_kw, **rel_kw)

data_slct.data = data_slct.data.fillna(0)

layout_kw = {'left': 0.1, 'right': 0.8, 'wspace': 0.8, 'hspace': 0.4,
             'bottom': 0.1}
label_kw = {'label_format':' ', 'label_subset':[-1], 'label_threshold':1e-6,
            'label_ha': 'left'}
plot_kw = {'kind_def': 'StackedGroupedBar',
           'kind_dict': {'DMND': 'LinePlot'},
           'colormap': mps.color_pt,
           'marker': 'd', 'reset_xticklabels': True,
           'ylabel': 'El. production mix [MWh]',
           'xlabel': 'Wind+solar penetration level [a.u.]',
           }

with plt.style.context(('ggplot')):
	plt_0 = pltpg.PlotTiled(data_slct, **layout_kw, **label_kw, **plot_kw)

# %% How does the output from fossil fuel plants change for increasing
#    solar+wind penetration under different CO2 emission prices?

import grimsel.plotting.plotpage as pltpg
reload(pltpg)

ind_pltx = ['nd']
ind_plty = None
ind_axx = ['swvr_vl']
values = ['erg_yr_sy']

series = ['swco_vl','sf']
table = sc_out + '.plant_run_tot'

filt=[
      ('pt', ['%NUC%', '%HCO%', '%GAS%'], ' LIKE ', ' OR '),
      ('run_id', [-1], '>')
     ]
post_filt = []

data_kw = {'filt': filt, 'post_filt': post_filt, 'data_scale': 100,
           'data_threshold': 1e-9, 'aggfunc': np.sum}
rel_kw = {'ind_rel': 'swvr_vl', 'relto': '00.0%', 'reltype': 'share'}

data_slct = pltpg.PlotPageData(db, ind_pltx, ind_plty, ind_axx, values, series,
                     table, **data_kw, **rel_kw)

layout_kw = {'left': 0.1, 'right': 0.7, 'wspace': 0.1, 'hspace': 0.5,
             'bottom': 0.12}
label_kw = {'label_format':' ', 'label_subset':[-1], 'label_threshold':1e-6,
            'label_ha': 'left'}
plot_kw = {'kind_def': 'LinePlot', 'stacked': False,
           'colormap': mps.get_color_dict(series[-1]),
           'marker': 'd', 'reset_xticklabels': False,
           'ylabel': 'Relative change of el production [%]',
           'xlabel': 'Wind+solar penetration level [a.u.]',
           }
legend_kw = {'legend_pos': [0, 0]}

with plt.style.context(('ggplot')):
	plt_0 = pltpg.PlotTiled(data_slct, **layout_kw, **label_kw, **plot_kw)


# %% PLANT_RUN: value by CO2 price and wind+solar penetration

ind_pltx = ['swco_vl']
ind_plty = ['nd']
ind_axx = ['swvr_vl']
values = ['val_yr_0_cap']

series = ['pt']
table = sc_out + '.plant_run_tot'

filt=[
#      ('pt', ['DMND', 'HYD_STO'], '<>', ' AND '),
      ('bool_out', ['False']),
      ('run_id', [-1], '>')
     ]
post_filt = []

data_kw = {'filt': filt, 'post_filt': post_filt, 'data_scale': 100,
           'data_threshold': 1e-9, 'aggfunc': np.sum}
rel_kw = {'ind_rel': 'swvr_vl', 'relto': '10.0%', 'reltype': 'share'}

data_slct = pltpg.PlotPageData(db, ind_pltx, ind_plty, ind_axx, values, series,
                     table, **data_kw)

layout_kw = {'left': 0.1, 'right': 0.8, 'wspace': 0.8, 'hspace': 0.4}
label_kw = {'label_format':' ', 'label_subset':[-1], 'label_threshold':1e-6,
            'label_ha': 'left'}
plot_kw = {'kind_def': 'LinePlot', 'colormap': mps.color_pt,
           'marker': 'd', 'reset_xticklabels': True, 'sharey': 'col',
		   }
legend_kw = {'legend_pos': [0, 0]}

with plt.style.context(('ggplot')):
	plt_0 = pltpg.PlotTiled(data_slct, **layout_kw, **label_kw, **plot_kw)




# %% AVERAGE DAYS SELECTED MONTHS
import grimsel.plotting.plotpage as pltpg

ind_pltx = ['pwrerg_cat'] #['mt_id']
ind_plty = ['nd']
ind_axx = ['sy'] #['how']
values = ['value_posneg']
post_filt = []

series = ['bool_out', 'pt']
table = sc_out + '.analysis_time_series'

data_threshold = 1e-1
filt=[
#      ('mt_id', [0, 7])
      ]

data_kw = {'filt': filt, 'post_filt': post_filt, 'data_scale': {'DMND': -1},
           'data_threshold': 1e-9, 'aggfunc': np.mean,
           'series_order': mps.pp_type_order}

do = pltpg.PlotPageData(db, ind_pltx, ind_plty, ind_axx, values, series,
                               table, **data_kw)

do.data = do.data.fillna(0)

layout_kw = {'left': 0.1, 'right': 0.8, 'wspace': 0.1, 'hspace': 0.1,
             'bottom': 0.1}
label_kw = {'label_format':' ', 'label_subset':[-1], 'label_threshold':1e-6,
            'label_ha': 'left'}
plot_kw = {'kind_def': 'StackedArea', 'colormap': mps.color_pt,
           'kind_dict': {'DMND': 'StepPlot'},
           'sharey': 'col', 'sharex': True,
           'marker': 'd', 'edgecolor': None, 'reset_xticklabels': False}
legend_kw = {'legend_pos': [0, 0]}

plt_0 = pltpg.PlotTiled(do, **layout_kw, **label_kw, **plot_kw)


# WEEKLY AVERAGES FOR SELECTED MONTHS

do_1 = do.copy()

do_1.ind_pltx = ['mt_id'] #['mt_id']
do_1.ind_plty = ['pwrerg_cat'] #['mt_id']
do_1.ind_axx = ['how'] #['how']

do_1.filt=[
           ('mt_id', [0, 4, 8])
          ]

do_1.series = ['bool_out', 'pt']
do_1.update()
do_1.data = do_1.data.fillna(0)

plot_kw['colormap'] =  mps.color_pt
plot_kw['sharey'] =  'row'

plt_1 = pltpg.PlotTiled(do_1, **layout_kw, **label_kw, **plot_kw)





