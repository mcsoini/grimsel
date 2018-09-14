import sys
import matplotlib as mpl
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np

import grimsel_h.core.model_loop as model_loop
from grimsel_h.analysis.sql_analysis import SqlAnalysis
import grimsel_h.plotting.plotpage as pltpg
import grimsel_h.auxiliary.maps as maps

from grimsel_h.core.model_base import ModelBase as MB

# %%
## %%%%%%%%%%%%%%%%%%%%%%% INIT MODEL %%%%%%%%%%%%%%%%%%%%%%%

reload(model_loop)

# additional kwargs for the model
mkwargs = {
           'slct_encar': ['HT', 'EL', 'H2'],
           'slct_node': ['FR0', 'DE0'],
           'sc_out': 'out_dev',
           'nhours': 1,
           'tm_filt': [['doy', (1, 350, 175, 176)]],
           'constraint_groups': MB.get_constraint_groups(excl=['chp', 'hydro'])
           }

# additional kwargs for the i/o
iokwargs ={'sc_warmstart': False,
           'sc_inp': 'lp_input',
           'resume_loop': False,
           'db': 'flexible_fuels',
           'autocompletion': True
          }

nsteps_default = [
                  ('swco', 11, np.linspace),   # price CO2 emissions
                 ]
ml = model_loop.ModelLoop(nsteps=nsteps_default, dev_mode=True,
                          mkwargs=mkwargs, iokwargs=iokwargs)

# %%%%%%%%%%%%%%%%%%%%%%% RUN MODEL %%%%%%%%%%%%%%%%%%%%%%%


ml.perform_model_run(zero_run=True)

irow_0 = 0

irow = irow_0
for irow in list(range(irow_0, len(ml.df_def_loop))):
    run_id = irow

    # sets ml attributes dct_* to the values corresponding to this run_id
    ml.select_run(run_id)
    ml.set_value_co2_price_allnodes()
    ml.perform_model_run()

## %%%%%%%%%%%%%%%%%%%%%%% ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%

sqa = SqlAnalysis(sc_out='out_dev', db='flexible_fuels')
print(sqa.build_tables_plant_run(list_timescale=['']))
print(sqa.build_table_plant_run_tot_balance())

slct_run_id = [-1]
sqa = SqlAnalysis(sc_out='out_dev', db='flexible_fuels',
                  slct_run_id=slct_run_id)
print(sqa.generate_analysis_time_series(False))

print(sqa.cost_disaggregation_high_level())


# %% PLOT TIME SERIES

mps = maps.Maps('out_dev', 'flexible_fuels')

reload(pltpg)

ind_pltx = ['ca', 'nd']
ind_plty = ['pwrerg_cat']
ind_axx = ['sy']
values = ['value_posneg']

series = ['bool_out', 'pt']
table = 'out_dev' + '.analysis_time_series'

filt=[('run_id', [-1])]
post_filt = []
series_order = []

data_kw = {'filt': filt, 'post_filt': post_filt, 'data_scale': {'DMND': -1},
           'data_threshold': 1e-9, 'aggfunc': np.sum, 'series_order': series_order,}
rel_kw = {'ind_rel': False, 'relto': False, 'reltype': 'share'}

do = pltpg.PlotPageData('flexible_fuels', ind_pltx, ind_plty,
                               ind_axx, values, series,
                               table, **data_kw, **rel_kw)

do.data = do.data.fillna(0)

layout_kw = {'left': 0.1, 'right': 0.8, 'wspace': 0.1, 'hspace': 0.1,
             'bottom': 0.1}
label_kw = {'label_format':' ', 'label_subset': [-1, 60, 0],
            'label_threshold':1e-6,
            'label_ha': 'left'}
plot_kw = {'kind_def': 'StackedArea', 'edgecolor': None, 'linewidth': 0,
           'kind_dict': {'DMND': 'StepPlot'},
           'colormap': mps.get_color_dict(series[-1]), 'sharex': True,
           'marker': None, 'reset_xticklabels': False}

with plt.style.context(('ggplot')):
	plt_0 = pltpg.PlotTiled(do, **layout_kw, **label_kw, **plot_kw)

dict_val = {'erg': 'Stored Energy [MWh]', 'pwr': 'Power [MW]'}
dict_nd = {'DE0': 'Node DE0', 'FR0': 'Node FR0'}
dict_ca = {'HT': 'Heat', 'H2': 'Hydrog.', 'EL': 'Elec.'}

for (ix, nx, iy, ny, ax) in plt_0.get_ax_list():

    ax.set_ylabel(dict_val[ny[0]] if ix == 0 else '')
    ax.set_xlabel('Hours' if iy!= 0 else '')
    ax.set_title(dict_nd[nx[1]] + ', ' + dict_ca[nx[0]] if iy == 0 else '')

# %% PLOT BY CO2 PRICE

ind_pltx = ['nd', 'ca']
ind_plty = []
ind_axx = ['swco_vl']
values = ['erg_yr_sy_posneg']

series = ['bool_out', 'pt']

table = 'out_dev' + '.analysis_plant_run_tot_balance'

filt=[]
post_filt = []

data_kw = {'filt': filt, 'post_filt': post_filt, 'data_scale': {'DMND': -1},
           'data_threshold': 1e-9, 'aggfunc': np.sum}
rel_kw = {'ind_rel': False, 'relto': False, 'reltype': 'share'}

data_slct = pltpg.PlotPageData('flexible_fuels', ind_pltx, ind_plty, ind_axx, values, series,
                     table, **data_kw, **rel_kw)

data_slct.data = data_slct.data.fillna(0)

layout_kw = {'left': 0.05, 'right': 0.8, 'wspace': 0.8, 'hspace': 0.4,
             'bottom': 0.2}
label_kw = {'label_format':' ', 'label_subset': [-1], 'label_threshold':1e-6,
            'label_ha': 'left'}
plot_kw = {'kind_def': 'StackedGroupedBar', 'edgecolor': 'k', 'edgewidth': 2,
           'kind_dict': {'DMND': 'LinePlot'},
           'colormap': mps.get_color_dict(series[-1]),
           'marker': None, 'reset_xticklabels': True}

with plt.style.context(('ggplot')):
	plt_0 = pltpg.PlotTiled(data_slct, **layout_kw, **label_kw, **plot_kw)


