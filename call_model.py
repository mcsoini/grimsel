import sys, os

import numpy as np
import pandas as pd
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

import pyAndy.core.plotpage as pltpg
import grimsel.auxiliary.maps as maps
mps = maps.Maps(sc_inp, db)


# %%

reload(model_loop)
reload(model_loop_modifier)
reload(config)

sqlc = aql.sql_connector(**dict(db=db,
                                password=config.PSQL_PASSWORD,
                                user=config.PSQL_USER,
                                port=config.PSQL_PORT,
                                host=config.PSQL_HOST))


slct_pt = pd.read_csv(os.path.join(config.PATH_CSV, 'def_pp_type.csv'))
slct_pt = slct_pt.loc[-slct_pt.pt.str.contains('|'.join(['SOL', 'LOL', 'WIN', 'NUC', 'HCO', 'HYD', 'LIG', 'GEO', 'WAS', 'BAL', 'OIL']))]
slct_pt = slct_pt.pt.tolist()


# additional kwargs for the model
mkwargs = {
           'slct_encar': ['EL'],
           'slct_node': ['DE0', 'FR0', 'IT0'],#, 'CH0', 'FR0', 'AT0'],#, 'DE0', 'CH0', 'FR0'],
           'nhours': 1,
           'slct_pp_type': slct_pt,
#           'skip_runs': True,
           'tm_filt': [('wk', [20]), ('dow', [4,5,6])],
#           'verbose_solver': False,
           'constraint_groups': MB.get_constraint_groups(excl=['chp', 'ror',
                                                               'hydro',
                                                               'monthly_total',
                                                               'chp_new'])
           }
# additional kwargs for the i/o
iokwargs = {'sc_warmstart': False,
            'resume_loop': False,
            'no_output': True,
#            'autocomplete_curtailment': True
           }



nsteps_default = [
                  ('swhy', 1, np.arange),    # historic years
#                  ('swsyrs', len(list_raise_dmnd) + 1, np.arange),    # vre scaling
#                  ('swcfcap', 2, np.arange),    #
#                  ('swchp', 2, np.arange)
                 ]

mlkwargs = {#'sc_inp': 'lp_input_calibration_years_linonly',
            'sc_out': 'out_cal',
            'db': db,
            'nsteps': nsteps_default,
            'sql_connector': sqlc,
            'dev_mode': True
            }

sc_out = mlkwargs['sc_out']

ml = model_loop.ModelLoop(**mlkwargs, mkwargs=mkwargs, iokwargs=iokwargs)

self = ml.m
# %

if new:

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMBINE transmission dirs
    df = self.df_node_connect

    df['dir'] = df.nd_id < df.nd_2_id

    df_e = df.loc[df.dir].assign(nd_id = self.df_node_connect.nd_2_id,
                                        nd_2_id = self.df_node_connect.nd_id,
                                        cap_trmi_leg = self.df_node_connect.cap_trm_leg)
    dfn = pd.concat([df_e,
                     df.loc[-df.dir].assign(cap_trme_leg = self.df_node_connect.cap_trm_leg)],
                     sort=False)
    dfn = dfn.drop('cap_trm_leg', axis=1).fillna(0)
    dfn = dfn.pivot_table(index=['nd_id', 'nd_2_id', 'ca_id', 'mt_id'],
                    values=['cap_trme_leg', 'cap_trmi_leg'], aggfunc=sum)

    self.df_node_connect = dfn.reset_index()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEFINE reduced ndcnn set

    from grimsel.auxiliary.aux_m_func import cols2tuplelist
    import pyomo.environ as po

    ml.m.delete_component('ndcnn')

    df = self.df_node_connect[['nd_id', 'nd_2_id', 'ca_id']]
    self.ndcnn = po.Set(within=self.nd * self.nd * self.ca,
                        initialize=cols2tuplelist(df), ordered=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEFINE cap_trmi/e_leg

    ml.m.delete_component('cap_trm_leg')
    ml.m.delete_component('cap_trme_leg')
    ml.m.delete_component('cap_trmi_leg')

    self.padd('cap_trme_leg', (self.mt, self.ndcnn,), 'df_node_connect') # Cross-node transmission capacity.
    self.padd('cap_trmi_leg', (self.mt, self.ndcnn,), 'df_node_connect') # Cross-node transmission capacity.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEFINE TRM VARIABLE

    ml.m.delete_component('trm_sd')
    ml.m.delete_component('trm_rv')
    ml.m.delete_component('trm')

    self.vadd('trm', (self.sy, self.ndcnn), (None, None))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW TRM CONSTRAINTS

    self.delete_component('trm_symm') # drop
    self.delete_component('yrtrsd') # drop
    self.delete_component('yrtrrv') # drop

    self.delete_component('trm_sd_capac_rule')
    self.delete_component('trm_rv_capac_rule')
    self.delete_component('trme_capac')
    self.delete_component('trmi_capac')

    for sy, nd1, nd2, ca in self.trm:

        mt = self.dict_soy_month[sy]

        self.trm[(sy, nd1, nd2, ca)].setub(self.cap_trme_leg[mt, nd1, nd2, ca])
        self.trm[(sy, nd1, nd2, ca)].setlb(-self.cap_trmi_leg[mt, nd1, nd2, ca])




#self.trme_capac.display()
#self.trmi_capac.display()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW TRM CONSTRAINTS


    from grimsel.auxiliary.aux_m_func import set_to_list

    self.delete_component('supply')

    def supply_rule(self, sy, nd, ca):

        prod = (# power output; negative if energy selling plant
                sum(self.pwr[sy, pp, ca]
                    * (-1 if pp in self.setlst['sll']
                                 + self.setlst['curt']
                                 else 1)
                    for (pp, nd, ca)
                    in set_to_list(self.ppall_ndca,
                                   [None, nd, ca]))
                + sum(self.trm[sy, nd, nd_2, ca] for (nd, nd_2, ca)
                      in set_to_list(self.ndcnn, [None, nd, ca]))

                # incoming inter-node transmission
    #            + sum(self.trm_rv[sy, nd, nd_2, ca] for (nd, nd_2, ca)
    #                  in set_to_list(self.ndcnn, [nd, None, ca]))
               )
        dmnd = (self.dmnd[sy, nd, ca]
                + sum(self.trm[sy, nd, nd_2, ca] for (nd, nd_2, ca)
                      in set_to_list(self.ndcnn, [nd, None, ca]))
                + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                      in set_to_list(self.st_ndca, [None, nd, ca]))
                # demand of plants using ca as an input
                + sum(self.pwr[sy, pp, ca_out] / self.pp_eff[pp, ca_out]
                      for (pp, nd, ca_out, ca)
                      in set_to_list(self.pp_ndcaca,
                                     [None, nd, None, ca]))
                )
        return prod * (1 - self.grid_losses[nd, ca]) == dmnd
    self.supply = po.Constraint(self.sy, self.ndca, rule=supply_rule)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SET MINIMUM DEMAND TO ZERO

df_dmnd = IO.param_to_df(ml.m.dmnd, ('sy', 'nd_id', 'ca_id'))

offs = df_dmnd.groupby('nd_id')['value'].min().rename('value_min')
df_dmnd = df_dmnd.join(offs, on=offs.index.name)
df_dmnd = df_dmnd.assign(value=df_dmnd.value - df_dmnd.value_min)
dict_dmnd = df_dmnd.set_index(['sy', 'nd_id', 'ca_id']).value.to_dict()

for key, val in dict_dmnd.items():

    ml.m.dmnd[key].value = val

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SET GRID LOSSES TO ZERO

for key in ml.m.grid_losses:

    ml.m.grid_losses[key] = 0





# %

# init ModelLoopModifier
mlm = model_loop_modifier.ModelLoopModifier(ml)

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
#    mlm.set_historic_year()

#    mlm.availability_cf_cap()

#    mlm.chp_on_off(ml.m.slct_node)

#    mlm.new_inflow_profile_for_ch()

#    mlm.chp_on_off(['DE0'])
#
#    mlm.cost_adjustment_literature()

#    mlm.raise_demand(list_raise_dmnd, 'DE0')

    #########################################
    ############### RUN MODEL ###############

    ml.m.setlst['peak'] = ml.m.setlst['pp']
    ml.m.fill_peaker_plants(demand_factor=20)

    ml.m._limit_prof_to_cap()

    ml.perform_model_run()

# %

#df_pwr_ch = ml.io.variab_to_df(ml.m.pwr_st_ch, ('sy', 'pp_id', 'ca_id'))[0]
#df_pwr_ch = df_pwr_ch.join(ml.m.df_def_plant.set_index('pp_id')[['pp', 'pt_id', 'nd_id', 'fl_id']], 'pp_id')
#df_pwr_ch = df_pwr_ch.join(ml.m.df_def_node.set_index('nd_id')['nd'], 'nd_id')
#df_pwr_ch = df_pwr_ch.join(ml.m.df_def_pp_type.set_index('pt_id')['pt'], 'pt_id')
#df_pwr_ch['bool_out'] = True

dict_dmnd = ml.m.df_def_plant.loc[ml.m.df_def_plant.pp.str.endswith('DMND')].set_index('nd_id')['pp_id'].to_dict()

df_dmnd = ml.io.param_to_df(ml.m.dmnd, ('sy', 'nd_id', 'ca_id'))
df_dmnd['pp_id'] = df_dmnd.nd_id.replace(dict_dmnd)
df_dmnd = df_dmnd.join(ml.m.df_def_plant.set_index('pp_id')[['pp', 'pt_id', 'fl_id']], 'pp_id')
df_dmnd = df_dmnd.join(ml.m.df_def_node.set_index('nd_id')['nd'], 'nd_id')
df_dmnd = df_dmnd.join(ml.m.df_def_pp_type.set_index('pt_id')['pt'], 'pt_id')
df_dmnd['bool_out'] = True

df_pwr = ml.io.variab_to_df(ml.m.pwr, ('sy', 'pp_id', 'ca_id'))[0]
df_pwr = df_pwr.join(ml.m.df_def_plant.set_index('pp_id')[['pp', 'pt_id', 'nd_id', 'fl_id']], 'pp_id')
df_pwr = df_pwr.join(ml.m.df_def_node.set_index('nd_id')['nd'], 'nd_id')
df_pwr = df_pwr.join(ml.m.df_def_pp_type.set_index('pt_id')['pt'], 'pt_id')
df_pwr['bool_out'] = False


if new:
    df_trm = ml.io.variab_to_df(ml.m.trm, ('sy', 'nd1', 'nd2', 'ca_id'))[0]
    df_trm = pd.merge(df_trm, ml.m.df_def_node[['nd_id', 'nd']], left_on='nd1', right_on='nd_id').rename(columns={'nd': 'nd_1'})
    df_trm = pd.merge(df_trm, ml.m.df_def_node[['nd_id', 'nd']], left_on='nd2', right_on='nd_id').rename(columns={'nd': 'nd_2'})

    df_trm['pt'] = df_trm[['nd_1', 'nd_2']].apply(lambda x: '_'.join(x), axis=1)

    df_trm = pd.concat([df_trm[['sy', 'pt', 'nd_1', 'value']].assign(value=-df_trm.value).rename(columns={'nd_1': 'nd'}),
                        df_trm[['sy', 'pt', 'nd_2', 'value']].rename(columns={'nd_2': 'nd'}),
                        ])
    df_trm.loc[df_trm.value < 0, 'bool_out'] = True
    df_trm.loc[df_trm.value >= 0, 'bool_out'] = False

    df_trm['value'] = df_trm.value.abs()
else:

    df_trm_sd = ml.io.variab_to_df(ml.m.trm_sd, ('sy', 'nd1', 'nd2', 'ca_id'))[0]
    df_trm_sd = pd.merge(df_trm_sd, ml.m.df_def_node[['nd_id', 'nd']], left_on='nd1', right_on='nd_id').rename(columns={'nd': 'nd_1'})
    df_trm_sd = pd.merge(df_trm_sd, ml.m.df_def_node[['nd_id', 'nd']], left_on='nd2', right_on='nd_id').rename(columns={'nd': 'nd_2'})

    df_trm_rv = ml.io.variab_to_df(ml.m.trm_rv, ('sy', 'nd1', 'nd2', 'ca_id'))[0]
    df_trm_rv = pd.merge(df_trm_rv, ml.m.df_def_node[['nd_id', 'nd']], left_on='nd1', right_on='nd_id').rename(columns={'nd': 'nd_1'})
    df_trm_rv = pd.merge(df_trm_rv, ml.m.df_def_node[['nd_id', 'nd']], left_on='nd2', right_on='nd_id').rename(columns={'nd': 'nd_2'})

    df_trm_rv['pt'] = df_trm_rv[['nd_2', 'nd_1']].apply(lambda x: '_'.join(x), axis=1)
    df_trm_sd['pt'] = df_trm_sd[['nd_1', 'nd_2']].apply(lambda x: '_'.join(x), axis=1)

    df_trm_rv['bool_out'] = False
    df_trm_sd['bool_out'] = True

    df_trm = pd.concat([df_trm_rv, df_trm_sd], axis=0)

    df_trm = df_trm[['sy', 'pt', 'nd_1', 'bool_out', 'value']].rename(columns={'nd_1': 'nd'})

# %
df_tot = pd.concat([df_pwr[df_trm.columns],
                    df_dmnd[df_trm.columns],
                    df_trm], axis=0, sort=False)

df_tot.loc[df_tot.bool_out, 'value'] *= -1

series_order = ['IT0_FR0', 'FR0_DE0', 'FR0_IT0', 'DE0_FR0',
                'NUC_ELC', 'HCO_LIN', 'WIN_ONS', ]

df_tot = df_tot.loc[-df_tot.pt.str.contains('NUC|HCO|LOL|GAS|DMND')]

do = pltpg.PlotPageData.from_df(df_tot, [], [],
                           ['sy'], ['value'], ['bool_out', 'pt'],
                           series_order=series_order, harmonize=False,
                           totals={'total': ['all']})

colormap = False#mps.get_color_dict('pt')6

plot_kws = dict(kind_def='StackedArea',
                kind_dict={'total': 'LinePlot'}, marker='o',
                sharey=True, legend='plots', colormap=colormap)
plt0 = pltpg.PlotTiled(do, **plot_kws, )

for ix, nx, iy, ny, plot, ax, kind in plt0.get_plot_ax_list():

    ax.legend()

    ax.set_title(new)
#plt0.add_page_legend(plt0.current_plot, *plt0.get_legend_handles_labels())


