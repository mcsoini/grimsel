import pyomo.environ as po
import pandas as pd
import numpy as np

from grimsel.auxiliary.aux_m_func import cols2tuplelist
from grimsel import _get_logger

logger = _get_logger(__name__)

class Sets:
    '''
    Mixin class for set definition.
    Methods:
    - define_sets: adds all relevant sets
    - get_setlst: generate dictionary with lists of set elements
    '''

    def define_sets(self):

        self.nd = po.Set(initialize=self.setlst['nd'], doc='Nodes')
        self.ca = po.Set(initialize=self.setlst['ca'], doc='Energy carriers')
        self.fl = po.Set(initialize=self.setlst['fl'], doc='Sub-fuels')
        self.pf = po.Set(initialize=self.setlst['pf'], doc='Profiles')

        df_ndca = self.df_def_plant[['pp_id', 'nd_id']].set_index('pp_id')
        df_ndca = self.df_plant_encar[['pp_id', 'ca_id']].join(df_ndca,
                                                               on='pp_id')
        df_ndca = df_ndca[['pp_id', 'nd_id', 'ca_id']]

        slct_cols = ['pp_id', 'ca_id']
        slct_sets = ['ppall', 'pp', 'st', 'pr', 'ror', 'lin',
                     'hyrs', 'chp', 'add', 'rem', 'winsol', 'scen',
                     'curt', 'sll']

        for iset in slct_sets:

            print(iset, self.setlst[iset])

            ''' SUB SETS PP'''
            setattr(self, iset,
                    po.Set(within=(None if iset == 'ppall' else self.ppall),
                           initialize=self.setlst[iset])
                           if iset in self.setlst.keys()
                           else po.Set(within=self.ppall, initialize=[]))

            ''' SETS PP x ENCAR '''
            _df = self.df_plant_encar.copy()
            _df = _df.loc[_df['pp_id'].isin(getattr(self, iset))]
            setattr(self, iset + '_ca',
                    po.Set(within=getattr(self, iset) * self.ca,
                           initialize=cols2tuplelist(_df[slct_cols])))

            ''' SETS PP x ND x ENCAR '''
            _df = df_ndca.copy()
            _df = _df.loc[df_ndca['pp_id'].isin(self.setlst[iset]
                          if iset in self.setlst.keys() else [])]
            setattr(self, iset + '_ndca',
                    po.Set(within=getattr(self, iset) * self.nd * self.ca,
                           initialize=cols2tuplelist(_df)))


        # no scf fuels in the _cafl and _ndcafl
        # These are used to calculate fuel costs, that's why we don't want
        # the generated fuels in there.
        df_0 = self.df_def_plant[['pp_id', 'nd_id', 'fl_id']]
        df_0 = df_0.set_index('pp_id')
        df_0 = self.df_plant_encar[['pp_id', 'ca_id']].join(df_0, on='pp_id')
        df_0 = df_0.loc[df_0.fl_id.isin(self.setlst['fl'])]

        list_sets = ['ppall', 'hyrs', 'pp', 'chp', 'ror', 'st', 'lin']
        list_sets = [st for st in list_sets if st in self.setlst.keys()]

        for iset in list_sets:

            cols_ppcafl = ['pp_id', 'ca_id', 'fl_id']
            df = df_0.loc[df_0['pp_id'].isin(self.setlst[iset]), cols_ppcafl]

            setattr(self, iset + '_cafl',
                    po.Set(within=getattr(self, iset) * self.ca * self.fl,
                           initialize=cols2tuplelist(df)))

            slct_cols_ppndcafl = ['pp_id', 'nd_id', 'ca_id', 'fl_id']
            df = df_0.loc[df_0.pp_id.isin(self.setlst[iset]),
                          slct_cols_ppndcafl]

            setattr(self, iset + '_ndcafl',
                    po.Set(within=(getattr(self, iset) * self.nd
                                 * self.ca * self.fl),
                           initialize=cols2tuplelist(df)))

        # plants selling fuels ... only ppall, therefore outside the loop
        setattr(self, 'pp' + '_ndcafl_sll',
                po.Set(within=self.pp_ndcafl,
                       initialize=
                       cols2tuplelist(df.loc[df.pp_id.isin(self.setlst['sll'])])))

        # temporal
        self.sy = po.Set(initialize=list(self.df_tm_soy.sy.unique()),
                         ordered=True)
        self.tm = po.Set(initialize=self.df_tm_soy.tm_id.unique(),
                         ordered=True)

        self.sy_hydbc = po.Set(within=self.sy,
                               initialize=self.df_plant_month.sy.tolist())

        self.mt = (po.Set(initialize=list(self.df_def_month['mt_id']))
                   if not self.df_def_month is None else None)
        self.wk = (po.Set(initialize=list(self.df_def_week['wk_id']))
                   if not self.df_def_week is None else None)

        # pp_cacafcl; used to account for conversions of ca in the supply rule
        df_cafl = self.df_def_encar.set_index('fl_id')['ca_id']
        df_cafl = df_cafl.rename('ca_fl_id')
        df_ppca = self.df_plant_encar.set_index('pp_id')['ca_id']
        df = (self.df_def_plant.join(df_ppca, on='pp_id')
                               .join(df_cafl, on='fl_id'))
        df = df.loc[-df.ca_fl_id.isnull()
                  & -df.ca_id.isnull(), ['pp_id', 'nd_id', 'ca_id',
                                         'ca_fl_id']]
        self.pp_ndcaca = po.Set(within=self.pp_ndca * self.ca,
                                  initialize=cols2tuplelist(df))

        # inter-node connections
        if not self.df_node_connect is None:
            df = self.df_node_connect[['nd_id', 'nd_2_id', 'ca_id']]
            self.ndcnn = po.Set(within=self.nd * self.nd * self.ca,
                             initialize=cols2tuplelist(df), ordered=True)

            df = self.df_symin_ndcnn[['symin', 'nd_id', 'nd_2_id', 'ca_id']]
            self.symin_ndcnn = po.Set(within=self.sy * self.nd
                                             * self.nd * self.ca,
                                      initialize=cols2tuplelist(df),
                                      ordered=True)
        else:
            self.ndcnn = po.Set(within=self.nd * self.nd * self.ca)
            self.symin_ndcnn = po.Set(within=self.sy * self.nd
                                             * self.nd * self.ca)


        # ndca for electricity only; mainly used for flexible demand;
        # then again: why would only EL have flexible demand?
        df = pd.concat([pd.Series(self.slct_node_id, name='nd_id'),
                        pd.Series(np.ones(len(self.slct_node_id))
                                  * self.mps.dict_ca_id['EL'],
                                  name='ca_id')], axis=1)
        self.ndca_EL = po.Set(within=self.nd * self.ca,
                              initialize=cols2tuplelist(df), ordered=True)

        # general ndca
        df = self.df_node_encar[['nd_id', 'ca_id']].drop_duplicates()
        self.ndca = po.Set(within=self.nd * self.ca,
                           initialize=cols2tuplelist(df), ordered=True)

        # general ndcafl
        if not self.df_fuel_node_encar is None:
            df = self.df_fuel_node_encar[['nd_id', 'ca_id', 'fl_id']]
            self.ndcafl = po.Set(within=self.nd * self.ca * self.fl,
                              initialize=cols2tuplelist(df), ordered=True)
        else:
            self.ndcafl = None

        # fuels with energy constraints
        lst = self.df_def_fuel.loc[self.df_def_fuel.is_constrained==1,
                                       'fl_id'].tolist()
        self.fl_erg = po.Set(within=self.fl, initialize=lst, ordered=True)

        # all plants with ramping costs
        vcrp_pos = (self.df_plant_encar.loc[self.df_plant_encar.vc_ramp > 0]
                        .set_index(['pp_id', 'ca_id']).index.values)
        rp = [ppca for ppca in vcrp_pos if ppca[0] in self.setlst['rp']]
        self.rp_ca = po.Set(within=self.ppall_ca, initialize=rp, ordered=True)

        # set pf_id for profiles
        for pf_set in ['dmnd_pf', 'supply_pf', 'pricesll_pf', 'pricebuy_pf']:
            setattr(self, pf_set,
                    po.Set(within=self.pf, initialize=self.setlst[pf_set],
                           ordered=True))

        self._init_tmsy_sets()


    def _init_tmsy_sets(self):
        '''


        Through the node-specific time resolution the plant ids and the
        time slots are connected.
        '''

        list_tmsy = cols2tuplelist(self.df_tm_soy[['tm_id', 'sy']])
        self.tmsy = po.Set(within=self.tm*self.sy, initialize=list_tmsy,
                           ordered=True)

        # only constructed if self.mt exists
        self.tmsy_mt = (po.Set(within=self.tmsy * self.mt,
                               initialize=cols2tuplelist(
                                    self.df_tm_soy[['tm_id', 'sy', 'mt_id']]))
                        if not self.mt is None else None)

        df = pd.merge(self.df_def_plant, self.df_plant_encar,
                      on='pp_id', how='outer')[['nd_id', 'ca_id']]
        df = df.loc[~df.ca_id.isna()].drop_duplicates()
        df['tm_id'] = df.nd_id.replace(self.dict_nd_tm_id)
        cols = ['sy', 'nd_id', 'ca_id']
        list_syndca = pd.merge(self.df_tm_soy[['tm_id', 'sy']],
                                df, on='tm_id', how='outer')[cols]

        self.sy_ndca = po.Set(within=self.sy * self.ndca, ordered=True,
                           initialize=cols2tuplelist(list_syndca))

        cols = ['sy', 'pp_id', 'ca_id']
        for slct_set in ['ppall', 'rp', 'st', 'hyrs', 'pr', 'pp', 'chp',
                         'ror', 'lin']:

            set_name = 'sy_%s_ca'%slct_set
            self.delete_component(set_name)

            logger.info('Defining set ' + set_name)

            mask_pp = self.df_plant_encar.pp_id.isin(self.setlst[slct_set])
            df = self.df_plant_encar.loc[mask_pp, ['pp_id', 'ca_id']].copy()
            df['tm_id'] = (df.pp_id.replace(self.mps.dict_plant_2_node_id)
                             .replace(self.dict_nd_tm_id))

            list_syppca = pd.merge(self.df_tm_soy[['sy', 'tm_id']],
                                    df, on='tm_id', how='outer')[cols]

            list_syppca = list_syppca.loc[~(list_syppca.pp_id.isna()
                                            | list_syppca.ca_id.isna())]

            setattr(self, set_name,
                    po.Set(within=self.sy * self.ppall_ca, ordered=True,
                           initialize=cols2tuplelist(list_syppca)))

    def get_setlst(self):
        '''
        Lists of indices for all model components are extracted from the
        input tables and stored in a dictionary self.
        '''
        # define lists for set initialization
        self.setlst = {}
        _df = self.df_def_plant
        self.setlst['ppall'] = _df.loc[_df['set_def_tr'] == 0,
                                       'pp_id'].tolist()
        for ippset in _df.columns[_df.columns.str.contains('set_def')]:
            # Note: index starting at 8 removes prefix set_def_
            self.setlst[ippset[8:]] = _df.loc[_df[ippset] == 1,
                                              'pp_id'].tolist()
        mask_node = self.df_def_node['nd_id'].isin(self.slct_node_id)
        self.setlst['nd'] = self.df_def_node.loc[mask_node]['nd_id'].tolist()
        self.setlst['ca'] = self.df_def_encar['ca_id'].get_values().tolist()

        # fuels are bought fuels only, not generated encars used as input
        df = self.df_def_fuel.copy()
        df = df.loc[df.is_ca.isin([0]), 'fl_id']
        self.setlst['fl'] = df.get_values().tolist()

        df = self.df_plant_encar.copy()
        df = df.loc[-df.supply_pf_id.isna(), 'supply_pf_id']
        self.setlst['supply_pf'] = df.drop_duplicates().get_values().tolist()

        df = self.df_fuel_node_encar.copy()
        df = df.loc[-df.pricesll_pf_id.isna(), 'pricesll_pf_id']
        self.setlst['pricesll_pf'] = df.drop_duplicates().get_values().tolist()

        df = self.df_fuel_node_encar.copy()
        df = df.loc[-df.pricebuy_pf_id.isna(), 'pricebuy_pf_id']
        self.setlst['pricebuy_pf'] = df.drop_duplicates().get_values().tolist()

        df = self.df_node_encar
        df = df.loc[-df.dmnd_pf_id.isna(), 'dmnd_pf_id']
        self.setlst['dmnd_pf'] = df.drop_duplicates().get_values().tolist()

        self.setlst['pf'] = (self.setlst['dmnd_pf']
                             + self.setlst['pricesll_pf']
                             + self.setlst['pricebuy_pf']
                             + self.setlst['supply_pf'])

        self.setlst['rp'] = (self.setlst['pp']
                             + self.setlst['ror']
                             + self.setlst['hyrs'])


        # hydro and storage together
        self.setlst['sthyrs'] = self.setlst['st'] + self.setlst['hyrs']

        mask_node = self.df_def_node['nd_id'].isin(self.slct_node_id)

