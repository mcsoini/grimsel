'''
Model sets
=================


'''


import pyomo.environ as po
import pyomo.core.base.sets as poset
import pandas as pd
import numpy as np

from grimsel.auxiliary.aux_general import silence_pd_warning
from grimsel.auxiliary.aux_m_func import cols2tuplelist
from grimsel import _get_logger

logger = _get_logger(__name__)


DICT_SETS_DOC = {r'sy': r'model time slots : df_tm_soy : t',
                 r'ppall': r'all power plant types : df_def_plant : p',
                 r'pp': r'dispatchable power plants with fuels : df_def_plant : p',
                 r'st': r'storage plants : df_def_plant : p',
                 r'pr': r'variable renewables with fixed profiles : df_def_plant : p',
                 r'ror': r'run-of-river plants : df_def_plant : p',
                 r'lin': r'dispatchable plants with linear supply curve : df_def_plant : p',
                 r'hyrs': r'hydro reservoirs : df_def_plant : p',
                 'chp': r'plants with co-generation : df_def_plant : p',
                 r'add': r'plants with capacity additions : df_def_plant : p',
                 r'rem': r'plants with capacity retirements : df_def_plant : p',
                 r'curt': r'dedicated curtailment technology : df_def_plant : p',
                 r'sll': r'plants selling produced energy carriers : df_def_plant : p',
                 r'rp': r'dispatchable plants with ramping costs : df_def_plant : p',
                 r'ppall_nd': r'combined :math:`\mathrm{ppall\times nd}` set; equivalent for all subsets of :math:`\mathrm{ppall}` : df_def_plant : (p,n)',
                 r'ppall_ndca': r'combined :math:`\mathrm{ppall\times nd\times ca}` set; equivalent for all subsets of :math:`\mathrm{ppall}` : merge(df_def_plant, df_plant_encar) : (p,n,c)',
                 r'ppall_ndcafl': r'combined :math:`\mathrm{ppall\times nd\times ca\times fl}` set; equivalent for all subsets of :math:`\mathrm{ppall}` : merge(df_def_plant, df_plant_encar) : (p,n,c,f)',
                 r'pp_ndcafl_sll': r'"fuels" sold by power plants :math:`\mathrm{pp}` consuming energy carrier :math:`\mathrm{ca}` : merge(df_def_plant, df_plant_encar) : (p,n,c,f)',
                 r'sy_hydbc\subset sy': r'Time slots with exogenously defined storage level boundary conditions. : df_plant_month : t',
                 r'mt': r'months : df_plant_month : m',
                 r'wk': r'weeks : df_plant_week : w',
                 r'ndcnn': r'combined node sets :math:`\mathrm{nd\times nd\times ca}` for inter-nodal transmission : df_node_connect : (n,n_2,c)',
                 r'symin_ndcnn': r'combined node sets :math:`\mathrm{sy\times nd\times nd\times ca}` for inter-nodal transmission : merge(df_tm_soy, df_node_connect) : (t,n,n_2,c)',
                 r'fl_erg': r'fuels with energy production constraints : df_def_fuel : f',
                 r'tm': r'time maps : df_tm_soy : \tau',
                 r'tmsy': r'combination of time maps and slots : df_tm_soy : (\tau,t)',
                 r'tmsy_mt': r'all relevant combinations :math:`\mathrm{tm\times sy\times mt}` : df_tm_soy : (\tau,t,m)',
                 r'sy_ppall_ca': r'combined :math:`\mathrm{sy\times ppall\times nd}` set; equivalent for all subsets of :math:`\mathrm{ppall}` : merge(df_plant_encar, df_tm_soy) : (t,p,c)',
                 r'nd': r'Nodes : df_def_node : n',
                 r'ca': r'Output energy carriers : df_def_encar : c',
                 r'fl': r'Fuels : df_def_fuel : f',
                 r'ndcafl': r'Relevant combinations of nodes, produced energy carriers, and fuels : df_node_fuel_encar : (n,c,f)',
                 r'pf': r'Profiles (demand, supply, price, etc) : df_def_profile : \phi',
                 r'sy_ndca': r'Combined :math:`\mathrm{sy\times nd\times ca}` set : merge(df_node_encar, df_tm_soy) : (t,n,c)',
                 r'pp_ndcaca': r'combined :math:`\mathrm{pp\times nd\times ca\times ca}` set describing plants which convert one produced energy carrier into another : merge(df_def_encar, df_def_plant, df_plant_encar) : (p,n,c_{out},c)'
                 }

class Sets:
    '''
    Mixin class for set definition.

    '''

    # base power plant subsets
    slct_sets = ['ppall', 'pp', 'st', 'pr', 'ror', 'lin',
                 'hyrs', 'chp', 'add', 'rem',
                 'curt', 'sll', 'rp']


    def define_sets(self):
        r'''
        Add all required sets to the model.

        Adds sets as defined by

        * the ``setlst`` dictionary initialized in the :func:`get_setlst`
          method
        * the DataFrame attributes of the :class:`ModelBase` class for more
          complex derived sets

        %s


        '''

        self.nd = po.Set(initialize=self.setlst['nd'])
        self.ca = po.Set(initialize=self.setlst['ca'])
        self.fl = po.Set(initialize=self.setlst['fl'])
        self.pf = po.Set(initialize=self.setlst['pf'])

        df_ndca = self.df_def_plant[['pp_id', 'nd_id']].set_index('pp_id')
        df_ndca = self.df_plant_encar[['pp_id', 'ca_id']].join(df_ndca,
                                                               on='pp_id')
        df_ndca = df_ndca[['pp_id', 'nd_id', 'ca_id']]

        slct_cols = ['pp_id', 'ca_id']


        for iset in self.slct_sets:

            logger.info('Defining basic sets for {}'.format(iset))

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
#        list_sets = [st for st in list_sets if st in self.setlst.keys()]

        for iset in list_sets:

            if iset in self.setlst:
                cols_ppcafl = ['pp_id', 'ca_id', 'fl_id']
                df = df_0.loc[df_0['pp_id'].isin(self.setlst[iset]),
                              cols_ppcafl]
                new_set = po.Set(within=getattr(self, iset) * self.ca * self.fl,
                                 initialize=cols2tuplelist(df))
                setattr(self, iset + '_cafl', new_set)

                slct_cols_ppndcafl = ['pp_id', 'nd_id', 'ca_id', 'fl_id']
                df = df_0.loc[df_0.pp_id.isin(self.setlst[iset]),
                              slct_cols_ppndcafl]

                setattr(self, iset + '_ndcafl',
                        po.Set(within=(getattr(self, iset) * self.nd
                                     * self.ca * self.fl),
                               initialize=cols2tuplelist(df)))
            else:

                new_set_cafl = po.Set(within=getattr(self, iset) * self.ca * self.fl,
                                 initialize=[])
                setattr(self, iset + '_cafl', new_set_cafl)

                new_set_ndcafl = po.Set(within=getattr(self, iset) * self.nd * self.ca * self.fl,
                                        initialize=[])
                setattr(self, iset + '_ndcafl', new_set_ndcafl)


        # plants selling fuels ... only ppall, therefore outside the loop
        lst = cols2tuplelist(df.loc[df.pp_id.isin(self.setlst['sll']
                                                  if 'sll' in self.setlst
                                                  else [])])
        setattr(self, 'pp_ndcafl_sll',
                po.Set(within=self.pp_ndcafl, initialize=lst))

        # temporal
        self.sy = po.Set(initialize=list(self.df_tm_soy.sy.unique()),
                         ordered=True)

        self.sy_hydbc = (po.Set(within=self.sy,
                               initialize=set(self.df_plant_month.sy))
                         if not self.df_plant_month is None else None)

        self.mt = (po.Set(initialize=list(self.df_def_month['mt_id']))
                   if not self.df_def_month is None else None)
        self.wk = (po.Set(initialize=list(self.df_def_week['wk_id']))
                   if not self.df_def_week is None else None)

        # pp_cacafcl; used to account for conversions of ca in the supply rule
        if 'fl_id' in self.df_def_encar:
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
        else:
            self.pp_ndcaca = None

        # inter-node connections
        if not self.df_node_connect is None and not self.df_node_connect.empty:
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
        if 'is_constrained' in self.df_def_fuel:
            lst = self.df_def_fuel.loc[self.df_def_fuel.is_constrained==1,
                                           'fl_id'].tolist()
            self.fl_erg = po.Set(within=self.fl, initialize=lst, ordered=True)
        else:
            self.fl_erg = po.Set(within=self.fl, initialize=[])

        # set pf_id for profiles
        for pf_set in ['dmnd_pf', 'supply_pf', 'pricesll_pf', 'pricebuy_pf']:
            setattr(self, pf_set,
                    po.Set(within=self.pf, initialize=self.setlst[pf_set],
                           ordered=True))

        self._init_tmsy_sets()


    def _init_tmsy_sets(self):
        '''

        The plant ids and the time slots are connected
        through the node-specific time resolution.
        '''

        self.tm = po.Set(initialize=self.df_tm_soy.tm_id.unique(),
                         ordered=True)

        list_tmsy = cols2tuplelist(self.df_tm_soy[['tm_id', 'sy']])
        self.tmsy = po.Set(within=self.tm*self.sy, initialize=list_tmsy,
                           ordered=True)

        # only constructed if self.mt exists
        self.tmsy_mt = (po.Set(within=self.tmsy * self.mt,
                               initialize=cols2tuplelist(
                                    self.df_tm_soy[['tm_id', 'sy', 'mt_id']]))
                        if not self.mt is None else None)

        df = pd.merge(self.df_def_node, self.df_node_encar,
                      on='nd_id', how='outer')[['nd_id', 'ca_id']]
        df = df.loc[~df.ca_id.isna()].drop_duplicates()
        df['tm_id'] = df.nd_id.replace(self.dict_nd_tm_id)
        cols = ['sy', 'nd_id', 'ca_id']
        list_syndca = pd.merge(self.df_tm_soy[['tm_id', 'sy']],
                                df, on='tm_id', how='outer')[cols]

        self.sy_ndca = po.Set(within=self.sy*self.ndca, ordered=True,
                              initialize=cols2tuplelist(list_syndca))

        mask_pp = self.df_plant_encar.pp_id.isin(self.setlst['ppall'])
        df = self.df_plant_encar.loc[mask_pp, ['pp_id', 'ca_id']].copy()
        df['tm_id'] = (df.pp_id
                         .replace(self.mps.dict_plant_2_node_id)
                         .replace(self.dict_nd_tm_id))
        cols = ['sy', 'pp_id', 'ca_id']
        list_syppca = pd.merge(self.df_tm_soy[['sy', 'tm_id']],
                                df, on='tm_id', how='outer')[cols]

        list_syppca = list_syppca.loc[~(list_syppca.pp_id.isna()
                                        | list_syppca.ca_id.isna())]

        list_syppca = cols2tuplelist(list_syppca)

        for slct_set in ['ppall', 'rp', 'st', 'hyrs', 'pr',
                         'pp', 'chp', 'ror', 'lin']:

            set_name = 'sy_%s_ca'%slct_set
            within = self.sy * getattr(self, slct_set) * self.ca

            if slct_set in self.setlst:

                logger.info('Defining set ' + set_name)

                set_pp = set(self.setlst[slct_set])

                setattr(self, set_name,
                        po.Set(within=within, ordered=True,
                               initialize=[row for row in list_syppca
                                           if row[1] in set_pp]))
            else:
                setattr(self, set_name, po.Set(within=within, initialize=[]))

    def get_setlst(self):
        '''
        Lists of indices for all model components are extracted from the
        input tables and stored in a dictionary ``ModelBase.setlst``.

        For the most part power plant subset definitions are based on the
        binary columns *set_def_..* in the ``df_def_plant`` input table.

        '''
        # define lists for set initialization
        self.setlst = {st: [] for st in self.slct_sets}

        df = self.df_def_plant

        qry = ' & '.join(['{} == 0'.format(sd)
                         for sd in ('set_def_tr', 'set_def_dmd')
                         if sd in df.columns])

        self.setlst['ppall'] = (df.query(qry).pp_id.tolist())

        for ippset in df.columns[df.columns.str.contains('set_def')]:
            # Note: index starting at 8 removes prefix set_def_ from col name
            self.setlst[ippset[8:]] = df.loc[df[ippset] == 1, 'pp_id'].tolist()

        mask_node = self.df_def_node['nd_id'].isin(self.slct_node_id)
        self.setlst['nd'] = self.df_def_node.loc[mask_node]['nd_id'].tolist()
        self.setlst['ca'] = self.df_def_encar['ca_id'].get_values().tolist()

        # fuels are bought fuels only, not generated encars used as input
        df = self.df_def_fuel.copy()
#        df = df.loc[df.is_ca.isin([0]), 'fl_id']
        self.setlst['fl'] = df.fl_id.get_values().tolist()


        for col, df in [('supply_pf_id', self.df_plant_encar),
                        ('pricesll_pf_id', self.df_fuel_node_encar),
                        ('pricebuy_pf_id', self.df_fuel_node_encar),
                        ('dmnd_pf_id', self.df_node_encar)]:

            if col in df.columns:
                df_ = df.copy()
                df_ = df_.loc[-df_[col].isna(), col]
                self.setlst[col.replace('_id', '')] = df_.unique().tolist()
            else:
                self.setlst[col.replace('_id', '')] = []

        self.setlst['pf'] = (self.setlst['dmnd_pf']
                             + self.setlst['pricesll_pf']
                             + self.setlst['pricebuy_pf']
                             + self.setlst['supply_pf'])

        self.setlst['rp'] = ((self.setlst['pp'] if 'pp' in self.setlst else [])
                             + (self.setlst['ror'] if 'ror' in self.setlst else [])
                             + (self.setlst['hyrs'] if 'hyrs' in self.setlst else [])
                             + (self.setlst['st'] if 'st' in self.setlst else []))

    @silence_pd_warning
    @staticmethod
    def _get_set_docs():
        '''
        Convenience method to extract all set docs from a :class:`ModelBase`
        instance.

        '''

        import tabulate

        to_math = lambda x: ':math:`\mathrm{%s}`'%x

        comb_sets = ['ndcnn', 'tmsy']

        cols = ['Set', 'Members', 'Description', 'Source table']
        df_doc = pd.Series(DICT_SETS_DOC).reset_index()
        df_doc[['Description', 'Source', 'Members']] = pd.DataFrame(df_doc[0].apply(lambda x: tuple(x.split(' : '))).tolist())
        df_doc = df_doc.drop(0, axis=1)
        df_doc.columns = ['Set', 'Description', 'Source table', 'Members']
        df_doc.Members = df_doc.Members.apply(lambda x: ':math:`\mathrm{%s}`'%(x))
        df_doc['Source table'] = df_doc['Source table'].apply(lambda x: '``%s``'%(x))
        df_doc = df_doc[cols]


        mask_not_under = ~df_doc.Set.str.contains('_')
        mask_not_ndca = ~df_doc.Set.str.contains('_ndca')
        mask_not_ca = ~df_doc.Set.str.contains('_ca')


        list_pp = ['ppall', 'pp', 'st', 'pr', 'ror', 'lin', 'hyrs', 'chp',
                   'add', 'rem', 'curt', 'sll', 'rp']

        df_doc_oth = df_doc.loc[~df_doc.Set.isin(list_pp)
                                & ~df_doc.Set.isin(comb_sets)
                                & mask_not_under]
        df_doc_oth['Set'] = df_doc_oth.Set.apply(to_math)
        table_base = (tabulate.tabulate(df_doc_oth,
                                headers=cols,
                                tablefmt='rst', showindex=False))



        df_doc_pp = df_doc.loc[df_doc.Set.isin(list_pp)
                               & mask_not_under]
        df_doc_pp['Set'] = df_doc_pp.Set.apply(to_math)
        df_doc_pp.Set = df_doc_pp.Set.apply(lambda x: x.replace('}', '\subset ppall}').replace('ppall\subset ', ''))
        table_pp = (tabulate.tabulate(df_doc_pp,
                                tablefmt='rst', headers=cols,
                                showindex=False))



        df_doc_oth = df_doc.loc[(df_doc.Set.isin(comb_sets)
                                | ~mask_not_under)]
        df_doc_oth['Set'] = df_doc_oth.Set.apply(to_math)

        df_doc_ppca = df_doc.loc[~mask_not_ca | ~mask_not_ndca]
        df_doc_ppca['Set'] = df_doc_ppca.Set.apply(to_math)
        df_doc_ppca = df_doc_ppca.loc[df_doc_ppca.Set.str.contains('ppall')
                        | df_doc_ppca.Set.str.endswith('sll')]
        df_doc_oth = pd.concat([df_doc_ppca, df_doc_oth])
        df_doc_oth.Set = df_doc_oth.Set.apply(lambda x: x.replace('_', '\_'))
        table_derived = tabulate.tabulate(df_doc_oth, headers=cols,
                                tablefmt='rst', showindex=False)

        doc_str = r'''

.. table:: **Primary base sets**
    :widths: 15 10 100 30

    %s


.. _power_plant_sets:

.. table:: **Primary power plant sets and subsets**
    :widths: 15 10 100 30

    %s

.. table:: **Key derived sets**
    :widths: 15 10 100 30

    %s

.. note::

   * :math:`\mathrm{sy\_ppall\_ca}`: The operation (power production/charging)
     of a given plant :math:`\mathrm{pp}` is defined for each of the time
     slots of the
     corresponding node. Since the time resolution and hence the
     number of time slots potentially depends on the node, this
     combined set is necessary to limit the variable and constraint
     definitions to the relevant time slots of any power plant.
   * :math:`\mathrm{symin\_ndcnn}`: If two nodes with different time
     resolutions are connected, the transmission variable has the
     higher time resolution of the two (*min* as in "smallest time slot
     duration"). This combined set expresses this relationship for
     each of the connected nodes.

'''%(table_base.replace('\n', '\n    '),
           table_pp.replace('\n', '\n    '),
           table_derived.replace('\n', '\n    '))

        return doc_str




Sets.define_sets.__doc__ = Sets.define_sets.__doc__%Sets._get_set_docs()
