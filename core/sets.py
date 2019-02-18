import pyomo.environ as po
import pandas as pd
import numpy as np

from grimsel.auxiliary.aux_general import silence_pd_warning
from grimsel.auxiliary.aux_m_func import cols2tuplelist
from grimsel import _get_logger

logger = _get_logger(__name__)


DICT_SETS_PP = {'ppall': 'all power plant types',
                 'pp': 'dispatchable power plants with fuels',
                 'st': 'storage plants',
                 'pr': 'variable renewables with fixed profiles',
                 'ror': 'run-of-river plants',
                 'lin': 'dispatchable plants with linear supply curve',
                 'hyrs': 'hydro reservoirs',
                 'chp': 'plants with co-generation',
                 'add': 'plants with capacity additions',
                 'rem': 'plants with capacity retirements',
                 'curt': 'dedicated curtailment technology',
                 'sll': 'plants selling produced energy carriers',
                 'rp': 'dispatchable plants with ramping costs'}

class Sets:
    '''
    Mixin class for set definition.

    '''

    def define_sets(self):
        r'''
        Add all required sets to the model.

        Adds sets as defined by

        * the ``setlst`` dictionary initialized in the :func:`get_setlst`
          method (for basic power plant sets)
        * the DataFrame attributes of the :class:`ModelBase` class

        Consult the :class:`Variables`, :class:`Parameters`,
        and :class:`Constraints` class for usage

        Primary base sets:

        ======================  ============================================
        Set                     Description
        ======================  ============================================
        :math:`\mathrm{ppall}`  all power plant types
        :math:`\mathrm{pp}`     dispatchable power plants with fuels
        :math:`\mathrm{st}`     storage plants
        :math:`\mathrm{pr}`     variable renewables with fixed profiles
        :math:`\mathrm{ror}`    run-of-river plants
        :math:`\mathrm{lin}`    dispatchable plants with linear supply curve
        :math:`\mathrm{hyrs}`   hydro reservoirs
        :math:`\mathrm{chp}`    plants with co-generation
        :math:`\mathrm{add}`    plants with capacity additions
        :math:`\mathrm{rem}`    plants with capacity retirements
        :math:`\mathrm{curt}`   dedicated curtailment technology
        :math:`\mathrm{sll}`    plants selling produced energy carriers
        :math:`\mathrm{rp}`     dispatchable plants with ramping costs
        ======================  ============================================

        Primary power plant sets and subsets:

        ===================  ==========================
        Set                  Description
        ===================  ==========================
        :math:`\mathrm{nd}`  Nodes
        :math:`\mathrm{ca}`  (Produced) energy carriers
        :math:`\mathrm{fl}`  Fuels
        :math:`\mathrm{pf}`  Profiles
        :math:`\mathrm{sy}`  model time slots
        :math:`\mathrm{mt}`  months
        :math:`\mathrm{wk}`  weeks
        :math:`\mathrm{tm}`  time maps
        ===================  ==========================

        Example derived sets:

        ==============================  ==============================================================================================
        Set                             Description
        ==============================  ==============================================================================================
        :math:`\mathrm{ppall\_ca}`      combined :math:`\mathrm{ppall\times nd}` set
        :math:`\mathrm{ppall\_ndca}`    combined :math:`\mathrm{ppall\times nd\times ca}` set
        :math:`\mathrm{sy\_ppall\_ca}`  combined :math:`\mathrm{sy\times ppall\times nd}` set
        :math:`\mathrm{ndcnn}`          combined node sets :math:`\mathrm{nd\times nd\times ca}` for inter-nodal transmission
        :math:`\mathrm{symin\_ndcnn}`   combined node sets :math:`\mathrm{sy\times nd\times nd\times ca}` for inter-nodal transmission
        :math:`\mathrm{tmsy}`           combination of time maps and slots
        :math:`\mathrm{tmsy\_mt}`       all relevant combinations :math:`\mathrm{tm\times sy\times mt}`
        ==============================  ==============================================================================================




        .. note::

           * :math:`\mathrm{sy\_ppall\_ca}`: The dispatch of a given power plant
             :math:`\mathrm{pp}` is defined for the time slots of the corresponding
             node. Since the time resolution and hence the number of time slots
             potentially depends on the node, this combined set is necessary to
             limit the variable and constraint definitions to the relevant time
             slots of any power plant.
           * :math:`\mathrm{symin\_ndcnn}`: If two nodes with different time
             resolutions are connected, the transmission variable has the higher
             time resolution of the two (*min* as in shortest time slot duration).
             This combined set expresses this relationship for each of the
             connected nodes.

        '''

        self.nd = po.Set(initialize=self.setlst['nd'], doc='Nodes')
        self.ca = po.Set(initialize=self.setlst['ca'], doc='(Produced) energy carriers')
        self.fl = po.Set(initialize=self.setlst['fl'], doc='Fuels')
        self.pf = po.Set(initialize=self.setlst['pf'], doc='Profiles')

        df_ndca = self.df_def_plant[['pp_id', 'nd_id']].set_index('pp_id')
        df_ndca = self.df_plant_encar[['pp_id', 'ca_id']].join(df_ndca,
                                                               on='pp_id')
        df_ndca = df_ndca[['pp_id', 'nd_id', 'ca_id']]

        slct_cols = ['pp_id', 'ca_id']
        slct_sets = ['ppall', 'pp', 'st', 'pr', 'ror', 'lin',
                     'hyrs', 'chp', 'add', 'rem',
                     'curt', 'sll', 'rp']


        for iset in slct_sets:

            logger.info('Defining basic sets for {}'.format(iset))

            ''' SUB SETS PP'''
            doc = dict(doc=DICT_SETS_PP[iset])
            setattr(self, iset,
                    po.Set(within=(None if iset == 'ppall' else self.ppall),
                           initialize=self.setlst[iset], **doc)
                           if iset in self.setlst.keys()
                           else po.Set(within=self.ppall, initialize=[], **doc))

            ''' SETS PP x ENCAR '''
            doc = dict(doc=r'combined :math:`\mathrm{%s\times nd}` set'%iset)
            _df = self.df_plant_encar.copy()
            _df = _df.loc[_df['pp_id'].isin(getattr(self, iset))]
            setattr(self, iset + '_ca',
                    po.Set(within=getattr(self, iset) * self.ca, **doc,
                           initialize=cols2tuplelist(_df[slct_cols])))

            ''' SETS PP x ND x ENCAR '''
            doc = dict(doc=r'combined :math:`\mathrm{%s\times nd\times ca}` set'%iset)
            _df = df_ndca.copy()
            _df = _df.loc[df_ndca['pp_id'].isin(self.setlst[iset]
                          if iset in self.setlst.keys() else [])]
            setattr(self, iset + '_ndca',
                    po.Set(within=getattr(self, iset) * self.nd * self.ca,
                           initialize=cols2tuplelist(_df), **doc))


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
        lst = cols2tuplelist(df.loc[df.pp_id.isin(self.setlst['sll'])])
        doc = ('"fuels" solds by power plant :math:`pp` '
               'consuming energy carrier :math:`ca`')
        setattr(self, 'pp' + '_ndcafl_sll',
                po.Set(within=self.pp_ndcafl, initialize=lst, doc=doc))

        # temporal
        self.sy = po.Set(initialize=list(self.df_tm_soy.sy.unique()),
                         ordered=True, doc='model time slots')

        self.sy_hydbc = po.Set(within=self.sy,
                               initialize=set(self.df_plant_month.sy))

        self.mt = (po.Set(initialize=list(self.df_def_month['mt_id']),
                          doc='months')
                   if not self.df_def_month is None else None)
        self.wk = (po.Set(initialize=list(self.df_def_week['wk_id']),
                          doc='weeks')
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
            doc = dict(doc='combined node sets '
                           ':math:`\mathrm{nd\\times nd\\times ca}` '
                           'for inter-nodal transmission')
            df = self.df_node_connect[['nd_id', 'nd_2_id', 'ca_id']]
            self.ndcnn = po.Set(within=self.nd * self.nd * self.ca,
                             initialize=cols2tuplelist(df), ordered=True, **doc)

            doc = dict(doc='combined node sets '
                           ':math:`\mathrm{sy\\times nd\\times nd\\times ca}` '
                           'for inter-nodal transmission')
            df = self.df_symin_ndcnn[['symin', 'nd_id', 'nd_2_id', 'ca_id']]
            self.symin_ndcnn = po.Set(within=self.sy * self.nd
                                             * self.nd * self.ca, **doc,
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

        self.tm = po.Set(initialize=self.df_tm_soy.tm_id.unique(),
                         ordered=True, doc='time maps')

        list_tmsy = cols2tuplelist(self.df_tm_soy[['tm_id', 'sy']])
        self.tmsy = po.Set(within=self.tm*self.sy, initialize=list_tmsy,
                           ordered=True,
                           doc='combination of time maps and slots')

        # only constructed if self.mt exists
        doc = 'all relevant combinations :math:`\mathrm{tm\\times sy\\times mt}`'
        self.tmsy_mt = (po.Set(within=self.tmsy * self.mt, doc=doc,
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
        df['tm_id'] = (df.pp_id.replace(self.mps.dict_plant_2_node_id)
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
            doc = r'combined :math:`\mathrm{sy\times %s\times nd}` set'%slct_set
            self.delete_component(set_name)

            logger.info('Defining set ' + set_name)

            set_pp = set(self.setlst[slct_set])

            setattr(self, set_name,
                    po.Set(within=self.sy * getattr(self, slct_set) * self.ca,
                           ordered=True, doc=doc,
                           initialize=[row for row in list_syppca
                                       if row[1] in set_pp]))


    def get_setlst(self):
        '''
        Lists of indices for all model components are extracted from the
        input tables and stored in a dictionary ``ModelBase.setlst``.

        For the most part power plant subset definitions are based on the
        binary columns *set_def_..* in the ``df_def_plant`` input table.

        Exceptions are:

        * Power plant subsets defined as set unions of the above
          (:math:`pf`, :math:`rp`)
        * Profile sets from the ``df_fuel_node_encar`` (price profiles),
          ``df_plant_encar`` (supply profiles) and ``df_node_encar``
          (demand profiles) input tables.

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

#         # hydro and storage together
#        self.setlst['sthyrs'] = self.setlst['st'] + self.setlst['hyrs']
#
#        mask_node = self.df_def_node['nd_id'].isin(self.slct_node_id)

#

    @silence_pd_warning
    def _get_set_docs(self):
        '''
        Convenience method to extract all set docs from a :class:`ModelBase`
        instance.

        '''

        import tabulate

        to_math = lambda x: ':math:`\mathrm{%s}`'%x

        dict_doc = {name: comp.doc for name, comp in self.__dict__.items()
                    if type(comp) in (poset.SimpleSet, poset.OrderedSimpleSet)
#                    and (not '_' in name)
                    and comp.doc
                    }

        comb_sets = ['ndcnn', 'tmsy']

        cols = ['Set', 'Description']
        df_doc = pd.Series(dict_doc).reset_index()
        df_doc.columns = cols

        mask_not_under = ~df_doc.Set.str.contains('_')
        mask_not_ndca = ~df_doc.Set.str.contains('_ndca')
        mask_not_ca = ~df_doc.Set.str.contains('_ca')

        print('\nPrimary base sets:\n')

        df_doc_pp = df_doc.loc[df_doc.Set.isin(DICT_SETS_PP)
                               & mask_not_under]
        df_doc_pp['Set'] = df_doc_pp.Set.apply(to_math)
        print(tabulate.tabulate(df_doc_pp.applymap(lambda x: x.replace('_', '\_')),
                                tablefmt='rst', headers=cols,
                                showindex=False))

        print('\nPrimary power plant sets and subsets:\n')


        df_doc_oth = df_doc.loc[~df_doc.Set.isin(DICT_SETS_PP)
                                & ~df_doc.Set.isin(comb_sets)
                                & mask_not_under]
        df_doc_oth['Set'] = df_doc_oth.Set.apply(to_math)
        print(tabulate.tabulate(df_doc_oth.applymap(lambda x: x.replace('_', '\_')),
                                headers=cols,
                                tablefmt='rst', showindex=False))

        df_doc_oth = df_doc.loc[(df_doc.Set.isin(comb_sets)
                                | ~mask_not_under) & mask_not_ndca & mask_not_ca]
        df_doc_oth['Set'] = df_doc_oth.Set.apply(to_math)

        print('\nExample derived sets:\n')

        df_doc_ppca = df_doc.loc[~mask_not_ca | ~mask_not_ndca]
        df_doc_ppca['Set'] = df_doc_ppca.Set.apply(to_math)
        df_doc_ppca = df_doc_ppca.loc[df_doc_ppca.Set.str.contains('ppall')
                        | df_doc_ppca.Set.str.endswith('sll')]
        print(tabulate.tabulate(pd.concat([df_doc_ppca, df_doc_oth]).applymap(lambda x: x.replace('_', '\_')), headers=cols,
                                tablefmt='rst', showindex=False))




