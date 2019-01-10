import sys

import grimsel.auxiliary.sqlutils.aux_sql_func as aql
from grimsel.auxiliary.aux_general import get_config
import numpy as np
import pandas as pd

class Maps:

    def __init__(self, sc, db, color_style='p1'):
        self.sc = sc
        self.make_style_dicts(color_style)


        self.db = get_config('sql_connect')['db'] if db is None else db
        self.make_id_dicts()
        self.make_hard_coded_dicts()

        print(color_style)

    def display_colormap(self):
        all_cols = {'c' + str(i): getattr(self, 'c' + str(i))
                    for i in range(1, 200)
                    if 'c' + str(i) in self.__dict__.keys()}

        df = pd.DataFrame(np.abs(np.sin(np.random.randn(2, len(all_cols)))),
                          columns=all_cols.keys())
        df.plot.area(stacked=True, color=[all_cols[c] for c in df.columns])


    def make_style_dicts(self, color_style):


        if color_style == 'p1':

            self.c1 = '#b85917' # dark orange
            self.c2 = '#956c36' # siena
            self.c3 = '#d3bca0' # light brown
            self.c4 = '#484823' # dark olive green
            self.c5 = '#a2b3ac' # dark sea green
            self.c6 = '#49524E'
            self.c7 = '#141716'
            self.c8 = 'k'
            self.c9 = 'k'

        if color_style == 'p2':
            self.c1 = '#6E0A25'
            self.c2 = '#29539B'
            self.c3 = '#8D96C8'
            self.c4 = '#C4BFBA'
            self.c5 = '#B15C79'

            self.c1 = '#4A4E38'
            self.c2 = '#373617'
            self.c3 = '#A8A27A'
            self.c4 = '#7B7542'
            self.c5 = '#656F73'
            self.c6 = '#7092B9'
            self.c7 = '#D4E2ED'
            self.c8 = '#B6C8D6'
            self.c9 = '#93AFCD'

        if color_style == 'p22':


            self.c1 = '#1B9E77'
            self.c2 = '#D95F02'
            self.c3 = '#7570B3'
            self.c4 = '#66A61E'
            self.c5 = '#E6AB02'
            self.c6 = '#A6761D'
            self.c7 = '#666666'
            self.c8 = '#EEEEEE'
            self.c9 = '#E7298A'





        self.dict_pt_name = {'LIO_STO': 'Battery-type storage',
                             'CAS_STO': 'Long-term storage',
                             'HYD_STO': 'PHS',
                             'HYD40_STO': 'PHS',
                             'HYD6_STO': 'PHS',
                             'HYD20_STO': 'PHS',
                             'HYD2_STO': 'PHS',
                             'HYD10_STO': 'PHS',
                             }


        self.color_swtc_vl = {'CAS': self.c4, 'LIO': self.c3}

        self.dict_marker_bins_coarse = {'(0,6]': ('s', 1),
                                        '(6,12]': ('h', 1.2),
                                        '(12,24]': ('v', 1.2),
                                        '(24,48]': ('D', 1),
                                        '(48,168]': ('X', 1.1),
                                        '(168,8760)': ('o', 1.1)}



    def make_hard_coded_dicts(self):


        ######### style colors ##############

        self.color_swco_vl = {'05EUR/t_CO2': self.c1,
                              '40EUR/t_CO2': self.c3,
                              '80EUR/t_CO2': self.c5}

        self.color_coarse_bin = \
            {'(0,6]': self.c8,
            '(6,12]': self.c1,
            '(12,24]': self.c6,
            '(24,48]': self.c4,
            '(48,168]': self.c9,
            '(168,8760)': self.c3,
            '(168,10000]': self.c3,


             '(0,16]': (0.988, 0.553, 0.384, 1.0),
             '(0,10]': self.c8,
             '(10,120]': self.c1,
             '(0,120]': self.c6,
             '(120,10000]': (0.702, 0.702, 0.702, 1.0),
             '(16,48]': (0.553, 0.627, 0.796, 1.0),
             '(0,14]': self.c3,
             '(14,48]': self.c6,
             '(0,12]': (0.988, 0.553, 0.384, 1.0),
             '(12,48]': (0.553, 0.627, 0.796, 1.0),
             '(10,48]': (0.553, 0.627, 0.796, 1.0),
             '(48,120]': self.c1,
             '(48,672]': self.c1,
             '(120,672]': self.c5,
             '(672,10000]': self.c9,
             'nan': 'k'}

        self.color_coarse_bin.update({
            ']0,6]': self.c8,
            ']6,12]': self.c1,
            ']12,24]': self.c6,
            ']24,48]': self.c4,
            ']48,168]': self.c9,
            ']168,8760[': self.c3,
            })

        #####################################

        self.color_daytime = {'Noon': self.c1,
                              'Night': self.c6,
                              'Morning': self.c7,
                              'Evening': self.c9,
                              'Afternoon': self.c3}


        self.color_energy_balance = {
                'production': '#00281F',
                'charging': '#037367',
                'grid losses': '#7B895B',
                'demand': '#BEA42E',
                'imports': '#7BD4CC',
                'exports': '#61C0BF',
                }

        self.color_bool_chg = {True: '#3C0C66',
                               False: '#E5853B'}


        self.dict_swtc_name = {'CAS': 'Long-term storage',
                               'LIO': 'Battery-type storage'}

        for tb_name, iind in [('pp_type', 'pt'),
                              ('node', 'nd'),
                              ('fuel', 'fl')]:

            print('--->', self.db, self.sc, 'def_' + tb_name, iind)


            df = aql.read_sql(self.db, self.sc, 'def_' + tb_name).set_index(iind)['color']
            setattr(self, 'color_' + iind, df.to_dict())


        # get color_pp from color_fl
        dict_pp_fl = aql.read_sql(self.db, self.sc, 'def_plant').set_index('pp')['fl_id'].to_dict()
        self.color_pp = {pp: self.color_fl[
                                           self.dict_fl[
                                                        dict_pp_fl[pp]
                                                       ]
                                          ]
                         for pp in dict_pp_fl}


        self.color_dict_cat = {
                                 'Bio and geo':               '#8ECC4A',
                                 'Cogeneration':              '#4159B2',
                                 'Conventional dispatchable': '#8A94B2',
                                 'Reservoir and pumped':      '#7696FF',
                                 'Variable renewable':        '#FFC356',
                                 'Power exported':            '#FFF841',
                                 'Power imported':            '#DF6E21',
                                 'Other storage':             '#80A06F',
                                }


        self.color_swtc_vl = {
         'CAS': '#80A06F',
         'LIO': '#FF624C',
        }


        self.color_country = {c[:2]: self.color_nd[c]
                              for c in self.color_nd}


        # Add color_node to color_pp_type for imports/exports
        self.color_pt.update({'EXPORT_TO_' + k + '0': v for k, v
                              in self.color_country.items()})
        self.color_pt.update({'IMPORT_FROM_' + k + '0': v for k, v
                              in self.color_country.items()})

        self.dict_ctry = {
        'AT': 'Austria',
        'DE': 'Germany',
        'FR': 'France',
        'CH': 'Switzerland',
        'IT': 'Italy',
        }


        self.dict_market_area = {
        'AT': 'DE/AT',
        'DE': 'DE/AT',
        'FR': 'FR',
        'CH': 'CH',
        'IT': 'Italy',
        'AT0': 'DE/AT',
        'DE0': 'DE/AT',
        'FR0': 'FR',
        'CH0': 'CH',
        'IT0': 'Italy',
        }

        self.dict_ctry.update({kk + '0': vv
                               for kk, vv in self.dict_ctry.items()})

        self.dict_ctry_id = {self.dict_nd_id[kk]: vv
                             for kk, vv in self.dict_ctry.items()
                             if kk in self.dict_nd_id.keys()}


        self.time_series_fuel_long_order = \
        [# Negative
         'Exports',
         'Demand',
         'Flexible demand',
         'Pumped h. charg.',
         'Compressed air storage charg.',
         'Li-ion storage charg.',
         'Hybrid r. charg.',
         # Base load
         'Nuclear',
         'Geothermal power',
         'Solid biomass',
         'Lignite',
         'Biogas',
         'All bio',
         'Biofuels',
         'Hard coal',
         'Non-ren. waste',
         'Mixed waste',
         'Waste',
         # Profiles
         'Offshore wind',
         'Run-of-river',
         'Onshore wind',
         'Photovoltaics',
         'Tidal power',

         # Peak
         'Syngas',
         'Natural gas',
         'Mineral oil light',
         'Mineral oil heavy',

         # Flexible peak
         'Hydro reservoirs',
         'Hybrid reservoirs',
         'Pumped h. disch',
         'Compressed air storage',
         'Li-ion storage',
         'Imports',
         'Exchange',
         'none']


        self.pp_type_order = [
         'DMND_FLEX',
         'TRNS_ST',

         'EXPORT_TO_DE0',
         'EXPORT_TO_CH0',
         'EXPORT_TO_FR0',
         'EXPORT_TO_AT0',
         'EXPORT_TO_IT0',


         'HCO_CHP',
         'HCO_ELC',

         'GEO_ELC',
         'NUC_ELC',
         'BAL_ELC',
         'BAL_CHP',


         'GAS_CC_CHP',
         'GAS_CC_ELC',
         'GAS_TB_CHP',
         'GAS_TB_ELC',

         'LIG_CHP',
         'LIG_ELC',

         'WAS_CHP',
         'WAS_ELC',

         'WIN_OFF',
         'WIN_ONS',
         'SOL_PHO',


         'HYD_ROR',
         'HYD_RES',


         'OIL_CHP',
         'OIL_ELC',
         'OIL_PEAK',
        # Peak

         'CAS_STO',
         'LIO_STO',
         'HYD_STO',
         'TRNS_RV',

         'IMPORT_FROM_DE0',
         'IMPORT_FROM_AT0',
         'IMPORT_FROM_CH0',
         'IMPORT_FROM_FR0',
         'IMPORT_FROM_IT0'
        ]



        self.fuel_dict = {
         'reservoir_charg': 'Hybrid r. charg.',
         'pumped_hydro_charg': 'Pumped h. charg.',
         'pumped_hydro': 'Pumped h. disch',
         'biogas': 'Biogas',
         'mineral_oil_light': 'Mineral oil light',
         'wind_onshore': 'Onshore wind',
         'tidal': 'Tidal power',
         'solid_biomass': 'Solid biomass',
         'mineral_oil_heavy': 'Mineral oil heavy',
         'geothermal': 'Geothermal power',
         'run_of_river': 'Run-of-river',
         'photovoltaics': 'Photovoltaics',
         'non-renewable_waste': 'Non-ren. waste',
         'nuclear_fuel': 'Nuclear',
         'natural_gas': 'Natural gas',
         'none': 'none',
         'hard_coal': 'Hard coal',
         'wind_offshore': 'Offshore wind',
         'biowaste': 'Waste',
         'lignite': 'Lignite',
         'biofuels': 'Biofuels',
         'syn_gas': 'Syngas',
         'reservoir': 'Hydro reservoirs',
         'hybrid_reservoir': 'Hybrid reservoirs',
         'power_sent': 'Exports',
         'power_received': 'Imports',
         'export': 'Exports',
         'import': 'Imports',
         'exchange': 'Exchange',
         'caes': 'Compressed air storage',
         'caes_charg': 'Compressed air storage charg.',
         'li_ion': 'Li-ion storage',
         'li_ion_charg': 'Li-ion storage charg.',
         'waste_mix': 'Mixed waste',
         'bio_all': 'All bio',
         'dmnd': 'Demand',
         'dmnd_flex': 'Flexible demand'
        }


        tech_res = [self.fuel_dict[c] for c in self.fuel_dict
                    if not self.fuel_dict[c] in self.time_series_fuel_long_order]
        if len(tech_res) > 0:
            raise ValueError('time_series_tech_order doesn\'t cover all fuel')


        self.time_series_fuel_order = [{self.fuel_dict[c]: c for c in self.fuel_dict}[fl]
                                      for fl in self.time_series_fuel_long_order]


    def get_color_dict(self, iind):
        dict_name = 'color_' + iind
        if hasattr(self, dict_name):
            color_dict = getattr(self, 'color_' + iind)
            color_dict.update({itot: 'black' for itot in
                               ['total', 'all', 'totals', 'Total', 'TOTAL']})
            color_dict.update({itot: 'green' for itot in
                               ['GRDLSS', 'gridlosses']})
            return color_dict

        else:
            return False

    def make_id_dicts(self):

        for inme, iind in [('fuel', 'fl'),
                           ('encar', 'ca'),
                           ('node', 'nd'),
                           ('plant', 'pp'),
                           ('pp_type', 'pt')]:

            df = aql.read_sql(self.db, self.sc, 'def_' + inme).set_index(iind + '_id')
            dct = {ir: df.loc[ir, iind] for ir in df.index}
            setattr(self, 'dict_' + iind, dct)
            df = df.reset_index().set_index(iind)
            dct_id = {ir: df.loc[ir, iind + '_id'] for ir in df.index}
            setattr(self, 'dict_' + iind + '_id', dct_id)

        self.dict_nd_2 = self.dict_nd
        self.dict_nd_2_id = self.dict_nd_id

        df_0 = aql.read_sql(self.db, self.sc, 'def_plant')[['pp_id', 'pt_id', 'nd_id']]
        self.dict_plant_2_node_id = df_0.set_index('pp_id')['nd_id'].to_dict()
        self.dict_plant_2_pp_type_id = df_0.set_index('pp_id')['pt_id'].to_dict()
        df = self.id_to_name(df_0, ['nd'])
        self.dict_plant_2_node = df.set_index('pp_id')['nd_id'].to_dict()
        df = self.id_to_name(df_0, ['pt'])
        self.dict_plant_2_pp_type = df.set_index('pp_id')['pt_id'].to_dict()

    def id_to_name(self, df, name_list):
        for iid in name_list:
            idict = getattr(self, 'dict_' + iid)

            df[iid + '_id'] = df[iid + '_id'].replace(idict)
        return df

# OBSOLETE???
#df_def_plant = aql.read_sql('storage1', 'lp_input', 'def_plant')
#list_st = df_def_plant.loc[df_def_plant.set_def_st == 1,
#                           'pp_id'].unique().tolist()

# OBSOLETE
fuel_dict_short = {
 'reservoir_charg': 'HYB_IN',
 'pumped_hydro_charg': 'PHS_IN',
 'pumped_hydro': 'PHS_OUT',
 'biogas': 'BG',
 'mineral_oil_light': 'OILL',
 'wind_onshore': 'WI_ON',
 'tidal': 'TID',
 'solid_biomass': 'BM',
 'mineral_oil_heavy': 'OILH',
 'geothermal': 'GEO',
 'run_of_river': 'ROR',
 'photovoltaics': 'PV',
 'non-renewable_waste': 'WNR',
 'nuclear_fuel': 'NU',
 'natural_gas': 'NG',
 'none': 'NN',
 'hard_coal': 'HC',
 'wind_offshore': 'WI_OF',
 'biowaste': 'WRN',
 'lignite': 'LIG',
 'biofuels': 'BF',
 'syn_gas': 'SY',
 'reservoir': 'RES',
 'hybrid_reservoir': 'HYB',
 'power_sent': 'EX',
 'power_received': 'IM',
 'export': 'EX',
 'import': 'IM',
 'exchange': 'XC',
 'caes': 'CA_IN',
 'caes_charg': 'CA_OUT',
 'li_ion': 'LI_IN',
 'li_ion_charg': 'LI_OUT',
 'waste_mix': 'WAS',
 'bio_all': 'BIO_ALL',
 'dmnd': 'DMD',
 'dmnd_flex': 'DMD_FLX',
}

dict_broad_cat = {
    'VARIABLE': 'Variable renewable',
    'CONVDISP': 'Conventional dispatchable',
    'RENDISP': 'Bio and geo',
    'CHP': 'Cogeneration',
    'HYDRO': 'Reservoir and pumped',
    'NONE': 'None',
    'SENT': 'Power exported',
    'RCVD': 'Power imported',
    'STONONHYD': 'Other storage'}


fuel_dict = {
 u'bio': 'Bioenergy',
 u'coa': 'Coal',
 u'gas': 'Gas',
 u'geo': 'Geothermal',
 u'hydro': 'Hydropower',
 u'none': 'Others',
 u'nuc': 'Nuclear',
 u'oil': 'Fossil liquids',
 u'sol': 'Solar',
 u'was': 'Waste',
 u'win': 'Wind power',
 u'_hydro_disch': 'Pumped h. discharging',
 u'_hydro_charg': 'Pumped h. charging',
 u'_hybrid_reservoir_charg': 'Hybrid h. charging',
 u'_hybrid_reservoir_disch': 'Hybrid h. discharging',
 u'tidal': 'Tidal',
}

