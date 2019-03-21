#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for minimal model configurations. Input data is defined directly
in the module and written to temporary CSV files. Tests focus on simple model
mechanics and compare the solution objective value to the expected value.

"""

import unittest

import wrapt
import os
import shutil

import numpy as np
import pandas as pd
import grimsel.core.model_base as model_base
import grimsel.core.io as grimsel_io

from grimsel import logger
logger.setLevel(10)

@wrapt.decorator
def write_table(f, _, args, kwargs):

    ret = f(*args, **kwargs)

    fn = 'test_files/{}.csv'.format(ret[1])

    logger.info('Writing {}'.format(fn))

    ret[0].to_csv(fn, index=False)

    return ret

@write_table
def make_def_node():

    df_def_node = pd.DataFrame({'nd': ['Node1'], 'nd_id':[0],
                                'price_co2': [40], 'nd_weight': [1]})
    dict_nd = df_def_node.set_index('nd').nd_id.to_dict()
    return df_def_node, 'def_node', dict_nd

@write_table
def make_def_fuel():

    df_def_fuel = pd.DataFrame({'fl_id': range(3),
                                'fl': ['natural_gas',
                                       'hard_coal',
                                       'wind'],
                                'co2_int': [0.2, 0.3, 0]})
    dict_fl = df_def_fuel.set_index('fl').fl_id.to_dict()
    return df_def_fuel, 'def_fuel', dict_fl

@write_table
def make_def_encar():

    df_def_encar = pd.DataFrame({'ca_id': [0], 'ca': ['EL']})
    dict_ca = df_def_encar.set_index('ca').ca_id.to_dict()
    return df_def_encar, 'def_encar', dict_ca

@write_table
def make_def_pp_type():

    pt = ['GAS_LIN', 'GAS_NEW', 'WIND', 'HCO_ELC']

    df_def_pp_type = pd.DataFrame({'pt_id': range(len(pt)), 'pt': pt})
    dict_pt = df_def_pp_type.set_index('pt').pt_id.to_dict()
    return df_def_pp_type, 'def_pp_type', dict_pt

@write_table
def make_tm_soy():
    df_tm_soy = pd.DataFrame({'sy': range(4), 'weight': 8760 / 4})
    return df_tm_soy, 'tm_soy'

@write_table
def make_def_plant(dict_pt, dict_nd, dict_fl):
    df_def_plant = pd.DataFrame({'pp': ['ND1_GAS_LIN', 'ND1_GAS_NEW',
                                        'ND1_WIND', 'ND1_HCO_ELC'],
                                 'pt_id': ['GAS_LIN', 'GAS_NEW', 'WIND',
                                           'HCO_ELC'],
                                 'nd_id': ['Node1'] * 4,
                                 'fl_id': ['natural_gas', 'natural_gas',
                                           'wind', 'hard_coal'],
                                 'set_def_pr': [0, 0, 1, 0],
                                 'set_def_pp': [1, 1, 0, 1],
                                 'set_def_lin': [1, 0, 0, 0],
                                 'set_def_add': [0, 1, 1, 0],
                                })

    df_def_plant.index.name = 'pp_id'
    df_def_plant = df_def_plant.reset_index()

    # translate columns to id using the previously defined def tables
    df_def_plant = df_def_plant.assign(pt_id=df_def_plant.pt_id.replace(dict_pt),
                                       nd_id=df_def_plant.nd_id.replace(dict_nd),
                                       fl_id=df_def_plant.fl_id.replace(dict_fl))
    dict_pp = df_def_plant.set_index('pp').pp_id.to_dict()

    return df_def_plant, 'def_plant', dict_pp

@write_table
def make_def_profile():

    df_def_profile = pd.DataFrame({'pf_id': range(2),
                                   'pf': ['SUPPLY_WIND', 'DMND_NODE1']})
    df_def_profile = pd.DataFrame({'pf_id': [0], 'pf': ['DMND_NODE1']})
    dict_pf = df_def_profile.set_index('pf').pf_id.to_dict()
    return df_def_profile, 'def_profile', dict_pf


@write_table
def make_plant_encar(dict_pp, dict_ca, dict_pf={}):

    eff_gas_min = 0.4
    eff_gas_max = 0.6
    cap_gas = 7000.
    f0_gas = 1/eff_gas_min
    f1_gas = 1/cap_gas * (f0_gas - 1/eff_gas_max)

    dr, lt = 0.06, 20 # assumed discount rate 6% and life time
    fact_ann = ((1+dr)**lt * dr) / ((1+dr)**lt - 1)

    fc_cp_gas = 800
    fc_cp_gas_ann = fact_ann * fc_cp_gas

    fc_cp_wind = 1500 # assumed capital cost wind power
    fc_cp_wind_ann = fact_ann * fc_cp_wind

    df_plant_encar = pd.DataFrame({'pp_id': ['ND1_GAS_LIN', 'ND1_GAS_NEW', 'ND1_WIND', 'ND1_HCO_ELC'],
                                   'ca_id': ['EL'] * 4,
                                   'supply_pf_id': [None, None, 'SUPPLY_WIND', None],
                                   'pp_eff': [None, eff_gas_max, None, 0.4],
                                   'factor_lin_0': [f0_gas, None, None, None],
                                   'factor_lin_1': [f1_gas, None, None, None],
                                   'cap_pwr_leg': [cap_gas, 0, 0, 10000],
                                   'fc_cp_ann': [None, fc_cp_gas_ann, fc_cp_wind_ann, None],
                                  })

    df_plant_encar = df_plant_encar.assign(pp_id=df_plant_encar.pp_id.replace(dict_pp),
                                           ca_id=df_plant_encar.ca_id.replace(dict_ca))

    if 'supply_pf_id' in df_plant_encar.columns:
        df_plant_encar.supply_pf_id = df_plant_encar.supply_pf_id.replace(dict_pf)


    return df_plant_encar, 'plant_encar'

@write_table
def make_node_encar(dict_nd, dict_ca, dict_pf={}):

    df_node_encar = pd.DataFrame({'nd_id': ['Node1'], 'ca_id': ['EL'],
                                  'dmnd_pf_id': ['DMND_NODE1']
                                  })
    df_node_encar = df_node_encar.assign(nd_id=df_node_encar.nd_id.replace(dict_nd),
                                         ca_id=df_node_encar.ca_id.replace(dict_ca),
                                         dmnd_pf_id=df_node_encar.dmnd_pf_id.replace(dict_pf))

    return df_node_encar, 'node_encar'

@write_table
def make_fuel_node_encar(dict_fl, dict_nd, dict_ca):

    df_fuel_node_encar = pd.DataFrame({'fl_id': ['natural_gas', 'hard_coal'],
                                       'nd_id': ['Node1'] * 2,
                                       'ca_id': ['EL'] * 2,
                                       'vc_fl': [40, 10],
                                       })

    df_fuel_node_encar = df_fuel_node_encar.assign(fl_id=df_fuel_node_encar.fl_id.replace(dict_fl),
                                           nd_id=df_fuel_node_encar.nd_id.replace(dict_nd),
                                           ca_id=df_fuel_node_encar.ca_id.replace(dict_ca))
    return df_fuel_node_encar, 'fuel_node_encar'


@write_table
def make_profsupply(dict_pf):
    prf = [0.169, 0.122, 0.176, 0.284]

    df_profsupply = pd.DataFrame({'supply_pf_id': [dict_pf['SUPPLY_WIND']] * len(prf),
                                  'hy': range(len(prf)), 'value': prf})
    return df_profsupply, 'profsupply'


@write_table
def make_profdmnd(dict_pf):
    prf = [6500, 6000, 6500, 6800]

    df_profdmnd = pd.DataFrame({'dmnd_pf_id': [dict_pf['DMND_NODE1']] * len(prf),
                                'hy': range(len(prf)), 'value': prf})

    return df_profdmnd, 'profdmnd'




# %%

class ModelCaller():
    '''
    Tests construct input tables saved as csv, ModelCaller calls the model
    resulting from these files.

    '''

    iokwargs_default = {'cl_out': 'tmp.hdf5',
                        'no_output': True,
                        'data_path': 'test_files', # config.PATH_CSV,
                        'output_target': 'hdf5',
                        'dev_mode': True}
    mkwargs_default = {'tm_filt': [('hy', range(4))],
                       'symbolic_solver_labels': True}


    def __init__(self, iokwargs=None, mkwargs=None):

        self.iokwargs = iokwargs if iokwargs else self.iokwargs_default.copy()
        self.mkwargs = mkwargs if mkwargs else self.mkwargs_default.copy()


    def run_model(self, hold=False):

        m = model_base.ModelBase(**self.mkwargs)
        io = grimsel_io.IO(**self.iokwargs, model=m)

        grimsel_io.IO._close_all_hdf_connections()
        io.read_model_data()

        m.init_maps()
        m.map_to_time_res()

        m.get_setlst()
        m.define_sets()
        m.add_parameters()

        m.define_variables()
        m.add_all_constraints()
        m.init_solver()

        if not hold:
            m.run()

        return m

        print('\n' * 2)


class UpDown():

    def setUp_0(self):

        try:
            shutil.rmtree('test_files')
        except Exception as e:
            logger.error(('Encountered error {e} when trying to delete '
                          'directory test_files in test class {cl}'
                          ).format(e=e, cl=self.__class__))

        os.mkdir('test_files')

    def tearDown_0(self):

        for fn in [f for f in list(os.walk('.'))[0][2] if f.startswith('tmp')]:
            os.remove(fn)

        for fn in [f for f in list(os.walk('./test_files'))[0][2]]:
            os.remove('test_files/' + fn)



class TestFuelAndCO2Cost(unittest.TestCase, UpDown):

    def setUp(self):

        super(TestFuelAndCO2Cost, self).setUp_0()

        _, _, dict_nd = make_def_node()
        _, _ = make_tm_soy()
        _, _, dict_ca = make_def_encar()
        _, _, dict_pt = make_def_pp_type()
        _, _, dict_fl = make_def_fuel()
        _, _, dict_pf = make_def_profile()
        _, _, dict_pp = make_def_plant(dict_pt, dict_nd, dict_fl)
        _, _ = make_fuel_node_encar(dict_fl, dict_nd, dict_ca)
        _, _ = make_node_encar(dict_nd, dict_ca, dict_pf)
        _, _ = make_plant_encar(dict_pp, dict_ca)
        _, _ = make_plant_encar(dict_pp, dict_ca)
        _, _ = make_profdmnd(dict_pf)

    def tearDown(self):

        super().tearDown_0()

    def test_linear_fuel_and_co2_cost(self):

        # ~~~~~~~~~ fuel price only

        mc = ModelCaller()
        mc.mkwargs['slct_pp_type'] = ['GAS_LIN']
        m = mc.run_model(hold=True)
        for key in m.price_co2: m.price_co2[key] = 0
        m.run()


        eff_gas_min = 0.4
        eff_gas_max = 0.6
        cap_gas = 7000.
        f0_gas = 1/eff_gas_min
        f1_gas = 1/cap_gas * (f0_gas - 1/eff_gas_max)

        dmnd = np.array([6500, 6000, 6500, 6800])

        vc_fl = 40

        cost_total = 8760 / 4 * sum(dmnd * vc_fl
                                    * (f0_gas + 0.5 * f1_gas * dmnd))

        self.assertEqual(int(m.objective_value * 1e5) / 1e5, cost_total)

        # ~~~~~~~~~ co2 price only

        m.reset_all_parameters()
        for key in m.vc_fl: m.vc_fl[key] = 0
        m.run()

        price_co2 = 40
        co2_int = 0.2

        cost_total = 8760 / 4 * sum(dmnd * price_co2 * co2_int
                                    * (f0_gas + dmnd * f1_gas * 0.5))

        self.assertAlmostEqual(round(m.objective_value * 1e5) / 1e5,
                               round(cost_total * 1e5) / 1e5)

    def test_constant_fuel_and_co2_cost(self):

        # ~~~~~~~~~ fuel price only

        mc = ModelCaller()
        mc.mkwargs['slct_pp_type'] = ['HCO_ELC']
        m = mc.run_model(hold=True)
        for key in m.price_co2: m.price_co2[key] = 0
        m.run()

        eff_hco = 0.4
        dmnd = np.array([6500, 6000, 6500, 6800])

        vc_fl = 10

        cost_total = 8760 / 4 * sum(dmnd * vc_fl / eff_hco)

        self.assertEqual(round(m.objective_value * 1e5) / 1e5, cost_total)

        # ~~~~~~~~~ co2 price only

        m.reset_all_parameters()
        for key in m.vc_fl: m.vc_fl[key] = 0
        m.run()

        price_co2 = 40
        co2_int = 0.3

        cost_total = 8760 / 4 * sum(dmnd * price_co2 * co2_int / eff_hco)

        self.assertEqual(round(m.objective_value * 1e5) / 1e5, cost_total)


class TestFixedCapitalAndOMCost(unittest.TestCase, UpDown):

    def tearDown(self):

        super().tearDown_0()

    def setUp(self):

        super(TestFixedCapitalAndOMCost, self).setUp_0()

        _, _, dict_nd = make_def_node()
        _, _ = make_tm_soy()
        _, _, dict_ca = make_def_encar()
        _, _, dict_pt = make_def_pp_type()
        _, _, dict_fl = make_def_fuel()
        _, _, dict_pf = make_def_profile()
        _, _, dict_pp = make_def_plant(dict_pt, dict_nd, dict_fl)
        _, _ = make_fuel_node_encar(dict_fl, dict_nd, dict_ca)
        _, _ = make_node_encar(dict_nd, dict_ca, dict_pf)
        _, _ = make_plant_encar(dict_pp, dict_ca)
        _, _ = make_plant_encar(dict_pp, dict_ca)
        _, _ = make_profdmnd(dict_pf)

    def test_fixed_cost(self):

        # ~~~~~~~~~ O&M full year

        mc = ModelCaller()
        mc.mkwargs['slct_pp_type'] = ['GAS_NEW']
        m = mc.run_model(hold=True)
        for key in m.price_co2: m.price_co2[key] = 0
        for key in m.vc_fl: m.vc_fl[key] = 0
        for key in m.fc_cp_ann: m.fc_cp_ann[key] = 0
        m.run()

        fc_om_gas = 24000
        dmnd = np.array([6500, 6000, 6500, 6800])
        cap_new = dmnd.max()

        cost_total = fc_om_gas * cap_new

        self.assertEqual(int(m.objective_value * 1e5) / 1e5, cost_total)

        # ~~~~~~~~~~~ capital cost full year

        m.dict_par['fc_cp_ann'].init_update()
        for key in m.fc_om: m.fc_om[key] = 0
        m.run()

        dr, lt = 0.06, 20 # assumed discount rate 6% and life time
        fact_ann = ((1+dr)**lt * dr) / ((1+dr)**lt - 1)
        fc_cp_gas = 0.8*1e6
        fc_cp_gas_ann = round(fact_ann * fc_cp_gas/10000) * 10000
        dmnd = np.array([6500, 6000, 6500, 6800])
        cap_new = dmnd.max()

        cost_total = fc_cp_gas_ann * cap_new

        self.assertEqual(int(m.objective_value * 1e5) / 1e5, cost_total)

    def test_fixed_cost_part_year(self):

        # ~~~~~~~~~ O&M part year

        red = [0]

        _, _ = make_tm_soy(red=red) # overwrite
        _, _, dict_pf = make_def_profile() # overwrite
        _, _ = make_profdmnd(dict_pf, red=red) # overwrite

        mc = ModelCaller()
        mc.mkwargs['slct_pp_type'] = ['GAS_NEW']
        m = mc.run_model(hold=True)
        for key in m.price_co2: m.price_co2[key] = 0
        for key in m.vc_fl: m.vc_fl[key] = 0
        for key in m.fc_cp_ann: m.fc_cp_ann[key] = 0
        m.run()


        fc_om_gas = 24000
        dmnd = np.array([6500, 6000, 6500, 6800])[red]
        cap_new = dmnd.max()

        cost_total = fc_om_gas * cap_new * len(red)/4

        self.assertEqual(int(m.objective_value * 1e5) / 1e5, cost_total)

        # ~~~~~~~~~~~ capital cost part year

        m.dict_par['fc_cp_ann'].init_update()
        for key in m.fc_om: m.fc_om[key] = 0
        m.run()

        dr, lt = 0.06, 20 # assumed discount rate 6% and life time
        fact_ann = ((1+dr)**lt * dr) / ((1+dr)**lt - 1)
        fc_cp_gas = 0.8*1e6
        fc_cp_gas_ann = round(fact_ann * fc_cp_gas/10000) * 10000

        cost_total = fc_cp_gas_ann * cap_new * len(red)/4

        self.assertEqual(int(m.objective_value * 1e5) / 1e5, cost_total)


if __name__ == '__main__':

    unittest.main()



