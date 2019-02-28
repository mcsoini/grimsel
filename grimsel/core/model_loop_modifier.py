# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:23:06 2018

@author: martin-c-s
"""


class ModelLoopModifier():
    '''
    The purpose of this class is to modify the parameters of the BaseModel
    objects in dependence on the current run_id.

    This might include the modification of parameters like CO2 prices,
    capacities, but also more profound modifications like the
    redefinition of constraints.
    '''

    def __init__(self, ml):
        '''
        To initialize the ModelBase object is made an instance attribute.
        '''

        self.ml = ml

    def set_value_co2_price(self, dict_co2=None):
        '''
        Example method changing the CO2 prices.
        '''


        sw_name = 'swco'

        if dict_co2 is None:
            dict_co2 = {0: 40, 1: 5, 2: 80}

        slct_co2 = dict_co2[self.ml.dct_step[sw_name]]

        for kk in self.ml.m.price_co2:
            self.ml.m.price_co2[kk].value = slct_co2

        self.ml.dct_vl[sw_name + '_vl'] = str(slct_co2) + 'EUR/t_CO2'

