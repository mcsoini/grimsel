#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import grimsel.auxiliary.sqlutils.aux_sql_func as aql

class DecoratorsSqlAnalysis():
    '''
    Mixin class
    '''


    def append_sw_columns(func_name):
        def _append_sw_columns(f):
            def wrapper(self, *args, **kwargs):
                f(self, *args, **kwargs)
                if len(self.sw_columns) > 0:
                    aql.joinon(self.db, self.sw_columns, ['run_id'],
                               [self.sc_out, func_name],
                               [self.sc_out, 'def_loop'], verbose=True)
            return wrapper
        return _append_sw_columns

    def append_pp_id_columns(func_name):
        def _append_pp_id_columns(f):
            def wrapper(self, *args, **kwargs):
                f(self, *args, **kwargs)
                aql.joinon(self.db, ['pt_id', 'fl_id', 'nd_id', 'pp'], ['pp_id'],
                           [self.sc_out, func_name], [self.sc_out, 'def_plant'])
            return wrapper
        return _append_pp_id_columns

    def append_pt_id_columns(func_name):
        def _append_pt_id_columns(f):
            def wrapper(self, *args, **kwargs):
                f(self, *args, **kwargs)
                aql.joinon(self.db, ['pt'], ['pt_id'],
                           [self.sc_out, func_name], [self.sc_out, 'def_pp_type'])
            return wrapper
        return _append_pt_id_columns

    def append_fl_id_columns(func_name):
        def _append_fl_id_columns(f):
            def wrapper(self, *args, **kwargs):
                f(self, *args, **kwargs)
                aql.joinon(self.db, ['fl'], ['fl_id'],
                           [self.sc_out, func_name], [self.sc_out, 'def_fuel'])
            return wrapper
        return _append_fl_id_columns

    def append_nd_id_columns(func_name):
        def _append_nd_id_columns(f):
            def wrapper(self, *args, **kwargs):
                f(self, *args, **kwargs)
                aql.joinon(self.db, ['nd'], ['nd_id'],
                           [self.sc_out, func_name], [self.sc_out, 'def_node'])
            return wrapper
        return _append_nd_id_columns


