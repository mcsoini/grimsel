#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 07:41:11 2018

@author: mcsoini
"""

import pandas as pd
import itertools as it
import matplotlib as mpl

import abc

import logging

    list_acclasses = []
logging.getLogger().setLevel(logging.INFO)


class AutoComplete(abc.ABC):
    '''
    Base class for the autocompletion of model input DataFrames.
    '''

    def __init__(self, m):

        format_list = [self.df_name, type(self).__name__]
        logging.info('Autocompletion '
                     '{} in {}'.format(*format_list))

        self.flag_add = True
        self.m = m

        # determine whether autocompletion is feasible based on
        # available model dataframes
        flag_feas = True
        if 'lst_req_df' in self.__dict__.keys():
            lst_mss_df = [df for df in self.lst_req_df
                          if (not df in self.m.__dict__.keys()
                              or getattr(self.m, df) is None)]
            if lst_mss_df:
                # set flag to False if any of the required dataframes are missing
                flag_feas = False

        if flag_feas:

            # Control attributes need to be assigned in child classes, otherwise
            # they are None, which won't work.
            if not '_df' in self.__dict__.keys():
                self._df = None # DataFrame to be completed
            if not '_add_col' in self.__dict__.keys():
                self._add_col = None # Reference column(s)

            self.df_add = None
            self.lst_add = []

            self.get_row_list()
            self.filter_rows()
            self.check_add()

            if self.flag_add:
                self.generate_df_add()
                self.reset_index()
                self.complement_columns()
                self.concatenate()

                logging.info('done. Added: '
                             '{}'.format(', '.join(map(str, self.lst_add))))
            else:
                logging.info('nothing added.')
        else:
            logging.info('infeasible. Missing model DataFrames: '
                         '{}'.format(', '.join(lst_mss_df)))




    def filter_row_list(self):
        ''' Filter rows by additionality or relevance. '''
        pass

    def check_add(self):
        ''' Checks whether there are any rows to be added. '''

        # set flag_add to False/0 if lst_add is empty
        self.flag_add *= (True if self.lst_add else False)

    @abc.abstractmethod
    def get_row_list(self):
        ''' List of rows to potentially be added. Implemented in children. '''

        pass

    def filter_rows_existing(self):
        ''' Remove rows already included in _df. By names in _add_col. '''

        # filter for existing rows
        # everything based on list of tuples; converted accordingly if necessary
        ttt = lambda x: tuple(x)
        slct_col = ([self._add_col]
                    if not type(self._add_col) is list
                    else self._add_col)
        self.lst_add = [pc for pc in self.lst_add
                        if not tuple([pc]
                                     if not type(pc) is list
                                     else pc)
                        in self._df[slct_col].apply(ttt, axis=1).tolist()]

    def filter_rows(self):
        ''' Overridden by child classes if other filters need to be applied '''

        self.filter_rows_existing()

    def generate_df_add(self):
        '''
        Generate full DataFrame to be appended to _df.
        '''

        # some children assign df_add before this (e.g. for pp name generation)
        if self.df_add is None:

            cols = (self._add_col
                    if type(self._add_col) is list
                    else [self._add_col])
            self.df_add = pd.DataFrame(self.lst_add, columns=cols)

    def reset_index(self):
        ''' Implemented in children. Calls _reset_index if required. '''
        pass

    def _reset_index(self):
        '''
        Reset index so it starts from 0, then shift to maximum of _df plus one.

        This is called from the child classes, since not relevant for all.
        E.g. not for non-def_ dataframes, which don't have a
        '''

#        print('Resetting index.')
        id_col = self._add_col + '_id'
        self.df_add = (self.df_add.reset_index(drop=True).reset_index()
                                  .rename(columns={'index': id_col}))
        self.df_add[id_col] += self._df[id_col].max() + 1

    def add_color_col(self, cmap=None):

        _cmap = mpl.cm.get_cmap(cmap)

        self.df_add['color'] = ''

        # reset index so we can iterate using loc
        self.df_add = self.df_add.reset_index(drop=True)

        for irow in range(self.df_add.iloc[:, 0].size):
            if cmap is None:
                color = '#ffffff'
            else:
                color = mpl.colors.to_hex(_cmap(irow))

            self.df_add.loc[irow, 'color'] = color


    def add_zero_cols(self):
        ''' Check which columns are missing in df_add and fill with zeros. '''

        list_cols = [c for c in self._df.columns
                     if not c in self.df_add.columns]

        for icol in list_cols:
            self.df_add[icol] = 0


    def concatenate(self):
        ''' Concatenate existing and new rows. '''

        add_cols = [c for c in self.df_add.columns if c in self._df.columns]
        setattr(self.m, self.df_name,
                pd.concat([self._df, self.df_add[add_cols]], sort=False))
        self._df = getattr(self.m, self.df_name)


    def complement_columns(self):
        ''' Implemented in child classes if additional columns are required '''
        pass

    def _get_dmnd_list(self, name_type):
        '''
        Returns a list of the relevant fuel or pp_type names.

        The returned list depends on the ``autocomplete_curtailment``
        bool value.

        Args:
            name_type (str): one of ('fuel', 'plant')

        Returns:
            list. The selected list of demand-like plants or fuels.
        '''

        if name_type == 'fuel':
            flx, dmd = 'dmnd_flex', 'dmnd'
        elif name_type == 'plant':
            flx, dmd = 'DMND_FLEX', 'DMND'

        return ([flx, dmd] if self.autocomplete_curtailment else [dmd])



class AutoCompletePpType(AutoComplete):

    df_name = 'df_def_pp_type'

    def __init__(self, m, autocomplete_curtailment):

        self._df = getattr(m, self.df_name)
        self._add_col = 'pt'

        self.autocomplete_curtailment = autocomplete_curtailment

        super().__init__(m)

    def get_row_list(self):
        ''' Static pt names plus one for each energy carrier in def_encar. '''

        self.lst_add = ['TRNS_ST', 'TRNS_RV'] + self._get_dmnd_list('plant')
        self.lst_add += ['CONS_' + ca for ca in self.m.df_def_encar.ca]

#        print('Potential additions: {}'.format(', '.join(self.lst_add)))

    def reset_index(self):
        ''' Calls parent _reset_index method. '''

        super()._reset_index()

    def complement_columns(self):
        ''' Add pp_broad_cat and color. '''

        self.df_add['pp_broad_cat'] = self.df_add['pt']
        self.add_color_col('Set1')

class AutoCompleteFuel(AutoComplete):

    df_name = 'df_def_fuel'
    def __init__(self, m):

        self._df = getattr(m, self.df_name)
        self._add_col = 'fl'

        # define column is_ca based on entries of df_def_encar
        self._df['is_ca'] = 0
        mask_ca = (self._df.fl_id.isin(m.df_def_encar.fl_id))
        self._df.loc[mask_ca, 'is_ca'] = 1

        super().__init__(m)

    def reset_index(self):
        ''' Calls parent class _reset_index '''

        self._reset_index()

    def complement_columns(self):

        self.add_color_col('Set3')
        self.add_zero_cols()

class AutoCompleteTrns():
    ''' Mixin class holding methods relevant for inter-node transmission. '''

    def filter_multi_node(self):
        ''' Delete all entries if there is only one node.'''

        self.lst_add = (self.lst_add
                        if self.m.df_def_node.nd_id.count() > 1
                        else [])

class AutoCompleteFuelTrns(AutoCompleteFuel, AutoCompleteTrns):

    def __init__(self, m):

        super().__init__(m)

    def get_row_list(self):
        ''' Static fl names. '''

        self.lst_add = ['import', 'export']

    def filter_rows(self):

        super().filter_multi_node()
        super().filter_rows_existing()



class AutoCompleteFuelConsumed(AutoCompleteFuel):

    def __init__(self, m):

        super().__init__(m)

    def get_row_list(self):
        ''' fl names generated from ca names in df_def_encar. '''

        self.lst_add = self.m.df_def_encar['ca'].tolist()
        self.lst_add = ['consumed_' + fl.lower() for fl in self.lst_add
                        if not fl in self._df[self._add_col]]

class AutoCompletePlant(AutoComplete):

    df_name = 'df_def_plant'
    def __init__(self, m):

        self._df = getattr(m, self.df_name)
        self._add_col = 'pp'

        super().__init__(m)

    def reset_index(self):
        ''' Calls parent class _reset_index '''

        self._reset_index()

    def add_id_cols(self, add_names=False):
        '''
        Add id columns based on name columns.

        Argument:
        add_names -- bool; add name columns based on id columns instead.
        '''

        for name, df_def in [('nd', 'df_def_node'),
                             ('fl', 'df_def_fuel'),
                             ('pt', 'df_def_pp_type')]:

            _name = name
            _id = name + '_id'

            if add_names:
                # reverse
                _name = name + '_id'
                _id = name

            if ((not _id in self.df_add.columns)
                and (_name in self.df_add.columns)):

                mp = getattr(self.m, df_def).set_index(_name)[_id]
                self.df_add = self.df_add.join(mp, on=_name)

class AutoCompletePlantTrns(AutoCompletePlant, AutoCompleteTrns):

    def __init__(self, m):
        # list of required model dataframes

        self.lst_req_df = ('df_node_connect',)

        super().__init__(m)

    def get_row_list(self):
        ''' pp names from reshaped df_node_connect table. '''

        mask_trns = ((self.m.df_node_connect.cap_trmi_leg != 0) |
                     (self.m.df_node_connect.cap_trme_leg != 0))
        self.df_add = (self.m.df_node_connect
                           .loc[mask_trns, ['nd_id', 'nd_2_id']]
                           .drop_duplicates()
                           .stack()
                           .reset_index()
                           .rename(columns={'level_1': 'imex', 0: 'nd_id'})
                           .join(self.m.df_def_node.set_index('nd_id')['nd'],
                                 on='nd_id')[['imex', 'nd_id', 'nd']]
                           .drop_duplicates())

        pt_dict = {'nd_id': 'TRNS_ST',
                   'nd_2_id': 'TRNS_RV'}
        fl_dict = {'nd_id': 'export',
                   'nd_2_id': 'import'}

        self.df_add['pt'] = self.df_add['imex'].replace(pt_dict)
        self.df_add['fl'] = self.df_add['imex'].replace(fl_dict)
        self.df_add['pp'] = (self.df_add['nd'].apply(lambda x: x[:2] + '_')
                             + self.df_add['pt'])

        self.lst_add = self.df_add['pp'].tolist()

    def filter_rows(self):

        super().filter_multi_node()
        super().filter_rows_existing()

    def complement_columns(self):

        self.add_color_col('Accent')
        self.add_id_cols()
        self.add_zero_cols()

        self.df_add['set_def_tr'] = 1


class AutoCompleteFuelDmnd(AutoCompleteFuel):

    def __init__(self, m, autocomplete_curtailment):

        self.autocomplete_curtailment = autocomplete_curtailment

        super().__init__(m)


    def get_row_list(self):
        ''' Static fl names. '''

        self.lst_add = self._get_dmnd_list('fuel')

class AutoCompletePlantDmnd(AutoCompletePlant):

    def __init__(self, m, autocomplete_curtailment):

        self.autocomplete_curtailment = autocomplete_curtailment

        super().__init__(m)

    def get_row_list(self):
        ''' Cross product fixed pt names and nodes. '''

        self.lst_add = self.m.df_def_node['nd'].apply(lambda x: x[:2]).tolist()
        self.lst_add = ['_'.join(pp) for pp in
                        list(it.product(self.lst_add,
                                        self._get_dmnd_list('plant')))]

    def complement_columns(self):

        get_nd = lambda x: x.replace('_DMND', '').replace('_FLEX', '')
        self.df_add['nd'] = self.df_add.pp.apply(get_nd)

        get_pt = lambda x: x.pp.replace(x.nd + '_', '')
        self.df_add['pt'] = self.df_add[['pp', 'nd']].apply(get_pt, axis=1)

        self.df_add['fl'] = self.df_add['pt'].apply(str.lower)

        self.add_id_cols()
        self.add_zero_cols()

        mask_flex = self.df_add.pt.str.contains('FLEX')
        self.df_add.loc[mask_flex, 'set_def_curt'] = 1


class AutoCompletePlantCons(AutoCompletePlant):

    def __init__(self, m):

        super().__init__(m)

    def get_row_list(self):
        ''' Plants in df_def_plant which consume energy carriers. '''

        self.df_add = self._df.loc[self._df.fl_id.isin(self.m.df_def_encar.fl_id.tolist()), ['nd_id', 'fl_id']].drop_duplicates()
        self.df_add = self.df_add.join(self.m.df_def_encar.set_index('fl_id')['ca'], on='fl_id')

        # adding nd column
        self.add_id_cols(add_names=True)

        self.df_add['pt'] = 'CONS_' + self.df_add['ca']
        self.df_add['pp'] = self.df_add.nd + '_' + self.df_add.pt

        self.lst_add = self.df_add.pp.tolist()

    def complement_columns(self):

        self.add_id_cols()
        self.add_zero_cols()


class AutoCompletePpCa(AutoComplete):

    df_name = 'df_plant_encar'
    completion_type = ''

    def __init__(self, m):

        self._df = getattr(m, self.df_name)
        self._add_col = ['pp_id', 'ca_id']

        super().__init__(m)

    def filter_rows(self):
        self.filter_rows_existing()


    def complement_columns(self):

        self.add_zero_cols()


class AutoCompletePpCaFlex(AutoCompletePpCa):

    def __init__(self, m, autocomplete_curtailment):

        self.autocomplete_curtailment = autocomplete_curtailment

        super().__init__(m)

    def get_row_list(self):
        ''' . '''

        if self.autocomplete_curtailment:

            mask_flex = self.m.df_def_plant.pp.str.contains('FLEX')
            list_pp = self.m.df_def_plant.loc[mask_flex, 'pp_id'].tolist()
            list_ca = self.m.df_def_encar['ca_id'].tolist()

            self.lst_add = [list(pc) for pc in it.product(list_pp, list_ca)]
        else:
            self.lst_add = []


if __name__ == '__main__':

    self = AutoCompletePpType(ml.m)
    self = AutoCompleteFuelTrns(ml.m)
    self = AutoCompleteFuelDmnd(ml.m)
    self = AutoCompletePlantTrns(ml.m)
    self = AutoCompletePlantDmnd(ml.m)
    self = AutoCompletePlantCons(ml.m)
    self = AutoCompletePpCaFlex(ml.m)

# %%






