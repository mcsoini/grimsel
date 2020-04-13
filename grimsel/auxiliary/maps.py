

import os
import numpy as np
import pandas as pd
import wrapt
from glob import glob
import re

import grimsel.auxiliary.sqlutils.aux_sql_func as aql
from grimsel.auxiliary.aux_general import silence_pd_warning
from grimsel import _get_logger

logger = _get_logger(__name__)

rev_dict = lambda dct: {val: key for key, val in dct.items()}


class Maps():
    '''
    Transforms definition tables into dictionaries.

    Uses the input tables with prefix *def_* to define dictionaries between
    ids and names. This includes
    - ``dict_node = {node id (int): node name (str)}``
    - ``dict_plant = {plant id (int): plant name (str)}``
    - ``dict_pp_type = {pp_type id (int): pp_type name (str)}``
    - ``dict_fuel = {fuel id (int): fuel name (str)}``
    - ``dict_profile = {profile id (int): profile name (str)}``


    Constructors:
    - default: PSQL schema and database.
    - ``from_df``: List of DataFrames.


    '''

    list_id_tbs = {'fuel': 'fl',
                   'profile': 'pf',
                   'encar': 'ca',
                   'node': 'nd',
                   'plant': 'pp',
                   'pp_type': 'pt',
                   'run': 'run'}

    list_id_tbs_rev = {val: key for key, val in list_id_tbs.items()}


    def __init__(self, sc, db, dict_tb=None):

        self.sc = sc
        self.db = db

        if not dict_tb:
            self._dict_tb = self._read_tables_sql()
        else:
            self._dict_tb = self._adjust_input_tables(dict_tb)

#
#        if 'run' in self._dict_tb and self._dict_tb['run'] is not None:
#            self.list_id_tbs['run'] = list(c for c in self._dict_tb['run']
#                                           if c.endswith('_vl'))

        self._make_id_dicts()
        self._make_color_dicts()


    def _adjust_input_tables(self, dict_tb):
        '''
        Adjust input table formatting.

        Makes sure that the tables provided through the alternative
        initializer :method:``from_dicts`` are formatted in the same way
        as the ones from the SQL input (:method:`_read_tables_sql`).
        '''

        for name, df in dict_tb.items():

            logger.debug('{}, {}'.format(name, Maps.list_id_tbs[name]))
            name_idx = Maps.list_id_tbs[name] + '_id'


            if name_idx in df.columns:

                df.set_index(name_idx, inplace=True)

        return dict_tb


    @classmethod
    def from_dicts(cls, dict_tb):
        '''
        Alternative constructor accepting input tables in a dictionary.

        Args:
            :type:`dict` like ``{short table name (str): DataFrame}``

        '''

        dict_tb = {key: df.copy() for key, df in dict_tb.items()
                   if key in Maps.list_id_tbs}

        self = cls(None, None, dict_tb)

        return self

    @classmethod
    def from_hdf5(cls, fn):

        with pd.HDFStore(fn) as store:

            keys = store.keys()

            dict_tb = {key.replace('/def_', ''): store.get(key)
                       for key in keys
                       if key.replace('/', '').replace('def_', '')
                       in Maps.list_id_tbs}

        if not dict_tb:
            raise IOError('File %s not found.'%fn)

        self = cls(None, None, dict_tb)

        return self

    @classmethod
    def from_parquet(cls, dirc):

        dict_fn_key = {fn: fn.split(os.sep)[-1]
                             .replace('def_', '').replace('.parq', '')
                       for fn in glob(dirc + '/*.parq')}

        dict_tb = {key: pd.read_parquet(fn)
                   for fn, key in dict_fn_key.items()
                   if key in Maps.list_id_tbs}

        if not dict_tb:
            raise IOError('Parquet directory %s not found.'%dirc)

        self = cls(None, None, dict_tb)

        return self


    def _read_tables_sql(self):
        '''
        Reads all relevant tables from the input PSQL schema.

        Returns:
            :returns: :type:`dict` like ``{short table name (str): DataFrame}``

        '''

        list_tb_sql = aql.get_sql_tables(self.sc, self.db)
        dict_tb = {}

        for inme, iind in Maps.list_id_tbs.items():

            tb_name = 'def_' + inme

            if tb_name in list_tb_sql:
                 tb = aql.read_sql(self.db, self.sc, tb_name)
                 tb = tb.set_index(iind + '_id')
                 dict_tb[inme] = tb

        return dict_tb



    def display_colormap(self):
        all_cols = {'c' + str(i): getattr(self, 'c' + str(i))
                    for i in range(1, 200)
                    if 'c' + str(i) in self.__dict__.keys()}

        df = pd.DataFrame(np.abs(np.sin(np.random.randn(2, len(all_cols)))),
                          columns=all_cols.keys())
        df.plot.area(stacked=True, color=[all_cols[c] for c in df.columns])

    def get_color_dict(self, iind, list_complement=[], color_complement='k'):
        '''
        Returns a color dictionary based on the *color* columns in the input.

        Parameters
        ----------
        iind : str
            index like fl, fl_id, pp, pp_id, etc.
        list_complement : list
            list of entries which are not included in the input table but
            which are required in the color dictionary
        color_complement : str
            color string of any format for the complemented color dictionary
            entries

        Returns
        -------
        dict
            dictionary ``{index: color}``

        bool
            Returns ``False`` if the the corresponding table doesn't exist
            or if it has no *color* column.


        '''

        dict_name = '_color_' + iind
        if hasattr(self, dict_name):
            color_dict = getattr(self, '_color_' + iind)

            color_dict_compl = {key: color_complement
                                for key in list_complement
                                if not key in color_dict}

            color_dict.update(color_dict_compl)

            return color_dict

        else:
            return False

    def _make_color_dicts(self):
        '''

        Note:
            Color maps for plants are generated from the plant type color maps,
            if required.
        '''


        inme = 'fuel'
        df = self._dict_tb[inme]
        for inme, df in self._dict_tb.items():

            iind = Maps.list_id_tbs[inme]

            if 'color' in df.columns:
                setattr(self, '_color_%s'%iind, df.set_index(iind)['color']
                                                 .to_dict())

                setattr(self, '_color_%s_id'%iind, df['color'].to_dict())

        if (hasattr(self, 'dict_plant_2_pp_type')
            and hasattr(self, '_color_pt')
            and not hasattr(self, '_color_pp')):
            # generate pp color map from pt

            self._color_pp_id = {pp_id: self._color_pt_id[pt] for pp_id, pt
                                 in self.dict_plant_2_pp_type.items()}
            self._color_pp = {self.dict_pp[pp_id]: col for pp_id, col
                              in self._color_pp_id.items()}

    def _make_id_dicts(self):
        '''
        '''

        for inme, df in self._dict_tb.items():
            if not inme == 'run':
                idx = Maps.list_id_tbs[inme]

                dict_0 = df[idx].to_dict()
                setattr(self, 'dict_%s'%idx, dict_0)
                setattr(self, 'dict_%s_id'%idx, rev_dict(dict_0))

        self.dict_nd_2 = getattr(self, 'dict_nd', None)
        self.dict_nd_2_id = getattr(self, 'dict_nd_id', None)


        self.dict_plant_2_node_id = self._dict_tb['plant']['nd_id'].to_dict()
        self.dict_plant_2_pp_type_id = self._dict_tb['plant']['pt_id'].to_dict()
        self.dict_plant_2_fuel_id = self._dict_tb['plant']['fl_id'].to_dict()


        def get_name_dict(name, id2):
            df = self.id_to_name(self._dict_tb[name][[id2 + '_id']], [id2])
            return df[id2 + '_id'].to_dict()

        self.dict_plant_2_pp_type = get_name_dict('plant', 'pt')
        self.dict_plant_2_node = get_name_dict('plant', 'nd')



    def _add_pp_maps(self, df, inplace=False):

        cols = ['nd_id', 'fl_id', 'pt_id']
        cols = list(set(cols) - set(df.columns))

        pp_df = self._dict_tb['plant'][cols]

        if not inplace:
            df = df.copy()

        return df.join(pp_df, on='pp_id')

    @silence_pd_warning
    def id_to_name(self, df, name_list=None, inplace=False,
                   keep_cols=True):

        if 'pp_id' in df.columns:
            df = self._add_pp_maps(df, inplace)

        if not name_list:
            name_list = [c.replace('_id', '') for c
                         in df.columns if c.endswith('_id')]

        if not inplace:
            df = df.copy()

        for iid in name_list:
            idict = getattr(self, 'dict_' + iid, None)

            if idict:

                col_name = iid + ('_id' if not keep_cols else '')
                df.loc[:, col_name] = df[iid + '_id'].replace(idict)
            elif iid == 'run':
                df = self.run_id_to_names(df)

        return df

    def run_id_to_names(self, df):

        if 'run' in self._dict_tb:
            ddfrun = self._dict_tb['run']
            ddfrun = ddfrun[[c for c in ddfrun.columns
                             if re.match('.*[_vl|_id]$', c)]]

            df = df.join(ddfrun, on='run_id')

        return df

    @wrapt.decorator
    def param_to_list(f, self, *args, **kwargs):
        if not isinstance(args[0][0], (set, list, tuple)):
            args = (([args[0][0]],),)

        return f(*args[0], **kwargs)

    @param_to_list
    def nd2nd(self, list_nd):
        df = self._dict_tb['node']
        if list_nd:
            return set(df.loc[df.nd.str.contains('|'.join(list_nd))].index.tolist())
        else:
            return {}

    @param_to_list
    def nd2pp(self, list_nd):
        if list_nd:
            df = self._dict_tb['plant']
            return set(df.loc[df.nd_id.isin(self.nd2nd(list_nd))].index.tolist())
        else:
            return {}


    @param_to_list
    def pt2pt(self, list_pt):
        if list_pt:
            df = self._dict_tb['pp_type']
            return set(df.loc[df.pt.str.contains('|'.join(list_pt))].index.tolist())
        else:
            return {}

    @param_to_list
    def pp2pp(self, list_pp):
        if list_pp:
            df = self._dict_tb['plant']
            return set(df.loc[df.pp.str.contains('|'.join(list_pp))].index.tolist())
        else:
            return {}

    @param_to_list
    def fl2fl(self, list_fl):
        if list_fl:
            df = self._dict_tb['fuel']
            return set(df.loc[df.fl.str.contains('|'.join(list_fl))].index.tolist())
        else:
            return {}

    @param_to_list
    def pt2pp(self, list_pt):
        if list_pt:
            df = self._dict_tb['plant']
            return set(df.loc[df.pt_id.isin(self.pt2pt(list_pt))].index.tolist())
        else:
            return {}

    @param_to_list
    def fl2pp(self, list_fl):
        if list_fl:
            df = self._dict_tb['plant']
            return set(df.loc[df.fl_id.isin(self.fl2fl(list_fl))].index.tolist())
        else:
            return {}



if __name__ == '__main__':

    self = Maps('lp_input_levels', 'storage2')

#    ml.m.init_maps()
#
#    self = ml.m.mps
