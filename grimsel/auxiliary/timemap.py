'''
The timemap module contains the TimeMap class for

* mapping between hours of the year and model time slots of the year,
* convenient mapping between time slots/hours and other temporal indices
  (month ids, week ids, seasons, hours of the week/month, etc)

'''

import pandas as pd
import numpy as np
from grimsel import _get_logger

logger = _get_logger(__name__)

MONTH_DICT = {0:'JAN', 1:'FEB', 2:'MAR', 3:'APR', 4:'MAY', 5:'JUN',
               6:'JUL', 7:'AUG', 8:'SEP', 9:'OCT', 10:'NOV', 11:'DEC'}

SEASON_DICT = {'JAN': 'WINTER', 'FEB': 'WINTER', 'DEC': 'WINTER',
               'MAR': 'SPRING', 'APR': 'SPRING', 'MAY': 'SPRING',
               'JUN': 'SUMMER', 'JUL': 'SUMMER', 'AUG': 'SUMMER',
               'SEP': 'FALL', 'OCT': 'FALL', 'NOV': 'FALL'}
DOW_DICT = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU',
            4: 'FRI', 5: 'SAT', 6: 'SUN'}
DOW_TYPE_DICT = {**{d: 'WEEKDAY' for d in range(5)}, **{5: 'SAT', 6: 'SUN'}}

TM_DICT = {}

def _tm_hash(nhours, freq, start, stop, tm_filt):

    hash_val = hash((nhours, freq, start, stop, str(tm_filt)))
    return hash_val

class _UniqueInstancesMeta(type):
    '''
    Load ``TimeMap`` instance from the module dictionary, if exists.

    To avoid the generation of multiple time maps with the same
    parameters, they are stored in the ``TM_DICT`` module attribute
    and retrieved when appropriate.

    '''

    def __call__(cls, nhours=1, freq=1,
                 start='2015-1-1 00:00', stop='2015-12-31 23:59',
                 tm_filt=False, *args, **kwargs):

        key = _tm_hash(nhours, freq, start, stop, tm_filt)

        if key in TM_DICT:
            logger.warning(('TimeMap (%s) exists. Reading time '
                         'map from module dict.')%key)
            return TM_DICT[key]
        else:
            return super().__call__(nhours, freq, start, stop, tm_filt,
                                    *args, **kwargs)

class TimeMap(metaclass=_UniqueInstancesMeta):
    '''
    The ``TimeMap`` class generates various DataFrames for time mapping.

    For example, if we start from input data with 15 minutes (0.25 hour)
    time resolution and run the model with 2 hour time resolution, we
    need a ``TimeMap`` instance with ``freq=0.25`` and ``nhours=2``.

    After calling the :func:`gen_soy_timemap` (which occurs automatically
    in the ``__init__`` method) three attribute ``DataFrames`` are generated:

    * ``df_time_map``, indexed by the hour of the year *hy*
    * ``df_time_red``, reduced time map indexed by the time slot of the year *sy*
    * ``df_hoy_soy``, a minimal map between the *sy* and the *hy*

    Args
    ----
    nhours : float
        target time resolution in hours
    freq : float
        frequency of the base time map in hours
    start : str
        start datetime of time map; format ``2015-1-1 00:00``
    stop : str
        end datetime of time map; format ``2015-1-1 00:00``
    tm_filt : list
        list of tuples with shape ``(timemap_column, [values])``
        for time map filtering. For example,
        ``tm_filt=[('dow', [0, 1]), ('mt_id', [2])]`` generates a
        ``TimeMap`` instance whose tables only contain the days of the
        week 0 and 1 for the *mt_id* 2
    keep_datetime : bool
        Remove the *DateTime* column if False
    minimum : bool
        don't generate non-essential columns

    Raises
    ------
    AssertionError
        If ``nhours`` is not a multiple of ``freq``.

    Examples
    --------

    Initialize a ``TimeMap`` instance mapping the original 0.25 hour time
    resolution to 1.25 hours:

    >>> tm = TimeMap(nhours=1.25, freq=0.25, tm_filt=[('mt_id', [0]), ('hom', range(2))], minimum=True)

    Base ``df_time_map`` table:

    >>> print(tabulate.tabulate(tm.df_time_map, headers=tm.df_time_map.columns, tablefmt='rst'))
    ====  ======  =====  =======  =====  ======  =====  =====  =====  =======  ===========  ====  ====
      ..    hour    doy    mt_id    day    year    dow    how    hom    wk_id    wk_weight    hy    sy
    ====  ======  =====  =======  =====  ======  =====  =====  =====  =======  ===========  ====  ====
       0       0      1        0      1    2015      3     72      0        0          384  0        0
       1       0      1        0      1    2015      3     72      0        0          384  0.25     0
       2       0      1        0      1    2015      3     72      0        0          384  0.5      0
       3       0      1        0      1    2015      3     72      0        0          384  0.75     0
       4       1      1        0      1    2015      3     73      1        0          384  1        0
       5       1      1        0      1    2015      3     73      1        0          384  1.25     1
       6       1      1        0      1    2015      3     73      1        0          384  1.5      1
       7       1      1        0      1    2015      3     73      1        0          384  1.75     1
    ====  ======  =====  =======  =====  ======  =====  =====  =====  =======  ===========  ====  ====

    Reduced ``df_time_red`` table:

    >>> print(tabulate.tabulate(tm.df_time_red, headers=tm.df_time_red.columns, tablefmt='rst'))
    ====  ======  ====  =====  =====  =====  =====  ======  =====  ====  =======  ========  =======  ===========
      ..    year    sy    day    dow    doy    hom    hour    how    hy    mt_id    weight    wk_id    wk_weight
    ====  ======  ====  =====  =====  =====  =====  ======  =====  ====  =======  ========  =======  ===========
       0    2015     0      1      3      1      0       0     72  0           0      1.25        0          384
       1    2015     1      1      3      1      1       1     73  1.25        0      0.75        0          384
    ====  ======  ====  =====  =====  =====  =====  ======  =====  ====  =======  ========  =======  ===========

    Minimal convenience table ``df_hoy_soy`` only containing the *hy* |rarr| *sy*
    map:

    >>> print(tabulate.tabulate(tm.df_hoy_soy, headers=tm.df_hoy_soy.columns, tablefmt='rst'))
    ====  ====  ====
      ..    sy    hy
    ====  ====  ====
       0     0  0
       1     0  0.25
       2     0  0.5
       3     0  0.75
       4     0  1
       5     1  1.25
       6     1  1.5
       7     1  1.75
    ====  ====  ====
    '''



    def __repr__(self):

        return 'TimeMap (%s)'%hash(self)

    def __hash__(self):

        return _tm_hash(self.nhours, self.freq, self.start,
                       self.stop, self.tm_filt)

    def __init__(self, nhours=1, freq=1,
                 start='2015-1-1 00:00', stop='2015-12-31 23:59',
                 tm_filt=False, keep_datetime=False, minimum=False):

        self.freq = freq
        self.num_freq = (float(self.freq[:-1])
                         if isinstance(self.freq, str) else self.freq)
        self.str_freq = '%sH'%self.num_freq

        self.nhours = nhours

        self.start, self.stop = start, stop

        self.tm_filt = tm_filt
        self.keep_datetime = keep_datetime

        self.minimum = minimum

        self.df_time_red = pd.DataFrame()
        self.df_hoy_soy = pd.DataFrame()
        self.df_time_map = pd.DataFrame()

        TM_DICT[hash(self)] = self

        if nhours:
            self.gen_soy_timemap()

    def gen_hoy_timemap(self):


        logger.info('Generating time map with freq='
                    '{} nhours={} from {} to {}'.format(self.freq, self.nhours,
                                                        self.start, self.stop))

        df_time_map = pd.DataFrame(index=pd.date_range(self.start, self.stop,
                                                       freq=self.str_freq))
        df_time_map = (df_time_map.reset_index()
                                  .rename(columns={'index': 'DateTime'}))

        df_time_map['hour'] = df_time_map['DateTime'].dt.hour
        df_time_map['doy'] = df_time_map['DateTime'].dt.dayofyear
        df_time_map['mt_id'] = df_time_map['DateTime'].dt.month - 1
        df_time_map['day'] = df_time_map['DateTime'].dt.day
        df_time_map['year'] = df_time_map['DateTime'].dt.year
        df_time_map['dow'] = df_time_map['DateTime'].dt.weekday
        df_time_map['how'] = df_time_map['dow'] * 24 + df_time_map['hour']
        df_time_map['hom'] = (df_time_map['day'] - 1) * 24 + df_time_map['hour']
        df_time_map['wk_id'] = df_time_map['DateTime'].dt.week - 1

        # add number of hours per week
        df_time_map = (df_time_map
            .join(df_time_map.pivot_table(values='how', index='wk_id', aggfunc=len)
            .rename(columns={'how': 'wk_weight'}), on='wk_id'))


        # remove February 29
        mask_feb29 = ((df_time_map.mt_id == 1) & (df_time_map.day == 29))
        df_time_map = df_time_map.loc[-mask_feb29].reset_index(drop=True)


        # add hour of the year column
        get_hoy = lambda x: ((x.hour + 24 * (x.doy - 1)
                             + x.DateTime.dt.minute / 60).rename('hy')
                             .reset_index())
        df_time_map['hy'] = (df_time_map.groupby(['year']).apply(get_hoy)
                                        .reset_index(drop=True)['hy'])
        if not self.minimum:

            df_time_map['month'] = df_time_map['DateTime'].dt.month
            df_time_map['mt'] = df_time_map['mt_id'].map(MONTH_DICT)
            df_time_map['season'] = df_time_map['mt'].map(SEASON_DICT)
            df_time_map['wk'] = df_time_map['wk_id']
            df_time_map['dow_name'] = df_time_map['dow'].replace(DOW_DICT)
            df_time_map['dow_type'] = df_time_map['dow'].replace(DOW_TYPE_DICT)

            # week of the month
            dfwom = df_time_map[['wk_id', 'mt_id']]
            dfwkmt = (df_time_map[['wk_id', 'mt_id']]
                            .pivot_table(index='wk_id', values=['mt_id'],
                                         aggfunc=np.median))
            dfwkmt = dfwkmt.rename(columns={'mt_id': 'wk_mt'})
            dfwom = (dfwom.drop('mt_id', axis=1).drop_duplicates()
                          .join(dfwkmt, on=dfwkmt.index.names))
            dfwk_max = (dfwom.pivot_table(index='wk_mt', values=['wk_id'],
                                         aggfunc=max)
                             .rename(columns={'wk_id': 'wk_max'})
                             .reset_index() + 1)
            dfwk_max = dfwk_max.set_index('wk_mt')
            dfwom = dfwom.join(dfwk_max, on=dfwk_max.index.names).fillna(0)
            dfwom['wom'] = dfwom['wk_id'] - dfwom['wk_max']
            dfwom = dfwom.set_index('wk_id')['wom']
            df_time_map = df_time_map.join(dfwom, on=dfwom.index.names)

            df_time_map_ndays = pd.DataFrame(df_time_map.loc[:,['mt', 'year', 'day']]
                                            .drop_duplicates()
                                            .pivot_table(values='day',
                                                         index=['year','mt'],
                                                               aggfunc=len))['day'].rename('ndays')
            df_time_map = df_time_map.join(df_time_map_ndays,
                                           on=df_time_map_ndays.index.names)


        # apply filtering
        mask = df_time_map.mt_id.apply(lambda x: True).rename('mask')
        if self.tm_filt:
            for ifilt in self.tm_filt:
                mask &= df_time_map[ifilt[0]].isin(ifilt[1])

        if mask.sum() == 0:
            raise RuntimeError('Trying to generate and TimeMap which is empty '
                               'after filtering.')

        self.df_time_map = df_time_map.loc[mask].reset_index(drop=True)


        if not self.keep_datetime:
            self.df_time_map = self.df_time_map.drop('DateTime', axis=1)


    def gen_soy_timemap(self):
        '''
        Reduces the original timemap to a lower time resolution.

        * Adds additional columns *weight* and *sy* to the ``df_time_map`` DataFrame:
            - *sy* are the time slot, each of which covers ``nhours/num_freq`` of the
              original *hy*-indexed ``df_time_map`` rows
            - The *weight* is the number of hours per reduced time slot. It is used
              to calculate energy from average power.
        * Generates an attribute ``df_time_red``, which is *sy* indexed. For any given
          *sy*, the minimum value of the temporal columns (e.g. *mt_id*, *how*, etc)
          over the corresponding *hy* is selected.

        '''

        assert (self.nhours / self.num_freq)%1 == 0, \
                ('TimeMap.gen_soy_timemap: The time slot duration nhours must '
                 'be a multiple of the original time map frequency freq. '
                 'num_freq=%f, nhours=%f'%(self.num_freq, self.nhours))

        if self.df_time_map.empty:
            self.gen_hoy_timemap()

        df_time_map = self.df_time_map

        # add soy column to dataframe, based on nhours
        len_rep = self.nhours / self.num_freq
        len_rge = np.ceil(len(df_time_map)/ len_rep)
        df_tm = pd.DataFrame(np.repeat(np.arange(len_rge), [len_rep]),
                             columns=['sy']).iloc[:len(df_time_map)]
        df_time_map['sy'] = df_tm

        # add weight column to dataframe
        df_weight = df_time_map.pivot_table(values=['hy'], index='sy',
                                            aggfunc=len)
        df_weight = df_weight.rename(columns={'hy': 'weight'}) * self.num_freq
        df_time_map = df_time_map.join(df_weight, on='sy')

        self.df_hoy_soy = df_time_map[['sy', 'hy']]

        if self.nhours == self.num_freq:
            self.df_time_red = df_time_map
        else:

            df_time_map_num = df_time_map.select_dtypes(include=['integer',
                                                                 'floating'])
            col_nonnum = [c for c in df_time_map.columns
                          if not c in df_time_map_num.columns]
            df_time_map_oth = df_time_map[col_nonnum + ['hy']].set_index('hy')

            self.df_time_red = (df_time_map_num
                        .pivot_table(aggfunc=min, index=['year', 'sy'])
                        .reset_index())

            self.df_time_red = self.df_time_red.join(df_time_map_oth, on='hy')

    def get_year_share(self):
        '''
        For a given time map, returns the share after applying ``tm_filt``.

        The ``tm_filt`` parameter of the :class:`TimeMap` class allows
        to limit the time map definition to certain months, days, etc.
        The year share corresponds to the duration of the filtered time map
        as a share of the total year length.

        Returns
        -------
        float : year share of the time map

        '''

        return len(self.df_time_red) / (8760 / self.nhours)

    def _get_dst_days(self, list_months=['MAR', 'OCT']):

        # filter by relevant months
        _df = self.df_time_map.loc[self.df_time_map.mt.isin(list_months)]
        mask_dy = _df.dow_name == 'SUN'

        _df = _df.loc[mask_dy].pivot_table(index=['mt', 'year'],
                                           values='doy', aggfunc=np.max)
        _df = _df.reset_index()[['year', 'mt', 'doy']]

        dict_dst = _df.set_index(['year', 'mt'])['doy'].to_dict()


        return dict_dst
# %%

if __name__ == '__main__':
    import tabulate
    import doctest

    doctest.testmod()
