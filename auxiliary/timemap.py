import pandas as pd
import numpy as np

MONTH_DICT = {0:'JAN', 1:'FEB', 2:'MAR', 3:'APR', 4:'MAY', 5:'JUN',
               6:'JUL', 7:'AUG', 8:'SEP', 9:'OCT', 10:'NOV', 11:'DEC'}

SEASON_DICT = {'JAN': 'WINTER', 'FEB': 'WINTER', 'DEC': 'WINTER',
               'MAR': 'SPRING', 'APR': 'SPRING', 'MAY': 'SPRING',
               'JUN': 'SUMMER', 'JUL': 'SUMMER', 'AUG': 'SUMMER',
               'SEP': 'FALL', 'OCT': 'FALL', 'NOV': 'FALL'}
DOW_DICT = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
DOW_TYPE_DICT = {**{d: 'WEEKDAY' for d in range(5)}, **{5: 'SAT', 6: 'SUN'}}

TM_DICT = {}

def tm_hash(nhours, freq, start, stop, tm_filt):

    hash_val = hash((nhours, freq, start, stop, str(tm_filt)))
    return hash_val

class UniqueInstancesMeta(type):

    def __call__(cls, nhours=1, freq=1,
                 start='2015-1-1 00:00', stop='2015-12-31 23:59',
                 tm_filt=False, *args, **kwargs):

        key = tm_hash(nhours, freq, start, stop, tm_filt)

        if key in TM_DICT:
            return TM_DICT[key]
        else:
            return super().__call__(nhours, freq, start, stop, tm_filt,
                                    *args, **kwargs)

class TimeMap(metaclass=UniqueInstancesMeta):

    def __hash__(self):

        return tm_hash(self.nhours, self.freq, self.start,
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


        print('Generating time map with freq='
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

            # add number of hours per week
            df_time_map = (df_time_map
                .join(df_time_map.pivot_table(values=['how'],
                                              index='wk', aggfunc=len)
                .rename(columns={'how': 'wk_weight'}), on='wk'))

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

        self.tm_filt_weight = mask.size / mask.sum()

        self.df_time_map = df_time_map.loc[mask].reset_index(drop=True)


        if not self.keep_datetime:
            self.df_time_map = self.df_time_map.drop('DateTime', axis=1)


    def gen_soy_timemap(self):
        '''
        Reduces the original timemap to a lower time resolution.


        The *weight* is the number of hours per reduced time slot. It is used
        to calculate energy from average power.

        Args:
            nhours (float): final time resolution in hours

        Raises:
            AssertionError: If nhours is not a multiple of freq.
        '''


        assert (self.nhours / self.num_freq)%1 == 0, \
                ('TimeMap.gen_soy_timemap: The time slot duration nhours must '
                 'be a multiple of the original time map frequency freq. '
                 'num_freq=%f, nhours=%f'%(self.num_freq, self.nhours))

        if self.df_time_map.empty:
            self.gen_hoy_timemap()

        df_time_map = self.df_time_map

        # add soy column to dataframe, based on nhours
        len_rge = np.ceil(len(df_time_map)/nhours)
        len_rep = nhours
        df_tm = pd.DataFrame(np.repeat(np.arange(len_rge), [len_rep]),
                             columns=['sy']).iloc[:len(df_time_map)]
        df_time_map['sy'] = df_tm

        # add weight column to dataframe
        df_weight = df_time_map.pivot_table(values=['hy'], index='sy',
                                            aggfunc=len)
        df_weight = df_weight.rename(columns={'hy': 'weight'})
        df_time_map = df_time_map.join(df_weight, on='sy')

        self.df_hoy_soy = df_time_map[['sy', 'hy']]

        if self.nhours == self.num_freq:
            self.df_time_red = df_time_map
        else:

            df_time_map_num = df_time_map.select_dtypes(['int', 'float'])
            col_nonnum = [c for c in df_time_map.columns
                          if not c in df_time_map_num.columns]
            df_time_map_oth = df_time_map[col_nonnum + ['hy']].set_index('hy')

            self.df_time_red = (df_time_map_num.groupby(['year', 'sy'])
                                               .apply(min))
            self.df_time_red = self.df_time_red.join(df_time_map_oth, on='hy')



    def get_dst_days(self, list_months=['MAR', 'OCT']):

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
    pass
