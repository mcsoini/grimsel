# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:28:17 2017

@author: m-c-soini
"""

import warnings

import grimsel.auxiliary.sqlutils.aux_sql_func as aql
import grimsel.auxiliary.timemap as tm

from grimsel.analysis.sql_analysis_hourly import SqlAnalysisHourly
from grimsel.analysis.decorators import DecoratorsSqlAnalysis





class SqlAnalysis(SqlAnalysisHourly, DecoratorsSqlAnalysis):
    ''' Performs various SQL-based analyses on the output tables. '''

    def __init__(self, sc_out, db, slct_run_id=None, bool_run=True, nd_id=False,
                 suffix=False, slct_pt=False, sw_year_col='swyr_vl'):
        ''' Init extracts model run parameter names from def_run table. '''


        self.bool_run = bool_run

        self.db = db

        self.sc_out = sc_out
        self.slct_run_id = (aql.read_sql(self.db, self.sc_out, 'def_run',
                                         keep='run_id', distinct=True).tolist()
                            if not slct_run_id else slct_run_id)
        self._suffix = '_' + suffix if suffix else ''
        self.sw_columns = [c for c in
                           aql.read_sql(self.db, sc_out,
                                        'def_run').columns
                           if 'sw' in c and 'vl' in c]

        self.sw_columns = [c for c in
                           aql.get_sql_cols('def_run', sc_out, self.db).keys()
                           if 'sw' in c and 'vl' in c]


        self.in_run_id = '(' + ', '.join(map(str, self.slct_run_id)) + ')'
        self._nd_id = nd_id if nd_id else aql.read_sql(self.db, sc_out, 'def_node',
                                                       filt=[('0', ['0'])],
                                                       keep=['nd_id'])['nd_id'].tolist()
        self.in_nd_id = '(' + ', '.join(map(str, self._nd_id)) + ')'


        _nd = aql.read_sql(self.db, sc_out, 'def_node', filt=[('0', ['0'])],
                           keep=['nd'])['nd'].tolist()
        self.in_nd = '(' + ', '.join(['\'' + str(c) + '\'' for c in _nd]) + ')'



        # get time resolution
        self.time_res = aql.read_sql(self.db, sc_out, 'tm_soy').iloc[0]['weight']

        try:
            self.tm_cols = aql.read_sql(self.db, self.sc_out, 'tm_soy_full').columns.tolist()
        except:
            print('Generating complete timemap...', end=' ')
            timemap = tm.TimeMap(self.time_res)
            self.tm_cols = timemap.df_time_red.columns.tolist()
            aql.write_sql(timemap.df_time_red, self.db, self.sc_out, 'tm_soy_full', 'replace')
            print('done.')


        self.tm_cols = [c for c in self.tm_cols if not c in
                        ['sy', 'year', 'day', 'dow_name', 'doy', 'hy',
                         'mt', 'ndays', 'wk', 'wk_weight']]

        self.list_to_str = lambda x: '({})'.format(', '.join(map(str, x if type(x) is list else x.tolist())))


#        list_pt_id = self.list_to_str(aql.read_sql(self.db, self.sc_out, 'def_pp_type',
#                                                   filt=[('pt', slct_pt + ['%STO%'], ' LIKE ')] if slct_pt else False)['pt_id'])
        list_pt_id = aql.read_sql(self.db, self.sc_out, 'def_pp_type',
                                  filt=[('pt', slct_pt, ' LIKE ')] if slct_pt else False)['pt_id']
        self.slct_pp_id = aql.read_sql(self.db, self.sc_out, 'def_plant',
                                  filt=[('pt_id', list_pt_id),
                                        ('nd_id', self._nd_id)])['pp_id'].tolist()

        list_slct_pp_id =  self.list_to_str(aql.read_sql(self.db, self.sc_out, 'def_plant',
                                                         filt=[('pt_id', list_pt_id.tolist(), ' = '),
                                                               ('nd_id', self._nd_id)] if slct_pt else False)['pp_id'])

        self.format_kw = {'sfx': self._suffix, 'sc_out': self.sc_out,
                          'in_run_id': self.in_run_id,
                          'in_nd_id': self.in_nd_id,
                          'in_nd': self.in_nd,
                          'list_slct_pp_id': list_slct_pp_id,
                          'sw_year_col': sw_year_col
                          }

        try:
            print('Calling generate_view_time_series_subset')
            self.generate_view_time_series_subset()
        except:
            warnings.warn('Method generate_view_time_series_subset could not be executed.')


    def analysis_storage_capacity_utilization(self, threshold_cf_sub=0):

        # %% /* WHAT MAXIMUM SHARE OF cap_erg_tot IS FILLED */
        exec_str = '''
                    DROP TABLE IF EXISTS {sc_out}.analysis_max_share_cap_erg CASCADE;
                    WITH soc_max AS (
                        SELECT pp_id, ca_id, run_id, bool_out,
                            MAX(value) AS max_soc, MIN(value) AS min_soc
                        FROM {sc_out}.analysis_time_series_view_energy
                        GROUP BY pp_id, ca_id, run_id, bool_out
                    )
                    SELECT *,
                        (max_soc - min_soc) / NULLIF(cap_erg_tot, 0) AS cap_erg_share
                    INTO {sc_out}.analysis_max_share_cap_erg
                    FROM soc_max
                    NATURAL LEFT JOIN (
                        SELECT pp_id, ca_id, run_id, value AS cap_erg_tot
                        FROM {sc_out}.var_yr_cap_erg_tot) AS cap
                    '''.format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)

        # %%  /* DURING HOW MANY HOURS IS THE OUTPUT/INPUT NOT ZERO? */
        #      --> SELECTICE CAPACITY FACTOR
        exec_str = '''
                  DROP TABLE IF EXISTS {sc_out}.analysis_selective_cf_storage CASCADE;
                   WITH hr_count AS (
                       SELECT pp_id, ca_id, run_id, bool_out,
                           COUNT(sy) AS hours_nonzero
                       FROM {sc_out}.analysis_time_series_view_power
                       WHERE pp_id
                           IN (SELECT pp_id FROM {sc_out}.def_plant WHERE pp LIKE '%STO%')
                           AND ABS(value) > {threshold_cf_sub}
                       GROUP BY pp_id, ca_id, run_id, bool_out
                   )
                   SELECT *,
                       erg_yr_sy / (cap_pwr_tot * hours_nonzero) AS cf_sub
                    INTO {sc_out}.analysis_selective_cf_storage
                   FROM hr_count
                   NATURAL LEFT JOIN (
                       SELECT
                           pp_id, ca_id, run_id, bool_out,
                           erg_yr_sy, cap_pwr_tot, cf_sy
                       FROM {sc_out}.analysis_plant_run_tot) AS erg
                   '''.format(**self.format_kw, threshold_cf_sub=threshold_cf_sub)
        aql.exec_sql(exec_str, db=self.db)


        for itb in ['analysis_selective_cf_storage', 'analysis_max_share_cap_erg']:

            aql.joinon(self.db, self.sw_columns, ['run_id'],
                       [self.sc_out, itb], [self.sc_out, 'def_run'])
            aql.joinon(self.db, ['nd_id', 'pt_id', 'fl_id', 'pp'], ['pp_id'],
                       [self.sc_out, itb], [self.sc_out, 'def_plant'])
            aql.joinon(self.db, ['fl'], ['fl_id'],
                       [self.sc_out, itb], [self.sc_out, 'def_fuel'])
            aql.joinon(self.db, ['nd'], ['nd_id'],
                       [self.sc_out, itb], [self.sc_out, 'def_node'])
            aql.joinon(self.db, ['pt'], ['pt_id'],
                       [self.sc_out, itb], [self.sc_out, 'def_pp_type'])


    def generate_total_vc(self):

        tb_name = 'analysis_total_vc'

        exec_str = '''
                   DROP TABLE IF EXISTS {sc_out}.{tb_name} CASCADE;

                   SELECT vcfl.pp_id, vcfl.run_id,
                       vcfl.value AS vc_fl, vcom.value AS vc_om, vcco.value AS vc_co,
                       vcfl.value + vcom.value + vcco.value AS vc_tot
                   INTO {sc_out}.{tb_name}
                   FROM {sc_out}.par_vc_fl AS vcfl
                   LEFT JOIN {sc_out}.par_vc_om AS vcom
                       ON vcfl.run_id = vcom.run_id AND vcfl.pp_id = vcom.pp_id
                   LEFT JOIN {sc_out}.par_vc_co2 AS vcco
                       ON vcfl.run_id = vcco.run_id AND vcfl.pp_id = vcco.pp_id;
                   '''.format(**self.format_kw, tb_name=tb_name)
        aql.exec_sql(exec_str, db=self.db)

        aql.joinon(self.db, self.sw_columns, ['run_id'],
                   [self.sc_out, tb_name], [self.sc_out, 'def_run'])
        aql.joinon(self.db, ['nd_id', 'pt_id', 'fl_id', 'pp'], ['pp_id'],
                   [self.sc_out, tb_name], [self.sc_out, 'def_plant'])
        aql.joinon(self.db, ['fl'], ['fl_id'],
                   [self.sc_out, tb_name], [self.sc_out, 'def_fuel'])
        aql.joinon(self.db, ['nd'], ['nd_id'],
                   [self.sc_out, tb_name], [self.sc_out, 'def_node'])
        aql.joinon(self.db, ['pt'], ['pt_id'],
                   [self.sc_out, tb_name], [self.sc_out, 'def_pp_type'])

    def generate_view_time_series_subset(self):
        '''
        This creates a view providing an extraction of data from the main
        time series output tables.
        '''

        exec_str = ('''
                    DROP VIEW IF EXISTS {sc_out}.analysis_time_series_view_power CASCADE;
                    CREATE VIEW {sc_out}.analysis_time_series_view_power AS
                    SELECT
                        pwr.sy, ca_id, pwr.pp_id,
                        bool_out,
                        value, pwr.run_id,
                        'pwr'::VARCHAR AS pwrerg_cat,
                        (CASE WHEN bool_out = True THEN -1 ELSE 1 END) * value AS value_posneg
                    FROM {sc_out}.var_sy_pwr AS pwr
                    WHERE pwr.run_id IN {in_run_id}
                    AND pp_id IN (SELECT pp_id FROM {sc_out}.def_plant
                                  WHERE nd_id in {in_nd_id})
                    AND pp_id in {list_slct_pp_id};
                    ''').format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)


        if 'var_sy_erg_st' in aql.get_sql_tables(self.sc_out, self.db):
            exec_str = ('''
                        DROP VIEW IF EXISTS {sc_out}.analysis_time_series_view_energy CASCADE;
                        CREATE VIEW {sc_out}.analysis_time_series_view_energy AS
                        SELECT ergst.sy, ca_id, ergst.pp_id,
                            False::BOOLEAN AS bool_out,
                            value, ergst.run_id,
                            'erg'::VARCHAR AS pwrerg_cat,
                            value AS value_posneg
                        FROM {sc_out}.var_sy_erg_st AS ergst
                        WHERE ergst.run_id IN {in_run_id}
                        AND pp_id IN (SELECT pp_id FROM {sc_out}.def_plant
                                      WHERE nd_id in {in_nd_id})
                        AND pp_id in {list_slct_pp_id};
                        ''').format(**self.format_kw)
            aql.exec_sql(exec_str, db=self.db)

        exec_str = ('''
                    DROP VIEW IF EXISTS {sc_out}.analysis_time_series_view_crosssector_0 CASCADE;

                    CREATE VIEW {sc_out}.analysis_time_series_view_crosssector_0 AS
                    WITH slct_pp_id AS (SELECT pp_id FROM {sc_out}.def_plant
                                        WHERE fl_id IN (
                                          SELECT fl_id
                                          FROM {sc_out}.def_fuel
                                          WHERE is_ca = 1))
                    SELECT sy, dfca.ca_id, dfnd.nd_id, nd, ca, dfpp.fl_id,
                           regexp_replace(nd, '0+$', '') || '_CONS_' || ca AS pp,
                           True AS bool_out, value / pp_eff AS value,
                           run_id,  'pwr'::VARCHAR AS pwrerg_cat, -value / pp_eff AS value_posneg
                    FROM (
                        SELECT sy, pp_id, ca_id AS ca_out_id, True AS bool_out, value, run_id
                        FROM {sc_out}.var_sy_pwr
                        WHERE bool_out=False
                        AND pp_id IN (SELECT pp_id FROM slct_pp_id)
                    ) AS pwr
                    LEFT JOIN (SELECT pp_id, fl_id, nd_id FROM {sc_out}.def_plant) AS dfpp ON dfpp.pp_id = pwr.pp_id
                    LEFT JOIN {sc_out}.def_encar AS dfca ON dfca.fl_id = dfpp.fl_id
                    LEFT JOIN {sc_out}.def_node AS dfnd ON dfnd.nd_id = dfpp.nd_id
                    LEFT JOIN (SELECT pp_id, ca_id, pp_eff FROM {sc_out}.plant_encar)
                        AS ppeff ON ppeff.pp_id = pwr.pp_id
                        AND ppeff.ca_id = pwr.ca_out_id;

                    DROP VIEW IF EXISTS {sc_out}.analysis_time_series_view_crosssector CASCADE;

                    CREATE VIEW {sc_out}.analysis_time_series_view_crosssector AS
                    SELECT sy, ca_id, pp_id, bool_out, SUM(value) AS value, run_id, pwrerg_cat, SUM(value_posneg) AS value_posneg
                    FROM {sc_out}.analysis_time_series_view_crosssector_0 AS tb
                    LEFT JOIN (SELECT pp, pp_id FROM {sc_out}.def_plant) AS dfpp ON dfpp.pp = tb.pp
                    GROUP BY sy, ca_id, pp_id, bool_out, run_id, pwrerg_cat;
                    ''').format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)


    def post_analysis_erg_sto_res(self):

        '''
        '''


        '''
        --DROP VIEW IF EXISTS tb_sto;
        --CREATE VIEW tb_sto AS
        WITH tb_slct AS (
            SELECT run_id, tb.pp_id, AVG(1 - st_lss_rt) AS eff,
                mt_id, bool_out, SUM(erg_yr_sy)
            FROM out_long.analysis_plant_mt_id_run_tot AS tb
            LEFT JOIN lp_input.plant_encar AS ppca ON ppca.pp_id = tb.pp_id
            WHERE tb.pp_id IN (SELECT pp_id FROM lp_input.def_plant
                               WHERE set_def_st = 1 OR set_def_hyrs = 1)
            GROUP BY run_id, tb.pp_id, mt_id, bool_out
        ), tb_true AS (
            SELECT *, sum AS sum_chg FROM tb_slct WHERE bool_out = True
            UNION ALL
            SELECT run_id, pp_id, eff, mt_id, bool_out, sum, sum_chg FROM (
                SELECT pp_id, AVG(1) AS eff, mt_id, True AS bool_out,
                    SUM(infl.value) AS sum, SUM(infl.value) AS sum_chg
                FROM lp_input.profinflow AS infl
                NATURAL LEFT JOIN (SELECT sy AS hy, mt_id FROM out_nucspreadvr_2.tm_soy) AS mt_map
                GROUP BY pp_id, mt_id
            ) AS tb
            FULL OUTER JOIN (SELECT DISTINCT run_id FROM out_long.analysis_plant_mt_id_run_tot) AS list_run_id ON True
        ), tb_flse AS (SELECT *, sum AS sum_dch FROM tb_slct WHERE bool_out = False),
        tb_net AS (
            SELECT tb_true.run_id, tb_true.pp_id, tb_true.mt_id, tb_true.eff AS eff_chg, tb_flse.eff AS eff_dch,
                sum_chg, sum_dch,
                sum_chg * SQRT(tb_true.eff) - sum_dch / SQRT(tb_flse.eff) AS netchgdch
                FROM tb_true
            LEFT JOIN tb_flse ON tb_flse.run_id = tb_true.run_id
                AND tb_flse.pp_id = tb_true.pp_id
                AND tb_flse.mt_id = tb_true.mt_id
        ), tb_erg AS (
            SELECT tb1.run_id, tb1.pp_id, tb1.eff_chg, tb1.eff_dch, tb1.mt_id, tb1.sum_chg, tb1.sum_dch, tb1.netchgdch, SUM(tb2.netchgdch) AS erg_0 FROM tb_net AS tb1
            LEFT JOIN tb_net AS tb2 ON tb2.run_id = tb1.run_id
                AND tb2.pp_id = tb1.pp_id
                AND tb2.mt_id <= tb1.mt_id
            GROUP BY tb1.run_id, tb1.pp_id, tb1.mt_id, tb1.netchgdch, tb1.sum_chg, tb1.sum_dch, tb1.eff_chg, tb1.eff_dch
            ORDER BY run_id, pp_id, mt_id
        ), tb_erg_min AS (
            SELECT run_id, pp_id, MIN(erg_0) AS erg_min
            FROM tb_erg
            GROUP BY run_id, pp_id
        )
        SELECT dfpp.*, tb_erg.*, erg_min, (tb_erg.erg_0 - tb_erg_min.erg_min) AS erg FROM tb_erg
        NATURAL LEFT JOIN tb_erg_min
        NATURAL LEFT JOIN (SELECT pp_id, pp FROM lp_input.def_plant) AS dfpp

        ;

        '''


    # %%
    def build_tables_plant_run(self, list_timescale=None):
        '''
        Calculates the main indicators of interest from the model output
        tables by (run_id, plant_id, bool_out).
        Keyword arguments:
        list_timescale -- list of time_map_soy columns to define the temporal
                          granularity of the analysis. 'mt_id' will calculate
                          all indicators for each month separately. Empty
                          string '' aggregates the whole year.
        '''


        if list_timescale is None:
            list_timescale = ['']



        for timescale in list_timescale:#, 'wk', 'doy', 'hy']:

            join_timescale_id = \
                    ('''
                     LEFT JOIN (
                     SELECT {timescale}, sy, tm_id FROM {sc_out}.tm_soy_full) AS dftsc
                     ON dftsc.sy = pwr.sy AND dftsc.tm_id = dfnd.tm_id
                     ''').format(sc_out=self.sc_out,
                                 timescale=timescale)
            if timescale == '':
                join_timescale_id = ''
            print('join_timescale_id: ', join_timescale_id)

            col_timescale_id = 'dftsc.' + timescale + ',' if timescale != '' else ''
            tb_mod = timescale + '_' if timescale != '' else ''

            time_scale_kw = {'join_timescale_id': join_timescale_id,
                             'col_timescale_id': col_timescale_id,
                             'tb_mod': tb_mod}

            exec_str = (
                        '''
                        /* ####################################### */
                        /* GENERATE GENERAL VIEW plant_{tb_mod}run */
                        /* ####################################### */

                        DROP VIEW IF EXISTS plant_{tb_mod}run_0 CASCADE;
                        CREATE VIEW plant_{tb_mod}run_0 AS
                        SELECT
                            pwr.run_id,
                            pwr.bool_out,
                            pwr.ca_id,
                            pwr.pp_id,
                            dfpp.fl_id,
                            dfnd.tm_id,
                            {col_timescale_id}
                            SUM(dffl.co2_int * pwr.value * tmsoy.weight * (ppca.factor_lin_0 + 0.5 * pwr.value * ppca.factor_lin_1)) AS co2_yr_sy,
                            SUM(mc.value * pwr.value * tmsoy.weight) AS val_yr_sy,
                            SUM(pwr.value * tmsoy.weight) AS erg_yr_sy,
                            COUNT(pwr.run_id) AS count_check
                        FROM (SELECT * FROM {sc_out}.var_sy_pwr WHERE run_id IN {in_run_id}) AS pwr
                        NATURAL LEFT JOIN (SELECT nd_id, tm_id FROM {sc_out}.def_node) AS dfnd
                        LEFT JOIN (SELECT run_id, sy, nd_id, value FROM {sc_out}.dual_supply) AS mc
                            ON mc.run_id = pwr.run_id AND mc.nd_id = dfplt.nd_id AND mc.sy = pwr.sy
                        LEFT JOIN {sc_out}.par_pp_eff AS eff
                            ON eff.pp_id = pwr.pp_id AND eff.run_id = pwr.run_id AND eff.ca_id = pwr.ca_id
                        LEFT JOIN (SELECT pp_id, ca_id, factor_lin_0, factor_lin_1 FROM {sc_out}.plant_encar) AS ppca
                            ON ppca.pp_id = pwr.pp_id AND ppca.ca_id = pwr.ca_id
                        LEFT JOIN (SELECT pp_id, fl_id, nd_id FROM {sc_out}.def_plant) AS dfpp
                            ON dfpp.pp_id = pwr.pp_id
                        LEFT JOIN (SELECT co2_int, fl_id FROM {sc_out}.def_fuel) AS dffl
                            ON dfpp.fl_id = dffl.fl_id
                        LEFT JOIN (SELECT nd_id, price_co2 FROM {sc_out}.def_node) AS dfnd
                            ON dfpp.nd_id = dfnd.nd_id
                        LEFT JOIN (SELECT tm_id, weight, sy FROM {sc_out}.tm_soy) AS tmsoy
                            ON pwr.sy = tmsoy.sy AND dfnd.tm_id = tmsoy.tm_id
                        {join_timescale_id}
                        GROUP BY pwr.run_id, pwr.ca_id, {col_timescale_id}
                                 pwr.pp_id, pwr.bool_out;

                        /* ADD DERIVED COLUMNS */
                        DROP VIEW IF EXISTS plant_{tb_mod}run_01 CASCADE;
                        CREATE VIEW plant_{tb_mod}run_01 AS
                        WITH totdmnd AS (
                            SELECT nd_id, run_id, SUM(value) AS sum_dmnd
                            FROM {sc_out}.par_dmnd AS dmnd
                            GROUP BY nd_id, run_id
                        )
                        SELECT pr0.*,
                            pr0.erg_yr_sy / NULLIF(sum_dmnd, 0) AS erg_yr_sy_share_dmnd
                        FROM plant_{tb_mod}run_0 AS pr0
                        LEFT JOIN totdmnd
                            ON totdmnd.run_id = pr0.run_id
                            AND totdmnd.nd_id = pr0.nd_id;

                        /* ADD CAPACITIES (DONE HERE TO AVOID CAPACITY IN
                                           AGGREGATING FUNCTION) */
                        DROP VIEW IF EXISTS plant_{tb_mod}run_1 CASCADE;
                        CREATE VIEW plant_{tb_mod}run_1 AS
                        SELECT
                            pr0.*,
                            tb_cap.value AS cap_pwr_tot,
                           --     tb_cap_new.value AS cap_pwr_new,
                           --     -tb_cap_rem.value AS cap_pwr_rem,
                            tb_cap_leg.value AS cap_pwr_leg
                        FROM plant_{tb_mod}run_01 AS pr0
                        LEFT JOIN {sc_out}.var_yr_cap_pwr_tot AS tb_cap
                            ON tb_cap.run_id = pr0.run_id
                            AND tb_cap.pp_id = pr0.pp_id
                        LEFT JOIN {sc_out}.par_cap_pwr_leg AS tb_cap_leg
                            ON tb_cap_leg.run_id = pr0.run_id
                            AND tb_cap_leg.pp_id = pr0.pp_id;

                        DROP VIEW IF EXISTS plant_{tb_mod}run_2;
                        CREATE VIEW plant_{tb_mod}run_2 AS
                        SELECT pr1.*,
                            pr1.val_yr_0 / NULLIF(cap_pwr_tot, 0) AS val_yr_0_cap,
                            erg_yr_sy / NULLIF(cap_pwr_tot * 8760, 0) AS cf_sy
                        FROM plant_{tb_mod}run_1 AS pr1;

                        DROP TABLE IF EXISTS {sc_out}.analysis_plant_{tb_mod}run_tot CASCADE;
                        SELECT * INTO {sc_out}.analysis_plant_{tb_mod}run_tot
                        FROM plant_{tb_mod}run_2 AS pr2;
                        ''').format(**self.format_kw, **time_scale_kw)

            if self.bool_run:
                print(exec_str)
                aql.exec_sql(exec_str, db=self.db, ret_res=False)

                itb = '''
                      analysis_plant_{tb_mod}run_tot
                      '''.format(**self.format_kw, **time_scale_kw)

                '''
                /* ################## */
                /* ADD HELPER COLUMNS */
                /* ################## */
                '''

                print(itb)

                aql.joinon(self.db, ['pp_broad_cat', 'pt'], ['pt_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_pp_type'])
                aql.joinon(self.db, self.sw_columns, ['run_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_run'])
                aql.joinon(self.db, ['fl_id', 'pp'], ['pp_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_plant'])
                aql.joinon(self.db, ['fl'], ['fl_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_fuel'])
                aql.joinon(self.db, ['nd'], ['nd_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_node'])

        return exec_str


    # %%
    def build_tables_plant_run_new(self, list_timescale=None):
        '''
        Calculates the main indicators of interest from the model output
        tables by (run_id, plant_id, bool_out).
        Keyword arguments:
        list_timescale -- list of time_map_soy columns to define the temporal
                          granularity of the analysis. 'mt_id' will calculate
                          all indicators for each month separately. Empty
                          string '' aggregates the whole year.
        '''

        if list_timescale is None:
            list_timescale = ['']

        for timescale in list_timescale:#, 'wk', 'doy', 'hy']:

            join_timescale_id = \
                    ('''
                     LEFT JOIN (
                     SELECT {timescale}, sy FROM {sc_out}.tm_soy) AS dftsc
                     ON dftsc.sy = pwr.sy
                     ''').format(sc_out=self.sc_out,
                                 timescale=timescale)
            if timescale == '':
                join_timescale_id = ''
            print('join_timescale_id: ', join_timescale_id)

            col_timescale_id = 'dftsc.' + timescale + ',' if timescale != '' else ''
            tb_mod = timescale + '_' if timescale != '' else ''

            time_scale_kw = {'join_timescale_id': join_timescale_id,
                             'col_timescale_id': col_timescale_id,
                             'tb_mod': tb_mod}


            tb_name = 'analysis_plant_{tb_mod}run_tot'.format(**time_scale_kw)
            cols = [('run_id', 'SMALLINT'),
                    ('bool_out', 'BOOLEAN'),
                    ('ca_id', 'SMALLINT'),
                    ('pp_id', 'SMALLINT'),
                    ('nd_id', 'SMALLINT'),
                    ('weight', 'FLOAT'),
                    ('val_sy_yr', 'DOUBLE PRECISION'),
                    ('co2_sy_yr', 'DOUBLE PRECISION'),
                    ('erg_yr_sy', 'DOUBLE PRECISION'),
                   ] + ([(timescale, 'SMALLINT')] if timescale else [])
            pk = ['run_id', 'pp_id', 'ca_id', 'bool_out'] + ([timescale] if timescale else [])
            cols = aql.init_table(tb_name=tb_name, cols=cols, schema=self.sc_out,
                                  pk=pk, db=self.db)


            exec_str = ('''
                        /* ####################################### */
                        /* GENERATE GENERAL VIEW plant_{tb_mod}run */
                        /* ####################################### */

                        WITH tb_final AS (
                        SELECT
                            pwr.run_id, pwr.bool_out, pwr.ca_id, pwr.pp_id,
                            dfpp.nd_id, AVG(weight) AS weight,
                            {col_timescale_id}
                            SUM((CASE WHEN pwr.bool_out = True THEN -1 ELSE 1 END) * pwr.value * mc.value) AS val_sy_yr,
                            SUM(co2_int * weight * pwr.value * (factor_lin_0 + 0.5 * pwr.value * factor_lin_1)) AS co2_sy_yr,
                            SUM(pwr.value * weight) AS erg_yr_sy,
                            COUNT(pwr.run_id) AS count_check
                        FROM (SELECT * FROM {sc_out}.var_sy_pwr) AS pwr
                        LEFT JOIN (SELECT pp_id, nd_id, fl_id FROM {sc_out}.def_plant) AS dfpp ON dfpp.pp_id = pwr.pp_id
                        LEFT JOIN (SELECT nd_id, ca_id, tm_id,
                                   weight_0 as weight
                                   FROM {sc_out}.nd_tm_pf_map) AS ndtmpf
                            ON ndtmpf.nd_id = dfpp.nd_id AND ndtmpf.ca_id = pwr.ca_id
                        LEFT JOIN (SELECT run_id, sy, nd_id, value
                                   FROM {sc_out}.dual_supply) AS mc
                            ON mc.run_id = pwr.run_id AND mc.nd_id = dfpp.nd_id AND mc.sy = pwr.sy
                        LEFT JOIN (SELECT nd_id, tm_id FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = dfpp.nd_id
                        LEFT JOIN (SELECT fl_id, co2_int FROM {sc_out}.def_fuel) AS dffl ON dffl.fl_id = dfpp.fl_id
                        LEFT JOIN (SELECT pp_id, ca_id, factor_lin_1, factor_lin_0 FROM {sc_out}.plant_encar) AS ppca
                            ON ppca.pp_id = pwr.pp_id AND ppca.ca_id = pwr.ca_id
                        GROUP BY pwr.run_id,  pwr.ca_id, pwr.pp_id, dfpp.nd_id, pwr.bool_out
                        )
                        INSERT INTO {sc_out}.{tb_name}
                        SELECT {cols} FROM tb_final;
                        '''
                        ).format(cols=cols, tb_name=tb_name,
                                 **self.format_kw, **time_scale_kw)
            if self.bool_run:
                aql.exec_sql(exec_str, db=self.db, ret_res=False)

                itb = '''
                      analysis_plant_{tb_mod}run_tot
                      '''.format(**self.format_kw, **time_scale_kw)

                aql.joinon(self.db, ['fl_id', 'pt_id', 'pp'], ['pp_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_plant'])
                aql.joinon(self.db, ['fl'], ['fl_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_fuel'])
                aql.joinon(self.db, ['nd'], ['nd_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_node'])
                aql.joinon(self.db, ['pt'], ['pt_id'],
                           [self.sc_out, itb], [self.sc_out, 'def_pp_type'])
                aql.joinon(self.db, {'value': 'cap_pwr_tot'},
                           ['pp_id', 'ca_id', 'run_id'],
                           [self.sc_out, itb], [self.sc_out, 'var_yr_cap_pwr_tot'])
                aql.joinon(self.db, {'value': 'cap_pwr_leg'},
                           ['pp_id', 'ca_id', 'run_id'],
                           [self.sc_out, itb], [self.sc_out, 'par_cap_pwr_leg'])

                if len(self.sw_columns) > 0:
                    aql.joinon(self.db, self.sw_columns, ['run_id'],
                               [self.sc_out, itb], [self.sc_out, 'def_run'])

        return exec_str

    def analysis_basic_pp_nd_map(self):
        '''
        Auxiliary tables for the maps

        * *pp_id*+*ca_id* |rarr| *tm_id/weight/supply_pd_id*
        * *nd_id*+*ca_id* |rarr| *tm_id/weight/dmnd_pf_id*

        '''

        tb_name = 'pp_tm_pf_map'
        cols = [('pp_id', 'SMALLINT'), ('ca_id', 'SMALLINT'),
                ('nd_id', 'SMALLINT'), ('tm_id', 'SMALLINT'),
                ('supply_pf_id', 'SMALLINT'), ('weight_0', 'FLOAT'),
               ]
        pk = ['pp_id', 'ca_id']
        cols = aql.init_table(tb_name=tb_name, cols=cols, schema=self.sc_out,
                              pk=pk, db=self.db)

        exec_strg = '''
        INSERT INTO {sc_out}.pp_tm_pf_map
        SELECT {cols} FROM {sc_out}.plant_encar
        NATURAL LEFT JOIN (SELECT nd_id, pp_id FROM {sc_out}.def_plant) AS dfpp
        NATURAL LEFT JOIN (SELECT tm_id, nd_id
                           FROM {sc_out}.def_node) AS dfnd
        NATURAL LEFT JOIN (SELECT weight AS weight_0, tm_id
                           FROM {sc_out}.tm_soy WHERE sy = 0) AS tmsy;
        '''.format(**self.format_kw, cols=cols)
        aql.exec_sql(exec_strg, db=self.db)

        tb_name = 'nd_tm_pf_map'
        cols = [('nd_id', 'SMALLINT'), ('ca_id', 'SMALLINT'),
                ('tm_id', 'SMALLINT'),
                ('dmnd_pf_id', 'SMALLINT'), ('weight_0', 'FLOAT'),
               ]
        pk = ['nd_id', 'ca_id']
        cols = aql.init_table(tb_name=tb_name, cols=cols, schema=self.sc_out,
                              pk=pk, db=self.db)

        exec_strg = '''
        INSERT INTO {sc_out}.nd_tm_pf_map
        SELECT nd_id, ca_id, tm_id, dmnd_pf_id, weight_0
            FROM {sc_out}.node_encar
        NATURAL LEFT JOIN (SELECT tm_id, nd_id
                           FROM {sc_out}.def_node) AS dfnd
        NATURAL LEFT JOIN (SELECT weight AS weight_0, tm_id
                           FROM {sc_out}.tm_soy WHERE sy = 0) AS tmsy;
        '''.format(**self.format_kw, cols=cols)
        aql.exec_sql(exec_strg, db=self.db)


# %%

    def add_plant_run_diff(self, diff_idx):
        '''
        Calculate differences along a certain index.
        '''

        order_cols = ', '.join([c for c in self.sw_columns
                                if not c == '{}_vl'.format(diff_idx)])
        order_cols += ', ' if not order_cols == '' else ''

        exec_str = '''
                   DROP TABLE IF EXISTS temp_add_plant_run_diff CASCADE;

                   SELECT tb.*,
                   erg_yr_sy - lag(erg_yr_sy) OVER (ORDER BY pt, bool_out, {order_cols}run_id, {diff_idx}_vl) AS erg_yr_sy_{diff_idx}
                   INTO temp_add_plant_run_diff
                   FROM {sc_out}.plant_run_tot{sfx} AS tb;

                   DROP TABLE {sc_out}.plant_run_tot{sfx} CASCADE;
                   SELECT *
                   INTO {sc_out}.plant_run_tot{sfx}
                   FROM temp_add_plant_run_diff;

                   DROP TABLE temp_add_plant_run_diff CASCADE;
                   '''.format(order_cols=order_cols, diff_idx=diff_idx,
                              **self.format_kw)

        if self.bool_run:
            aql.exec_sql(exec_str)

        return exec_str




    # %% Table plant_run for different time scales
    def build_tables_plant_run_quick(self):
        '''
        SAME as build_tables_plant_run, but based on the yearly energy
        variables. Hence no alternative time aggregations (obviously).
        '''

        exec_str = '''
                    DROP VIEW IF EXISTS plant_run_quick_0 CASCADE;
                    CREATE VIEW plant_run_quick_0 AS
                    SELECT
                        erg.run_id, erg.bool_out, erg.ca_id, erg.pp_id,
                        dfplt.nd_id, dfplt.set_def_st, dfplt.pt_id,
                        SUM(ABS(erg.value)) AS erg_yr_yr,
                        SUM(co2_int / NULLIF(eff.value * erg.value, 0)) AS co2_el,
                        COUNT(erg.run_id) AS count_check
                    FROM (SELECT * FROM {sc_out}.var_yr_erg_yr
                          WHERE run_id IN {in_run_id}) AS erg
                    LEFT JOIN (
                        SELECT pp_id, pt_id, fl_id, nd_id, set_def_st
                        FROM {sc_out}.def_plant) AS dfplt
                    ON dfplt.pp_id = erg.pp_id
                    LEFT JOIN (
                        SELECT fl_id, co2_int
                        FROM {sc_out}.def_fuel) AS dffl
                    ON dfplt.fl_id = dffl.fl_id
                    LEFT JOIN {sc_out}.par_pp_eff AS eff
                    ON eff.pp_id = erg.pp_id AND eff.run_id = erg.run_id AND eff.ca_id = erg.ca_id
                    GROUP BY erg.run_id, erg.ca_id, dfplt.pt_id,
                             dfplt.fl_id, erg.pp_id, dfplt.nd_id,
                             erg.bool_out, dfplt.set_def_st;
                    '''.format(**self.format_kw)
        if self.bool_run:
            aql.exec_sql(exec_str, db=self.db, ret_res=False)

        exec_strg = '''
        DROP VIEW IF EXISTS plant_run_quick_0d CASCADE;
        CREATE VIEW plant_run_quick_0d AS
        WITH tb_dmnd_0 AS (
            SELECT nd_id, ca_id, SUM(value) AS erg_yr_yr
            FROM {sc_out}.profdmnd
            GROUP BY nd_id, ca_id
        ), tb_dmnd AS (
            SELECT * FROM tb_dmnd_0
            FULL OUTER JOIN (SELECT run_id FROM {sc_out}.def_run) AS dflp ON 1 = 1
        )
        SELECT run_id, True::BOOLEAN AS bool_out, ca_id, pp_id,
                nd_id, 0::SMALLINT AS set_def_st, pt_id, erg_yr_yr,
                NULL::DOUBLE PRECISION AS co2_el, 1 AS count_check FROM tb_dmnd
        NATURAL LEFT JOIN (SELECT nd_id, pp_id, pt_id
                           FROM {sc_out}.def_plant
                           WHERE pp LIKE '%DMND') AS dfpp;
        '''.format(**self.format_kw)
        if self.bool_run:
            aql.exec_sql(exec_strg, db=self.db, ret_res=False)



        exec_str = ('''
                    /* ADD DERIVED COLUMNS */
                    DROP VIEW IF EXISTS plant_run_quick_01 CASCADE;
                    CREATE VIEW plant_run_quick_01 AS
                    WITH totdmnd AS (
                        SELECT nd_id, run_id, erg_yr_yr AS sum_dmnd
                        FROM plant_run_quick_0d
                    )
                    SELECT pr0.*,
                        pr0.erg_yr_yr / sum_dmnd AS erg_yr_yr_share_dmnd
                    FROM (SELECT * FROM plant_run_quick_0
                          UNION ALL
                          SELECT * FROM plant_run_quick_0d) AS pr0
                    LEFT JOIN totdmnd
                        ON totdmnd.run_id = pr0.run_id
                        AND totdmnd.nd_id = pr0.nd_id;

                    /* ADD CAPACITIES (DONE HERE TO AVOID CAPACITY IN
                                       AGGREGATING FUNCTION) */
                    DROP VIEW IF EXISTS plant_run_quick_1 CASCADE;
                    CREATE VIEW plant_run_quick_1 AS
                    SELECT
                        pr0.*,
                        tb_cap.value AS cap_pwr_tot,
                        tb_cap_new.value AS cap_pwr_new,
                        -tb_cap_rem.value AS cap_pwr_rem,
                        tb_cap_leg.value AS cap_pwr_leg
                    FROM plant_run_quick_01 AS pr0
                    LEFT JOIN {sc_out}.var_yr_cap_pwr_tot AS tb_cap
                        ON tb_cap.run_id = pr0.run_id
                        AND tb_cap.pp_id = pr0.pp_id
                    LEFT JOIN {sc_out}.var_yr_cap_pwr_new AS tb_cap_new
                        ON tb_cap_new.run_id = pr0.run_id
                        AND tb_cap_new.pp_id = pr0.pp_id
                    LEFT JOIN {sc_out}.var_yr_cap_pwr_rem AS tb_cap_rem
                        ON tb_cap_rem.run_id = pr0.run_id
                        AND tb_cap_rem.pp_id = pr0.pp_id
                    LEFT JOIN {sc_out}.par_cap_pwr_leg AS tb_cap_leg
                        ON tb_cap_leg.run_id = pr0.run_id
                        AND tb_cap_leg.pp_id = pr0.pp_id;


                    DROP VIEW IF EXISTS plant_run_quick_2;
                    CREATE VIEW plant_run_quick_2 AS
                    SELECT pr1.*,
                        erg_yr_yr / NULLIF(cap_pwr_tot * 8760, 0) AS cf_sy
                    FROM plant_run_quick_1 AS pr1;

                    DROP TABLE IF EXISTS {sc_out}.analysis_plant_run_quick{sfx} CASCADE;
                    SELECT * INTO {sc_out}.analysis_plant_run_quick{sfx}
                    FROM plant_run_quick_2 AS pr2;
                    ''').format(**self.format_kw)
        if self.bool_run:
            aql.exec_sql(exec_str, db=self.db, ret_res=False)

        '''
        /* ################## */
        /* ADD HELPER COLUMNS */
        /* ################## */
        '''

        tb = 'analysis_plant_run_quick{sfx}'.format(**self.format_kw)
        if self.bool_run:
            aql.joinon(self.db, ['pp_broad_cat', 'pt'], ['pt_id'], [self.sc_out, tb],
                       [self.sc_out, 'def_pp_type'])
            aql.joinon(self.db, self.sw_columns, ['run_id'],
                       [self.sc_out, tb], [self.sc_out, 'def_run'])
            aql.joinon(self.db, ['set_def_winsol', 'fl_id', 'pp'], ['pp_id'], [self.sc_out, tb],
                       [self.sc_out, 'def_plant'])
            aql.joinon(self.db, ['fl'], ['fl_id'], [self.sc_out, tb],
                       [self.sc_out, 'def_fuel'])
            aql.joinon(self.db, ['nd'], ['nd_id'], [self.sc_out, tb],
                       [self.sc_out, 'def_node'])

        return exec_str



    # %% Table

    def build_table_weighted_mix(self):

        exec_str = ('''
                    DROP TABLE IF EXISTS {sc_out}.analysis_weighted_mix;

                    WITH pwr_mask AS (
                        SELECT pwr.sy, nd_id, pp AS pp_mask, swtc_vl, pwr.pp_id AS pp_id_mask, ca_id, bool_out,
                            value * weight AS pwrw_mask, pwr.run_id, weight,
                            SUM(value * weight) OVER (PARTITION BY pwr.run_id, pp, bool_out) AS value_mask_tot
                        FROM {sc_out}.var_sy_pwr AS pwr
                        LEFT JOIN {sc_out}.def_run AS dflp ON dflp.run_id = pwr.run_id
                        LEFT JOIN {sc_out}.def_plant AS dfpp ON dfpp.pp_id = pwr.pp_id
                        LEFT JOIN {sc_out}.tm_soy AS tm ON tm.sy = pwr.sy
                        WHERE pp LIKE '%' || swtc_vl || '%' OR pp LIKE '%HYD_STO%'
                    ), pwr_pp AS (
                        SELECT run_id, pwr.sy, nd_id, pp,
                        value * weight AS pwrw_pp,
                        SUM(value * weight) OVER (PARTITION BY run_id, pwr.sy, nd_id, bool_out) AS value_tot_all_pp
                        FROM {sc_out}.var_sy_pwr AS pwr
                        LEFT JOIN {sc_out}.def_plant AS dfpp ON dfpp.pp_id = pwr.pp_id
                        LEFT JOIN {sc_out}.tm_soy AS tm ON tm.sy = pwr.sy
                        WHERE bool_out = False AND NOT pp LIKE '%STO%'
                    ), pp_shares AS (
                        SELECT pwr_pp.run_id, pwr_pp.sy, pwr_pp.nd_id, pwr_pp.pp,
                            pwrw_pp / NULLIF(value_tot_all_pp, 0) AS pwrw_pp_sh
                        FROM pwr_pp
                    ), mask_shares AS (
                        SELECT pwr_mask.run_id, pwr_mask.sy, pwr_mask.nd_id, pwr_mask.pp_mask, pwr_mask.bool_out,
                            pwrw_mask / NULLIF(value_mask_tot, 0) AS pwrw_mask_sh
                        FROM pwr_mask
                    )
                    SELECT mask_shares.nd_id, mask_shares.run_id, pp, pp_mask, bool_out,
                        SUM(pwrw_pp_sh * pwrw_mask_sh) AS pwrw_share_all
                    INTO {sc_out}.analysis_weighted_mix
                    FROM mask_shares
                    LEFT JOIN pp_shares
                    ON pp_shares.run_id = mask_shares.run_id
                        AND pp_shares.nd_id = mask_shares.nd_id
                        AND pp_shares.sy = mask_shares.sy
                    GROUP BY mask_shares.nd_id, mask_shares.run_id, pp, pp_mask, bool_out;
                    ''').format(**self.format_kw)


        if self.bool_run:
            aql.exec_sql(exec_str)

            '''
            /* ################## */
            /* ADD HELPER COLUMNS */
            /* ################## */
            '''

            aql.joinon(self.sw_columns, ['run_id'],
                       [self.sc_out, 'analysis_weighted_mix'],
                       [self.sc_out, 'def_run'])
            aql.joinon(['fl_id', 'pt_id'], ['pp'],
                       [self.sc_out, 'analysis_weighted_mix'],
                       [self.sc_out, 'def_plant'])
            aql.joinon(['pt'], ['pt_id'],
                       [self.sc_out, 'analysis_weighted_mix'],
                       [self.sc_out, 'def_pp_type'])
            aql.joinon(['fl'], ['fl_id'],
                       [self.sc_out, 'analysis_weighted_mix'],
                       [self.sc_out, 'def_fuel'])
            aql.joinon(['nd'], ['nd_id'],
                       [self.sc_out, 'analysis_weighted_mix'],
                       [self.sc_out, 'def_node'])
        return exec_str


# %%
    def mc_histograms(self, nbins=10, valmin=0, valmax=100, pp_like='%STO%',
                      list_timescale=None):


        if list_timescale is None:
            list_timescale = ['']

        for timescale in list_timescale:#, 'wk', 'doy', 'hy']:

            join_timescale_id = \
                    ('''
                     LEFT JOIN (
                     SELECT {timescale}, sy FROM {sc_out}.tm_soy) AS dftsc
                     ON dftsc.sy = pwr.sy
                     ''').format(sc_out=self.sc_out,
                                 timescale=timescale)
            if timescale == '':
                join_timescale_id = ''
            print(join_timescale_id)

            col_timescale_id_0 = timescale + ','
            col_timescale_id = 'dftsc.' + col_timescale_id_0
            tb_mod = timescale + '_'
            col_timescale_join = 'vals.{ts} = complete.{ts} AND'.format(ts=timescale)

            time_scale_keys = {'join_timescale_id': join_timescale_id,
                               'col_timescale_id': col_timescale_id,
                               'col_timescale_id_0': col_timescale_id_0,
                               'tb_mod': tb_mod,
                               'col_timescale_join': col_timescale_join}
            for kk, vv in time_scale_keys.items():
                time_scale_keys[kk] = vv if timescale != '' else ''

            exec_str = '''
                        DROP VIEW IF EXISTS mc_chdc CASCADE;
                        CREATE VIEW mc_chdc AS
                        SELECT pwr.*, dlsp.value AS mc, dfpp.nd_id, dfpp.pt_id,
                               {col_timescale_id}
                               width_bucket(dlsp.value, {valmin}, {valmax}, {nbins}) AS bucket
                        FROM {sc_out}.var_sy_pwr AS pwr
                        LEFT JOIN {sc_out}.def_plant AS dfpp
                            ON dfpp.pp_id = pwr.pp_id
                        LEFT JOIN {sc_out}.dual_supply AS dlsp
                            ON dlsp.sy = pwr.sy
                                AND dlsp.run_id = pwr.run_id
                                AND dlsp.nd_id = dfpp.nd_id
                        {join_timescale_id}
                        WHERE pp LIKE '{pp_like}' AND pwr.value <> 0
                            AND pwr.run_id IN {in_run_id};

                        DROP TABLE IF EXISTS {sc_out}.analysis_{tb_mod}mc_hist CASCADE;
                        WITH complete AS (
                            -- Generate table with complete list of buckets
                            SELECT
                                {valmin} + (bucket - 1) * ({valmax} - {valmin})::FLOAT / {nbins}::FLOAT AS low,
                                {valmin} + (bucket) * ({valmax} - {valmin})::FLOAT / {nbins}::FLOAT AS high,
                                ({valmax} - {valmin}) / {nbins} AS bin_width,
                                bucket,
                                ppcat.*
                            FROM (SELECT generate_series(0, {nbins} + 1) AS bucket, 1 AS dummy) AS bcks
                            FULL OUTER JOIN (SELECT DISTINCT pp_id, pt_id, bool_out, run_id, nd_id, {col_timescale_id_0} 1 AS dummy FROM mc_chdc) AS ppcat
                            ON bcks.dummy = ppcat.dummy
                        ), vals AS (
                            SELECT pp_id, pt_id, bool_out, run_id, nd_id, {col_timescale_id_0}
                                COUNT(*) AS count,
                                SUM(value) AS erg,
                                bucket,
                                min(mc) AS mc_min, max(mc) AS mc_max
                            FROM mc_chdc
                            GROUP BY pp_id, {col_timescale_id_0} pt_id,
                                     bool_out, run_id, nd_id, bucket
                        )
                        SELECT complete.*,
                        COALESCE(count, 0) AS count, COALESCE(erg, 0) AS erg,
                        COALESCE(mc_max, 0) AS mc_max, COALESCE(mc_min, 0) AS mc_min
                        INTO {sc_out}.analysis_{tb_mod}mc_hist
                        FROM complete
                        LEFT JOIN vals
                        ON
                        vals.pp_id = complete.pp_id AND
                        vals.bool_out = complete.bool_out AND
                        vals.run_id = complete.run_id AND
                        {col_timescale_join}
                        vals.bucket = complete.bucket;

                        ALTER TABLE {sc_out}.analysis_{tb_mod}mc_hist
                        ADD COLUMN IF NOT EXISTS center DOUBLE PRECISION;

                        UPDATE {sc_out}.analysis_{tb_mod}mc_hist
                        SET center = 0.5 * (low + high);

                        ;
                        '''.format(nbins=nbins, valmin=valmin,
                                   valmax=valmax, pp_like=pp_like,
                                   **time_scale_keys,
                                   **self.format_kw)

            tb = 'analysis_{tb_mod}mc_hist'.format(**self.format_kw,
                                                   **time_scale_keys)

            aql.exec_sql(exec_str, db=self.db)
            aql.joinon(self.db, ['run_name'] + self.sw_columns, ['run_id'],
                       [self.sc_out, tb], [self.sc_out, 'def_run'])
            aql.joinon(self.db, ['pt'], ['pt_id'],
                       [self.sc_out, tb], [self.sc_out, 'def_pp_type'])
            aql.joinon(self.db, ['nd'], ['nd_id'],
                       [self.sc_out, tb], [self.sc_out, 'def_node'])
            return exec_str




#
#/*
#                        DROP VIEW IF EXISTS mc_chdc CASCADE;
#                        CREATE VIEW mc_chdc AS
#                        SELECT pwr.*, dlsp.value AS mc, dfpp.nd_id, dfpp.pt_id,
#                            {col_timescale_id}
#                            width_bucket(dlsp.value, {valmin}, {valmax}, {nbins}) AS bucket
#                        FROM {sc_out}.var_sy_pwr AS pwr
#                        LEFT JOIN {sc_out}.def_plant AS dfpp
#                            ON dfpp.pp_id = pwr.pp_id
#                        LEFT JOIN {sc_out}.dual_supply AS dlsp
#                            ON dlsp.sy = pwr.sy
#                                AND dlsp.run_id = pwr.run_id
#                                AND dlsp.nd_id = dfpp.nd_id
#                        {join_timescale_id}
#                        WHERE pp LIKE '{pt_like}' AND pwr.value <> 0;
#
#                        DROP TABLE IF EXISTS {tb_mod}mc_hist CASCADE;
#                        SELECT pp_id, pt_id, bool_out, run_id, nd_id, {col_timescale_id_0}
#                            COALESCE(COUNT(*), 0) AS count,
#                            COALESCE(SUM(value), 0) AS erg,
#                            {valmin} + (bucket - 1) * ({valmax} - {valmin})::FLOAT / {nbins}::FLOAT AS low,
#                            {valmin} + (bucket) * ({valmax} - {valmin})::FLOAT / {nbins}::FLOAT AS high,
#                            ({valmax} - {valmin}) / {nbins} AS bin_width,
#                            min(mc) AS mc_min, max(mc) AS mc_max
#                        INTO {tb_mod}mc_hist
#                        FROM mc_chdc
#                        GROUP BY {col_timescale_id_0} pp_id, pt_id, bool_out, run_id, nd_id, bucket
#                        ORDER BY {col_timescale_id_0} pp_id, pt_id, bool_out, run_id, nd_id, bucket
#*/
    # %% Node-node run

    def build_table_node_node_run(self):
        '''
        /* ############################### */
        /* GENERATE TABLE node_node_run    */
        /* ############################### */
        '''

        exec_str = ('''
                    DROP TABLE IF EXISTS node_node_run CASCADE;
                    WITH vrtr AS (
                        SELECT * FROM {sc_out}.var_tr_trm_rv
                        UNION ALL
                        SELECT
                            sy, nd_id, nd_2_id, ca_id, bool_out,
                            -value AS value,
                            run_id FROM {sc_out}.var_tr_trm_sd
                    )
                    SELECT dfnd.nd AS nd, dfnd_2.nd AS nd_2, bool_out, vrtr.run_id,
                        SUM(vrtr.value * weight) AS erg_yr_sy,
                        AVG(vrtr.value / (NULLIF(captrm.value, 0))) AS cf_sy
                    INTO node_node_run
                    FROM vrtr
                    LEFT JOIN {sc_out}.def_node AS dfnd
                        ON dfnd.nd_id = vrtr.nd_id
                    LEFT JOIN {sc_out}.def_node AS dfnd_2
                        ON dfnd_2.nd_id = vrtr.nd_2_id
                    LEFT JOIN {sc_out}.tm_soy AS dfsy
                        ON dfsy.sy = vrtr.sy
                    LEFT JOIN {sc_out}.par_cap_trm_leg AS captrm
                        ON captrm.nd_id = vrtr.nd_id
                        AND captrm.nd_2_id = vrtr.nd_2_id
                        AND captrm.mt_id = dfsy.mt_id
                        AND captrm.run_id = vrtr.run_id
                    GROUP BY dfnd.nd, dfnd_2.nd, bool_out, vrtr.run_id;
                    ''').format(**self.format_kw)
        if self.bool_run:
            aql.exec_sql(exec_str)

            aql.joinon(['run_name'] + self.sw_columns, ['run_id'],
                       ['public', 'node_node_run'], [self.sc_out, 'def_run'])
        return exec_str



    # %% table run
    def table_run(self):
        '''
        /* ############################### */
        /* GENERATE TABLE run              */
        /* ############################### */
        '''

        exec_str = ('''
                    DROP TABLE IF EXISTS run CASCADE;
                    DROP VIEW IF EXISTS run_0 CASCADE;

                    CREATE VIEW run_0 AS
                    SELECT
                        run_id, bool_out, nd_id, nd, run_name,
                        SUM(CASE WHEN set_def_chp = 1 THEN cap_pwr_tot ELSE 0 END) AS cap_pwr_chp_tot,
                        SUM(CASE WHEN set_def_winsol = 1 THEN erg_yr_sy ELSE 0 END) AS erg_yr_sy_ws_tot,
                        SUM(CASE WHEN set_def_st = 1 THEN erg_yr_sy ELSE 0 END) AS erg_yr_sy_st,
                        SUM(erg_yr_sy) AS erg_yr_sy_tot
                    FROM plant_run_tot
                    GROUP BY run_id, bool_out, nd_id, nd, run_name;

                    SELECT *,
                        erg_yr_sy_ws_tot / NULLIF(erg_yr_sy_tot, 0) AS erg_yr_sy_ws_share
                    INTO run
                    FROM run_0
                    ''')
        aql.exec_sql(exec_str)


        print(exec_str)

    # %% Table energy balance

    def build_table_plant_run_tot_balance(self, from_quick=False):

        erg_col = 'erg_yr_yr' if from_quick else 'erg_yr_sy'
        tb_base = 'analysis_plant_run_quick' if from_quick else 'analysis_plant_run_tot'

        exec_str = ('''
                    DROP TABLE IF EXISTS
                        {sc_out}.analysis_plant_run_tot_balance CASCADE;
                    SELECT
                        run_id, bool_out,
                        (CASE WHEN bool_out = False THEN 1 ELSE -1 END ) * ABS({erg_col}) AS {erg_col}_posneg,
                        pp_broad_cat, pt, fl, nd, nd_id, ca_id, pp
                    INTO {sc_out}.analysis_plant_run_tot_balance
                    FROM {sc_out}.{tb_base};

                    /* GRID LOSSES */
                    INSERT INTO {sc_out}.analysis_plant_run_tot_balance
                    SELECT
                        prt.run_id, True AS bool_out,
                        - SUM(pp_sign * {erg_col} * grid_losses) AS erg_yr_sy,
                        'GRIDLOSSES' AS pp_broad_cat,
                        'GRDLSS' AS pt, 'gridlosses' AS fl,
                        dfnd.nd, grdlss.nd_id, prt.ca_id,
                        prt.nd::VARCHAR || '_GRIDLSS' AS pp
                    FROM {sc_out}.{tb_base} AS prt
                    LEFT JOIN (SELECT nd_id, ca_id, run_id,
                                      value AS grid_losses
                               FROM {sc_out}.par_grid_losses) AS grdlss
                        ON grdlss.nd_id = prt.nd_id
                        AND grdlss.ca_id = prt.ca_id
                        AND grdlss.run_id = prt.run_id
                    LEFT JOIN (SELECT nd_id, nd
                               FROM {sc_out}.def_node) AS dfnd
                        ON dfnd.nd_id = prt.nd_id
                    LEFT JOIN (SELECT pp, CASE WHEN set_def_curt = 1 OR set_def_sll = 1 THEN -1 ELSE 1 END AS pp_sign
                               FROM {sc_out}.def_plant) AS map_sign ON map_sign.pp = prt.pp
                    WHERE
                        pp_broad_cat IN ('HYDRO', 'CHP', 'CONVDISP',
                                         'VARIABLE', 'RENDISP',
                                         'HYDROSTORAGE', 'NEW_STORAGE',
                                         'TRNS_RV')
                        AND prt.bool_out = False OR (prt.bool_out = True AND pp_broad_cat = 'DMND_FLEX')
                    GROUP BY prt.run_id, prt.bool_out, prt.nd, grdlss.nd_id, dfnd.nd, prt.ca_id
                    ORDER BY run_id;
                    ''').format(**self.format_kw, erg_col=erg_col, tb_base=tb_base)
        aql.exec_sql(exec_str, db=self.db)

        if len(self.sw_columns) > 0:
            aql.joinon(self.db, self.sw_columns,
                       ['run_id'], [self.sc_out, 'analysis_plant_run_tot_balance'],
                       [self.sc_out, 'def_run'])

        aql.joinon(self.db, ['nd'],
                   ['nd_id'], [self.sc_out, 'analysis_plant_run_tot_balance'],
                   [self.sc_out, 'def_node'])
        aql.joinon(self.db, ['ca'],
                   ['ca_id'], [self.sc_out, 'analysis_plant_run_tot_balance'],
                   [self.sc_out, 'def_encar'])
        return exec_str


    # %% Simultaneous charging/discharging

    def check_simultaneous_charging_discharging(self):
        exec_str=('''
                    DROP TABLE IF EXISTS check_simultaneous_ch_dc CASCADE;

                    WITH tb_false AS (
                        SELECT run_id, pp_id, sy, value
                            AS disch FROM {sc_out}.var_sy_pwr
                        WHERE pp_id IN
                            (SELECT pp_id FROM {sc_out}.def_plant
                                 WHERE pp LIKE '%STO%')
                        AND bool_out = False
                    ), tb_true AS (
                        SELECT run_id, pp_id, sy, value
                            AS charg FROM {sc_out}.var_sy_pwr
                        WHERE pp_id IN
                            (SELECT pp_id FROM {sc_out}.def_plant
                                 WHERE pp LIKE '%STO%')
                        AND bool_out = True
                    )
                    SELECT tb_true.*, tb_false.disch
                    INTO check_simultaneous_ch_dc
                    FROM tb_false
                    LEFT JOIN tb_true
                        ON tb_false.run_id = tb_true.run_id
                            AND tb_false.sy = tb_true.sy
                            AND tb_false.pp_id = tb_true.pp_id
                    WHERE charg <> 0 AND disch <> 0
                    ORDER BY run_id, sy
                    ''').format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)
        return exec_str


    # %% Comparison to yearly stats

    def comparison_yearly_stats(self, sc_inp):

        self.format_kw.update({'sc_inp': sc_inp})

#        exec_str = ('''
#                    DROP TABLE IF EXISTS comp_erg CASCADE;
#
#                    SELECT dffl.fl, erg.nd_id, value
#                    INTO comp_erg
#                    FROM {sc_inp}.erg_yr_comp AS erg
#                    LEFT JOIN {sc_out}.def_fuel AS dffl
#                    ON dffl.fl_id = erg.fl_id
#                    UNION
#                    SELECT
#                        variable AS fl, nd_id,
#                        (CASE WHEN variable = 'export' THEN -1 ELSE 1 END) * SUM(value) AS value
#                    FROM {sc_inp}.imex_comp
#                    GROUP BY variable, nd_id
#                    UNION
#                    SELECT
#                        'pumped_hydro_charg' AS fl,
#                        nd_id, -1 * charging AS value
#                    FROM {sc_out}.def_node;
#
#                    DROP TABLE IF EXISTS var_sy_pwr_slct CASCADE;
#
#                    WITH pwr_m1 AS (
#                        SELECT pp_id, bool_out,
#                            SUM(value * weight) AS sum_value_m1
#                        FROM {sc_out}.var_sy_pwr AS pwr
#                        LEFT JOIN {sc_out}.tm_soy AS dfsy ON pwr.sy = dfsy.sy
#                        WHERE run_id=-1
#                        GROUP BY pp_id, bool_out
#                    ), pwr_0 AS (
#                        SELECT pp_id, bool_out,
#                            SUM(value * weight) AS sum_value_0
#                        FROM {sc_out}.var_sy_pwr AS pwr
#                        LEFT JOIN {sc_out}.tm_soy AS dfsy ON pwr.sy = dfsy.sy
#                        WHERE run_id=0
#                        GROUP BY pp_id, bool_out
#                    ), erg_m1 AS (
#                        SELECT pp_id, bool_out,
#                            value AS sum_value_erg_m1
#                        FROM {sc_out}.var_yr_erg_yr
#                        WHERE run_id=-1
#                    ), erg_0 AS (
#                        SELECT pp_id, bool_out,
#                            value AS sum_value_erg_0
#                        FROM {sc_out}.var_yr_erg_yr
#                        WHERE run_id=0
#                    )
#                    SELECT pwr_m1.*, pwr_0.sum_value_0, erg_m1.sum_value_erg_m1,
#                           erg_0.sum_value_erg_0,
#                        dfpl.fl_id, dfpl.nd_id, dfpl.pt_id
#                    INTO var_sy_pwr_slct
#                    FROM pwr_m1
#                    LEFT JOIN {sc_out}.def_plant AS dfpl
#                        ON pwr_m1.pp_id = dfpl.pp_id
#                    LEFT JOIN pwr_0
#                        ON pwr_0.pp_id = pwr_m1.pp_id
#                            AND pwr_0.bool_out = pwr_m1.bool_out
#                    LEFT JOIN erg_m1
#                        ON pwr_m1.pp_id = erg_m1.pp_id
#                            AND pwr_m1.bool_out = erg_m1.bool_out
#                    LEFT JOIN erg_0
#                        ON pwr_m1.pp_id = erg_0.pp_id
#                            AND pwr_m1.bool_out = erg_0.bool_out
#
#                    ''').format(sc_out=self.sc_out, sc_inp=self.sc_inp)
        exec_str = ('''
                    /* COMPARISON TO STATS */
                    DROP TABLE IF EXISTS comp_erg_analysis CASCADE;
                    SELECT fl, nd, run_name, erg
                    INTO comp_erg_analysis
                    FROM (
                        SELECT fl, nd, run_id, SUM(erg_yr_sy) AS erg
                        FROM plant_run_tot
                        WHERE run_id in (-1, 0)
                        GROUP BY fl, nd, run_id
                        UNION ALL
                        SELECT fl, nd, CAST(-2 AS SMALLINT) AS run_id, value AS erg FROM {sc_inp}.erg_yr_comp AS cmp
                        LEFT JOIN {sc_out}.def_node AS dfnd ON dfnd.nd_id = cmp.nd_id
                        LEFT JOIN {sc_out}.def_fuel AS dffl ON dffl.fl_id = cmp.fl_id
                        UNION ALL
                        SELECT
                            variable AS fl, nd, CAST(-2 AS SMALLINT) AS run_id,
                            (CASE WHEN variable = 'import' THEN 1 ELSE -1 END) * SUM(value) AS erg
                        FROM {sc_inp}.imex_comp AS cmp
                        LEFT JOIN {sc_out}.def_node AS dfnd ON dfnd.nd_id = cmp.nd_id
                        GROUP BY fl, nd
                    ) AS mrg
                    LEFT JOIN def_run_name AS dfrn ON dfrn.run_id = mrg.run_id;
                    ''').format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)
        return(exec_str)


    # %% total CO2 emissions from linear factors
    @DecoratorsSqlAnalysis.append_nd_id_columns('analysis_emissions_lin')
    @DecoratorsSqlAnalysis.append_pt_id_columns('analysis_emissions_lin')
    @DecoratorsSqlAnalysis.append_fl_id_columns('analysis_emissions_lin')
    @DecoratorsSqlAnalysis.append_pp_id_columns('analysis_emissions_lin')
    @DecoratorsSqlAnalysis.append_sw_columns('analysis_emissions_lin')

    def analysis_emissions_lin(self):


        tb_name = 'analysis_emissions_lin'
        cols = [('pp_id', 'SMALLINT'),
                ('ca_id', 'SMALLINT'),
                ('run_id', 'SMALLINT'),
                ('emissions', 'DOUBLE PRECISION'),
                ('emissions_cost', 'DOUBLE PRECISION'),
               ]
        pk = ['pp_id', 'ca_id', 'run_id']
        cols = aql.init_table(tb_name=tb_name, cols=cols, schema=self.sc_out,
                              pk=pk, db=self.db)


        exec_strg = '''
        WITH tb_final AS (
            SELECT pwr.pp_id, pwr.ca_id, pwr.run_id,
            SUM(pwr.value * weight
                * (factor_vc_co2_lin_0
                   + 0.5 * pwr.value * factor_vc_co2_lin_1)) AS emissions,
            SUM(pwr.value * weight * price_co2
                * (factor_vc_co2_lin_0
                   + 0.5 * pwr.value * factor_vc_co2_lin_1)) AS emissions_cost
            FROM {sc_out}.var_sy_pwr AS pwr

            /* factor 0 */
            LEFT JOIN (SELECT pp_id, ca_id, run_id,
                       value AS factor_vc_co2_lin_0
                       FROM {sc_out}.par_factor_vc_co2_lin_0) AS parco0
            ON parco0.pp_id = pwr.pp_id AND parco0.ca_id = pwr.ca_id
                AND parco0.run_id = pwr.run_id

            /* factor 1 */
            LEFT JOIN (SELECT pp_id, ca_id, run_id,
                       value AS factor_vc_co2_lin_1
                       FROM {sc_out}.par_factor_vc_co2_lin_1) AS parco1
            ON parco1.pp_id = pwr.pp_id AND parco1.ca_id = pwr.ca_id
                AND parco1.run_id = pwr.run_id

            /* def_plant*/
            LEFT JOIN (SELECT fl_id, nd_id, pp_id
                       FROM {sc_out}.def_plant) AS dfpp
            ON dfpp.pp_id = pwr.pp_id

            /* tm soy/ weight*/
            LEFT JOIN (SELECT sy, weight, mt_id
                       FROM {sc_out}.tm_soy) AS tm
            ON pwr.sy = tm.sy

            /* CO2 price  */
            LEFT JOIN (SELECT mt_id, nd_id, run_id, value AS price_co2
            FROM {sc_out}.par_price_co2) AS parprco
            ON parprco.nd_id = dfpp.nd_id AND parprco.mt_id = tm.mt_id
                AND parprco.run_id = pwr.run_id

            WHERE pwr.pp_id IN (SELECT pp_id
                                FROM {sc_out}.def_plant
                                WHERE set_def_lin = 1)
                AND pwr.run_id IN {in_run_id}
            GROUP BY pwr.pp_id, pwr.ca_id, pwr.run_id
        )
        INSERT INTO {sc_out}.analysis_emissions_lin ({cols})
        SELECT {cols} FROM tb_final;
        '''.format(**self.format_kw, cols=cols)

        aql.exec_sql(exec_strg, db=self.db)



    # %% COST DISAGGREGATION NEW

    @DecoratorsSqlAnalysis.append_nd_id_columns('analysis_cost_disaggregation_lin')
    @DecoratorsSqlAnalysis.append_pt_id_columns('analysis_cost_disaggregation_lin')
    @DecoratorsSqlAnalysis.append_fl_id_columns('analysis_cost_disaggregation_lin')
    @DecoratorsSqlAnalysis.append_pp_id_columns('analysis_cost_disaggregation_lin')
    @DecoratorsSqlAnalysis.append_sw_columns('analysis_cost_disaggregation_lin')
    def analysis_cost_disaggregation_lin(self):

        tb_name = 'analysis_cost_disaggregation_lin'
        cols = [('pp_id', 'SMALLINT'),
                ('ca_id', 'SMALLINT'),
                ('run_id', 'SMALLINT'),
                ('value', 'DOUBLE PRECISION'),
                ('type', 'VARCHAR'),
               ]
        pk = ['pp_id', 'ca_id', 'run_id', 'type']
        aql.init_table(tb_name=tb_name,
                       cols=cols, schema=self.sc_out,
                       pk=pk, db=self.db)


        exec_strg = '''
        DROP TABLE IF EXISTS temp_vc_lin CASCADE;
        SELECT pwr.pp_id, pwr.ca_id, pwr.run_id,
        SUM(pwr.value * weight * vc_fl * (factor_vc_fl_lin_0 + 0.5 * pwr.value * factor_vc_fl_lin_1)) AS value_vc_fl_lin,
        SUM(pwr.value * weight * price_co2 * (factor_vc_co2_lin_0 + 0.5 * pwr.value * factor_vc_co2_lin_1)) AS value_vc_co2_lin
        INTO temp_vc_lin
        FROM {sc_out}.var_sy_pwr AS pwr
        LEFT JOIN (SELECT pp_id, ca_id, run_id, value AS factor_vc_fl_lin_0
                   FROM {sc_out}.par_factor_vc_fl_lin_0) AS parfl0
            ON parfl0.pp_id = pwr.pp_id AND parfl0.ca_id = pwr.ca_id
            AND parfl0.run_id = pwr.run_id
        LEFT JOIN (SELECT pp_id, ca_id, run_id, value AS factor_vc_fl_lin_1
                   FROM {sc_out}.par_factor_vc_fl_lin_1) AS parfl1
            ON parfl1.pp_id = pwr.pp_id AND parfl1.ca_id = pwr.ca_id
            AND parfl1.run_id = pwr.run_id
        LEFT JOIN (SELECT pp_id, ca_id, run_id, value AS factor_vc_co2_lin_0 FROM {sc_out}.par_factor_vc_co2_lin_0) AS parco0
        ON parco0.pp_id = pwr.pp_id AND parco0.ca_id = pwr.ca_id AND parco0.run_id = pwr.run_id
        LEFT JOIN (SELECT pp_id, ca_id, run_id, value AS factor_vc_co2_lin_1 FROM {sc_out}.par_factor_vc_co2_lin_1) AS parco1
        ON parco1.pp_id = pwr.pp_id AND parco1.ca_id = pwr.ca_id AND parco1.run_id = pwr.run_id
        LEFT JOIN (SELECT fl_id, nd_id, pp_id FROM {sc_out}.def_plant) AS dfpp
        ON dfpp.pp_id = pwr.pp_id
        LEFT JOIN (SELECT sy, weight, mt_id FROM {sc_out}.tm_soy) AS tm
        ON pwr.sy = tm.sy
        LEFT JOIN (SELECT mt_id, fl_id, nd_id, run_id, value AS vc_fl FROM {sc_out}.par_vc_fl) AS parvcfl
        ON parvcfl.fl_id = dfpp.fl_id AND parvcfl.nd_id = dfpp.nd_id AND parvcfl.mt_id = tm.mt_id AND parvcfl.run_id = pwr.run_id
        LEFT JOIN (SELECT mt_id, nd_id, run_id, value AS price_co2 FROM {sc_out}.par_price_co2) AS parprco
        ON parprco.nd_id = dfpp.nd_id AND parprco.mt_id = tm.mt_id AND parprco.run_id = pwr.run_id
        WHERE pwr.pp_id IN (SELECT pp_id FROM {sc_out}.def_plant WHERE set_def_lin = 1)
        AND pwr.run_id IN {in_run_id}
        GROUP BY pwr.pp_id, pwr.ca_id, pwr.run_id;
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)

        for vc in ['fl', 'co2']:
            print('Inserting vc_%s_lin ... '%vc, end='')
            exec_strg = '''
            INSERT INTO {sc_out}.analysis_cost_disaggregation_lin (pp_id, ca_id, run_id, value, type)
            SELECT pp_id, ca_id, run_id, value_vc_{vc}_lin AS value, 'vc_{vc}_lin'::VARCHAR AS type
            FROM temp_vc_lin;
            '''.format(**self.format_kw, vc=vc)
            aql.exec_sql(exec_strg, db=self.db)
            print('done.')

        print('Inserting vc_fl ... ', end='')
        exec_strg = '''
        DELETE FROM {sc_out}.analysis_cost_disaggregation_lin
        WHERE type = 'vc_fl';

        WITH tb_final AS (
            SELECT pwr.pp_id, pwr.ca_id, pwr.run_id,
            SUM(pwr.value * weight * vc_fl / pp_eff) AS value_vc_fl
            FROM {sc_out}.var_sy_pwr AS pwr
            LEFT JOIN (SELECT fl_id, nd_id, pp_id, pp FROM {sc_out}.def_plant) AS dfpp
            ON dfpp.pp_id = pwr.pp_id
            LEFT JOIN (SELECT pp_id, ca_id, run_id, value AS pp_eff FROM {sc_out}.par_pp_eff) AS eff
            ON eff.pp_id = pwr.pp_id AND eff.ca_id = pwr.ca_id AND eff.run_id = pwr.run_id
            LEFT JOIN (SELECT sy, weight, mt_id FROM {sc_out}.tm_soy) AS tm
            ON pwr.sy = tm.sy
            LEFT JOIN (SELECT mt_id, fl_id, nd_id, run_id, value AS vc_fl FROM {sc_out}.par_vc_fl) AS parvcfl
            ON parvcfl.fl_id = dfpp.fl_id AND parvcfl.nd_id = dfpp.nd_id AND parvcfl.mt_id = tm.mt_id AND parvcfl.run_id = pwr.run_id
            WHERE pwr.pp_id IN (SELECT pp_id FROM {sc_out}.def_plant WHERE set_def_lin = 0 AND set_def_pp = 1)
            AND pwr.run_id IN {in_run_id}
            GROUP BY pwr.pp_id, pwr.ca_id, pwr.run_id
        )
        INSERT INTO {sc_out}.analysis_cost_disaggregation_lin (pp_id, ca_id, run_id, value, type)
        SELECT pp_id, ca_id, run_id, value_vc_fl AS value, 'vc_fl'::VARCHAR AS type
        FROM tb_final;
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)
        print('done.')


        print('Inserting vc_fl_agg ... ', end='')
        exec_strg = '''
        DELETE FROM {sc_out}.analysis_cost_disaggregation_lin
        WHERE type = 'vc_fl_agg';

        INSERT INTO {sc_out}.analysis_cost_disaggregation_lin (pp_id, ca_id, run_id, value, type)
        SELECT pp_id, ca_id, run_id, value, 'vc_fl_agg'::VARCHAR AS type
        FROM {sc_out}.var_yr_vc_om_pp_yr
        WHERE run_id IN {in_run_id}
        AND pp_id IN (SELECT pp_id FROM {sc_out}.def_plant WHERE set_def_pp = 1 AND set_def_lin = 0);
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)
        print('done.')

        print('Inserting vc_om ... ', end='')
        exec_strg = '''
        DELETE FROM {sc_out}.analysis_cost_disaggregation_lin
        WHERE type = 'vc_om';

        WITH tb_final AS (
            SELECT pwr.pp_id, pwr.ca_id, pwr.run_id,
            SUM(pwr.value * weight * vc_om) AS value_vc_om
            FROM {sc_out}.var_sy_pwr AS pwr
            LEFT JOIN (SELECT sy, weight, mt_id FROM {sc_out}.tm_soy) AS tm
            ON pwr.sy = tm.sy
            LEFT JOIN (SELECT pp_id, ca_id, run_id, value AS vc_om FROM {sc_out}.par_vc_om) AS parvcom
            ON parvcom.pp_id = pwr.pp_id AND parvcom.ca_id = pwr.ca_id AND parvcom.run_id = pwr.run_id
            WHERE pwr.run_id IN {in_run_id}
            GROUP BY pwr.pp_id, pwr.ca_id, pwr.run_id
        )
        INSERT INTO {sc_out}.analysis_cost_disaggregation_lin (pp_id, ca_id, run_id, value, type)
        SELECT pp_id, ca_id, run_id, value_vc_om AS value, 'vc_om'::VARCHAR AS type
        FROM tb_final;
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)
        print('done.')

        print('Inserting fc_om ... ', end='')
        exec_strg = '''
        DELETE FROM {sc_out}.analysis_cost_disaggregation_lin
        WHERE type = 'fc_om';

        WITH tb_final AS (
            SELECT cap.pp_id, cap.ca_id, run_id, cap.value * fc_om AS value
            FROM {sc_out}.var_yr_cap_pwr_tot AS cap
            LEFT JOIN (SELECT pp_id, ca_id, fc_om FROM {sc_out}.plant_encar) AS ppca ON ppca.pp_id = cap.pp_id AND ppca.ca_id = cap.ca_id
            LEFT JOIN (SELECT pp_id, pp FROM {sc_out}.def_plant) AS dfpp ON dfpp.pp_id = cap.pp_id
            WHERE run_id IN {in_run_id}
        )
        INSERT INTO {sc_out}.analysis_cost_disaggregation_lin (pp_id, ca_id, run_id, value, type)
        SELECT pp_id, ca_id, run_id, value, 'fc_om'::VARCHAR AS type
        FROM tb_final;
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)
        print('done.')


        print('Inserting vc_ramp ... ', end='')
        exec_strg = '''
        DELETE FROM {sc_out}.analysis_cost_disaggregation_lin
        WHERE type = 'vc_ramp';

        INSERT INTO {sc_out}.analysis_cost_disaggregation_lin (pp_id, ca_id, run_id, value, type)
        SELECT pp_id, ca_id, run_id, value, 'vc_ramp'::VARCHAR AS type
        FROM {sc_out}.var_yr_vc_ramp_yr
        WHERE pp_id IN (SELECT pp_id FROM {sc_out}.def_plant
                        WHERE set_def_pp = 1  OR set_def_ror = 1 OR set_def_hyrs = 1)
            AND run_id IN {in_run_id}
        ;
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)
        print('done.')

        print('Inserting total_total ... ', end='')
        exec_strg = '''
        DELETE FROM {sc_out}.analysis_cost_disaggregation_lin
        WHERE type = 'total_total';

        INSERT INTO {sc_out}.analysis_cost_disaggregation_lin (pp_id, ca_id, run_id, value, type)
        SELECT -1 AS pp_id, -1 AS ca_id, run_id, SUM(COALESCE(value, 0)), 'total_total'::VARCHAR AS type
        FROM {sc_out}.analysis_cost_disaggregation_lin
        WHERE type IN ('fc_om', 'vc_co2_lin', 'vc_fl',
                       'vc_fl_lin', 'vc_om', 'vc_ramp')
        AND run_id IN {in_run_id}
        GROUP BY run_id;
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)
        print('done.')


        print('Inserting objective ... ', end='')
        exec_strg = '''
        DELETE FROM {sc_out}.analysis_cost_disaggregation_lin
        WHERE type = 'objective';

        INSERT INTO {sc_out}.analysis_cost_disaggregation_lin (pp_id, ca_id, run_id, value, type)
        SELECT 0 AS pp_id, 0 AS ca_id, run_id, objective AS value, 'objective'::VARCHAR AS type
        FROM {sc_out}.def_run
        WHERE run_id IN {in_run_id};
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)
        print('done.')

    # %% COST DISAGGREGATION
    def cost_disaggregation(self):
        exec_str = ('''
              /*      /* CAPITAL COST OF INVESTMENT */
                    DROP VIEW IF EXISTS cc_fc_cp CASCADE;
                    CREATE VIEW cc_fc_cp AS
                    SELECT pp, cpnw.pp_id, cpnw.value AS cap_pwr_new, cpnw.run_id, dfpp.nd_id, dr, fc_cp, lt,
                     --      (dr * (1+dr)^lt) / ((1 + dr)^lt - 1) AS crf,
                     --      fc_cp * (dr * (1+dr)^lt) / ((1 + dr)^lt - 1) AS fc_cp_ann,
                     --      fc_cp * (dr * (1+dr)^lt) / ((1 + dr)^lt - 1) * cpnw.value AS value
                    FROM {sc_out}.var_yr_cap_pwr_new AS cpnw
                    LEFT JOIN {sc_out}.def_plant AS dfpp ON cpnw.pp_id = dfpp.pp_id
                    LEFT JOIN (SELECT nd_id, discount_rate AS dr FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = dfpp.nd_id
                    LEFT JOIN {sc_out}.plant_encar AS ppca ON ppca.pp_id = cpnw.pp_id
                    ORDER BY crf;

                    /* CAPITAL COST OF INVESTMENT */
                    DROP VIEW IF EXISTS cc_fc_cp CASCADE;
                    CREATE VIEW cc_fc_cp AS
                    SELECT pp, cpnw.pp_id, cpnw.value AS cap_pwr_new, cpnw.run_id, dfpp.nd_id, dr, fc_cp, lt,
                           fc_cp_ann,
                           fc_cp_ann * cpnw.value AS value
                    FROM {sc_out}.var_yr_cap_pwr_new AS cpnw
                    LEFT JOIN {sc_out}.def_plant AS dfpp ON cpnw.pp_id = dfpp.pp_id
                    LEFT JOIN (SELECT nd_id, discount_rate AS dr FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = dfpp.nd_id
                    LEFT JOIN {sc_out}.plant_encar AS ppca ON ppca.pp_id = cpnw.pp_id;
              */

                    /* CAPITAL COST OF INVESTMENT */
                    DROP VIEW IF EXISTS cc_fc_cp CASCADE;
                    CREATE VIEW cc_fc_cp AS
                    SELECT pp, cpnw.pp_id, dfpp.nd_id, cpnw.run_id, cpnw.value AS cap_pwr_new, fcann.value AS fc_cp_ann,
                           fcann.value * cpnw.value AS value
                    FROM {sc_out}.var_yr_cap_pwr_new AS cpnw
                    LEFT JOIN {sc_out}.def_plant AS dfpp ON cpnw.pp_id = dfpp.pp_id
                    LEFT JOIN (SELECT nd_id, discount_rate AS dr FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = dfpp.nd_id
                    LEFT JOIN {sc_out}.par_fc_cp_ann AS fcann ON fcann.pp_id = cpnw.pp_id AND fcann.run_id = cpnw.run_id;

                    /* FIXED O&M COST */
                    DROP VIEW IF EXISTS cc_fc_om CASCADE;
                    CREATE VIEW cc_fc_om AS
                    SELECT pp, cptt.pp_id, cptt.value AS cap_pwr_tot,
                    cptt.run_id, dfpp.nd_id, ppca.fc_om,
                    ppca.fc_om * cptt.value AS value
                    FROM {sc_out}.var_yr_cap_pwr_tot AS cptt
                    LEFT JOIN {sc_out}.def_plant AS dfpp ON cptt.pp_id = dfpp.pp_id
                    LEFT JOIN {sc_out}.plant_encar AS ppca ON ppca.pp_id = cptt.pp_id;

                    /* VARIABLE O&M AND FUEL COSTS */
                    DROP VIEW IF EXISTS cc_vc CASCADE;
                    CREATE VIEW cc_vc AS
                    SELECT pp, pprn.pp_id, pprn.erg_yr_sy AS erg_yr_sy, pprn.run_id, nd_id,
                    ppca.vc_om AS sp_vc_om, ppca.vc_fl AS sp_vc_fl,
                    ppca.vc_om * erg_yr_sy AS cc_vc_om,
                    ppca.vc_fl * erg_yr_sy AS cc_vc_fl
                    FROM plant_run_tot AS pprn
                    LEFT JOIN {sc_out}.plant_encar AS ppca ON ppca.pp_id = pprn.pp_id
                    WHERE bool_out = False;

                    /* VARIABLE O&M AND FUEL COSTS */
                    DROP VIEW IF EXISTS cc_vc_co2 CASCADE;
                    CREATE VIEW cc_vc_co2 AS
                    SELECT pp, pprn.pp_id, pprn.erg_yr_sy AS erg_yr_sy, pprn.run_id, nd_id,
                    parco2.value AS sp_vc_co2,
                    parco2.value * erg_yr_sy AS cc_vc_co2
                    FROM plant_run_tot AS pprn
                    LEFT JOIN {sc_out}.par_vc_co2 AS parco2 ON parco2.pp_id = pprn.pp_id AND parco2.run_id = pprn.run_id;

                    /* VARIABLE RAMPING COST */
                    DROP VIEW IF EXISTS cc_rp CASCADE;
                    CREATE VIEW cc_rp AS
                    SELECT pp, vrrp.pp_id, nd_id, vrrp.run_id, vrrp.value AS tt_ramp, vcrp.value AS vc_ramp, vrrp.value * vcrp.value AS tc
                    FROM {sc_out}.var_yr_pwr_ramp_yr AS vrrp
                    LEFT JOIN {sc_out}.def_plant AS dfpp ON vrrp.pp_id = dfpp.pp_id
                    LEFT JOIN {sc_out}.par_vc_ramp AS vcrp ON vrrp.pp_id = vcrp.pp_id AND vrrp.run_id = vcrp.run_id;

                    /* FLEXIBLE DEMAND */
                    DROP VIEW IF EXISTS cc_fx CASCADE;
                    CREATE VIEW cc_fx AS
                    SELECT pp, pp_id, pprn.nd_id, run_id, -erg_yr_sy AS driver, vc_dmnd_flex AS spc, -erg_yr_sy * vc_dmnd_flex AS ttc
                    FROM plant_run_tot AS pprn
                    LEFT JOIN {sc_out}.def_node AS dfnd ON pprn.nd_id = dfnd.nd_id
                    WHERE pp LIKE '%DMND_FLEX%';


                    DROP VIEW IF EXISTS plant_run_cost_disaggregation_0 CASCADE;
                    CREATE VIEW plant_run_cost_disaggregation_0 AS
                    SELECT pp, pp_id, nd_id, run_id, CAST('fc_cp' AS VARCHAR) AS comp, cap_pwr_new AS driver, fc_cp_ann AS spc, value AS ttc
                    FROM cc_fc_cp
                    UNION ALL
                    SELECT pp, pp_id, nd_id, run_id, CAST('fc_om' AS VARCHAR) AS comp, cap_pwr_tot AS driver, fc_om AS spc, value AS ttc FROM cc_fc_om
                    UNION ALL
                    SELECT pp, pp_id, nd_id, run_id, CAST('vc_om' AS VARCHAR) AS comp, erg_yr_sy AS driver, sp_vc_om AS spc, cc_vc_om AS ttc FROM cc_vc
                    UNION ALL
                    SELECT pp, pp_id, nd_id, run_id, CAST('vc_fl' AS VARCHAR) AS comp, erg_yr_sy AS driver, sp_vc_fl AS spc, cc_vc_fl AS ttc FROM cc_vc
                    UNION ALL
                    SELECT pp, pp_id, nd_id, run_id, CAST('vc_co2' AS VARCHAR) AS comp, erg_yr_sy AS driver, sp_vc_co2 AS spc, cc_vc_co2 AS ttc FROM cc_vc_co2
                    UNION ALL
                    SELECT 'TOTAL' AS pp, -1 AS pp_id, -1 AS nd_id, run_id, CAST('tc' AS VARCHAR), 1 AS driver, 1 AS spc, value AS ttc FROM {sc_out}.par_objective
                    UNION ALL
                    SELECT pp, pp_id, nd_id, run_id, CAST('vc_rp' AS VARCHAR), tt_ramp AS driver, vc_ramp AS spc, tc AS ttc FROM cc_rp
                    UNION ALL
                    SELECT pp, pp_id, nd_id, run_id, CAST('vc_fx' AS VARCHAR), driver AS driver, spc AS spc, ttc AS ttc FROM cc_fx;


                    DROP TABLE IF EXISTS plant_run_cost_disaggregation CASCADE;

                    SELECT ccda.*, nd, run_name
                    INTO plant_run_cost_disaggregation
                    FROM plant_run_cost_disaggregation_0 AS ccda
                    LEFT JOIN {sc_out}.def_node AS dfnd ON dfnd.nd_id = ccda.nd_id
                    LEFT JOIN {sc_out}.def_run AS dflp ON dflp.run_id = ccda.run_id;

                    ''').format(**self.format_kw)
        aql.exec_sql(exec_str)


        aql.joinon(['pt_id'], ['pp_id'],
                   ['public', 'plant_run_cost_disaggregation'],
                   [self.sc_out, 'def_plant'])
        aql.joinon(['pt'], ['pt_id'],
                   ['public', 'plant_run_cost_disaggregation'],
                   [self.sc_out, 'def_pp_type'])
        aql.joinon(self.sw_columns, ['run_id'],
                   ['public', 'plant_run_cost_disaggregation'],
                   [self.sc_out, 'def_run'])

        return(exec_str)

    # %% COST DISAGGREGATION AS IN OBJECTIVE CONSTRAINT
    def cost_disaggregation_high_level(self):

        exec_str = ('''
                    DROP TABLE IF EXISTS
                        {sc_out}.analysis_plant_run_cost_disaggregation_highlevel
                        CASCADE;
                    SELECT pp_id, run_id, 'vcfl' AS comp, SUM(value) AS ttc
                    INTO {sc_out}.analysis_plant_run_cost_disaggregation_highlevel
                    FROM {sc_out}.var_yr_vc_fl_pp_yr AS vcfl
                    GROUP BY pp_id, run_id, comp
                    UNION ALL
                    SELECT pp_id, run_id, 'vcco2' AS comp, value AS ttc
                        FROM {sc_out}.var_yr_vc_co2_pp_yr AS vcco2
                    UNION ALL
                    SELECT pp_id, run_id, 'vcom' AS comp, value AS ttc
                        FROM {sc_out}.var_yr_vc_om_pp_yr AS vcom
                    UNION ALL
                    SELECT pp_id, run_id, 'vcfx' AS comp, value AS ttc
                        FROM {sc_out}.var_yr_vc_dmnd_flex_yr AS vcfx
                    LEFT JOIN (SELECT pp_id, nd_id FROM {sc_out}.def_plant
                        WHERE pp LIKE '%DMND_F%') AS ppfx ON ppfx.nd_id = vcfx.nd_id
                    UNION ALL
                    SELECT pp_id, run_id, 'vcrp' AS comp, value AS ttc
                        FROM {sc_out}.var_yr_vc_ramp_yr AS vcrp
                    UNION ALL
                    SELECT pp_id, run_id, 'fcom' AS comp, value AS ttc
                        FROM {sc_out}.var_yr_fc_om_pp_yr AS fcom
                    UNION ALL
                    SELECT pp_id, run_id, 'fccp' AS comp, value AS ttc
                        FROM {sc_out}.var_yr_fc_cp_pp_yr AS fccp
                    UNION ALL
                    SELECT -1 AS pp_id, run_id, 'tc'::VARCHAR AS comp, objective AS ttc
                        FROM {sc_out}.def_run
                    ''').format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)

        aql.joinon(self.db, ['nd_id', 'pp', 'fl_id'], ['pp_id'],
                   [self.sc_out, 'analysis_plant_run_cost_disaggregation_highlevel'],
                   [self.sc_out, 'def_plant'])
        aql.joinon(self.db, ['nd'], ['nd_id'],
                   [self.sc_out, 'analysis_plant_run_cost_disaggregation_highlevel'],
                   [self.sc_out, 'def_node'])
        aql.joinon(self.db, ['fl'], ['fl_id'],
                   [self.sc_out, 'analysis_plant_run_cost_disaggregation_highlevel'],
                   [self.sc_out, 'def_fuel'])
        if len(self.sw_columns) > 0:
            aql.joinon(self.db, self.sw_columns, ['run_id'],
                       [self.sc_out, 'analysis_plant_run_cost_disaggregation_highlevel'],
                       [self.sc_out, 'def_run'])

        return exec_str

    @DecoratorsSqlAnalysis.append_sw_columns
    def analysis_chp_shares(self):
        '''
        Compares the chp profile generated from
        '''

        tb_name = 'analysis_chp_shares'
        cols = [('fl', 'VARCHAR'),
                ('nd', 'VARCHAR'),
                ('run_id', 'SMALLINT'),
                ('erg_chp_prf', 'DOUBLE PRECISION'),
                ('erg_chp_input', 'DOUBLE PRECISION'),
                ('erg_tot', 'DOUBLE PRECISION'),
                ('share_chp', 'DOUBLE PRECISION')
               ]
        pk = ['fl', 'nd', 'run_id']
        slct_cols = aql.init_table(tb_name=tb_name,
                                  cols=cols, schema=self.sc_out,
                                  pk=pk, db=self.db)

        exec_strg = '''
        WITH tb_tot AS (
            SELECT fl, nd, run_id, SUM(value) AS erg_tot FROM {sc_out}.var_yr_erg_yr AS erg
            LEFT JOIN (SELECT pp_id, pp, fl_id, nd_id FROM {sc_out}.def_plant) AS dfpp ON dfpp.pp_id = erg.pp_id
            LEFT JOIN (SELECT fl_id, fl FROM {sc_out}.def_fuel) AS dffl ON dffl.fl_id = dfpp.fl_id
            LEFT JOIN (SELECT nd_id, nd FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = dfpp.nd_id
            GROUP BY fl, nd, run_id
        ), tb_chp AS (
            SELECT fl, erg.run_id, nd, SUM(prf.value * tm.weight * erg.value) AS erg_chp_prf, MIN(flndca.erg_chp) AS erg_chp_input FROM {sc_out}.par_chpprof AS prf
            FULL OUTER JOIN {sc_out}.par_erg_chp AS erg ON erg.nd_id = prf.nd_id AND erg.ca_id = prf.ca_id AND erg.run_id = prf.run_id
            LEFT JOIN (SELECT nd_id, nd FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = prf.nd_id
            LEFT JOIN (SELECT fl_id, fl FROM {sc_out}.def_fuel) AS dffl ON dffl.fl_id = erg.fl_id
            LEFT JOIN (SELECT sy, weight FROM {sc_out}.tm_soy) AS tm ON tm.sy = prf.sy
            LEFT JOIN (SELECT is_chp, fl_id, nd_id, ca_id, erg_chp FROM {sc_out}.fuel_node_encar) AS flndca
                ON flndca.fl_id = erg.fl_id AND flndca.nd_id = prf.nd_id
            WHERE is_chp = 1
            GROUP BY nd, erg.run_id, fl
        ), tb_final AS (
            SELECT tb_chp.*, erg_tot, erg_chp_prf / erg_tot AS share_chp FROM tb_chp
            LEFT JOIN tb_tot ON tb_tot.fl = tb_chp.fl AND tb_tot.nd = tb_chp.nd AND tb_tot.run_id = tb_chp.run_id
        )
        INSERT INTO {sc_out}.analysis_chp_shares ({slct_cols})
        SELECT {slct_cols} FROM tb_final;

        '''.format(slct_cols=slct_cols, **self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)



if __name__ == '__main__':

    pass
