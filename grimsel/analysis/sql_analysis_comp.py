# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:11:09 2018

"""

#import sys
#from importlib import reload
#
import pandas as pd
import numpy as np
import grimsel.auxiliary.sqlutils.aux_sql_func as aql
#
import grimsel.auxiliary.maps as maps

import grimsel.analysis.sql_analysis as  sql_analysis

from grimsel.analysis.decorators import DecoratorsSqlAnalysis


class SqlAnalysisComp(sql_analysis.SqlAnalysis):


    def __init__(self, sc_out, db, **kwargs):

        self.mps = maps.Maps(sc_out, db)

        super().__init__(sc_out, db, **kwargs)



    @DecoratorsSqlAnalysis.append_sw_columns('analysis_monthly_comparison')
    def analysis_monthly_comparison(self):

        tb_name = 'analysis_monthly_comparison'
        cols = [('fl', 'VARCHAR'),
                ('mt_id', 'SMALLINT'),
                ('nd', 'VARCHAR'),
                ('run_id', 'SMALLINT'),
                ('erg', 'DOUBLE PRECISION'),
                ('input', 'VARCHAR'),
                ('input_simple', 'VARCHAR'),
               ]
        pk = ['fl', 'mt_id', 'nd', 'run_id', 'input']
        aql.init_table(tb_name=tb_name,
                       cols=cols, schema=self.sc_out,
                       pk=pk, db=self.db)


        exec_strg = '''
        WITH map_input AS (
            SELECT input::VARCHAR, input_simple::VARCHAR
            FROM (VALUES ('agora_month_sum', 'stats'), ('rte_month_sum', 'stats'),
                         ('econtrol_betriebsstatistik', 'stats'),
                         ('ch_elektrizitaetsstatistik', 'stats'),
                         ('model_erg_max_from_cf', 'model_max'),
                         ('model_var_sy_pwr', 'model'), ('model_tr', 'model'),
                         ('entsoe_cross_border', 'stats'),
                         ('entsoe_commercial_exchange', 'stats'),
                         ('terna', 'stats'))
            AS temp (input, input_simple)
/*        ), tb_model_erg_max_from_cf AS (
            SELECT fl, tbcf.mt_id, nd, tbcf.run_id,
                SUM(cap.value * tbcf.value * month_weight) AS erg,
                'model_erg_max_from_cf'::VARCHAR AS input
            FROM {sc_out}.par_cf_max AS tbcf
            NATURAL LEFT JOIN (SELECT pp_id, pp, fl_id, nd_id FROM {sc_out}.def_plant) AS dfpp
            NATURAL LEFT JOIN (SELECT fl_id, fl FROM {sc_out}.def_fuel) AS dffl
            NATURAL LEFT JOIN (SELECT nd_id, nd FROM {sc_out}.def_node) AS dfnd
            LEFT JOIN (SELECT mt_id, month_weight FROM {sc_out}.def_month) AS dfmt ON dfmt.mt_id = tbcf.mt_id
            LEFT JOIN (SELECT pp_id, value, run_id FROM {sc_out}.par_cap_pwr_leg) AS cap ON tbcf.run_id = cap.run_id AND tbcf.pp_id = cap.pp_id
            GROUP BY fl, tbcf.mt_id, nd, tbcf.run_id
 */       ), tb_model_var_sy_pwr AS (
            SELECT fl, mt_id, nd, run_id,
                SUM(value * weight) AS erg,
                'model_var_sy_pwr'::VARCHAR AS input
            FROM {sc_out}.var_sy_pwr AS ergsy
            NATURAL LEFT JOIN (SELECT pp_id, pp, fl_id, nd_id FROM {sc_out}.def_plant) AS dfpp
            NATURAL LEFT JOIN (SELECT fl_id, fl FROM {sc_out}.def_fuel) AS dffl
            NATURAL LEFT JOIN (SELECT nd_id, nd FROM {sc_out}.def_node) AS dfnd
            LEFT JOIN (SELECT mt_id, weight, sy FROM {sc_out}.tm_soy) AS tm ON tm.sy = ergsy.sy
            GROUP BY fl, mt_id, nd, run_id
        ), tb_agora_month_sum AS (
            SELECT fl_id AS fl, mt_id, nd_id AS nd, 0 AS run_id, SUM(value) * 1000 AS erg, 'agora_month_sum'::VARCHAR AS input
            FROM profiles_raw.agora_profiles AS ag
            LEFT JOIN (SELECT mt_id, datetime FROM profiles_raw.timestamp_template) AS ts ON ts.datetime = ag."DateTime"
            WHERE year = 2015
            GROUP BY fl_id, mt_id, nd_id, run_id
        ), tb_rte_month_sum AS (
            SELECT fl_id AS fl, mt_id, nd_id AS nd, 0 AS run_id, SUM(value) AS erg, 'rte_month_sum'::VARCHAR AS input
            FROM profiles_raw.rte_production_eco2mix AS rte
            WHERE year = 2015
            GROUP BY fl_id, mt_id, nd_id, run_id
        ), tb_entsoe_xborder AS (
            SELECT fl, mt_id, nd, 0::SMALLINT AS run_id, SUM(value) AS erg,
                'entsoe_cross_border'::VARCHAR AS input
            FROM {sc_out}.analysis_time_series
            WHERE sta_mod = 'stats_imex_entsoe'
            GROUP BY fl, mt_id, nd, run_id
        ), tb_rte_eco2mix AS (
            SELECT fl, mt_id, nd, 0::SMALLINT AS run_id, SUM(value) AS erg,
                'rte_eco2mix'::VARCHAR AS input
            FROM {sc_out}.analysis_time_series
            WHERE sta_mod = 'stats_rte_eco2mix'
            GROUP BY fl, mt_id, nd, run_id
        ), tb_model_tr AS (
            SELECT
                CASE WHEN bool_out = True THEN 'export_' || nd2 ELSE 'import_' || nd2 END AS fl,
                mt_id, nd, run_id,
                SUM((CASE WHEN bool_out = True THEN -1 ELSE +1 END) * value * weight) AS erg,
                'model_tr'::VARCHAR AS input
            FROM {sc_out}.var_tr_trm AS tbtr
            LEFT JOIN (SELECT nd_id, nd FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = tbtr.nd_id
            LEFT JOIN (SELECT nd_id, nd AS nd2 FROM {sc_out}.def_node) AS dfnd2 ON dfnd2.nd_id = tbtr.nd_2_id
            LEFT JOIN (SELECT sy, mt_id, weight FROM {sc_out}.tm_soy) AS dftm ON dftm.sy = tbtr.sy
            GROUP BY fl, mt_id, nd, run_id
        ), tb_model_tr_cap_imp AS (
            SELECT 'import_' || nd2 AS fl, mt_id, nd, run_id, value AS erg, 'model_cap_tr'
            FROM {sc_out}.par_cap_trmi_leg AS tbrv
            LEFT JOIN (SELECT nd_id, nd AS nd2 FROM {sc_out}.def_node) AS dfnd2 ON dfnd2.nd_id = tbrv.nd_id
            LEFT JOIN (SELECT nd_id, nd AS nd FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = tbrv.nd_2_id
            ORDER BY nd, nd2
        ), tb_model_tr_cap_exp AS (
            SELECT 'export_' || nd2 AS fl, mt_id, nd, run_id, value AS erg, 'model_cap_tr'
            FROM {sc_out}.par_cap_trme_leg AS tbrv
            LEFT JOIN (SELECT nd_id, nd AS nd FROM {sc_out}.def_node) AS dfnd2 ON dfnd2.nd_id = tbrv.nd_id
            LEFT JOIN (SELECT nd_id, nd AS nd2 FROM {sc_out}.def_node) AS dfnd ON dfnd.nd_id = tbrv.nd_2_id
            ORDER BY nd, nd2
        ), tb_entsoe_comm AS (
            WITH tb_raw AS (
                SELECT nd_to, nd_from, mt_id, tb.year, SUM(value) AS erg
                FROM profiles_raw.entsoe_commercial_exchange AS tb
                LEFT JOIN (SELECT mt_id, year, datetime
                           FROM profiles_raw.timestamp_template) AS ts
                ON ts.datetime = tb."DateTime"
                GROUP BY nd_to, nd_from, mt_id, tb.year
            ), tb AS (
                SELECT 'export_comm_'::VARCHAR || LOWER(nd_to) AS fl,  nd_from AS nd, *
                FROM tb_raw WHERE nd_from IN (SELECT nd FROM {sc_out}.def_node)
                UNION ALL
                SELECT 'import_comm_'::VARCHAR || LOWER(nd_from) AS fl, nd_to AS nd, *
                FROM tb_raw WHERE nd_to IN (SELECT nd FROM {sc_out}.def_node)
            )
            SELECT fl, mt_id, nd, 0::SMALLINT AS run_id,
                erg, 'entsoe_commercial_exchange'::VARCHAR AS input
            FROM tb
        ), tb_monthly AS (
            SELECT fl, mt_id, nd, 0::SMALLINT AS run_id, erg, input
            FROM profiles_raw.monthly_production
            WHERE year = 2015
        ), tb_all AS (
        SELECT * FROM tb_agora_month_sum
        UNION ALL
        SELECT * FROM tb_rte_month_sum
        UNION ALL
        SELECT * FROM tb_model_var_sy_pwr
        UNION ALL
        SELECT * FROM tb_entsoe_xborder
        UNION ALL
        SELECT * FROM tb_model_tr
        UNION ALL
        SELECT * FROM tb_model_tr_cap_imp
        UNION ALL
        SELECT * FROM tb_model_tr_cap_exp
        UNION ALL
        SELECT * FROM tb_monthly
        UNION ALL
        SELECT * FROM tb_entsoe_comm
        UNION ALL
        SELECT * FROM tb_rte_eco2mix
        )
        INSERT INTO {sc_out}.analysis_monthly_comparison (fl, mt_id, nd, run_id,
                                                          erg, input, input_simple)
        SELECT fl, mt_id, nd, run_id, erg, input, input_simple
        FROM tb_all
        NATURAL LEFT JOIN map_input;

/*
        /* ADD CAPACITY FACTORS CROSS-BORDER TRANSMISSION */
        INSERT INTO {sc_out}.analysis_monthly_comparison (
                        fl, mt_id, nd, run_id, erg, input, input_simple)
        SELECT
            dfnd2.fl, tbrv.mt_id, dfnd.nd, tbrv.run_id,
            erg / (value * month_weight) AS erg,
            'model_tr_cf'::VARCHAR AS input, 'model'::VARCHAR AS input_simple
        FROM {sc_out}.par_cap_trm_leg AS tbrv
        LEFT JOIN (SELECT nd_id, nd AS nd FROM {sc_out}.def_node) AS dfnd
            ON dfnd.nd_id = tbrv.nd_2_id
        LEFT JOIN (SELECT nd_id, 'import_' || nd AS fl
                   FROM {sc_out}.def_node) AS dfnd2
            ON dfnd2.nd_id = tbrv.nd_id
        LEFT JOIN (SELECT mt_id, month_weight FROM {sc_out}.def_month) AS dfmt
            ON dfmt.mt_id = tbrv.mt_id
        LEFT JOIN (SELECT fl, mt_id, nd, run_id, erg
                   FROM {sc_out}.analysis_monthly_comparison
                   WHERE input = 'model_tr' AND fl LIKE 'import_%') AS tban
            ON tban.mt_id = tbrv.mt_id AND tban.run_id = tbrv.run_id
                AND tban.fl = dfnd2.fl AND tban.nd = dfnd.nd;

        INSERT INTO {sc_out}.analysis_monthly_comparison (
                        fl, mt_id, nd, run_id, erg, input, input_simple)
        SELECT
            dfnd2.fl, tbrv.mt_id, dfnd.nd, tbrv.run_id,
            erg / (value * month_weight) AS erg,
            'model_tr_cf'::VARCHAR AS input, 'model'::VARCHAR AS input_simple
        FROM {sc_out}.par_cap_trm_leg AS tbrv
        LEFT JOIN (SELECT nd_id, nd AS nd FROM {sc_out}.def_node) AS dfnd
            ON dfnd.nd_id = tbrv.nd_id
        LEFT JOIN (SELECT nd_id, 'export_' || nd AS fl
                   FROM {sc_out}.def_node) AS dfnd2
            ON dfnd2.nd_id = tbrv.nd_2_id
        LEFT JOIN (SELECT mt_id, month_weight FROM {sc_out}.def_month) AS dfmt
            ON dfmt.mt_id = tbrv.mt_id
        LEFT JOIN (SELECT fl, mt_id, nd, run_id, erg
                   FROM {sc_out}.analysis_monthly_comparison
                   WHERE input = 'model_tr' AND fl LIKE 'export_%') AS tban
            ON tban.mt_id = tbrv.mt_id AND tban.run_id = tbrv.run_id
                AND tban.fl = dfnd2.fl AND tban.nd = dfnd.nd;

*/

        ALTER TABLE {sc_out}.analysis_monthly_comparison
        ADD COLUMN fl2 VARCHAR;

        UPDATE {sc_out}.analysis_monthly_comparison
        SET fl2 = fl;
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)



    def analysis_production_comparison_hourly(self, stats_years=['2015'],
                                              sy_only=False):

        # generating model data time series
        self.generate_analysis_time_series(False)

        # rename original table to _soy and expand to hours
        exec_strg = '''
                    DROP TABLE IF EXISTS {sc_out}.analysis_time_series_soy
                    CASCADE;

                    ALTER TABLE {sc_out}.analysis_time_series
                    RENAME TO analysis_time_series_soy;
                    '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)

        if not sy_only:

            exec_strg = '''
                        SELECT
                        run_id, bool_out, fl, nd, {sw_year_col}, hy AS sy, value,
                        value_posneg,
                        dow, dow_type, hom, hour, how, mt_id,
                        season, wk_id, wom, 'model'::VARCHAR AS sta_mod, pwrerg_cat
                        INTO {sc_out}.analysis_time_series
                        FROM {sc_out}.hoy_soy AS hs
                        LEFT JOIN {sc_out}.analysis_time_series_soy AS ts
                        ON hs.sy = ts.sy;
                        '''.format(**self.format_kw)
            aql.exec_sql(exec_strg, db=self.db)

            # insert rows of entsoe data after adding some convenience columns
            exec_strg = '''
                        ALTER TABLE {sc_out}.analysis_time_series
                        DROP CONSTRAINT IF EXISTS analysis_time_series_fl_fkey;

                        DELETE FROM {sc_out}.analysis_time_series
                        WHERE sta_mod <> 'model';

                        INSERT INTO
                            {sc_out}.analysis_time_series(run_id, bool_out, fl, nd,
                                                         {sw_year_col}, sy, value,
                                                         value_posneg, dow,
                                                         dow_type, hom, hour,
                                                         how, mt_id, season, wk_id,
                                                         wom, sta_mod, pwrerg_cat)
                        SELECT -1::SMALLINT AS run_id, False::BOOLEAN AS bool_out, --'EL' AS ca,
                            fl_id AS fl, nd_id AS nd,
                            'yr' || tm.year::VARCHAR AS {sw_year_col},
                            tm.hy AS sy, value, value AS value_posneg,
                            dow, dow_type, hom, hour, how, mt_id,
                            season, wk_id, wom, 'stats_entsoe'::VARCHAR AS sta_mod,
                            'pwr'::VARCHAR AS pwrerg_cat
                        FROM profiles_raw.entsoe_generation AS ent
                        LEFT JOIN {sc_out}.tm_soy_full AS tm ON tm.sy = ent.hy
                        WHERE ent.year IN ({st_yr})
                            AND ent.nd_id IN {in_nd};

                        INSERT INTO
                            {sc_out}.analysis_time_series(run_id, bool_out, fl, nd,
                                                         {sw_year_col}, sy, value,
                                                         value_posneg, dow,
                                                         dow_type, hom, hour,
                                                         how, mt_id, season, wk_id,
                                                         wom, sta_mod, pwrerg_cat)
                        SELECT -1::SMALLINT AS run_id, False::BOOLEAN AS bool_out, --'EL' AS ca,
                            fl_id AS fl, nd_id AS nd,
                            'yr' || tm.year::VARCHAR AS {sw_year_col},
                            tm.hy AS sy, value, value AS value_posneg,
                            dow, dow_type, hom, hour, how, ent.mt_id,
                            season, wk_id, wom, 'stats_rte_eco2mix'::VARCHAR AS sta_mod,
                            'pwr'::VARCHAR AS pwrerg_cat
                        FROM profiles_raw.rte_production_eco2mix AS ent
                        LEFT JOIN {sc_out}.tm_soy_full AS tm ON tm.sy = ent.hy
                        WHERE ent.year IN ({st_yr});

                        INSERT INTO
                            {sc_out}.analysis_time_series(run_id, bool_out, fl, nd,
                                                         {sw_year_col}, sy, value,
                                                         value_posneg, dow,
                                                         dow_type, hom, hour,
                                                         how, mt_id, season, wk_id,
                                                         wom, sta_mod, pwrerg_cat)
                        SELECT -1::SMALLINT AS run_id,
                            False::BOOLEAN AS bool_out, --'EL' AS ca,
                            fl_id AS fl, nd_id AS nd,
                            'yr' || tb.year::VARCHAR AS {sw_year_col},
                            ts.slot AS sy,
                            CASE WHEN fl_id = 'dmnd' THEN -1 ELSE 1 END * 1000 * value,
                            CASE WHEN fl_id = 'dmnd' THEN -1 ELSE 1 END * 1000 * value AS value_posneg,
                            dow, dow_type, hom, hour, how, mt_id,
                            season, wk_id, wom, 'stats_agora'::VARCHAR AS sta_mod,
                            'pwr'::VARCHAR AS pwrerg_cat
                        FROM profiles_raw.agora_profiles AS tb
                        LEFT JOIN (SELECT datetime, slot
                                   FROM profiles_raw.timestamp_template
                                   WHERE year IN ({st_yr})) AS ts
                        ON ts.datetime = tb."DateTime"
                        LEFT JOIN {sc_out}.tm_soy_full AS tm ON tm.sy = ts.slot
                        WHERE tb.year IN ({st_yr});

                        WITH tb_raw AS (
                            SELECT nd_to AS nd, 'import_' || nd_from AS fl,
                                    False::BOOLEAN AS bool_out, value, year, hy
                            FROM profiles_raw.entsoe_cross_border
                            UNION ALL
                            SELECT nd_from AS nd, 'export_' || nd_to AS fl,
                                True::BOOLEAN AS bool_out, -value AS value,
                                year, hy
                            FROM profiles_raw.entsoe_cross_border
                        )
                        INSERT INTO
                            {sc_out}.analysis_time_series(run_id, bool_out, fl, nd,
                                                         {sw_year_col}, sy, value,
                                                         value_posneg, dow,
                                                         dow_type, hom, hour,
                                                         how, mt_id, season, wk_id,
                                                         wom, sta_mod, pwrerg_cat)
                        SELECT
                        -1::SMALLINT AS run_is, bool_out, fl, nd,
                                        'yr2015'::VARCHAR AS {sw_year_col}, tb_raw.hy AS sy,
                                        value, value AS value_posneg,
                                        dow, 'NONE'::VARCHAR AS dow_type, hom, hour, how, mt_id, season,
                                        EXTRACT(week FROM datetime)::SMALLINT - 1 AS wk_id,
                                        wom, 'stats_imex_entsoe'::VARCHAR AS sta_mod,
                                        'pwr'::VARCHAR AS pwrerg_cat
                        FROM (SELECT * FROM profiles_raw.timestamp_template WHERE year = 2015) AS ts
                        LEFT  JOIN tb_raw ON tb_raw.year = ts.year AND tb_raw.hy = ts.slot
                        '''.format(**self.format_kw, st_yr=', '.join(stats_years))
            aql.exec_sql(exec_strg, db=self.db)

            aql.joinon(self.db, self.sw_columns, ['run_id'],
                       [self.sc_out, 'analysis_time_series'],
                       [self.sc_out, 'def_run'])

            for col in self.sw_columns:
                exec_strg = '''
                            UPDATE {sc_out}.analysis_time_series
                            SET {col} = 'none'
                            WHERE {col} IS NULL;
                            '''.format(**self.format_kw, col=col)
                aql.exec_sql(exec_strg, db=self.db)


    def analysis_production_comparison(self):
        '''
        This merges the model results from sc_out.var_yr_erg_yr with the
        input data erg_inp from sc_out.fuel_node_encar as well as the
        inter-node transmission from lp_input.imex_comp with for
        comparison and calibration.
        TODO: Copy imex_comp to the output schema!!!
        '''

        # get model results
        exec_str = '''
                    DROP TABLE IF EXISTS {sc_out}.analysis_production_comparison CASCADE;
                    SELECT fl_id, nd_id, ca_id, value, bool_out,
                        'model'::VARCHAR AS sta_mod, run_id
                    INTO {sc_out}.analysis_production_comparison
                    FROM {sc_out}.var_yr_erg_yr AS erg
                    NATURAL LEFT JOIN {sc_out}.def_run AS dflp
                    NATURAL LEFT JOIN {sc_out}.def_plant AS dfpp
                    WHERE run_id IN (0, -1)
                    ;
                    '''.format(**self.format_kw)

        aql.exec_sql(exec_str, db=self.db)

        # add stats
        df_erg_inp = aql.read_sql(self.db, self.sc_out, 'fuel_node_encar').set_index(['fl_id', 'nd_id', 'ca_id'])
        df_erg_inp = df_erg_inp[[c for c in df_erg_inp.columns if 'erg_inp' in c]]
        df_erg_inp = df_erg_inp.stack().reset_index().rename(columns={'level_3': 'swhy_vl', 0: 'value'})
        df_erg_inp['swhy_vl'] = df_erg_inp['swhy_vl'].replace({'erg_inp': 'erg_inp_yr2015'}).map(lambda x: x[-6:])
        df_erg_inp['sta_mod'] = 'stats'
        df_erg_inp['run_id'] = -1
        df_erg_inp['bool_out'] = False


        # add imex stats
#        if 'export' in self.mps.dict_fl_id.keys():
#
#            df_imex = aql.read_sql(self.db, self.sc_out, 'imex_comp')
#
#            df_imex = df_imex.set_index(['nd_id', 'nd_2_id']).stack().reset_index()
#            df_imex = df_imex.rename(columns={'level_2': 'swhy_vl', 0: 'value'})
#
#            df_imex['value'] *= 1000
#
#            df_imex['swhy_vl'] = df_imex['swhy_vl'].replace({'erg_trm': 'erg_trm_yr2015'}).map(lambda x: x[-6:])
#            df_imex['sta_mod'] = 'stats'
#            df_imex['ca_id'] = self.mps.dict_ca_id['EL']
#            df_imex = df_imex.join(df_swhy, on=df_swhy.index.names)
#            df_imex = df_imex.loc[-df_imex.run_id.apply(np.isnan)].drop('swhy_vl', axis=1)
#
#            df_imex_exp = df_imex.groupby(['nd_id', 'sta_mod', 'ca_id', 'run_id'])['value'].sum().reset_index()
#            df_imex_exp['fl_id'] = self.mps.dict_fl_id['export']
#            df_imex_exp['bool_out'] = True
#            df_imex_imp = df_imex.groupby(['nd_2_id', 'sta_mod', 'ca_id', 'run_id'])['value'].sum().reset_index()
#            df_imex_imp['fl_id'] = self.mps.dict_fl_id['import']
#            df_imex_imp['bool_out'] = False
#            df_imex_imp = df_imex_imp.rename(columns={'nd_2_id': 'nd_id'})
#        else:
#            df_imex_imp = pd.DataFrame()
#            df_imex_exp = pd.DataFrame()

        df_erg_inp = pd.concat([df_erg_inp])

        aql.write_sql(df_erg_inp, self.db, self.sc_out,
                      'analysis_production_comparison', 'append')


        aql.joinon(self.db, self.sw_columns, ['run_id'],
                   [self.sc_out, 'analysis_production_comparison'],
                   [self.sc_out, 'def_run'])
        aql.joinon(self.db, ['fl'], ['fl_id'],
                   [self.sc_out, 'analysis_production_comparison'],
                   [self.sc_out, 'def_fuel'])
        aql.joinon(self.db, ['nd'], ['nd_id'],
                   [self.sc_out, 'analysis_production_comparison'],
                   [self.sc_out, 'def_node'])

    def analysis_cf_comparison(self):

        exec_str = '''
                   DROP TABLE IF EXISTS {sc_out}.analysis_cf_comparison CASCADE;

                   SELECT mt_id, pp_id, ca_id, run_id,
                       'var'::VARCHAR AS var_par,
                       value / NULLIF(cap * 8760, 0) AS cf
                   INTO {sc_out}.analysis_cf_comparison
                   FROM {sc_out}.var_mt_erg_mt
                   NATURAL LEFT JOIN (
                       SELECT pp_id, ca_id, run_id, value AS cap
                       FROM {sc_out}.par_cap_pwr_leg) AS cap
                   UNION ALL
                   SELECT mt_id, pp_id, ca_id, run_id,
                       'par'::VARCHAR AS var_par, value AS cf
                   FROM {sc_out}.par_cf_max
                   '''.format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)

        aql.joinon(self.db, self.sw_columns, ['run_id'],
                   [self.sc_out, 'analysis_cf_comparison'],
                   [self.sc_out, 'def_run'])
        aql.joinon(self.db, ['pp', 'nd_id', 'pt_id', 'fl_id'], ['pp_id'],
                   [self.sc_out, 'analysis_cf_comparison'],
                   [self.sc_out, 'def_plant'])
        aql.joinon(self.db, ['nd'], ['nd_id'],
                   [self.sc_out, 'analysis_cf_comparison'],
                   [self.sc_out, 'def_node'])
        aql.joinon(self.db, ['pt'], ['pt_id'],
                   [self.sc_out, 'analysis_cf_comparison'],
                   [self.sc_out, 'def_pp_type'])
        aql.joinon(self.db, ['fl'], ['fl_id'],
                   [self.sc_out, 'analysis_cf_comparison'],
                   [self.sc_out, 'def_fuel'])


    def analysis_price_comparison(self, valmin=-20, valmax=150, nbins=170):
        '''
        Merge the supply constraint shadow prices with the historic electricity
        prices and bin them.
        Output tables are
         -- analysis_price_comparison: Complete hourly model and historic
                 electricity prices with corresponding bins.
        '''


        self.format_kw.update(dict(valmin=valmin, valmax=valmax, nbins=nbins))

        exec_str = '''
                   DROP TABLE IF EXISTS
                       {sc_out}.analysis_price_comparison CASCADE;
                   DROP VIEW IF EXISTS
                       {sc_out}._view_analysis_prices_complete CASCADE;
                   DROP VIEW IF EXISTS
                       {sc_out}._view_analysis_prices_stats_0 CASCADE;
                   DROP VIEW IF EXISTS
                       {sc_out}._view_analysis_prices_stats_at CASCADE;



                   CREATE VIEW {sc_out}._view_analysis_prices_stats_0 AS
                   SELECT hy, nd_id, ca_id,
                     price_eur_mwh AS price,
                     volume_mwh AS volume,
                     price_eur_mwh * volume_mwh AS price_volume,
                     -1 AS run_id, 'stats' AS sta_mod
                   FROM {sc_out}.profprice_comp
                   NATURAL LEFT JOIN {sc_out}.def_run
                   WHERE swhy_vl = 'yr2015';

                   /* Double Germany for Austria */
                   CREATE VIEW {sc_out}._view_analysis_prices_stats_at AS
                   SELECT hy,
                       (SELECT nd_id
                        FROM {sc_out}.def_node
                        WHERE nd = 'AT0') AS nd_id,
                        ca_id, price, volume, price_volume, run_id, sta_mod
                   FROM  {sc_out}._view_analysis_prices_stats_0
                   WHERE nd_id IN (SELECT nd_id
                                   FROM {sc_out}.def_node
                                   WHERE nd = 'DE0');

                   CREATE VIEW {sc_out}._view_analysis_prices_complete AS
                   SELECT hy, nd_id, ca_id,
                       value / weight AS price,
                       volume * weight AS volume,
                       value * volume AS price_volume,
                       run_id, 'model' AS sta_mod
                   FROM {sc_out}.hoy_soy
                   NATURAL LEFT JOIN {sc_out}.dual_supply
                   NATURAL LEFT JOIN {sc_out}.tm_soy
                   NATURAL LEFT JOIN (
                       SELECT sy, nd_id, run_id, SUM(value) AS volume
                       FROM {sc_out}.var_sy_pwr
                       NATURAL LEFT JOIN {sc_out}.def_plant
                       WHERE bool_out=False
                       GROUP BY sy, nd_id, run_id) AS tb_volume

                   UNION ALL

                   SELECT * FROM {sc_out}._view_analysis_prices_stats_0

                   UNION ALL

                   SELECT * FROM {sc_out}._view_analysis_prices_stats_at
                   ;

                   WITH table_complete_binned AS (
                       SELECT tb_complete.*,
                       WIDTH_BUCKET(tb_complete.price, {valmin},
                                        {valmax}, {nbins}) AS bucket
                       FROM {sc_out}._view_analysis_prices_complete
                           AS tb_complete
                   ), bucket_list AS (
                   SELECT
                       {valmin} + (bucket - 1)
                                   * ({valmax} - {valmin})::FLOAT
                                   / {nbins}::FLOAT AS low,
                       {valmin} + (bucket)
                                   * ({valmax} - {valmin})::FLOAT
                                   / {nbins}::FLOAT AS high,
                       ({valmax} - {valmin}) / {nbins} AS bin_width, bucket
                   FROM (SELECT generate_series(0, {nbins} + 1)
                         AS bucket, 1 AS dummy) AS bcks
                   )
                   SELECT *,
                   0.5 * (low + high) AS bin_center,
                   CASE WHEN price < low THEN 1 ELSE 0 END AS check_low,
                   CASE WHEN price > high THEN 1 ELSE 0 END AS check_high
                   INTO {sc_out}.analysis_price_comparison
                   FROM table_complete_binned
                   NATURAL LEFT JOIN bucket_list
                   '''.format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)

        aql.joinon(self.db, self.sw_columns, ['run_id'],
                   [self.sc_out, 'analysis_price_comparison'],
                   [self.sc_out, 'def_run'])
        aql.joinon(self.db, ['nd'], ['nd_id'],
                   [self.sc_out, 'analysis_price_comparison'],
                   [self.sc_out, 'def_node'])
        aql.joinon(self.db, ['ca', 'fl_id'], ['ca_id'],
                   [self.sc_out, 'analysis_price_comparison'],
                   [self.sc_out, 'def_encar'])
        aql.joinon(self.db, ['fl'], ['fl_id'],
                   [self.sc_out, 'analysis_price_comparison'],
                   [self.sc_out, 'def_fuel'])
        exec_strg = '''
                     ALTER TABLE {sc_out}.analysis_price_comparison
                     ADD COLUMN IF NOT EXISTS season VARCHAR(20),
                     ADD COLUMN IF NOT EXISTS dow_name VARCHAR(10),
                     ADD COLUMN IF NOT EXISTS hour SMALLINT,
                     ADD COLUMN IF NOT EXISTS wom SMALLINT,
                     ADD COLUMN IF NOT EXISTS hom SMALLINT,
                     ADD COLUMN IF NOT EXISTS mt_id SMALLINT,
                     ADD COLUMN IF NOT EXISTS how SMALLINT;

                     UPDATE {sc_out}.analysis_price_comparison AS prc
                     SET season = tm.season,
                         dow_name = tm.dow_name,
                         hour = tm.hour,
                         wom = tm.wom,
                         hom = tm.hom,
                         mt_id = tm.mt_id,
                         how = tm.how
                     FROM {sc_out}.tm_soy_full AS tm
                     WHERE prc.hy = tm.sy;
                     '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)

        exec_str = '''
                   DROP TABLE IF EXISTS {sc_out}.analysis_price_weighted;

                   SELECT nd_id, ca_id, run_id, sta_mod,
                       SUM(price * volume) / SUM(volume) AS price_weighted,
                       AVG(price) AS price_averaged
                   INTO {sc_out}.analysis_price_weighted
                   FROM {sc_out}._view_analysis_prices_complete
                   GROUP BY nd_id, ca_id, run_id, sta_mod;
                   '''.format(**self.format_kw)
        aql.exec_sql(exec_str, db=self.db)

        aql.joinon(self.db, self.sw_columns, ['run_id'],
                   [self.sc_out, 'analysis_price_weighted'],
                   [self.sc_out, 'def_run'])
        aql.joinon(self.db, ['nd'], ['nd_id'],
                   [self.sc_out, 'analysis_price_weighted'],
                   [self.sc_out, 'def_node'])
