# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:28:17 2017

@author: m-c-soini


"""

import grimsel.auxiliary.sqlutils.aux_sql_func as aql

from grimsel.analysis.decorators import DecoratorsSqlAnalysis

class SqlAnalysisHourly():
    ''' Performs various SQL-based analyses on the output tables. '''


    @DecoratorsSqlAnalysis.append_sw_columns('analysis_time_series_dual_supply')
    @DecoratorsSqlAnalysis.append_nd_id_columns('analysis_time_series_dual_supply')
    def generate_complete_dual_supply(self):
        ''' Adds loop and sy columns to the dual_supply table.'''
        print(self.in_run_id)


        tb_name = 'analysis_time_series_dual_supply'
        aql.init_table(tb_name, (['sy', 'nd_id', 'ca_id', 'run_id',
                                  'value', 'nd', *self.tm_cols] +
                                 [(c, 'VARCHAR') for c in self.sw_columns]
                                  + [('mc', 'DECIMAL')]),
                                 schema=self.sc_out,
                                 ref_schema=self.sc_out,
                                 pk=['sy', 'nd_id', 'run_id'], db=self.db)

        exec_str_0 = ('''
                      INSERT INTO {sc_out}.{tb_name} (sy, nd_id, ca_id, value, run_id)
                      SELECT *
                      FROM {sc_out}.dual_supply
                      WHERE run_id in {in_run_id}
                      ''').format(tb_name=tb_name, sc_out=self.sc_out, in_run_id=self.in_run_id)
        aql.exec_sql(exec_str_0, db=self.db)

        # add timemap indices
        aql.joinon(self.db, [c for c in self.tm_cols if not c == 'sy'], ['sy'],
                   [self.sc_out, 'analysis_time_series_dual_supply'],
                   [self.sc_out, 'tm_soy_full'],  new_columns=False)

        exec_str_1 = ('''
                      UPDATE {sc_out}.analysis_time_series_dual_supply
                      SET mc = value / weight;
                      '''.format(sc_out=self.sc_out))
        aql.exec_sql(exec_str_1, db=self.db)

        return(exec_str_0 + exec_str_1)


    @DecoratorsSqlAnalysis.append_nd_id_columns('analysis_time_series')
    @DecoratorsSqlAnalysis.append_fl_id_columns('analysis_time_series')
    @DecoratorsSqlAnalysis.append_pp_id_columns('analysis_time_series')
    @DecoratorsSqlAnalysis.append_sw_columns('analysis_time_series')
    def generate_analysis_time_series(self, energy_only=False):
        ''' Generates basic (full time resolution) x (pp_type) x (run_id) table. '''
        print(self.in_run_id)

        tb_name = 'analysis_time_series'

        self.generate_view_time_series_subset()


        aql.init_table(tb_name, (['sy', 'ca_id', 'pp_id', 'bool_out', 'run_id',
                                 'pwrerg_cat', 'nd_id', 'fl_id', 'pt_id',
                                 * self.tm_cols,
                                 ('value', 'DECIMAL'), 'fl', 'nd', 'pt',
                                 ('value_posneg', 'DECIMAL')]
                                 + [(c, 'VARCHAR')
                                    for c in self.sw_columns]),
                                 schema=self.sc_out,
                                 ref_schema=self.sc_out, db=self.db)

        if not energy_only:
            print('Inserting power...')
            exec_str = ('''
                        INSERT INTO {sc_out}.{tb_name}
                            (sy, ca_id, pp_id, bool_out, value, run_id, pwrerg_cat, value_posneg)
                        SELECT * FROM {sc_out}.analysis_time_series_view_power
                        WHERE run_id IN {in_run_id};
                        ''').format(tb_name=tb_name, **self.format_kw)
            aql.exec_sql(exec_str, db=self.db)


            print('Inserting cross-sector consumption...')
            exec_str = ('''
                        INSERT INTO {sc_out}.{tb_name}
                            (sy, ca_id, pp_id, bool_out, value, run_id, pwrerg_cat, value_posneg)
                        SELECT * FROM {sc_out}.analysis_time_series_view_crosssector
                        WHERE run_id IN {in_run_id};
                        ''').format(tb_name=tb_name, **self.format_kw)
            aql.exec_sql(exec_str, db=self.db)


        if 'var_sy_erg_st' in aql.get_sql_tables(self.sc_out, self.db):
            print('Inserting energy...')
            exec_str = ('''
                        INSERT INTO {sc_out}.{tb_name}
                            (sy, ca_id, pp_id, bool_out, value, run_id, pwrerg_cat, value_posneg)
                        SELECT * FROM {sc_out}.analysis_time_series_view_energy
                        WHERE run_id IN {in_run_id};

                        ''').format(tb_name=tb_name, **self.format_kw)
            aql.exec_sql(exec_str, db=self.db)

        # add timemap indices
        aql.joinon(self.db, [c for c in self.tm_cols if (not c == 'sy')], ['sy'],
                   [self.sc_out, tb_name], [self.sc_out, 'tm_soy_full'], new_columns=False)

        return exec_str


    def chgdch_filtered_difference(self, use_views_only):
        '''
        This query applies a binary filter charging for charging/discharging
        and subtracts the zero-storage case for each hour.
        '''

        list_sw_not_st = [c for c in self.sw_columns if not c == 'swst_vl']
        sw_not_st = ', '.join(list_sw_not_st) + (', ' if len(list_sw_not_st) > 0 else '')

        join_ref = ' AND ' + '\nAND '.join(['tbrel.{} = tbrf.{}'.format(c, c)
                                           for c in list_sw_not_st])


        slct_pp_id_st = aql.read_sql(self.db, self.sc_out, 'def_plant',
                               filt=[('pp', ['%STO%'], ' LIKE '),
                                     ('nd_id', self._nd_id)])['pp_id']
        lst_pp_id_st = self.list_to_str(slct_pp_id_st)

        slct_pp_id_non_st = [pp for pp in self.slct_pp_id
                             if not pp in slct_pp_id_st.tolist()]
        lst_pp_id_non_st = self.list_to_str(slct_pp_id_non_st)

        run_id_zero_st = self.list_to_str(aql.read_sql(self.db, self.sc_out, 'def_loop',
                                      filt=[('swst', [0]),
                                            ('run_id', self.slct_run_id)])['run_id'])
        run_id_nonzero_st = self.list_to_str(aql.read_sql(self.db, self.sc_out, 'def_loop',
                                        filt=[('swst', [0], ' <> '),
                                            ('run_id', self.slct_run_id)])['run_id'])


        self.format_kw.update({'sw_not_st': sw_not_st,
                               'join_ref': join_ref,
                               'lst_pp_id_st': lst_pp_id_st,
                               'lst_pp_id_non_st': lst_pp_id_non_st,
                               'run_id_zero_st': run_id_zero_st,
                               'run_id_nonzero_st': run_id_nonzero_st})


        if use_views_only:
            self.format_kw['out_tb'] = 'analysis_agg_filtdiff_views' + self._suffix
        else:
            self.format_kw['out_tb'] = 'analysis_agg_filtdiff_tables' + self._suffix



        if not use_views_only:


            aql.init_table('temp_analysis_time_series_subset',
                           ['sy', 'ca_id', 'pp_id', 'bool_out', 'value', 'run_id',
                            ('value_posneg', 'DECIMAL'), ('value_diff', 'DECIMAL'),
    #                        ('value_ref', 'DECIMAL')
                            ],
                           ref_schema=self.sc_out)
            exec_str = '''
                        /* FILTER TIME SERIES TABLE */
                        INSERT INTO temp_analysis_time_series_subset
                            (sy, ca_id, pp_id, bool_out, value, run_id, value_posneg)
                        SELECT sy, ca_id, ts.pp_id, bool_out, value, ts.run_id, value_posneg
                        FROM {sc_out}.analysis_time_series_view_power AS ts
                        LEFT JOIN {sc_out}.def_plant AS dfpp ON dfpp.pp_id = ts.pp_id
                        LEFT JOIN {sc_out}.def_loop AS dflp ON dflp.run_id = ts.run_id
                        LEFT JOIN {sc_out}.def_pp_type AS dfpt ON dfpt.pt_id = dfpp.pt_id
                        WHERE (ts.pp_id NOT IN {lst_pp_id_st} or pp LIKE '%HYD_STO'
                              OR pt LIKE swtc_vl||'_STO')
                              AND dfpp.pp_id IN {list_slct_pp_id};
                        '''.format(**self.format_kw)
    #        print(exec_str)
            aql.exec_sql(exec_str, time_msg='Filter var_sy_pwr and write to table')


            aql.init_table('temp_tbrf', ['sy', 'pp_id', 'bool_out', 'run_id', ('value_ref', 'DECIMAL')],
                           ref_schema=self.sc_out)
            exec_str = '''
                        /* CREATE TABLE WITH NO-STORAGE REFERENCE VALUES */
                        INSERT INTO temp_tbrf (sy, pp_id, bool_out, run_id, value_ref)
                        SELECT sy, pp_id, bool_out, run_id, value AS value_ref
                        FROM temp_analysis_time_series_subset
                        WHERE run_id IN {run_id_zero_st};
                        '''.format(**self.format_kw)
            aql.exec_sql(exec_str, time_msg='Write data to ref table')
            aql.exec_sql('''ALTER TABLE temp_tbrf
                            ADD PRIMARY KEY(sy, pp_id, bool_out, run_id)''',
                         time_msg='Add pk to temp_tbrf')

            aql.init_table('temp_map_run_id_ref',
                           ['run_id',
                            ('run_id_ref', 'SMALLINT',
                             self.sc_out + '.def_loop(run_id)')],
                           ref_schema=self.sc_out, pk=['run_id'])
            exec_str = '''
                        /* CREATE MAP BETWEEN run_ids and reference run_ids */
                        INSERT INTO temp_map_run_id_ref (run_id, run_id_ref)
                        WITH ref_st AS (
                            SELECT {sw_not_st}run_id AS run_id_rf FROM {sc_out}.def_loop
                            WHERE swst_vl = '0.0%'
                        )
                        SELECT dflp.run_id, ref_st.run_id_rf FROM {sc_out}.def_loop AS dflp
                        NATURAL JOIN ref_st
                        ORDER BY run_id;
                        '''.format(**self.format_kw)
            aql.exec_sql(exec_str, time_msg='Create reference run_id map')


            aql.joinon(['run_id_ref'], ['run_id'],
                       ['public', 'temp_analysis_time_series_subset'],
                       ['public', 'temp_map_run_id_ref'])

            exec_str = '''
                        UPDATE temp_analysis_time_series_subset AS ts
                        SET value_diff = ts.value - rf.value_ref
                          --  ,value_ref = rf.value_ref
                        FROM temp_tbrf AS rf
                        WHERE rf.run_id = ts.run_id_ref AND rf.sy = ts.sy
                            AND rf.pp_id = ts.pp_id
                            AND rf.bool_out = ts.bool_out
                        '''.format(**self.format_kw)
    #        print(exec_str)
            aql.exec_sql(exec_str, time_msg='Add difference column to ts table')


            aql.init_table('temp_tbst',
                           ['run_id', 'sy', ('pp_id_st', 'SMALLINT'),
                            ('bool_out_st', 'BOOLEAN'), ('value_mask', 'DECIMAL')],
                           ref_schema=self.sc_out, pk=['run_id', 'pp_id_st', 'sy', 'bool_out_st'])
            exec_str = '''IF EXISTS
                        /* GET CHARGING/DISCHARGING MASK */
                        INSERT INTO temp_tbst
                            (run_id, sy, pp_id_st, bool_out_st, value_mask)
                        SELECT run_id, sy, pp_id AS pp_id_st,
                            bool_out AS bool_out_st, value AS value_mask
                        FROM temp_analysis_time_series_subset
                        WHERE pp_id IN {lst_pp_id_st}
                        AND run_id IN {run_id_nonzero_st}
                        ORDER BY sy, run_id, pp_id, bool_out;
                        '''.format(**self.format_kw)
            print(exec_str)
            aql.exec_sql(exec_str)




            aql.exec_sql('''ALTER TABLE temp_analysis_time_series_subset
                            ADD PRIMARY KEY(sy, ca_id, pp_id, bool_out, run_id)''',
                         time_msg='Add pk to temp_analysis_time_series_subset')



            # expand temp_analysis_time_series_subset to (bool_out_st = True/False)

            # expand temp_tbst to pp_id, bool_out, best on creation/insertion

            aql.init_table(self.format_kw['out_tb'],
                           ['pp_id', ('pp_id_st', 'SMALLINT'), 'bool_out', ('bool_out_st', 'BOOLEAN'), 'run_id', ('run_id_ref', 'SMALLINT'), ('swst_vl_ref', 'VARCHAR(5)'), 'value'],
                           schema=self.sc_out,
                           ref_schema=self.sc_out, pk=['pp_id', 'pp_id_st',
                                                       'bool_out', 'bool_out_st',
                                                       'run_id'])
            exec_str = '''
                        INSERT INTO {sc_out}.{out_tb}
                            (pp_id, pp_id_st, bool_out, bool_out_st, run_id, run_id_ref, value)
                        WITH tb AS (
                            SELECT * FROM temp_analysis_time_series_subset AS tb
                            CROSS JOIN (SELECT DISTINCT pp_id_st, bool_out_st FROM temp_tbst) AS exp_bool_out_st
                            WHERE run_id IN {run_id_nonzero_st}
                        ), st AS (
                            SELECT * FROM temp_tbst
                            CROSS JOIN (SELECT DISTINCT tb.pp_id, bool_out FROM tb) AS exp_pp
                        )
                        SELECT pp_id, pp_id_st, bool_out, bool_out_st, run_id, run_id_ref,
                        SUM(value_diff) AS value FROM tb
                        NATURAL LEFT JOIN st
                        WHERE value_mask <> 0
                        GROUP BY pp_id, pp_id_st, bool_out, bool_out_st, run_id, run_id_ref;
                        '''.format(**self.format_kw)
            print(exec_str)
            aql.exec_sql(exec_str)
#
#            exec_str = '''
#                        UPDATE {sc_out}.analysis_agg_filtdiff AS tb
#                        SET swst_vl_ref = dflp.swst_vl
#                        FROM {sc_out}.def_loop AS dflp
#                        WHERE dflp.run_id = tb.run_id_ref
#                        '''.format(**format_kw)
#            print(exec_str)
#            aql.exec_sql(exec_str)
#
    #
    #        aql.exec_sql('''
    #                     DROP TABLE IF EXISTS temp_analysis_time_series_subset CASCADE;
    #                     DROP TABLE IF EXISTS temp_tbrf CASCADE;
    #                     DROP TABLE IF EXISTS temp_tbst CASCADE;
    #                     DROP TABLE IF EXISTS temp_map_run_id_ref CASCADE;
    #                     ''')


        else:

            aql.init_table(self.format_kw['out_tb'],
                           ['pp_id',
                            ('pp_id_st', 'SMALLINT'),
                            'bool_out',
                            ('bool_out_st', 'BOOLEAN'),
                            'run_id',
                            ('run_id_ref', 'SMALLINT'),
                            ('swst_vl_ref', 'VARCHAR(5)'),
                            'value'],
                           schema=self.sc_out,
                           ref_schema=self.sc_out, pk=['pp_id', 'pp_id_st',
                                                       'bool_out', 'bool_out_st',
                                                       'run_id'], db=self.db)

            exec_str = '''
                        /* CREATE MAP BETWEEN run_ids and reference run_ids */
                        DROP VIEW IF EXISTS temp_map_run_id_ref CASCADE;
                        CREATE VIEW temp_map_run_id_ref AS
                        WITH ref_st AS (
                            SELECT {sw_not_st}run_id AS run_id_rf FROM {sc_out}.def_loop
                            WHERE swst_vl = '0.00%'
                        )
                        SELECT dflp.run_id, ref_st.run_id_rf FROM {sc_out}.def_loop AS dflp
                        NATURAL JOIN ref_st
                        ORDER BY run_id;

                        /* FILTER TIME SERIES TABLE */
                        DROP TABLE IF EXISTS temp_analysis_time_series_subset CASCADE;
                        SELECT sy, ca_id, nd_id, ts.pp_id, bool_out, value, ts.run_id, value_posneg, run_id_rf
                        INTO temp_analysis_time_series_subset
                        FROM {sc_out}.analysis_time_series_view_power AS ts
                        LEFT JOIN {sc_out}.def_plant AS dfpp ON dfpp.pp_id = ts.pp_id
                        LEFT JOIN {sc_out}.def_loop AS dflp ON dflp.run_id = ts.run_id
                        LEFT JOIN {sc_out}.def_pp_type AS dfpt ON dfpt.pt_id = dfpp.pt_id
                        LEFT JOIN temp_map_run_id_ref AS run_id_ref ON run_id_ref.run_id = ts.run_id
                        WHERE dfpp.pp_id IN {list_slct_pp_id};

                        /* CREATE TABLE WITH NO-STORAGE REFERENCE VALUES */
                        DROP VIEW IF EXISTS temp_tbrf CASCADE;
                        CREATE VIEW temp_tbrf AS
                        SELECT sy, pp_id, bool_out, run_id, value AS value_ref
                        FROM temp_analysis_time_series_subset
                        WHERE run_id IN {run_id_zero_st};

                        /* GET CHARGING/DISCHARGING MASK */
                        DROP VIEW IF EXISTS temp_tbst CASCADE;
                        CREATE VIEW temp_tbst AS
                        SELECT run_id, sy, nd_id AS nd_id_st, pp_id AS pp_id_st,
                            bool_out AS bool_out_st, value AS value_mask
                        FROM temp_analysis_time_series_subset
                        WHERE pp_id IN {lst_pp_id_st}
                        AND run_id IN {run_id_nonzero_st};

                        DROP TABLE IF EXISTS temp_analysis_time_series_subset_ref CASCADE;
                        SELECT ts.*, ts.value - rf.value_ref AS value_diff
                        INTO temp_analysis_time_series_subset_ref
                        FROM temp_analysis_time_series_subset AS ts
                        LEFT JOIN temp_tbrf AS rf
                        ON rf.run_id = ts.run_id_rf AND rf.sy = ts.sy
                            AND rf.pp_id = ts.pp_id
                            AND rf.bool_out = ts.bool_out;

                        DROP VIEW IF EXISTS temp_analysis_time_series_subset_reffilt CASCADE;
                        CREATE VIEW temp_analysis_time_series_subset_reffilt AS
                        WITH tb AS (
                            SELECT * FROM temp_analysis_time_series_subset_ref AS tb
                            FULL OUTER JOIN (SELECT DISTINCT pp_id_st, nd_id_st, bool_out_st FROM temp_tbst) AS exp_bool_out_st
                            ON exp_bool_out_st.nd_id_st = tb.nd_id
                            WHERE run_id IN {run_id_nonzero_st}
                        ), st AS (
                            SELECT * FROM temp_tbst
                            CROSS JOIN (SELECT DISTINCT tb.pp_id, bool_out FROM tb) AS exp_pp
                        ), mt_map AS (
                            SELECT sy, mt_id FROM {sc_out}.tm_soy
                        )
                        SELECT pp_id, pp_id_st, bool_out, bool_out_st, run_id, mt_id, run_id_rf,
--                        SELECT pp_id, pp_id_st, bool_out, bool_out_st, run_id, run_id_rf,
                        SUM(value_diff) AS value FROM tb
                        NATURAL LEFT JOIN st
                        LEFT JOIN mt_map ON mt_map.sy = tb.sy
                        WHERE value_mask <> 0
                        GROUP BY
                            pp_id, pp_id_st, bool_out, bool_out_st,
                            run_id, mt_id, run_id_rf;
--                            run_id, run_id_rf;

                        DROP TABLE IF EXISTS {sc_out}.{out_tb} CASCADE;
                        SELECT *
                        INTO {sc_out}.{out_tb}
                        FROM temp_analysis_time_series_subset_reffilt
                        '''.format(**self.format_kw)
            aql.exec_sql(exec_str, db=self.db)

        aql.exec_sql('''
                     DROP VIEW IF EXISTS temp_map_run_id_ref CASCADE;
                     DROP TABLE IF EXISTS temp_analysis_time_series_subset CASCADE;
                     DROP VIEW IF EXISTS temp_tbrf CASCADE;
                     DROP VIEW IF EXISTS temp_tbst CASCADE;
                     DROP TABLE IF EXISTS temp_analysis_time_series_subset_ref CASCADE;
                     ''', db=self.db)

        aql.exec_sql('''
                     /* ADD PP_TYPE COLUMN FOR STORAGE FILTER PLANTS */
                     ALTER TABLE {sc_out}.{out_tb}
                             DROP COLUMN IF EXISTS pt_st,
                             DROP COLUMN IF EXISTS pp_st,
                             ADD COLUMN pp_st VARCHAR(15),
                             ADD COLUMN pt_st VARCHAR(15);
                     UPDATE {sc_out}.{out_tb} AS tb
                     SET pt_st = map_pt.pt, pp_st = map_pt.pp
                     FROM (
                         SELECT pp_id AS pp_id, pt, pp
                             FROM {sc_out}.def_plant AS dfpp
                         LEFT JOIN {sc_out}.def_pp_type AS dfpt
                        ON dfpp.pt_id = dfpt.pt_id
                     ) AS map_pt
                     WHERE map_pt.pp_id = tb.pp_id_st;
                     '''.format(**self.format_kw), db=self.db)

        # add loop indices
        aql.joinon(self.db, self.sw_columns, ['run_id'],
                   [self.sc_out, self.format_kw['out_tb']],
                   [self.sc_out, 'def_loop'])
        # add pp indices
        aql.joinon(self.db, ['pt_id', 'fl_id', 'nd_id'], ['pp_id'],
                   [self.sc_out, self.format_kw['out_tb']],
                   [self.sc_out, 'def_plant'])
        # add pt indices
        aql.joinon(self.db, ['pt'], ['pt_id'], [self.sc_out, self.format_kw['out_tb']],
                   [self.sc_out, 'def_pp_type'])
        # add fl indices
        aql.joinon(self.db, ['fl'], ['fl_id'], [self.sc_out, self.format_kw['out_tb']],
                   [self.sc_out, 'def_fuel'])
        # add nd indices
        aql.joinon(self.db, ['nd'], ['nd_id'], [self.sc_out, self.format_kw['out_tb']],
                   [self.sc_out, 'def_node'])


    def append_nd_id_columns(f):
        def wrapper(self, *args, **kwargs):
            f(self, *args, **kwargs)
            aql.joinon(self.db, ['nd'], ['nd_id'],
                       [self.sc_out, f.__name__], [self.sc_out, 'def_node'])
        return wrapper


    @DecoratorsSqlAnalysis.append_nd_id_columns('analysis_agg_filtdiff')
    @DecoratorsSqlAnalysis.append_pt_id_columns('analysis_agg_filtdiff')
    @DecoratorsSqlAnalysis.append_fl_id_columns('analysis_agg_filtdiff')
    @DecoratorsSqlAnalysis.append_pp_id_columns('analysis_agg_filtdiff')
    @DecoratorsSqlAnalysis.append_sw_columns('analysis_agg_filtdiff')
    @DecoratorsSqlAnalysis.append_nd_id_columns('analysis_agg_filtdiff_agg')
    @DecoratorsSqlAnalysis.append_pt_id_columns('analysis_agg_filtdiff_agg')
    @DecoratorsSqlAnalysis.append_fl_id_columns('analysis_agg_filtdiff_agg')
    @DecoratorsSqlAnalysis.append_pp_id_columns('analysis_agg_filtdiff_agg')
    @DecoratorsSqlAnalysis.append_sw_columns('analysis_agg_filtdiff_agg')
    def analysis_agg_filtdiff(self):
        '''
        New style.
        '''

        list_sw_not_st = [c for c in self.sw_columns if not c == 'swst_vl']
        sw_not_st = (', '.join(list_sw_not_st)
                     + (', ' if len(list_sw_not_st) > 0 else ''))

        join_ref = ' AND ' + '\nAND '.join(['tbrel.{} = tbrf.{}'.format(c, c)
                                            for c in list_sw_not_st])

        slct_pp_id_st = aql.read_sql(self.db, self.sc_out, 'def_plant',
                               filt=[('pp', ['%STO%'], ' LIKE '),
                                     ('pp_id', self.slct_pp_id),
                                     ('nd_id', self._nd_id)])['pp_id'].tolist()
        lst_pp_id_st = self.list_to_str(slct_pp_id_st)

        slct_pp_id_non_st = [pp for pp in self.slct_pp_id
                             if not pp in slct_pp_id_st]
        lst_pp_id_non_st = self.list_to_str(slct_pp_id_non_st)

        run_id_zero_st = self.list_to_str(aql.read_sql(self.db, self.sc_out, 'def_loop',
                                      filt=[('swst', [0]),
                                            ('run_id', self.slct_run_id)])['run_id'])
        run_id_nonzero_st = self.list_to_str(aql.read_sql(self.db, self.sc_out, 'def_loop',
                                        filt=[('swst', [0], ' <> '),
                                              ('run_id', self.slct_run_id)])['run_id'])


        self.format_kw.update({'sw_not_st': sw_not_st,
                               'join_ref': join_ref,
                               'lst_pp_id_st': lst_pp_id_st,
                               'lst_pp_id_non_st': lst_pp_id_non_st,
                               'run_id_zero_st': run_id_zero_st,
                               'run_id_nonzero_st': run_id_nonzero_st,
                               'out_tb': 'analysis_agg_filtdiff'})


        cols = aql.init_table('analysis_agg_filtdiff',
               [
                ('sy', 'SMALLINT'),
                ('pp_id', 'SMALLINT'),
                ('bool_out', 'BOOLEAN'),
                ('run_id', 'SMALLINT'),
                ('run_id_rf', 'SMALLINT'),
                ('pp_id_st', 'SMALLINT'),
                ('bool_out_st', 'BOOLEAN'),
                ('value_st', 'DOUBLE PRECISION'),
                ('value', 'DOUBLE PRECISION'),
                ('value_ref', 'DOUBLE PRECISION'),
                ('value_diff', 'DOUBLE PRECISION'),
               ],
               schema=self.sc_out,
               ref_schema=self.sc_out,
               pk=['sy', 'pp_id', 'bool_out', 'run_id', 'run_id_rf',
                   'pp_id_st', 'bool_out_st'],
               db=self.db)



        exec_strg = '''
        /* CREATE MAP BETWEEN run_ids and reference run_ids */
        DROP VIEW IF EXISTS temp_map_run_id_ref CASCADE;
        CREATE VIEW temp_map_run_id_ref AS
        WITH ref_st AS (
            SELECT swvr_vl, swtc_vl, swpt_vl, swyr_vl, swco_vl, run_id AS run_id_rf
            FROM {sc_out}.def_loop
            WHERE swst_vl = '0.00%' AND run_id IN {run_id_zero_st}
        )
        SELECT dflp.run_id, ref_st.run_id_rf FROM {sc_out}.def_loop AS dflp
        NATURAL JOIN ref_st
        WHERE run_id IN {run_id_nonzero_st}
        ORDER BY run_id;
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)

        exec_strg = '''
        /* FILTER TIME SERIES TABLE */
        DROP TABLE IF EXISTS temp_analysis_time_series_subset CASCADE;
        SELECT sy, ca_id, nd_id, ts.pp_id, pp, bool_out, value, ts.run_id, value_posneg, run_id_rf
        INTO temp_analysis_time_series_subset
        FROM {sc_out}.analysis_time_series_view_power AS ts
        LEFT JOIN {sc_out}.def_plant AS dfpp ON dfpp.pp_id = ts.pp_id
        LEFT JOIN {sc_out}.def_loop AS dflp ON dflp.run_id = ts.run_id
        LEFT JOIN {sc_out}.def_pp_type AS dfpt ON dfpt.pt_id = dfpp.pt_id
        LEFT JOIN temp_map_run_id_ref AS run_id_ref ON run_id_ref.run_id = ts.run_id
        WHERE dfpp.pp_id IN {list_slct_pp_id} AND ts.run_id IN {in_run_id};
        '''.format(**self.format_kw)
        aql.exec_sql(exec_strg, db=self.db)

        exec_strg = '''
        WITH mask_st AS (
            SELECT sy, pp_id AS pp_id_st, bool_out AS bool_out_st, value AS value_st, run_id
            FROM temp_analysis_time_series_subset
            WHERE pp_id IN {lst_pp_id_st}
                AND run_id IN (SELECT DISTINCT run_id FROM temp_map_run_id_ref
                               WHERE NOT run_id
                               IN (SELECT DISTINCT run_id_rf FROM temp_map_run_id_ref))
        ), tb_pp AS (
            SELECT sy, pp_id, bool_out, value AS value, run_id, run_id_rf
            FROM temp_analysis_time_series_subset
            WHERE pp_id IN {lst_pp_id_non_st}
                AND run_id IN (SELECT DISTINCT run_id FROM temp_map_run_id_ref
                               WHERE NOT run_id
                               IN (SELECT DISTINCT run_id_rf FROM temp_map_run_id_ref))
        ), tb_pp_ref AS (
            SELECT sy, pp_id, bool_out, value AS value_ref, run_id_rf
            FROM temp_analysis_time_series_subset
            WHERE pp_id IN {lst_pp_id_non_st}
                AND run_id IN (SELECT DISTINCT run_id_rf FROM temp_map_run_id_ref)
        )
        INSERT INTO {sc_out}.{out_tb} ({cols})
        SELECT tb_pp.sy, tb_pp.pp_id, tb_pp.bool_out, tb_pp.run_id, tb_pp.run_id_rf,
            pp_id_st, bool_out_st, value_st,
            value, value_ref, value - value_ref AS value_diff
        FROM tb_pp
        LEFT JOIN tb_pp_ref
            ON tb_pp.sy = tb_pp_ref.sy
            AND tb_pp.pp_id = tb_pp_ref.pp_id
            AND tb_pp.bool_out = tb_pp_ref.bool_out
            AND tb_pp.run_id_rf = tb_pp_ref.run_id_rf
        FULL OUTER JOIN mask_st ON mask_st.sy = tb_pp.sy
                                AND mask_st.run_id = tb_pp.run_id
        WHERE ABS(value_st) > 1;
        '''.format(**self.format_kw, cols=cols)
        aql.exec_sql(exec_strg, db=self.db)

        aql.joinon(self.db, ['mt_id', 'season'], ['sy'],
                   [self.sc_out, 'analysis_agg_filtdiff'],
                   [self.sc_out, 'tm_soy_full'])

        # aggregated table
        cols = aql.init_table('analysis_agg_filtdiff_agg',
               [
                ('pp_id', 'SMALLINT'),
                ('bool_out', 'BOOLEAN'),
                ('run_id', 'SMALLINT'),
                ('run_id_rf', 'SMALLINT'),
                ('pp_id_st', 'SMALLINT'),
                ('bool_out_st', 'BOOLEAN'),
                ('value_diff_agg', 'DOUBLE PRECISION'),
               ],
               schema=self.sc_out,
               ref_schema=self.sc_out,
               pk=['pp_id', 'bool_out', 'run_id', 'run_id_rf',
                   'pp_id_st', 'bool_out_st'],
               db=self.db)

        exec_strg = '''
        INSERT INTO {sc_out}.analysis_agg_filtdiff_agg ({cols})
        SELECT pp_id, bool_out, run_id, run_id_rf, pp_id_st, bool_out_st,
        SUM(value_diff) AS value_diff_agg
        FROM {sc_out}.analysis_agg_filtdiff
        GROUP BY pp_id, bool_out, run_id, run_id_rf, pp_id_st, bool_out_st
        '''.format(**self.format_kw, cols=cols)
        aql.exec_sql(exec_strg, db=self.db)

        for tb in ['analysis_agg_filtdiff_agg',
                   'analysis_agg_filtdiff']:

            aql.joinon(self.db,
                       {'pp': 'pp_st', 'pt_id': 'pt_id_st'},
                       {'pp_id': 'pp_id_st'},
                       [self.sc_out, tb], [self.sc_out, 'def_plant'])

            aql.joinon(self.db,
                       {'pt': 'pt_st'},
                       {'pt_id': 'pt_id_st'},
                       [self.sc_out, tb], [self.sc_out, 'def_pp_type'])


    def generate_analysis_time_series_diff(self):

        exec_str = ('''
                    DROP TABLE IF EXISTS analysis_time_series_diff CASCADE;
                    WITH tb_1 AS (
                        SELECT * FROM analysis_time_series
                        WHERE run_id = {run_id_1}
                    ), tb_2 AS (
                        SELECT
                            bool_out, ca_id, pp_id, sy,
                            value_posneg AS value_other
                        FROM analysis_time_series
                        WHERE run_id = {run_id_2}
                    )
                    SELECT tb_1.*,
                        tb_1.value_posneg - tb_2.value_other AS value_diff
                    INTO analysis_time_series_diff{sff}
                    FROM tb_1
                    LEFT JOIN tb_2
                    ON tb_1.bool_out = tb_2.bool_out
                        AND tb_1.ca_id = tb_2.ca_id
                        AND tb_1.pp_id = tb_2.pp_id
                        AND tb_1.sy = tb_2.sy;
                    ''').format(run_id_1=self.run_id[0],
                                run_id_2=self.run_id[1],
								sff=self._suffix)
        aql.exec_sql(exec_str, db=self.db)
        return exec_str


    def generate_analysis_time_series_transmission(self):
        exec_str = ('''
                    DROP VIEW IF EXISTS
                        analysis_time_series_transmission_0 CASCADE;
                    DROP TABLE IF EXISTS
                        {sc_out}.analysis_time_series_transmission CASCADE;

                    CREATE VIEW analysis_time_series_transmission_0 AS
                    WITH trrv AS (
                        SELECT sy, run_id, dfnd.nd AS nd,
                            CAST('IMPORT_FROM_' AS VARCHAR) || dfnd_2.nd AS pt,
                            False AS bool_out,
                            CAST('TRANSMISSION' AS VARCHAR) AS pp_broad_cat,
                            CAST('import' AS VARCHAR) AS fl,
                            value,
                            value AS value_posneg
                        FROM {sc_out}.var_tr_trm_rv AS tr
                        LEFT JOIN {sc_out}.def_node AS dfnd
                            ON dfnd.nd_id = tr.nd_id
                        LEFT JOIN {sc_out}.def_node AS dfnd_2
                            ON dfnd_2.nd_id = tr.nd_2_id
                        WHERE run_id IN {in_run_id}
                    ), trsd AS (
                        SELECT sy, run_id, dfnd.nd AS nd,
                            CAST('EXPORT_TO_' AS VARCHAR) || dfnd_2.nd AS pt,
                            True AS bool_out,
                            CAST('TRANSMISSION' AS VARCHAR) AS pp_broad_cat,
                            CAST('export' AS VARCHAR) AS fl,
                            value, - value AS value_posneg
                            FROM {sc_out}.var_tr_trm_sd AS tr
                        LEFT JOIN {sc_out}.def_node AS dfnd
                            ON dfnd.nd_id = tr.nd_id
                        LEFT JOIN {sc_out}.def_node AS dfnd_2
                            ON dfnd_2.nd_id = tr.nd_2_id
                        WHERE run_id IN {in_run_id}
                    ), tm AS (
                        SELECT *
                        FROM profiles_raw.timestamp_template
                        WHERE year = 2015
                    ), trall AS (
                        SELECT * FROM trrv
                        UNION ALL
                        SELECT * FROM trsd
                    )
                    SELECT 0::SMALLINT AS bool_out,
                           trall.value, trall.run_id, tm.*,
                           trall.pt, trall.pp_broad_cat, trall.nd,
                           trall.fl, trall.value_posneg
                    FROM trall
                    LEFT JOIN tm ON tm.slot = trall.sy;

                    SELECT *
                    INTO {sc_out}.analysis_time_series_transmission{sff}
                    FROM {sc_out}.analysis_time_series
                    WHERE NOT pp LIKE '%TRNS%'
                    UNION ALL
                    SELECT * FROM analysis_time_series_transmission_0;
                    ''').format(sc_out=self.sc_out, in_run_id=self.in_run_id,
    								 sff=self._suffix)
        aql.exec_sql(exec_str, db=self.db)


        # add timemap indices
        aql.joinon(self.db, [c for c in self.tm_cols if (not c == 'sy')], ['sy'],
                   ['public', 'analysis_time_series_transmission' + self._suffix],
                   ['public', 'timemap_' + str(self.time_res)])


        return exec_str

    def generate_analysis_time_series_chp_comparison(self):
        ''' Filters production data by CHP plants and appends exogenous CHP
            profile for comparison '''

        exec_str = ('''
                    /* Extension to analysis_time_series */
                    DROP TABLE IF EXISTS analysis_time_series_chp_comparison
                    CASCADE;
                    WITH output AS (
                        SELECT sy, wk_id, how, mt_id, doy, hom,
                               weight, nd, run_id, pt,
                            SUM(value) AS value
                        FROM analysis_time_series
                        WHERE pp_broad_cat = 'CHP'
                        GROUP BY pt, sy, wk_id, how, mt_id, doy,
                                 hom, weight, nd, run_id
                    ), profile AS (
                        SELECT prf.sy, wk_id, how, mt_id, doy, hom,
                               weight, nd, prf.run_id,
                            CAST('profile' AS VARCHAR) AS pt,
                            cap_pwr_tot_chp * value AS value
                        FROM {sc_out}.par_chpprof AS prf
                        LEFT JOIN (SELECT sy, wk_id, how, mt_id,
                                          doy, hom, weight
                                   FROM timemap) AS tm
                        ON prf.sy = tm.sy
                        LEFT JOIN {sc_out}.def_node AS dfnd
                        ON prf.nd_id = dfnd.nd_id
                        LEFT JOIN (
                            SELECT
                                run_id, nd_id,
                                SUM(cap.value) AS cap_pwr_tot_chp
                            FROM {sc_out}.var_yr_cap_pwr_tot AS cap
                            LEFT JOIN {sc_out}.def_plant AS dfpp
                            ON cap.pp_id = dfpp.pp_id
                            WHERE pp LIKE '%CHP%'
                            GROUP BY run_id, nd_id
                            ORDER BY nd_id, run_id
                            ) AS cap
                        ON prf.nd_id = cap.nd_id AND prf.run_id = cap.run_id
                        ORDER BY run_id, nd, sy
                    )
                    SELECT * INTO analysis_time_series_chp_comparison{sff}
                    FROM output
                    UNION ALL
                    SELECT * FROM profile
                    ''').format(sc_out=self.sc_out, sff=self._suffix)
        aql.exec_sql(exec_str, db=self.db)
        return exec_str



if __name__ == '__main__':
    pass


#    res = aql.exec_sql('''
#                 EXPLAIN ANALYZE SELECT *
#                 --INTO out_nucspreadvr.analysis_agg_filtdiff_views
#                 FROM temp_analysis_time_series_subset_reffilt
#                 ''')



