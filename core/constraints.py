
import pyomo.environ as po
import numpy as np

from grimsel.auxiliary.aux_m_func import set_to_list

class Constraints:
    '''
    Mixin class containing all constraints.
    '''
    def add_supply_rules(self):

        print('Supply rule')
        def supply_rule(self, sy, nd, ca):

            prod = (# power output; negative if energy selling plant
                    sum(self.pwr[sy, pp, ca]
                        * (-1 if pp in self.setlst['sll']
                                     + self.setlst['curt']
                                     else 1)
                        for (pp, nd, ca)
                        in set_to_list(self.ppall_ndca - self.curt_ndca,
                                       [None, nd, ca]))
                    # incoming inter-node transmission
                    + sum(self.trm_rv[sy, nd, nd_2, ca] for (nd, nd_2, ca)
                          in set_to_list(self.ndcnn, [nd, None, ca]))
                   )
            dmnd = (self.dmnd[sy, nd, ca]
                    # wasted
                    + sum(self.trm_sd[sy, nd, nd_2, ca] for (nd, nd_2, ca)
                          in set_to_list(self.ndcnn, [nd, None, ca]))
                    + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                          in set_to_list(self.st_ndca, [None, nd, ca]))
                    # demand of plants using ca as an input
                    + sum(self.pwr[sy, pp, ca_out] / self.pp_eff[pp, ca_out]
                          for (pp, nd, ca_out, ca)
                          in set_to_list(self.pp_ndcaca,
                                         [None, nd, None, ca]))
                    )
            return prod * (1 - self.grid_losses[nd, ca]) == dmnd
        self.supply = po.Constraint(self.sy, self.ndca, rule=supply_rule)


    # decorator to print function name???
    def add_energy_aggregation_rules(self):
        '''
        Calculation of yearly sums of energy from
        time slot power variables.
        '''

        print('Calculation of yearly totals rules')
        def YearlyEnergy_rule(self, pp, ca):
            ''' Yearly energy production consumed per power plant and output
            energy carrier. Relevant only for fuel-consuming power plants.'''
            return (self.erg_yr[pp, ca]
                    == sum(self.pwr[sy, pp, ca]
                       * self.weight[sy] for sy in self.sy))
        self.YearlyEnergy = po.Constraint(self.ppall_ca,
                                          rule=YearlyEnergy_rule)

#        print('Yearly flexible demand calculation rule')
#        def yearly_flex_dmnd_rule(self, nd, ca):
#            ''' Yearly amount of flexible demand (curtailment). '''
#            return (self.dmnd_flex_yr[nd, ca]
#                    == sum(self.dmnd_flex[sy, nd, ca] * self.weight[sy]
#                           for sy in self.sy))
#        self.yrfxdd = po.Constraint(self.ndca_EL, rule=yearly_flex_dmnd_rule)

        print('Yearly ramping calculation rule')
        def yearly_ramp_rule(self, pp, ca):
            ''' Yearly ramping in MW/yr. Up and down aggregated. '''
            return (self.pwr_ramp_yr[pp, ca]
                    == sum(self.pwr_ramp_abs[sy, pp, ca] for sy in self.sy))
        self.yrrmp = po.Constraint(self.pp_ca | self.hyrs_ca | self.ror_ca,
                                   rule=yearly_ramp_rule)

        print('Yearly fuel consumption rule')
        def yearly_fuel_cons_rule(self, pp, nd, ca, fl):
            ''' Yearly fuel consumed per power plant and
            output energy carrier. '''
            return (self.erg_fl_yr[pp, nd, ca, fl]
                    == self.erg_yr[pp, ca])
        self.yrflcs = po.Constraint(self.pp_ndcafl,
                                    rule=yearly_fuel_cons_rule)

        print('Yearly charging rule')
        def yearly_chg_rule(self, pp, ca):
            agg_erg_chg_yr = sum(self.pwr_st_ch[sy, pp, ca] * self.weight[sy]
                                 for sy in self.sy)
            return self.erg_ch_yr[pp, ca] == agg_erg_chg_yr
        self.yrstcg = po.Constraint(self.st_ca, rule=yearly_chg_rule)
        self.yrstcg.doc = 'Yearly storage charging energy.'

        print('Yearly transmission send rule')
        def yearly_trm_sd_rule(self, nd, nd_2, ca):
            return (self.erg_trm_sd_yr[nd, nd_2, ca]
                    == sum(self.trm_sd[sy, nd, nd_2, ca]
                       * self.weight[sy] for sy in self.sy))
        self.yrtrsd = po.Constraint(self.ndcnn, rule=yearly_trm_sd_rule)
        self.yrtrsd.doc = 'Yearly cross border exchanged energy (send).'

        print('Yearly transmission receive rule')
        def yearly_trm_rv_rule(self, nd, nd_2, ca):
            return (self.erg_trm_rv_yr[nd, nd_2, ca]
                    == sum(self.trm_rv[sy, nd, nd_2, ca]
                       * self.weight[sy] for sy in self.sy))
        self.yrtrrv = po.Constraint(self.ndcnn, rule=yearly_trm_rv_rule)
        self.yrtrrv.doc = 'Yearly cross border exchanged energy (receive).'

    def add_transmission_rules(self):
        print('Symmetry of transmission rule')
        def trm_symm_rule(self, sy, nd1, nd2, ca):
            return (self.trm_sd[sy, nd1, nd2, ca]
                    == self.trm_rv[sy, nd2, nd1, ca])
        self.trm_symm = po.Constraint(self.sy, self.ndcnn, rule=trm_symm_rule)

    def add_capacity_rules(self):

        print('Calculate total capacity')
        def calc_cap_pwr_tot_rule(self, pp, ca):
            cap_tot = self.cap_pwr_leg[pp, ca]
            if (pp in self.setlst['add']):
                cap_tot += self.cap_pwr_new[pp, ca]
            if (pp in self.setlst['rem']):
                cap_tot -= self.cap_pwr_rem[pp, ca]
            return (self.cap_pwr_tot[pp, ca] == cap_tot)
        self.calc_cap_pwr_tot = po.Constraint(self.ppall_ca,
                                              rule=calc_cap_pwr_tot_rule)

        print('Calculate total energy capacity from power capacity')
        def calc_cap_erg_tot_rule(self, pp, ca):
            return (self.cap_erg_tot[pp, ca]
                    == self.cap_pwr_tot[pp, ca]
                       * self.discharge_duration[pp, ca])
        self.calc_cap_erg_tot = po.Constraint(self.st_ca | self.hyrs_ca,
                                              rule=calc_cap_erg_tot_rule)

        print('Capacity constraints rules...')
        print('- Transmission')
        def trm_sd_capac_rule(self, sy, nd1, nd2, ca):
            return (self.trm_sd[sy, nd1, nd2, ca]
                    <= self.cap_trm_leg[self.dict_soy_month[sy], nd1, nd2, ca])
        self.trm_sd_capac_rule = po.Constraint(self.sy, self.ndcnn,
                                               rule=trm_sd_capac_rule)

        print('- Power capacity pps')
        def ppst_capac_rule(self, pp, ca, sy):
            ''' Produced power less than capacity. '''
            
            dict_nuc_fr_scale = {(0, 34): 0.9505061879261046,
                                 (1, 34): 0.8137875854801296,
                                 (2, 34): 0.8022824210606725,
                                 (3, 34): 0.6766355042299101,
                                 (4, 34): 0.6579847531960288,
                                 (5, 34): 0.6664230450313823,
                                 (6, 34): 0.7219366181477619,
                                 (7, 34): 0.6941977859795634,
                                 (8, 34): 0.66484147338352,
                                 (9, 34): 0.7473448651685072,
                                 (10, 34): 0.774015880758718,
                                 (11, 34): 0.874112544211129}

#            if pp == 34:
#                mt = set_to_list(self.sy_mt, [sy, None])[0][1]
#                cap_scale = dict_nuc_fr_scale[(mt, 34)]
#            else:
            cap_scale = 1
            
            return (self.pwr[sy, pp, ca]
                    <= self.cap_pwr_tot[pp, ca] * cap_scale)
        self.PpStCapac = po.Constraint(self.pp_ca - self.pr_ca | self.st_ca
                                       | self.hyrs_ca,
                                       self.sy, rule=ppst_capac_rule)

        print('- Power capacity storage charging')
        def st_capac_rule_pw_ch(self, pp, ca, sy):
            ''' Charging power less than nominal power capacity. '''
            return (self.pwr_st_ch[sy, pp, ca]
                    <= self.cap_pwr_tot[pp, ca])
        self.CapacStPwCh = po.Constraint(self.st_ca, self.sy,
                                         rule=st_capac_rule_pw_ch)

        print('- Energy capacity')
        def st_capac_rule_en(self, pp, ca, sy):
            return (self.erg_st[sy, pp, ca]
                    <= self.cap_erg_tot[pp, ca])
        self.CapacStEn = po.Constraint((self.st_ca | self.hyrs_ca), self.sy,
                                       rule=st_capac_rule_en)


    def add_chp_new_rules(self):
        def chp_prof_rule(model, sy, nd, ca, fl):
            '''Produced power greater than CHP output profile.'''

            sum_pwr = sum(self.pwr[sy, pp, ca] for (pp, nd, ca, fl)
                          in set_to_list(self.pp_ndcafl, (None, nd, ca, fl)))
            chp_prf = self.chpprof[sy, nd, ca] * self.erg_chp[nd, ca, fl]

            return sum_pwr >= chp_prf

        self.chp_prof = po.Constraint(self.sy, self.ndcafl_chp,
                                      rule=chp_prof_rule)


    def add_monthly_total_rules(self):

        print('Calculation of monthly totals rule')
        def monthly_totals_rule(self, mt, pp, ca):
            ''' Monthly sums hydro. '''
            return (self.erg_mt[mt, pp, ca]
                    == sum(self.pwr[sy, pp, ca]
                       * self.weight[sy]
                       for sy in self.dict_month_soy[mt]))
        self.monthly_totals = po.Constraint(self.mt,
                                            self.pp_ca | self.hyrs_ca,
                                            rule=monthly_totals_rule)

    def add_chp_rules(self):
        # TODO explicit demand for all ca --> constrain pwr-to-heat ratios
#        pass
        def chp_prof_rule(self, sy, pp, nd, ca):
            '''Produced power greater than CHP output profile.'''
            left = self.pwr[sy, pp, ca]

            if ca == 0:
                return left >= self.chpprof[sy, nd, ca] * self.cap_pwr_tot[pp, ca]
            else:
                return po.Constraint.Skip

        self.chp_prof = po.Constraint(self.sy, self.chp_ndca,
                                      rule=chp_prof_rule)

        print('CHP capacity share must not change')
        def set_chp_cap_rule(self, nd):
            total_chp = sum(self.cap_pwr_tot[pp, ca] for (pp, nd, ca)
                            in set_to_list(self.chp_ndca, (None, nd, 0)))
            return total_chp >= self.chp_cap_pwr_leg[nd]

        self.set_chp_cap = po.Constraint(self.nd, rule=set_chp_cap_rule)

    def add_variables_rules(self):
        print('Profile rule variables')
        def variables_prof_rule(self, sy, pp, nd, ca):
            ''' Produced power equal output profile '''
            left = self.pwr[sy, pp, ca]
            return left == self.supprof[sy, pp, ca] * self.cap_pwr_tot[pp, ca]
        self.variables_prof = po.Constraint(self.sy, self.pr_ndca,
                                            rule=variables_prof_rule)
    def add_ramp_rate_rules(self):

        print('Calculation of ramp rates rule')
        def calc_ramp_rate_rule(self, pp, ca, sy):
            this_soy = sy
            last_soy = (sy - 1) if this_soy != self.sy[1] else self.sy[-1]

            return (self.pwr_ramp[sy, pp, ca]
                    == self.pwr[this_soy, pp, ca]
                     - self.pwr[last_soy, pp, ca])
        self.calc_ramp_rate = po.Constraint(self.pp_ca | self.hyrs_ca
                                            | self.ror_ca, self.sy,
                                            rule=calc_ramp_rate_rule)

        print('Calculation of absolute ramp rates rule')
        def ramp_rate_abs_rule(self, pp, ca, sy):
            return (flag_abs * self.pwr_ramp[sy, pp, ca]
                    <= self.pwr_ramp_abs[sy, pp, ca])

        flag_abs = 1
        self.ramp_rate_abs_pos = po.Constraint(self.pp_ca | self.hyrs_ca
                                               | self.ror_ca, self.sy,
                                               rule=ramp_rate_abs_rule)
        flag_abs = -1
        self.ramp_rate_abs_neg = po.Constraint(self.pp_ca | self.hyrs_ca
                                               | self.ror_ca, self.sy,
                                               rule=ramp_rate_abs_rule)

    def add_energy_constraint_rules(self):

        if 'cf_max' in self.parameter_month_list:

            print('Capacity factor limitation rule')
            def pp_cf_rule(self, mt, pp, ca):

#                if 34 == pp:
#                    return po.Constraint.Skip
#                else:
                return (self.erg_mt[mt, pp, ca]
                        <= self.cf_max[mt, pp, ca]
                           * self.cap_pwr_tot[pp, ca]
                           * self.month_weight[mt])
            self.pp_cf = po.Constraint(self.mt, self.pp_ca - self.pr_ca,
                                       rule=pp_cf_rule,
                                       doc='Capacity factor constrained.')
        else:
            print('Capacity factor limitation rule')
            def pp_cf_rule(self, pp, ca):
#                if 34 == pp:
#                    return po.Constraint.Skip
#                else:
                return (self.erg_yr[pp, ca]
                        <= self.cf_max[pp, ca]
                            * self.cap_pwr_tot[pp, ca]
                            * 8760)
            self.pp_cf = po.Constraint(self.pp_ca - self.pr_ca,
                                       rule=pp_cf_rule,
                                       doc='Capacity factor constrained.')

        print('Fuel constraint rule')
        def pp_max_fuel_rule(self, nd, ca, fl):
            '''Maximum energy produced from a certain fuel in a certain pp.
               Note: this should be defined in fuels_base, not plant_fuel.'''

            # making sure we are dealing with a constrained fuel
            is_constr = fl in self.fl_erg

            # Skipping the case where erg_yr = 0 and no corresponding plants
            erg_inp_is_zero = self.erg_inp[nd, ca, fl] == 0
            no_plants_for_fuel = set_to_list(self.ppall_ndcafl,
                                             [None, nd, ca, fl]) == []

            if is_constr and not erg_inp_is_zero and not no_plants_for_fuel:
                left = sum(self.erg_fl_yr[pp, nd_1, ca_1, fl_1]
                           for (pp, nd_1, ca_1, fl_1) in
                           set_to_list(self.ppall_ndcafl, [None, nd, ca, fl]))
                right = self.erg_inp[nd, ca, fl]

                return left <= right
            else:
                return po.Constraint.Skip

        self.pp_max_fuel = po.Constraint(self.ndcafl, rule=pp_max_fuel_rule)

#        print('Fixed VRE share')
#        def set_win_sol_rule(self, nd):
#            'Calculation of wind and solar share rule; note: activated in loop'
#
#
#
#            prod_ws = sum(self.erg_yr[(pp, 0)] for pp in self.setlst['winsol']
#                        if (pp, nd, 0) in [i for i in self.pr_ndca])
#            if not self.share_ws_set[nd].value == None:
#                return self.dmnd_sum[nd] * self.share_ws_set[nd] == prod_ws
#            else:
#                return po.Constraint.Skip
#        self.set_win_sol = po.Constraint(self.nd, rule=set_win_sol_rule)
#        self.set_win_sol.deactivate()

    def add_charging_level_rules(self):

        print('Storage level rule')
        def erg_store_level_rule(self, pp, nd, ca, fl, sy):
            ''' Charging state for storage and hydro. '''
            this_soy = sy
            last_soy = (sy - 1) if this_soy != self.sy[1] else self.sy[-1]

            left = 0
            right = 0

            # last time slot's energy level for storage and hyrs
            # this excludes run-of-river, which doesn't have an energy variable
            if pp in self.setlst['st'] + self.setlst['hyrs']:
                left += self.erg_st[this_soy, pp, ca] # in MWh of stored energy
                right += self.erg_st[last_soy, pp, ca] #* (1-self.st_lss_hr[pp, ca])

            if pp in self.setlst['st']:
                right += ((- self.pwr[this_soy, pp, ca]
                           / (1 - self.st_lss_rt[pp, ca])**(1/2)
                           * self.weight[sy]
                         ) + (
                           self.pwr_st_ch[this_soy, pp, ca]
                           * (1 - self.st_lss_rt[pp, ca])**(1/2)
                           * self.weight[sy]))
            elif pp in self.setlst['hyrs'] + self.setlst['ror']:
                right += (
                          # inflowprof profiles are normalized to one!!
                          (self.inflowprof[this_soy, pp, ca]
                           * self.erg_inp[nd, ca, fl]
                          - self.pwr[sy, pp, ca]) * self.weight[sy]
                         )
            return left == right
        self.erg_store_level = po.Constraint(self.st_ndcafl | self.hyrs_ndcafl | self.ror_ndcafl,
                                             self.sy, rule=erg_store_level_rule)

    def add_hydro_rules(self):

        print('Reservoir boundary conditions rule')
        def hy_reservoir_boundary_conditions_rule(self, pp, ca, sy):
            if (sy, pp) in [i for i in self.hyd_erg_bc.sparse_iterkeys()]:
                return (self.erg_st[sy, pp, ca]
                        == self.hyd_erg_bc[sy, pp] * self.cap_erg_tot[pp, ca])
            else:
                return po.Constraint.Skip
        self.hy_reservoir_boundary_conditions = (
                po.Constraint(self.hyrs_ca, self.sy,
                              rule = hy_reservoir_boundary_conditions_rule))
#
        
        exclude_hyd = [self.mps.dict_pp_id[pp] for pp in [
#                                                          'DE_HYD_RES',
#                                                          'AT_HYD_RES',
#                                                          'CH_HYD_RES',
#                                                          'IT_HYD_RES',
#                                                          'FR_HYD_RES',
                                                          ]]
        
        print('Hydro minimum monthly generation as fraction of maximum monthly inflow')
        def hy_month_min_rule(self, mt, pp, nd, ca, fl):
            if pp in exclude_hyd:
                return po.Constraint.Skip
            else:
                return (self.erg_mt[mt, pp, ca]
                        >= self.max_erg_mt_in_share[pp]
                         * self.min_erg_mt_out_share[pp]
                         * self.erg_inp[nd, ca, fl])
        self.hy_month_min = po.Constraint(self.mt, self.hyrs_ndcafl,
                                          rule=hy_month_min_rule)
        #### MINIMUM RESERVOIR LEVEL HYDRO
        def hy_erg_min_rule(self, pp, ca, sy):
            if not pp in [h for h in self.min_erg_share]:
                return po.Constraint.Skip
            else:
                return (self.erg_st[sy, pp, ca]
                        >=
                        self.min_erg_share[pp]
                        * self.cap_erg_tot[pp, ca])
        self.hy_erg_min = po.Constraint(self.hyrs_ca, self.sy,
                                        rule=hy_erg_min_rule)

    def add_ror_rules(self):

        print('Weekly totals rule')
        def weekly_totals_rule(self, wk, pp, ca):
            return (self.erg_wk[wk, pp, self.mps.dict_ca_id['EL']]
                    == sum(self.pwr[sy, pp, self.mps.dict_ca_id['EL']]
                       * self.weight[sy]
                    for sy in self.dict_week_soy[wk]))
        self.weekly_totals = po.Constraint(self.wk, self.ror_ca,
                                           rule=weekly_totals_rule)

        print('Run-of-river weekly production constraint')
        def ror_weekly_energy_rule(self, wk, pp):
            return (self.erg_wk[wk, pp, self.mps.dict_ca_id['EL']]
                    <= self.week_ror_output[wk, pp])
        self.ror_weekly_energy = po.Constraint(self.wk, self.ror,
                                               rule=ror_weekly_energy_rule)

        print('Run-of-river minimum power output rule')
        def ror_min_base_load_rule(self, sy, pp):
            wk = self.dict_soy_week[sy]
            return (self.erg_wk[wk, pp, self.mps.dict_ca_id['EL']]
                    * 0.8 / self.wk_weight[wk]
                    <= self.pwr[sy, pp, self.mps.dict_ca_id['EL']])
        self.ror_min_base_load = po.Constraint(self.sy, self.ror,
                                               rule=ror_min_base_load_rule)

#
#    def add_scenario_rules(self):
#        '''
#        Constraints which are only active during certain model runs
#        '''
#        def chg_only_var_ren_rule(self, nd, ca, sy):
#            '''
#            Total electric charging power is limited to VRE output
#            '''
#            pch = sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
#                      in set_to_list(self.st_ndca & self.scen_ndca,
#                                     [None, nd, ca]))
#            ppr = sum(self.pwr[sy, pr, ca] for (pr, nd, ca)
#                      in set_to_list(self.winsol_ndca, [None, nd, ca]))
#            return pch <= ppr
#        self.chg_only_var_ren = po.Constraint(self.ndca_EL, self.sy,
#                            rule=chg_only_var_ren_rule)
#
#        self.chg_only_var_ren.deactivate()

    def add_yearly_cost_rules(self):
        print('Fuel consumption calculation rule')
        def calc_vc_fl_pp_rule(self, pp, nd, ca, fl):

            sign = -1 if pp in self.setlst['sll'] else 1

            # Case 1: fuel has price profile
            if (pp, nd, ca, fl) in self.pp_ndcafl_prof:
                sums = (sign * sum(self.weight[sy] * self.priceprof[sy, nd, fl]
                                   / self.pp_eff[pp, ca]
                                   * self.pwr[sy, pp, ca] for sy in self.sy))
            # Case 2: monthly adjustment factors have been applied to vc_fl
            elif 'vc_fl' in self.parameter_month_list:
                sums = (sign * sum(self.weight[sy] * self.vc_fl[mt, fl, nd]
                                                   / self.pp_eff[pp, ca]
                                                   * self.pwr[sy, pp, ca]
                                   for (sy, mt) in set_to_list(self.sy_mt,
                                                               [None, None])))
            # Case 3: ordinary single fuel price
            else:
                sums = (sign * self.erg_fl_yr[pp, nd, ca, fl]
                             / self.pp_eff[pp, ca]
                             * self.vc_fl[fl, nd])

            return self.vc_fl_pp_yr[pp, ca, fl] == sums

        self.calc_vc_fl_pp = po.Constraint(self.pp_ndcafl - self.lin_ndcafl,
                                           rule=calc_vc_fl_pp_rule)

        print('OM VC calculation rule')
        def calc_vc_om_pp_rule(self, pp, ca):
            return (self.erg_yr[pp, ca] * self.vc_om[pp, ca]
                    == self.vc_om_pp_yr[pp, ca])
        self.calc_vc_om_pp = po.Constraint(self.ppall_ca,
                                           rule=calc_vc_om_pp_rule)

        print('CO2 VC calculation rule --> OBSOLETE: Linear cost included directly in objective!!')
#        def calc_vc_co2_pp_rule(self, pp, nd, ca, fl):
#
#            # Case 1: monthly adjustment factors have been applied to vc_fl
#            if 'price_co2' in self.parameter_month_list:
#                sums = sum(self.pwr[sy, pp, ca] # POWER!
#                           / self.pp_eff[pp, ca] * self.weight[sy]
#                           * self.price_co2[mt, nd] * self.co2_int[fl]
#                           for (sy, mt) in set_to_list(self.sy_mt,
#                                                       [None, None]))
#            # Case 2: ordinary single CO2 price
#            else:
#                sums = (self.erg_fl_yr[pp, nd, ca, fl] # ENERGY!
#                            / self.pp_eff[pp, ca]
#                            * self.price_co2[nd] * self.co2_int[fl])
#
#            return self.vc_co2_pp_yr[pp, ca] == sums
#
#        self.calc_vc_co2_pp = po.Constraint(self.pp_ndcafl - self.lin_ppndcafl,
#                                            rule=calc_vc_co2_pp_rule)
#







#        print('Flexible demand VC calculation rule')
#        def calc_vc_flex_dmnd_rule(self, nd, ca):
#            return (self.vc_dmnd_flex_yr[nd, ca]
#                    == self.dmnd_flex_yr[nd, ca] * self.vc_dmnd_flex[nd, ca])
#        self.calc_vc_flex_dmnd = po.Constraint(self.ndca,
#                                               rule=calc_vc_flex_dmnd_rule)

        print('Ramp VC calculation rule')
        def calc_vc_ramp_rule(self, pp, ca):
            
#            if pp == 34:
#                return po.Constraint.Skip
#            else:
            return (self.vc_ramp_yr[pp, ca]
                    == self.pwr_ramp_yr[pp, ca] * self.vc_ramp[pp, ca])
        self.calc_vc_ramp = po.Constraint(self.pp_ca
                                          | self.hyrs_ca | self.ror_ca,
                                          rule=calc_vc_ramp_rule)

        print('Fixed O&M cost calculation rule')
        def calc_fc_om_rule(self, pp, ca):
            return (self.fc_om_pp_yr[pp, ca]
                    == self.cap_pwr_tot[pp, ca] * self.fc_om[pp, ca])
        self.calc_fc_om = po.Constraint(self.ppall_ca, rule=calc_fc_om_rule)

        print('Fixed capital cost calculation rule')
        def calc_fc_cp_rule(self, pp, ca):
            return (self.fc_cp_pp_yr[pp, ca]
                    == self.cap_pwr_new[pp, ca] * self.fc_cp_ann[pp, ca])
        self.calc_fc_cp = po.Constraint(self.add_ca, rule=calc_fc_cp_rule)

#    def add_objective_rules(self):
#        print('Objective rule')
#        nnn = [None] * 3
#        nn = [None] * 2
#        def objective_rule(self):
#            return (sum(self.vc_fl_pp_yr[pp, ca, fl]
#                        for (pp, ca, fl) in set_to_list(self.pp_cafl, nnn))
#                  + sum(self.vc_co2_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.pp_ca, nn))
#                  + sum(self.vc_om_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
##                  + sum(self.vc_dmnd_flex_yr[nd, ca]
##                        for (nd, ca) in set_to_list(self.ndca_EL, nn))
#                  + sum(self.vc_ramp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.pp_ca | self.hyrs_ca
#                                                    | self.ror_ca, nn))
#                  + sum(self.fc_om_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
#                  + sum(self.fc_cp_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.add_ca, nn))
#                )
#        self.objective = po.Objective(rule=objective_rule, sense=po.minimize,
#                                   doc='Define objective function')


    def add_objective_rules(self):
        print('Objective rule quadratic')
        nnn = [None] * 3
        nn = [None] * 2
        def objective_rule_quad(self):
            return (# FUEL COST CONSTANT
                    sum(self.vc_fl_pp_yr[pp, ca, fl]
                        for (pp, ca, fl)
                        in set_to_list(self.pp_cafl - self.lin_cafl, nnn))
                    # FUEL COST LINEAR
                  + sum(sum(self.pwr[sy, lin, ca]
                            * self.weight[sy]
                            * (self.vc_fl_lin_0[lin, ca]
                               + self.pwr[sy, lin, ca]
                               * self.vc_fl_lin_1[lin, ca])
                            for sy in self.sy)
                        for (lin, ca) in set_to_list(self.lin_ca, nn))
                    # EMISSION COST LINEAR (NOTE: all fossil plants are linear)
                  + sum(sum(self.pwr[sy, lin, ca]
                            * self.weight[sy]
                            * (self.price_co2[mt, nd]
                               if 'price_co2' in self.parameter_month_list
                               else self.price_co2[nd])
                            * (self.factor_vc_co2_lin_0[lin, ca]
                               + self.pwr[sy, lin, ca]
                               * self.factor_vc_co2_lin_1[lin, ca])
                            for (sy, mt) in set_to_list(self.sy_mt, nn))
                        for (lin, nd, ca) in set_to_list(self.lin_ndca, nnn))
#                  + sum(self.vc_co2_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.pp_ca, nn))
                  + sum(self.vc_om_pp_yr[pp, ca]
                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
#                  + sum(self.vc_dmnd_flex_yr[nd, ca]
#                        for (nd, ca) in set_to_list(self.ndca_EL, nn))
                  + sum(self.vc_ramp_yr[pp, ca]
                        for (pp, ca) in set_to_list(self.pp_ca | self.hyrs_ca
                                                    | self.ror_ca, nn))
                  + sum(self.fc_om_pp_yr[pp, ca]
                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
                  + sum(self.fc_cp_pp_yr[pp, ca]
                        for (pp, ca) in set_to_list(self.add_ca, nn))
                )
        self.objective_quad = po.Objective(rule=objective_rule_quad,
                                           sense=po.minimize,
                                           doc='Quadratic objective function')
#
#        print('Objective rule linear')
#        nnn = [None] * 3
#        nn = [None] * 2
#        def objective_rule(self):
#            return (sum(self.vc_fl_pp_yr[pp, ca, fl]
#                        for (pp, ca, fl) in set_to_list(self.pp_cafl - self.lin_cafl, nnn))
#                  + sum(self.vc_co2_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.pp_ca, nn))
#                  + sum(self.vc_om_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
##                  + sum(self.vc_dmnd_flex_yr[nd, ca]
##                        for (nd, ca) in set_to_list(self.ndca_EL, nn))
#                  + sum(self.vc_ramp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.pp_ca | self.hyrs_ca
#                                                    | self.ror_ca, nn))
#                  + sum(self.fc_om_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
#                  + sum(self.fc_cp_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.add_ca, nn))
#                )
#        self.objective_lin = po.Objective(rule=objective_rule,
#                                          sense=po.minimize,
#                                          doc='Objective function')
#


