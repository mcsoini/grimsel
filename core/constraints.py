
from functools import wraps

import pyomo.environ as po

from grimsel.auxiliary.aux_m_func import set_to_list
from grimsel import _get_logger

logger = _get_logger(__name__)


nnnn = [None] * 4
nnn = [None] * 3
nn = [None] * 2



def _limit_max_sy(dct):
    '''
    Decorator limiting the constraints to a timemap-dependent range

    Note: Only useful if the sy_pp_ca sets are not defined (which they will
    probably always be, since they are required by the variables).

    Parameters
    ----------
        dct: dict
            maximum time slot in dependence on the other parameters
            (typically ``{nd: sy_max}`` or ``{pp: sy_max}``)

    '''
    def wrapper(f):
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            if args[0] <= dct[args[1]]:
                return f(self, *args, **kwargs)
            else:
                return po.Constraint.Skip
        return wrapped
    return wrapper



class Constraints:
    '''
    Mixin class containing all constraints.
    '''


    def add_transmission_bounds_rules(self):
        '''
        Add transmission bounds.

        :math:`-C_\mathrm{import} \leqslant p_{\mathrm{trm}, t} \leqslant C_\mathrm{export}`

        *Note*: This method modifies the ``trm`` transmission power Pyomo
        variable object by calling its ``setub`` and ``setlb`` methods.

        '''


        if hasattr(self, 'trm'):
            for sy, nd1, nd2, ca in self.trm:

                tm = self.dict_ndnd_tm_id[nd1, nd2]

                mt = self.dict_soy_month[(tm, sy)]
                ub = self.cap_trme_leg[mt, nd1, nd2, ca]
                lb = - self.cap_trmi_leg[mt, nd1, nd2, ca]
                self.trm[(sy, nd1, nd2, ca)].setub(ub)
                self.trm[(sy, nd1, nd2, ca)].setlb(lb)

    def add_supply_rules(self):
        r'''
        Adds the supply rule.

        .. math::

           & p_\mathrm{pp} - p_\mathrm{sell} - p_\mathrm{curt} \\
           & + p_\mathrm{trm} \\
           & \leqslant \\
           & p_\mathrm{dmnd} + p_\mathrm{st,chg} \\
           & + p_\mathrm{pp, ca_out} / \eta_\mathrm{pp}


        '''

# %%

        def get_transmission(sy, nd, nd_2, ca, export=True):
            '''
            If called by supply rule, the order nd, nd_2 is always the ndcnn
            order, therefore also trm order.

            Case 1: nd has higher time resolution (min) -> just use
                    trm[tm, sy, nd, nd_2, ca]
            Case 2: nd has lower time resolution (not min) -> average
                    avg(trm[tm, all the sy, nd, nd_2, ca])

                    self.dict_sysy[nd, nd_2, sy]

            Parameters
            ----------
                - sy (int): current time slot in nd
                - nd (int): outgoing node
                - nd_2 (int): incoming node
                - ca (int): energy carrier
                - export (bool): True if export else False

            '''
            if self.is_min_node[(nd if export else nd_2,
                                 nd_2 if export else nd)]:
                trm = self.trm[sy, nd, nd_2, ca]
                return trm

            else: # average over all of the other sy
                list_sy2 = self.dict_sysy[nd if export else nd_2,
                                          nd_2 if export else nd, sy]

                avg = 1/len(list_sy2) * sum(self.trm[_sy, nd, nd_2, ca]
                                            for _sy in list_sy2)
                return avg

        logger.info('Supply rule')
        def supply_rule(self, sy, nd, ca):

            list_neg = self.setlst['sll'] + self.setlst['curt']
            prod = (# power output; negative if energy selling plant
                    sum(self.pwr[sy, pp, ca]
                        * (-1 if pp in list_neg else 1)
                        for (pp, nd, ca)
                        in set_to_list(self.ppall_ndca, [None, nd, ca]))
                    # incoming inter-node transmission
                    + sum(get_transmission(sy, nd, nd_2, ca,    False)
                          for (nd, nd_2, ca)
                          in set_to_list(self.ndcnn, [None, nd, ca]))
                   )
            exports = sum(get_transmission(sy, nd, nd_2, ca, True)
                          for (nd, nd_2, ca)
                          in set_to_list(self.ndcnn, [nd, None, ca]))
            dmnd = (self.dmnd[sy, self.dict_dmnd_pf[(nd, ca)]]
                    + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                          in set_to_list(self.st_ndca, [None, nd, ca]))
                    # demand of plants using ca as an input
                    + sum(self.pwr[sy, pp, ca_out] / self.pp_eff[pp, ca_out]
                          for (pp, nd, ca_out, ca)
                          in set_to_list(self.pp_ndcaca,
                                         [None, nd, None, ca]))
                    )
            return prod == dmnd * (1 + self.grid_losses[nd, ca]) + exports
        self.supply = po.Constraint(self.sy_ndca, rule=supply_rule)


    def add_energy_aggregation_rules(self):
        '''
        Calculation of yearly sums of energy from
        time slot power variables.
        '''

        logger.info('Calculation of yearly totals rules')
        def yearly_energy_rule(self, pp, ca):
            ''' Yearly energy production consumed per power plant and output
            energy carrier. Relevant only for fuel-consuming power plants.'''

            tm = self.dict_pp_tm_id[pp]

            return (self.erg_yr[pp, ca]
                    == sum(self.pwr[sy, pp, ca] * self.weight[tm, sy]
                           for tm, sy in set_to_list(self.tmsy, [tm, None])))

        self.yearly_energy = po.Constraint(self.ppall_ca,
                                           rule=yearly_energy_rule)

        logger.info('Yearly ramping calculation rule')
        def yearly_ramp_rule(self, pp, ca):
            ''' Yearly ramping in MW/yr. Up and down aggregated. '''
            tm = self.dict_pp_tm_id[pp]
            tmsy_list = set_to_list(self.tmsy, [tm, None])

            return (self.pwr_ramp_yr[pp, ca]
                    == sum(self.pwr_ramp_abs[sy, pp, ca]
                           for tm, sy in tmsy_list))

        self.yearly_ramping = po.Constraint(self.rp_ca, rule=yearly_ramp_rule)

        logger.info('Yearly fuel consumption rule')
        def yearly_fuel_cons_rule(self, pp, nd, ca, fl):
            ''' Yearly fuel consumed per power plant and
            output energy carrier. '''
            return (self.erg_fl_yr[pp, nd, ca, fl]
                    == self.erg_yr[pp, ca])
        self.yrflcs = po.Constraint(self.pp_ndcafl,
                                    rule=yearly_fuel_cons_rule)

        logger.info('Yearly charging rule')
        def yearly_chg_rule(self, pp, ca):

            tm = self.dict_pp_tm_id[pp]
            tmsy_list = set_to_list(self.tmsy, [tm, None])

            agg_erg_chg_yr = sum(self.pwr_st_ch[sy, pp, ca]
                                 * self.weight[tm, sy]
                                 for tm, sy in tmsy_list)
            return self.erg_ch_yr[pp, ca] == agg_erg_chg_yr
        self.yearly_charging = po.Constraint(self.st_ca, rule=yearly_chg_rule)

    def add_capacity_calculation_rules(self):

        logger.info('Calculate total capacity')
        def calc_cap_pwr_tot_rule(self, pp, ca):
            cap_tot = self.cap_pwr_leg[pp, ca]
            if (pp in self.setlst['add']):
                cap_tot += self.cap_pwr_new[pp, ca]
            if (pp in self.setlst['rem']):
                cap_tot -= self.cap_pwr_rem[pp, ca]
            return (self.cap_pwr_tot[pp, ca] == cap_tot)
        self.calc_cap_pwr_tot = po.Constraint(self.ppall_ca,
                                              rule=calc_cap_pwr_tot_rule)

        logger.info('Calculate total energy capacity from power capacity')
        def calc_cap_erg_tot_rule(self, pp, ca):
            return (self.cap_erg_tot[pp, ca]
                    == self.cap_pwr_tot[pp, ca]
                       * self.discharge_duration[pp, ca])
        self.calc_cap_erg_tot = po.Constraint(self.st_ca | self.hyrs_ca,
                                              rule=calc_cap_erg_tot_rule)


    def add_capacity_constraint_rules(self):

        logger.info('Capacity constraints rules...')
        logger.info('- Power capacity pps')
        def ppst_capac_rule(self, sy, pp, ca):
            ''' Produced power less than capacity. '''

            tm = self.dict_pp_tm_id[pp]

            mt = self.dict_soy_month[(tm, sy)]

            if pp in self.setlst['pp']:
                return (self.pwr[sy, pp, ca] <= self.cap_pwr_tot[pp, ca]
                                                * self.cap_avlb[mt, pp, ca])
            else:
                return (self.pwr[sy, pp, ca] <= self.cap_pwr_tot[pp, ca])

        self.ppst_capac = po.Constraint((self.sy_pp_ca - self.sy_pr_ca)
                                         | self.sy_st_ca | self.sy_hyrs_ca,
                                         rule=ppst_capac_rule)

        logger.info('- Power capacity storage charging')
        def st_chg_capac_rule(self, sy, pp, ca):
            ''' Charging power less than nominal power capacity. '''

            return (self.pwr_st_ch[sy, pp, ca]
                    <= self.cap_pwr_tot[pp, ca])
        self.st_chg_capac = po.Constraint(self.sy_st_ca,
                                          rule=st_chg_capac_rule)

        logger.info('- Energy capacity')
        def st_erg_capac_rule(self, sy, pp, ca):
            return (self.erg_st[sy, pp, ca]
                    <= self.cap_erg_tot[pp, ca])
        self.st_erg_capac = po.Constraint(self.sy_st_ca | self.sy_hyrs_ca,
                                          rule=st_erg_capac_rule)


    def add_chp_rules(self):

        def chp_prof_rule(model, sy, pp, ca):
            '''Produced power greater than CHP output profile.'''

            nd = self.mps.dict_plant_2_node_id[pp]

            return (self.pwr[sy, pp, ca]
                    >= self.chpprof[sy, nd, ca] * self.erg_chp[pp, ca])

        self.chp_prof = po.Constraint(self.sy_chp_ca, rule=chp_prof_rule)


    def add_monthly_total_rules(self):

        logger.info('Calculation of monthly totals rule')
        def monthly_totals_rule(self, mt, pp, ca):
            ''' Monthly sums hydro. '''

            tm = self.dict_pp_tm_id[pp]

            if (tm, mt) in self.dict_month_soy:
                list_sy = self.dict_month_soy[(tm, mt)]
                return (self.erg_mt[mt, pp, ca]
                        == sum(self.pwr[sy, pp, ca] * self.weight[tm, sy]
                               for sy in list_sy))
            else:
                return po.Constraint.Skip
        self.monthly_totals = po.Constraint(self.mt, self.hyrs_ca,
                                            rule=monthly_totals_rule)

    def add_variables_rules(self):

        logger.info('Profile rule variables')
        def variables_prof_rule(self, sy, pp, ca):
            ''' Produced power equal output profile '''
            left = self.pwr[sy, pp, ca]
            return left == (self.supprof[sy, self.dict_supply_pf[(pp, ca)]]
                            * self.cap_pwr_tot[pp, ca])
        self.variables_prof = po.Constraint(self.sy_pr_ca,
                                            rule=variables_prof_rule)

    def add_ramp_rate_rules(self):

        logger.info('Calculation of ramp rates rule')
        def calc_ramp_rate_rule(self, sy, pp, ca):

            tm = self.dict_pp_tm_id[pp]

            list_sy = self.dict_tm_sy[tm]

            this_soy = sy
            last_soy = (sy - 1) if this_soy != list_sy[0] else list_sy[-1]

            return (self.pwr_ramp[sy, pp, ca]
                    == self.pwr[this_soy, pp, ca]
                     - self.pwr[last_soy, pp, ca])
        self.calc_ramp_rate = po.Constraint(self.sy_rp_ca,
                                            rule=calc_ramp_rate_rule)

        logger.info('Calculation of absolute ramp rates rule')
        def ramp_rate_abs_rule(self, sy, pp, ca):
            return (flag_abs * self.pwr_ramp[sy, pp, ca]
                    <= self.pwr_ramp_abs[sy, pp, ca])

        flag_abs = 1
        self.ramp_rate_abs_pos = po.Constraint(self.sy_rp_ca,
                                               rule=ramp_rate_abs_rule)
        flag_abs = -1
        self.ramp_rate_abs_neg = po.Constraint(self.sy_rp_ca,
                                               rule=ramp_rate_abs_rule)

    def add_energy_constraint_rules(self):
#
        logger.info('Fuel constraint rule')
        def pp_max_fuel_rule(self, nd, ca, fl):
            '''Maximum energy produced from a certain fuel in a certain pp.
               Note: this should be defined in fuels_base, not plant_fuel.'''

            # making sure we are dealing with a constrained fuel
            is_constr = fl in self.fl_erg

            # Skipping the case where erg_yr = 0 and no corresponding plants
            erg_inp_is_zero = self.erg_inp[nd, ca, fl] == 0

            plant_list = set_to_list(self.ppall_ndcafl, [None, nd, ca, fl])

            if is_constr and not erg_inp_is_zero and plant_list:
                left = sum(self.erg_fl_yr[pp, nd_1, ca_1, fl_1]
                           for (pp, nd_1, ca_1, fl_1) in plant_list)
                right = self.erg_inp[nd, ca, fl]

                return left <= right
            else:
                return po.Constraint.Skip

        self.pp_max_fuel = po.Constraint(self.ndcafl, rule=pp_max_fuel_rule)

    def add_charging_level_rules(self):

        logger.info('Storage level rule')
        def erg_store_level_rule(self, sy, pp, ca):
            ''' Charging state for storage and hydro. '''

            nd = self.mps.dict_plant_2_node_id[pp]
            fl = self.mps.dict_plant_2_fuel_id[pp]
            tm = self.dict_nd_tm_id[nd]

            list_sy = self.dict_tm_sy[tm]

            this_soy = sy
            last_soy = (sy - 1) if this_soy != list_sy[0] else list_sy[-1]

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
                           * self.weight[tm, sy]
                         ) + (
                           self.pwr_st_ch[this_soy, pp, ca]
                           * (1 - self.st_lss_rt[pp, ca])**(1/2)
                           * self.weight[tm, sy]))
            elif pp in self.setlst['hyrs'] + self.setlst['ror']:
                right += (
                          # inflowprof profiles are normalized to one!!
                          (self.inflowprof[this_soy, pp, ca]
                           * self.erg_inp[nd, ca, fl]
                          - self.pwr[sy, pp, ca]) * self.weight[tm, sy]
                         )
            return left == right

        self.erg_store_level = po.Constraint(self.sy_st_ca | self.sy_hyrs_ca
                                             | self.sy_ror_ca,
                                             rule=erg_store_level_rule)

    def add_hydro_rules(self):

        logger.info('Reservoir boundary conditions rule')
        def hy_reservoir_boundary_conditions_rule(self, sy, pp, ca):
            if (sy, pp) in [i for i in self.hyd_erg_bc.sparse_iterkeys()]:
                return (self.erg_st[sy, pp, ca]
                        == self.hyd_erg_bc[sy, pp] * self.cap_erg_tot[pp, ca])
            else:
                return po.Constraint.Skip
        self.hy_reservoir_boundary_conditions = (
                po.Constraint(self.sy_hyrs_ca,
                              rule = hy_reservoir_boundary_conditions_rule))

        logger.info('Hydro minimum monthly generation as fraction of maximum monthly inflow')
        def hy_month_min_rule(self, mt, pp, nd, ca, fl):
            return (self.erg_mt[mt, pp, ca]
                    >= self.max_erg_mt_in_share[pp]
                     * self.min_erg_mt_out_share[pp]
                     * self.erg_inp[nd, ca, fl])
        self.hy_month_min = po.Constraint(self.mt, self.hyrs_ndcafl,
                                          rule=hy_month_min_rule)

        logger.info('Hydro minimum stored energy as a fraction of energy capacitiy')
        def hy_erg_min_rule(self, sy, pp, ca):
            if not pp in [h for h in self.min_erg_share]:
                return po.Constraint.Skip
            else:
                return (self.erg_st[sy, pp, ca]
                        >=
                        self.min_erg_share[pp]
                        * self.cap_erg_tot[pp, ca])

        self.delete_component('hy_erg_min')
        self.hy_erg_min = po.Constraint(self.sy_hyrs_ca,
                                        rule=hy_erg_min_rule)

#    def add_ror_rules(self):
#
#        print('Weekly totals rule')
#        def weekly_totals_rule(self, wk, pp, ca):
#            return (self.erg_wk[wk, pp, self.mps.dict_ca_id['EL']]
#                    == sum(self.pwr[sy, pp, self.mps.dict_ca_id['EL']]
#                       * self.weight[sy]
#                    for sy in self.dict_week_soy[wk]))
#        self.weekly_totals = po.Constraint(self.wk, self.ror_ca,
#                                           rule=weekly_totals_rule)
#
#        print('Run-of-river weekly production constraint')
#        def ror_weekly_energy_rule(self, wk, pp):
#            return (self.erg_wk[wk, pp, self.mps.dict_ca_id['EL']]
#                    <= self.week_ror_output[wk, pp])
#        self.ror_weekly_energy = po.Constraint(self.wk, self.ror,
#                                               rule=ror_weekly_energy_rule)
#
#        print('Run-of-river minimum power output rule')
#        def ror_min_base_load_rule(self, sy, pp):
#            wk = self.dict_soy_week[sy]
#            return (self.erg_wk[wk, pp, self.mps.dict_ca_id['EL']]
#                    * 0.8 / self.wk_weight[wk]
#                    <= self.pwr[sy, pp, self.mps.dict_ca_id['EL']])
#        self.ror_min_base_load = po.Constraint(self.sy, self.ror,
#                                               rule=ror_min_base_load_rule)

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

    logger.info('Fuel consumption calculation rule')
    def add_yearly_cost_rules(self):
        def calc_vc_fl_pp_rule(self, pp, nd, ca, fl):

            tm = self.dict_nd_tm_id[nd]
            list_sy = self.dict_tm_sy[tm]

            sign = -1 if pp in self.setlst['sll'] else 1

            # Case 1: fuel has price profile
            if (fl, nd, ca) in {**self.dict_pricebuy_pf,
                                **self.dict_pricesll_pf}:

                pprf = (self.pricesllprof if sign is -1 else self.pricebuyprof)

                pf = (self.dict_pricesll_pf[(fl, nd, ca)] if sign is -1 else
                      self.dict_pricebuy_pf[(fl, nd, ca)])

                sums = (sign * sum(self.weight[tm, sy]
                                   * pprf[sy, pf]
                                   / self.pp_eff[pp, ca]
                                   * self.pwr[sy, pp, ca]
                                   for sy in list_sy))

            # Case 2: monthly adjustment factors have been applied to vc_fl
            elif 'vc_fl' in self.parameter_month_list:
                sums = (sign * sum(self.weight[tm, sy]
                                   * self.vc_fl[self.dict_soy_month[(tm, sy)], fl, nd]
                                   / self.pp_eff[pp, ca]
                                   * self.pwr[sy, pp, ca] for (sy, _pp, _ca)
                                   in set_to_list(self.sy_pp_ca,
                                                  [None, pp, ca])))

            # Case 3: ordinary single fuel price
            else:
                sums = (sign * self.erg_fl_yr[pp, nd, ca, fl]
                             / self.pp_eff[pp, ca]
                             * self.vc_fl[fl, nd])

            return self.vc_fl_pp_yr[pp, ca, fl] == sums

        self.calc_vc_fl_pp = po.Constraint(self.pp_ndcafl - self.lin_ndcafl,
                                           rule=calc_vc_fl_pp_rule)

        logger.info('OM VC calculation rule')
        def calc_vc_om_pp_rule(self, pp, ca):
            return (self.erg_yr[pp, ca] * self.vc_om[pp, ca]
                    == self.vc_om_pp_yr[pp, ca])
        self.calc_vc_om_pp = po.Constraint(self.ppall_ca,
                                           rule=calc_vc_om_pp_rule)


        logger.info('CO2 VC calculation rule --> OBSOLETE: Linear cost included directly in objective!!')
        def calc_vc_co2_pp_rule(self, pp, nd, ca, fl):

            tm = self.dict_nd_tm_id[nd]

            # Case 1: monthly adjustment factors have been applied to vc_fl
            if 'price_co2' in self.parameter_month_list:
                sums = sum(self.pwr[sy, pp, ca] # POWER!
                           / self.pp_eff[pp, ca] * self.weight[tm, sy]
                           * self.price_co2[mt, nd] * self.co2_int[fl]
                           for (_tm, sy, mt) in set_to_list(self.tmsy_mt,
                                                       [tm, None, None]))
            # Case 2: ordinary single CO2 price
            else:
                sums = (self.erg_fl_yr[pp, nd, ca, fl] # ENERGY!
                            / self.pp_eff[pp, ca]
                            * self.price_co2[nd] * self.co2_int[fl])

            return self.vc_co2_pp_yr[pp, ca] == sums

        self.calc_vc_co2_pp = po.Constraint(self.pp_ndcafl - self.lin_ndcafl,
                                            rule=calc_vc_co2_pp_rule)

        logger.info('Ramp VC calculation rule')
        def calc_vc_ramp_rule(self, pp, ca):

            return (self.vc_ramp_yr[pp, ca]
                    == self.pwr_ramp_yr[pp, ca] * self.vc_ramp[pp, ca])
        self.calc_vc_ramp = po.Constraint(self.rp_ca, rule=calc_vc_ramp_rule)

        logger.info('Fixed O&M cost calculation rule')
        def calc_fc_om_rule(self, pp, ca):
            return (self.fc_om_pp_yr[pp, ca]
                    == self.cap_pwr_tot[pp, ca] * self.fc_om[pp, ca])
        self.calc_fc_om = po.Constraint(self.ppall_ca, rule=calc_fc_om_rule)

        logger.info('Fixed capital cost calculation rule')
        def calc_fc_cp_rule(self, pp, ca):
            return (self.fc_cp_pp_yr[pp, ca]
                    == self.cap_pwr_new[pp, ca] * self.fc_cp_ann[pp, ca])
        self.calc_fc_cp = po.Constraint(self.add_ca, rule=calc_fc_cp_rule)

    def get_vc_fl(self):
        return \
        sum(self.pwr[sy, lin, ca]
            * self.weight[self.dict_pp_tm_id[lin], sy]
            * self.vc_fl[self.dict_soy_month[(self.dict_pp_tm_id[lin], sy)],
                         self.mps.dict_plant_2_fuel_id[lin],
                         self.mps.dict_plant_2_node_id[lin]]
            * (self.factor_lin_0[lin, ca]
               + 0.5 * self.pwr[sy, lin, ca]
                     * self.factor_lin_1[lin, ca])
            for (sy, lin, ca) in set_to_list(self.sy_lin_ca, nnnn))

    def get_vc_co(self):
        return \
        sum(self.pwr[sy, lin, ca] * self.weight[self.dict_pp_tm_id[lin], sy]
            * (self.price_co2[self.dict_soy_month[(self.dict_pp_tm_id[lin], sy)],
                              self.mps.dict_plant_2_node_id[lin]]
               if 'price_co2' in self.parameter_month_list
               else self.price_co2[self.mps.dict_plant_2_node_id[lin]])
            * self.co2_int[self.mps.dict_plant_2_fuel_id[lin]]
                * (self.factor_lin_0[lin, ca]
                   + 0.5 * self.pwr[sy, lin, ca]
                   * self.factor_lin_1[lin, ca])
            for (sy, lin, ca) in set_to_list(self.sy_lin_ca, nnnn))

    def add_objective_rules(self):
        print('Objective rule quadratic')

        def objective_rule_quad(self):
            return (# FUEL COST CONSTANT
                    sum(self.vc_fl_pp_yr[pp, ca, fl]
                        for (pp, ca, fl)
                        in set_to_list(self.pp_cafl - self.lin_cafl, nnn))
                    # FUEL COST LINEAR
                  + self.get_vc_fl()
                    # EMISSION COST LINEAR (NOTE: all fossil plants are linear)
                  + self.get_vc_co()
#                  + sum(self.vc_co2_pp_yr[pp, ca]
#                        for (pp, ca) in set_to_list(self.pp_ca, nn))
                  + sum(self.vc_om_pp_yr[pp, ca]
                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
#                  + sum(self.vc_dmnd_flex_yr[nd, ca]
#                        for (nd, ca) in set_to_list(self.ndca_EL, nn))
                  + sum(self.vc_ramp_yr[pp, ca]
                        for (pp, ca) in set_to_list(self.rp_ca, nn))
                  + sum(self.fc_om_pp_yr[pp, ca]
                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
                  + sum(self.fc_cp_pp_yr[pp, ca]
                        for (pp, ca) in set_to_list(self.add_ca, nn))
                )
        self.objective_quad = po.Objective(rule=objective_rule_quad,
                                           sense=po.minimize,
                                           doc='Quadratic objective function')


