
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
    Decorator limiting the constraints to a timemap-dependent range.

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

    def cadd(self, name, *args, objclass=po.Constraint, **kwargs):
        '''
        Add constraints or objectives to the model after logging.

        Parameters
        ----------
        name : str
            name of the new component
        objclass : pyomo class
            one of ``{po.Constraint, po.Objective}``
        args, kwargs
            passed to the ``objclass`` initialization

        '''

        ls = 'Adding {} {}: {}.'.format(objclass.__name__.lower(), name,
                                        kwargs['rule'].__doc__)
        logger.info(ls)

        obj = objclass(*args, **kwargs)
        setattr(self, name, obj)

    def add_transmission_bounds_rules(self):
        '''
        Add transmission bounds.

        .. math::

           -C_\mathrm{import} \leqslant p_{\mathrm{trm}, t}
           \leqslant C_\mathrm{export}

        .. note::
           This method modifies the ``trm`` transmission power Pyomo
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
        Adds the supply rule: Balance supply/demand in each node for teach
        time slot :math:`sy`, node :math:`nd` and, produced energy
        carrier :math:`ca`.

        * The supply side consists of the power production
          :math:`p_\mathrm{sy\_pp\_ca}` from all plants producing energy
          carrier :math:`ca`. This is reduced by the curtailed and the sold
          power. Additionally, imports or exports from connected nodes
          enter this side of the equation depending on the directionality
          definition in the combinaed set :math:`ndcnn` (see the **note**
          below).
        * The demand side consists of the exogenous demand profile, the storage
          charging power, the relevant imports and exports, depending on the
          direction definition, and the consumtion of energy carrier
          :math:`\mathrm{ca}` for the production of energy carrier
          :math:`\mathrm{ca_{out}}` (see the **note** below).

        .. math::

           & \sum_\mathrm{ppall\setminus (sell\cup curt)} p_\mathrm{sy\_pp\_ca}
               - \sum_\mathrm{sell}p_\mathrm{sy\_sell\_ca}
               - \sum_\mathrm{curt}p_\mathrm{sy\_curt\_ca} \\
           & + \sum_\mathrm{nd_2} p_\mathrm{trm, nd_2 \rightarrow nd} \\
           & = \\
           & \phi_\mathrm{dmnd, sy(nd), pf(nd, ca)} \\
           & + \sum_\mathrm{st_{nd}} p_\mathrm{sy\_st\_ca}\\
           & + \sum_\mathrm{nd_2} p_\mathrm{trm, nd \rightarrow nd_2} \\
           & + \sum_\mathrm{pp} p_\mathrm{sy\_pp\_ca_{out}} /
               \eta_\mathrm{pp\_ca} \\


        .. note::

            * **Directionality of inter-nodal transmission**: Transmission
              between nodes is expressed through the variables
              :math:`p_\mathrm{trm, symin_ndcnn}`. These variables can be
              positive or negative depending on the power flow direction.
              Since only one direction is included in the :math:`ndcnn` set,
              e.g.

              .. math::

                 & \mathrm{(nd_1=0, nd_2=1, ca=0) \in ndcnn} \\
                 & \Rightarrow \mathrm{(nd_1=1, nd_2=0, ca=0) \notin ndcnn}, \\

              they enter the supply constraint on both sides.
            * **Transmission between nodes with different time
              resolutions**: The transmission power variable
              :math:`p_\mathrm{trm}` has the higher time resolution of the two
              connected nodes. On the side of the lower time resolution node,
              it is averaged over the corresponding time slots.
            * **Consumption of produced energy carriers :math:`ca`**:
              Generators which consume an endogenously produced energy carrier
              (e.g. electricity) to produce another energy carrier (e.g. heat)
              enter the supply constraint on the demand side
              as :math:`\sum_\mathrm{pp} p_\mathrm{sy\_pp\_ca_{out}} /
              \eta_\mathrm{pp\_ca}`

        '''

        def get_transmission(sy, nd, nd_2, ca, export=True):
            '''
            If called by supply rule, the order nd, nd_2 is always the ndcnn
            order, therefore also trm order.

            Case 1: nd has higher time resolution (min) |rarr| just use
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

        def supply_rule(self, sy, nd, ca):

            list_neg = self.setlst['sll'] + self.setlst['curt']
            prod = (# power output; negative if energy selling plant
                    sum(self.pwr[sy, pp, ca]
                        * (-1 if pp in list_neg else 1)
                        for (pp, nd, ca)
                        in set_to_list(self.ppall_ndca, [None, nd, ca]))
                    # incoming inter-node transmission
                    + sum(get_transmission(sy, nd, nd_2, ca, False)
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

        self.cadd('supply', self.sy_ndca, rule=supply_rule)


    def add_energy_aggregation_rules(self):
        r'''
        Calculation of yearly totals from time slot power variables.


        * Total energy production by plant and output energy carrier

          .. math::

             & E_\mathrm{ppall\_ca}
             = \sum_\mathrm{sy} p_\mathrm{sy\_ppall\_ca} \mathrm{w_{tmsy}} \\
             & \forall \mathrm{ppall, ca} \\

        * Total absolute ramping power

        .. math::

           & \Delta p_\mathrm{rp\_ca}|
           = \sum_\mathrm{sy} |\delta p_\mathrm{sy\_rp\_ca}| \\
           & \qquad \forall \mathrm{rp, ca} \\

        * Total energy production by plant, fuel and output energy carrier

        .. math::

           & E_\mathrm{ppall\_ca}
           = \sum_\mathrm{sy} p_\mathrm{sy\_ppall\_ca} \mathrm{w_{tmsy}} \\
           & \forall \mathrm{ppall, ca} \\

        '''

        def yearly_energy_rule(self, pp, ca):
            ''' Sets variable erg_yr for fuel-consuming plants. '''

            tm = self.dict_pp_tm_id[pp]

            return (self.erg_yr[pp, ca]
                    == sum(self.pwr[sy, pp, ca] * self.weight[tm, sy]
                           for tm, sy in set_to_list(self.tmsy, [tm, None])))

        self.cadd('yearly_energy', self.ppall_ca, rule=yearly_energy_rule)

        def yearly_ramp_rule(self, pp, ca):
            ''' Yearly ramping in MW/yrm, absolute aggregated up and down. '''

            tm = self.dict_pp_tm_id[pp]
            tmsy_list = set_to_list(self.tmsy, [tm, None])

            return (self.pwr_ramp_yr[pp, ca]
                    == sum(self.pwr_ramp_abs[sy, pp, ca]
                           for tm, sy in tmsy_list))

        self.cadd('yearly_ramping', self.rp_ca, rule=yearly_ramp_rule)

        def yearly_fuel_cons_rule(self, pp, nd, ca, fl):
            ''' Fuel consumed per plant and output energy carrier. '''

            return (self.erg_fl_yr[pp, nd, ca, fl]
                    == self.erg_yr[pp, ca])
        self.cadd('yearly_fuel_cons', self.pp_ndcafl,
                  rule=yearly_fuel_cons_rule)



    def add_capacity_calculation_rules(self):
        r'''
        Constraints concerning endogenous capacity calculations.

        * The total power capacity is composed of exogenous legacy capacity,
          retirements, and new capacity investments.

          .. math::

             & P_\mathrm{tot, p\_ca} = P_\mathrm{leg, p\_ca}
             - P_\mathrm{ret, p\_ca} + P_\mathrm{new, p\_ca} \\
             & \forall \mathrm{p\in ppall}

        * The storage and reservoir energy capacity follows from the total
          power capacity and the fixed exogenous discharge duration.

          .. math::

             & C_\mathrm{tot, p,c} = P_\mathrm{tot, p,c} * \zeta_\mathrm{p,c} \\
             & \forall \mathrm{(p,c) \in st\_ca \setunion hyrs\_ca} \\

        '''



        def calc_cap_pwr_tot_rule(self, pp, ca):
            '''Calculate total power capacity (leg + add - rem).'''

            cap_tot = self.cap_pwr_leg[pp, ca]
            if (pp in self.setlst['add']):
                cap_tot += self.cap_pwr_new[pp, ca]
            if (pp in self.setlst['rem']):
                cap_tot -= self.cap_pwr_rem[pp, ca]
            return (self.cap_pwr_tot[pp, ca] == cap_tot)
        self.cadd('calc_cap_pwr_tot', self.ppall_ca,
                  rule=calc_cap_pwr_tot_rule)

        def calc_cap_erg_tot_rule(self, pp, ca):
            '''Calculate total energy capacity from total power capacity.'''

            return (self.cap_erg_tot[pp, ca]
                    == self.cap_pwr_tot[pp, ca]
                       * self.discharge_duration[pp, ca])
        self.cadd('calc_cap_erg_tot', self.st_ca | self.hyrs_ca,
                  rule=calc_cap_erg_tot_rule)

    def add_capacity_constraint_rules(self):
        
        def ppst_capac_rule(self, sy, pp, ca):
            ''' Produced power must be less than capacity. '''

            if pp in self.setlst['pp']:

                tm = self.dict_pp_tm_id[pp]
                mt = self.dict_soy_month[(tm, sy)]
                return (self.pwr[sy, pp, ca] <= self.cap_pwr_tot[pp, ca]
                                                * self.cap_avlb[mt, pp, ca])
            else:
                return (self.pwr[sy, pp, ca] <= self.cap_pwr_tot[pp, ca])

        self.cadd('ppst_capac',
                  (self.sy_pp_ca - self.sy_pr_ca) | self.sy_st_ca
                       | self.sy_hyrs_ca, rule=ppst_capac_rule)

        def st_chg_capac_rule(self, sy, pp, ca):
            ''' Charging power must be less than capacity. '''

            return (self.pwr_st_ch[sy, pp, ca]
                    <= self.cap_pwr_tot[pp, ca])

        self.cadd('st_chg_capac', self.sy_st_ca, rule=st_chg_capac_rule)

        def st_erg_capac_rule(self, sy, pp, ca):
            ''' Stored energy must be less than energy capacity. '''

            return (self.erg_st[sy, pp, ca]
                    <= self.cap_erg_tot[pp, ca])

        self.cadd('st_erg_capac', self.sy_st_ca | self.sy_hyrs_ca,
                  rule=st_erg_capac_rule)

    def add_chp_rules(self):
        r'''
        Adds all co-generation related constraints.
        
        * Certain generators need to produce power following heat demand.
          This is implemented through the node-specific normalized CHP
          profile :math:`\phi_\mathrm{chp, sy\_pf}`: The production from 
          power plants :math:`p_\mathrm{pp,t}` in the
          set :math:`\mathrm{sy\_chp\_ca}` must be larger than the scaled
          CHP profile

        :math:`\phi_\mathrm{chp,n,t}`:

        .. math::

           & p_\mathrm{pp,t} \geqslant \phi_\mathrm{chp,n,t}
           \mathrm{e}_\mathrm{chp,pp} \\
           & \forall \mathrm{(sy,pp,ca)} \in \mathrm{sy\_chp\_ca}


        '''

        def chp_prof_rule(model, sy, pp, ca):
            '''Produced power greater than CHP output profile.'''

            nd = self.mps.dict_plant_2_node_id[pp]

            return (self.pwr[sy, pp, ca]
                    >= self.chpprof[sy, nd, ca] * self.erg_chp[pp, ca])

        self.cadd('chp_prof', self.sy_chp_ca, rule=chp_prof_rule)


    def add_monthly_total_rules(self):

        def monthly_totals_rule(self, mt, pp, ca):
            '''Calculate monthly total production (hydro only). '''

            tm = self.dict_pp_tm_id[pp]

            list_sy = self.dict_month_soy[(tm, mt)]
            return (self.erg_mt[mt, pp, ca]
                    == sum(self.pwr[sy, pp, ca] * self.weight[tm, sy]
                           for sy in list_sy))

        if  len(set(self.df_tm_soy.mt_id)) == 12:
            self.cadd('monthly_totals', self.mt, self.hyrs_ca,
                      rule=monthly_totals_rule)
        else:
            logger.warning('Constraint monthly_totals: skipping. Temporal '
                           'model scope doesn\'t cover all months.')

    def add_variables_rules(self):
        '''Produced power equals output profile '''

        def variables_prof_rule(self, sy, pp, ca):
            ''' Produced power equal output profile '''
            left = self.pwr[sy, pp, ca]
            return left == (self.supprof[sy, self.dict_supply_pf[(pp, ca)]]
                            * self.cap_pwr_tot[pp, ca])

        self.cadd('variables_prof', self.sy_pr_ca, rule=variables_prof_rule)

    def add_ramp_rate_rules(self):

        def calc_ramp_rate_rule(self, sy, pp, ca):
            '''Ramp rates are power output differences.'''

            tm = self.dict_pp_tm_id[pp]

            list_sy = self.dict_tm_sy[tm]

            this_soy = sy
            last_soy = (sy - 1) if this_soy != list_sy[0] else list_sy[-1]

            return (self.pwr_ramp[sy, pp, ca]
                    == self.pwr[this_soy, pp, ca]
                     - self.pwr[last_soy, pp, ca])

        self.cadd('calc_ramp_rate', self.sy_rp_ca, rule=calc_ramp_rate_rule)


        def ramp_rate_abs_rule(self, sy, pp, ca):
            '''Standard LP absolute value constraints.'''

            return (flag_abs * self.pwr_ramp[sy, pp, ca]
                    <= self.pwr_ramp_abs[sy, pp, ca])

        flag_abs = 1
        self.cadd('ramp_rate_abs_pos', self.sy_rp_ca, rule=ramp_rate_abs_rule)
        flag_abs = -1
        self.cadd('ramp_rate_abs_neg', self.sy_rp_ca, rule=ramp_rate_abs_rule)

    def add_energy_constraint_rules(self):
#
        def pp_max_fuel_rule(self, nd, ca, fl):
            '''Constrain energy produced from certain fuels.'''

            is_constr = fl in self.fl_erg
            erg_inp_is_zero = self.erg_inp[nd, ca, fl] == 0

            ret = po.Constraint.Skip

            if is_constr and not erg_inp_is_zero:

                plant_list = set_to_list(self.ppall_ndcafl, [None, nd, ca, fl])

                if plant_list:

                    left = sum(self.erg_fl_yr[pp, nd_1, ca_1, fl_1]
                               for (pp, nd_1, ca_1, fl_1) in plant_list)
                    right = self.erg_inp[nd, ca, fl]
                    ret = left <= right

            return ret

        self.cadd('pp_max_fuel', self.ndcafl, rule=pp_max_fuel_rule)

    def add_charging_level_rules(self):

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

        self.cadd('erg_store_level',
                  self.sy_st_ca | self.sy_hyrs_ca | self.sy_ror_ca,
                  rule=erg_store_level_rule)

    def add_hydro_rules(self):

        def hy_reservoir_boundary_conditions_rule(self, sy, pp, ca):
            '''Reservoirs stored energy boundary conditions.'''

            if (sy, pp) in [i for i in self.hyd_erg_bc.sparse_iterkeys()]:
                return (self.erg_st[sy, pp, ca]
                        == self.hyd_erg_bc[sy, pp] * self.cap_erg_tot[pp, ca])
            else:
                return po.Constraint.Skip

        self.cadd('hy_reservoir_boundary_conditions', self.sy_hyrs_ca,
                  rule=hy_reservoir_boundary_conditions_rule)

        def hy_month_min_rule(self, mt, pp, nd, ca, fl):
            '''Reservoirs minimum monthlyl power production.'''

            return (self.erg_mt[mt, pp, ca]
                    >= self.max_erg_mt_in_share[pp]
                     * self.min_erg_mt_out_share[pp]
                     * self.erg_inp[nd, ca, fl])

        self.cadd('hy_month_min', self.mt, self.hyrs_ndcafl,
                  rule=hy_month_min_rule)

        def hy_erg_min_rule(self, sy, pp, ca):
            '''Reservoirs minimum filling level.'''

            if not pp in [h for h in self.min_erg_share]:
                return po.Constraint.Skip
            else:
                return (self.erg_st[sy, pp, ca]
                        >=
                        self.min_erg_share[pp]
                        * self.cap_erg_tot[pp, ca])

        self.cadd('hy_erg_min', self.sy_hyrs_ca, rule=hy_erg_min_rule)

    def add_yearly_cost_rules(self):
        '''
        Groups the yearly cost calculation constraints:

        * Fuel cost of plants with constant supply curves
        .. math::
           \mathrm{vc}_\mathrm{fl, ppall\_cafl\setminus lin\_cafl}


        *

        .. note::
           The fuel and emission costs of power plants with linear supply
           curves must be calculated directly in the objective function.
           This is because CPLEX only supports a quadratic objective, not
           quadratic constraints.


        '''


        def calc_vc_fl_pp_rule(self, pp, nd, ca, fl):
            '''Yearly fuel cost calculation (constant supply curve plants).'''

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

        self.cadd('calc_vc_fl_pp', self.pp_ndcafl - self.lin_ndcafl,
                  rule=calc_vc_fl_pp_rule)

        def calc_vc_om_pp_rule(self, pp, ca):
            '''Yearly variable O&M cost calculation rule.'''

            return (self.erg_yr[pp, ca] * self.vc_om[pp, ca]
                    == self.vc_om_pp_yr[pp, ca])

        self.cadd('calc_vc_om_pp', self.ppall_ca, rule=calc_vc_om_pp_rule)


        def calc_vc_co2_pp_rule(self, pp, nd, ca, fl):
            '''Yearly emission ncost calculation (constant supply curves).'''

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

        self.cadd('calc_vc_co2_pp', self.pp_ndcafl - self.lin_ndcafl,
                  rule=calc_vc_co2_pp_rule)

        def calc_vc_ramp_rule(self, pp, ca):
            '''Yearly ramping variable cost calculation rule.'''

            return (self.vc_ramp_yr[pp, ca]
                    == self.pwr_ramp_yr[pp, ca] * self.vc_ramp[pp, ca])

        self.cadd('calc_vc_ramp', self.rp_ca, rule=calc_vc_ramp_rule)

        def calc_fc_om_rule(self, pp, ca):
            '''Fixed O&M cost calculation rule.'''

            return (self.fc_om_pp_yr[pp, ca]
                    == self.cap_pwr_tot[pp, ca] * self.fc_om[pp, ca])

        self.cadd('calc_fc_om', self.ppall_ca, rule=calc_fc_om_rule)

        def calc_fc_cp_rule(self, pp, ca):
            '''Fixed capital cost calculation rule'''

            return (self.fc_cp_pp_yr[pp, ca]
                    == self.cap_pwr_new[pp, ca] * self.fc_cp_ann[pp, ca])

        self.cadd('calc_fc_cp', self.add_ca, rule=calc_fc_cp_rule)

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
        '''
        Auxiliary methods to generate the

        '''

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
        ''' Quadratic objective function.

        The objective function to be minimized is the sum over all system
        costs:



        * Fuel costs of power plants with constant supply curve:

        .. math::
            \sum_\mathrm{ppall\setminus lin} \mathrm{vc}_\mathrm{fl, ppall\_cafl\setminus lin\_cafl}

        * Variable O\&M costs of power plants:

        .. math::
            \sum_\mathrm{ppall,ca} \mathrm{vc}_\mathrm{om, ppall\_ca}


        .. note::
           The variable fuel and emission cost terms of the power plants with
           linear supply curves

           * :math:`\sum_\mathrm{lin,ca}\mathrm{vc_{fl\_lin\_ca}} =`
           * :math:`\sum_\mathrm{lin,ca}\mathrm{vc_{em\_lin\_ca}} =`

           are not model variables but calculated directly in the auxiliary methods
           :func:`get_vc_fl` and :func:`get_vc_fl`. See the note in the
           :func:`add_yearly_cost_rules` method documentation.

        '''

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
                        for (pp, ca) in set_to_list(self.add_ca, nn)))

        self.cadd('objective_quad', rule=objective_rule_quad,
                  sense=po.minimize, objclass=po.Objective)


