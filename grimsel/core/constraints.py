'''
Model constraints
===================

All model constraints are defined in the :class:`Constraints` class, which
serves as a mixin class to :class:`grimsel.core.model_base.ModelBase`.

'''


from functools import wraps

import pyomo.environ as po

from grimsel.core.io import IO
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
    Mixin class containing all constraints, included in the
    :class:`grimsel.core.model_base.ModelBase`
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
        r'''
        Add transmission bounds.

        .. math::

           & -P_\mathrm{imp,m,n,n_2,c} \leqslant p_\mathrm{trm,t,n,n_2,c}
           \leqslant P_\mathrm{exp,m,n,n_2,c} \\
           & \forall \mathrm{(t,n,n_2,c) \in symin\_ndcnn} \\

        .. note::
           This method modifies the ``trm`` transmission power Pyomo
           variable object by calling its ``setub`` and ``setlb`` methods.

        '''

        dict_weight = IO.param_to_df(self.nd_weight).set_index('nd_id').value.to_dict()

        if hasattr(self, 'trm'):
            for sy, nd1, nd2, ca in self.trm:

                # get node with max weight
                nd_w = max([nd1, nd2], key=lambda x: dict_weight[x])

                tm = self.dict_ndnd_tm_id[nd1, nd2]

                mt = self.dict_soy_month[(tm, sy)]

                ub = (self.cap_trme_leg[mt, nd1, nd2, ca]
                      * self.nd_weight[nd_w])
                lb = - (self.cap_trmi_leg[mt, nd1, nd2, ca]
                        * self.nd_weight[nd_w])

                self.trm[(sy, nd1, nd2, ca)].setub(ub)
                self.trm[(sy, nd1, nd2, ca)].setlb(lb)

    def add_supply_rules(self):
        r'''
        Adds the supply rule: Balance supply/demand in each node for each
        time slot :math:`\mathrm{t}`, node :math:`\mathrm{n}` and,
        produced energy carrier :math:`\mathrm{c}`.

        * The supply side consists of the power production
          :math:`p_\mathrm{t,p,c}` from all plants producing energy
          carrier :math:`\mathrm{c}`. This is reduced by the curtailed
          and the sold power. Additionally, imports or exports from
          connected nodes enter this side of the equation depending on the
          directionality definition in the combined set :math:`\mathrm{ndcnn}`
          (see the **note** below).
        * The demand side consists of the exogenous demand profile, the storage
          charging power, the relevant imports and exports, depending on the
          direction definition, and the consumption of energy carrier
          :math:`\mathrm{c}` for the production of other energy carriers
          :math:`\mathrm{c_{out}}`.

        .. math::

           & \sum_\mathrm{ppall\setminus (sell\cup curt)} p_\mathrm{sy\_pp\_ca}
               - \sum_\mathrm{sell}p_\mathrm{sy\_sell\_ca}
               - \sum_\mathrm{curt}p_\mathrm{sy\_curt\_ca} \\
           & + \sum_\mathrm{nd_2} p_\mathrm{trm, nd_2 \rightarrow nd} \\
           & = \phi_\mathrm{dmd, t, n, c} \\
           & + \sum_\mathrm{p \in st_n} p_\mathrm{chg,t,p,c}\\
           & + \sum_\mathrm{nd_2} p_\mathrm{trm, nd \rightarrow nd_2} \\
           & + \sum_\mathrm{(p, n, c_{out}, c)\in pp\_ndcaca}
               p_\mathrm{t,p,c_{out}} / \eta_\mathrm{p,c_{out}} \\
           & \forall \mathrm{(t,n,c) \in sy\_nd\_ca} \\

        .. note::

            * **Directionality of inter-nodal transmission**: Transmission
              between nodes is expressed through the variables
              :math:`p_\mathrm{trm, symin_ndcnn}`. These variables can be
              positive or negative depending on the power flow direction.
              Since only one direction is included in the
              :math:`\mathrm{ndcnn}` set, e.g.

              .. math::

                 & \mathrm{(nd_1=0, nd_2=1, ca=0) \in ndcnn} \\
                 & \Rightarrow \mathrm{(nd_1=1, nd_2=0, ca=0) \notin ndcnn}, \\

              they enter the supply constraint on both sides.
            * **Transmission between nodes with different time
              resolutions**: The transmission power variable
              :math:`p_\mathrm{trm}` has the higher time resolution of the two
              connected nodes. On the side of the lower time resolution node,
              it is averaged over the corresponding time slots.
            * **Consumption of produced energy carriers** :math:`\mathrm{ca}`:
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

            * **Case 1**: ``nd`` has higher time resolution (min) |rarr| just
              use ``trm[tm, sy, nd, nd_2, ca]``
            * **Case 2**: ``nd`` has lower time resolution (not min) |rarr|
              average ``avg(trm[tm, sy_2, nd, nd_2, ca])`` for all ``sy_2``
              defined by the
               ``grimsel.core.model_base.ModelBase.dict_sysy[nd, nd_2, sy]``

            Parameters
            ----------
            sy : int
                current time slot in nd
            nd : int
                outgoing node
            nd_2 : int
                incoming node
            ca : int
                energy carrier
            export : bool
                True if export else False

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
            ''' Balance supply/demand '''

            list_neg = self.sll | self.curt
            prod = (# power output; negative if energy selling plant
                    sum(self.pwr[sy, pp, ca]
                        * (-1 if pp in list_neg else 1)
                        for (pp, nd, ca)
                        in set_to_list(self.ppall_ndca, [None, nd, ca]))
                    # incoming inter-node transmission
                    + sum(get_transmission(sy, nd, nd_2, ca, False)
                          / self.nd_weight[nd_2]
                          for (nd, nd_2, ca)
                          in set_to_list(self.ndcnn, [None, nd, ca]))
                   )
            exports = sum(get_transmission(sy, nd, nd_2, ca, True)
                          / self.nd_weight[nd]
                          for (nd, nd_2, ca)
                          in set_to_list(self.ndcnn, [nd, None, ca]))
            dmnd = (self.dmnd[sy, nd, ca]
                    + sum(self.pwr_st_ch[sy, st, ca] for (st, nd, ca)
                          in set_to_list(self.st_ndca, [None, nd, ca])))


            # demand of plants using ca as an input
            ca_cons = (po.ZeroConstant if not self.pp_ndcaca else
                       sum(self.pwr[sy, pp, ca_out] / self.pp_eff[pp, ca_out]
                           for (pp, nd, ca_out, ca)
                           in set_to_list(self.pp_ndcaca,
                                          [None, nd, None, ca])))
            gl = self.grid_losses[nd, ca]

            return prod == (dmnd + ca_cons) * (1 + gl) + exports

        self.cadd('supply', self.sy_ndca, rule=supply_rule)

    def add_energy_aggregation_rules(self):
        r'''
        Calculation of yearly totals from time slot power variables.

        * Total energy production by plant and output energy carrier:

          .. math::

             & E_\mathrm{p,c}
             = \sum_\mathrm{t} p_\mathrm{t,p,c} {w_\mathrm{\tau,t}} \\
             & \forall \mathrm{(p, c) \in ppall\_ca} \\

        * Total absolute ramping power:

        .. math::

           & |\Delta p_\mathrm{p,c}|
           = \sum_\mathrm{t} |\delta p_\mathrm{t,p,c}| \\
           & \forall \mathrm{(p, c) \in rp\_ca} \\

        * Total energy production by plant, fuel and output energy carrier:

          .. math::

             & E_\mathrm{p,n,c,f} = E_\mathrm{p,c} \\
             & \forall \mathrm{(p, n,c,f) \in pp\_ndcafl} \\

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
          optimized retirements, and optimized newly installed capacity:

          .. math::

             & P_\mathrm{tot, p, c} = P_\mathrm{leg, p, c}
             - P_\mathrm{ret, p, c} + P_\mathrm{new, p, c} \\
             & \forall \mathrm{(p,c)\in ppall\_ca}

        * The storage and reservoir energy capacity follows from the total
          power capacity and the fixed exogenous discharge duration.

          .. math::

             & C_\mathrm{tot, p,c} = P_\mathrm{tot, p,c} \zeta_\mathrm{p,c} \\
             & \forall \mathrm{(p,c) \in st\_ca \cup hyrs\_ca} \\

        '''



        def calc_cap_pwr_tot_rule(self, pp, ca):
            '''Calculate total power capacity (leg + add - rem).'''

            cap_tot = self.cap_pwr_leg[pp, ca]
            if pp in self.add:
                cap_tot += self.cap_pwr_new[pp, ca]
            if pp in self.rem:
                cap_tot -= self.cap_pwr_rem[pp, ca]
            return self.cap_pwr_tot[pp, ca] == cap_tot
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
        r'''
        Power and stored energy are constrained by the respective capacities:

        * Output power of generators (note: variable renewables with
          exogenous supply (i.e. capacity factor) profile are explicitly
          excluded):

          .. math::

             & P_\mathrm{tot, p, c} \geqslant p_\mathrm{t, p, c} \\
             & \forall \mathrm{(t, p, c) \in
             (sy\_pp\_ca \setminus sy\_pr\_ca)
             \cup sy\_st\_ca \cup sy\_hyrs\_ca} \\

        * Charging power of storage assets:

          .. math::

             & P_\mathrm{tot, p, c} \geqslant p_\mathrm{chg,t,p,c} \\
             & \forall \mathrm{(t, p, c) \in sy\_st\_ca} \\

        * Stored energy of storage assets:

          .. math::

             & C_\mathrm{tot, p, c} \geqslant e_\mathrm{t,p,c} \\
             & \forall \mathrm{(t, p, c) \in sy\_st\_ca\cup sy\_hyrs\_ca} \\


        '''

        def ppst_capac_rule(self, sy, pp, ca):
            ''' Produced power must be less than capacity. '''

            if pp in self.setlst['pp'] and has_cap_avlb:

                tm = self.dict_pp_tm_id[pp]
                mt = self.dict_soy_month[(tm, sy)]

                cap_avlb = (self.cap_avlb[mt, pp, ca]
                            if self.dict_par['vc_fl'].has_monthly_factors
                            else self.cap_avlb[pp, ca])

                return (self.pwr[sy, pp, ca] <= self.cap_pwr_tot[pp, ca]
                                                * cap_avlb)
            else:
                return (self.pwr[sy, pp, ca] <= self.cap_pwr_tot[pp, ca])

        has_cap_avlb = hasattr(self, 'cap_avlb')
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

        def pwr_pot_add_rule(self, pp, ca):
            ''' Capcity added + legacy must be less than potential capacity '''

            return (self.cap_pwr_tot[pp, ca]
                    <= self.pwr_pot[pp, ca])

        self.cadd('pwr_pot_add', self.add_ca, rule=pwr_pot_add_rule)

    def add_chp_rules(self):
        r'''
        Adds all co-generation related constraints.

        * Certain generators need to produce power following heat demand.
          This is implemented through the node-specific normalized CHP
          profile :math:`\phi_\mathrm{chp, sy\_pf}`: The production from
          power plants must be larger than the scaled CHP profile:

        .. math::

           & p_\mathrm{t,p,c} \geqslant \phi_\mathrm{chp,t,n,c}
           \mathrm{e}_\mathrm{chp,p,c} \\
           & \forall \mathrm{(t,p,c)} \in \mathrm{sy\_chp\_ca}


        '''

        def chp_prof_rule(model, sy, pp, ca):
            '''Produced power greater than CHP output profile.'''

            nd = self.mps.dict_plant_2_node_id[pp]

            return (self.pwr[sy, pp, ca]
                    >= self.chpprof[sy, nd, ca] * self.erg_chp[pp, ca])

        self.cadd('chp_prof', self.sy_chp_ca, rule=chp_prof_rule)


    def add_monthly_total_rules(self):
        r'''
        Adds the ``monthly_totals`` constraint which sets the monthly energy
        production variables:

        .. math::

           & E_\mathrm{m,p,c} = \sum_\mathrm{t\in sy_m} w_\mathrm{\tau(p),t}
           p_\mathrm{t,p,c} \\
           & \forall \mathrm{(m,p,c) \in mt \times hyrs\_ca} \\

        .. note::

           * :math:`\mathrm{sy_m}` is an ad-hoc set defining the time slots
             :math:`\mathrm{sy}` for any given month :math:`\mathrm{m}`.
           * The resulting monthly totals :math:`E_\mathrm{m,p,c}` are
             primarily used for hydro power constraints.

        '''


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
        r'''
        Produced power equals output profile.

        Variable renewable energy generators produce at a given output profile
        (capacity factor per time slot).

        .. note::
            Curtailments are included at the system level through dedicated
            technologies (set :math:`\mathrm{curt \subset ppall}`).

        .. math::

           & p_\mathrm{t,p,c} = \Phi_\mathrm{supply,t,p,c} P_\mathrm{tot,p,c}\\
           & \forall \mathrm{(t,p,c) \in sy\_pr\_ca}

        '''

        def variables_prof_rule(self, sy, pp, ca):
            ''' Produced power equal output profile '''
            left = self.pwr[sy, pp, ca]
            return left == (self.supprof[sy, pp, ca]
                            * self.cap_pwr_tot[pp, ca])

        self.cadd('variables_prof', self.sy_pr_ca, rule=variables_prof_rule)

    def add_ramp_rate_rules(self):
        r'''
        Three constraints are used to obtain the absolute hourly ramp rates
        :math:`|\delta_\mathrm{sy\_rp\_ca}|`:

        * Difference of power production (as in the
          :func:`add_charging_level_rules` constraint time is circular):

          .. math::

             & \delta_\mathrm{t,p,c} = p_\mathrm{t,p,c} - p_\mathrm{t - 1,p,c} \\
             & \forall \mathrm{(t,p,c) \in sy\_rp\_ca} \\

        * Two constraints to calculate the absolute values:

          .. math::

             & +1 \cdot \delta_\mathrm{t,p,c} \leqslant |\delta_\mathrm{t,p,c}| \\
             & -1 \cdot \delta_\mathrm{t,p,c} \leqslant |\delta_\mathrm{t,p,c}| \\
             & \forall \mathrm{(t,p,c) \in sy\_rp\_ca} \\

        '''


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
        r'''
        Adds the ``pp_max_fuel`` constraint which limits the amount of
        output energy produced from certain fuels to the
        value of the exogenous parameter :math:`E_\mathrm{inp,n,c,f}`:

        .. math::

            & \sum_\mathrm{p \in ppall} E_\mathrm{p,n,c,f}
            \leqslant E_\mathrm{inp,n,c,f} \\
            & \forall \mathrm{(n,c,f) \in ndcafl} \\

        '''

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

        if hasattr(self, 'erg_inp'):
            self.cadd('pp_max_fuel', self.ndcafl, rule=pp_max_fuel_rule)

    def add_charging_level_rules(self):
        r'''
        Adds the constraint determining the stored energy: The energy in the
        current time slot :math:`e_\mathrm{t,p,c}` is equal what's left from
        the last time slot :math:`e_\mathrm{t-1,p,c}` plus

        * inflow minus production in the case of reservoirs and run-of-river
          plants
        * charging minus discharging in the case of pure storage plants
          without inflow

        Time is circular, i.e. the first time slot follows after the last.

        .. math::

           e_\mathrm{t,p,c} = e_\mathrm{t-t,p,c} +
           \begin{cases}
           (\phi_\mathrm{inflow,t,p} e_\mathrm{inp,n(p),c,f(p)} - p_\mathrm{t, p, c}) w_\mathrm{t, n(p)} \\
           \qquad \qquad \forall \mathrm{(t,p,c)\in sy\_hyrs\_ca \cup sy\_ror\_ca}\\
           (\eta_\mathrm{p,c}^{1/2} p_\mathrm{chg, t, p, c} - \eta_\mathrm{p,c}^{-1/2} p_\mathrm{t, p, c}) w_\mathrm{t} \\
           \qquad \qquad \forall \mathrm{(t,p,c)\in sy\_st\_ca}\\
           \end{cases}

        '''

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
        r'''
        Various rules constraining the operation of the hydro reservoir plants.

        * The ``hy_reservoir_boundary_conditions`` constraint requires
          the reservoir filling level to assume an exogenously defined share
          of the energy capacity during certain time slots:

          .. math::

             & \rho_\mathrm{t,p,c}
             = e_\mathrm{hyd\_bc,p,c} C_\mathrm{tot, p, c} \\
             & \forall \mathrm{(t,p,c) \in sy\_hydbc \times hyrs\_ca} \\

        * The ``hy_month_min`` constraint forces a certain minimum output
          production from hydro reservoirs each month. This minimum is
          expressed as a share of maximum monthly inflow (which itself is
          a share :math:`\rho_\mathrm{max\_erg\_in,p}` of the total yearly
          inflow energy :math:`e_\mathrm{inp,n,c,f}`):

          .. math::

             E_\mathrm{m, p, c} \geqslant & \rho_\mathrm{max\_erg\_in,p}\\
                                & \cdot \rho_\mathrm{min\_erg\_out,p}\\
                                & \cdot e_\mathrm{inp,n(p),c,f(p)}\\

        * The ``hy_erg_min`` constraint sets a lower bound on the stored
          energy as a fraction of the total energy capacity.

          .. math::

             e_\mathrm{t,p,c} \geqslant & C_\mathrm{tot,p,c} \\
                                        & \cdot \rho_\mathrm{min\_cap,p} \\
                                        & \forall \mathrm{(t,p,c)
                                        \in sy\_hyrs\_ca}

        '''


        def hy_reservoir_boundary_conditions_rule(self, sy, pp, ca):
            '''Reservoirs stored energy boundary conditions.'''

            if (sy, pp) in [i for i in self.hyd_erg_bc.sparse_iterkeys()]:
                return (self.erg_st[sy, pp, ca]
                        == self.hyd_erg_bc[sy, pp]
                           * self.cap_erg_tot[pp, ca])
            else:
                return po.Constraint.Skip

        if hasattr(self, 'hyd_erg_bc'):
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
        r'''
        Groups the yearly cost calculation constraints:

        * Variable fuel costs :math:`c_\mathrm{fuel,p,c,f}` of
          plants with constant supply curves:

          - For fixed fuel costs :math:`\mathrm{vc}_\mathrm{fuel,fl\_nd}`:

             .. math::

                & c_\mathrm{fuel,p,c,f}
                = E_\mathrm{p,n,c,f} \mathrm{vc}_\mathrm{f,n} \\
                & \forall \mathrm{(p,n,c,f)\in ppall\_ndcafl
                                            \setminus lin\_ndcafl} \\

          - For monthly fuel costs :math:`vc_\mathrm{m,d,n}`:

             .. math::

                & c_\mathrm{fuel,p,c,f}
                = \sum_\mathrm{t\in sy\_pp\_ca|_{p,c}} \mathrm{vc_{m(t),f,n}}
                  p_\mathrm{t,p,c} / \eta_\mathrm{p,c}\\
                & \forall \mathrm{(p,n,c,f)\in ppall\_ndcafl
                                             \setminus lin\_ndcafl} \\

          - For plants with cost profiles:

              .. math::

                & c_\mathrm{fuel,p,c,f}
                = \sum_\mathrm{t\in sy\_ppall\_ca|_{p,c}}
                p_\mathrm{t,p,c} / \eta_\mathrm{p,c}
                w_\mathrm{\tau(p),t} \cdot
                \begin{cases}
                -1 \cdot \Phi_\mathrm{psll,t,\phi(p)}
                    & \text{if } \mathrm{p\in sll} \\
                +1 \cdot \Phi_\mathrm{pbuy,t,\phi(p)}
                    & \text{if } \mathrm{p\notin sll} \\
                \end{cases} \\
                & \forall \mathrm{(p,n,c,f)
                \in ppall\_ndcafl \setminus lin\_ndcafl} \\

        * Variable operation and maintenance costs
          :math:`\mathrm{vc}_\mathrm{om,p,c}` for all plants:

             .. math::

                & c_\mathrm{om_v,p,c}
                = E_\mathrm{p,c} \mathrm{vc}_\mathrm{om,p,c} \\
                & \forall \mathrm{(p,c)\in ppall\_ca} \\

        * Variable |CO2|  emission costs: Same as variable fuel costs
          (fixed fuel costs and monthly fuel costs cases), but with
          specific cost
           :math:`\pi_\mathrm{CO_2,[m],n}\cdot i_\mathrm{CO_2, f}`.

        * Total ramping cost:

          .. math::

             & c_\mathrm{rp,p,c}
             = |\Delta_\mathrm{p,c}| \mathrm{vc}_\mathrm{ramp,p,c} \\
             & \forall \mathrm{(p,c)\in rp\_ca} \\

        * Total fixed O&M cost (only relevant for plants with capacity
          additions and retirements):

          .. math::

             & c_\mathrm{om_f,p,c}
             = P_\mathrm{tot,p,c} \mathrm{fc}_\mathrm{om,p,c} \\
             & \forall \mathrm{(p,c)\in add\_ca \cup rem\_ca} \\


        * Total annualized fixed investment cost:

          .. math::

             & c_\mathrm{cp,p,c}
             = P_\mathrm{new,p,c} {c}_\mathrm{cp,p,c} \\
             & \forall \mathrm{(p,c)\in add\_ca} \\


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

            sign = -1 if pp in self.sll else 1

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
                                   for (sy, _, _)
                                   in set_to_list(self.sy_pp_ca,
                                                  [None, pp, ca])))

            # Case 2: monthly adjustment factors have been applied to vc_fl
            elif self.dict_par['vc_fl'].has_monthly_factors:
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

        if hasattr(self, 'vc_om'):
            self.cadd('calc_vc_om_pp', self.ppall_ca, rule=calc_vc_om_pp_rule)


        def calc_vc_co2_pp_rule(self, pp, nd, ca, fl):
            '''Yearly emission ncost calculation (constant supply curves).'''

            tm = self.dict_nd_tm_id[nd]

            # Case 1: monthly adjustment factors have been applied to vc_fl
            if self.dict_par['price_co2'].has_monthly_factors:
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

        if hasattr(self, 'vc_ramp'):
            self.cadd('calc_vc_ramp', self.rp_ca, rule=calc_vc_ramp_rule)

        def calc_fc_om_rule(self, pp, ca):
            '''Fixed O&M cost calculation rule.'''

            return (self.fc_om_pp_yr[pp, ca]
                    == self.cap_pwr_tot[pp, ca] * self.fc_om[pp, ca])

        if hasattr(self, 'fc_om'):
            self.cadd('calc_fc_om', self.add_ca | self.rem_ca,
                      rule=calc_fc_om_rule)

        def calc_fc_cp_rule(self, pp, ca):
            '''Fixed capital cost calculation rule'''

            return (self.fc_cp_pp_yr[pp, ca]
                    == self.cap_pwr_new[pp, ca] * self.fc_cp_ann[pp, ca])

        self.cadd('calc_fc_cp', self.add_ca, rule=calc_fc_cp_rule)

    def add_objective_rules(self):
        ''' Quadratic objective function.


        The objective function to be minimized is the sum of all system
        costs:

        * Fuel costs of power plants with constant efficiency/supply curve:

        .. math::
            \sum_\mathrm{(p,c,f) \in pp\_cafl \setminus lin\_cafl}
                    c_\mathrm{fuel,p,c,f}

        * |CO2| emission costs of power plants with constant efficiency:

        .. math::
            \sum_\mathrm{(p,c) \in pp\_ca \setminus lin\_ca} c_\mathrm{em,p,c}

        * Variable O\&M costs of power plants:

        .. math::
            \sum_\mathrm{(p,c) \in ppall\_ca} c_\mathrm{om_v,p,c}

        * Fixed O\&M costs of power plants:

        .. math::
            \sum_\mathrm{(p,c) \in ppall\_ca} c_\mathrm{om_f,p,c}


        * Variable ramping costs of power plants:

        .. math::
            \sum_\mathrm{(p,c) \in rp\_ca} c_\mathrm{rp,p,c}

        * Fixed investment costs of power plants:

          .. math::
             \sum_\mathrm{(p,c) \in add\_ca} c_\mathrm{cp,p,c}

        * Variable fuel costs for plants with linear supply curves are
          calculated in the method :func:`get_vc_fl`.

        * Variable |CO2| emission costs for plants with linear supply curves
          are calculated in the method :func:`get_vc_co`.

        .. note::
           The variable fuel and emission cost terms of the power plants with
           linear supply curves are not model variables but calculated directly
           in the auxiliary methods
           :func:`get_vc_fl` and :func:`get_vc_fl`. See the note in the
           :func:`add_yearly_cost_rules` method documentation.

        '''

        def objective_rule_quad(self):

            return (# FUEL COST CONSTANT
                    sum(self.vc_fl_pp_yr[pp, ca, fl]
                        * self.nd_weight[self.mps.dict_plant_2_node_id[pp]]
                        for (pp, ca, fl)
                        in set_to_list(self.pp_cafl - self.lin_cafl, nnn))
                    # FUEL COST LINEAR
                  + self.get_vc_fl()
                    # EMISSION COST LINEAR
                  + self.get_vc_co()
                  + sum(self.vc_co2_pp_yr[pp, ca]
                        * self.nd_weight[self.mps.dict_plant_2_node_id[pp]]
                        for (pp, ca) in set_to_list(self.pp_ca - self.lin_ca, nn))
                  + sum(self.vc_om_pp_yr[pp, ca]
                        * self.nd_weight[self.mps.dict_plant_2_node_id[pp]]
                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
                  + sum(self.vc_ramp_yr[pp, ca]
                        * self.nd_weight[self.mps.dict_plant_2_node_id[pp]]
                        for (pp, ca) in set_to_list(self.rp_ca, nn))
                  + sum(self.fc_om_pp_yr[pp, ca]
                        * self.nd_weight[self.mps.dict_plant_2_node_id[pp]]
                        for (pp, ca) in set_to_list(self.ppall_ca, nn))
                  + sum(self.fc_cp_pp_yr[pp, ca]
                        * self.nd_weight[self.mps.dict_plant_2_node_id[pp]]
                        for (pp, ca) in set_to_list(self.add_ca, nn)))

        self.cadd('objective_quad', rule=objective_rule_quad,
                  sense=po.minimize, objclass=po.Objective)

# %%
#
#plin = ml.io.modwr.dict_comp_obj['pwr'].get_df().rename(columns={'value': 'pwr'})#.query('pp_id in %s'%ml.m.setlst['lin'])
#
#plin['fl_id'] = plin.pp_id.replace(ml.m.mps.dict_plant_2_fuel_id)
#plin['nd_id'] = plin.pp_id.replace(ml.m.mps.dict_plant_2_node_id)
#plin['tm_id'] = plin.nd_id.replace(ml.m.dict_nd_tm_id)
#
#plin = plin.join(ml.m.df_def_plant.set_index(['pp_id'])[['set_def_lin', 'set_def_sll']], on=['pp_id'])
#
#plin = plin.join(ml.m.df_tm_soy.set_index(['tm_id', 'sy'])[['mt_id', 'weight']], on=['tm_id', 'sy'])
#plin = plin.join(ml.io.modwr.dict_comp_obj['vc_fl'].get_df().set_index(['mt_id', 'fl_id', 'nd_id']).value.rename('vc_fl'), on=['mt_id', 'fl_id', 'nd_id'])
#plin = plin.join(ml.io.modwr.dict_comp_obj['vc_om'].get_df().set_index(['pp_id']).value.rename('vc_om'), on=['pp_id'])
#
#plin = plin.join(ml.io.modwr.dict_comp_obj['factor_lin_0'].get_df().set_index(['pp_id']).value.rename('f0'), on=['pp_id'])
#plin = plin.join(ml.io.modwr.dict_comp_obj['factor_lin_1'].get_df().set_index(['pp_id']).value.rename('f1'), on=['pp_id'])
#plin = plin.join(ml.io.modwr.dict_comp_obj['nd_weight'].get_df().set_index(['nd_id']).value.rename('nd_weight'), on=['nd_id'])
#plin = plin.join(ml.io.modwr.dict_comp_obj['pp_eff'].get_df().set_index(['pp_id']).value.rename('pp_eff'), on=['pp_id'])
#plin = plin.join(ml.io.modwr.dict_comp_obj['co2_int'].get_df().set_index(['fl_id']).value.rename('co2_int'), on=['fl_id'])
#plin = plin.join(ml.io.modwr.dict_comp_obj['price_co2'].get_df().set_index(['nd_id', 'mt_id']).value.rename('price_co2'), on=['nd_id', 'mt_id'])
#
#prfsll = ml.io.modwr.dict_comp_obj['pricesllprof'].get_df()
#prfsll['fl_id'] = prfsll.pf_id.replace({v: k[0] for k, v in ml.m.dict_pricesll_pf.items()})
#prfbuy = ml.io.modwr.dict_comp_obj['pricebuyprof'].get_df()
#prfbuy['fl_id'] = prfbuy.pf_id.replace({v: k[0] for k, v in ml.m.dict_pricebuy_pf.items()})
#
#plin = plin.join(prfsll.set_index(['sy', 'fl_id']).value.rename('prfsll'), on=['sy', 'fl_id'])
#plin = plin.join(prfbuy.set_index(['sy', 'fl_id']).value.rename('prfbuy'), on=['sy', 'fl_id'])
#
#plin = ml.m.mps.id_to_name(plin)
#
#plin['vcom'] = plin.eval('pwr * weight * vc_om * nd_weight')
#
#plin['c_trm'] = 1e4
#plin['c_trm']  = plin.c_trm.where(plin.fl == 'electricity', 0)
#plin['CC_trm'] = (plin.pwr**2) * plin.nd_weight * plin.c_trm
#
#plin.loc[plin.fl == 'electricity']
#
##plin.assign(pp_id=lambda x: x.pp_id.astype(np.int)).groupby('pp_id').sum().reset_index().join(dfan.set_index('pp_id'), on='pp_id').set_index('pp_id').iloc[:,-2:].reset_index().assign(dff=lambda x: x.diff(axis=1).iloc[:, -1], pp=lambda x: x.pp_id.replace(ml.m.mps.dict_pp)).sort_values('dff').dff
#
#VC_FL_L = plin.query('set_def_lin == 1').eval('pwr * weight * vc_fl * nd_weight * (f0 + 0.5 * pwr * f1)').sum()
#VC_FL_C = plin.query('set_def_lin == 0').eval('pwr * weight * vc_fl * nd_weight / pp_eff').sum()
#VC_CO_L = plin.query('set_def_lin == 1').eval('pwr * weight * co2_int * price_co2 * nd_weight * (f0 + 0.5 * pwr * f1)').sum()
#VC_CO_C = plin.query('set_def_lin == 0').eval('pwr * weight * co2_int * price_co2 * nd_weight / pp_eff').sum()
#VC_OM = plin.eval('pwr * weight * vc_om * nd_weight').sum()
#VC_RAMP = ml.io.modwr.dict_comp_obj['vc_ramp_yr'].get_df().value.sum()
#VC_TRM = plin.query('fl == "electricity"').CC_trm.sum()
#
#VC_SLL_HH = - plin.query('fl == "electricity" and set_def_sll == 1').eval('nd_weight * weight * pwr * prfsll').sum()
#VC_BUY_HH = plin.query('fl == "electricity" and set_def_sll == 0').eval('nd_weight * weight * pwr * prfbuy').sum()
#
#VC_TRM + (VC_FL_L + VC_FL_C + VC_CO_L + VC_CO_C + VC_OM + VC_RAMP + VC_SLL_HH + VC_BUY_HH)
#
#ml.m.objective_value

# %%


    def get_vc_fl(self):
        r'''
        Get total fuel cost calculated directly from power production:

        .. math::
           \sum_\mathrm{(t,p,c)\in sy\_lin\_ca}
           p_\mathrm{t,p,c} w_\mathrm{\tau(p),t}
           \cdot \mathrm{vc_{f(p),n(p)}}
           \cdot (f_\mathrm{0,p,c} + 0.5 p_\mathrm{t,p,c} f_\mathrm{1,p,c})

        '''


        def spec_vc_fl(lin, sy):

            fl_id, nd_id = (self.mps.dict_plant_2_fuel_id[lin],
                            self.mps.dict_plant_2_node_id[lin])

            if self.dict_par['vc_fl'].has_monthly_factors:
                mt_id = self.dict_soy_month[(self.dict_pp_tm_id[lin], sy)]
                return self.vc_fl[mt_id, fl_id, nd_id]
            else:
                return self.vc_fl[fl_id, nd_id]

        return \
        sum(self.pwr[sy, lin, ca]
            * self.weight[self.dict_pp_tm_id[lin], sy]
            * spec_vc_fl(lin, sy)
            * (self.factor_lin_0[lin, ca]
               + 0.5 * self.pwr[sy, lin, ca]
                     * self.factor_lin_1[lin, ca])
            * self.nd_weight[self.mps.dict_plant_2_node_id[lin]]
            for (sy, lin, ca) in set_to_list(self.sy_lin_ca, nnn))

    def get_vc_co(self):
        r'''
        Get total |CO2| emission cost calculated directly from power
        production:

        .. math::
           \sum_\mathrm{(t,p,c)\in sy\_lin\_ca}
           p_\mathrm{t,p,c} w_\mathrm{\tau(p),t}
           \cdot \pi_\mathrm{CO_2, m(t), n(p)} i_\mathrm{CO_2,f}
           \cdot (f_\mathrm{0,p,c} + 0.5 p_\mathrm{t,p,c} f_\mathrm{1,p,c})

        '''

        return \
        sum(self.pwr[sy, lin, ca] * self.weight[self.dict_pp_tm_id[lin], sy]
            * (self.price_co2[self.dict_soy_month[(self.dict_pp_tm_id[lin], sy)],
                              self.mps.dict_plant_2_node_id[lin]]
               if self.dict_par['price_co2'].has_monthly_factors
               else self.price_co2[self.mps.dict_plant_2_node_id[lin]])
            * self.co2_int[self.mps.dict_plant_2_fuel_id[lin]]
                * (self.factor_lin_0[lin, ca]
                   + 0.5 * self.pwr[sy, lin, ca] * self.factor_lin_1[lin, ca])
            * self.nd_weight[self.mps.dict_plant_2_node_id[lin]]
            for (sy, lin, ca) in set_to_list(self.sy_lin_ca, nnn))


