
import pyomo.environ as po
from collections import namedtuple

from grimsel import _get_logger

logger = _get_logger(__name__)


class Variables:
    '''
    Mixin class containing all variables.
    '''





    def define_variables(self):
        Var = namedtuple('Var', ['name', 'sets', 'bounds', 'doc'])

        vars_ = [Var('pwr', self.sy_ppall_ca, None,
                     ':math:`p_\mathrm{sy\_ppall\_ca} \in (0,\infty)`: Per '
                     'time slot production of energy carriers '
                     ':math:`\mathrm{ca}` from all plants'),
                 Var('pwr_ramp', self.sy_rp_ca, (None, None),
                     ':math:`\Delta p_\mathrm{sy\_rp\_ca} \in '
                     '(-\infty,\infty)`: Per time slot ramping power '
                     'difference for relevant plants :math:`\mathrm{rp}`'),
                 Var('pwr_ramp_abs', self.sy_rp_ca, None,
                     ':math:`|\delta p_\mathrm{sy\_rp\_ca}| \in (0,\infty)`: '
                     'Per time slot absolute ramping power difference'),
                 Var('pwr_st_ch', self.sy_st_ca, None,
                     ':math:`p_\mathrm{sy\_st\_ca} \in (0,\infty)`: Per '
                     'time slot charging power of storage plants.'),
                 Var('erg_st', self.sy_st_ca | self.sy_hyrs_ca, None,
                     ':math:`e_\mathrm{sy\_st\_ca\cup sy\_hyrs\_ca} \in '
                     '(0,\infty)`: Stored energy in storage and reservoirs '
                     'each time slot'),

                 Var('trm', self.symin_ndcnn, (None, None),
                     ':math:`p_\mathrm{trm,symin\_ndcnn} \in '
                     '(-\infty,\infty)`: Internodal power transmission for '
                     'each of the time slots'),
                 Var('erg_mt', (self.mt, self.hyrs_ca | self.pp_ca,), None,
                     ':math:`E_\mathrm{mt,hyrs\_ca\cup pp\_ca} \in '
                     '(0,\infty)`: Monthly produced energy from hydro '
                     'reservoirs and dispatchable plants'),
                 Var('erg_fl_yr', self.ppall_ndcafl, None,
                     ':math:`E_\mathrm{ppall\_ndcafl} \in (0,\infty)`: '
                     'Yearly produced energy by fuel'),
                 Var('erg_yr', self.ppall_ca, None,
                     ':math:`E_\mathrm{ppall\_ca} \in (0,\infty)`: Yearly '
                     'produced energy by plant'),
                 Var('pwr_ramp_yr', self.rp_ca, None,
                     ':math:`\Delta p_\mathrm{rp\_ca} \in (0,\infty)`: Yearly '
                     'aggregated absolute ramping'),

                 Var('vc_fl_pp_yr', self.ppall_cafl - self.lin_cafl,
                     (None, None),
                     ':math:`\mathrm{vc}_\mathrm{fl, ppall\_cafl\setminus '
                     'lin\_cafl} \in (-\infty,\infty)`: Yearly '
                     'variable fuel cost (only constant supply curves).'),
                 Var('vc_om_pp_yr', self.ppall_ca, None,
                     ':math:`\mathrm{vc}_\mathrm{om, ppall\_ca} \in '
                     '(0,\infty)`: Yearly variable O\&M cost.'),
                 Var('fc_om_pp_yr', self.ppall_ca, None,
                     ':math:`\mathrm{fc}_\mathrm{om, ppall\_cafl} \in '
                     '(0,\infty)`: Yearly fixed O\&M cost.'),
                 Var('fc_cp_pp_yr', self.add_ca, None,
                     ':math:`\mathrm{fc}_\mathrm{cp, add\_ca} \in '
                     '(0,\infty)`: Yearly capital investment cost.'),
                 Var('vc_co2_pp_yr', self.pp_ca, None,
                     ':math:`\mathrm{vc}_\mathrm{em, pp\_ca} \in '
                     '(0,\infty)`: Yearly CO:sub:`2` emission cost.'),
                 Var('vc_ramp_yr', self.rp_ca, None,
                     ':math:`\mathrm{vc}_\mathrm{rp, rp\_ca} \in '
                     '(0,\infty)`: Yearly ramping cost.'),
                 Var('cap_pwr_tot', self.ppall_ca, None,
                     ':math:`P_\mathrm{tot, ppall\_ca} \in '
                     '(0,\infty)`: Net installed power capacity.'),
                 Var('cap_pwr_new', self.add_ca, None,
                     ':math:`P_\mathrm{new, add\_ca} \in '
                     '(0,\infty)`: New installed power capacity.'),
                 Var('cap_pwr_rem', self.rem_ca, None,
                     ':math:`P_\mathrm{ret, rem\_ca} \in '
                     '(0,\infty)`: Retired power capacity.'),
                 Var('cap_erg_tot', self.st_ca | self.hyrs_ca, None,
                     ':math:`C_\mathrm{ret, rem\_ca} \in '
                     '(0,\infty)`: Total energy capacity ofr storage and '
                     'reservoirs.'),
                 ]

        for var in vars_:
            self.delete_component(var.name)
            self.vadd(var.name, var.sets,
                      var.bounds if var.bounds else (0, None), doc=var.doc)

#        self.vadd('erg_ch_yr',       (self.st_ca,))                        # Yearly amount of charging energy MWh/yr



    def vadd(self, variable_name, variable_index, bounds=(0, None),
             domain=po.Reals, doc=''):
        if not type(variable_index) is tuple:
            variable_index = (variable_index,)

        logger.info('Defining variable %s ...'%variable_name)

        if not self.check_valid_indices(variable_index):
            return None
        else:
            logger.info('... ok.')

        setattr(self, variable_name, po.Var(*variable_index, bounds=bounds,
                                            domain=domain, doc=doc))

