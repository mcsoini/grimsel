
import warnings
import pyomo.environ as po
import numpy as np

class Variables:
    '''
    Mixin class containing all variables.
    '''

    def define_variables(self):

        self.vadd('pwr',             (self.sy_ppall_ca,))
        self.vadd('pwr_ramp',        (self.sy_rp_ca), (None, None))
        self.vadd('pwr_ramp_abs',    (self.sy_rp_ca))              # Absolute value ramp rates for cost calculation
        self.vadd('pwr_st_ch',       (self.sy_st_ca,))               #
        self.vadd('erg_st',          (self.sy_st_ca | self.sy_hyrs_ca,)) # Hourly energy stored

        self.vadd('trm',             (self.symin_ndcnn), (None, None))  # Positive or negative power sent through cross-border connections

        self.vadd('erg_mt',          (self.mt, self.hyrs_ca | self.pp_ca,)) #

        self.vadd('erg_fl_yr',       (self.ppall_ndcafl,))                 # Yearly energy production in MWh per year
        self.vadd('erg_yr',          (self.ppall_ca,))                     # Yearly energy production in MWh per year
        self.vadd('dmnd_flex_yr',    (self.ndca,))                         # Yearly amount of flexible loads
        self.vadd('pwr_ramp_yr',     (self.pprp_ca,))                      # Yearly amount of ramping MW/yr
        self.vadd('erg_ch_yr',       (self.st_ca,))                        # Yearly amount of charging energy MWh/yr

        self.vadd('vc_fl_pp_yr',     (self.ppall_cafl - self.lin_cafl,), (None, None))     # Yearly variable cost of fuel
        self.vadd('vc_om_pp_yr',     (self.ppall_ca,))                     # Yearly variable cost of O&M
        self.vadd('fc_om_pp_yr',     (self.ppall_ca,))                     # Yearly fixed cost of O&M
        self.vadd('fc_cp_pp_yr',     (self.add_ca,))                       # Annualized investment cost of capacity
        self.vadd('fc_dc_pp_yr',     (self.ppall_ca | self.st_ca,))        # Decommissioning cost
        self.vadd('vc_co2_pp_yr',    (self.pp_ca,))                        # Yearly variable cost of CO2 emissions
        self.vadd('vc_dmnd_flex_yr', (self.ndca,))                         # Yearly variable cost of flexible loads
        self.vadd('vc_ramp_yr',      (self.pprp_ca,))                      # Yearly variable cost of ramping
        self.vadd('cap_pwr_tot',     (self.ppall_ca,))                     # Total capacity
        self.vadd('cap_pwr_new',     (self.add_ca,), (0, self.capchnge_max))   # New capacity
        self.vadd('cap_pwr_rem',     (self.rem_ca,), (0, self.capchnge_max))   # Retired capacity
        self.vadd('cap_erg_tot',     (self.st_ca | self.hyrs_ca,))         # Total energy capacity hydro+storage calculated from cap_pwr_tot


    def vadd(self, variable_name, variable_index, bounds=(0, None),
             domain=po.Reals):
        if not type(variable_index) is tuple:
            variable_index = (variable_index,)

        print('Defining variable ', variable_name, end='... ')

        if not self.check_valid_indices(variable_index):
            return None
        else:
            print('ok.')

        setattr(self, variable_name, po.Var(*variable_index, bounds=bounds,
                                            domain=domain))
