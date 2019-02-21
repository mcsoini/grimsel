
import pyomo.environ as po

from grimsel import _get_logger

logger = _get_logger(__name__)


class Variables:
    '''
    Mixin class containing all variables.
    '''





#        self.vadd('erg_ch_yr',       (self.st_ca,))                        # Yearly amount of charging energy MWh/yr



    def vadd(self, variable_name, variable_index, bounds=(0, None),
             domain=po.Reals):
        if not type(variable_index) is tuple:
            variable_index = (variable_index,)

        logger.info('Defining variable %s ...'%variable_name)

        if not self.check_valid_indices(variable_index):
            return None
        else:
            logger.info('... ok.')

        setattr(self, variable_name, po.Var(*variable_index, bounds=bounds,
                                            domain=domain))
