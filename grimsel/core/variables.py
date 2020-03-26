'''
Model variables
=================


'''

import pyomo.environ as po
from collections import namedtuple

from grimsel import _get_logger
from grimsel.auxiliary.aux_general import silence_pd_warning

logger = _get_logger(__name__)


VAR_DOCS = {'pwr': ':math:`p_\\mathrm{t,p,c} \\forall sy\\_ppall\\_ca \\in (0,\\infty)`: per time slot production of energy carriers :math:`\\mathrm{ca}` from all plants',
         'pwr_ramp': ':math:`\\delta p_\\mathrm{t,p,c} \\forall sy\\_rp\\_ca \\in (-\\infty,\\infty)`: per time slot ramping power difference for relevant plants :math:`\\mathrm{rp}`',
#         'pwr_ramp_abs': ':math:`|\\delta p_\\mathrm{t,p,c}| \\forall sy\\_rp\\_ca \\in (0,\\infty)`: per time slot absolute ramping power difference',
         'pwr_ramp_abs': '|pwr_ramp_abs| `\\forall sy\\_rp\\_ca \\in (0,\\infty)`: per time slot absolute ramping power difference',
         'pwr_st_ch': ':math:`p_\\mathrm{chg,t,p,c} \\forall sy\\_st\\_ca \\in (0,\\infty)`: per time slot charging power of storage plants',
         'erg_st': ':math:`e_\\mathrm{t,p,c} \\forall sy\\_st\\_ca\\cup sy\\_hyrs\\_ca \\in (0,\\infty)`: stored energy in storage and reservoirs each time slot',
         'trm': ':math:`p_\\mathrm{trm,t,n,n_2,c} \\forall symin\\_ndcnn \\in (-\\infty,\\infty)`: internodal power transmission for each of the time slots',
         'erg_mt': ':math:`E_\\mathrm{m,p,c} \\forall mt \\times hyrs\\_ca\\in (0,\\infty)`: monthly produced energy from hydro reservoirs',
         'erg_fl_yr': ':math:`E_\\mathrm{p,n,c,f} \\forall ppall\\_ndcafl\\in (0,\\infty)`: yearly produced energy by plant and fuel',
         'erg_yr': ':math:`E_\\mathrm{p,c} \\forall ppall\\_ca \\in (0,\\infty)`: yearly produced energy by plant',
         'pwr_ramp_yr': ':math:`\\Delta p_\\mathrm{p,c} \\forall rp\\_ca \\in (0,\\infty)`: yearly aggregated absolute ramping',
         'vc_fl_pp_yr': ':math:`{c}_\\mathrm{fuel, p,c,f} \\forall ppall\\_cafl\\setminus lin\\_cafl \\in (-\\infty,\\infty)`: yearly variable fuel cost (only constant supply curves)',
         'vc_om_pp_yr': ':math:`{c}_\\mathrm{om_v,p,c} \\forall ppall\\_ca \\in (0,\\infty)`: yearly variable O\\&M cost',
         'fc_om_pp_yr': ':math:`{c}_\\mathrm{om_f, p,c} \\forall ppall\\_ca \\in (0,\\infty)`: yearly fixed O\\&M cost',
         'fc_cp_pp_yr': ':math:`{c}_\\mathrm{cp, p,c} \\forall add\\_ca \\in (0,\\infty)`: yearly capital investment cost',
         'vc_co2_pp_yr': ':math:`{c}_\\mathrm{em, p,c} \\forall pp\\_ca \\in (0,\\infty)`: yearly |CO2| emission cost',
         'vc_ramp_yr': ':math:`{c}_\\mathrm{rp, p,c} \\forall rp\\_ca \\in (0,\\infty)`: yearly ramping cost',
         'cap_pwr_tot': ':math:`P_\\mathrm{tot, p, c} \\forall ppall\\_ca \\in (0,\\infty)`: net installed power capacity',
         'cap_pwr_new': ':math:`P_\\mathrm{new, p,c} \\forall add\\_ca \\in (0,\\infty)`: new installed power capacity',
         'cap_pwr_rem': ':math:`P_\\mathrm{ret, p,c} \\forall rem\\_ca \\in (0,\\infty)`: retired power capacity',
         'cap_erg_tot': ':math:`C_\\mathrm{ret, p,c} \\forall rem\\_ca \\in (0,\\infty)`: total energy capacity ofr storage and reservoirs'}

class Variables:
    '''
    Mixin class containing all variable definitions.
    '''





    def define_variables(self):
        r'''
        Adds all variables to the model instance by calling :func:`vadd`.

        '''

        Var = namedtuple('Var', ['name', 'sets', 'bounds'])

        vars_ = [Var('pwr', self.sy_ppall_ca, None),
                 Var('pwr_ramp', self.sy_rp_ca, (None, None)),
                 Var('pwr_ramp_abs', self.sy_rp_ca, None),
                 Var('pwr_st_ch', self.sy_st_ca, None),
                 Var('erg_st', self.sy_st_ca | self.sy_hyrs_ca, None),

                 Var('trm', self.symin_ndcnn, (None, None)),
                 Var('erg_mt', (self.mt, self.hyrs_ca), None),
                 Var('erg_fl_yr', self.ppall_ndcafl, None),
                 Var('erg_yr', self.ppall_ca, None),
                 Var('pwr_ramp_yr', self.rp_ca, None),

                 Var('vc_fl_pp_yr', self.ppall_cafl - self.lin_cafl, (None, None)),
                 Var('vc_om_pp_yr', self.ppall_ca, None),
                 Var('fc_om_pp_yr', self.ppall_ca, None),
                 Var('fc_cp_pp_yr', self.add_ca, None),
                 Var('vc_co2_pp_yr', self.pp_ca, None),
                 Var('vc_ramp_yr', self.rp_ca, None),
                 Var('cap_pwr_tot', self.ppall_ca, None),
                 Var('cap_pwr_new', self.add_ca, None),
                 Var('cap_pwr_rem', self.rem_ca, None),
                 Var('cap_erg_tot', self.st_ca | self.hyrs_ca, None),
                 ]

        for var in vars_:
            self.delete_component(var.name)
            self.vadd(var.name, var.sets,
                      var.bounds if var.bounds else (0, None),
                      doc=VAR_DOCS[var.name])



    def vadd(self, variable_name, variable_index, bounds=(0, None),
             domain=po.Reals, doc=''):
        '''
        Adds a single variable to the model instance.

        Makes sure the ``variable_index`` is a tuple of pyomo sets and
        checks whether the sets are None or empty

        Parameters
        ----------
        variable_name : str
            name of the variable attribute
        variable_index : pyomo set
            parameter index set
        bounds : tuple
            upper and lower bounds of the variable defined through len-2 tuple;
            default: ``(0, None)``
        domain : pyomo domain
            default: ``pyomo.environ.Reals``
        doc : str
            parameter docstring; default ''

        '''

        if not type(variable_index) is tuple:
            variable_index = (variable_index,)

        logger.info('Defining variable %s ...'%variable_name)

        if not self.check_valid_indices(variable_index):
            return None
        else:
            logger.info('... ok.')

        setattr(self, variable_name, po.Var(*variable_index, bounds=bounds,
                                            domain=domain, doc=doc))


    @silence_pd_warning
    @staticmethod
    def _get_set_docs():

        import pandas as pd
        import tabulate


        var_doc = pd.Series(VAR_DOCS).apply(lambda x: r'{}'.format(x))
        var_doc = var_doc.reset_index()
        var_doc.columns = ['Attribute', 'Symbol']

        def get_doc_row(x):
            x_spl = x.Symbol.split('\in ')
            x_spl2 = x_spl[1].split('infty)`:')
            x_spl3 = x_spl[0].split(r'\forall')

            dct_row = {'Attribute': '``%s``'%x.Attribute,
                       'Symbol': x_spl3[0].strip(' ') + '`',
                       'Set': ':math:`\mathrm{' + x_spl3[1].strip(' ') + '}`',
                       'Bounds': ':math:`' + x_spl2[0] + 'infty)`',
                       'Doc': x_spl2[1].strip(' ').strip('\n')}

            return pd.Series(dct_row)

        var_doc = var_doc.apply(get_doc_row, axis=1)


        return tabulate.tabulate(var_doc, tablefmt='rst', showindex=False,
                          headers=var_doc.columns)


Variables.define_variables.__doc__ += '\n'*3 + Variables._get_set_docs()


