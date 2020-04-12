import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import wrapt

@wrapt.decorator
def silence_pd_warning(f, self, args, kwargs):

    def f_new():
        pd.options.mode.chained_assignment = None
        ret = f(*args, **kwargs)
        pd.options.mode.chained_assignment = 'warn'
        return ret

    return f_new()


def print_full(x):
    with pd.set_option('display.max_rows', len(x)) as _:
        print(x)



def get_ols(df, add_constant=True, verbose=False):
    '''
    Get a simple linear regression, based on a two-columns dataframe.

    Parameters
    ----------
        add_constant: bool

    '''


    y = df.iloc[:, 1]
    X = df.iloc[:, 0]
    if add_constant:
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model


    logit_model = sm.OLS(y,X.astype(float))


    result=logit_model.fit()
    if verbose:
        print(result.summary())

    dfres = pd.concat([pd.concat([result.params], keys=['val'], names=['result']),
                       pd.concat([result.pvalues], keys=['p_val'], names=['result']),
                       pd.concat([pd.DataFrame([result])], keys=['result_object'], names=['result'])
                      ]).T

    return dfres

def expand_rows(df, all_cols, val_cols, asindex=False):
#    df = df_ren_spec_energy_slct_0
#    all_cols = ['bdg_type','bdg_period','scenario']
#    val_cols = [bdg_type,bdg_period,scenario]

    # for each column all_cols containing the key string 'all', copy the whole dataframe for each of the values in val_cols
    all_cols_slct = [x for x in all_cols if x in df.columns.tolist()]
    val_cols_slct = [val_cols[x] for x in range(len(val_cols)) if all_cols[x] in df.columns.tolist()]

    icol = 0
    df_out = df.copy()
    icol = 3
    for icol in range(len(all_cols_slct)):

        df_rest = df_out[df_out[all_cols_slct[icol]].apply(str) != 'all']
        df_expd = df_out[df_out[all_cols_slct[icol]].apply(str) == 'all']

        try:
            all_col_unq = df_expd[all_cols_slct[icol]].unique()
            if (all_col_unq[0] == 'all') and (len(all_col_unq) == 1):
                df_out = pd.DataFrame()

                ival = 1
                for ival in range(len(val_cols_slct[icol])):
                    df_add = df_expd.copy()
                    df_add[all_cols_slct[icol]] = val_cols_slct[icol][ival]

                    df_out= df_out.append(df_add)
                df_out = pd.concat([df_out.copy(),df_rest])
        except:
            print('Error expand_rows')


    if df_out.columns.size != df.columns.size:
        raise ValueError('expand_rows: Column number changed')
    else:
        pass

    if asindex:
        df_out = df_out.set_index(all_cols, append=True, drop=True)

    return df_out

def read_xlsx_table(wb, sheets, columns, value_col=None, sub_table='', drop=[]):
    ''' Read tables from Excel file '''

#    wb
#    sheets = ['PLANT_ENCAR']
#    columns=ppca_cols


    ncols = len(columns)
    table_complete = pd.DataFrame()
    sheetname = sheets[0]
    for sheetname in sheets:
        sheet = wb.sheet_by_name(sheetname)

        rows = []
        number_of_rows = sheet.nrows

        row = 7
        for row in range(number_of_rows):
            value0  = (sheet.cell(row,0).value)

            if value0 == '-->' + sub_table:
                row_add = [c.value if not c.ctype in [5] else np.nan for c in sheet.row(row)[1:ncols+1]]
                rows.append(row_add)

        table_complete = pd.concat([table_complete,
                                    pd.DataFrame(rows, columns=columns)])

        for icol in table_complete.columns:
            table_complete.loc[(table_complete[icol].apply(str) == 'Inf')
                             | (table_complete[icol].apply(str) == 'inf'),
                               [icol]] = float('inf')
    return table_complete[[c for c in table_complete if not c in drop]]


def translate_id(df, dft, n):
    '''
    translate id columns
    '''

    if type(n) == str:
        n = [n] * 2
    n_id = [i_n + '_id' for i_n in n]
    dft = dft[[n[0], n_id[0]]]
    dict_id = dft.set_index(n[0])[n[0] + '_id']
    df[n_id[1]] = df[n_id[1]].replace(dict_id)
    return df, dict_id


