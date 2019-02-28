import itertools
import pandas as pd

def pdef(df, sets=[], val='value', fixnan=0):
    '''
    Transform a dataframe into a dictionary;
    primarily used for parameter setting.

    Keyword arguments:
    df -- input dataframe
    sets -- selection of columns (default all columns except val)
    val -- value column (default "value")
    fixnan -- If not False: used as keyword argument for fillna;
              if False: drop rows with null values
    '''
    df = df.reset_index(drop=True)
    sets = [c for c in df.columns if not c == val] if len(sets) == 0 else sets
    df = df[sets + [val]]

    if not (type(fixnan) == bool and fixnan == False):
        df = df.fillna(fixnan)
    else:
        df = df.loc[-df[val].isnull()]
    dct = df.set_index(sets)[val].to_dict()
    return dct

def set_to_list(st, vl):
    '''
    Takes a (multi-dimensional) pyomo set and returns all index combinations
    which satisfy the conditions vl
    Input:
    - st: pyomo set, e.g. pp_ndca composed of pp, nd, ca
    - vl: tuple with same dimensions, e.g. (None, None, 0) for all set members with ca == 0
    '''
    vl = [[v] if not (type(v) == list or v == None)  else v for v in vl]
    ind = [i[0] for i in enumerate(vl) if not i[1]==None]
    cbs = list(itertools.product(*[v for v in vl if not v == None]))
    return [a for a in st if tuple([a[ia] for ia in ind]) in cbs]

def cols2tuplelist(*args, return_df=False):
    '''
    Converts dataframes to lists of tuples.

    Multiple inputs are joined by crossproducts, i.e. yielding all
    combinations of the input rows.
    '''

    tl = []
    cols = []
    for idf in args:

        if type(idf) == pd.core.frame.Series:
            idf = pd.DataFrame(idf)

#        appkwargs = dict(func=tuple, axis=1)

        cols += idf.columns.tolist()

#        tl.append(list(idf.drop_duplicates().apply(**appkwargs)))
        tl.append([tuple(row) for row in idf.drop_duplicates().values])

    prod = list(itertools.product(*tl))
    prod = [tuple([ccc for cc in c for ccc in cc]) for c in prod]

    if return_df:
        prod = pd.DataFrame(prod, columns=cols)
    return prod



def get_ilst(df, cols=None):
    '''
    Get index list from dataframe columns, using all or a selection of the
    columns.
    - df: input dataframe
    - cols: selection of columns
    '''
#    cols = df.columns if cols == None else cols
#    return [tuple(l) for l in
#            df.loc[:, cols].drop_duplicates().get_values()]
    print('get_ilst got superseeded by cols2tuplelist')
    return cols2tuplelist(df[cols] if cols else df)

