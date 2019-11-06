#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:31:46 2019

@author: user
"""
from multiprocessing import Pool
from multiprocessing import current_process


def _call_list_run_id(func, list_run_id):

    for run_id in list_run_id:
        func(run_id)


def run_sequential(ml, func):
    ''' Sequential execution of all model runs. '''


    _call_list_run_id(func, ml.get_list_run_id())


def run_parallel(ml, func, nproc=None, groupby=None):
    '''
    Parameters
    ----------
    func : function(run_id)
        function to be sent to the workers
    nproc : int
        Number of processes
    groupby : list of `ModelLoop.df_run` columns
        Determines the groups of runs which are passed to the processes. This
        is necessary if certain model runs depend on each other.
    '''

    p = Pool(nproc)

    if groupby:
        # list of lists of run_ids grouped by groupby
        grouped_run_id = (ml.df_def_run.groupby(groupby)
                                       .run_id.apply(list).tolist())

        args = zip([func] * len(grouped_run_id), grouped_run_id)
        p.starmap(_call_list_run_id, args)

    else:
        list_run_id = ml.get_list_run_id()
        p.map(func, list_run_id)

    p.close()
    p.join()

    ml._merge_df_run_files()
