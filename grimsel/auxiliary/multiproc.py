#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:31:46 2019

@author: user
"""
from multiprocessing import Pool
from multiprocessing import current_process
import contextlib
from grimsel.core.model_loop import logger_parallel
from grimsel import logger

def _call_list_run_id(func, list_run_id):

    for run_id in list_run_id:
        func(run_id)



@contextlib.contextmanager
def _adjust_logger_levels(do, ml, grimsel_level, parallel_level, verbose_solver):

    _verbose_solver = ml.m.verbose_solver
    old_grimsel_level = grimsel_level
    old_parallel_level = parallel_level

    if do:
        ml.m.verbose_solver = verbose_solver
        logger.setLevel(grimsel_level)
        logger_parallel.setLevel(parallel_level)

    yield

    ml.m.verbose_solver = _verbose_solver
    logger.setLevel(old_grimsel_level)
    logger_parallel.setLevel(old_parallel_level)


def run_sequential(ml, func, adjust_logger_levels=True):
    ''' Sequential execution of all model runs. '''

    with _adjust_logger_levels(adjust_logger_levels,
                               ml, 'DEBUG', 'ERROR', True):

        _call_list_run_id(func, ml.get_list_run_id())



def run_parallel(ml, func, nproc=None, groupby=None,
                 adjust_logger_levels=True):
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

    with _adjust_logger_levels(adjust_logger_levels,
                               ml, 'ERROR', 'INFO', False):

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




