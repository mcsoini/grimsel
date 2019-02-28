
Running Grimsel
=================

Grimsel is instantiated most conventiently through the :class:`grimsel.core.model_loop` class. This class takes care of data i/o (through the :class:`grimsel.core.io` class), model initialization (:class:`grimsel.core.model_base` class), as  well as parameter variation/scenario management. This shows a simple example on how to set up the model and run it.

The input data consists in either a PostgreSQL schema of normalized tables (see :doc:`doc_input_data` section) or a set of csv files with the same structure (see the `example files included with Grimsel on Github <https://github.com/mcsoini/grimsel/tree/master/input_data/>`_).


.. code:: ipython3

    import numpy as np
    
    from grimsel.core.model_base import ModelBase as MB
    import grimsel.core.model_loop as model_loop
    import grimsel.config as config
    from grimsel import logger
    logger.setLevel(1000)

Defining model parameters
-------------------------

The model parameters are collected in a dictionary ``mkwargs``. They are passed to the :class:`ModelLoop` initializer as dictionary and serve as keyword arguments to the :class:`ModelBase` ``__init__`` class.


.. code:: ipython3

    mkwargs = {'slct_encar': ['EL'],
               'slct_node': ['CH0', 'DE0', 'SFH_AA', 'SFH_AB'],
               'nhours': {'CH0': (1, 1), 'SFH_AA': (0.25, 0.25), 'SFH_AB': (0.25, 0.5)},
               'slct_pp_type': [],
               'skip_runs': False,
               'tm_filt': [('mt_id', [1, 4, 7, 10]), ('day', [1])],
               'constraint_groups': MB.get_constraint_groups(excl=['chp'])
               }


* ``slct_encar``: Which energy carriers to include. Any subset of the entries in the *def_encar* input table's *ca*     column. All input tables are filtered accordingly.
* ``slct_node``: Which nodes to include. Any subset of the entries in the *def_node* input table's *nd* column. All input tables are filtered accordingly. In the example two country-nodes ``[CH0, DE0]`` and two household-nodes ``[SFH_AA, SFH_AB]`` are included.
* ``nhours``: Original and target time resolution of all profiles in the selected nodes. The value pairs correspond to the ``freq`` and ``nhours`` parameters of the :class:`grimsel.auxiliary.timemap.TimeMap` class. In the example, the country nodes have 1 hour time resolution. For ``CH0``, this remains explicitly unchanged: ``(1, 1)``. ``SFH_AA`` and ``SFH_AB`` have 15 minute inpute data time resolution which is maintained for ``SFH_AA`` and averaged to 30 minutes for ``SFH_AB``: ``(0.25, 0.5)``. In principle, any combination of time resolutions is possible. However, :class:`grimsel.auxiliary.timemap.TimeMap` throws an error if the target time resolution is not a multiple of the input data time resolution.
* ``slct_pp_type``: Which power plant types to include. Any subset of the entries in the *def_pp_type* input table's *pt*     column. All input tables are filtered accordingly.
* ``skip_runs``: Doesn't perform model runs but only writes the constructed model parameters to the output data. Occasionally useful.
* ``tm_filt``: The parameter of the :class:`grimsel.auxiliary.timemap.TimeMap` class. In the example we limit the temporal scope of the model to the first day of four selected months.
* ``constraint_groups``: The constraint groups are the methods of the :class:`grimsel.core.constraints.Constraints` class whose name follows the pattern ``add_*_rules``. Through this model parameter it is possible to select a subset of the active constraints (e.g. to investigate infeasibilities). For convenience, the :func:`grimsel.core.model_base.ModelBase.get_constraint_groups` class method allows to select all constraint groups except for those specified by its ``excl`` parameter.

.. code:: ipython3

    MB.get_constraint_groups(excl=['chp'])  # demonstration of the ``get_constraint_groups`` method




.. parsed-literal::

    ['capacity_calculation',
     'capacity_constraint',
     'charging_level',
     'energy_aggregation',
     'energy_constraint',
     'hydro',
     'monthly_total',
     'objective',
     'ramp_rate',
     'supply',
     'transmission_bounds',
     'variables',
     'yearly_cost']



Defining input/output parameters
---------------------------------

The input/output parameters are collected in a dictionary ``iokwargs``. They are passed to the :class:`ModelLoop` initializer as dictionary and serve as keyword arguments to the :class:`grimsel.core.io.IO` ``__init__`` class.


.. code:: ipython3

    iokwargs = {# input
                'sc_inp': None,
                'data_path': config.PATH_CSV,
                # output
                'output_target': 'hdf5',
                'cl_out': 'grimsel_out',
                'no_output': False,
                'resume_loop': False,
                'replace_runs_if_exist': False,
                # general
                'dev_mode': True,
                'sql_connector': None,
                'autocomplete_curtailment': False,
               }

**Data input parameters**

* ``sc_inp``: Name of the input PostgreSQL schema if data is to be read from the database.
* ``data_path``: Name of the path holding the input data CSV files if applicable.

**Data output parameters**

* ``output_target``: One of ``'hdf5'`` (write to hdf5 file) or ``'psql'`` (write to PostgreSQL database).
* ``cl_out``: Name of the output table collection. This could either be a PostgreSQL schema or an hdf5 file.
* ``no_output``: If ``True``, no output is written to selected target, but only the model runs are performed.
* ``resume_loop``: Resume the model runs at a certain ``run_id``. If this is ``False`` (default), the output table collection (file or database schema) is re-initialized.
* ``replace_runs_if_exist``: By default, if ``resume_loop`` is an integer, all output data with ``run_id >= resume_loop`` is deleted prior to the first model run. If ``replace_runs_if_exist`` is ``True``, individual model runs are replaced instead.

**General parameters**

* ``dev_mode``: Re-initialize the output data target without the default warning.
* ``sql_connector``: Instance of the :class:`grimsel.auxiliary.sqlutils.aux_sql_func.SqlConnector` class. This is only required if either the input reading or model output writing makes use of a database connection.

Defining model loop parameters
---------------------------------

Apart from the ``iokwkargs`` and ``mkwargs`` dictionaries, the :class:`ModelLoop` class' only parameter is the nsteps list:

.. code:: ipython3

    nsteps = [('swco', 3, np.linspace),  # CO2 emission price
              ('swfy', 3, np.arange),    # Future years
             ]

It defines the steps of parameter variations/scenarios and hence the model runs. In this example, 3 steps of both a ``swco`` axis and a ``swfy`` axis are defined, e.g. to vary the |CO2| emission price and the future years separately. The third item of each tuple (numpy functions) specify whether the corresponding axis values are defined as 

* equally spaced steps between 0 and 1 (``np.linspace``); this might be convenient e.g. if the emission price is to varied between 0 and an upper maximum, e.g. 100. In this case the resulting swco loop value can just be used as a multiplicator.
* *n* steps (``np.arange``), e.g. to select discrete values (years, scenarios, etc) from a dictionary ``{0: 'yr2015', 1: 'yr2020', 3: 'yr2030'}``

Then, the ``ModelLoop`` instance is constructed as follows:

.. code:: ipython3

    ml = model_loop.ModelLoop(nsteps=nsteps, mkwargs=mkwargs, iokwargs=iokwargs)

Calling the :func:`grimsel.core.model_loop.ModelLoop.init_run_table` method generates a table with all combinations of the steps specified through the ``nsteps`` parameter above:

.. code:: ipython3

    ml.init_run_table()
    ml.df_def_loop




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>run_id</th>
          <th>swco_id</th>
          <th>swfy_id</th>
          <th>swco</th>
          <th>swfy</th>
          <th>swco_vl</th>
          <th>swfy_vl</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.5</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>2.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.5</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>5</th>
          <td>5</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>6</th>
          <td>6</td>
          <td>0.0</td>
          <td>2.0</td>
          <td>0.0</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>7</th>
          <td>7</td>
          <td>1.0</td>
          <td>2.0</td>
          <td>0.5</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>8</th>
          <td>8</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



The *run_id* column is the unique integer index, the *\*_id* columns are unique step ids for each axis, the columns bearing the axes names without suffix are the results of the step type specified in the ``nsteps`` parameter (``np.arange`` or ``np.linspace``). Finally, the *\*_vl* columns are actual names of the model run variations and are set later.

For detailed parameter studies this table gets quite large. It is typically filtered to limited the model runs:

.. code:: ipython3

    ml.df_def_loop = ml.df_def_loop.query('swco_id != 2 or swfy_id == 1')
    ml.df_def_loop




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>run_id</th>
          <th>swco_id</th>
          <th>swfy_id</th>
          <th>swco</th>
          <th>swfy</th>
          <th>swco_vl</th>
          <th>swfy_vl</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.5</td>
          <td>0.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.5</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>5</th>
          <td>5</td>
          <td>0.0</td>
          <td>2.0</td>
          <td>0.0</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>6</th>
          <td>6</td>
          <td>1.0</td>
          <td>2.0</td>
          <td>0.5</td>
          <td>2.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



Note that the *run_id* column is automatically reset for any change to this table.

Model setup
---------------------------------

A sequence of method calls is used to read the input data and to set up the model instance:

.. code:: ipython3

    ml.io.read_model_data()

The :class:`grimsel.core.io.IO.read_model_data` method reads the model data from the selected data source and adds all tables to the :class:`grimsel.core.model_base.ModelBase` instance.

.. code:: ipython3

    ml.m.init_maps()

The method :func:`grimsel.core.model_base.ModelBase.init_maps` uses the input table to generate a :class:`grimsel.auxiliary.maps.Maps` object. This is based on the ``def_*`` tables and serves primarily to convert ids to names and vice versa (of nodes, power plants, profiles, etc).

.. code:: ipython3

    ml.m.map_to_time_res()

The method :func:`grimsel.core.model_base.ModelBase.map_to_time_res` takes care of all model aspects related to the selected time resolution:

* It maps all input profiles to the desired model time resolution as specified by the ``nhours`` parameter (see above). This results in a set of attributes like ``ml.m.df_profdmnd_soy`` (equivalent for the other profile tables) which are filtered according to the ``tm_filt`` parameter and have potentially reduced time resolution.
* It generates the required tables to define the transmission between nodes, especially concerning the mapping of time slots for inter-nodal energy transmission between nodes with different time resolution.

.. code:: ipython3

    ml.io.write_runtime_tables()

The method :func:`grimsel.core.io.write_runtime_tables` writes input and runtime tables to the output data container. Runtime tables are time maps between model time slots and hours of the year.

.. code:: ipython3

    ml.m.get_setlst()
    ml.m.define_sets()

Sets up the pyomo set objects as attributes of the
:class:``grimsel.core.model_base.ModelBase`` class.

The method :func:``grimsel.core.sets.define_sets`` generates all
necessary pyomo set objects.

The method :func:``grimsel.core.sets.get_setlst`` generates a dictionary
``ml.m.setlst`` with the most basic sets, especially those defined by
the *set_def_\** columns of the ``ml.m.df_def_plant`` input table:

.. code:: ipython3

    print(ml.m.setlst['st'])  # for example all storage plants
    {pp: ml.m.mps.dict_pp[pp] for pp in ml.m.setlst['st']}  


.. parsed-literal::

    [57, 60, 102, 103]




.. parsed-literal::

    {57: 'DE_HYD_STO', 60: 'CH_HYD_STO', 102: 'SFH_AA_STO', 103: 'SFH_AB_STO'}



.. code:: ipython3

    ml.m.add_parameters()
    ml.m.define_variables()
    ml.m.add_all_constraints()

Adds all parameter, variable, and constraint attributes to the model (see :class:`grimsel.core.parameters`, :class:`grimsel.core.variables`, and :class:`grimsel.core.constraints`, ).

.. code:: ipython3

    ml.m.init_solver()

The method :func:`grimsel.core.model_base.ModelBase.init_solver` initializes a pyomo SolverFactory instance. Note that assumption on the CPLEX executable are hardcoded here in dependence on the operating system. If this doesn't work, manual manipulation is required.

.. code:: ipython3

    ml.io.init_output_tables()

:func:`grimsel.core.io.IO.init_output_tables` generates the output table handler objects :class:`grimsel.core.io.CompIO` and initializes the SQL tables (if applicable).

Generating a model loop modifier 
---------------------------------



.. code:: ipython3

    mlm = model_loop_modifier.ModelLoopModifier(ml)


Loop over model runs 
---------------------



Basic model data access
-------------------------


