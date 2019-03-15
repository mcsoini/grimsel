
======================
Introductory example
======================

This example provides a minimal example of the modeling process with Grimsel, including the definition of the input tables.

.. code:: ipython3

    from grimsel.core.io import IO
    import numpy as np
    import pandas as pd
    import grimsel.core.model_loop as model_loop
    from grimsel.core.model_base import ModelBase as MB
    from grimsel import logger
    logger.setLevel(0)

Definition of the input data
=============================

The ``def_node`` table
-------------------------

The ``def_node`` table contains the mapping between *nd_id*s and node names *nd* as well as all parameters which are indexed by the nodes only. We define a node with name ``Node1`` with a certain |CO2| price ``price_co2``:

.. code:: ipython3

    df_def_node = pd.DataFrame({'nd': ['Node1'], 'price_co2': [40]})
    df_def_node.index.name = 'nd_id'
    df_def_node = df_def_node.reset_index()
    dict_nd = df_def_node.set_index('nd').nd_id.to_dict()
    display(df_def_node)



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
          <th>nd_id</th>
          <th>nd</th>
          <th>price_co2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>Node1</td>
          <td>40</td>
        </tr>
      </tbody>
    </table>
    </div>


.. important::
   The indices *nd_id* are best generated after the finalization of the table. The same holds for all other basic indices (power plants *pp_id*, fuels *fl_id* etc). It is good practice to never assume numerical indices as fixed, but to use translated names instead. The :class:`grimsel.core.auxiliary.maps.Maps` class provides convenience methods and attributes for this. For example, instead of referring to ``nd_id = 0`` we would always use ``nd_id = mps.dict_nd['Node1']`` instead (with ``mps`` an instance of the ``Maps`` class).

The ``def_fuel`` table
--------------------------

Every power plant in the broadest sense has fuel. This includes fuel-less renewable generators (wind, solar), demand-like model components, and transmission between nodes. The same holds for the ``def_plant`` and ``def_pp_type`` tables (see below).

.. note::
   Abstract fuels which follow directly from the model structure (demand, curtailment, transmission) don't have to be defined in the input table. They are automatically appended by the :module:`grimsel.core.autocomplete` module.

.. code:: ipython3

    df_def_fuel = pd.DataFrame({'fl': ['natural_gas', 'hard_coal', 'photovoltaics'], 
                                'co2_int': [0.20196, 0.34596, 0], 
                                })
    df_def_fuel.index.name = 'fl_id'
    df_def_fuel = df_def_fuel.reset_index()
    dict_fl = df_def_fuel.set_index('fl').fl_id.to_dict()
    display(df_def_fuel)



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
          <th>fl_id</th>
          <th>fl</th>
          <th>co2_int</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>natural_gas</td>
          <td>0.20196</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>hard_coal</td>
          <td>0.34596</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>photovoltaics</td>
          <td>0.00000</td>
        </tr>
      </tbody>
    </table>
    </div>


The ``def_encar`` table
--------------------------

Definition of produced energy carriers. 

.. note::
   In systems where a produced energy carrier can be consumed by other plants (e.g. electricity to produce heat), a column *fl_id* is required, which maps the output (e.g. electricity) carrier to the input (e.g. electricity) fuel.

.. code:: ipython3

    df_def_encar = pd.DataFrame({'ca_id': [0],
                                 'ca': ['EL']})
    df_def_encar.index.name = 'ca_id'
    dict_ca = df_def_encar.set_index('ca').ca_id.to_dict()
    display(df_def_encar)



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
          <th>ca_id</th>
          <th>ca</th>
        </tr>
        <tr>
          <th>ca_id</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>EL</td>
        </tr>
      </tbody>
    </table>
    </div>


The ``def_pp_type`` table
---------------------------
* mainly used for analysis

.. code:: ipython3

    df_def_pp_type = pd.DataFrame({'pt': ['GAS_LIN', 'SOL_PHO', 'HCO_ELC']})
    df_def_pp_type.index.name = 'pt_id'
    df_def_pp_type = df_def_pp_type.reset_index()
    dict_pt = df_def_pp_type.set_index('pt').pt_id.to_dict()
    display(df_def_pp_type)



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
          <th>pt_id</th>
          <th>pt</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>GAS_LIN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>SOL_PHO</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>HCO_ELC</td>
        </tr>
      </tbody>
    </table>
    </div>


The ``def_plant`` table
--------------------------


.. code:: ipython3

    df_def_plant = pd.DataFrame({'pp': ['ND1_GAS_LIN', 'ND1_SOL_PHO', 'ND1_HCO_ELC'],
                                 'pt_id': ['GAS_LIN', 'SOL_PHO', 'HCO_ELC'],
                                 'nd_id': ['Node1'] * 3,
                                 'fl_id': ['natural_gas', 'photovoltaics', 'hard_coal'],
                                 'set_def_pr': [0, 1, 0],
                                 'set_def_pp': [1, 0, 1],
                                 'set_def_lin': [1, 0, 0],
                                })
    df_def_plant.index.name = 'pp_id'
    df_def_plant = df_def_plant.reset_index()
    
    # translate columns to id using the previously defined def tables
    df_def_plant = df_def_plant.assign(pt_id=df_def_plant.pt_id.replace(dict_pt),
                                       nd_id=df_def_plant.nd_id.replace(dict_nd),
                                       fl_id=df_def_plant.fl_id.replace(dict_fl))
    dict_pp = df_def_plant.set_index('pp').pp_id.to_dict()
    
    display(df_def_plant)



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
          <th>pp_id</th>
          <th>pp</th>
          <th>pt_id</th>
          <th>nd_id</th>
          <th>fl_id</th>
          <th>set_def_pr</th>
          <th>set_def_pp</th>
          <th>set_def_lin</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>ND1_GAS_LIN</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>ND1_SOL_PHO</td>
          <td>1</td>
          <td>0</td>
          <td>2</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>ND1_HCO_ELC</td>
          <td>2</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>


The ``def_profile`` table
--------------------------




.. code:: ipython3

    df_def_profile = pd.DataFrame({'pf': ['SUPPLY_SOL_PHO', 'DMND_NODE1']})
    df_def_profile.index.name = 'pf_id'
    df_def_profile = df_def_profile.reset_index()
    dict_pf = df_def_profile.set_index('pf').pf_id.to_dict()
    df_def_profile





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
          <th>pf_id</th>
          <th>pf</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>SUPPLY_SOL_PHO</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>DMND_NODE1</td>
        </tr>
      </tbody>
    </table>
    </div>



The ``plant_encar`` table
--------------------------


.. code:: ipython3

    eff_gas_min = 0.4
    eff_gas_max = 0.6
    cap_gas = 4000.
    f0_gas = 1/eff_gas_min
    f1_gas = 1/cap_gas * (f0_gas - 1/eff_gas_max)
    
    df_plant_encar = pd.DataFrame({'pp_id': ['ND1_GAS_LIN', 'ND1_SOL_PHO', 'ND1_HCO_ELC'],
                                   'ca_id': ['EL'] * 3,
                                   'pf_id': [None, 'SUPPLY_SOL_PHO', None],
                                   'pp_eff': [None, None, 0.4],
                                   'factor_lin_0': [f0_gas, None, None],
                                   'factor_lin_1': [f1_gas, None, None],
                                   'cap_pwr_leg': [2000, 1000, 4000],
                                  })
    
    df_plant_encar = df_plant_encar.assign(pf_id=df_plant_encar.pf_id.replace(dict_pf),
                                           pp_id=df_plant_encar.pp_id.replace(dict_pp),
                                           ca_id=df_plant_encar.ca_id.replace(dict_ca))
    df_plant_encar
    





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
          <th>pp_id</th>
          <th>ca_id</th>
          <th>pf_id</th>
          <th>pp_eff</th>
          <th>factor_lin_0</th>
          <th>factor_lin_1</th>
          <th>cap_pwr_leg</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>0</td>
          <td>None</td>
          <td>NaN</td>
          <td>2.5</td>
          <td>0.000208</td>
          <td>2000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>0</td>
          <td>None</td>
          <td>0.4</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>4000</td>
        </tr>
      </tbody>
    </table>
    </div>



The ``fuel_node_encar`` table
--------------------------------


.. code:: ipython3

    df_fuel_node_encar = pd.DataFrame({'fl_id': ['natural_gas', 'hard_coal'],
                                   'nd_id': ['Node1'] * 2,
                                   'ca_id': ['EL'] * 2,
                                   'vc_fl': [40, 10],
                                  })
    df_fuel_node_encar = df_fuel_node_encar.assign(fl_id=df_fuel_node_encar.fl_id.replace(dict_fl),
                                           nd_id=df_fuel_node_encar.nd_id.replace(dict_nd),
                                           ca_id=df_fuel_node_encar.ca_id.replace(dict_ca))
    df_fuel_node_encar




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
          <th>fl_id</th>
          <th>nd_id</th>
          <th>ca_id</th>
          <th>vc_fl</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>40</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>10</td>
        </tr>
      </tbody>
    </table>
    </div>



The ``profsupply`` table
--------------------------


.. code:: ipython3

    hy = np.arange(24)
    sp = (-(hy - 12)**2 + 50) / 60
    sp[sp < 0] = 0
    
    df_profsupply = pd.DataFrame({'supply_pf_id': [dict_pf['SUPPLY_SOL_PHO']] * len(hy),
                                  'hy': hy, 'value': sp})
    
    df_profsupply.head()




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
          <th>supply_pf_id</th>
          <th>hy</th>
          <th>value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0</td>
          <td>1</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0</td>
          <td>2</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0</td>
          <td>3</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0</td>
          <td>4</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    </div>



The ``profdmnd`` table
--------------------------


.. code:: ipython3

    hy = np.arange(24)
    sp = np.sin(hy / 23 * np.pi *3) + 5
    
    df_profdmnd = pd.DataFrame({'dmnd_pf_id': [dict_pf['DMND_NODE1']] * len(hy),
                                  'hy': hy, 'value': sp})
    
    df_profdmnd.head()




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
          <th>dmnd_pf_id</th>
          <th>hy</th>
          <th>value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>5.000000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>1</td>
          <td>5.398401</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>2</td>
          <td>5.730836</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>3</td>
          <td>5.942261</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>4</td>
          <td>5.997669</td>
        </tr>
      </tbody>
    </table>
    </div>



Writing input tables to disk
---------------------------------


.. code:: ipython3

    for dftb, tbname in [(df_def_node, 'def_node'),
                       (df_def_plant, 'def_plant'),
                       (df_def_fuel, 'def_fuel'),
                       (df_def_encar, 'def_encar'),
                       (df_def_pp_type, 'def_pp_type'),
                       (df_plant_encar, 'plant_encar'),
                       (df_def_profile, 'def_profile'),
                       (df_fuel_node_encar, 'fuel_node_encar'),
                       (df_profsupply, 'profsupply'),
                       (df_profdmnd, 'profdmnd')]:
        dftb.to_csv('introductory_example_files/{}.csv'.format(tbname), index=False)
        

.. code:: ipython3

    import os
    
    mkwargs = {}
    iokwargs = {'data_path': os.path.abspath('introductory_example_files/'),
                'output_target': 'hdf5',
                'cl_out': 'introductory_example_files/output.hdf5'}
    nsteps = [('swco', 3, np.linspace),  # CO2 emission price
             ]
    ml = model_loop.ModelLoop(nsteps=nsteps, mkwargs=mkwargs, iokwargs=iokwargs)
    print(ml.io.datrd.data_path)


.. parsed-literal::

    > 21:22:55 - ERROR - grimsel.core.io - 'No object named def_run in the file'
    > 21:22:55 - WARNING - grimsel.core.io - reset_hdf_file: Could not determine max_run_id ... setting to None.


.. parsed-literal::

    
    ~~~~~~~~~~~~~~~   WARNING:  ~~~~~~~~~~~~~~~~
    You are about to delete existing file introductory_example_files/output.hdf5.
    The maximum run_id is None.
    
    Hit enter to proceed.
    
    /mnt/data/Dropbox/GRIMSEL_SOURCE/grimsel/notebooks/introductory_example_files


.. code:: ipython3

    IO._close_all_hdf_connections()
    ml.init_run_table()
    ml.df_def_run
    
    print(ml.io.datrd.data_path)



.. parsed-literal::

    /mnt/data/Dropbox/GRIMSEL_SOURCE/grimsel/notebooks/introductory_example_files


.. parsed-literal::

    Closing remaining open files:introductory_example_files/output.hdf5...done


.. code:: ipython3

    logger.setLevel(1000)
    ml.io.read_model_data()


.. code:: ipython3

    ml.m.init_maps()
    
    ml.m.map_to_time_res()
    
    
    # %
    ml.io.write_runtime_tables()
    
    ml.m.get_setlst()
    ml.m.define_sets()
    ml.m.add_parameters()
    ml.m.define_variables()
    ml.m.add_all_constraints()
    ml.m.init_solver()
    ml.io.init_output_tables()
    ml.select_run(0)



::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-17-86181a344f5c> in <module>
          1 ml.m.init_maps()
          2 
    ----> 3 ml.m.map_to_time_res()
          4 
          5 


    /mnt/data/Dropbox/GRIMSEL_SOURCE/grimsel/grimsel/core/model_base.py in map_to_time_res(self)
        762 
        763         self._init_time_map()
    --> 764         if not self.df_node_connect.empty:
        765             self._init_time_map_connect()
        766 


    AttributeError: 'NoneType' object has no attribute 'empty'

