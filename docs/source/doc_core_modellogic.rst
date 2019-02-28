.. |pwr_ramp_abs| replace:: :math:`|\delta p _ \mathrm{t,p,c}|`
.. |CO2| replace:: CO\ :sub:`2`\ \


===============================
Grimsel default model structure
===============================


Grimsel provides a flexible framework for the definition of energy optimization 
models. The included default case corresponds to a typical national-scale 
dispatch model. This section describes the mathematical formulation of the
default model. It is defined by its :ref:`sets <sets>`,
:ref:`variables <variables>`, :ref:`parameters <parameters>`, 
and :ref:`constraints <constraints>`. The :ref:`sets <sets>` and 
:ref:`parameter <parameters>` values are defined by the 
:ref:`input tables<input_data_structure>`.

.. todo::

   This section mixes the mathematical model formulation and the Python module
   documentation. This should better be separated.

.. _sets:
.. automodule:: grimsel.core.sets
    :members:
    :undoc-members:
    :show-inheritance:

.. _parameters:
.. automodule:: grimsel.core.parameters
    :members:
    :undoc-members:
    :show-inheritance:

.. _variables:
.. automodule:: grimsel.core.variables
    :members:
    :undoc-members:
    :show-inheritance:

.. _constraints:
.. automodule:: grimsel.core.constraints
    :members:
    :undoc-members:
    :show-inheritance:


