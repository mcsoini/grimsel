## Companion branch replicating the results of paper [...].

* The input data consists of a collection of [csv files](input_data/).
* Grimsel generally requires a live PostgreSQL server for output writing and a local CPLEX installation.
* PostgreSQL settings can set a config_local.py file, as described in the error message raised by the [config.py](config.py) file (or set by replacing the config attributes)
* Grimsel assumes default installation locations for CPLEX. These are defined in the [model_base](core/model_base.py) module.
