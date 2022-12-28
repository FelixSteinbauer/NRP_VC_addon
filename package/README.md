# Neurorobotics Platform (NRP) Virtual Coach (VC) add-on

This package abstracts functionalities of the NRP virtual coach. On what hardware a simulation is run should be independent of the experiment files and their results. Therefore, this package aims to separate between the experiment setup (src/experiments) and how these experiments are run on a NRP server instance. A short overview of the respective concepts/classes (for more details on respective classes see the class description):

## Simulators

The "Simulators" (src/NRP_Simulator.py) specify different ways of running a Experiment. "**NRP_Simulator_Local**" is meant for local docker installations. As fewer issues occur for local installations, this class does not need to include complicated error handling. On the other hand, "**NRP_Simulator_Online**" tires to include catch different errors that may occur when running on an online NRP instance. These two classes are meant to simulate a single experiment each. "**NRP_Simulator_Paralell**" inherits from the online simulators and allows to specific a list of experiments which are automatically assigned and run on server instances in parallel. 

## Experiments

For each experiment folder in the NRP, one or more Experiment classes can be specified. Such Objects handle parameter-setting, file in and output and result post-processing. As each experiment object specifies the *exp_id* of its experiment, it can be given to a Simulator for simulation. In src/Experiments/Tigrillo is an exemplary class for the "Tigrillo SNN Learning in Closed Loop" NRP template experiment. The classes for my thesis experiment are in "Mouse_experiment". 


## TODOs:
Some stuff that might be improved.

### Code
- Proper logging (with levels, verbosity and stuff). Not just python print statements
- Loading/saving of network. Currently, you can just pickle the whole Experiment object and store it. However, it might be nice if one could just export the SNN.

<!--
### Repository
 - Make official package out of it (only if really ) -> https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives
 - 
 - 
 - -->

