# Neurorobotics Platform (NRP) Virtual Coach (VC) add-on

This package abstracts functionalities of the NRP virtual coach. 
Its main goals is to to provide seperate classes for the experiment setup (src/experiments) and the way these experiments are run on a NRP server instance. On what hardware the simulation is executed should be independent from the experiment files and its results.


## TODOs:
Some stuff that might be improved.

### Code
- Proper logging (with levels, verbosity and stuff). Not just python print statements
- Loading/saving of network. Currently you can just pickle the whole Experiment object and store it. However, it might be nice if one could just export the SNN.

<!--
### Repository
 - Make official package out of it (only if really ) -> https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives
 - 
 - 
 - -->

