## Introduction

This repository supplements my master’s thesis on FORCE-learning with musculoskeletal models in the NRP. As I utilized the NRP virtual coach often, I tried to create a more general interface that abstracts some functionalities of the VC ("package/" folder). This package might also be useful for other applications besides my thesis.

## The VC Package
For a description how the VC code package works, see the README in the "package" folder. I set up the package to be an actual python package. However, I did not make it installation ready as it itself is not necessarily bug free/complete. Consequently, I currently cannot provide an installation guide and the python files need to be included manually with the correct path (like in the notebooks).

## Reproducing Thesis Results

For reproducing results of thesis the following steps are necessary:
1. Clone this repository
2. Go to your NRP web frontend and upload the experiment folder "nrpexperiment_robobrain_mouse_v5" (currently not included in this repository).
3. If you do not work with a local NRP docker instance (localhost), create a file called "password.py" in the root of this repository, fill in the following code and replace the username and password string with your server login credentials (your EBRAINS account):
    ```python3
    OCID_username = "username"
    OCID_password = "password"
    ```
4. Start a jupyter notebook in the root of this repository (```jupyter notebook```)

### Run Experiment Simulations
5. Go into the "Mouse_Experiment_FORCE_Interpolate" notebook.
6. If you do not work with a local NRP docker instance, change  ```online = False ``` to True in the first cell and change the  ```VC_address``` parameter in the first "Simulation" cell (scroll down in the notebook) to the address of the server you want to use.
7. If the experiment folder name on the server is not "nrpexperiment_robobrain_mouse_v5_0", you need to correct the "exp_id=" parameter in the "Mouse_Experiment_FORCE_Interpolate(" call that generated the experiment object.
8. Run the whole notebook
9. Get the figures directly from the notebook (png) or as .svg from the newly generated "output/" folder. Images used in the following thesis figures are generated in this order (and with some other unimportant stuff happening in between): 5.8, 4.2, 5.1, 4.6, 5.5, 5.6, 5.7, 4.7, 5.3
10. Change the experiment parameters (section "Experiment"). By default, the variables for "speed inter-/extrapolation" are set. Comment out the lines for "pattern extrapolation" or "1 Hz only" to change the experiment and run the notebook again. *Note:* On the submission CD, pre-generated experiment files for these three experiments exist. In the GitHub repository I omitted these (big) files.

Notes:
- If you want to re-generate a experiment (because you have too much time), just rename the respective .pkl files in the output folder or remove them.

### Generate Other Figures
The following figures are generated in other notebooks:
- Figure 4.3: "CMAES_Visualization"
  - TODO: I currently cannot find the cmaes pickled file that I used for the thesis. However, that is the script for generating the figures and the "CMAES_Runner" notebook is responsible for running the CMA-ES algorithm. 
- Figure 4.4: TODO: update the resprective script so it is compatible with the current package version. 
- Figure 4.5: "Mouse_Experiment_GT_Generate"
  - generates the ground truth muscle innervations for all our scenarios using our naive appraoch
  - For each scenario two images are generated which make up figure 4.5.
  - The script expects a local simulation environment. This can be changed by replacing but this can esaily be changed by replacing ```simulator = NRP_Simulator_Local()``` with ```simulator = NRP_Simulator_Online(OCID_username=OCID_username,OCID_password=OCID_password, maxTries=1, VC_address="http://148.187.150.190")```
  - The resulting datasets are written to "data/FFT/". We pre-generated them, as it takes some time to regenerate them.


- Not all figures are generated by the "Mouse_Experiment_FORCE_Interpolate"
<!-- TODO: create script for generating the other figures -->


