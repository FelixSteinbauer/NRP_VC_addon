# -*- coding: utf-8 -*-
"""


"""
from __future__ import print_function
from builtins import str
# pragma: no cover

__author__ = 'Template'

from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np
import logging
import time
import sys
import Population_Utils as PU
import SNN_column
try:
    reload(SNN_column)
except:
    import importlib
    importlib.reload(SNN_column)

logger = logging.getLogger(__name__)

FORCE_dur = 3.87 #3.3 #3.87 #2.8 #10**x
    # -> 7413
delay_dur = 3.87 #3.3 #3.87 #10**x

##########################################################################################################################


#===============================================================
#                        default SNN
#===============================================================
def default_SNN(seed=None):
    if seed:
    	rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState(seed=np.random.seed())

    #setup dictionary parent structure
    SNN_dict = {}

    SNN_dict['networkStructure'] = {}
    SNN_dict['networkStructure']['n_in'] = {}
    SNN_dict['p_connect'] = {}
    SNN_dict['p_connect']['p_connect_in'] = {}
    SNN_dict['p_connect']['p_connect_intra'] = {}
    SNN_dict['weights'] = {}
    SNN_dict['weights']['w_in'] = {}
    SNN_dict['weights']['w_intra'] = {}
    SNN_dict['NexcNinh'] = {}
    SNN_dict['neuron_parameters'] = {}

    #Network structure
    SNN_dict['networkStructure']['n_in']['n_sensor'] = 1 #amount input neurons: body sensors ()
    SNN_dict['networkStructure']['n_in']['n_hli'] = 0 #amount input neurons: 
    SNN_dict['networkStructure']['n_in']['n_fb'] = 0 #amount input neurons: feedback
    SNN_dict['networkStructure']['n_res'] = [MARKER_COLUMNS] #amount reservoir populations (mini-columns)
    SNN_dict['networkStructure']['n_out'] = 0 #amount readout neurons
    n_sensor = SNN_dict['networkStructure']['n_in']['n_sensor']
    n_hli = SNN_dict['networkStructure']['n_in']['n_hli']
    n_fb = SNN_dict['networkStructure']['n_in']['n_fb'] 
    n_res = SNN_dict['networkStructure']['n_res'] 
    n_out = SNN_dict['networkStructure']['n_out'] 

    #Synaptic connectivity
    SNN_dict['p_connect']['p_connect_in']['p_connect_sensor'] = 0.1
    SNN_dict['p_connect']['p_connect_in']['p_connect_hli'] = 0
    SNN_dict['p_connect']['p_connect_in']['p_connect_fb'] = 0
    SNN_dict['p_connect']['p_connect_res'] = PU.createConnectivityMatrix(n_res)
    SNN_dict['p_connect']['p_connect_intra']['EE'] = 0   #Inra population connection p: Exitory to exitory
    SNN_dict['p_connect']['p_connect_intra']['EI'] = 0.1 #Inra population connection p: Exitory to inhibitory
    SNN_dict['p_connect']['p_connect_intra']['IE'] = 0.1 #Inra population connection p: inhibitory to exitory

    SNN_dict['scale_fb'] = 0
    SNN_dict['spectral_R'] = 1.15
    SNN_dict['weights']['w_in']['w_sensor'] =   rng.randn(n_res,n_sensor) #np.random.randn(n_res,n_sensor)
    SNN_dict['weights']['w_in']['w_hli'] = rng.randn(n_res,n_hli) #np.random.randn(n_res,n_hli)
    SNN_dict['weights']['w_in']['w_fb'] = rng.randn(n_res,n_fb) #np.random.randn(n_res,n_fb)
    SNN_dict['weights']['w_res'] = PU.get_rand_mat(n_res,SNN_dict['spectral_R'], seed=seed)
    SNN_dict['weights']['w_out'] = rng.randn(n_out,n_res) #np.random.randn(n_out,n_res)
    SNN_dict['weights']['w_intra']['EE'] = 0.1
    SNN_dict['weights']['w_intra']['EI'] = 1.0
    SNN_dict['weights']['w_intra']['IE'] = -1.0

    #Simulation Noise
    SNN_dict['noise_SD'] = 2.0 #SD for noise to be injected. Set to 0 for no noise 

    #Simulation delay
    SNN_dict['max_delay'] = 10
    SNN_dict['min_delay'] = 1.0
    SNN_dict['delays'] = rng.random_sample((n_res,n_res))*(SNN_dict['max_delay']-SNN_dict['min_delay'])+SNN_dict['min_delay']
    #SNN_dict['delays'] = np.random.sample((n_res,n_res))*(SNN_dict['max_delay']-SNN_dict['min_delay'])+SNN_dict['min_delay']

    #Reservoir population (mini-column) definition 
    SNN_dict['NexcNinh']['Nexc'] = [MARKER_Nexc] #amount exitory neurons
    SNN_dict['NexcNinh']['Ninh'] = [MARKER_Ninh] #amount inhibitory neurons
    # -> 80 to 20 -> 4 to 1

    #Neuron parameters
    SNN_dict['neuron_parameters']['regular'] = {
                'cm': 0.2,
                'v_reset': -75,
                'v_rest': -65,
                'v_thresh': -50,
                'tau_m': 30.0,  # sim.RandomDistribution('uniform', (10.0, 15.0)),
                'tau_refrac': 2.0,
                'tau_syn_E': 0.5,  # np.linspace(0.1, 20, hiddenP_size),
                'tau_syn_I': 0.5,
                'i_offset': 0.0
            }
    SNN_dict['neuron_parameters']['monitor'] = {
                'cm': 0.2,
                'v_reset': -15,
                'v_rest': 0,
                'v_thresh': float('inf'),
                'tau_m': 30.0,  # sim.RandomDistribution('uniform', (10.0, 15.0)),
                'tau_refrac': 0.1,
                'tau_syn_E': 5.5,  # np.linspace(0.1, 20, hiddenP_size),
                'tau_syn_I': 5.5,
                'i_offset': 0.0
            }

    return SNN_dict
# ===============================================================

time0 = time.time()
seed = 31337
SNN_dict = default_SNN(seed=seed)
n_res = SNN_dict['networkStructure']['n_res']
rng = np.random.RandomState(seed=seed)

# ===============================================================
#                       set params (override default)
# ===============================================================

# param = np.random.sample() * (max-min) + min
lamb = 6.15
SNN_dict['p_connect']['p_connect_res'] = PU.createConnectivityMatrix(n_res, lamb)

SNN_dict['spectral_R'] = 4.13 #3.4 #1.16 #2.8 #4.13
SNN_dict['weights']['w_res'] = PU.get_rand_mat(n_res,SNN_dict['spectral_R'], negative_weights=True, seed=seed)

#SNN_dict['noise_SD'] = np.random.sample() * (5-0) + 0
SNN_dict['p_connect']['p_connect_intra']['EE'] = 0.59
SNN_dict['p_connect']['p_connect_intra']['EI'] = 0.03
SNN_dict['p_connect']['p_connect_intra']['IE'] = 0.01
# ===============================================================
#change structure
SNN_dict['networkStructure']['n_in']['n_sensor'] = 4 #TODO: 8
n_sensor = SNN_dict['networkStructure']['n_in']['n_sensor']
SNN_dict['weights']['w_in']['w_sensor'] = rng.randn(n_res,n_sensor)#rng.randn(n_res,n_sensor) #

SNN_dict['networkStructure']['n_out'] = 4 #TODO: 8
n_out = SNN_dict['networkStructure']['n_out']
SNN_dict['weights']['w_out'] = rng.randn(n_out,n_res)#rng.randn(n_out,n_res) #

# set n_hli to 1 to ensure at least a dummy population is created, if its used depends on previous value (see prev lines). This hacky constellation is due to nrp transfer functions with rigid decorators/restricted python
n_hli = 1
SNN_dict['networkStructure']['n_in']['n_hli'] = n_hli 
SNN_dict['weights']['w_in']['w_hli'] = rng.randn(n_res,n_hli)

# ===============================================================

SNN_dict['neuron_parameters']['monitor']['tau_m'] = 80 #test it, acts as low pass filter -->seems to at least not hurt

# ===============================================================
# set up
SNN = SNN_column.SNN_column(SNN_dict)


##########################################################################################################################

SNN.set_recordings()

#TODO: add 4 more for muscles
sensor_population0 = SNN.sensor_populations[0]
sensor_population1 = SNN.sensor_populations[1]
sensor_population2 = SNN.sensor_populations[2]
sensor_population3 = SNN.sensor_populations[3]
HLI_population = SNN.sensor_populations[4]
print('created SNN, took '+str(time.time()-time0)+' s')

circuit = sensor_population0+sensor_population1+sensor_population2+sensor_population3
