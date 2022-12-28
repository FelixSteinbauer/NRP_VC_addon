"""
Applies the CMAES on the Mouse Experiment to find proper muscle innervation signals.
The appraoch however could be more generalized to also fit other experiments.
But as I just needed the Mouse experiment for my thesis, it is rather hardcoded for this specific experiment (Mouse_Experiment_CMAES)

"""

import os  # to load/save files
import pickle  # to pickle/unpickle
import cma
import time  # to get current time
import datetime  # to format time
import numpy as np

import matplotlib.pyplot as plt

from .NRP_Simulator import NRP_Simulator_Local, NRP_Simulator_Paralell, NRP_Simulator_Online

from .experiments.Mouse_Experiment.Mouse_Experiment_CMAES import Mouse_Experiment_CMAES


class CMAES_Runner():
    """Runs the CMAES algorythm

    CMAESfilePath -- path to file including the CMAES object. If exists the object
        there will be loaded an ALL other parametres are ignored.
    maxIteration -- after which amount of iterations the loop shall terminate
    trhead -- will be ignored of not online

    ExperimentParameter consitency: Normally all experiment parameter are stored in the cmaes file and the
    template Experiment are just relevant at the first instanciation of this the Runner not afterwards.
    However, the exp_id and the brainFileName are overwritten when this class is instanciated to allow changing of
    experiment name as well as the brain.

    Simulatior Consitancy: The simulator is not stored and has to be provided when creating the class.

    """

    def save(self, filePath):
        print("Saving - ", end='')
        with open(filePath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print("done")

    def load(self, filePath):
        with open(filePath, 'rb') as f:
            l = pickle.load(f)

        print("Continuing after iteration " +
              str(l.cmaes_object.result.iterations))
        return l


    def __init__( #notice, that first __new__ is called in python!
        self,
        templateExperiment,
        templateSimulator,
        CMAESfilePath,
        phaseName,
        len_x,  # how many steps (x) per phase
        x0,  # has to be lex_x long
        maxInstances=None, #how many paralell experiment instances shall be simulated. None for CMAES determined (might be too many for the cluster!)
        maxIteration=float("inf")
        ):

        # If the CMAES object(path) allready exisis, load it as object instance.
        if (os.path.exists(CMAESfilePath)):
            print(
                f'CMAES file \"{CMAESfilePath}\" exists. Loading and continuing where we stopped last time...')

            previousObject = self.load(CMAESfilePath)  # get previous object

            # override relevant properties
            previousObject.maxInstances = maxInstances

            previousObject.phaseName = phaseName
            for p in templateExperiment.phases: #get actual phase object from Name
                if (phaseName == p["name"]):
                    previousObject.phase = p
                    break

            if (previousObject.templateExperiment != None):
                # as we might continue our algorythem on a different cluster
                previousObject.templateExperiment.exp_id = templateExperiment.exp_id
                previousObject.templateExperiment.brainFileName = templateExperiment.brainFileName
                previousObject.templateExperiment.bibiMode = templateExperiment.bibiMode

            if (previousObject.templateSimulator != None):
                previousObject.templateSimulator = templateSimulator

            previousObject.maxIteration = maxIteration
            previousObject.CMAESfilePath = CMAESfilePath

            #basically replaces current object values with values form another instance
            self.__dict__.update(previousObject.__dict__)

            return None
        
        else: #we really create a new object
            self.iteration = 0

        # CMAES specific stuff
        self.CMAESfilePath = CMAESfilePath
        self.maxIteration = maxIteration
        self.maxInstances = maxInstances
        print(f"INSTANCES MAX1 = {self.maxInstances}")
        self.phaseName = phaseName
        self.x0 = x0
        self.len_x = len_x

        # Experiment specific stuff
        self.templateExperiment = templateExperiment
        for p in templateExperiment.phases: #get actual phase object from Name
            if (phaseName == p["name"]):
                self.phase = p
                break

        # Simulator specific stuff
        self.templateSimulator = templateSimulator
        

        # Otherwise really create a new one (now also the other parameters get relevant)

        # create CMAES object
        # x0
        # initial vector has to have same length as the
        assert len(self.x0) == self.len_x
        # Sigma
        #   The problem variables should have been scaled, such that a
        #     single standard deviation on all variables is useful and the optimum is expected to
        #     lie within about `x0` +- ``3*sigma0``."""
        self.sigma = 0.5
        # Options
        self.opts = cma.CMAOptions()
        #opts.set("tolstagnation", 2000)
        #opts.set('t/var/lib/docker/volumes/nrp_user_data/_data/MT/VC/package/src/NRP_VC_addon/experiments/Mouse_Experiment/Mouse_Experiment_CMAES.pyolfun', 0.5)
        #opts['tolx'] = 0.05/500
        #opts["minstd"] = 0.01/500
        #opts["maxstd"] = 0.01/500
        #opts = {"tolfun" : 10}

        self.x0_flat = np.array(x0).flatten()  # shape: (len_x, 8) -> (len_x*8)
        self.cmaes_object = cma.CMAEvolutionStrategy(
            self.x0_flat, self.sigma, self.opts)

        # Track history
        self.history = {  # for each iteration there is a new entry
            "inputs": [],
            "trajectories": [],
            "errors": [],  # distances from GT
            "losses": []  # tota error (mean distances)
        }

    #### RUN PIPELINE ####

    def __generateExperiments(self, inputs):
        '''
        Generate as much Experiments as necesarry to process all inputs.

        '''

        exps = []
        for in_data in inputs:

            exps.append(
                Mouse_Experiment_CMAES(
                    exp_id=self.templateExperiment.exp_id,
                    dataDirectory=self.templateExperiment.dataDirectory,
                    # might otherwise generate many identical folders #self.templateExperiment.workingFolderDirectory,
                    workingFolderDirectory=None,
                    brainFileName=self.templateExperiment.brainFileName,
                    muscleGTdir=self.templateExperiment.muscleGTdir,
                    templateFolder=self.templateExperiment.templateFolder,
                    CMAES_mixed_input=in_data if self.phaseName == "CMAES mixed" else None,
                    CMAES_input=in_data if self.phaseName == "CMAES" else None,
                    scenarioName=self.templateExperiment.scenarioName,
                    muscleValuesGTsource=self.templateExperiment.muscleValuesGTsource,
                    frequency=self.templateExperiment.frequency,
                    folderInfix=self.templateExperiment.folderInfix,
                    # Phases
                    pid_init_duration=self.templateExperiment.pid_init_duration,  # seconds
                    pid_movement_duration=self.templateExperiment.pid_movement_duration,  # Periods
                    CMAES_mixed_duration=self.templateExperiment.CMAES_mixed_duration,  # Periods
                    CMAES_duration=self.templateExperiment.CMAES_duration  # Periods
                ))

        return exps

    def __simulateExperiments(self, experiments):

        if (isinstance(self.templateSimulator, NRP_Simulator_Paralell)):  # if its online simulation (paralel)

            print(f'[SIMULATION] Running {len(experiments)} experimetns in paralell...')
            self.templateSimulator.setExperiments(experiments)
            self.templateSimulator.run()

        elif (isinstance(self.templateSimulator, NRP_Simulator_Local)):  # if its local
            # runn sequentially
            for i, exp in enumerate(experiments):
                print(f'[SIMULATION] Running experiment {i+1}/{len(experiments)}...')
                self.templateSimulator.setExperiment(exp)
                self.templateSimulator.run()

                print(f"DEBUG: loss = {exp.phases[2]['loss']}")

        elif (isinstance(self.templateSimulator, NRP_Simulator_Online)):  # if its online (sequential)
            # runn sequentially
            for i, exp in enumerate(experiments):
                print(f'[SIMULATION] Running experiment {i+1}/{len(experiments)}...')
                self.templateSimulator.setExperiment(exp)
                self.templateSimulator.run()

                print(f"DEBUG: loss = {exp.phases[2]['loss']}")

        else:
            raise RuntimeError(
                "Please use either the NRP_Simulator_Local or NRP_Simulator_Paralell for simulation!")

    def __extractOutput(self, experiments):

        trajectories = []
        errors = []
        losses = []

        for exp in experiments:
            trajectories.append(exp.trajectory) #HIER das ist zu viel trajectory

            for phase in exp.phases:
                if(phase["name"] == self.phaseName):
                    errors.append(phase["distances"])
                    losses.append(phase["loss"])
                    break

        self.history["trajectories"].append(trajectories)
        self.history["errors"].append(errors)
        self.history["losses"].append(losses)

        #print(f'trajectories = {repr(trajectories)}')
        #print(f'errors = {repr(errors)}')
        #print(f'losses = {repr(losses)}')

        return losses

    def run(self):

        print("Running optimizer...")
        tstart = time.time()
        while (self.cmaes_object.stop() == {}) and (self.iteration < self.maxIteration):

            print(
                f'[{datetime.timedelta(seconds=time.time()-tstart)}] ITERATION: Processing iteration {self.iteration}')
            istart = time.time()

            # Get (and split) Inputs for this iteration
            inputs = np.array(self.cmaes_object.ask())

            #Split inputs in chunks and process them in chunks
            solutions = []
            print(f"INSTANCES MAX = {self.maxInstances}")
            chunksSize = len(inputs) if self.maxInstances == None else self.maxInstances
            chunks = int(np.ceil(len(inputs) / chunksSize))
            for chunkID in range(chunks):
                inputs_chunk = inputs[chunkID*chunksSize:chunkID*chunksSize+chunksSize]
                print(f'[{datetime.timedelta(seconds=time.time()-tstart)}] ITERATION - CHUNK: Processing chunk nr {chunkID+1}/{chunks} (size={len(inputs_chunk)})')
            
                # shape: (N,len_x*8) -> (N,len_x, 8)
                inputs_chunk = inputs_chunk.reshape(len(inputs_chunk), -1, 8)

                # Generate experiments for all inputs
                print(
                    f'[{datetime.timedelta(seconds=time.time()-tstart)}] GENERATE EXPERIMENTS')
                exps = self.__generateExperiments(inputs_chunk)

                # Simulate experiments until all done
                print(
                    f'[{datetime.timedelta(seconds=time.time()-tstart)}] SIMULATE EXPERIMENTS]')

                outputs = self.__simulateExperiments(exps)

                # extract outputs and update object
                print(
                    f'[{datetime.timedelta(seconds=time.time()-tstart)}] EXTRACT OUTPUTS')

                solutions_chunk = self.__extractOutput(exps)
                solutions.extend(solutions_chunk)

            self.history["inputs"].append(inputs)
            print(
                f'Losses (min,max) = [ {min(solutions)}\t, {max(solutions)} ]')

            # save object
            print(f'[{datetime.timedelta(seconds=time.time()-tstart)}] SAVE')
            # remove VC instance and SIM so the AuthenticationString is gone and the object can be pickeled
            #self.templateSimulator.vc_instance = None
            self.templateSimulator.sim = None
            self.save(self.CMAESfilePath)

            # update CMA-ES
            print(f'[{datetime.timedelta(seconds=time.time()-tstart)}] UPDATE CMAES')
            self.cmaes_object.tell(inputs, solutions)
            self.cmaes_object.disp()

            self.iteration += 1  # make one iteration

            # print time per iteration
            print(
                f" Time per experiment: {(time.time()-istart)/len(inputs):.2f} seconds")
            print(
                f" Time per iteration: {(time.time()-istart)/60:.2f} minutes")

        # In the case the algorythm actually finishes, explain why
        if (self.cmaes_object.stop() != {}):
            print(
                f'[DONE] CMAES termination critera ({self.cmaes_object.stop()}) was reached after {self.iteration} iterations.')
        else:
            print(
                f'[DONE] Maximal iteration count ({self.maxIteration}) was reached after {self.iteration} iterations.')

    #### VISUALIZATIONS #### TODO: make labeling etc. more convinient/possible

    def getMinimalRun(self):
        #save minimal run specifically
        losses = self.history["losses"]
        interation_minimum = np.min(losses,axis=1)
        interation = np.argmin(interation_minimum)
        run_minimum = np.min(losses[interation])
        run = np.argmin(losses[interation])

        #self.minimalRun = {}
        #for key in self.history.keys():
        #    self.minimalRun[key] = self.history[key][interation][run]

        return interation, run

    #show trajectry development over time
    def plotLosses(
        self,
        minimalRun,
        amplitudeScale=True,
        title="Losses", xlabel="Iteration", ylabel="%Amplitude",
        drawMinMax = True,
        log=False,
        visibRange = (0.1,0.6), lw=0.5, draw_segements = False):

        #prepare data
        data = np.array(self.history["losses"])
        
        #Scale Amplitude
        if(amplitudeScale):
            data = data/self.templateExperiment.scenario["A_max"] #scale to percent of Amplitude (if necesarry)
        min_data = data[minimalRun[0]][minimalRun[1]] #get now for later usage
        
        #Get Mean
        means = np.mean(data,axis=1,keepdims=True)

        plt.figure()
        
        if(drawMinMax):
            plt.plot( np.min(data,axis=1), color="black", linewidth=lw,linestyle=':')
            plt.plot( np.max(data,axis=1), color="black",  linewidth=lw,linestyle=':')
            
        plt.plot( means, color="black", linewidth=lw)
        
        if(log):
            plt.yscale("log")
        if(xlabel != None):
            plt.xlabel(xlabel)
        if(ylabel != None):
            plt.ylabel(ylabel)
        if(title != None):
            plt.title(title)
            
        plt.show()

    #show error development over time
    def plotErrors(
        self,
        minimalRun,
        meanIteration=True, onlyLast=False, amplitudeScale=True,
        title="Error over time", xlabel="time (s)", ylabel="Error (%A)",
        visibRange = (0.1,0.6), lw=0.5, draw_segements = False):

        #prepare data
        data = np.array(self.history["errors"])
        
        #Scale Amplitude
        if(amplitudeScale):
            data = data/self.templateExperiment.scenario["A_max"] #scale to percent of Amplitude (if necesarry)
        min_data = data[minimalRun[0]][minimalRun[1]] #get now for later usage
        
        #Get Mean
        if(meanIteration):
            data = np.mean(data,axis=1,keepdims=True)
        data_shape = data.shape
        #print("Shape data:"+str(data_shape))
        
        #Get X axis
        xs = np.linspace(self.phase["t_start"], self.phase["t_start"]+self.phase["duration"],data_shape[-1]) #get timesteps
        #print("Shape xs:"+str(xs.shape))
            
        #create figure
        opacStep = (visibRange[1]-visibRange[0])/len(data)
        plt.figure()
        
        if(onlyLast == False): #draw all iterations until last
            for i, itData in enumerate(data[:-1]):  
                alpha=visibRange[0]+i*opacStep
                
                for l, line in enumerate(itData):
                    plt.plot(xs, line, color="black", alpha=alpha, linewidth=lw)
                    
        #draw last iteration
        itData = data[-1]
                    
        for l, line in enumerate(itData):
            plt.plot(xs, line, color="red", linewidth=lw*2.5, linestyle=':')

        #draw best
        plt.plot(xs, min_data, linewidth=lw*2.5, color="red")

        if(xlabel != None):
            plt.xlabel(xlabel)
        if(ylabel != None):
            plt.ylabel(ylabel)
        if(title != None):
            plt.title(title)
            
        plt.show()

    #show trajectory development over time
    def plotTrajectory(
        self,
        minimalRun,
        meanIteration=True, onlyLast=False, amplitudeScale=False, flatPlots = True,
        title="Error over time", xlabel="time (s)", ylabel="Error (%A)",
        visibRange = (0.1,0.6), lw=0.5, draw_segements = False, limits=(-0.06,0.06),
        index_start=None, index_end=None
        ):

        #prepare data
        fullTrajectory = np.array(self.history["trajectories"]) #contains the full trajectroy.
            # -> we now want to filter only for the relevant phase
        if(index_start == None):
            index_start = int( (self.phase["t_start"]/self.templateExperiment.stepSize))
        if(index_end == None):
            index_end = int((self.phase["t_start"]+self.phase["duration"])/self.templateExperiment.stepSize)    
        
        data = np.array(fullTrajectory[:,:,index_start:index_end])
        print(f'From {index_start} to {index_end}')
        
        #Scale Amplitude
        if(amplitudeScale):
            data = data/self.templateExperiment.scenario["A_max"] #scale to percent of Amplitude (if necesarry)
        
        
        data = data.transpose(0,1,3,2) # shape: (Iters, Runs, timepoints, 3) -> (Iters, Runs, 3, timepoints)
        min_data = data[minimalRun[0], minimalRun[1]] #get now for later usage
        
        #Get Mean
        if(meanIteration):
            data = np.mean(data,axis=1,keepdims=True)
        
        data_shape = data.shape
        #print("Shape data:"+str(data_shape))
        
        #get XS
        if(flatPlots):
            xs = np.linspace(self.phase["t_start"], self.phase["t_start"]+self.phase["duration"],data_shape[-1]) #get timesteps
            #print("Shape xs:"+str(xs.shape))
        
        #create figure
        opacStep = (visibRange[1]-visibRange[0])/len(data)
        fig, axs = plt.subplots(2 if flatPlots else 1)
            
        if(onlyLast == False): #draw all iterations until last
            for i, itData in enumerate(data[:-1]):  
                alpha=visibRange[0]+i*opacStep

                for l, line in enumerate(itData):
                    if(flatPlots == False):
                        #print("X values: "+str(line[0]))
                        #print("Y values: "+str(line[1]))
                        axs.plot(line[0], line[1], color="black", alpha=alpha, linewidth=lw)
                    else:
                        axs[0].plot(xs, line[0], color="black", alpha=alpha, linewidth=lw)
                        axs[1].plot(xs, line[1], color="black", alpha=alpha, linewidth=lw)

                    
        #draw last iteration
        itData = data[-1]
        
        #print("itData "+str(np.shape(itData)))
        for l, line in enumerate(itData):
            if(flatPlots == False):
                axs.plot(line[0], line[1], color="red", linewidth=lw*2.5, linestyle=":")
            else:
                axs[0].plot(xs, line[0], color="red", linewidth=lw*2.5, linestyle=":")
                axs[1].plot(xs, line[1], color="red", linewidth=lw*2.5, linestyle=":")
                
        #draw best
        #print("min_data "+str(np.shape(min_data)))
        if(flatPlots == False):
            axs.plot(min_data[0], min_data[1], color="red", linewidth=lw*2.5)
        else:
            axs[0].plot(xs, min_data[0], color="red", linewidth=lw*2.5)
            axs[1].plot(xs, min_data[1], color="red", linewidth=lw*2.5)
            
        #draw GT
        gt = self.templateExperiment.getGT(self.phase)
        gt = np.array(gt).transpose()
        if(flatPlots == False):
            axs.plot(gt[0], gt[1], color="green", linewidth=lw*2.5)
        else:
            axs[0].plot(xs, gt[0], color="green", linewidth=lw*2.5)
            axs[1].plot(xs, gt[1], color="green", linewidth=lw*2.5)

        #if(xlabel != None):
        #    axs.xlabel(xlabel)
        #if(ylabel != None):
        #    axs.ylabel(ylabel)
        #if(title != None):
        #    axs.title(title)

        if(flatPlots == False):
            plt.xlim(limits)
            plt.ylim(limits)
    
        plt.show()

    #plot Muscle Innervation
    def plotMuscleInput(
        self,
        minimalRun,
        meanIteration=True, onlyLast=False, scale=5,
        showInterpolation=False,
        title="Error over time", xlabel=r"Time (s)", ylabels=["Foot1","Foot2","Radius1","Radius2","Humerus1","Humerus2","Humerus3","Humerus4"],
        visibRange = (0.1,0.6), lw=0.5):
        
        ylim = (-scale,scale)

        #prepare data
        data = np.array(self.history["inputs"])
        
        if(showInterpolation):
            data =  np.array(
                [
                    [
                        self.templateExperiment.getMuscleInnervation(CMAES_input, self.phase, self.templateExperiment.stepSize)
                        for CMAES_input in iteration]
                    for iteration in CMAES_input]
            )
        
        data = data.transpose(0,1,3,2) # shape: (Iters, Runs, timepoints, 8) -> (Iters, Runs, 8, timepoints)
        min_data = data[minimalRun[0], minimalRun[1]] #get now for later usage
        
        #Get Mean
        if(meanIteration):
            data = np.mean(data,axis=1,keepdims=True)
        
        data_shape = data.shape
        #print("Shape data:"+str(data_shape))
        
        #get XS
        ts = np.linspace(self.phase["t_start"], self.phase["t_start"]+self.phase["duration"],data_shape[-1]) #get timesteps
        #print("Shape ts:"+str(ts.shape))
        
        #create figure
        opacStep = (visibRange[1]-visibRange[0])/len(data)
        fig, axs = plt.subplots(nrows=8, ncols=1,
                            sharey='row', sharex='all',
                            figsize=(10,16),
                            squeeze=False)
            
        if(onlyLast == False): #draw all iterations until last
            for i, itData in enumerate(data[:-1]):  
                alpha=visibRange[0]+i*opacStep

                for r, runData in enumerate(itData):
                    for m_i, muscleValues in enumerate(runData):
                        axs[m_i][0].plot(ts, muscleValues, color="black", alpha=alpha, linewidth=lw)
                    
        #draw last iteration
        itData = data[-1]
        
        #print("itData "+str(np.shape(itData)))
        for l, runData in enumerate(itData):
            for m_i, muscleValues in enumerate(runData):
                axs[m_i][0].plot(ts, muscleValues, color="red", linewidth=lw*2.5, linestyle=":")
                axs[m_i][0].set_ylim(ylim)
                axs[m_i][0].set_ylabel(ylabels[m_i])            
                
        axs[-1][0].set_xlabel(xlabel)
                
        #draw best
        #print("min_data "+str(np.shape(min_data)))
        for m_i, muscleValues in enumerate(min_data):
            axs[m_i][0].plot(ts, muscleValues , color="red", linewidth=lw*2.5)
            
        #draw GT
        #gt = self.templateExperiment.getGT(self.phase)
        #gt = np.array(gt).transpose()
        #for m_i, FFT_muscleValues in FFT:
        #    axs[m_i][0].plot(ts, FFT_muscleValues, color="green", linewidth=lw*2.5)

        #if(xlabel != None):
        #    axs.xlabel(xlabel)
        #if(ylabel != None):
        #    axs.ylabel(ylabel)
        #if(title != None):
        #    axs.title(title)
    
        plt.show()

    def plotx0(
        self,
        showInterpolation=False,
        ylim=(0,1),
        title="Error over time", xlabel=r"Time (s)", ylabels=["Foot1","Foot2","Radius1","Radius2","Humerus1","Humerus2","Humerus3","Humerus4"],
        lw=0.5):
        
        #prepare data
        data = self.cmaes_object.x0.reshape(-1,8).transpose()
        
        #get XS
        ts = np.linspace(self.phase["t_start"], self.phase["t_start"]+self.phase["duration"],data.shape[-1]) #get timesteps
        #print("Shape ts:"+str(ts.shape))
        
        #create figure
        fig, axs = plt.subplots(nrows=8, ncols=1,
                            sharey='row', sharex='all',
                            figsize=(10,16),
                            squeeze=False)
            
  
        #draw best
        for m_i, muscleValues in enumerate(data):
            axs[m_i][0].plot(ts, muscleValues , color="black", linewidth=lw*2.5)
            axs[m_i][0].set_ylim(ylim)
            axs[m_i][0].set_ylabel(ylabels[m_i])  
        axs[-1][0].set_xlabel(xlabel)
            
        plt.show()



# TODO:integrate
blacklist = [
    "prod_latest_backend-11-443",  # 504
    "prod_latest_backend-15-443",  # 504
    "prod_latest_backend-25-443",  # 504
    "prod_latest_backend-30-443",  # 504
]
# 504: says error but indeed starts and is unavialiable afetrwards

whitelist = [  # tested server
    "prod_latest_backend-2-443",
    "prod_latest_backend-4-443",
    "prod_latest_backend-6-443",
    "prod_latest_backend-7-443",
    "prod_latest_backend-8-443",
    "prod_latest_backend-9-443",
    "prod_latest_backend-10-443",
    "prod_latest_backend-12-443",
    "prod_latest_backend-13-443",
    "prod_latest_backend-14-443",
    "prod_latest_backend-16-443",
    "prod_latest_backend-17-443",
    "prod_latest_backend-18-443",
    "prod_latest_backend-19-443",
    "prod_latest_backend-22-443",
    "prod_latest_backend-27-443",
    "prod_latest_backend-28-443",
    "prod_latest_backend-29-443",
    "prod_latest_backend-1-80",
    "prod_latest_backend-2-80",
    "prod_latest_backend-6-80",
    "prod_latest_backend-7-80"
]
# blacklist.extend(whitelist)
