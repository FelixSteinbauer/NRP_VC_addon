
from .Mouse_Experiment import *
import matplotlib.pyplot as plt

class Mouse_Experiment_FORCE_Interpolate(Mouse_Experiment):
    """FORCE learning on the mouse experiment


    phase_pid_init_duration -- Phase 1 length in seconds

    noiseSD - gaussian noise SD injected into the feedback population neurons and all reservoir population neurons (happens in SNN_column.py)

    Note: if either the contribution values or the target values are None at any point in time,
    FORECE learning (including weight updates) will be disables for this timepoint completely.
    However, the network dynamics as well as the inputs (controls, feedback) stay online. 

    """

    def __init__(
        self,
        exp_id,
        dataDirectory,
        workingFolderDirectory,
        muscleGTdir,
        brainFileName, #where the brain lies or is supposed to be stored
        templateFolder,
        stepSize=0.020,
        scenarios=["circle"],
        frequencies=[0.25,0.5,1],
        interpolate_fXY=[(1/3,1,1),(2,1,1)], #contains triplets that describe phases for interpolation phase
        repeatFactor=2, #how often the training sequence made up of the permuatiosn shall be repated (vandesomple default =2)
        brainRecreate=False,
        bibiMode=None, #just no line in the bibi file if None
        experimentName="Mouse_Experiment_FORCE_Interpolation",
        muscleValuesGTsource="FFT",
        folderPrefix="", folderInfix="", folderSuffix="",
        pid_init_duration=2,  # seconds
        pid_movement_duration=1,  # Periods
        FORCE_open_loop_duration=10,  # Periods
        FORCE_mixed_duration=10,  # Periods
        FORCE_closed_loop_duration=10,  # Periods
        FORCE_testing_duration=10,
        FORCE_interpolation_duration=10,
        columns=3*3*2, Nexec=30, Ninh=10,
        NESTthreads=8,
        noiseSD=2.0, lamb=6.15, spectralR=4.13,
        feedbackFactor=35.0, #how much the feedback signak is scaled
        p_shotNoise=0.01,  #probability of spike (impulse noise)
        shotSizeMax = 0.05,  # maximal noise size. so a signal value of 1 gets moved somewhere to [1,1.5]
        #LPfilter=1, #brauchen wir das als parameter?
        # From which algorythem the muscle ground truth values shall come #TODO: make an "CMAES" version
        gtNoiseSD=0.0
    ):

        super().__init__(
            exp_id=exp_id,
            frequency=min(frequencies), #START/Beginning scenario
            dataDirectory=dataDirectory,
            muscleGTdir=muscleGTdir,
            stepSize=stepSize,
            # be overwritten later by phase definition
            duration=(float("inf"), float("inf")),
            workingFolderDirectory=workingFolderDirectory,
            templateFolder=templateFolder,
            scenarioName=scenarios[0], #START/Beginning scenario
            brainFileName=brainFileName, 
            brainRecreate=brainRecreate,
            bibiMode=bibiMode,
            experimentName=experimentName,
            muscleValuesGTsource=muscleValuesGTsource,
            folderPrefix=folderPrefix, folderInfix=folderInfix, folderSuffix=folderSuffix,
            columns=columns, Nexec=Nexec, Ninh=Ninh,
            NESTthreads=NESTthreads,
            noiseSD=noiseSD, lamb=lamb, spectralR=spectralR,
            #p_shotNoise=p_shotNoise
            #LPfilter=LPfilter, #brauchen wir das als parameter?
        )

        # Training stuff
        self.gtNoiseSD = gtNoiseSD  # noise added during learning
        self.repeatFactor = repeatFactor
        self.frequencies = np.array(frequencies)
        self.scenarios = scenarios

        self.subsections = []
        for iteration in range(repeatFactor):
            for scenarioName in scenarios:
                for frequency in frequencies:
                    scenario = self.getScenarioParameters(scenarioName)
                    scenario["frequency"] = frequency
                    self.subsections.append(scenario)

        self.interpolate_fXY = np.array(interpolate_fXY) #interpolation phases
        self.interpolateSubsections = []
        for iteration in range(repeatFactor):
            for controls in self.interpolate_fXY:
                f,X,Y = controls
                scenario = {
                    "scenario": f"f={f}_X={X}_Y={Y}",
                    "A_max": 0.05*max([X,Y]),  # maximal distance from origin. required for error metric computation
                    "x_A": 0.05*X, "x_p": 0,
                    "y_A": 0.05*Y, "y_p": 0.5*np.pi,
                    "f":f, "X": X, "Y":Y
                }
                scenario["frequency"] = f
                self.interpolateSubsections.append(scenario)
        
        #Sensor2Brain TF
        self.feedbackFactor = feedbackFactor
        self.p_shotNoise = p_shotNoise
        self.shotSizeMax = shotSizeMax

        # Generate data for all phases
        self.setPhases(  # also sets the muscel and pid values and contributions
            pid_init_duration=pid_init_duration,  # seconds
            pid_movement_duration=pid_movement_duration,  # Periods
            FORCE_open_loop_duration=FORCE_open_loop_duration,  # Periods
            FORCE_mixed_duration=FORCE_mixed_duration,  # Periods
            FORCE_closed_loop_duration=FORCE_closed_loop_duration,  # Periods
            FORCE_testing_duration=FORCE_testing_duration,
            FORCE_interpolation_duration=FORCE_interpolation_duration
        )  # set FORCE specific phases. Also overrides/correct the simulation-duration

        # Set timeout (other pysics parameter etc. are set in the parent class allready)
        self.fileContents["experiment_configuration.exc"] = {"content": self.getExecConfiguration(
            templatePath=templateFolder + "experiment_configuration.exc", timeout=self.duration[0]) }

        # Set TF modifications: PIF control and Muscle control values
        self.TFcontents["joint_controller"] = {"replacements": self.getPIDreplacements(
            self.PIDvalues, self.PIDcontributionValues)}
        #self.TFcontents["muscle_controller"] = {
        #    "replacements": self.getMuscleReplacements(self.MuscleValues)}
        # we dont need this as the tf_SNN_FORCE will set all muscle values (either GT or network output)

        # Set TF modifications: FORCE control and contribution values
        self.TFcontents["FRC_send_sensor2brain"] = {
            "replacements":  self.getSensor2BrainReplacements()}    #FORCE Control and parameters

        self.TFcontents["FRC_tf_SNN_FORCE"] = {
            "replacements": self.getFORCEtfReplacements()}    #FORCE contribution

        # FORCE specific result files (will be filled after simulation)
        self.resultFiles["CSV"]["FORCE_inputs.csv"] = None
        self.resultFiles["CSV"]["SNN_FORCE.csv"] = None

    ##### PHASES AND THEIR VALUES #####

    def setPhases(self,
                  pid_init_duration=2,  # seconds
                  pid_movement_duration=1,  # Periods
                  FORCE_open_loop_duration=10,  # Periods
                  FORCE_mixed_duration=10,  # Periods
                  FORCE_closed_loop_duration=10,  # Periods
                  FORCE_testing_duration=10, #Periods
                  FORCE_interpolation_duration=10
                  ):
        """
        Side effect: creates self.PIDvalues, self.MuscleValues, self.FORCEcontributionValues and self.PIDcontributionValues.
        The FORCE control are (in this case) implicitly generated from the scenario parameters.
        """

        self.phases = []

        subsectionDuration = sum(1/self.frequencies)*self.repeatFactor*len(self.scenarios) #period length * repeat * scenarios
        #how long the subsections would take if only 1 period per scenario and speed were executed
        

        # add phases
        self.phases.append({
            "name": "pid init",  # phase name
            "t_start": 0,  # absolute start time in s
            "duration": pid_init_duration  # phase duration (in seconds)
        })
        self.phases.append({
            "name": "pid movement",  # phase name
            "t_start": self.phases[-1]["t_start"]+self.phases[-1]["duration"],
            "duration": pid_movement_duration*(1.0/self.frequency)  # phase duration
        })
        self.phases.append({
            "name": "FORCE open loop",  # phase name
            "t_start": self.phases[-1]["t_start"]+self.phases[-1]["duration"],
            "duration": FORCE_open_loop_duration*subsectionDuration,  # phase duration
            "periods": FORCE_open_loop_duration
        })
        self.phases.append({
            "name": "FORCE mixed",  # phase name
            "t_start": self. phases[-1]["t_start"]+self.phases[-1]["duration"],
            "duration": FORCE_mixed_duration*subsectionDuration,  # phase duration
            "periods": FORCE_mixed_duration
        })
        self.phases.append({
            "name": "FORCE closed loop",  # phase name
            "t_start": self.phases[-1]["t_start"]+self.phases[-1]["duration"],
            # phase duration (in Periods)
            "duration": FORCE_closed_loop_duration*subsectionDuration,
            "periods": FORCE_closed_loop_duration
        })
        self.phases.append({
            "name": "FORCE testing",  # phase name
            "t_start": self.phases[-1]["t_start"]+self.phases[-1]["duration"],
            # phase duration (in Periods)
            "duration":FORCE_testing_duration*subsectionDuration,
            "periods": FORCE_testing_duration
        })
        interp_duration = sum(1/(self.interpolate_fXY.transpose()[0]))*self.repeatFactor #Period lengths * repeat
        self.phases.append({
            "name": "FORCE interpolation",  # phase name
            "t_start": self.phases[-1]["t_start"]+self.phases[-1]["duration"],
            # phase duration (in Periods)
            "duration":FORCE_interpolation_duration*interp_duration,
            "periods": FORCE_interpolation_duration
        })

        # set duration
        minRuntime = self.phases[-1]["t_start"]+self.phases[-1]["duration"]

        self.duration = (minRuntime, minRuntime+10)

        # get PID, muslce, contribution etc. values for the total runtime
        self.PIDvalues = self.__generatePIDvalues()
        self.MuscleValues, self.MuscleValues_noised = self.__generateMuscleValues()
        self.FORCEcontributionValues = self.__generateFORCEContributionValues()
        self.FORCElearningValues = self.__generateFORCELearningValues()
        self.PIDcontributionValues = self.__generatePIDContributionValues()
        self.controlValues = self.__generateControlValues()

    def __generatePIDvalues(self):

        #Fron start/beginning scenario (first scenario in array, lowest frequency)
        f = self.frequency
        A1 = self.scenario["x_A"]
        p1 = self.scenario["x_p"]
        A2 = self.scenario["y_A"]
        p2 = self.scenario["y_p"]

        values = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            if (phase["name"] == "pid init"):  # -> static position of beginning of next phase
                t = phase["t_start"]+phase["duration"]
                # because we want to set the values now for the next timestep to be correct
                t = t + self.stepSize
                # longitudinal movement (Y) - joystick_helper_joint -front/back
                Y = A1*np.sin(t*f*2*np.pi+p1)
                # Transversal movement (X) - joystick_world_joint - left/right
                X = A2*np.sin(t*f*2*np.pi+p2)
                values_temp = [(X, Y)]*steps
                values.extend([tuple(value) for value in values_temp])
            elif (phase["name"] == "pid movement"):  # -> actual movment according to target
                # should be steps elment. beginnign inclusive. end exclusive
                ts = np.arange(
                    phase["t_start"], phase["t_start"]+phase["duration"], self.stepSize)
                # because we want to set the values now for the next timestep to be correct
                ts = ts + self.stepSize
                # longitudinal movement (Y) - joystick_helper_joint
                Y = A1*np.sin(ts*f*2*np.pi+p1)
                # Transversal movement (X) - joystick_world_joint
                X = A2*np.sin(ts*f*2*np.pi+p2)
                # shape: (2, steps) -> (steps, 2)
                values_temp = np.transpose([X, Y])
                values.extend([tuple(value) for value in values_temp])
            elif (phase["name"] in ["FORCE open loop", "FORCE mixed", "FORCE closed loop", "FORCE testing", "FORCE interpolation"]):  # -> idle PID
                # no PID control during force learning!
                values.extend([None]*steps)
            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating PID values')

        return values

    def __addNoise(self, muscleValues, gtNoiseSD):
        """
        
        nosie -- add percentage of signal spectrum (max-min) as gaussian noise to the signal (default 0.0)
        """

        #transpose
        muscleValues = np.array(muscleValues).transpose()

        # add gaussian noise to GT
        muscleValues_noised = np.zeros_like(muscleValues)
        for m, values in enumerate(muscleValues):
            scaleFactor = max(values)-min(values) #set in realtion to signal level
            muscleValues_noised[m] = values + np.random.normal(loc=0,scale=gtNoiseSD,size=len(values))*scaleFactor

        #re-tranpose
        muscleValues_noised = np.transpose(muscleValues_noised)
        muscleValues_noised = [tuple(value) for value in muscleValues_noised]

        return muscleValues_noised

    def __generateMuscleValues(self):

        values = []
        values_noised = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            # FYI: muscle values in this order: ['Foot1', 'Foot2', 'Radius1', 'Radius2', 'Humerus1', 'Humerus2', 'Humerus3', 'Humerus4']

            if (phase["name"] in ["pid init", "pid movement"]):  # -> muscles idle
                v = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*steps
                v_noise = v
            elif (phase["name"] in ["FORCE open loop","FORCE mixed","FORCE closed loop","FORCE testing"] ):  # -> GT with/without noise

                t_start=phase["t_start"]
                v = []
                for section in self.subsections:
                    #controls
                    f = section["frequency"]
                    X = section["x_A"]/section["A_max"]
                    Y = section["y_A"]/section["A_max"]
                    #duration
                    duration = phase["periods"]/f
                    t_end=t_start+duration
                    v.extend(self.getMuscleGT(phase,section["scenario"],f,X,Y,t_start=t_start, t_end=t_end)) #get values for section
                    t_start = t_end #update temporal position
                
                if(phase["name"] != "FORCE testing"): #while testing and interpolation we do not add noise
                    v_noise = self.__addNoise(v,self.gtNoiseSD)
                else:
                    v_noise = v

                assert len(v_noise) == steps, f'{len(v_noise)} == {steps}'

            elif (phase["name"] == "FORCE interpolation"):

                t_start=phase["t_start"]
                v = []
                for section in self.interpolateSubsections:
                    #controls
                    f = section["f"]
                    X = section["X"]
                    Y = section["Y"]
                    #duration
                    duration = phase["periods"]/f
                    t_end=t_start+duration
                    #GT scenario name
                    if(X!=0 and Y!=0):
                        scenarioName = "circle"
                    elif(X!=0 and Y == 0):
                        scenarioName = "longitudinal"
                    elif(X==0 and Y != 0):
                        scenarioName = "transversal"

                    #apply scaling
                    mgt = self.getMuscleGT(phase,scenarioName,f,X,Y,t_start=t_start, t_end=t_end)
                    scaling = max(X,Y)
                    scaled_vs = [[val*scaling for val in t] for t in mgt]
                    
                    v.extend(scaled_vs) #get values for section
                    t_start = t_end #update temporal position
                
                v_noise = v
                assert len(v_noise) == steps, f'{len(v_noise)} == {steps}'

            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating muscle values')
            
            values.extend(v)
            values_noised.extend(v_noise)

        return values, values_noised

    def __generateFORCEContributionValues(self):
        """Defines how much the SNN netowrk output shall contibute to the muscle innervation in each phase."""

        values = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            if (phase["name"] in ["pid init", "pid movement", "FORCE open loop"]):  # -> no learning = no contribution = FORCE offline
                values.extend([0]*steps)
            elif (phase["name"] == "FORCE mixed"): # -> contribution ttransition from 0 to 1
                # from 0 to 1 in steps steps (both ends inclusive)
                values.extend(np.linspace(0, 1, steps))
            elif (phase["name"] in ["FORCE closed loop", "FORCE testing", "FORCE interpolation"]):  # -> contribution = 1
                values.extend([1]*steps)
            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating FORCE contribution values')

        return values

    def __generateFORCELearningValues(self):
        """Defines wheter the SNN learning (weight update) is enabled or not."""

        values = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            if (phase["name"] in ["pid init", "pid movement",  "FORCE testing","FORCE interpolation"]):
                values.extend([False]*steps)
            elif (phase["name"] in ["FORCE open loop", "FORCE mixed", "FORCE closed loop",]):
                values.extend([True]*steps)
            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating FORCE learning values')

        return values

    def __generatePIDContributionValues(self):
        """Defines how much the PID controller shall contibute to the joystik movelemt

        The output values range from 1 to 0, where 1 represents 100% pid force and 0 means 0% PID force."""

        values = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            # -> no learning = full contribution
            if (phase["name"] in ["pid init", "pid movement"]):
                values.extend([1]*steps)
            # -> contribution = 0/offline
            elif (phase["name"] in ["FORCE open loop", "FORCE mixed", "FORCE closed loop", "FORCE testing", "FORCE interpolation"]):
                values.extend([None]*steps)
            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating PID contribution values')

        return values

    def __generateControlValues(self):

        frequency_control = []
        X_control = []
        Y_control = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            if(phase["name"] in ["pid init", "pid movement"]): #staring phases
                f = self.frequency
                X = self.scenario["x_A"]/self.scenario["A_max"]
                Y = self.scenario["y_A"]/self.scenario["A_max"]

                frequency_control.extend( [f]*steps )
                X_control.extend( [X]*steps )
                Y_control.extend( [Y]*steps )
            elif(phase["name"] in ["FORCE open loop","FORCE mixed","FORCE closed loop","FORCE testing"] ):
                
                for section in self.subsections:
                    #controls
                    f = section["frequency"]
                    X = section["x_A"]/section["A_max"]
                    Y = section["y_A"]/section["A_max"]
                    #duration
                    duration = phase["periods"]/f
                    steps_section = int(duration/self.stepSize)  # steps in this phase
                    frequency_control.extend( [f]*steps_section )
                    X_control.extend( [X]*steps_section )
                    Y_control.extend( [Y]*steps_section )

            elif (phase["name"] == "FORCE interpolation"):

                for section in self.interpolateSubsections:
                    #controls
                    f = section["frequency"]
                    X = section["X"]#section["x_A"]/section["A_max"]
                    Y = section["Y"]#section["y_A"]/section["A_max"]
                    #duration
                    duration = phase["periods"]/f
                    steps_section = int(duration/self.stepSize)  # steps in this phase
                    frequency_control.extend( [f]*steps_section )
                    X_control.extend( [X]*steps_section )
                    Y_control.extend( [Y]*steps_section )

            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating control values')

        return {"f":frequency_control,"X":X_control,"Y":Y_control}

    ##### TRANSFER FUNCTION CODE #####

    #different to the nfomrla FORCE class
    def getSensor2BrainReplacements(self):
        replacements = []
        
        #Create the TF replacements for the FORCE control values in the respecitve TF """

        fileName_f = "INJECTION_cont_f.txt" 
        self.fileContents[fileName_f] = {"content": json.dumps(self.controlValues['f'])}

        fileName_X = "INJECTION_cont_X.txt" 
        self.fileContents[fileName_X] = {"content": json.dumps(self.controlValues['X'])}

        fileName_Y = "INJECTION_cont_Y.txt" 
        self.fileContents[fileName_Y] = {"content": json.dumps(self.controlValues['Y'])}

        marker = "    #INJECTION_MARKER_CONTROLS\n"
        code = f"""

        all_controls_f = all_controls_f.value
        all_controls_X = all_controls_X.value
        all_controls_Y = all_controls_Y.value

        if isinstance(all_controls_f, type(None)):
            import json

            with open(\"{fileName_f}\", "r") as f:
                all_controls_f_str = f.read()
                all_controls_f = json.loads(all_controls_f_str)

            with open(\"{fileName_X}\", "r") as f:
                all_controls_X_str = f.read()
                all_controls_X = json.loads(all_controls_X_str)

            with open(\"{fileName_Y}\", "r") as f:
                all_controls_Y_str = f.read()
                all_controls_Y = json.loads(all_controls_Y_str)

        #t to index
        index = int(t/{self.stepSize})

        f = all_controls_f[index] if index < len(all_controls_f) else 0.5
        X = all_controls_X[index] if index < len(all_controls_X) else 1
        Y = all_controls_Y[index] if index < len(all_controls_Y) else 1
        
        HLI_pop0.amplitude = float(f) #Hz
        HLI_pop1.amplitude = float(X) #X percent
        HLI_pop2.amplitude = float(Y) #Y percent
        
        """
        replacements.append((marker,    code))

        #sensor feedback parameters
        marker = "    #INJECTION_MARKER_PARAMETERS\n"
        code = f"""
        
    feedbackFactor = {self.feedbackFactor} #empirically found value we need so scale our ~1.0 signal values with
    # increase this factor and the network will produce more overall spikes, decrease to starve the network
    impulse_p = {self.p_shotNoise} #probability of spike (impulse noise)
    impulse_size = {self.shotSizeMax} # maximal noise size. so a signal value of 1 gets moved somewhere to [1,1.5]
    f = {self.frequency}

    """
        replacements.append((marker,    code))
        return replacements

    ##same as in the normal FORCE class
    def getFORCEtfReplacements(self):
        """Create the TF replacements for the FORCE contribution values in the respecitve TF """

        #convert values to textfile
        fileName_mv_noised = "INJECTION_MuscleValues_noised.txt" 
        self.fileContents[fileName_mv_noised] = {"content": json.dumps(self.MuscleValues_noised)}

        fileName_contrib = "INJECTION_FORCEcontrib.txt" 
        self.fileContents[fileName_contrib] = {"content": json.dumps(self.FORCEcontributionValues)}

        fileName_learning= "INJECTION_learning.txt" 
        self.fileContents[fileName_learning] = {"content": json.dumps(self.FORCElearningValues)}
        
        #FORCE contribution replacements
        marker = "    #INJECTION_MARKER\n"
        code = f"""
        
    all_muscle_values = all_muscle_values.value
    all_contribution_values = all_contribution_values.value
    all_learning_values = all_learning_values.value

    if isinstance(all_muscle_values, type(None)):
        import json
        
        with open(\"{fileName_mv_noised}\", "r") as f:
            all_muscle_values_str = f.read()
            all_muscle_values = json.loads(all_muscle_values_str)

        with open(\"{fileName_contrib}\", "r") as f:
            all_contribution_values_str = f.read()
            all_contribution_values = json.loads(all_contribution_values_str)

        with open(\"{fileName_learning}\", "r") as f:
            all_learning_values_str = f.read()
            all_learning_values = json.loads(all_learning_values_str)

    #t to index
    index = int(t/{self.stepSize})

    target_values = all_muscle_values[index] if index < len(all_muscle_values) else (0,0, 0,0, 0,0,0,0)
    contribution = all_contribution_values[index] if index < len(all_contribution_values) else 1
    learning = all_learning_values[index] if index < len(all_learning_values) else False
    
    """
        replacements = [(marker,    code)]
        return replacements

    ##### EXPERIMENT RESULTS #####

    #see self.resultFiles for all the data extracted from the CSVs
    def postprocessResults(self):

        super().postprocessResults()  # do the higer abstraction postprocessing

        # now the FORCE specific postprocessing
        SNN_FORCE = self.resultFiles["CSV"]["SNN_FORCE.csv"]

        #Normalize for each section base don max amplitude
        SNN_FORCE["max_amps"] = np.ones_like(np.array(SNN_FORCE["error"]))
        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            if (phase["name"] in ["pid init", "pid movement"]):
                pass# default:ones
            elif (phase["name"] in ["FORCE open loop","FORCE mixed","FORCE closed loop","FORCE testing"] ):  # -> GT with/without noise

                t_start=phase["t_start"]
                for section in self.subsections:
                    i_start = int(t_start/self.stepSize)
                    duration = phase["periods"]/section["frequency"]
                    t_end=t_start+duration
                    i_end = int(t_end/self.stepSize)

                    t_start = t_end #update temporal position

                    muscleValues = np.array(self.MuscleValues[i_start:i_end])

                    SNN_FORCE["max_amps"][i_start:i_end] = np.max(muscleValues,axis=0)-np.min(muscleValues,axis=0)

            elif (phase["name"] == "FORCE interpolation"):

                t_start=phase["t_start"]
                for section in self.interpolateSubsections:
                    i_start = int(t_start/self.stepSize)
                    duration = phase["periods"]/section["f"]
                    t_end=t_start+duration
                    i_end = int(t_end/self.stepSize)

                    t_start = t_end #update temporal position

                    muscleValues = np.array(self.MuscleValues[i_start:i_end])

                    SNN_FORCE["max_amps"][i_start:i_end] = np.max(muscleValues,axis=0)-np.min(muscleValues,axis=0)

        SNN_FORCE["error"] = np.array(SNN_FORCE["error"])/SNN_FORCE["max_amps"] #normaize based on maximal amplitude of each muscle signal -> %Amp values
        #old version: (innacurate)
        # amplitudes = np.max(np.array(self.MuscleValues),axis=0)-np.min(np.array(self.MuscleValues),axis=0)
        #SNN_FORCE["error"] = np.array(SNN_FORCE["error"])/amplitudes #normaize based on maximal amplitude of each muscle signal -> %Amp values

        SNN_FORCE["error_abs"] = np.mean(np.abs(              # generate mean abolute error for all muscles
            SNN_FORCE["error"]), axis=1, keepdims=True)

        #where there is no new_w (becasue FORCE weight updates are off) we have 0
        new_w_forDiff = np.zeros((len(SNN_FORCE["new_w"]), self.columns, 8)) # shape: timesteps, columns, muscles
        for i, e in enumerate(SNN_FORCE["new_w"]):
            if e != None:
                new_w_forDiff[i] = e

        SNN_FORCE["new_w_diff"] = np.diff(                   # generate weight change matrix
            new_w_forDiff, axis=0)
        SNN_FORCE["new_w_diff_magnitude"] = np.linalg.norm(  # generate magnitude (L1 norm) of weight change matrix
            SNN_FORCE["new_w_diff"], axis=1)
        SNN_FORCE["new_w_magnitude"] = np.linalg.norm(  # generate magnitude (L1 norm) of weight matrix
            new_w_forDiff, axis=1)

        SNN_FORCE["sensor_monitor_1"] = [values[:8]
                                         for values in SNN_FORCE["sensor_monitor"]]
        SNN_FORCE["sensor_monitor_2"] = [values[8:]
                                         for values in SNN_FORCE["sensor_monitor"]]


    ##### VISUALIZATIONS #####

