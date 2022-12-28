
from .Mouse_Experiment import *

class Mouse_Experiment_CMAES(Mouse_Experiment):
    """

    Path to the CMAES file we are supposed to work on. None for new one (default None)

    CMAES_mixed_input -- input for the mixed phase
    CMAES_input -- input for the pure-CMAES phase

    """

    def __init__(
        self,
        exp_id,
        dataDirectory,
        workingFolderDirectory,
        templateFolder,
        muscleGTdir,
        brainFileName,
        CMAES_mixed_input,
        CMAES_input,
        stepSize=0.020,
        scenarioName="circle",
        frequency=0.5,
        bibiMode="<mode>SynchronousDirectNestSimulation</mode>",
        experimentName="Mouse_Experiment_CMAES",
        muscleValuesGTsource="FFT",
        folderPrefix="", folderInfix="", folderSuffix="",
        pid_init_duration=2,  # seconds
        pid_movement_duration=1,  # Periods
        CMAES_mixed_duration=1,  # Periods
        saturation=1,
        CMAES_duration=1,  # Periods
        NESTthreads=8
        # From which algorythem the muscle ground truth values shall come #TODO: make an "CMAES" version
        ):

        #Durations
        self.pid_init_duration = pid_init_duration
        self.pid_movement_duration = pid_movement_duration
        self.CMAES_mixed_duration = CMAES_mixed_duration
        self.CMAES_duration = CMAES_duration
        self.saturation = saturation

        super().__init__(
            exp_id=exp_id,
            frequency=frequency,
            dataDirectory=dataDirectory,
            muscleGTdir=muscleGTdir,
            stepSize=stepSize,
            # be overwritten later by phase definition
            duration=(float("inf"), float("inf")),
            workingFolderDirectory=workingFolderDirectory,
            # We are not interest in the muscle GT values (this varaible is also not used by the superclass anyways)
            templateFolder=templateFolder,
            scenarioName=scenarioName,
            bibiMode=bibiMode,
            experimentName=experimentName,
            brainFileName=brainFileName,
            brainRecreate=False,
            muscleValuesGTsource=muscleValuesGTsource,
            folderPrefix=folderPrefix, folderInfix=folderInfix, folderSuffix=folderSuffix,
            columns=0, Nexec=0, Ninh=0,
            NESTthreads=NESTthreads,
            disableBrain=True  # CMAES does not need the SNN running in the background!
            #noiseSD=noiseSD, lamb=lamb, spectralR=spectralR, shall fall back to default values
        )

        # CMAES stuff
        self.CMAES_mixed_input = CMAES_mixed_input
        self.CMAES_input = CMAES_input

        # Generate data for all phases
        self.setPhases(
            pid_init_duration=pid_init_duration,  # seconds
            pid_movement_duration=pid_movement_duration,  # Periods
            CMAES_mixed_duration=CMAES_mixed_duration,  # Periods
            CMAES_duration=CMAES_duration,  # Periods
        )  # set CMAES specific phases. Also overrides/correct the simulation-duration

        # Set timeout (other pysics parameter etc. are set in the parent class allready)
        self.fileContents["experiment_configuration.exc"] = { "content": self.getExecConfiguration(
            templatePath=templateFolder + "experiment_configuration.exc", timeout=self.duration[0]) }

        # Set TF modifications: PIF control and Muscle control values
        self.TFcontents["joint_controller"] = {"replacements": self.getPIDreplacements(self.PIDvalues, self.PIDcontributionValues)}
        self.TFcontents["muscle_controller"] = {"replacements":self.getMuscleReplacements(self.MuscleValues)}

    ##### PHASES AND THEIR VALUES #####

    def setPhases(self,
                  pid_init_duration,  # seconds
                  pid_movement_duration,  # Periods
                  CMAES_mixed_duration,  # Periods
                  CMAES_duration
                  ):
        """
        Side effect: creates self.PIDvalues, self.MuscleValues and self.FORCEcontributionValues.
        The FORCE control are (in this case) implicitly generated from the scenario parameters.
        """

        self.phases = []

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
            "name": "CMAES mixed",  # phase name
            "t_start": self.phases[-1]["t_start"]+self.phases[-1]["duration"],
            "duration": CMAES_mixed_duration*(1.0/self.frequency)  # phase duration
        })
        if(CMAES_duration > 0): #this phase is actually optional
            self.phases.append({
                "name": "CMAES",  # phase name
                "t_start": self. phases[-1]["t_start"]+self.phases[-1]["duration"],
                "duration": CMAES_duration*(1.0/self.frequency)  # phase duration
            })

        # set duration
        minRuntime = self.phases[-1]["t_start"]+self.phases[-1]["duration"]

        self.duration = (minRuntime, minRuntime+10)

        # get PID, muslce, contribution etc. values for the total runtime
        self.PIDvalues = self.__generatePIDvalues()
        self.MuscleValues = self.__generateMuscleValues()
        self.FORCEcontributionValues = self.__generateFORCEContributionValues()
        self.PIDcontributionValues = self.__generatePIDContributionValues()

    def __generatePIDvalues(self):

        # for easier readbility
        f = self.frequency
        A1 = self.scenario["x_A"]
        p1 = self.scenario["x_p"]
        A2 = self.scenario["y_A"]
        p2 = self.scenario["y_p"]
        P = 1/f

        values = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            if (phase["name"] == "pid init"):  # -> static position of beginning of next phase
                t = phase["t_start"]+phase["duration"]
                # because we want to set the values now for the next timestep to be correct
                t = t + self.stepSize
                # longitudinal movement (Y) - joystick_helper_joint
                Y = A1*np.sin(t*f*2*np.pi+p1)
                # Transversal movement (X) - joystick_world_joint
                X = A2*np.sin(t*f*2*np.pi+p2)
                values_temp = [(X, Y)]*steps
                values.extend([tuple(value) for value in values_temp])
            # -> actual movment according to target
            elif (phase["name"] in ["pid movement", "CMAES mixed"]):
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

            elif (phase["name"] in ["CMAES"]):  # -> idle PID
                values.extend([None]*steps)
            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating PID values')

        return values

    def __generateMuscleValues(self):

        # controls
        f = self.frequency
        X = self.scenario["x_A"]/self.scenario["A_max"]
        Y = self.scenario["y_A"]/self.scenario["A_max"]

        values = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            # FYI: muscle values in this order: ['Foot1', 'Foot2', 'Radius1', 'Radius2', 'Humerus1', 'Humerus2', 'Humerus3', 'Humerus4']

            if (phase["name"] in ["pid init", "pid movement"]):  # -> muscles idle
                values.extend([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*steps)
            elif (phase["name"] == "CMAES mixed"):  # -> CMAES mixed with PID controller
                if(self.CMAES_mixed_input is None):
                    values.extend(self.getMuscleGT(phase, self.scenarioName, f,X,Y, 
                    t_start=phase["t_start"], t_end=phase["t_start"]+phase["duration"]))
                else:
                    values.extend(self.getMuscleInnervation(
                            self.CMAES_mixed_input, phase, self.stepSize))

            elif (phase["name"] == "CMAES"):  # -> only CMAES control
                if(self.CMAES_input is None):
                    values.extend(self.getMuscleGT(phase,self.scenarioName,f,X,Y,t_start=phase["t_start"], t_end=phase["t_start"]+phase["duration"]))
                else:
                    values.extend(self.getMuscleInnervation(
                        self.CMAES_input, phase, self.stepSize))
            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating muscle values')

        return values

    def __generateFORCEContributionValues(self):
        """Defines how much the SNN netowrk output shall contibute to the muscle innervation in each phase  """

        values = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            values.extend([None]*steps)

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
            elif (phase["name"] == "CMAES mixed"):  # -> contribution transition from 1 to 0
                # from 1 to 0 in steps steps (both ends inclusive)
                values.extend(np.linspace(1, 0, steps))
            elif (phase["name"] == "CMAES"):  # -> np contribution = 0/offline
                values.extend([None]*steps)
            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating PID contribution values')

        return values

    ##### TRANSFER FUNCTION CODE #####

    #Nothing special here

    ##### EXPERIMENT RESULTS #####

    def getGT(self, phase=None):
        #from where to where
        if(phase == None):
            ts = np.arange(0, self.duration[0], self.stepSize)
        else:
            ts = np.arange(phase["t_start"],phase["t_start"]+phase["duration"], self.stepSize)
       
        # for easier readbility
        f = self.frequency
        A1 = self.scenario["x_A"]
        p1 = self.scenario["x_p"]
        A2 = self.scenario["y_A"]
        p2 = self.scenario["y_p"]
        P = 1/f

        #that here is actually the code from the PID controller which is also relying on the GT values
        Y = A1*np.sin(ts*f*2*np.pi+p1)   # longitudinal movement (Y) - joystick_helper_joint
        X = A2*np.sin(ts*f*2*np.pi+p2)  # Transversal movement (X) - joystick_world_joint
        
        values_temp = np.transpose([X, Y]) # shape: (2, steps) -> (steps, 2)
        GT = [tuple(value) for value in values_temp]

        return GT

    def loss(self, phase, GT, trajectory):

        index_start = int(phase["t_start"]/self.stepSize)
        steps = int(phase["duration"]/self.stepSize)

        phase["distances"] = []
        for index in range(index_start,index_start+steps):

            gt_x, gt_y = GT[index]
            t_x, t_y, t_z = trajectory[index]
            dx, dy = gt_x-t_x, gt_y-t_y

            dist = np.sqrt(dx**2+dy**2)  #distance from trajectoy to GT point
            
            phase["distances"].append(dist)

        phase["loss"] = np.mean(phase["distances"])

        return phase["loss"]

    def postprocessResults(self):

        super().postprocessResults()  # to the higher abstraction postprocessing

        self.GT = self.getGT()

        self.trajectory = [ state["position"]["rel"] for state in self.resultFiles["CSV"]["states.csv"]["jointStates"]]

        # now the CMAES specific postprocessing
        self.CMAES_mixed_loss, self.CMAES_loss = None, None
        for phase in self.phases:
            if(phase["name"] == "CMAES mixed"):
                self.CMAES_mixed_loss = self.loss(phase, self.GT, self.trajectory)
            elif(phase["name"] == "CMAES"):
                self.CMAES_loss = self.loss(phase, self.GT, self.trajectory)

    ##### VISUALIZATIONS #####

