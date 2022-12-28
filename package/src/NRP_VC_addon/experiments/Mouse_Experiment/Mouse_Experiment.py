from ..Experiment import Experiment

import numpy as np
import pickle  # for muscle GT files

import scipy.interpolate #for CMAES inut streching over all timetemps
import matplotlib.pyplot as plt
import os
import json # to dump stuff in files

class Mouse_Experiment(Experiment): 
    """Represents the experiment configuration of the muscoloskeletal mouse model.

    Expects experiment files to have proper capabilities. E.g. that the PID controller plugin is running.


    disableBrain -- if we do not want to do anything FORCE or SNN related (just physics for example). You can disable the
        Neural network and respective transfere function with this variable (default: False)

    """

    ##### UTILITY FUNCTIONS #####

    # only linear interpolation
    @classmethod
    def interp_numpy(self, in_y, out_x):
         # The input array we have
        in_x = np.linspace(out_x[0], out_x[-1], len(in_y))
        in_y = in_y
        # The output array we want
        out_x = out_x
        out_y = np.interp(out_x, in_x, in_y)
        return out_y

    # allows differnt kinds of interpolation splines
    @classmethod
    def interp_scipy(self, in_y, out_x, kind="linear"):
         # The input array we have
        in_x = np.linspace(out_x[0], out_x[-1], len(in_y))
        in_y = in_y

        in_f = scipy.interpolate.interp1d(in_x, in_y, kind=kind, axis=0)

        # The output array we want
        out_x = out_x
        out_y = in_f(out_x)

        return out_y

    @classmethod
    def getMuscleInnervation(self, CMAES_input, phase, stepSize):
        """ Converts a list of N muscle values from CMAES to an array with values for each time step in the given phase.
        If N < steps, this function linearly interpolates to get the missing values.

        This function assume sthat the first and last element of the CMAES_input realates to te first and last timestep in the phase.
        """

        steps = int(phase["duration"]/stepSize)
        out_y = self.interp_scipy(
            in_y=CMAES_input,
            out_x=np.linspace(phase["t_start"],
                              phase["t_start"]+phase["duration"], steps),
            kind="quadratic" #we assume that the muscle signals are a continuously differentiable function !
        )

        #reformat to a list of lists
        out_y = [list(values) for values in out_y]

        return out_y

    def computeBrainSize(self, plot=False):

        #to gnerate more complex commands
        from past.utils import old_div

        def getCoordinates(ID, xD, yD):
            """
            place population in grid based on id
            :param ID: id of population, long int
            :param xD: x dimensionality of grid, long int
            :param yD: y dimensionality of grid, long int
            :return: cartesian coordinates
            """
            if not (isinstance(ID, (int, int)) & isinstance(xD, (int, int)) & isinstance(yD, (int, int))):
                raise Exception('population ID, xDimension and yDimension must be integer types')
            zD = xD * yD

            z = old_div(ID, zD)
            y = old_div((ID - z * zD), xD)
            x = ID - z * zD - y * xD
            return x, y, z

        def getProb(ID0, ID1, xD, yD, C=0.3, lamb=1.0):
            """
            get distance-based connection probability for pair (ID0,ID1)
            :param ID0: id of population 0
            :param ID1: id of population 1
            :param xD: x dimensionality of grid
            :param yD: y dimensionality of grid
            :param C: parameter to weight connectivity based on connection type (not yet implemented, from maass 2002)
            :param lamb: parameter to in/decrease overall connectivity
            :return: Probability of connection between any two neurons of populations ID0 and ID1
            """
            if ID0 == ID1:
                prob = 0.
            else:
                x0, y0, z0 = getCoordinates(ID0, xD, yD)
                x1, y1, z1 = getCoordinates(ID1, xD, yD)
                d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)  # eucl distance
                prob = C * np.power(np.e, -np.square(old_div(d, lamb)))
            return prob

        def createConnectivityMatrix(N,lamb=1.0):
            """
            create distance based connectivity matrix for N populations
            currently defaults to stacking populations in 3by3 layers

            :param N: number of populations
            :return: connectivity matrix (to, from)
            """
            p_connect = np.empty((N, N))
            for fr in range(N):
                for to in range(N):
                    p_connect[fr, to] = getProb(to, fr, xD=3, yD=3, lamb=lamb)  # (to,from)
            return p_connect

        def get_rand_mat(dim, spec_rad, negative_weights=True, seed=None):
            "Return a square random matrix of dimension @dim given a spectral radius @spec_rad"
            
            if seed:
                rng = np.random.RandomState(seed=seed)
                mat = rng.randn(dim, dim)
            else:
                mat = np.random.randn(dim, dim)
            if not (negative_weights):
                mat = abs(mat)
            w, v = np.linalg.eig(mat)
            mat = np.divide(mat, (old_div(np.amax(np.absolute(w)), spec_rad)))

            return mat


        #Start extracting
        brainLines = self.brainConfig.split("\n")

        # filter dictionary definition lines
        evalStart ="SNN_dict"
        brainLinesEval = []
        for line in brainLines:
            if(evalStart in line or "lamb =" in line ):
                if(line[0:4] == "    "):
                    line = line[4:]
                brainLinesEval.append(line)

        seed = 31337
        import numpy as np
        rng = np.random.RandomState(seed=seed)

        #execute
        locls = locals()
        for line in brainLinesEval:
            try:
                #print("EXEC:"+str(line))
                exec( line, globals(), locls )
            except NameError  as e:
                pass
                print(f'\"{line}\"\n\tNameError: {e}')
            except SyntaxError as e:
                pass
                print(f'\"{line}\"\n\tSyntaxError: {e}')
            except KeyError as e:
                pass
                print(f'\"{line}\"\n\tKeyError: {e}')

        #extract definition
        SNN_dict = locls["SNN_dict"]
        n_sensor = locls["n_sensor"]
        n_hli = locls["n_hli"]
        n_fb = locls["n_fb"]
        n_res = locls["n_res"]
        n_out = locls["n_out"]
        lamb = locls["lamb"]


        #get variables
        n_in = SNN_dict['networkStructure']['n_in']['n_sensor']
        n_res = SNN_dict['networkStructure']['n_res']
        n_readout = SNN_dict['networkStructure']['n_out']
        n_hli = SNN_dict['networkStructure']['n_in']['n_hli']

        p_connect_sensor = SNN_dict['p_connect']['p_connect_in']['p_connect_sensor']
        p_connect_input_hli = SNN_dict['p_connect']['p_connect_in']['p_connect_hli']
        p_connect_fb = SNN_dict['p_connect']['p_connect_in']['p_connect_fb']
        #p_connect_inter = SNN_dict['p_connect']['p_connect_res']
        p_connect_intra_EE = SNN_dict['p_connect']['p_connect_intra']['EE']
        p_connect_intra_EI = SNN_dict['p_connect']['p_connect_intra']['EI']
        p_connect_intra_IE = SNN_dict['p_connect']['p_connect_intra']['IE']

        w_sensor = SNN_dict['weights']['w_in']['w_sensor']
        w_hli = SNN_dict['weights']['w_in']['w_hli']
        w_in = np.hstack((w_sensor,w_hli))
        w_gb = SNN_dict['weights']['w_in']['w_fb']
        #w_res = SNN_dict['weights']['w_res']
        w_out = SNN_dict['weights']['w_out']
        w_intra_EE, w_intra_EI, w_intra_IE = SNN_dict['weights']['w_intra']['EE'], SNN_dict['weights']['w_intra']['EI'], SNN_dict['weights']['w_intra']['IE']

        Nexc = SNN_dict['NexcNinh']['Nexc']
        Ninh = SNN_dict['NexcNinh']['Ninh']

        ##Create missing entries

        #SNN_dict['p_connect']['p_connect_res'] = PU.createConnectivityMatrix(n_res, lamb)
        print(f'Get connectivity matrix for {n_res}x{n_res}={n_res*n_res} connections...')
        SNN_dict['p_connect']['p_connect_res'] = createConnectivityMatrix(n_res,lamb)
        p_connect_inter = SNN_dict['p_connect']['p_connect_res']

        #SNN_dict['weights']['w_res'] = PU.get_rand_mat(n_res,SNN_dict['spectral_R'], negative_weights=True, seed=seed)
        print(f'Get random matrix for {n_res}x{n_res}={n_res*n_res} connections...')
        SNN_dict['weights']['w_res'] = get_rand_mat(n_res,SNN_dict['spectral_R'], negative_weights=True, seed=seed)
        w_res = SNN_dict['weights']['w_res']

        #Population size
        sensor_populations = (n_in+n_hli)*Nexc
        sensor_monitor_population = (n_in+n_hli)
        hidden_populations = n_res*(Nexc+Ninh)
        monitor_population = n_res
        readout_neuron_populations = n_readout

        print(f'Neuron populations (total={sensor_populations+sensor_monitor_population+hidden_populations+monitor_population+readout_neuron_populations}):')
        print(f'\tsonsor_population:\t{sensor_populations}')
        print(f'\tsensor_monitor_pop:\t{sensor_monitor_population}')
        print(f'\thidden_populations:\t{hidden_populations}')
        print(f'\tmonitor_population:\t{monitor_population}')
        print(f'\treadout_neuron_pop:\t{readout_neuron_populations}')

        #Synapse count
        projections_sensor_monitor = (n_in+n_hli)*Nexc
        projections_sensor_hiddenP = n_res*(n_in+n_hli)*Nexc*SNN_dict['p_connect']['p_connect_in']['p_connect_sensor'] #n_res not hidden_population because pop[0]
        projections_hiddenP_monitor = n_res*Nexc
        projections_hiddenP_readout = n_readout*n_res*Nexc
        projections_hiddenP_hiddenP = np.sum(p_connect_inter.flatten())*(Nexc*Nexc)#n_res*n_res*Nexc
        projections_intrapopulation = n_res * (Nexc*Ninh*p_connect_intra_EI+Nexc*Nexc*p_connect_intra_EE+Ninh*Nexc*p_connect_intra_IE)

        print(f'\nNeuron projections (total ca. {int(projections_sensor_monitor+projections_sensor_hiddenP+projections_hiddenP_monitor+projections_hiddenP_readout+projections_hiddenP_hiddenP+projections_intrapopulation)}):')
        print(f'\tprojections_sensor_monitor:\t{projections_sensor_monitor}')
        print(f'\tprojections_hiddenP_monitor:\t{projections_hiddenP_monitor}')
        print(f'\tprojections_hiddenP_readout:\t{projections_hiddenP_readout}')
        print(f'\tprojections_hiddenP_hiddenP:\t{projections_hiddenP_hiddenP}')
        print(f'\tprojections_intrapopulation:\t{projections_intrapopulation}')

        if(plot): #plots matrices
            import matplotlib.pyplot as plt

            plt.figure()
            plt.matshow(SNN_dict['p_connect']['p_connect_res'])
            plt.title("P connections reservoir neurons")
            plt.show()

            plt.figure()
            plt.matshow(SNN_dict['weights']['w_res'])
            plt.title("Connections weights - reservoir")
            plt.show()

    # get Scenario parameters based on the scenario name
    @classmethod
    def getScenarioParameters(self, scenarioName):
        if (scenarioName == "longitudinal"):
            scenario = {
                "scenario": scenarioName,
                "A_max": 0.05,  # maximal distance from origin. required for error metric computation
                "x_A": 0.05, "x_p": 0,
                "y_A": 0,    "y_p": 0
            }
        elif (scenarioName == "transversal"):
            scenario = {
                "scenario": scenarioName,
                "A_max": 0.05,
                "x_A": 0,    "x_p": 0,
                "y_A": 0.05, "y_p": np.pi*0.5
            }
        elif (scenarioName == "circle"):
            scenario = {
                "scenario": scenarioName,
                "A_max": 0.05,
                "x_A": 0.05,  "x_p": 0,
                "y_A": 0.05,  "y_p": np.pi*0.5
            }
        else:
            raise RuntimeError(
                f"ERROR: scenario \"{scenarioName}\" not defined")
        return scenario


    ##### INIT OF CLASS #####

    def __init__(
        self,
        exp_id,
        dataDirectory,
        workingFolderDirectory,
        templateFolder,
        muscleGTdir,
        stepSize,
        duration,
        frequency,
        scenarioName,
        brainFileName, #where the brain lies or is supposed to be stored
        brainRecreate=False, #wheter to create the big brain parts new if the file allready exists
        bibiMode=None,
        experimentName="Mouse_Experiment",
        muscleValuesGTsource="FFT",
        PID=(1.5, 0.0, 0.015),
        folderPrefix="", folderInfix="", folderSuffix="",
        columns=3*3*2, Nexec=30, Ninh=10,
        NESTthreads=8,
        noiseSD=0.2, lamb=6.15, spectralR=4.13,
        #p_shotNoise=0.01, #TODO: wo muss das hin? das wird momentan nur in der FORCE klasse genahdelt
        #LPfilter=1, brauchen wir das?
        disableBrain=False
        ):

        super().__init__(
            exp_id=exp_id,
            dataDirectory=dataDirectory,
            workingFolderDirectory=workingFolderDirectory,
            stepSize=stepSize,
            duration=duration,
            experimentName=experimentName,
            folderPrefix=folderPrefix, folderInfix=folderInfix, folderSuffix=folderSuffix
        )

        # Physics paramaters
        self.PID = PID

        # Experiment general parameters
        self.frequency = frequency

        # SNN specific stuff
        self.disableBrain = disableBrain
        self.brainFileName = brainFileName
        self.brainRecreate = brainRecreate
        self.columns = columns

        # Muscle GT
        self.muscleGTdir = muscleGTdir
        self.muscleValuesGTsource = muscleValuesGTsource
        self.muscleNames = ["Foot1","Foot2","Radius1","Radius2","Humerus1","Humerus2","Humerus3","Humerus4"] #mapping between indices and the muscle names (is mostly for convinience and human readability)
        self.GTfiles = self.getGTfiles(self.muscleGTdir,self.muscleValuesGTsource)

        # Scenario sepcific parameters
        self.scenarioName = scenarioName
        self.scenario = self.getScenarioParameters(scenarioName)

        # Get experiment files content based on template files
        self.templateFolder = templateFolder

        self.fileContents["robobrain_mouse_with_joystick/model.sdf"] = {"content": self.getSDFConfiguration(
            templatePath=templateFolder+"model.sdf",
            p=PID[0], i=PID[1], d=PID[2])}

        self.bibiMode = bibiMode
        self.fileContents["bibi_configuration.bibi"] = {"content": self.getBIBIconfiguration(
            templatePath=templateFolder+"bibi_configuration.bibi", bibiMode=bibiMode)}

        self.brainConfig = self.getBrainConfiguration( #we might need that for brain size computation
            templatePath=templateFolder+"FRC_CPG_brain.py",
            columns=columns, Nexec=Nexec, Ninh=Ninh, noiseSD=noiseSD, 
            brainFileName=brainFileName, brainRecreate=brainRecreate,
            threads=NESTthreads,
            lamb=lamb, spectralR=spectralR)
        self.fileContents["FRC_CPG_brain.py"] = {"content": self.brainConfig}
        # Timeout is not set here but in the child class because it depends on the actual phases

        # Define result files
        # every mouse experiment is supposed to log the musce & joint states
        self.resultFiles["CSV"]["states.csv"] = None

        #self.resultFiles["CSV"]["muscle_controller.csv"] = None
        self.resultFiles["CSV"]["joint_controller.csv"] = None

        ##Now initialize the brain
        if(self.disableBrain == False):
            self.initializeBrain()

    ##### PHASES AND THEIR VALUES #####

    #just read all the available GT files
    def getGTfiles(self, muscleGTdir, muscleValuesGTsource):

        if(muscleValuesGTsource=="FFT"):
            dir = muscleGTdir+muscleValuesGTsource+"/"

            fileNames = os.listdir(dir)
            files = []
            for fileName in fileNames:
                print(fileName)
                split = fileName[:-4] #remove extension
                split = split.split("_") # split into arguments
                # filenames have following format: f'{scenarioName}_f={f}_periods={periods}.pkl'

                #get Parameter values
                scenarioDict = {
                    "scenario": split[0],
                    "path":dir+fileName
                    }
                for parameterString in split[1:]:
                    parameter, value = parameterString.split("=")
                    scenarioDict[parameter] = value

                #now get Data
                path = dir+fileName
                with open(path, 'rb') as f:
                    scenarioDict["muscleGTWaveSpecification"] = pickle.load(f)

                files.append(scenarioDict)

            return files
        elif( muscleValuesGTsource == "CMA"):
            return None
            #TODO: do we need to do something here?
            pass
        else:
            raise RuntimeError(f'Unkown GT source \"{muscleValuesGTsource}\"')


    # - FFT: FFT found innervation (TODO: generate missing!)
    # - CMA: CMA optimized FFT version (TODO: generate these!)
    def getMuscleGT(self, phase, scenarioName, f, X, Y, t_start, t_end, periods=100
        ):
        """gets the muscle innervation signals we considere corect for the given scenario.
        This GT values might be based on different sources/algorythms.

        ATTENTION: when getting muscle innervatiosn from a CAMES file, the phase shift needs to match the CMAES runner phase shift.
        """

        #print(f'Get muscle values for phase {phase["name"]} (mode={self.muscleValuesGTsource}) : ')

        if (self.muscleValuesGTsource == "FFT"):
            
            #print( f'INFO: Searching for controls file(s) for f={f:.04f}, X={X:.04f}, Y={Y:.04f}')


            #get files that match closest
            closeness = []
            relevantFiles = []
            for fileDict in self.GTfiles:
                if fileDict["scenario"] != scenarioName:
                    #print(f'{fileDict["path"]} is wrong scenario')
                    continue
                if fileDict["periods"] != str(periods):
                    #print(f'{fileDict["path"]} has wrong periods ({fileDict["periods"]} instad of {periods})')
                    continue

                dist = f-float(fileDict["f"])
                closeness.append(dist)
                relevantFiles.append(fileDict)

            if(len(relevantFiles) == 0):
                raise RuntimeError(f'For controls f={f}, X={X}, Y={Y} were no GT files found in {self.muscleGTdir+self.muscleValuesGTsource+"/"}')       

            #extract wave specification from files
            if 0.0 in closeness: #there is one file matching exactly
                for i, dist in enumerate(closeness):
                    if dist == 0.0: #that is the file
                        self.muscleGTWaveSpecification = relevantFiles[i]["muscleGTWaveSpecification"]
                        print(f'Relevant GT file: {relevantFiles[i]["path"]}')

            else: #get two clostest files and interpolate between them
                d1,d2 = sorted(np.abs(closeness))[:2]
                f1 = None
                for i, dist in enumerate(closeness):
                    if np.abs(dist) == d1 and f1 == None: #closest file
                        f1 = relevantFiles[i]
                    elif np.abs(dist) == d2: #second closest file
                        f2 = relevantFiles[i]

                print(f'Relevant GT files:\n\tclosest:\t{f1["path"]} \n\t2. closest:\t{f2["path"]}')

                self.muscleGTWaveSpecification = {} #create new wave specification

                for muscleName in self.muscleNames:

                    wave1 = f1["muscleGTWaveSpecification"][muscleName]
                    wave2 = f2["muscleGTWaveSpecification"][muscleName]

                    #get percentage for interpolation
                    p1 = d1/(d1+d2)
                    p2 = d2/(d1+d2)
                    
                    #create components
                    components = []
                    for component in wave1["components"]:
                        freq, Aamp, period = component
                        Aamp *= p1
                        freq = f
                        components.append( (freq, Aamp, period) )
                    for component in wave2["components"]:
                        freq, Aamp, period = component
                        Aamp *= p2
                        freq = f
                        components.append( (freq, Aamp, period) )
                    
                    #add other information for wave speciication
                    self.muscleGTWaveSpecification[muscleName] = {
                            "components":components,
                            "Amod":wave1["Amod"]*p1+wave2["Amod"]*p2,
                            "y_offset":wave1["y_offset"]*p1+wave2["y_offset"]*p2,
                            "p_offset":wave1["p_offset"]*p1+wave2["p_offset"]*p2,
                            "MSE":wave1["MSE"]*p1+wave2["MSE"]*p2 #this entry does not amke much sense as there is no GT to compute a MSE for..
                            }

            # convert wave description to target value
            muscleNames = ['Foot1', 'Foot2', 'Radius1', 'Radius2',
                           'Humerus1', 'Humerus2', 'Humerus3', 'Humerus4']
            ts = np.arange(t_start, t_end, self.stepSize)
            muscleValues = np.zeros(
                (len(muscleNames), len(ts)))  # output strucutre
            for i, muscle in enumerate(muscleNames):
                specification = self.muscleGTWaveSpecification[muscle]
                components = specification["components"]
                Amod = -specification["Amod"] #TODO: I am not 100% certain why we need the inverse here!
                off = specification["y_offset"]
                p_off = specification["p_offset"]

                values = 0
                for f, A, p in components:
                    values += Amod*A*np.sin(f*2*np.pi*np.array(ts)+p+p_off)
                values += off

                muscleValues[i] = values

            # shape (8, len(ts) ) -> (len(ts), 8)
            muscleValues = np.transpose(muscleValues)
            muscleValues = [tuple(value) for value in muscleValues]

        elif self.muscleValuesGTsource == "CMA": #TODO: needs to be filled/fixed when we have the CMA-ES GT data!

            if(phase["name"] == "CMAES mixed"):
                # switch case for respective CMAES files TODO: maks this less hard coded...
                if f == 1 and X == 1 and Y == 1:
                    fileName = ""
                elif f == 1 and X == 1 and Y == 0:
                    fileName = ""
                elif f == 1 and X == 0 and Y == 1:
                    fileName = ""
                elif f == 0.5 and X == 1 and Y == 1:
                    fileName = "cmaes_mixed_0.5.pkl"
                elif f == 0.5 and X == 1 and Y == 0:
                    fileName = ""
                elif f == 0.5 and X == 0 and Y == 1:
                    fileName = ""
                elif f == 0.25 and X == 1 and Y == 1:
                    fileName = ""
                elif f == 0.25 and X == 1 and Y == 0:
                    fileName = ""
                elif f == 0.25 and X == 0 and Y == 1:
                    fileName = ""
                else:
                    raise RuntimeError(
                        f'No CMAES muscle innervation file found for controls f={f}, X={X} and Y={Y}.')
            elif(phase["name"] == "CMAES"): #TODO: generate CMAES files and then return their content here instead of this dummy data

                steps = int(phase["duration"]/self.stepSize)
                v = 0.5
                return [(v,v, v,v, v,v,v,v)]*steps

            # get GT target
            dataPath = self.muscleGTdir+self.muscleValuesGTsource+"/"+fileName
            #print(f'Loading {dataPath}')

            with open(dataPath, 'rb') as f:
                cmaes = pickle.load(f)

            xbest = cmaes.cmaes_object.result.xbest
            muscleValuesPeriod = self.getMuscleInnervation(xbest, cmaes.phase, self.stepSize) #muscle innervations for a single period
            
            #TODO: What if "CMAES" period needs the same values multiple times?!?! 
            #periods = int(phase["duration"]*self.frequency) #how many periods we actually need
            muscleValues = muscleValuesPeriod #* periods

        else:
            raise RuntimeError(
                f"Unknown muscle innervation gt source: \"{self.muscleValuesGTsource}\"")

        return muscleValues


    #### THE BRAIN NETWORK ####

    def initializeBrain(self):

        #filename = f'hidden_projectionsx{SNN_dict["networkStructure"]["n_res"]}.txt'
        pass


    ##### EXPERIMENT RESULTS #####

    def getExecConfiguration(self, templatePath, timeout):

        replacements = [
            ("[MARKER_TIMEOUT]", str(int(np.ceil(timeout))))  # set timeout
        ]

        fileContent = self.replaceInFile(
            filePath=templatePath, replacements=replacements)

        return fileContent

    def getSDFConfiguration(self,
                            templatePath,
                            p=1.5, i=0.0, d=0.015,
                            world_humerus=50e-4,
                            humerus_helper_joint=50e-4,
                            humerus_radius=12e-4,
                            radius_foot=60e-4,
                            foot_joystick=100e-4,
                            joystick_helper_joint=50e-4,
                            joystick_world_joint=50e-4
                            ):

        replacements = [
            ("[MARKER_P]",                      str(p)),
            ("[MARKER_I]",                      str(i)),
            ("[MARKER_D]",                      str(d)),
            ("[MARKER_world_humerus]",          str(world_humerus)),
            ("[MARKER_humerus_helper_joint]",   str(humerus_helper_joint)),
            ("[MARKER_humerus_radius]",         str(humerus_radius)),
            ("[MARKER_radius_foot]",            str(radius_foot)),
            ("[MARKER_foot_joystick]",          str(foot_joystick)),
            ("[MARKER_joystick_helper_joint]",  str(joystick_helper_joint)),
            ("[MARKER_joystick_world_joint]",   str(joystick_world_joint))
        ]

        fileContent = self.replaceInFile(
            filePath=templatePath, replacements=replacements)

        return fileContent

    def getBrainConfiguration(self,
                              templatePath,
                              columns, Nexec, Ninh,
                              brainFileName, brainRecreate,
                              threads=8,
                              noiseSD=0.2, lamb=6.15, spectralR=4.13
                              ):

        replacements = [
            ("[MARKER_THREADS]",   str(int(threads))),
            ("[MARKER_COLUMNS]",    str(int(columns))),
            ("[MARKER_NOISE_SD]",   str(float(noiseSD))), #defualt 2
            ("[MARKER_Nexc]",       str(int(Nexec))),
            ("[MARKER_Ninh]",       str(int(Ninh))),
            ("[MARKER_LAMB]",       str(float(lamb))), #parameter to in/decrease overall connectivity, default 6.15
            ("[MARKER_SPECTRAL_R]", str(float(spectralR))), #spectral radius, default 4.13
            ("[MARKER_BRAIN_FILE]", "\""+str(brainFileName)+"\"" if brainFileName != None else "None"),
            ("[MARKER_BRAIN_RECREATE]", str(brainRecreate)),
        ]

        fileContent = self.replaceInFile(
            filePath=templatePath, replacements=replacements)

        return fileContent

    def getBIBIconfiguration(self,
                             templatePath,
                             bibiMode):

        if (self.disableBrain):  # if we do not want the brain to be loaded
            replacements = [
                ("[MARKER_BRAIN_MODEL]", ""),
                ("[MARKER_FRC_tf_SNN_FORCE]", ""),
                ("[MARKER_FRC_send_sensor2brain]", ""),
                ("[MARKER_MODE]",bibiMode if bibiMode != None else "") #needs to be set even if we do not have a brain
            ]

        else:  # the default bibi has everything active
            replacements = [
                ("[MARKER_BRAIN_MODEL]",
                 "  <brainModel>\n    <file>FRC_CPG_brain.py</file>\n  </brainModel>"),
                ("[MARKER_FRC_tf_SNN_FORCE]",
                 f'  <transferFunction src="FRC_tf_SNN_FORCE.py" active="true" priority="0" xsi:type="PythonTransferFunction" />'),
                ("[MARKER_FRC_send_sensor2brain]",
                 f'  <transferFunction src="FRC_send_sensor2brain.py" active="true" priority="0" xsi:type="PythonTransferFunction" />'),
                ("[MARKER_MODE]",
                    bibiMode if bibiMode != None else "")

            ]  # currently we do not want to replace anything there. Just the content of the template

        fileContent = self.replaceInFile(
            filePath=templatePath, replacements=replacements)

        return fileContent

    ##### TRANSFER FUNCTION CODE #####

    # PID controller signals. Is called by the respective subclass
    def getPIDreplacements(self, PIDvalues, PIDcontributionValues):
        """Creates the TF replacements for the joint controller TF

        ATTENTION: requires self.PIDcontributionValues to be set beforehand
        """

        #convert PIDvalues to textfile
        fileName_PID = "INJECTION_PIDvalues.txt" 
        self.fileContents[fileName_PID] = {"content": json.dumps(PIDvalues)}

        #convert contribution Values to textfile
        fileName_PIDcont = "INJECTION_PIDcont.txt" 
        self.fileContents[fileName_PIDcont] = {"content": json.dumps(PIDcontributionValues)}
        
        #NOTE: NRP TF does not allow eval calls

        marker = "    #INJECTION_MARKER\n"
        code = f"""

    valueLists = valueLists.value
    contributionValues = contributionValues.value
    if isinstance(valueLists, type(None)):
        import json

        with open(\"{fileName_PID}\", "r") as f:
            valueLists_str = f.read()
            valueLists = json.loads(valueLists_str)

        with open(\"{fileName_PIDcont}\", "r") as f:
            contributionValues_str = f.read()
            contributionValues = json.loads(contributionValues_str)

    index = int(t/{self.stepSize})

    p, i, d = {self.PID}

    if( len(valueLists) > index):
        v = valueLists[index]
        c = contributionValues[index]

        #set contribution
        if( index > 0 and contributionValues[index] != contributionValues[index-1]): #only change PID controller values if they changed
            c = 0 if c==None else c
            pid_controller.value( SetPIDParametersRequest("joystick_world_joint", p*c, i*c, d*c ) )
            pid_controller.value( SetPIDParametersRequest("joystick_helper_joint", p*c, i*c, d*c ) )
            
        #set values
        if(v != None and valueLists[index] != valueLists[index-1]):
            X,Y = v
            joystick_world_joint.send_message(X)
            joystick_helper_joint.send_message(Y)
    
    """
        replacements = [(marker,    code)]
        return replacements

    # Muscle signals. Is called by the respective subclass
    def getMuscleReplacements(self, MuscleValues):
        """Create the TF replacements for the muscle controller TF """

        #convert contribution Values to textfile
        fileName = "INJECTION_MuscleValues.txt" 
        self.fileContents[fileName] = {"content": json.dumps(MuscleValues)}

        marker = "    #INJECTION_MARKER\n"
        code = f"""

    valueLists = valueLists.value
    if isinstance(valueLists, type(None)):    
        import json

        with open(\"{fileName}\", "r") as f:
            valueLists_str = f.read()
            valueLists = json.loads(valueLists_str)

    index = int(t/{self.stepSize})

    if( len(valueLists) > index):
        muscle_actuation.value = valueLists[index]
    
    """
        replacements = [(marker,    code)]
        return replacements

    ##### EXPERIMENT RESULTS #####

    #see self.resultFiles for all the data extracted from the CSVs
    def postprocessResults(self):

        # in the mouse experiment we just use CSV files as output
        for filename in self.resultFiles["CSV"].keys():
            print(f'[CSV] Postprocessing {filename}')
            self.resultFiles["CSV"][filename] = self.readCSV(
                rawFileContent=self.resultFiles["CSV"][filename])

    ##### VISUALIZATIONS #####

    def plotData(self,
            data, #either 1D or 2D. 2D might be table (dict!) or 
            key=None,
            phaseName=None, selectIndices = None,
            seperatePlots = False,
            fig_width = 8,
            fig_height = 4,
            flat=False,
            tlim = (None,None),
            ylim = (None,None),
            log=False,
            legend=True, title=None,
            fileName=None,
            pColor="red", pStyle="--", pWidth=0.5
            ):
        """
        
        key -- if None (default) assumes data is a 1D or 2D list we can directy plot. Otherwise we assume that
            data s a table and key specifies the column (consitig of 1D or 2D data) we want to plot.
        phaseName -- if None (default) draws all phases. Otherweise draws selected phase (by name).
        muscleNames -- filter for specific muscles (requires that the data actually contains muscle related data!)
        seperatePlots -- for each data row a seperate plot (default False)
        flat -- if data is really just one array of data
        tlim -- upper and lower bound for more restrictive display concerning time. is supposed to be in seconds. If the data is within tlim anyways,
            this parameter has no effect. Set to None to make boundary open (default: (None, None))
        legend -- plots legend or not (defualt: False)
        log -- logarythmic y scale (default: False)
        
        """
            
        tlim = list(tlim)
        #Get plot boundaries from phase
        if(phaseName != None):
            #find phase
            found = False
            for phase in self.phases:
                if(phase["name"] == phaseName):
                    if(tlim[0] == None or tlim[0] < phase["t_start"]):
                        tlim[0] = phase["t_start"]
                    if(tlim[1] == None or tlim[1] > phase["t_start"]+phase["duration"]):
                        tlim[1] = phase["t_start"]+phase["duration"]
                    found = True
                    break
            if(found == False):
                raise RuntimeError(f'Phase \"{phaseName}\" does not exist. Existing phases are: {[phase["name"] for phase in self.phases]}')
        
        #reformat data and add timestamps
        if(key != None): #If it is a table
            #set tlim if any values are none
            tlim = [
                0 if tlim[0] == None else tlim[0],
                self.duration[0] if tlim[1] == None else tlim[1] ]
            #print("Tlim"+str(tlim))
            
            #plot data
            #print(f'Table original shape: {np.array(data).shape}')
            column = np.array(data[key])
            #print(f'Column shape: {column.shape}')
            
            #time data
            ts = data["time"]
            
            #remove unnecesarry datapoints (based on tlim)        
            ts_temp = []
            d_temp = []
            for t, point in zip(ts,column):
                if( t >= tlim[0] and t<=tlim[1]):
                    ts_temp.append(t)
                    d_temp.append(point)
            
            ts = np.array(ts_temp)
            d = np.array(d_temp).transpose()
            
        else: #if it is no table
            #set tlim if any values are none
            tlim = [
                0 if tlim[0] == None else tlim[0],
                len(data)*self.stepSize if tlim[1] == None else tlim[1] ]
            #print("Tlim"+str(tlim))
            
            #plot data
            if(flat):
                data = [[value] for value in data]
            tupleLen = np.max([(len(value) if value != None else 0) for value in data]) #get tuple length
            data = [([None]*tupleLen if value == None else value) for value in data] #replace non entries with lists of none to allow transpose
            
            d = np.array(data).transpose()
            #print(f'Data shape: {d.shape}')
            
            #time data
            ts = np.arange(len(data))*self.stepSize
            
            #remove unnecesarry datapoints (based on tlim)
            stepLim = [ int(tlim[0]/self.stepSize), int(tlim[1]/self.stepSize)]
            print("stepLim"+str(stepLim))
            
            ts = ts[stepLim[0]:stepLim[1]]
            if(len(d.shape) == 1):
                d = d[stepLim[0]:stepLim[1]] 
            elif(len(d.shape) == 2): #2D
                d_temp = []
                for i, dim in enumerate(d):
                    d_temp.append( dim[stepLim[0]:stepLim[1]] )
                d = np.array(d_temp)
            
        #print("d.shape"+str(d.shape))
        #print("ts.shape"+str(ts.shape))
        
        if(selectIndices != None): #if we only want a selection of the data rows
            d = d[selectIndices]
            
        #create figure
        nrows = 1 if seperatePlots == False else len(d)
        
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharey='row', sharex='all',
            figsize=(fig_width,nrows*fig_height),
            squeeze=False)

        if(log): #log scale
            for row in range(nrows):
                axs[row][0].set_yscale("log")
        
        #1D data
        if(len(d.shape) == 1): 
            if(len(d) - len(ts) == -1): #if we work with differences or similar, we have one value too less
                ts = ts[1:]
            axs[0][0].plot(ts,d)
            
        #2D data
        elif(len(d.shape) == 2): #2D
            for i, dim in enumerate(d): #the last one is the HLI neuron
                if(len(d) - len(ts) == -1): #if we work with differences or similar, we have one value too less
                    ts = ts[1:]
                if(seperatePlots):
                    axs[i][0].plot(ts,dim,label=str(i))
                else:
                    axs[0][0].plot(ts,dim,label=str(i))
        else:
            plt.show()
            print(f'ERROR: data to be plotted has more than 2 dimenstions (shape={d.shape} )!')
            return None
        
        #draw phases
        for phase in self.phases:
            if(seperatePlots):
                for row in range(nrows):
                    if(phase["t_start"] >= tlim[0] and phase["t_start"] <= tlim[1]):
                        axs[row][0].axvline(phase["t_start"], color = pColor, linestyle=pStyle, linewidth=pWidth)
                    if(phase["name"] == self.phases[-1]["name"] and phase["t_start"]+phase["duration"] <= tlim[1]): #if its the last phase
                        axs[row][0].axvline(phase["t_start"]+phase["duration"], color = pColor, linestyle=pStyle, linewidth=pWidth)
            else:
                if(phase["t_start"] >= tlim[0] and phase["t_start"] <= tlim[1]):
                    axs[0][0].axvline(phase["t_start"], color = pColor, linestyle=pStyle, linewidth=pWidth)
                if(phase["name"] == self.phases[-1]["name"] and phase["t_start"]+phase["duration"] <= tlim[1]): #if its the last phase
                    axs[0][0].axvline(phase["t_start"]+phase["duration"], color = pColor, linestyle=pStyle, linewidth=pWidth)
                
        if(legend==True):
            for row in range(nrows):
                axs[row][0].legend()
        
        for row in range(nrows):
            axs[row][0].set_ylabel("Value")
            axs[row][0].set_ylim(ylim)
                
        axs[-1][0].set_xlabel("time (s)")
        if(title != None):
            axs[0][0].set_title(title)
        elif(key != None):
            axs[0][0].set_title(key)

        fig.tight_layout()

        if(fileName != None):
            plt.savefig(self.workingFolder+fileName)

        plt.show()
