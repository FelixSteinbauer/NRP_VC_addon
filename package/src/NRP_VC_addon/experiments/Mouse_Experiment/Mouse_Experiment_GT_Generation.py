
from .Mouse_Experiment import *

from scipy import signal #for find_peaks for FFT analysis
import pickle

from scipy.ndimage import uniform_filter1d # for sliding window filter

class Mouse_Experiment_GT_Generation(Mouse_Experiment):
    """

    Generates the Muscle GT values from muscle length recordings at PID forced movement 

    """

    def __init__(
        self,
        exp_id,
        dataDirectory,
        workingFolderDirectory,
        templateFolder,
        muscleGTdir, #where the data shall go
        brainFileName,
        stepSize=0.020,
        scenarioName="circle",
        frequency=0.5,
        experimentName="Mouse_Experiment_GT_Generation",
        muscleValuesGTsource="FFT", #this determines whcih algorythem is used to generate the GT
        folderPrefix="", folderInfix="", folderSuffix="",
        pid_init_duration=2,  # seconds
        pid_movement_duration=20,  # Periods
        NESTthreads=8
        ):

        #Durations
        self.pid_init_duration = pid_init_duration
        self.pid_movement_duration = pid_movement_duration

        super().__init__(
            exp_id=exp_id,
            frequency=frequency,
            dataDirectory=dataDirectory,
            muscleGTdir=muscleGTdir,
            brainFileName=brainFileName,
            stepSize=stepSize,
            # be overwritten later by phase definition
            duration=(float("inf"), float("inf")),
            workingFolderDirectory=workingFolderDirectory,
            # We are not interest in the muscle GT values (this varaible is also not used by the superclass anyways)
            templateFolder=templateFolder,
            scenarioName=scenarioName,
            experimentName=experimentName,
            muscleValuesGTsource=muscleValuesGTsource,
            folderPrefix=folderPrefix, folderInfix=folderInfix, folderSuffix=folderSuffix,
            columns=0, Nexec=0, Ninh=0,
            NESTthreads=NESTthreads,
            disableBrain=True  # GT generation does not need the SNN running in the background!
        )


        # Generate data for all phases
        self.setPhases(
            pid_init_duration=pid_init_duration,  # seconds
            pid_movement_duration=pid_movement_duration  # Periods
        )  # set pahse specific values. Also overrides/corrects the simulation-duration

        # Set timeout (other pysics parameter etc. are set in the parent class allready)
        self.fileContents["experiment_configuration.exc"] = { "content": self.getExecConfiguration(
            templatePath=templateFolder + "experiment_configuration.exc", timeout=self.duration[0]) }

        # Set TF modifications: PIF control and Muscle control values
        self.TFcontents["joint_controller"] = {"replacements": self.getPIDreplacements(self.PIDvalues, self.PIDcontributionValues)}
        self.TFcontents["muscle_controller"] = {"replacements":self.getMuscleReplacements(self.MuscleValues)}

    ##### PHASES AND THEIR VALUES #####

    def setPhases(self,
                  pid_init_duration,  # seconds
                  pid_movement_duration  # Periods
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

        # set duration
        minRuntime = self.phases[-1]["t_start"]+self.phases[-1]["duration"]

        self.duration = (minRuntime, minRuntime+10) #TODO: use the upper limit for anything and decide on what the 10 means here

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
            elif (phase["name"] in ["pid movement"]):
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

            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating PID values')

        return values

    def __generateMuscleValues(self):

        values = []

        for phase in self.phases:
            steps = int(phase["duration"]/self.stepSize)  # steps in this phase

            # FYI: muscle values in this order: ['Foot1', 'Foot2', 'Radius1', 'Radius2', 'Humerus1', 'Humerus2', 'Humerus3', 'Humerus4']

            if (phase["name"] in ["pid init", "pid movement"]):  # -> muscles idle
                values.extend([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]*steps)
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

            # full contribution
            if (phase["name"] in ["pid init", "pid movement"]):
                values.extend([1]*steps)
            else:
                raise RuntimeError(
                    f'ERROR: unkown phase name \"{phase["name"]}\" for generating PID contribution values')

        return values

    ##### TRANSFER FUNCTION CODE #####

    #Nothing special here

    ##### EXPERIMENT RESULTS #####

    # get FFT values (frequencies, amplitudes, phase(in rad), raw complex values)
    # freqRange narrows down allowed frequenceies. limits are inclusive
    def getFFT(self, ts, values, freqRange=(None,None), freq_amp_thr=0.01):

        frequencies = np.fft.fftfreq(len(values), self.stepSize) #frequencies (x axis)
        Z = (2.0/len(values)) * np.fft.fft(values) # complex part (y-axis)

        cuttoff=int(len(values)/2)
        frequencies = frequencies[:cuttoff]
        Z = Z[:cuttoff]
        
        #convert the frequency range to indices representing this range
        lim = [0,len(frequencies)-1] 
        if freqRange != (None,None):
            for i,f in enumerate(frequencies):
                if freqRange[0] != None and f < freqRange[0]:
                    lim[0] = i
                if freqRange[1] and f > freqRange[1]:
                    lim[1] = i
                    break

            frequencies = frequencies[lim[0]:lim[1]+1]
            Z = Z[lim[0]:lim[1]+1]
        
        #get polar coordinates
        A = np.abs(Z)[:cuttoff] # np.sqrt(a*a+b*b)
        #rad = np.angle(comp)

        # from https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/
        Z_filtered=Z.copy(); #store the FFT results in another array
        
        #detect noise (very small numbers (eps)) and ignore them
        eps = 0.0001
        Z_filtered[A<eps] = 0; #maskout values that are below the threshold
        #p=np.angle(Z_filtered)
        p=np.arctan2(Z_filtered.imag,Z_filtered.real); #phase information

        return frequencies, A, p, Z

    #computes MSE of two equally long lists
    def mse(self,l1,l2): #reuqire for FFT vs actual signal error computaiton
        l1 = np.array(l1)
        l2 = np.array(l2)
        error = l1-l2
        squared_error = np.power(error,2)
        mean_squared_error = np.mean(squared_error)
        return mean_squared_error

    def find_peaks(self, values):

        #SCIPY - FIND_PEAKS (not satisfactory!)
        #prominence = 0.07*max(values) # 0.07 is found empirically and describes the trade-off between false positives and sensitivity
        #peak_indices, peak_dict = signal.find_peaks(values, prominence=prominence)
        #print(f"Peak dicts: {peak_dict}")

        #SCIPY - FIND_PEAKS_CWT (not satisfactory!)
        #width = 5
        #min_snr=2
        #window_size=len(fs) #all data
        #peak_indecs = signal.find_peaks_cwt(values,width,min_snr=min_snr, window_size=window_size)

        #SELF MADE PEAK FINDER (based on threshold noise filtering, not satisfactory because not generic enough)
        # Assumption: additive noise from 0
        # Assumption: less signal peaks than noise peaks (otherwise median is off!)
        # Assumption: much less peaks than noise peaks (otherwise median is not in the middle of noise) = accuracy

        #assume normal distribution of noise
        #p_positive = 0.005 # 1% 
        #thr = np.quantile(values, 1-p_positive)
        
        #peak_indices = np.where(values >= thr)

        # MODIFIED Z-SCORE (rather satisfactory because sientifically founded)
        # also assumes that the median and std is rather representative for a normal distribution
        m=5.5 # relatively optimized for our scenarios

        MAD = np.median(np.abs(values-np.median(values)))
        M = (0.6745*(values-np.median(values)))/MAD

        peak_indices = np.where(M>m)[0]


        return peak_indices

    #extracts possible sinus wave combinations making up the input signal
    def getWaveSpecification(self, Amod=1, offMod=1, freqRange=(None,None), freq_amp_thr=0.01,):
    
        start_index = int(self.phases[1]["t_start"]/self.stepSize)

        ts = np.array(self.resultFiles["CSV"]['states.csv']["time"][start_index:])
        ts_period = ts#/(1.0/self.frequency)
        muscleValuesDicts = self.resultFiles["CSV"]['states.csv']["muscleStates"][start_index:]


        #create muscle Value arrays (reformat csv output)
        muscleValues = dict.fromkeys(self.muscleNames) # dict with np.array for each muscle -> shape: (timestamps, parameters)
        parameters = ["lengthening_speed"] #which parameters shall be extracted into the muscleValues dict

        #initialize arrays
        for muscleName in muscleValues.keys():
            muscleValues[muscleName] = np.zeros((len(muscleValuesDicts),len(parameters)))

        #fill arrays
        for n, dicts in enumerate(muscleValuesDicts):
            for muscleName in muscleValues.keys():
                for p, parameter in enumerate(parameters):
                    muscleValues[muscleName][n][p] = dicts[muscleName][parameter]

        #convert units
        for p,parameter in enumerate(parameters):
            if(parameter == "lengthening_speed"):
                for muscleName in muscleValues.keys():
                    muscleValues[muscleName][:,p] *= 1000

        #Now extract wave specification using FFT
        waveSpecifications = {}

        #create plot in paralell and save to workFolder
        fig, ax = plt.subplots(nrows=len(muscleValues), ncols=3,
                #sharex='col',
                figsize=(3*3,len(muscleValues)*2),
                squeeze=False)
        plotTimepoints = int((1.0/self.frequency)/self.stepSize)*4 #just visualize 4 periods

        self.FFT ={
            "frequency":[],
            "amplitude":[],
            "phase":[],
            "complex":[]
        }

        specificationsText=f"scenario={self.scenarioName}\nfrequency={self.frequency}\nAmod={Amod}\noffMod={offMod}\n\n"

        for x, muscleName in enumerate(muscleValues.keys()):
            
            print(f"Applying FFT for {muscleName}")
            specificationsText += f"{muscleName}:\n"

            values = muscleValues[muscleName][:,0] #we only use one parameter, namely lengthening_speed

            #apply FFT
            windowSize = 20
            #values_filtered = uniform_filter1d(values, size=windowSize) #this is nice but does not really add much value.. 
            values_filtered = values

            frequencies, As, ps, Z = self.getFFT(ts,values_filtered,freqRange=freqRange, freq_amp_thr=0.01) #just get signals, the filtering is done by the peak finder
            self.FFT["frequency"].append(frequencies)
            self.FFT["amplitude"].append(As)
            self.FFT["phase"].append(ps)
            self.FFT["complex"].append(Z)

            #get frequency components (strongest signals in FFT -> our signal components)
            peak_indecs = self.find_peaks(As)
            print("PEAK INDICES:"+str(peak_indecs))
            components = []
            for i in peak_indecs:
                f = frequencies[i] #frequency
                A = As[i]  # amplitude of frequency
                p = ps[i]  # phase of wave - this phase information is actually pretty inaccurate. idk why

                components.append( (f,A,p) )

            #get p-offset (because the above p values are rather inaccurate)
            p_min = None
            mse_min = float("inf")
            original_values = np.zeros_like(ts) # reconstruction of original signal
            for p_off in np.linspace(0,2*np.pi,1000):
                v_temp = np.zeros_like(ts)
                for component in components:
                    f, A, p = component
                    v_temp += A*np.sin(f*2*np.pi*ts+p+p_off)

                MSE = self.mse(values,v_temp) #compute MSE

                if(MSE < mse_min): #update p_off
                    p_min = p_off
                    mse_min = MSE
                    original_values = v_temp
            p_off = p_min

            #First Column (signal)
            ax[x][0].plot(ts_period[:plotTimepoints], values[:plotTimepoints], color="black",alpha=0.25)#, label=muscleName)
            ax[x][0].plot(ts_period[:plotTimepoints], values_filtered[:plotTimepoints], color="pink",alpha=1)
            ax[x][0].plot(ts_period[:plotTimepoints], original_values[:plotTimepoints],color="black")
            ax[x][0].set_ylabel(muscleName)

            ax[-1][0].set_xlabel(r"Time (s)")
            ax[0][0].set_title("Signal: "+parameter)

            #Second Column (componets)
            values_inf = np.zeros_like(ts) #create filtered signal 
            for component in components:
                f, A, p = component
                v = Amod*A*np.sin(f*2*np.pi*np.array(ts)+p+p_off)
                values_inf += v 

                print("\t Sinus (f, A, p) = "+str(f)+", "+str(A)+", "+str(p/np.pi)+" π")
                specificationsText += f"{A}*sinus(2π*{f}*t + {p/np.pi} π + {p_off})\n"
        
                ax[x][1].plot(ts_period[:plotTimepoints], v[:plotTimepoints])#, label="("+str(f)+", "+str(A)+", "+str(p/np.pi)+" π)")
                #ax[x][1].set_ylabel("Amplitude")
            ax[-1][1].set_xlabel(r"Time (s)")
            ax[0][1].set_title(r"Inferred Parts of Signal")

            #get y-offset
            y_off = -min(values_inf) # move to the x axis (no negative values)
            y_off *= offMod
            values_inf += y_off

            #Put everything into the output dict
            waveSpecifications[muscleName] = {
                        "components":components,
                        "Amod":Amod,
                        "y_offset":y_off,
                        "p_offset":p_off,
                        "MSE":MSE
                        }

            print("\t-> Amod: "+str(Amod))
            print("\t-> y_offset: "+str(y_off))
            print("\t-> p_offset: "+str(p_off))
            print("\t-> MSE: "+str(MSE))
            specificationsText += f"MSE={MSE}\ny_offset={y_off}\p_offset={p_off}\n\n"

            #Third Column (signal)
            #ax[x][2].set_ylabel("Muscle Input")
            ax[-1][2].set_xlabel(r"Time (s)")
            ax[0][2].set_title("Inferred muscle input")
            ax[x][2].plot(ts_period[:plotTimepoints], values_inf[:plotTimepoints])

        #Save figute
        plt.tight_layout()
        plt.savefig(self.workingFolder+"FFT_splitup.png", dpi=200)
        plt.show()

        #Print text
        with open(self.workingFolder+'waveSpecifications.txt', 'w') as f:
            f.write(specificationsText)

        ####  Plot Frequency Spectrum and other stuff ####

        nrows = len(self.muscleNames)
        ncols = len(list(self.FFT.keys()))-1
        fig_height = nrows * 3
        fig_width = ncols * 4
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                        #sharex='col',
                        figsize=(fig_width,fig_height),
                        squeeze=False)

        for col, key in enumerate(list(self.FFT.keys())[1:]):
            
            for row, muscleName in enumerate(self.muscleNames):
            
                if(key == "amplitude"):
                    #thresholds
                    #ax[row][col].vlines(freqRange,ymin=freq_amp_thr,ymax=max(self.FFT[key][row]), color="red") #redundant becasse no other 
                    #ax[row][col].hlines(freq_amp_thr, xmin=min(self.FFT["frequency"][row]), xmax=max(self.FFT["frequency"][row]), color="red")
                    
                    #peaks
                    components = waveSpecifications[muscleName]["components"]
                    ax[row][col].scatter([f for f,A,p in components], [A for f,A,p in components], marker="x",color="red",s=50 )
                if(key=="complex"):
                    
                    ax[row][col].plot(self.FFT["frequency"][row],self.FFT[key][row].imag,label="imag")
                    ax[row][col].plot(self.FFT["frequency"][row],self.FFT[key][row].real,label="real")
                else:    
                    ax[row][col].plot(self.FFT["frequency"][row],self.FFT[key][row])

                ax[row][0].set_ylabel(muscleName)
                
            ax[nrows-1][col].set_xlabel("frequency")
            ax[0][col].set_title(key)

            if(key == "complex"):
                ax[0][col].legend()
            
        plt.savefig(self.workingFolder+"FFT_spectrum.png", dpi=200)
        plt.show()


        #### DONE ####

        return waveSpecifications

    #see self.resultFiles for all the data extracted from the CSVs
    def postprocessResults(self):
        # THIS function only extract based on FFT.
        # The data driven appraoch is not implemented here as it is not used in the masters thesis!
        # for the data driven appraoch (which is just: taking the redout values while PID movement and clean them up a bit)
        # contact the thesis author / github maintainer
        
        super().postprocessResults()  # do the higer abstraction postprocessing

        #different hand-selected parameters for FFT extraction for different scenarios
        if(self.scenarioName == "longitudinal"): #long
            freq_amp_thr=0.09
            Amod = 0.6
            offMod = 1
        elif(self.scenarioName == "transversal"):#trans
            freq_amp_thr=0.09
            Amod = 1.4
            offMod = 1
        elif(self.scenarioName == "circle"):#circle
            freq_amp_thr=0.04
            Amod = 0.4
            offMod = 1
        elif(self.scenarioName == "discont"):#discont
            freq_amp_thr=0.05
            Amod = 0.4
            offMod = 1
        elif(self.scenarioName == "circe_vib_ang"):#circle vib (angular)
            freq_amp_thr=0.04
            Amod = 0.4
            offMod = 1
        elif(self.scenarioName == "circe_vib_abs"):#circle vib (absolute)
            freq_amp_thr=0.04
            Amod = 0.4
            offMod = 1
        else:
            raise RuntimeError(f"Unknown scenario type {self.scenarioName}")
 
        freqRange = (self.frequency/4, self.frequency*4)
        # Frequency Upper limit (überschwindung): up to the 4th harmonic. Everything faster is probably noise anyways
        # Frequency lower limit (unterschwindung): 1/4 of base frequency 

        #self.waveSpecification = self.getWaveSpecification(Amod=Amod,offMod=offMod,freqRange=freqRange,freq_amp_thr=freq_amp_thr)
        self.waveSpecification = self.getWaveSpecification(Amod=1,offMod=1,freqRange=freqRange,freq_amp_thr=freq_amp_thr)

        #save waveSpecification

        filename = f'{self.scenarioName}_f={self.frequency}_periods={self.pid_movement_duration}.pkl'
        filePath = self.muscleGTdir+self.muscleValuesGTsource+"/"+filename

        with open(filePath, 'wb') as f:
            print("Saved to "+filePath)
            pickle.dump(self.waveSpecification, f)
    


    ##### VISUALIZATIONS #####

