

from pynrp import virtual_coach
from pynrp.virtual_coach import VirtualCoach

import time  # for sleeps and timing
# for Paralell Simulatordoes not pickel sim objects
from multiprocessing import Pool
import natsort  # for sorting lists properly


class NRP_Simulator():
    """A abstract class representing one simulation instance on the NRP

    Arguments:
    - VC_username -- virtual coach username (might be OCID name)
    - VC_password -- virtual coach password (might be OCID password)
    - VC_address -- virtual coach full address string including http, IP adress and port (if necesarry)
    - sleeptime -- specifies minimal waiting time between operation that cannot be executed directly after each other (default 0.0)

    """

    def __init__(self,
                 VC_username, VC_password, OCID_username, OCID_password, VC_address,
                 sleeptime):

        # VC stuff
        self.VC_username = VC_username
        self.VC_password = VC_password
        self.OCID_username = OCID_username
        self.OCID_password = OCID_password
        self.VC_address = VC_address

        # experiment stuff
        self.experiment = None

        self.sleeptime = sleeptime

    # VC INITIALIZATION
    def getVC(self, forceGenerate=False):
        """Refreshes the VC instance if necessary and returns it."""

        vc_age = time.time()-self.vc_timestamp

        if (forceGenerate == True or vc_age >= self.vc_TTL):
            print(
                f"Regenerate VC instance (age: {vc_age:.2f}s = {vc_age/60:.2f}min )")

            self.vc_timestamp = time.time()
            if (self.VC_username != None):
                self.vc_instance = VirtualCoach(
                    environment=self.VC_address,
                    storage_username=self.VC_username,
                    storage_password=self.VC_password)
            elif (self.OCID_username != None):
                self.vc_instance = VirtualCoach(
                    environment=self.VC_address,
                    oidc_username=self.OCID_username,
                    oidc_password=self.OCID_password)
            else:
                raise RuntimeError(
                    "Neither Storage nor OCID username was provided.")
        else:
            print(
                f'Reusing VC instance (age: {vc_age:.2f}s = {vc_age/60:.2f}min )')

        return self.vc_instance

    ##### EXPERIMENT RUNNING PIPELINE (generel. Others are implemented in subclass)
    def setFiles(self):
        """Writes the simulation parameters for the experiment onto the server files.

        ATTENTION: These changes are permanent and completly override the existing files on the server!"""

        for filePath in self.experiment.fileContents.keys():
            content = self.experiment.fileContents[filePath]["content"]
            lines = content.count("\n")
            print(
                f'Replacing \"{filePath}\" with {lines} lines from the respective template.')

            #If you get an HTTP 403 here, check if the experiment can be found (e.g. correct name!)
            self.vc_instance.set_experiment_file(
                exp_id=self.experiment.exp_id, file_name=filePath, file_content=content)

    def launchExperiment(self):
        """Abstract method. Please override in subclass!"""
        raise NotImplementedError()

    def setParameters(self, tid=None):
        """Writes the simulation parameters that go into the transfer functions and other stuff relevant before simulation launch.

        These changes are only present for the duration of the simulation"""

        if tid != None:
            exp = self.experiments[tid]
        else:
            exp = self.experiment

        for TFname in exp.TFcontents.keys():
            replacements = exp.TFcontents[TFname]["replacements"]
            code = self.sim.get_transfer_function(TFname)  # get TF
            lines_original = code.count("\n")
            newCode = exp.replaceInTF(
                code, replacements)  # modify TF
            exp.TFcontents[TFname]["content"] = newCode
            lines = newCode.count("\n")
            self.sim.edit_transfer_function(TFname, newCode)  # replace TF

            print(f'TF \"{TFname}\" went from {lines_original}\t to {lines}\tlines code.')

    def executeExperiment(self):
        """Abstract method. Please override in subclass!"""
        raise NotImplementedError()

        # function to rune one Simulation with a specific valueList

    def readResults(self):
        # read csv files
        for filename in self.experiment.resultFiles["CSV"].keys():
            print(f'[CSV] Try reading {filename}')
            self.experiment.resultFiles["CSV"][filename] = self.vc_instance.get_last_run_file(
                exp_id=self.experiment.exp_id, file_type="csv", file_name=filename)

        # read profiler (whatever that is)
        for filename in self.experiment.resultFiles["profiler"].keys():
            print(f'[profiler] Try reading {filename}')
            # I acutally have never used the profiler so I have no clue what to implement here
            raise NotImplementedError(
                "I dont know how to read profiler. You need to implement that yourself!")
            # TODO: implement this even if I dont need it.

        # read files from directory directly
        for filePath in self.experiment.resultFiles["filesystem"].keys():
            print(f'[filesystem] Try reading {filePath}')
            with open(filePath, "r") as f:
                self.experiment.resultFiles["filesystem"][filePath] = f.read()

    def run(self):
        if (self.experiment == None):
            raise RuntimeError(
                "No experiment set. Use 'setExperiment' before running the simulation.")

        print(f'STEP 0: Setting experiment files...')
        t_start = time.time()
        self.setFiles()
        step_time = time.time()-t_start
        print(f"Setting experiment files took {step_time:.02f}s")

        time.sleep(self.sleeptime)

        print(f'STEP 1: Launch instance...')
        t_start = time.time()
        self.launchExperiment()
        step_time = time.time()-t_start
        print(f"Launch took {step_time:.02f}s")

        time.sleep(self.sleeptime)

        print(f'STEP 2: Setting experiment parameters (Transfer functions)...')
        t_start = time.time()
        self.setParameters()
        step_time = time.time()-t_start
        print(f"Setting parameters took {step_time:.02f}s")

        time.sleep(self.sleeptime)

        simulationTime = self.experiment.duration[0]
        print(
            f'STEP 3: Executing simulation (min simulation time = {simulationTime:.02f}s = {simulationTime/60:.02f} min)...')
        t_start = time.time()
        self.executeExperiment()
        step_time = time.time()-t_start
        realTime = step_time
        RTF = realTime/simulationTime if simulationTime > 0 else 0
        print(f"Executing simulation took {step_time:.02f}s = {step_time/60:.02f}min = {step_time/3600:.02f}h  (rtf={RTF:.2f})")

        time.sleep(self.sleeptime)

        #return #TODO: remove this debugging statement

        print(f'STEP 4: Reading results...')
        t_start = time.time()
        self.vc_instance = self.getVC()  # refresh VC if necesarry
        self.readResults()
        step_time = time.time()-t_start
        print(f"Reading results took {step_time:.02f}s")

        time.sleep(self.sleeptime)

        print(f'STEP 5: Post-processing results...')
        t_start = time.time()
        self.experiment.postprocessResults()
        step_time = time.time()-t_start
        print(f"Post-processing results took {step_time:.02f}s")

    ##### GETTER & SETTER
    def setExperiment(self, experiment):
        """Initializes this simulator with a experiment. Everything from possible previos experiments are overwritten!"""

        self.experiment = experiment
        self.sim = None

    def getExperiment(self):
        return self.experiment


class NRP_Simulator_Local(NRP_Simulator):
    """Subclas representing a local NRP simulation (requires running docker)"""

    def __init__(self,
                 VC_username='nrpuser', VC_password="password", VC_address='http://172.19.0.1:9000', sleeptime=2):

        super().__init__(
            VC_username=VC_username, VC_password=VC_password, VC_address=VC_address,
            OCID_username=None, OCID_password=None, sleeptime=sleeptime)

        # init VC
        # basically the VC instance does not expire on a local macine (as tehre are not OCID tokens in the background)
        self.vc_TTL = float("inf")
        self.vc_timestamp = time.time()
        # forceGenerate needs to be true at this point as no refreshLimit is defined yet
        self.vc_instance = self.getVC(forceGenerate=True)

    ##### EXPERIMENT RUNNING PIPELINE  (local specific)

    def launchExperiment(self):
        """Launches Experiment on Server."""

        success = False
        while success == False:
            
            try:
                self.sim = self.vc_instance.launch_experiment(self.experiment.exp_id)
                success = True
            except ValueError as e:

                if("No available servers for" in str(e)):   #this error we can handle
                    print("[WARNING] Server currently unavailable. Waiting and trying again...")
                    time.sleep(self.sleeptime)
                else:
                    raise e

    def executeExperiment(self):
        """ """

        self.sim.start()
        time.sleep(self.sleeptime)

        # that how long the simulation AT LEAST (as RTF<1 is impossible on the current NRP version) takes
        time.sleep(self.experiment.duration[0])

        while (self.sim.get_state() == "started" or self.sim.get_state() == "created"):
            time.sleep(self.sleeptime)

        self.sim.stop()

        #simulationTime = 0
        # while simulationTime <= tLim: #sleep until the simulation is termination ready (needs to be <= because the are rounding to integers!!)
        #    time.sleep(0.5)
        #    if(simulationState==0):
        #        #print("No simulation state available")
        #        print("NoSimState",end = ' ')
        #    else:
        #        simulationTime = simulationState["simulationTime"]
        #        realTime = simulationState["realTime"]
        #RTF = realTime/simulationTime if simulationTime>0 else 0
        # print(f"(t_r={realTime},t_s={simulationTime},rtf={RTF:.2f})")

class NRP_Simulator_Online(NRP_Simulator):
    """Subclas representing a online NRP simulation (requires running docker).

    This class differs from the local version as it tracks the age of the VC instance and renews it if its to old.
    Also it has integrated error handling when a simulation fails (e.g. retrying).

    Arguments:
    - OCID_username -- virtual coach OCID username
    - OCID_password -- virtual coach OCID password
    - VC_address -- virtual coach full address string including http, IP adress and port (if necesarry)
    - VC_TTL -- maximal age (seconds) of VC instance after which new istance shall be tried to obtain. Wheter really an new instance is
    generate don the server-side noone can know... (default 14*16 = 14 minutes)
    - sleeptime -- specifies minimal waiting time between operation that cannot be executed directly after each other (default 2.0).
    Must be > 0 or buisy waiting (and server overload) might occur.
    - serverName -- spcifies server to run on. Non for any server (default None)
    - tid -- thread id. Relevant if run parallel threads. only the main thread (tid=0) will set experiment files! tid=None
        means that no other instance of this experiment is run in paralell (no multitrheading)

    """

    def __init__(self,
                 OCID_username,
                 OCID_password,
                 VC_address='http://148.187.149.212',
                 VC_TTL=14*60,
                 sleeptime=2,
                 maxTries=3,
                 serverName=None):
        super().__init__(
            OCID_username=OCID_username,
            OCID_password=OCID_password,
            VC_address=VC_address,
            VC_username=None,
            VC_password=None,
            sleeptime=sleeptime
        )

        # init VC
        self.vc_TTL = VC_TTL
        self.vc_timestamp = time.time()
        # forceGenerate needs to be true at this point as no refreshLimit is defined yet
        self.vc_instance = self.getVC(forceGenerate=True)

        # online NRP stuff
        self.serverName = serverName
        self.maxTries = maxTries  # on the same server before giving up

    ##### EXPERIMENT RUNNING PIPELINE (online specific)

    def launchExperiment(self):
        """Launches Experiment on Server."""

        self.getVC()  # refresh VC if necesarry

        tries = 0
        success = False
        while success == False:
            print(
                f' Try to launch {self.experiment.exp_id} on {self.serverName if self.serverName!=None else "ANY"} ...')
            try:
                self.sim = self.vc_instance.launch_experiment(
                    self.experiment.exp_id, server=self.serverName)
                success = True
                print(f'-> Successfull launched!')

            except Exception as e:

                print(f'Server launch error: {e}')
                if (tries < self.maxTries):
                    tries += 1
                    print(
                        f"\t -> Wait and try again (try {tries}/{self.maxTries}) ...")
                    time.sleep(self.sleeptime*5)
                else:
                    print(
                        f'Reached {self.maxTries} on server {self.serverName if self.serverName!=None else "ANY"}.')
                    raise e  # could not fix the server error. Simulation fails

    def executeExperiment(self):
        """ """

        vc = self.getVC()
        time.sleep(self.sleeptime)
        self.sim.start()

        # that how long the simulation AT LEAST (as RTF<1 is impossible on the current NRP version) takes
        time.sleep(self.experiment.duration[0])

        while (self.sim.get_state() == "started" or self.sim.get_state() == "created"):
            time.sleep(self.sleeptime)

        self.sim.stop()

class NRP_Simulator_Paralell(NRP_Simulator_Online):
    """Simulator taking multiple experiments and runs them in paralell

    ATTENTION: Every permanent file operation is just carried out for the first exeperiment as it is valid for all the other too.
    TF canges however are (as usual) simulation specific
    ATTENTION2: This class assumes that all experiments are from the same exp_id.
    If you want to run different experiments in paralell. Simply create more simulator instances (the paralellization however needs to be done manually)

    Arguments:
    """

    def __init__(self,
                 OCID_username,
                 OCID_password,
                 VC_address='http://148.187.149.212',
                 VC_TTL=14*60,
                 sleeptime=2,
                 maxTries=3):
        super().__init__(
            OCID_username,
            OCID_password,
            VC_address=VC_address,
            VC_TTL=VC_TTL,
            sleeptime=sleeptime,
            maxTries=maxTries,
            serverName=None
        )

    ##### EXPERIMENT RUNNING PIPELINE (paralell specific)

    # simular to single thread setFiles appiled on the first experiment in self.experiments

    def setFiles(self):
        """Writes the simulation parameters for the experiment onto the server files.

        ATTENTION: These changes are permanent and completly override the existing files on the server!"""

        for filePath in self.experiments[0].fileContents.keys():
            content = self.experiments[0].fileContents[filePath]["content"]
            lines = content.count("\n")
            print(
                f'Replacing \"{filePath}\" with {lines} lines from the respective template.')

            # HINT: If this call throws a HTTP 403 Forbidden, you might have used the wrong exp_id!
            self.vc_instance.set_experiment_file(
                exp_id=self.experiments[0].exp_id, file_name=filePath, file_content=content)

        # get available serverlist based on a blacklist

    def getAvailableServers(self, blacklist=[]):
        availableServers = natsort.natsorted(
            self.vc_instance.print_available_servers())

        # filter based on blacklist
        availableServers_filtered = []
        for server in availableServers:
            if (server in blacklist):
                continue
            availableServers_filtered.append(server)

        # enough servers availabe for the experiments?
        if (len(self.experiments) > len(availableServers_filtered)):
            raise RuntimeError(
                f'Not enough servers available! Requested {len(self.experiments)} but only {len(availableServers_filtered)}/{len(availableServers)} available .')

        self.availableServers = availableServers_filtered
        return availableServers_filtered

    def launchExperiment(self):
        """Launches Experiment on Server."""

        self.vc_instance = self.getVC()  # refresh VC if necesarry
        self.serverName = self.availableServers[self.tid]

        tries = 0
        success = False
        while success == False:
            print(f'[tid={self.tid}] Try to launch {self.experiments[self.tid].exp_id} on {self.serverName if self.serverName!=None else "ANY"} ...')
            try:
                self.sim = self.vc_instance.launch_experiment(
                    self.experiments[self.tid].exp_id, server=self.serverName)
                success = True
                print(f'[tid={self.tid}] -> Successfull launched!')

            except Exception as e:

                print(f'[tid={self.tid}] Server launch error: {e}')
                if (tries < self.maxTries):
                    tries += 1
                    print(
                        f"[tid={self.tid}] \t -> Wait and try again (try {tries}/{self.maxTries}) ...")
                    time.sleep(self.sleeptime*5)
                else:
                    print(
                        f'[tid={self.tid}] Reached {self.maxTries} tries on server {self.serverName if self.serverName!=None else "ANY"}.')
                    raise e  # could not fix the server error. Simulation fails

    def executeExperiment(self):
        """ """

        tries = 0
        success = False
        while success == False:
            print(f'[tid={self.tid}] Try to start experiment.')
            try:
                self.sim.start()  # If this throws a you might want to try again
                success = True
            except Exception as e:
                print(f'[tid={self.tid}] Experiment start error: {e}')

                if (str(e) == "Unable to set simulation state, HTTP status 400"):
                    # HTTP error 400 (bad request) we are not able to solve this here
                    raise e
                    # HINT: possible reason: you are using simulation "real" timout which has allready expired when starting
                elif (tries < self.maxTries):
                    tries += 1
                    print(
                        f"[tid={self.tid}] \t -> Wait and try again (try {tries}/{self.maxTries}) ...")
                    time.sleep(time.sleep(self.sleeptime))
                else:
                    print(
                        f'[tid={self.tid}] Reached {self.maxTries} tries on server {self.serverName if self.serverName!=None else "ANY"}.')
                    raise e  # could not fix the error. Simulation fails

        # time.sleep(30)
        # that how long the simulation AT LEAST (as RTF<1 is impossible on the current NRP version) takes
        time.sleep(self.experiments[self.tid].duration[0])

        while (self.sim.get_state() == "started" or self.sim.get_state() == "created"):
            time.sleep(self.sleeptime)

        self.sim.stop()

    # within this thread we should not write to the Simulator object as its is just a pickeld copy of the original.
    # We need to collect all outputs/results and retun them to the Pool-er
    def threadFunction(self, args):
        """Thread-paralell launch, parameter setting and execution of all experiments on the server.

        Using multiple Threads is required as the launch command is blocking (30-50s) on the NRP servers. All other tasks
        can be exicuted simply using a for loop. Also the simulation object cannot be pickeld (by pickle and dill as well),
        which is why all code requiring the simulation object also has to happen within this thread.
        """

        
        self.tid = args  # get thread id from trhead arguments

        # slightly deserialize (mainly for nice output)
        time.sleep(self.sleeptime*self.tid)

        #self.getVC(forceGenerate=True)
        self.getVC(forceGenerate=False)
        print(f'[tid={self.tid}] STEP 1: Launch instance...')
        t_start = time.time()
        self.launchExperiment()
        step_time = time.time()-t_start
        print(f"[tid={self.tid}] Launch took {step_time:.02f}s")

        time.sleep(self.sleeptime)

        print(
            f'[tid={self.tid}] STEP 2: Setting experiment parameters (Transfer functions)...')
        t_start = time.time()
        self.setParameters(self.tid)
        step_time = time.time()-t_start
        print(f"[tid={self.tid}] Setting parameters took {step_time:.02f}s")

        time.sleep(self.sleeptime)

        simulationTime = self.experiments[self.tid].duration[0]
        print(
            f'[tid={self.tid}] STEP 3: Executing simulation (min simulation time = {simulationTime:.02f} s = {simulationTime/60:.02f} min)...')
        t_start = time.time()
        self.executeExperiment()
        step_time = time.time()-t_start
        realTime = step_time
        RTF = realTime/simulationTime if simulationTime > 0 else 0
        print(
            f"[tid={self.tid}] Executing simulation took {step_time:.02f}s = {step_time/60:.02f}min = {step_time/3600:.02f}h  (rtf={RTF:.2f})")

    def readResults(self):
        """

        ATTENTION: the first experiment defined which files are relevant for all following experiments.
        """

        # read csv files
        # first experiment defined which files are relevant for all following experiments
        print(f'[CSV] Getting files...')
        try:
            csv_files = self.vc_instance._VirtualCoach__get_available_files(
                experiment_id=self.experiments[0].exp_id, file_type="csv")
        except Exception as e:
            
            raise e
            # HINT: If an error occurs at this point it is possibly due a very full CSV folder
            # you might want to delete some CSV files

        folderNames = natsort.natsorted(csv_files.keys())

        # get all folders that contain all the relevant files
        folderNames_filtered = []
        fileNames_wanted = set(self.experiments[0].resultFiles["CSV"].keys())
        for folderName in folderNames:
            fileNames_real = set(csv_files[folderName].keys())
            fileNames_wanted = set(
                self.experiments[0].resultFiles["CSV"].keys())
            if fileNames_wanted.intersection(fileNames_real) == fileNames_wanted:
                folderNames_filtered.append(folderName)

        # check if enough files exist
        if len(folderNames_filtered) < len(self.experiments):
            raise RuntimeError(
                f'Not enough CSV folders found ({len(folderNames_filtered)}) containing the following files: {fileNames_wanted}')

        # read files from last N folders
        for eid in range(len(self.experiments)):
            print(f'Acessing folder {folderName} for experiment {self.experiments[eid].experimentName}(nr {eid})... ')

            for filename in self.experiments[0].resultFiles["CSV"].keys(): 
                folderName = folderNames[-(len(self.experiments)-eid)] # last eleemnt in the list is for the last experiment (Nth). Nth last is for the first experiment    

                uuid = csv_files[folderName][filename]["uuid"]
                # last eleemnt in the list is for the last experiment (Nth). Nth last is for the first experiment

                self.experiments[eid].resultFiles["CSV"][filename] = self.vc_instance._VirtualCoach__get_file_content(
                    exp_id=self.experiments[eid].exp_id, file_uuid=uuid)           

                #except KeyError as e:
                #    print(f'CSV folder \"{folderName}\" is not ready yet. Waiting and retry.')
                #    time.sleep(self.sleeptime)
                #    folderNames = self.getCSVfolderNames()

                
        # read profiler (whatever that is)
        for filename in self.experiments[0].resultFiles["profiler"].keys():
            print(f'[profiler] Try reading {filename}')
            # I acutally have never used the profiler so I have no clue what to implement here
            raise NotImplementedError(
                "I dont know how to read profiler. You need to implement that yourself!")

        # read files from directory directly
        for filePath in self.experiments[0].resultFiles["filesystem"].keys():
            print(f'[filesystem] Try reading {filePath}')
            # I acutally have never used the profiler so I have no clue what to implement here
            raise NotImplementedError(
                "I never used custome file IO for paralell launches. You need to implement that yourself!")

            # with open(filePath, "r") as f:
            #    self.experiment.resultFiles["filesystem"][filePath] = f.read()

    def run(self):
        if (self.experiments == None):
            raise RuntimeError(
                "No experiment set. Use 'setExperiment' before running the simulation.")

        print(f'STEP 0: Setting experiment files...')
        t_start = time.time()
        self.setFiles()
        step_time = time.time()-t_start
        print(f"Setting experiment files took {step_time:.02f}s")

        time.sleep(self.sleeptime)

        # BEGINN MULTIPROCESSING
        # create a list of available servers
        self.availableServers = self.getAvailableServers()

        args = list(range(len(self.experiments)))  # create list of thread ids

        with Pool(len(args)) as p:
            # all changes that are made to the simultor object within this function
            rets = p.map(self.threadFunction, args)
            # are only temorary (copies) and do not have any effect on this instance (the real one) of the simulator
        # END MULTIPROCESSING

        time.sleep(self.sleeptime)

        print(f'STEP 4: Reading results...')
        t_start = time.time()
        self.vc_instance = self.getVC()  # refresh VC if necesarry
        self.readResults()
        step_time = time.time()-t_start
        print(f"Reading results took {step_time:.02f}s")

        time.sleep(self.sleeptime)

        print(f'STEP 5: Post-processing results...')
        t_start = time.time()
        for experiment in self.experiments:
            experiment.postprocessResults()
        step_time = time.time()-t_start
        print(f"Post-processing results took {step_time:.02f}s")

    ##### GETTER & SETTER

    def setExperiments(self, experiments):
        """Initializes this simulator with a list of experiments. The first element will be the experiment for which permanent files are set. """

        self.experiments = experiments

    def getExperiments(self):
        return self.experiments
