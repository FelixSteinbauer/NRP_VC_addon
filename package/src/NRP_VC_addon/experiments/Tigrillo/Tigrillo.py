from ..Experiment import Experiment

import numpy as np

class Tigrillo_Experiment(Experiment):
    """Default "Tigrillo SNN Learning in Closed Loop" Experiment

    duration -- min, max (max might not be enforcable)"""

    def __init__(self,
        exp_id,
        dataDirectory,
        workingFolderDirectory,
        templateFolder,
        duration=(60,None),
        stepSize=0.020,
        experimentName="Tigrillo",
        folderPrefix="", folderInfix="", folderSuffix="",
        columns=3*3*2, Nexec=30, Ninh=10):

        super().__init__(
            exp_id=exp_id,
            dataDirectory=dataDirectory,
            workingFolderDirectory=workingFolderDirectory,
            stepSize=stepSize,
            duration=duration,
            experimentName=experimentName,
            folderPrefix=folderPrefix, folderInfix=folderInfix, folderSuffix=folderSuffix)

        # Set timeout (other pysics parameter etc. are set in the parent class allready)
        self.fileContents["experiment_configuration.exc"] = {"content":  self.getExecConfiguration(
            templatePath=templateFolder + "experiment_configuration.exc", timeout=self.duration[0])}

        # set brain neuron configuration
        self.fileContents["CPG_brain.py"] = {"content": self.getBrainConfiguration(
            templatePath=templateFolder+"CPG_brain.py",
            columns=columns, Nexec=Nexec, Ninh=Ninh)}

    ##### EXPERIMENT FILES #####

    def getExecConfiguration(self, templatePath, timeout):

        replacements = [
            ("[MARKER_TIMEOUT]", str(int(np.ceil(timeout))))  # set timeout
        ]

        fileContent = self.replaceInFile(
            filePath=templatePath, replacements=replacements)

        return fileContent

    def getBrainConfiguration(self,
                              templatePath,
                              columns=3*3*2, Nexec=30, Ninh=10):

        replacements = [
            ("[MARKER_COLUMNS]",    str(int(columns))),
            ("[MARKER_Nexc]",       str(int(Nexec))),
            ("[MARKER_Ninh]",       str(int(Ninh)))
        ]

        fileContent = self.replaceInFile(
            filePath=templatePath, replacements=replacements)

        return fileContent

    ##### EXPERIMENT RESULTS #####

    def postprocessResults(self):
        pass  #we are not interested in experiment output for now
