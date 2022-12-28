
import datetime
from pathlib import Path #for creation of directories. requires python >= 3.5. Otherwise you need to use os.mkdir
import os #to read and write files
import numpy as np #to evaluate CSV entries of np arrays
import matplotlib.pyplot as plt

class Experiment:
    """A abstract class representing a NRP experiment configuration (not the results)"""

    # function that reads the experiment specific (if any) CSV files
    @classmethod
    def readCSV(self, rawFileContent):
        """reads selfmade csv file which consits of python objects converted to string via "str()"""

        # read data from the CSV (because of the (wrong) formatting we cannot use a default CSV reader)
        binary_csv_string = rawFileContent
        s = binary_csv_string.decode('ascii')
        s = s.replace("\n          ", " ")  # some wierd newline formatting
        s = s.replace("\n ", ", ")
        s = s.replace("nan", "np.nan")
        s = s.split("\n")

        # assmeble data
        data = {}

        # get headers
        headers = s[0].split(",")
        for i, header in enumerate(headers):
            data[header] = []

        for i_l, line in enumerate(s[1:-1]):
            values = [value for value in line.split("\"") if value != ',']
            for i_h, header in enumerate(headers):
                v = values[i_h]
                # type casting
                if (header == "time"):
                    v = eval(v[:-1])
                elif (header == "Simulation_reset"):
                    pass  # let it be a string
                # convert to original data type (default behavoir for any header)
                else:
                    # print(repr(v))
                    v = eval(v)

                data[header].append(v)

        return data

    @classmethod
    def replaceInFile(self, filePath, replacements):
        """ Returns the file content with the markers replaced 
        
        replacements -- list of tupels where the first string is the marker in the file which is to be replaced
        by the second string. (all occurences are replaced)
        workingFolderDirectory -- the folder where output and experiment files will be stored.
            Set to None to avoid filesystem usage (might be usefull if the output files do not matter).
        """

        with open(filePath, "r") as f:
            fileContent = f.read()

        for replacement in replacements:
            fileContent = fileContent.replace(replacement[0], replacement[1])

        return fileContent

    @classmethod
    def replaceInTF(self, code, replacements):
        """ Returns the changed TF
        
        replacements -- list of tupels where the first string is the marker in the text which is to be replaced
        by the second string. (all occurences are replaced)
        """

        for replacement in replacements:
            code = code.replace(replacement[0], replacement[1])

        return code


    def __init__(
            self,
            exp_id,
            dataDirectory,
            workingFolderDirectory,
            stepSize,
            duration,
            experimentName="Experiment",
            folderPrefix="", folderInfix="", folderSuffix=""):

        self.exp_id = exp_id
        self.dataDirectory = dataDirectory
        self.workingFolderDirectory = workingFolderDirectory
        self.stepSize = stepSize
        self.duration = duration
        self.experimentName = experimentName
        self.folderPrefix, self.folderInfix, self.folderSuffix = folderPrefix, folderInfix, folderSuffix

        # Setup folder structure
        if(self.workingFolderDirectory != None):
            self.workingFolder = self.__createWorkingFolder(
                parentFolder=self.workingFolderDirectory,
                experimentName=experimentName,
                prefix=folderPrefix, infix=folderInfix, suffix=folderSuffix)

            print("Working in folder: "+self.workingFolder)

        #File contents that shall be replaced by the simulator for the experiment (permanent file changes)
        self.fileContents = {} #dict: Keys: experiment file path. Values: {"content":  file Content as string} 

        #Transfer function contents that shall be replaced by the simulator for the duration of the simulation
        self.TFcontents = {} #dict: Keys: TF name. Values: {"replacements": replacement list, "content": final TF string} 

        #Result files (key = filename, value = content(initially empty))
        self.resultFiles = {
            "CSV" : {}, #files that are written to the CSV folder
            "profiler":{}, # files that written to the recording folder
            "filesystem":{} #files directly from the filesystem (expects relative path in the experiment folder)
        }

    # creates folder with timestamp as suffix
    def __createWorkingFolder(self, parentFolder, experimentName, prefix, infix, suffix):
        now = datetime.datetime.now()
        folderPath = parentFolder + prefix + experimentName + \
            infix + now.strftime("%Y-%m-%d_%H:%M:%S") + suffix + "/"
        Path(folderPath).mkdir(parents=True, exist_ok=True)
        return folderPath


    def postprocessResults(self):
        """This function is called after the simulat has read all the raw file content.
        Here, an experiment can do proper interpretation of the files content"""

        raise NotImplementedError("You need to override this function in your custom experiment, even if you dont have experiment output.")
