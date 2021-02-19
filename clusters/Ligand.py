import pandas as pd
import numpy as np

class Ligand:
    """
        simple ligand class to store the contents of ligand_information.csv.
        I made the style choice of having this take an array of inputs from the text file for easier useability.
        And to save efficiency so I dont need to store the memory of >8000 objects when it can condensed to one object

        Args:
          LigandID (array-like): Holds the ID label of each node
          score (array-like): Hold the doc score of each Ligand
          Smiles(array-like): smiles representation of the Ligand ID
          OnBits(array-like): raw onbit data where each number represents the index where a bit is 1
          Long(array-like): Will hold the 1024 dimension represention of the ligant onbit


        """
    def __init__(self,LigandID,score,SMILES,OnBits):
        self.LigandID=LigandID
        self.score=score
        self.Smiles=SMILES
        self.OnBits=OnBits
        self.Long=np.array(len(LigandID))
        self.OnbitToLong()

    def OnbitToLong(self):
        """
        Converts the raw onbit representation to a long 1024 array bit version of the data for clustering.
        """
        Long=[]
        for i in self.OnBits:
            tmp=[0]*1024
            for idx in i.split(","):
                idx=int(idx)
                tmp[idx-1]=1
            Long.append(tmp)
        Long=np.array(Long)


        self.long=Long
