import pandas as pd
import numpy as np

class Ligand:
    def __init__(self,LigandID,score,SMILES,OnBits):
        self.LigandID=LigandID
        self.score=score
        self.Smiles=SMILES
        self.OnBits=OnBits

    def OnbitToLong(self):
        Long=[]
        for i in self.OnBits:
            tmp=[0]*1024
            for idx in i.split(","):
                idx=int(idx)
                tmp[idx-1]=1
            Long.append(tmp)
        Long=np.array(Long)


        self.long=Long
