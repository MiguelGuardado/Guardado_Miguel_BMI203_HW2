from clusters import Ligand,algs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    LigandInformation=pd.read_csv("../ligand_information.csv",sep=",")
    LigandData=Ligand.Ligand(LigandID=LigandInformation['LigandID'],score=LigandInformation['Score'],
                             SMILES=LigandInformation['SMILES'],OnBits=LigandInformation['OnBits'])


    LigandData.OnbitToLong()
    # print(LigandData.long[1:10])

    # HC1=algs.HierarchicalClustering(LigandData.long[0:10])
    # HC1.CalculateEuclideanDistance()
    # HC1.runClustering()
    # print(HC1.DistanceMatrix)

    # rawdata=np.array([[10],[7],[28],[20],[35],[8]])
    # DistMat=np.array([[0.0,0.23,0.22,0.37,0.34,0.23],
    #                   [0.23,0.0,0.15,0.20,0.14,0.25],
    #                   [0.22,0.15,0.0,0.15,0.28,0.11],
    #                   [0.37,0.20,0.15,0.0,0.29,0.22],
    #                   [0.34,0.14,0.28,0.29,0.0,0.39],
    #                   [0.23,0.25,0.11,0.22,.39,0.0]])
    #
    # HC2=algs.HierarchicalClustering(rawdata=rawdata)
    # #HC2.CalculateEuclideanDistance()
    # HC2.DistanceMatrix=DistMat
    # #HC2.CalculateEuclideanDistance()
    # HC2.runClustering()
    # algs.SilouetteScore()


    # X = np.random.rand(50, 2)
    # Y = 2 + np.random.rand(50, 2)
    # Z = np.concatenate((X, Y))
    #
    # PT1=algs.PartitionClustering(rawdata=Z,n_clusters=3,max_iteration=102)
    # PT1.DistanceMatrix=algs.CalculateEuclideanDistance(PT1)
    # PT1.runClustering()
    #
    # #Plot output files
    #
    # LABEL_COLOR_MAP = {1: 'r',
    #                    2: 'b',
    #                    3: 'g'
    #
    # }
    # label_color = [LABEL_COLOR_MAP[l] for l in PT1.clusterassignment]
    # print(len(Z[:,0]))
    # print(len(Z[:,1]))
    # plt.scatter(Z[:,0], Z[:,1], c=label_color)
    # plt.show()

    # SetA = [['A', 'B', 'D'], ['C', 'E'], ['F']]
    # SetA = set(frozenset(i) for i in SetA)
    #
    # SetB=[['A','B','E'],['C','D'],['F'] ]
    # SetB = set(frozenset(i) for i in SetB)

    # print(SetA)
    # print(SetB)
    SetA=set((0,1,2,5,6))
    SetB=set((0,2,3,4,5,7,9))

    f00=SetA.union(SetB)
    f11=SetA.intersection(SetB)
    f01=SetA.difference(SetB)
    f10=SetB.difference(SetA)

    print(len(f11)/(len(f01)+len(f10)+len(f11)))








if __name__ == "__main__":
    main()