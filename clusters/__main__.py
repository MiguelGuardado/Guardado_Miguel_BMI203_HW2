from clusters import Ligand,algs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    LigandInformation=pd.read_csv("../ligand_information.csv",sep=",")
    LigandData=Ligand.Ligand(LigandID=LigandInformation['LigandID'],score=LigandInformation['Score'],
                             SMILES=LigandInformation['SMILES'],OnBits=LigandInformation['OnBits'])
    LigandData.OnbitToLong()


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
    # #
    # PT1=algs.PartitionClustering(rawdata=Z,n_clusters=3,max_iteration=10)

    # PT1=algs.PartitionClustering(rawdata=LigandData.long[0:1000],n_clusters=150,max_iteration=100)
    # PT1.DistanceMatrix=algs.CalculateEuclideanDistance(PT1)
    # PT1.runClustering()
    # algs.SilouetteScore(PT1)

    HC1=algs.HierarchicalClustering(LigandData.long[0:100])
    HC1.CalculateEuclideanDistance()
    HC1.runClustering()



    #
    # #Plot output files
    #
    # LABEL_COLOR_MAP = {1: 'r',
    #                    2: 'b',
    #                    3: 'g',
    #                    4: 'y',
    #                    5: 'b'
    #
    # }
    # label_color = [LABEL_COLOR_MAP[l] for l in PT1.clusterassignment]
    # plt.scatter(Z[:,0], Z[:,1], c=label_color)
    # plt.show()










if __name__ == "__main__":
    main()