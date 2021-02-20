from clusters import Ligand,algs
import pandas as pd
import numpy as np
import umap # comment if you are not using this! it takes forever to load locally
import matplotlib.pyplot as plt



def main():
    LigandInformation=pd.read_csv("../ligand_information.csv",sep=",")
    LigandData=Ligand.Ligand(LigandID=LigandInformation['LigandID'],score=LigandInformation['Score'],
                             SMILES=LigandInformation['SMILES'],OnBits=LigandInformation['OnBits'])
    LigandData.OnbitToLong()

    #Question 2
    #I am going to implement a Umap of the ligand
    fit = umap.UMAP()
    u= fit.fit_transform(LigandData.long[0:2000])

    #Used to just load the data for later questions/visualizations, if you uncomment, keep the loadtxt command to so the data
    #can be typecast to a numpy array, for array index notation consistentcy .
    np.savetxt('UmapDimensionalSpace.txt',[u[:, 0], u[:, 1]])
    u=np.loadtxt("UmapDimensionalSpace.txt")
    #
    plt.scatter(u[0], u[1])
    plt.title('UMAP embedding of Ligands')
    plt.show()

    LABEL_COLOR_MAP = {1: 'r',
                       2: 'b',
                       3: 'g',
                       4: 'y',
                       5: 'm',
                       6: 'c'

    }
    #Question 3 +4
    score=[]
    for i in range(1,10):
        print(i)
        PT=algs.PartitionClustering(LigandData.long[0:2000],n_clusters=i, max_iteration=100)
        PT.runClustering()
        score.append(algs.SilhouetteScore(PT))
        del PT
    print(score)

    #Will rerun singluar test on higest silscore to get data for generation.
    #Best score was found when K=6, see Guardado_Miguel_BMI203_HW2_WriteUp.pdf for more info.
    PT_k6=algs.PartitionClustering(LigandData.long[0:2000],n_clusters=6, max_iteration=100)
    PT_k6.runClustering()
    print(PT_k6.clusterassignment)
    label_color = [LABEL_COLOR_MAP[l] for l in PT_k6.clusterassignment]
    print(np.unique(PT_k6.clusterassignment))
    u=np.loadtxt("UmapDimensionalSpace.txt")

    plt.figure(figsize=(20, 10))
    plt.scatter(u[0], u[1],c=label_color)
    plt.title('UMAP embedding of Ligands,2000 Ligands, 6 clusters')
    plt.show()


    #Question 5+6
    score=[]
    for i in range(1,10):
        print(i)
        HC=algs.PartitionClustering(LigandData.long[0:2000],n_clusters=i, max_iteration=100)
        HC.runClustering()
        score.append(algs.SilhouetteScore(HC))
        print(score)
        del HC
    print(score)
    HC_k4=algs.HierarchicalClustering(LigandData.long[0:2000],n_clusters=4)
    HC_k4.runClustering()
    print(HC_k4.clusterassignment)
    label_color = [LABEL_COLOR_MAP[l] for l in HC_k4.clusterassignment]
    print(np.unique(HC_k4.clusterassignment))
    u=np.loadtxt("UmapDimensionalSpace.txt")

    plt.figure(figsize=(20, 10))
    plt.scatter(u[0], u[1],c=label_color)
    plt.title('UMAP embedding of Ligands,2000 Ligands, 4 clusters')
    plt.show()

    # #Question 7
    arr1=np.array([0.2035,0.0933,0.18810,0.0485,0.396001,0.22705,0.08660,0.29346])
    arr2=np.array([0.0953,0.08660,0.26153,0.081803,0.163898,0.233848,0.09873,-0.167866])
    print(np.sum(arr1-arr2))
    print(algs.CalculatePairWiseDistance(arr1,arr2))

    k=[4,6]
    for n_cluster in k:
        PT= algs.PartitionClustering(LigandData.long[0:2000],n_clusters=n_cluster,max_iteration=100)
        PT.runClustering()
        HC= algs.HierarchicalClustering(LigandData.long[0:2000],n_clusters=n_cluster)
        HC.runClustering()
        print(algs.TanimotoCoeff(PT.clusters,HC.clusters))






if __name__ == "__main__":
    main()