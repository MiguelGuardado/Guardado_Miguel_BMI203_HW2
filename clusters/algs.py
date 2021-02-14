import numpy as np


#This will calculate the Silouette Score
def SilouetteScore(clusters,DistanceMatrix):
    if len(clusters)==1:
        return (-1)

    for curr_cluster in clusters:
        #Calculate the WSS for each individual centroid in a cluster
        for indiv1 in curr_cluster:
            for indiv2 in curr_cluster:

                if indiv1!=indiv2:
                    print(indiv1,indiv2)
                    print(DistanceMatrix[indiv1][indiv2])




        exit(0)

def TanimotoCoeff(ClusterA,ClusterB):

    #I will convert the list of clusters into a set
    SetA = set(frozenset(i) for i in ClusterA)
    SetB = set(frozenset(i) for i in ClusterB)

    #Same Clusters in both sets
    f00 = SetA.union(SetB)
    #Same in Cluster 1, not in 2
    f11 = SetA.intersection(SetB)
    #Same Cluster 2, not in 1
    f01 = SetA.difference(SetB)
    #Different clusters in both
    f10 = SetB.difference(SetA)

    return(len(f11) / (len(f01) + len(f10) + len(f11)))


def CalculateEuclideanDistance(ClusteringObject):
    IterationLength = len(ClusteringObject.rawdata)
    DistanceMatrix=np.zeros([IterationLength,IterationLength])
    for i in range(0, IterationLength):
        for j in range(i, IterationLength):
            arr1 = ClusteringObject.rawdata[i]
            arr2 = ClusteringObject.rawdata[j]
            sum_sq = np.sum(np.sqrt(np.abs(np.square(arr1 - arr2))))
            DistanceMatrix[i][j] = sum_sq
            DistanceMatrix[j][i] = sum_sq

    return(DistanceMatrix)

def CalculateManhanttanDistance():
    pass







class HierarchicalClustering:
    def __init__(self, rawdata):
        self.rawdata=rawdata
        self.DistanceMatrix=np.zeros([len(rawdata),len(rawdata)])
        self.Dendrogram=[]

    def CalculateEuclideanDistance(self):
        IterationLength = len(self.rawdata)

        for i in range(0, IterationLength):
            for j in range(i, IterationLength):
                arr1 = self.rawdata[i]
                arr2 = self.rawdata[j]
                sum_sq = np.sum(np.sqrt(np.abs(np.square(arr1 - arr2))))
                self.DistanceMatrix[i][j] = sum_sq
                self.DistanceMatrix[j][i] = sum_sq


    def runClustering(self):
        clusters=[[i] for i in range(len(self.rawdata))]
        ClusterStack=[]
        ClusterStack.append(clusters.copy())
        UpdatedDistanceMatrix=self.DistanceMatrix

        while len(clusters)>1:

            #Find smallest distance in the matrix which will dictate which two clusters will get merged together
            ClustertoCombine=np.argwhere(UpdatedDistanceMatrix==np.min(UpdatedDistanceMatrix[np.nonzero(UpdatedDistanceMatrix)]))[0]
            #generate a temp list of the new cluster group that is formed
            tmplist=list(np.concatenate( (clusters[ClustertoCombine[0]],clusters[ClustertoCombine[1]])))
            #Reassign the first cluster to the combination of the other two clusters
            clusters[ClustertoCombine[0]]=tmplist
            #Delete second cluster from the clustering object
            clusters.pop(ClustertoCombine[1])
            #Deep copy of the clustering order onto cluster stack for
            ClusterStack.insert(0,clusters.copy())


            #Combine the two clusters together based off the lowest clustering technique
            ListtoCombine=UpdatedDistanceMatrix[ClustertoCombine]
            ListtoCombine=np.delete(ListtoCombine,ClustertoCombine[1] ,axis=1)
            ListtoCombine=np.minimum(ListtoCombine[0],ListtoCombine[1])



            #Drop the rows and column of the second cluster that got merged together
            UpdatedDistanceMatrix=np.delete(UpdatedDistanceMatrix,ClustertoCombine[1],axis=0)

            UpdatedDistanceMatrix=np.delete(UpdatedDistanceMatrix,ClustertoCombine[1],axis=1)

            UpdatedDistanceMatrix[ClustertoCombine[0],:] = ListtoCombine
            UpdatedDistanceMatrix[:,ClustertoCombine[0]] = ListtoCombine


        # ClusterStack.insert(0, ["ZERO BASED CLUSTERING MEANS NOTHING"])
        # print(ClusterStack)
        #
        # SilouetteScore(ClusterStack[2],self.DistanceMatrix)









class PartitionClustering():
    def __init__(self, rawdata,n_clusters,max_iteration):
        self.rawdata=rawdata
        self.n_clusters=n_clusters
        self.DistanceMatrix=np.zeros([len(rawdata),len(rawdata)])
        self.centroids=np.zeros(n_clusters)
        self.max_iteration=max_iteration

        #These variabled get changed later, assignment not important
        self.clusters=np.zeros(len(rawdata))
        self.clusterassignment=np.zeros(len(rawdata))


    def CalculatePairwiseDistance(self,ID,cluster_mean):
        arr1 = self.rawdata[ID]
        arr2 = cluster_mean
        sum_sq = np.sum(np.sqrt(np.abs(np.square(arr1 - arr2))))
        return (sum_sq)


    #This will update the centroid in the current clustering iteration
    def UpdateCentroid(self,new_cluster):
        UpdatedCentroid=[]
        #Will start by iterating one cluser at a time, calculate the mean of the cluster,
        # and assign a new centroid to the mean of the cluster centroid.
        for cur_cluster in new_cluster:
            #Unpack Raw Data
            rawdata=self.rawdata[cur_cluster]
            #Get the mean of each feature of the raw data set, row wise mean
            cluster_mean=np.mean(rawdata,axis=0)

            dis_from_mean=[]
            #Will now calculate the euclidean distance between each point in the cluster from the cluster mean
            for ID in cur_cluster:
                dis_from_mean.append(self.CalculatePairwiseDistance(ID,cluster_mean))

            #Assign closest point to the cluster mean as the new centroid
            new_centroid_id=int(np.argwhere(dis_from_mean==np.min(dis_from_mean))[0])

            UpdatedCentroid.append(cur_cluster[new_centroid_id])
        self.centroids=UpdatedCentroid

    def FitPreductions(self):
        clusterid = [i for i in range(len(self.rawdata))]

        clusters = [[] for i in range(len(self.centroids))]
        for i in clusterid:
            if i not in self.centroids:
                DistancefromCentroid = self.DistanceMatrix[i][self.centroids]

                idxofmin = int(np.argwhere(DistancefromCentroid == DistancefromCentroid.min())[0])

                clusters[idxofmin].append(i)

        clusteridx=1
        for centroid in self.centroids:
            self.clusterassignment[centroid]=clusteridx
            clusteridx+=1

        clusteridx = 1
        for cluster in clusters:
            for ID in cluster:
                self.clusterassignment[ID] = clusteridx
            clusteridx+=1
        self.clusters=cluster



    def runClustering(self):
        clusters = [i for i in range(len(self.rawdata))]

        #STEP 1, choose a random number where we assign the first round of k based centroids
        centroids=np.random.choice(clusters,self.n_clusters)
        self.centroids=centroids

        #STEP 2 Until best centroids found, with new set of centroids we will assign each intem into thier closest
        #centroid cluster.
        #create new tmp matirx to hold values of the current clustering implementation
        iteration=0
        prev_centroid = []
        while ( not(self.centroids in prev_centroid ) and (iteration < self.max_iteration) ):

            iteration+=1
            #assign previous centroid to list to check if the updated centoids does not convergence at a point
            prev_centroid.append(list(self.centroids))
            new_clusters = [[] for i in range(len(self.centroids))]
            for i in clusters:
                if i not in self.centroids:
                    #Want to modify CalculatePairwiseDistance so it can also be used here so I don't need to initalize a
                    #distance matrix for this clustering method, but the calculations should be the same
                    DistancefromCentroid=self.DistanceMatrix[i][self.centroids]
                    # DistancefromCentroid=self.CalculatePairwiseDistance(i,self.centroids)

                    #get index of the mimimum distance to all the centroid
                    idxofmin=int(np.argwhere(DistancefromCentroid==DistancefromCentroid.min())[0])

                    #Assign current ID to the cluster
                    new_clusters[idxofmin].append(i)



            #STEP 3 Calulcate the mean of each cluster and assign mean centroid the new cluster centroid.
            self.UpdateCentroid(new_clusters)

            self.clusters = new_clusters

        #Finally with the predicted centroid, we will assign each indiv their cluster assignment, with each index
        #in the array corresponding to a row inside self.rawdata.
        #self.clusterassignment will be a filled array with each ID corresponding the the cluster of that datapoint
        self.FitPreductions()






