import numpy as np

def SilouetteScore(ClusteringObject):
    sil_score = []
    #Will Iterate one cluster at a time
    if len(ClusteringObject.clusters)==1:
        return -1

    for cur_cluster in ClusteringObject.clusters:
        if len(cur_cluster)==1:
            sil_score.append(-1)
        #For each cluster we will go though each individual
        for ID in cur_cluster:
            print(ID)
            print(cur_cluster)
            # get similarity to other elements in cluster
            dists = [ClusteringObject.CalculatePairwiseDistance(ID,cluster_nodes) for cluster_nodes in cur_cluster if ID != cluster_nodes]
            a =  sum(dists)/len(dists)

            # get the minimum distance to elements of another cluster
            b=[]
            for tmp in ClusteringObject.clusters:
                print(tmp)
                if len(tmp) > 1:
                    if cur_cluster != tmp:
                        dists = [ClusteringObject.CalculatePairwiseDistance(ID, cluster_nodes) for cluster_nodes in tmp if ID != cluster_nodes]
                        b.append(sum(dists) / len(dists))
            b = min(b)

            # calculate s(i) based on vector operations of a and b variables
            sil_score.append((b - a) / max(a, b))

    #Now we will assign each silouette score to thier own
    print(np.mean(sil_score))
    return (np.mean(sil_score))




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
        print(self.DistanceMatrix)
        while len(clusters)>1:
            print(len(clusters))
            print(UpdatedDistanceMatrix)
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


    def calc_distance_from_mean(self, ID, cluster_mean):
        arr1 = self.rawdata[ID]
        arr2 = cluster_mean
        sum_sq = np.sum(np.sqrt(np.abs(np.square(arr1 - arr2))))
        return (sum_sq)

    def CalculatePairwiseDistance(self,ID1,ID2):
        arr1 = self.rawdata[ID1]
        arr2 = self.rawdata[ID2]
        sum_sq = np.sum(np.sqrt(np.abs(np.square(arr1 - arr2))))
        return (sum_sq)

    def empty_cluster(self,new_cluster):
        #I will iterate though each cluster finding all empty clusters and assign the furthest point in the largest cluster as the
        # cluster. Will do this for N<K empty clusters, where K is the number clusters defined by the user

        #Find the largest cluster and calculate the distance from the centroid to all the other nodes not in the largest cluster.
        cluster_len=[]
        for cluster in new_cluster:
            cluster_len.append(len(cluster))
        #Find index of largest cluster
        largest_cluster=cluster_len.index(max(cluster_len))
        #Convert the new_cluster to np list because thats just gonna make my life easier for this next step
        new_cluster=np.array(new_cluster,dtype=object)
        #Create a subset of new_cluster where we remove the largest cluster, we will now find the cluster furthest away from
        # largest cluster centroid
        # small_clusters=np.delete(new_cluster,largest_cluster)
        cluster_idx=0
        small_clusters_dist=[]
        for i in range(0,len(new_cluster)):
            if i != largest_cluster:
                print(i,largest_cluster)
                print(self.rawdata[i])
                print(self.rawdata[largest_cluster])
                dis=self.CalculatePairwiseDistance(self.rawdata[i],self.rawdata[self.centroids[largest_cluster]])
                small_clusters_dist.append(dis)
                #small_clusters_dist.append(np.sum(np.sqrt(np.abs(np.square(self.rawdata[i] - self.rawdata[largest_cluster])))))
            else:
                small_clusters_dist.append(-np.inf)

        #Get the cluster with the furthest centroid  distance to the largest cluster
        furthest_cluster_idx=small_clusters_dist.index(max(small_clusters_dist))

        furthest_cluster_indiv_distances=[]

        for ID in new_cluster[furthest_cluster_idx]:
            dis =  self.CalculatePairwiseDistance(self.rawdata[ID],self.rawdata[self.centroids[largest_cluster]])
            furthest_cluster_indiv_distances.append(dis)

        print(furthest_cluster_indiv_distances)












        exit(0)




    #This will update the centroid in the current clustering iteration
    def UpdateCentroid(self,new_cluster):
        UpdatedCentroid=np.zeros(self.n_clusters,dtype=int)
        #Will start by iterating one cluser at a time, calculate the mean of the cluster,
        # and assign a new centroid to the mean of the cluster centroid.
        idx=0
        for cur_cluster in new_cluster:
            #Unpack Raw Data
            rawdata=self.rawdata[cur_cluster]
            #Get the mean of each feature of the raw data set, row wise mean
            cluster_mean=np.mean(rawdata,axis=0)

            dis_from_mean=[]
            #Will now calculate the euclidean distance between each point in the cluster from the cluster mean
            for ID in cur_cluster:
                dis_from_mean.append(self.calc_distance_from_mean(ID, cluster_mean))
            print(dis_from_mean)
            #Assign closest point to the cluster mean as the new centroid
            new_centroid_id=int(np.argwhere(dis_from_mean==np.min(dis_from_mean))[0])

            UpdatedCentroid[idx]=cur_cluster[new_centroid_id]
            idx=idx+1

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

        self.clusters=clusters



    def runClustering(self):
        rawdataID = [i for i in range(len(self.rawdata))]

        #STEP 1, choose a random number where we assign the first round of k based centroids
        centroids=np.random.choice(rawdataID,self.n_clusters)
        self.centroids=centroids


        #STEP 2 Until best centroids found, with new set of centroids we will assign each intem into thier closest
        #centroid cluster.
        #create new tmp matirx to hold values of the current clustering implementation
        iteration=0
        prev_centroid = np.empty(0)
        while ( not(self.centroids in prev_centroid ) and (iteration < self.max_iteration) ):
            iteration+=1
            #assign previous centroid to list to check if the updated centoids does not convergence at a point
            np.append(prev_centroid,self.centroids)
            new_clusters = [[] for i in range(len(self.centroids))]
            for i in rawdataID:
                if i not in self.centroids:
                    #Want to modify CalculatePairwiseDistance so it can also be used here so I don't need to initalize a
                    #distance matrix for this clustering method, but the calculations should be the same
                    #DistancefromCentroid=self.DistanceMatrix[i][self.centroids]
                    DistancefromCentroid=[]
                    for j in self.centroids:
                        DistancefromCentroid.append(self.CalculatePairwiseDistance(i,j))
                    # DistancefromCentroid=self.CalculatePairwiseDistance(i,self.centroids)
                    DistancefromCentroid=np.array(DistancefromCentroid)
                    #get index of the mimimum distance to all the centroid
                    idxofmin=int(np.argwhere(DistancefromCentroid==DistancefromCentroid.min())[0])

                    #Assign current ID to the cluster
                    new_clusters[idxofmin].append(i)
                else:
                    clusteridx=int(np.argwhere(self.centroids == i)[0])
                    new_clusters[clusteridx].append(i)

            #STEP 3 Calulcate the mean of each cluster and assign mean centroid the new cluster centroid.
            self.UpdateCentroid(new_clusters)
            self.clusters = new_clusters

        #Finally with the predicted centroid, we will assign each indiv their cluster assignment, with each index
        #in the array corresponding to a row inside self.rawdata.
        #self.clusterassignment will be a filled array with each ID corresponding the the cluster of that datapoint
        self.FitPreductions()






