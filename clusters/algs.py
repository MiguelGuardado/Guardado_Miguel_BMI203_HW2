import numpy as np


def SilhouetteScore(ClusteringObject):
    """Calculates the Silhouette Score of your Cluster.
    This score can be used to evaluate the quality of clusters created by the clustering algorithim. This will calcualte
    the silhouette score for each observation/data point and return the mean of every point. The range of values
    calculated and returned will be between [-1,1]. The algorithim will calcute for each point
    a = Cohesion of each observation within a cluster
    b = Seperation of each observation within other clusters
    s(i) = b-a/max(a,b)

    Args:
      ClusteringObject (object): This can be either a HierarchicalClustering or a PartitionClustering object from algs class that has a filled ClusteringObject.clusters.

    Returns:
      int: Returns the mean Silhouette score across all individuals observations, value will be between [-1,1]. A value
      close to 1 will mean the clusters is well seperated and dense, 0 representing overlapping clusters, and -1 meaning
      the samples might have been assigned to the wrong cluster.


    """
    sil_score = []
    #Will Iterate one cluster at a time
    if len(ClusteringObject.clusters)==1:
        return -1

    for cur_cluster in ClusteringObject.clusters:

        if len(cur_cluster)==1:
            sil_score.append(-1)
            break
        #For each cluster we will go though each individual
        for ID in cur_cluster:
            # get similarity to other elements in cluster, or a
            #dists = [ClusteringObject.CalculatePairwiseDistance(ID,cluster_nodes) for cluster_nodes in cur_cluster if ID != cluster_nodes]
            dists = [CalculatePairWiseDistance(ClusteringObject.rawdata[ID], ClusteringObject.rawdata[cluster_nodes]) for cluster_nodes in cur_cluster if
                     ID != cluster_nodes]
            a =  sum(dists)/len(dists)

            # get the minimum distance to elements of another cluster, or b
            b=[]
            for tmp in ClusteringObject.clusters:
                if len(tmp) > 1:
                    if cur_cluster != tmp:
                        dists = [CalculatePairWiseDistance(ClusteringObject.rawdata[ID], ClusteringObject.rawdata[cluster_nodes])
                                 for cluster_nodes in tmp if ID != cluster_nodes]
                        b.append(sum(dists) / len(dists))
            b = min(b)

            # calculate s(i) based on vector operations of a and b variables
            sil_score.append((b - a) / max(a, b))

    #Now we will assign each silouette score to thier own
    return (np.mean(sil_score))




def TanimotoCoeff(ClusterA,ClusterB):
    """Calculates the Tanimoto similarity Coefficent for two given clusters.
    This is used to evaluate the similarity of one clustering compared to another clustering. The Tanimoto coefficient
    is implemented the same way as Jacard Distance, since T. Tanimoto created this implementation independently from
    Jaccard with the primary purpose for Cheminformatics. Caluclating this coefficeint will come from dividing the
    intersection of the two sets from the union of both sets.
    J(x,y)=|X insec Y|/|X union Y|

    Args:
      ClusterA (list): This can contain either a list of list where each list inside the main list represent a cluster.
      This can be a list type or a numpy array where each row is a cluster.
        ex: [[1,2,3],[4,5,6],[7,8]]

      ClusterB (list): Same as cluster A, make sure the type of list created is consistent and the same size as the first
      cluster.

    Returns:
      int: The range of this coefficient is between [0,1] with a value of 1 being identical clusters
      and 0 having no identical matches found in the cluster.

    """

    SetA=[]
    SetB=[]
    #Iterate though each cluster and create a two pair set for each item in a cluster
    for clus_a, clus_b in zip(ClusterA,ClusterB):
        for id1 in clus_a:
            for id2 in clus_a:
                setinput=[id1,id2]
                setinput=list(np.sort(setinput))
                #Input combination of pair as long as the index are not equal
                if (id1!=id2 ):
                    SetA.append(setinput)

        for id1 in clus_b:
            for id2 in clus_b:
                setinput=[id1,id2]
                setinput=list(np.sort(setinput))
                # Input combination of pair as long as the index are not equal
                if (id1!=id2 ):
                    SetB.append(setinput)

    #Remove duplicated inside each set file
    tmp = set(tuple(x) for x in SetA)
    SetA = [list(x) for x in tmp]

    tmp = set(tuple(x) for x in SetB)
    SetB = [list(x) for x in tmp]

    #I will convert the list of clusters into a set
    SetA = set(frozenset(i) for i in SetA)
    SetB = set(frozenset(i) for i in SetB)

    #Same Clusters in both sets
    f00 = SetA.union(SetB)
    # Different clusters in both
    f11 = SetA.intersection(SetB)
    #Same in Cluster 1, not in 2
    f01 = SetA.difference(SetB)
    #Same in Cluster 2, not in 1
    f10 = SetB.difference(SetA)

    return(len(f11) / (len(f01) + len(f10) + len(f11)))




def CalculatePairWiseDistance(arr1,arr2):
    """Calculates the Euclidean distance between two observations on N based features.
    This utilizes vector operation so when you input the data be sure to input it as a singular dimension of data.

    Args:
      arr1 (list): 1-d array of data. ex [1,2,4,4,5]

      arr2 (list): 1-d array of data. ex [2,5,6,3,1]

    Returns:
      int: Single value of the euclidean distance between these two obsevations.

    """
    if(len(arr1)!=len(arr2)):
        return(0)

    return(np.sum(np.sqrt(np.abs(np.square(arr1 - arr2)))))






class HierarchicalClustering:
    """This is a class is used to preform Hierarchial Clustering on a set of M observations by N features matrix.
    This type of clustering will be an Agglomerative clustering, or a bottom up approach, where we start off with each
    node in thier own cluster, combining pairs clusters one iteration at a time until we get to a desired amount of
    clusters to stop at. This is an implementation of a deterministic clustering algorithim

        Attributes:
            rawdata (array-like): Matrix of M observation x N features, must be set so the row represents an individual observation
            DistanceMatrix (array-like): Distance Matrix of [M x M] features where each index represents the euclidean distance
            clusters(array-like): empty at initialization, stores a cluster array with each list inside the main list being a cluster of obvservations.
            n_clusters(int): Specify the number of clusters we should stop our petitioning algorithim at.
            clustersassignment(array-like): empty at initialization, stores an 1-d array of size M and stores the cluster ID each nodes belings to

        """
    def __init__(self, rawdata,n_clusters):
        self.rawdata=rawdata
        self.DistanceMatrix=np.zeros([len(rawdata),len(rawdata)])
        self.clusters=[]
        self.n_clusters=n_clusters
        self.clusterassignment = np.zeros(len(rawdata))

        #Fill input matrix from the input of rawdata
        self.CalculateEuclideanDistance()

    def CalculateEuclideanDistance(self):
        """This helper function will help fill the distance matrix needed for the HierarchicalClustering clustering
        algorithim, this will iterate us though each observation calculating the Euclidean pairwise distance between the two obsevations.

            Returns:
              self.DistanceMatrix(array-like): Distance Matrix of the object. Will now be non-empty array

            """
        IterationLength = len(self.rawdata)

        for i in range(0, IterationLength):
            for j in range(i, IterationLength):
                if i==j:
                    self.DistanceMatrix[i][j]=np.NAN
                else:
                    sum_sq = CalculatePairWiseDistance(self.rawdata[i],self.rawdata[j])
                    self.DistanceMatrix[i][j] = sum_sq
                    self.DistanceMatrix[j][i] = sum_sq

    def FitPreductions(self):
        """This helper function will be used to fill self.clusterassignment, where we will go though each cluster and
        assign each observation in that cluster a numerical label, The number of unique numerical label will be equal
        to the n_clusters you define as a user.

        example:
        Input: [[1,2,3],[4,5],[6,7,8,9]]
        Output: [1,1,1,2,2,3,3,3,3]

        """
        idx = 1
        for cur_cluster in self.clusters:
            for ID in cur_cluster:
                self.clusterassignment[ID]=idx
            idx=idx+1


    def runClustering(self):
        """ This function is used to run my clustering algorithim. This will be a agglomerative clustering implementation.
        A minor psudocode implementation of our alg will be:
        1. Assign each node into their own cluster
        while the length of cluster > number of clusters desired
        2. Find the smallest non.nan distance in the Distance Matrix
        3 Combine the clusters where the distance is found
        4. Update distance matrix we combine the distances of the two observations and prune a row/col.
        done
        5. Assign each observation thier clusterID

            Returns:
              self.clusters (array-like): list of list of observations where each list represents a cluster and the observations found in that cluster
              self.clusterassignment(array-like): Cluster assignment for each of the observation

            """
        clusters=[[i] for i in range(len(self.rawdata))]
        ClusterStack=[]
        ClusterStack.append(clusters.copy())
        UpdatedDistanceMatrix=self.DistanceMatrix
        while len(clusters)>self.n_clusters:
            #Find smallest distance in the matrix which will dictate which two clusters will get merged together
            ClustertoCombine=np.argwhere(UpdatedDistanceMatrix==np.nanmin(UpdatedDistanceMatrix))[0]
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
        self.clusters=clusters
        self.FitPreductions()





class PartitionClustering():
    """
    This is a class is used to preform Partition Clustering on a set of M observations by N features matrix.
    This type of clustering will be an Agglomerative clustering, or a bottom up approach, where we start off with
    each node in thier own cluster, combining pairs clusters one iteration at a time until we get to a desired amount
    of clusters to stop at. This is an implementation of a non-deterministic clustering algorithim

        Attributes:
            rawdata (array-like): Matrix of M observation x N features, must be set so the row represents an individual observation
            DistanceMatrix (array-like): Distance Matrix of [M x M] features where each index represents the euclidean distance
            centroids(array-like): will hold the centoids value
            max_iteration(int): Since the time to converge to a centroid is unknown this parameter is here to allow us to break after a certain amount of iterations
            clusters(array-like): Empty at initialization, stores a cluster array with each list inside the main list being a cluster of obvservations.
            n_clusters(int): Specify the number of clusters we should stop our petitioning algorithim at.
            clustersassignment(array-like): empty at initialization, stores an 1-d array of size M and stores the cluster ID each nodes belings to

        """

    def __init__(self, rawdata,n_clusters,max_iteration):
        self.rawdata=rawdata
        self.n_clusters=n_clusters
        self.DistanceMatrix=np.zeros([len(rawdata),len(rawdata)])
        self.centroids=np.zeros(n_clusters)
        self.max_iteration=max_iteration

        #These variabled get changed later, assignment not important
        self.clusters=np.zeros(len(rawdata))
        self.clusterassignment=np.zeros(len(rawdata))


    #This will update the centroid in the current clustering iteration
    def UpdateCentroid(self,new_cluster):
        """This function is used as a helper function of my run.clustering() function to identify the new Centroid.
        This is used to take in the current clustering array, calculate the mean of each cluster, and assign the closet
        point to the mean as the new centroid. I implemented a style choice of having the centroid be the observation closest
        to the cluster mean rather then the cluster mean itself, while this is not the best Mathemetical implementation,
        it allows us to ignore most edge cases when a updated centroid mean may produde and empty cluster. Also provides
        a slightly faster implementation.

            Args:
                new_cluster (array-like): current iteration of clusters assignments,

            Returns:
                self.centoid (array-like): this will return an updated version of centorid nodes

            """
        UpdatedCentroid=np.zeros(self.n_clusters,dtype=int)
        #Will start by iterating one cluser at a time, calculate the mean of the cluster,
        # and assign a new centroid to the mean of the cluster centroid.
        idx=0
        for cur_cluster in new_cluster:
            #Unpack Raw Data
            rawdata=self.rawdata[cur_cluster]
            # print(new_cluster)
            #Get the mean of each feature of the raw data set, row wise mean
            cluster_mean=np.mean(rawdata,axis=0)

            dis_from_mean=[]
            #Will now calculate the euclidean distance between each point in the cluster from the cluster mean
            # print(cur_cluster)
            for ID in cur_cluster:
                arr1=self.rawdata[ID]
                dis_from_mean.append(CalculatePairWiseDistance(arr1,cluster_mean))
            # print(dis_from_mean)
            #Assign closest point to the cluster mean as the new centroid
            new_centroid_id=int(np.argwhere(dis_from_mean==np.min(dis_from_mean))[0])

            UpdatedCentroid[idx]=cur_cluster[new_centroid_id]
            idx=idx+1
        #Update centoid function
        self.centroids=UpdatedCentroid

    def FitPreductions(self):
        """This helper function will be used to fill self.clusterassignment, where we will go though each cluster and
        assign each observation in that cluster a numerical label, The number of unique numerical label will be equal
        to the n_clusters you define as a user.

        example:
        Input: [[1,2,3],[4,5],[6,7,8,9]]

        Output: [1,1,1,2,2,3,3,3,3]

        """
        clusterid = [i for i in range(len(self.rawdata))]

        clusters = [[] for i in range(len(self.centroids))]
        for i in clusterid:
            if i not in self.centroids:
                DistancefromCentroid=[]
                for centroid in self.centroids:
                    DistancefromCentroid.append(CalculatePairWiseDistance(self.rawdata[i],self.rawdata[centroid]))

                DistancefromCentroid=np.array(DistancefromCentroid)
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

        #self.clusters=clusters



    def runClustering(self):
        """ This function is used to run our clustering algorithim. This will be a k-means clustering implementation.
        A light psudocode implementation of my alg will be:
        1. Choose a random set of observations to be the initial centroids
        while centroids have not converged or reached max_iterations:
        2. Assign each observation to their closest cluster centroid
        3  Update the centroid based off the mean of the newly assigned cluster
        done
        4. Assign each observation thier clusterID


            Returns:
              self.clusters (array-like): list of list of observations where each list represents a cluster and the observations found in that cluster
              self.clusterassignment(array-like): Cluster assignment for each of the observation

            """
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
                        DistancefromCentroid.append(CalculatePairWiseDistance(self.rawdata[i],self.rawdata[j]))
                        #DistancefromCentroid.append(self.CalculatePairwiseDistance(i,j))
                    # DistancefromCentroid=self.CalculatePairwiseDistance(i,self.centroids)
                    DistancefromCentroid=np.array(DistancefromCentroid)
                    #get index of the mimimum distance to all the centroid
                    idxofmin=int(np.argwhere(DistancefromCentroid==DistancefromCentroid.min())[0])

                    #Assign current ID to the cluster
                    new_clusters[idxofmin].append(i)
                else:
                    clusteridx=int(np.argwhere(self.centroids == i)[0])
                    new_clusters[clusteridx].append(i)
                    #print(new_clusters)

            #STEP 3 Calulcate the mean of each cluster and assign mean centroid the new cluster centroid.
            self.UpdateCentroid(new_clusters)
            self.clusters = new_clusters
        #Finally with the predicted centroid, we will assign each indiv their cluster assignment, with each index
        #in the array corresponding to a row inside self.rawdata.
        #self.clusterassignment will be a filled array with each ID corresponding the the cluster of that datapoint
        self.FitPreductions()
