from clusters import algs
import numpy as np
import pytest


def test_partitioning():
	testcluster = np.array([[3.08232755e-01, 7.31276243e-01],
							[1.38059574e-01, 5.96831094e-01],
							[7.17477934e-01, 6.92660634e-01],
							[1.04842083e-01, 5.81815300e-01],
							[2.63517862e-01, 8.56987831e-01],
							[6.82660482e-01, 7.65745298e-01],
							[3.30899459e-01, 1.27005643e-01],
							[2.15388524e+00, 2.76495447e+00],
							[2.02847470e+00, 2.17510569e+00],
							[2.81339552e+00, 2.92175026e+00],
							[2.11079023e+00, 2.70619934e+00],
							[2.51975852e+00, 2.72664963e+00]])

	TestPT = algs.PartitionClustering(rawdata=testcluster, max_iteration=100, n_clusters=2)
	TestPT.runClustering()
	assert (len(TestPT.centroids) == 2)
	assert (len(TestPT.clusters) == 2)
	assert (sorted(TestPT.centroids) == [0, 10])
	assert (sorted(TestPT.clusters) == [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])


def test_hierarchical():
	testcluster = np.array([[3.08232755e-01, 7.31276243e-01],
							[1.38059574e-01, 5.96831094e-01],
							[7.17477934e-01, 6.92660634e-01],
							[1.04842083e-01, 5.81815300e-01],
							[2.63517862e-01, 8.56987831e-01],
							[6.82660482e-01, 7.65745298e-01],
							[3.30899459e-01, 1.27005643e-01],
							[2.15388524e+00, 2.76495447e+00],
							[2.02847470e+00, 2.17510569e+00],
							[2.81339552e+00, 2.92175026e+00],
							[2.11079023e+00, 2.70619934e+00],
							[2.51975852e+00, 2.72664963e+00]])

	TestHC = algs.HierarchicalClustering(rawdata=testcluster, n_clusters=2)
	TestHC.runClustering()

	assert (TestHC.DistanceMatrix.shape==(len(testcluster),len(testcluster)))
	assert (len(TestHC.clusters) == 2)
	sorted_clus = []
	for clus in TestHC.clusters:
		sorted_clus.append(sorted(clus))
	assert (sorted(sorted_clus) == [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])



def test_tanimoto():
	#Identical
	A = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
	B = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
	value = algs.TanimotoCoeff(A, B)
	assert (value==1)
	#Expected cluster = 0.25
	A = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
	B = [[1, 7, 3, 9, 5], [6, 2, 8, 4, 10]]
	value = algs.TanimotoCoeff(A, B)
	assert (value==0.25)

def test_pairwise_distance():
	#one dimension
	value = algs.CalculatePairWiseDistance(np.array([10]), np.array([5]))
	assert (value==5)
	#two dimesnions
	value = algs.CalculatePairWiseDistance(np.array([10,5]), np.array([5,10]))
	assert(value==10)
	#three dimensions
	value = algs.CalculatePairWiseDistance(np.array([10, 5, 2]), np.array([5, 10, 8]))
	assert (value == 16)


def test_similarity():
	testcluster = np.array([[3.08232755e-01, 7.31276243e-01],
							[1.38059574e-01, 5.96831094e-01],
							[7.17477934e-01, 6.92660634e-01],
							[1.04842083e-01, 5.81815300e-01],
							[2.63517862e-01, 8.56987831e-01],
							[6.82660482e-01, 7.65745298e-01],
							[3.30899459e-01, 1.27005643e-01],
							[2.15388524e+00, 2.76495447e+00],
							[2.02847470e+00, 2.17510569e+00],
							[2.81339552e+00, 2.92175026e+00],
							[2.11079023e+00, 2.70619934e+00],
							[2.51975852e+00, 2.72664963e+00]])

	TestPT = algs.PartitionClustering(rawdata=testcluster, n_clusters=2, max_iteration=100)
	TestPT.runClustering()

	score = algs.SilhouetteScore(TestPT)
	assert (score>0.80)

	TestPT = algs.HierarchicalClustering(rawdata=testcluster, n_clusters=2)
	TestPT.runClustering()

	score = algs.SilhouetteScore(TestPT)
	assert (score>0.80)


