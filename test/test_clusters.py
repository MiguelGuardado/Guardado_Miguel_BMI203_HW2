from clusters import algs
import numpy as np
import pytest

def test_partitioning():


	assert True

def test_hierarchical():
	assert True

def load_ligand_class():
	assert True

def test_tanimoto():
	A = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
	B = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
	value = algs.TanimotoCoeff(A, B)
	assert (value==1)

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
	assert True

