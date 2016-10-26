from math import exp
import numpy as np
from numpy import linalg as LA
import itertools
import random

v1 = [0,0]
v2 = [0,1]
v3 = [1,0]
v4 = [4,2]
v5 = [5,2]
v6 = [5,3]

v = [v1, v2, v3, v4, v5, v6]
W = []

def GaussianKernel(v1, v2, sigma=1):
	x = np.array(v1)
	y = np.array(v2)
	return exp(-LA.norm(x-y, 2)**2/(2.*sigma**2))

for i in v:
	W.append([])
	for j in v:
		W[(len(W)-1)].append(GaussianKernel(i,j))

def pretty_print(m):
	for i in m:
		print i

f_all = [list(i) for i in itertools.product([0,1], repeat=len(v))]

def calc_r(adj, f):
	r = 0
	for i, row in enumerate(adj):
		for j, w in enumerate(row):
			r += w * (f[i] - f[j])**2
	return r

def find_best_f():
	lowest = (float("inf"), None)
	for f in f_all[1:-1]:
		candidate = calc_r(W, f)
		if candidate < lowest[0]:
			lowest = (candidate, f)
	return lowest


W_matrix = np.array(W)
D_matrix = np.zeros(W_matrix.shape)

for x in xrange(W_matrix.shape[0]):
	D_matrix[x, x] = W_matrix[x].sum()

L = D_matrix - W_matrix

def return_next_smallest(arr, val):
	next_smallest = [float("inf"), float("inf")]
	for i, x in enumerate(arr):
		if x > val and x < next_smallest[0]:
			next_smallest = [x, i]
	return next_smallest

def find_smallest_eigenval_index(a, k):
	# Note: This returns the garbage '0' eigenvalue and corresponding eigenvector
	# as the first result, so remove it before returning
	eigenvalsandvecs = LA.eig(a)
	smallest = float("-inf")
	valindexes = []
	for x in xrange(k+1):
		nextcandidate = return_next_smallest(eigenvalsandvecs[0], smallest)
		smallest = nextcandidate[0]
		valindexes.append(nextcandidate[1])
	vectors = []
	for x in valindexes:
		vectors.append(eigenvalsandvecs[1][:,x])
	return vectors[1:]

vec = find_smallest_eigenval_index(L, 1)

def distance(p1, p2):
	return abs(p1 - p2)

def getPoints(matrix):
	return [p for p in matrix[0]]

def computeCentroid(points):
	return sum(points) / len(points)

def minAndMax(points):
	highestPoint = 0
	lowestPoint = 0
	for x in points:
		if x > highestPoint:
			highestPoint = x 
		if x < lowestPoint:
			lowestPoint = x 
	return {'highest': highestPoint, 'lowest': lowestPoint}

def assignClusters(clusterCenters, points):
	clusters = {}
	for centroid in clusterCenters:
		clusters[centroid] = []
	for pointIndex, point in enumerate(points):
		dist = float("inf")
		closestCentroid = None
		for centroid in clusterCenters:
			distCandidate = distance(clusterCenters[centroid], point)
			if distCandidate < dist:
				dist = distCandidate
				closestCentroid = centroid
		clusters[closestCentroid].append(pointIndex)
	return clusters

def calculateNewCenters(clusters, points):
	clusterCenters = {}
	for cluster in clusters:
		l = []
		for p in clusters[cluster]:
			l.append(points[p])
		clusterCenters[cluster] = computeCentroid(l)
	return clusterCenters

def compareClusters(c1, c2):
	for x in c1.keys():
		if c1[x] != c2[x]:
			return False
	return True

def k_means(points, k):
	# find lowest and highest num in points
	# loop through k and make an object where
	# the keys are the cluster number and the value is a random point
	# between lowest and highest
	clusterCenters = {}
	centersRange = minAndMax(points)
	for x in xrange(k):
		clusterCenters[x] = random.uniform(centersRange['lowest'], centersRange['highest'])
	clusters = assignClusters(clusterCenters, points)
	while True:
		updatedClusterCenters = calculateNewCenters(clusters, points)
		if not compareClusters(clusterCenters, updatedClusterCenters):
			clusterCenters = updatedClusterCenters
			clusters = assignClusters(clusterCenters, points)
		else:
			break
	return clusters

# Make work for > 2 clusters
