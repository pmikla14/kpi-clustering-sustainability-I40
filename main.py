from sklearn.cluster import AgglomerativeClustering
import numpy as np
from numpy import genfromtxt

# S ... symmetric k-dimensional array that stores the similarity between each pair of the k kpis
# similarity of kpis was computed by postgresql pg_trgm.word_similarity method
# so S(n,m) returns the similarity between the KPIs n and m
# S(n,m) = 1 ... same kpi (e.g., n=m)
# S(n,m) = 0 ... completely different kpi
S = genfromtxt('data/similarities_query_results.csv', delimiter=',')

# D ... symmetric k-dimensional array that stores the distances (= 1-similarity) between each pair of the k kpis
# so D(n,m) returns the difference between the KPIs n and m
# D(n,m) = 0 ... same kpi (e.g., n=m)
# D(n,m) = 1 ... completely different kpi
#
D = 1-S;

#

# create agglomerative clustering
# affinity='precomputed' indicates that the input will be a distance matrix
# distance_threshold=0.5 indicates that distances greater than 0.6 will not be clustered together
# e.g., two KPIs n and m will not be in the same cluster when D(n,m) > 0.6
# Note PM: threshold has been set to 0.5 initially, but obviously similar KPIs had differences greater than 0.5
#   -> reason: postgresql function
# Note PM: linkage was changed from 'single' to 'complete' to be more critical in cluster generation
#   -> by complete two clusters are only merged when all of the n:m differnces are greater than the threshold
clusteringKpis = AgglomerativeClustering(affinity='precomputed', n_clusters=None, distance_threshold=0.6, linkage='complete')
clusteringKpis.fit(D)
clusteredKpis = clusteringKpis.labels_

np.savetxt('data/clusters_thresh06_linkComplete.txt', clusteredKpis, delimiter='\n', fmt='%i')