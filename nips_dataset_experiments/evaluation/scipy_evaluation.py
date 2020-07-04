from scipy import stats
from scipy import sparse
from numpy import array
import numpy as np
from scipy.spatial import distance

beta = -1.0
i_list = []
j_list = []
v_list = []
fc = open("C_matrix.txt","r")

for fline in fc:
  l = fline.split(" ")
  i_list.append(int(l[0]))
  j_list.append(int(l[1]))
  v_list.append(-int(l[2]))
fc.close()

n = 2715
I = array(i_list)
J = array(j_list)
V = array(v_list)
edges_dict = {}
for i in range(len(I)):
    edges_dict[(I[i],J[i])] = abs(V[i])
    edges_dict[(J[i],I[i])] = abs(V[i])
C = sparse.coo_matrix((V,(I,J)),shape=(n,n))
#C = C.toarray()

C_sum = np.sum(C.toarray(),axis=0)

s_i_greather_than_10 = np.abs(C_sum) >= 10
s_i_greather_than_20 = np.abs(C_sum) >= 20

evaluate_euclidean_representations = False
recall_at_1 = 0.0
if evaluate_euclidean_representations:
    ambient_euclidean_dimensionality = 6
    print("Euclidean space of dimensionality %d" % ambient_euclidean_dimensionality)
    file_name = "nips_data/euclidean_%d/1/x.txt" % ambient_euclidean_dimensionality
    X = np.loadtxt(file_name, usecols=range(ambient_euclidean_dimensionality))
    K = distance.cdist(X, X, 'sqeuclidean')
    
else:

    dimensionality_of_ambient_space  = 7
    time_dimensions = 4
    print("dimensionality of the ambient space = %d" % dimensionality_of_ambient_space)
    if time_dimensions == 1:
        print("hyperbolic case")
    elif time_dimensions == dimensionality_of_ambient_space :
        print("spherical case")
    else:
        print("ultrahyperbolic case with %d time dimensions" % time_dimensions)

    file_name = "nips_data/d_%d_q_%d/1/x.txt" % (dimensionality_of_ambient_space , time_dimensions)
    X = np.loadtxt(file_name, usecols=range(dimensionality_of_ambient_space))
    G = np.identity(dimensionality_of_ambient_space )
    for i in range(time_dimensions):
        G[i,i] = -1
    K = np.matmul(np.matmul(X,G), X.transpose())
    hyperbolic_indices = K <= -1.0
    if time_dimensions == dimensionality_of_ambient_space :
        linear_approximation = K > 1.0
    else:
        linear_approximation = K > 0.0
    spherical_indices = ~hyperbolic_indices & ~linear_approximation
    K[hyperbolic_indices] = np.arccosh(K[hyperbolic_indices] / beta)
    K[spherical_indices] = np.arccos(K[spherical_indices] / beta)
    K[linear_approximation] = np.pi - K[linear_approximation] / beta
for u in range(K.shape[0]):
    K[u][u] = 0.0
K_sum = np.sum(K,axis=0)

print("Spearman's rank correlation coefficient for the whole dataset:")
print(stats.spearmanr(C_sum,K_sum))
print("Spearman's rank correlation coefficient for s_i >= 10")
print(stats.spearmanr(C_sum[s_i_greather_than_10],K_sum[s_i_greather_than_10]))
print("Spearman's rank correlation coefficient for s_i >= 20")
print(stats.spearmanr(C_sum[s_i_greather_than_20],K_sum[s_i_greather_than_20]))
for u in range(K.shape[0]):
    K[u][u] = np.Inf
argmin_distance = np.argmin(K, axis=0)
for u in range(K.shape[0]):
    if (u, argmin_distance[u]) in edges_dict:
        recall_at_1 += 1.0
print("recall at 1 = %f" % (100.0 * recall_at_1 / n))
