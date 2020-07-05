from scipy import stats
from scipy import sparse
from numpy import array
import numpy as np
from scipy.spatial import distance

evaluate_euclidean_representations = False
time_dimensions = 3



nb_splits = 5
ambient_euclidean_dimensionality = 6
dimensionality_of_ambient_space  = 5

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

n = 34
I = array(i_list)
J = array(j_list)
V = array(v_list)
edges_dict = {}
for i in range(len(I)):
    edges_dict[(I[i],J[i])] = abs(V[i])
    edges_dict[(J[i],I[i])] = abs(V[i])
C = sparse.coo_matrix((V,(I,J)),shape=(n,n))

C = C.toarray()
C = C + C.transpose()

C_sum = np.sum(C,axis=0)

top_10 = [33,0,32,2,1,31,23,3,8,13]
top_5 = [33,0,32,2,1]

recall_at_1 = 0.0
rank_first_leader = []
rank_second_leader = []
rho5_list = []
rho10_list = []

for i in range(nb_splits):
    if evaluate_euclidean_representations:
        file_name = "zachary_data/euclidean/%d/d.txt" % (i+1)
        D = np.loadtxt(file_name, usecols=range(n))
    else:

        file_name = "zachary_data/d_%d_q_%d/%d/d.txt" % (dimensionality_of_ambient_space , time_dimensions, i+1)
        D = np.loadtxt(file_name, usecols=range(n))
        
    D = np.sum(D,axis=0)
    sorted_D = np.argsort(D)
    search_second_leader = False
    for j in range(n):
        if (sorted_D[j] == 0) or (sorted_D[j] == n-1):
            if search_second_leader:
                rank_second_leader.append(j+1)
                continue
            else:
                search_second_leader = True
                rank_first_leader.append(j+1)
    rho5, pval5 = stats.spearmanr(C_sum[top_5],D[top_5])
    rho10, pval10 = stats.spearmanr(C_sum[top_10],D[top_10])
    rho5_list.append(rho5)
    rho10_list.append(rho10)

if evaluate_euclidean_representations:
    print("Euclidean space of dimensionality %d" % ambient_euclidean_dimensionality)
else:
    print("dimensionality of the ambient space = %d" % dimensionality_of_ambient_space)
    if time_dimensions == 1:
        print("hyperbolic case")
    elif time_dimensions == dimensionality_of_ambient_space :
        print("spherical case")
    else:
        print("ultrahyperbolic case with %d time dimensions" % time_dimensions)

ddofint = 1

print("rank of first leader")
print("mean = %f ----- std = %f" % (np.mean(rank_first_leader), np.std(rank_first_leader,ddof=ddofint)))

print("rank of second leader")
print("mean = %f ----- std = %f" % (np.mean(rank_second_leader), np.std(rank_second_leader,ddof=ddofint)))

print("top 5 Spearman's rho")
print("mean = %f ----- std = %f" % (np.mean(rho5_list), np.std(rho5_list,ddof=ddofint)))

print("top 10 Spearman's rho")
print("mean = %f ----- std = %f" % (np.mean(rho10_list), np.std(rho10_list,ddof=ddofint)))
