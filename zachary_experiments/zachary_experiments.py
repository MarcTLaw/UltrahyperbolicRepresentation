import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import random
import sys 
import shutil
import time

weighted_version = False

apply_standard_sgd = False
use_pseudoRiemannian_gradient = False

space_dimensions = 7
time_dimensions = 3

p = space_dimensions
q = time_dimensions
d = space_dimensions + time_dimensions

use_cuda = False #torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

beta_value = -1.0
nb_nodes = 34

step_size = 10**(-6)

zeros = torch.zeros([1], dtype=torch.long)
if use_cuda:
    print("training using cuda")
    zeros = zeros.to(device)   
else:
    print("training using cpu")

edge_list_78 = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1), (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4), (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2), (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1), (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23), (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8), (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8), (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23), (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13), (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22), (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30), (33, 31), (33, 32)]

weights_78 = [8, 10, 12, 6, 6, 6, 6, 6, 6, 4, 10, 4, 8, 8, 6, 4, 10, 2, 4, 6, 6, 6, 3, 6, 6, 10, 6, 6, 6, 6, 4, 2, 4, 4, 4, 4, 10, 4, 4, 8, 6, 4, 5, 8, 4, 6, 4, 4, 14, 4, 5, 7, 6, 6, 2, 6, 4, 10, 7, 6, 8, 7, 4, 6, 4, 8, 4, 2, 2, 3, 8, 4, 8, 4, 4, 6, 8, 10]
if weighted_version:
    ordered_c = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14]
else:
    ordered_c = [1]
    
edge_list = {}
for e in range(0,len(edge_list_78)):
    (j,i) = edge_list_78[e]
    if weighted_version:
        e_ij = weights_78[e]
    else:
        e_ij = 1
    edge_list[(i, j)] = e_ij
    edge_list[(j, i)] = e_ij



positive_indices = {}
total_negatives = 1
for i in range(0,nb_nodes-1):
    for j in range(i+1,nb_nodes):
        index_ij = i*nb_nodes+j
        if (j,i) in edge_list:
            k = edge_list[(j,i)]
            if k in positive_indices:
                positive_indices[k].append(index_ij)
            else:
                positive_indices[k] = [index_ij]
        else:
            if 0 in positive_indices:
                positive_indices[0].append(index_ij)
                total_negatives += 1
            else:
                positive_indices[0] = [index_ij]

           
            
class Model(nn.Module):
    def __init__(self, beta,q):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(nb_nodes, d)
        if q == 1:
            nn.init.uniform_(self.embedding.weight[:,0], 1, 1)
            nn.init.uniform_(self.embedding.weight[:,1:], -0.1,0.1)
        else:
            nn.init.uniform_(self.embedding.weight[:,0], 1, 1)
            nn.init.uniform_(self.embedding.weight[:,1:], -0.1,0.1)
        self.sqrtbeta = nn.Embedding(1, 1)
        self.sqrtbeta.weight.data.fill_(math.log(math.exp(abs(beta))-1))
        self.q = q
        self.d = d
        self.rad = nn.Embedding(1, 1)
        self.rad.weight.data.fill_(math.log(math.exp(0.5)-1))

        self.embedding.weight.requires_grad = True
        self.rad.weight.requires_grad = False
        self.sqrtbeta.weight.requires_grad =  False

    def map_on_pseudohyperboloid(self, beta_scaling = True):
        self.embedding.weight.data = self.embeddings(beta_scaling)
        
    def embeddings(self, beta_scaling = True, use_mapping = True):
        X = self.embedding.weight
        if self.q == 0:
            return X
        if not use_mapping:
            return X
        elif self.q == X.shape[0]:
            if beta_scaling:
                return self.psqrtbeta() * F.normalize(X)
            else:
                return F.normalize(X)
        elif self.q > 1:
            X0 = X[:,0].view(-1,1)
            Xtime = F.normalize(torch.cat((X0, X[:,1:self.q].view(-1,self.q-1)),1))
            Xspace = X[:,self.q:].div(self.psqrtbeta())
            spaceNorm = torch.sum(Xspace * Xspace, dim=1, keepdim=True)
            Xtime = torch.sqrt(spaceNorm.add(1.0)).expand_as(Xtime) * Xtime
            if beta_scaling:
                return self.psqrtbeta() * torch.cat((Xtime,Xspace),1)
            else:
                return torch.cat((Xtime,Xspace),1)
        elif self.q == 1:
            Xspace = X[:,self.q:].div(self.psqrtbeta())
            spaceNorm = torch.sum(Xspace * Xspace, dim=1, keepdim=True)
            Xtime = torch.sqrt((spaceNorm).add(1.0)).view(-1,1)
            if beta_scaling:
                return torch.cat((Xtime, Xspace),1)
            else:
                return self.psqrtbeta() * torch.cat((Xtime, Xspace),1)
        return 0
        

    def compute_similarity_matrix_without_mapping(self, beta_scaling = False, X = None, sample_list = None):
        if X is None:
            X = self.embedding.weight
        if sample_list is not None:
            X = X[sample_list,:]
        d = X.shape[1]    
        if self.q == 0:
            K = torch.matmul(X,X.t())        
        elif self.q == d:
            K = - torch.matmul(X,X.t())
        elif self.q >= 1:
            Xtime = X[:,0:self.q]
            Xspace = X[:,self.q:]
            K = torch.matmul(Xspace,Xspace.t()) - torch.matmul(Xtime,Xtime.t())
        if beta_scaling:
            return self.psqrtbeta() * self.psqrtbeta() * K
        return K

    def compute_similarity_matrix_with_mapping(self, X = None, sample_list = None):
        if X is None:
            X = self.embedding.weight
        if sample_list is not None: 
            X = X[sample_list,:]
        d = X.shape[1]      
        if self.q == 0:
            K = torch.matmul(X,X.t())        
        elif self.q == d:
            XX = F.normalize(X)
            K = - torch.matmul(XX,XX.t())  
        elif self.q > 1:
            Xtime = F.normalize(X[:,0:self.q])            
            Xspace = X[:,self.q:].div(self.psqrtbeta())
            spaceNorm = torch.sum(Xspace * Xspace, dim=1, keepdim=True)
            Xtime = torch.sqrt(spaceNorm.add(1.0)).expand_as(Xtime) * Xtime
            K = torch.matmul(Xspace,Xspace.t()) - torch.matmul(Xtime,Xtime.t())
        elif self.q == 1:
            Xspace = X[:,self.q:].div(self.psqrtbeta())
            spaceNorm = torch.sum(Xspace * Xspace, dim=1, keepdim=True)
            Xtime = torch.sqrt((spaceNorm).add(1.0))
            K = torch.matmul(Xspace,Xspace.t()) - torch.matmul(Xtime,Xtime.t())
        return K
            

    def compute_distance_matrix(self,K,beta_scaling = False):
        if beta_scaling:
            K = K / (self.psqrtbeta() * self.psqrtbeta())
        if self.q == 0:
            K = -K
        epsilon = 0.00001
        if self.q == d:
            euclidean_indices = (K < -1.0 + epsilon)
            K[euclidean_indices] = self.psqrtbeta() * torch.abs(2.0 * (1.0 + K[euclidean_indices]))
            K[~euclidean_indices] = self.psqrtbeta() * torch.acos(-K[~euclidean_indices])
        
        hyperbolic_indices = K < -1.0 - epsilon
        euclidean_indices = (K < -1.0 + epsilon) & (~hyperbolic_indices)
        positive_similarity = K >= 0.0
        spherical_indices = (~positive_similarity) & (~(K < -1.0 + epsilon)) 
        K[hyperbolic_indices] = self.psqrtbeta() * self.acosh(-K[hyperbolic_indices])
        K[euclidean_indices] = self.psqrtbeta() * torch.abs(2.0 * (1.0 + K[euclidean_indices]))
        K[positive_similarity] = self.psqrtbeta() * (math.pi/2 + K[positive_similarity])
        K[spherical_indices] = self.psqrtbeta() * torch.acos(-K[spherical_indices])
        return K
        



    def stopping_criterion(self, use_mapping = True):
        if use_mapping:
            K = self.compute_similarity_matrix_with_mapping()
        else:
            K = self.compute_similarity_matrix_without_mapping()
        n = K.shape[0]
        K = self.compute_distance_matrix(K,not use_mapping)
        cpt = 0
        negative_distance_list = []
        positive_distance_list = {}
        for i in range(0,n):
            for j in range(i+1,n):
                d = K[i,j]
                if (j,i) in edge_list:
                    if weighted_version:
                        k = weights_78[cpt]
                    else:
                        k = 1
                    cpt += 1
                    if k in positive_distance_list:
                        positive_distance_list[k].append(d)                        
                    else:
                        positive_distance_list[k] = []
                        positive_distance_list[k].append(d)                        
                    
                else:
                    negative_distance_list.append(d)

        for c in ordered_c:
            for positive in positive_distance_list[c]:
                for neg in negative_distance_list:
                    if positive > neg:
                        return False
            negative_distance_list = positive_distance_list[c]
            
        return True    


    def rescale_beta(self, X = None):
        if X is None:
            X = self.embedding.weight.data
        norm_X = X * X
        norm_Xtime = norm_X[:,0:self.q]            
        norm_Xspace = norm_X[:,self.q:]
        return X / torch.abs(torch.sum(norm_Xspace,dim=1, keepdim=True) - torch.sum(norm_Xtime,dim=1, keepdim=True) ).sqrt().expand_as(X) * self.psqrtbeta()        

    def perform_rescaling_beta(self, X = None):
        if X is None:
            X = self.embedding.weight
        norm_X = X * X
        norm_Xtime = norm_X[:,0:self.q]
        norm_Xspace = norm_X[:,self.q:]
        self.embedding.weight.data = X / torch.abs( torch.sum(norm_Xspace,dim=1, keepdim=True) - torch.sum(norm_Xtime,dim=1, keepdim=True) ).sqrt().expand_as(X) * self.psqrtbeta()

    def scalar_product(self, x, y):
        z = x * y
        return z[self.q:] - z[0:self.q]
    
    def change_metric(self, x):
        z = x.clone()
        z[:,0:self.q] = -z[:,0:self.q]
        return z

    def exponential_map(self, gradient, step_size, X = None):
        t = -step_size
        n = gradient.shape[0]
        d = gradient.shape[1]
        sqrt_beta = self.psqrtbeta()
        g = gradient.clone()
        if X is None:
            X = self.embedding.weight.clone()
        gX = g * X
        gXtime = gX[:,0:self.q]
        gXspace = gX[:,self.q:]
        gXsum = torch.sum(gX,dim=1, keepdim=True)
        gXsumq = torch.sum(gXspace,dim=1, keepdim=True) - torch.sum(gXtime,dim=1, keepdim=True)
        norm_X = X * X
        norm_Xtime = norm_X[:,0:self.q]
        norm_Xspace = norm_X[:,self.q:]
        sq_normX = torch.sum(norm_Xspace,dim=1, keepdim=True) - torch.sum(norm_Xtime,dim=1, keepdim=True)
        l2_sq_normX = torch.sum(norm_X,dim=1, keepdim=True)
        
        epsilon = 0.000000000001

        
        if use_pseudoRiemannian_gradient or (self.q == 1) or (self.q == d):
            apply_precondition = False
        else:
            apply_precondition = True
        if apply_precondition:
            #xi_preconditioned = g - self.change_metric(X) * (gXsum / sq_normX).expand_as(X) - X * ((torch.sum(gXspace,dim=1, keepdim=True) - torch.sum(gXtime,dim=1, keepdim=True)) / sq_normX).expand_as(X) + X * (l2_sq_normX * gXsum / (sq_normX * sq_normX)).expand_as(X)
            xi_preconditioned = g - self.change_metric(X) * (gXsum / sq_normX).expand_as(X) + X * ((l2_sq_normX * gXsum) / (sq_normX * sq_normX) - (gXsumq / sq_normX)).expand_as(X)
            sq_norm_xi_preconditioned_1 = xi_preconditioned * xi_preconditioned
            sq_norm_xitime_preconditioned = sq_norm_xi_preconditioned_1[:,0:self.q]
            sq_norm_xispace_preconditioned = sq_norm_xi_preconditioned_1[:,self.q:]
            sq_norm_xi_preconditioned = torch.sum(sq_norm_xispace_preconditioned,dim=1) - torch.sum(sq_norm_xitime_preconditioned,dim=1)
            norm_xi_preconditioned = torch.abs(sq_norm_xi_preconditioned).sqrt()
            sq_norm_xi = sq_norm_xi_preconditioned
            norm_xi = norm_xi_preconditioned
            xi = xi_preconditioned
        else:       
            xi_sr = self.change_metric(gradient) - X * (gXsum / sq_normX).expand_as(X)        
            sq_norm_xi_sr = xi_sr * xi_sr
            sq_norm_xitime_sr = sq_norm_xi_sr[:,0:self.q]
            sq_norm_xispace_sr = sq_norm_xi_sr[:,self.q:]
            sq_norm_xi_sr = torch.sum(sq_norm_xispace_sr,dim=1) - torch.sum(sq_norm_xitime_sr,dim=1)
            norm_xi_sr = torch.abs(sq_norm_xi_sr).sqrt()
            sq_norm_xi = sq_norm_xi_sr
            norm_xi = norm_xi_sr
            xi = xi_sr
            if (self.q == d):
                t = -t
        cpt = 0
        U = torch.empty(n,d)
        if use_cuda:
            U = U.to(device)

        time_like = sq_norm_xi < -epsilon
        space_like = sq_norm_xi > epsilon
        null_geodesic = (~space_like) & (~time_like)
        norm_xi = norm_xi.view(-1,1)
        U[time_like,:] = X[time_like,:] * torch.cos(t * norm_xi[time_like] / self.psqrtbeta()).repeat(1,d) + (self.psqrtbeta() * torch.sin(t * norm_xi[time_like] / self.psqrtbeta()) / norm_xi[time_like]).repeat(1,d) * xi[time_like,:]
        U[space_like,:] = X[space_like,:] * torch.cosh(t * norm_xi[space_like] / self.psqrtbeta()).repeat(1,d) + (self.psqrtbeta() * torch.sinh(t * norm_xi[space_like] / self.psqrtbeta()) / norm_xi[space_like]).repeat(1,d) * xi[space_like,:]
        U[null_geodesic,:] = X[null_geodesic,:] + t * xi[null_geodesic,:]

        if self.q == 1:
            U[:,0] = torch.abs(U[:,0])
        self.embedding.weight.data = self.rescale_beta(X = U)

    def loss_function(self, use_mapping = True, beta_scaling = True, similarity_matrix = None, sample_list = None):
        if similarity_matrix is None:
            if use_mapping:
                K = self.compute_similarity_matrix_with_mapping(sample_list = sample_list)
            else:
                K = self.compute_similarity_matrix_without_mapping(sample_list = sample_list)
        else:
                K = similarity_matrix
        total_loss = 0.0
        temperature_inv = 100.0 
        nb_positive = 0
        nb_negative = 0
        nb_satisfied = 0
        nb_constraints = 0
        celoss = nn.CrossEntropyLoss(reduction="sum")
        losses = None
        negative_distance_list = None
        positive_distance_list = {}
        cpt = 0
        n = K.shape[0]
        drawn_number = n
        K = (-self.compute_distance_matrix(K) * temperature_inv).view(-1)
        total_c = 0
        negative_distance_list = K[positive_indices[0]].view(1,-1)
        
        cptindx = 0
        nb_positives = 0
        for c in ordered_c:
            if c not in positive_indices:
                continue
            positives_c = K[positive_indices[c]]
            clength = positives_c.shape[0]
            total_loss += celoss(torch.cat((positives_c.view(-1,1),negative_distance_list.repeat(clength,1)),1),zeros.repeat(clength))
            negative_distance_list = torch.cat((negative_distance_list,positives_c.view(1,-1)),1)
        return total_loss

    def save_matrix(self, use_mapping = True):
        X = self.embeddings(use_mapping)
        n = X.shape[0]
        d = X.shape[1]
        file_name = "embedding_matrix_q_%d_d_%d.txt" % (self.q, self.d)
        f = open(file_name, "w")
        for i in range(0,n):
            for j in range(0,d):
                f.write("%17.12f " % X[i,j])
            f.write("\n")        
        f.close()
        shutil.copyfile(file_name, "x.txt")
        if use_mapping:
            K = self.compute_similarity_matrix_with_mapping()
        else:
            K = self.compute_similarity_matrix_without_mapping()
        K = self.compute_distance_matrix(K,not use_mapping)
        f = open("d.txt", "w")
        for i in range(0,n):
            for j in range(0,n):
                f.write("%17.12f " % K[i,j])
            f.write("\n")        
        f.close()


    def radius(self):
        return self.psqrtbeta() * F.softplus(self.rad.weight)

    def acosh(self,x):
        return torch.log(x+(x*x-1.0).sqrt())

    def psqrtbeta(self):
        return F.softplus(self.sqrtbeta.weight)
        
    def beta(self):
        return - (self.psqrtbeta() * self.psqrtbeta())



model = Model(math.sqrt(abs(beta_value)),q)
if use_cuda:
    model.to(device)
model.perform_rescaling_beta() 

if apply_standard_sgd:
    optimizer = optim.SGD(model.parameters(), lr=step_size, momentum=0.0)
else:
    optimizer = optim.SGD([ {'params': model.embedding.parameters(), 'lr': 0.0, 'momentum':0.0 }, {'params': model.sqrtbeta.parameters(), 'lr': 0.0000001, 'momentum':0.0}, {'params': model.rad.parameters(), 'lr': 0.0000001, 'momentum':0.0}], lr=0.000, momentum=0.0)
    model.perform_rescaling_beta()


f_loss = open("loss_values.txt","w")
iteration = 0
    
min_loss = float("inf")

optimize_exp_map = True

print("Data loaded. Starting training")
has_not_decreased = True
start_time = time.time()
while True:
    iteration += 1
    optimizer.zero_grad()
    loss = model.loss_function(use_mapping = apply_standard_sgd)
    if math.isnan(loss):
        print("nan loss")
        sys.exit()
    f_loss.write("%f\n" % loss)
    if not weighted_version:
        print("iteration %d : loss = %f" % (iteration, loss))
        if (iteration % 20):
            if model.stopping_criterion(apply_standard_sgd):
                model.save_matrix(apply_standard_sgd)
                break
    elif not(iteration % 20):
        print("iteration %d : loss = %f" % (iteration, loss))
        if loss < min_loss:
            min_loss = loss            
        else:            
            if iteration >= 10000:
                break
            continue
        model.save_matrix(apply_standard_sgd)
        print("files saved")
        if iteration >= 10000:
            break
        if model.stopping_criterion(apply_standard_sgd):
            break

    loss.backward()

    if has_not_decreased:
        if weighted_version:
            if loss < 400:
                has_not_decreased = False
                step_size = step_size * 0.2
        else:       
            if loss < 100:
                has_not_decreased = False
                step_size = step_size * 0.5
    if apply_standard_sgd:
        optimizer.step()
    else: 
        if optimize_exp_map:
            model.exponential_map(model.embedding.weight.grad, step_size)    
        else:
            optimizer.step()
    
f_loss.close()

print("End of training in %d seconds" % (time.time() - start_time))
