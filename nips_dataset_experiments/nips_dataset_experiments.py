import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import random
import sys 
import shutil
import time

weighted_version = True

apply_standard_sgd = False
use_pseudoRiemannian_gradient = False
negative_batch_size = 42000

space_dimensions = 2
time_dimensions = 3

p = space_dimensions
q = time_dimensions
d = space_dimensions + time_dimensions
if d <= 6:
    nb_max_iterations = 10000
else:
    nb_max_iterations = 25000


beta_value = -1.0
use_cuda = torch.cuda.is_available()

author_numbers = 2715
device = torch.device("cuda" if use_cuda else "cpu")
zeros = torch.zeros([1], dtype=torch.long)
if use_cuda:
    print("use cuda")
    zeros = zeros.to(device)   
else:
    print("use cpu")

f = open("nips_edges.txt","r")
edge_list = {}
for l in f:
    ee = l.split(" ")
    i = (int(ee[0]))-1
    j = (int(ee[1]))-1
    if weighted_version:
        e_ij = int(ee[2])
    else:
        e_ij = 1
    edge_list[(i, j)] = e_ij
    edge_list[(j, i)] = e_ij
f.close()

ordered_c = [1,2,3,4,5,6,7,8,9]

positive_indices = {}
total_negatives = 1
for i in range(0,author_numbers-1):
    for j in range(i+1,author_numbers):
        index_ij = i*author_numbers+j
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


                
                
negatives = range(0, total_negatives)   

            
class Model(nn.Module):
    def __init__(self, beta,q):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(author_numbers, d)
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

        self.euclidean_space = False

    def map_on_pseudohyperboloid(self, beta_scaling = True):
        self.embedding.weight.data = self.embeddings(beta_scaling)

    def print(self):
        print(self.embedding.weight)
        
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
        

    def compute_similarity_matrix_without_mapping(self, beta_scaling = False, X = None):
        if X is None:
            X = self.embedding.weight
        d = X.shape[1]
        if self.euclidean_space:
            K = torch.matmul(X,X.t())        
        elif self.q == 0:
            K = torch.matmul(X,X.t())        
        elif self.q == d:
            K = -torch.matmul(X,X.t())
        elif self.q >= 1:
            Xtime = X[:,0:self.q]
            Xspace = X[:,self.q:]
            K = torch.matmul(Xspace,Xspace.t()) - torch.matmul(Xtime,Xtime.t())
        if beta_scaling:
            return self.psqrtbeta() * self.psqrtbeta() * K
        return K

    def compute_similarity_matrix_with_mapping(self, X = None):
        if X is None:
            X = self.embedding.weight
        d = X.shape[1]
        if self.euclidean_space:
            K = torch.matmul(X,X.t())        
        elif self.q == 0:
            K = torch.matmul(X,X.t())        
        elif self.q == d:
            XX = F.normalize(X)
            K = -torch.matmul(XX,XX.t())  
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
        return False


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
            xi_preconditioned = g - self.change_metric(X) * (gXsum / sq_normX).expand_as(X) - X * ((torch.sum(gXspace,dim=1, keepdim=True) - torch.sum(gXtime,dim=1, keepdim=True)) / sq_normX).expand_as(X) + X * (l2_sq_normX * gXsum / (sq_normX * sq_normX)).expand_as(X)
            #xi_preconditioned = g - self.change_metric(X) * (gXsum / sq_normX).expand_as(X) + X * ((l2_sq_normX * gXsum / (sq_normX * sq_normX)) - (torch.sum(gXspace,dim=1, keepdim=True) - torch.sum(gXtime,dim=1, keepdim=True)) / sq_normX).expand_as(X)
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
            if self.q == d:
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

        #for i in range(0,n):
        #    if sq_norm_xi[i] < -epsilon:
        #        U[i,:] = X[i,:] * torch.cos(t * norm_xi[i] / self.psqrtbeta()) + self.psqrtbeta() * torch.sin(t * norm_xi[i] / self.psqrtbeta()) * (xi[i,:] / norm_xi[i])
        #    elif sq_norm_xi[i] > epsilon:
        #        U[i,:] = X[i,:] * torch.cosh(t * norm_xi[i] / self.psqrtbeta()) + self.psqrtbeta() * torch.sinh(t * norm_xi[i] / self.psqrtbeta()) * (xi[i,:] / norm_xi[i])
        #    else:
        #        U[i,:] = X[i,:] + t * xi[i,:]
        #    if q == 1:
        #        U[i,0] = torch.abs(U[i,0])
        self.embedding.weight.data = self.rescale_beta(X = U)

    def loss_function(self, use_mapping = True, beta_scaling = True, similarity_matrix = None):
        if similarity_matrix is None:
            if use_mapping:
                K = self.compute_similarity_matrix_with_mapping()
            else:
                K = self.compute_similarity_matrix_without_mapping()
        else:
                K = similarity_matrix

        total_loss = 0.0
        temperature_inv = 10000.0
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
        negative_distance_list = K[positive_indices[0]]
        weaker_positive = None
        
        cptindx = 0
        nb_positives = 0
        for c in ordered_c:
            if c not in positive_indices:
                continue
            positives_c = K[positive_indices[c]]
            clength = positives_c.shape[0]
            neg_indices = torch.randint(total_negatives, (clength *negative_batch_size,))
            if weaker_positive is None:
                total_loss += celoss(torch.cat((positives_c.view(-1,1),negative_distance_list[neg_indices].view(clength,-1)),1),zeros.repeat(clength))
            else:
                total_loss += celoss(torch.cat((positives_c.view(-1,1),weaker_positive.repeat(clength,1).view(clength,-1),negative_distance_list[neg_indices].view(clength,-1)),1),zeros.repeat(clength))

            if weaker_positive is None:
                weaker_positive = positives_c.view(1,-1)
            else:
                weaker_positive = torch.cat((weaker_positive,positives_c.view(1,-1)),1)
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
    optimizer = optim.SGD(model.parameters(), lr=0.00000001, momentum=0.0)
else:
    optimizer = optim.SGD([ {'params': model.embedding.parameters(), 'lr': 0.0, 'momentum':0.0 }, {'params': model.sqrtbeta.parameters(), 'lr': 0.0000001, 'momentum':0.0}, {'params': model.rad.parameters(), 'lr': 0.0000001, 'momentum':0.0}], lr=0.000, momentum=0.0)
    model.perform_rescaling_beta()

f_loss = open("loss_values.txt","w")
iteration = 0

min_loss = float("inf")

optimize_exp_map = True
step_size = 10**(-8)
have_not_decreased_step_size = True

print("starting training")
start_time = time.time()
while True:
    iteration += 1
    optimizer.zero_grad()
    loss = model.loss_function(use_mapping = apply_standard_sgd,)
    if math.isnan(loss):
        print("nan loss")
        sys.exit()
    print("iteration %d : loss = %f" % (iteration, loss))
    f_loss.write("%f\n" % loss)
    if not(iteration % 20):
        if loss < min_loss:
            min_loss = loss
        else:
            continue
        model.save_matrix(apply_standard_sgd)
        print("files saved")
    if loss < 50000:
        if have_not_decreased_step_size:
            have_not_decreased_step_size = False
            step_size = 0.5 * step_size
            print("step size decreased")
    if iteration >= nb_max_iterations:
        break

    loss.backward()

    if apply_standard_sgd:
        optimizer.step()
    else: 
        model.exponential_map(model.embedding.weight.grad, step_size)            
        
f_loss.close()
print("End of training in %d seconds" % (time.time() - start_time))
