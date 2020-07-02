clear all;

evaluate_euclidean_representations = false
time_dimensions = 4
dimensionality_of_embedded_space = 7
euclidean_dimension = 6
lower_bound_score = [1, 10, 20];
rho_score = [];
if evaluate_euclidean_representations
    directory_name = strcat('nips_data/euclidean_',int2str(euclidean_dimension));
else
    directory_name = strcat('nips_data/d_', int2str(dimensionality_of_embedded_space), '_q_',int2str(time_dimensions));
end
load('C.mat');

S = C;
degree = sum(S,1);
u = unique(degree);

for directory=1
    X = load(strcat(directory_name,'/',int2str(directory),'/','x.txt'));    
    d = size(X,2);
    n = size(X,1);
    if evaluate_euclidean_representations
        K = X*X';
        Kdiag = diag(K);
        D = repmat(Kdiag',n,1) + repmat(Kdiag,1,n) - 2 * K;
    else
        J = eye(d);
        for i=1:time_dimensions
            J(i,i) = -1;
        end
        K = X * J * X';
        D = zeros(size(K));

        i1 = K < -1;
        if d == time_dimensions
            disp('spherical');
            i3 = ~i1;
            D(i3) = acos(-K(i3));            
        else
            i2 = K > 0;
            i3 = ~i1 & ~i2;            
            D(i2) = pi/2 + K(i2);
            D(i3) = acos(-K(i3));
        end
        D(i1) = acosh(-K(i1));
    end
    for i = 1:n
        D(i,i) = 0.0;
    end
    N = sum(D,1);
    for topauthors = lower_bound_score
        topp = degree >= topauthors; 
        rho_score = [rho_score, corr(N(topp)',-degree(topp)','Type','Spearman')];
    end
    
    recall_at_1 = 0;
    for i=1:n
        D(i,i) = inf;
        [min_i, index_min] = min(D(i,:));
        recall_at_1 = recall_at_1 + (S(i,index_min) > 0);
    end
end

rho_score
recall_at_1 = 100*recall_at_1 / n