clear all;

load('C.mat');
C = sparse(C);
[c1,c2,cv] = find(C);
c1 = c1 - 1;
c2 = c2 - 1;

fileID = fopen('C_matrix.txt','w');


for i=1:length(c1)
    a = c1(i);
    b = c2(i);
    v = cv(i);
    nbytes = fprintf(fileID,'%d %d %d\n',a,b,v);    
end

fclose(fileID);
