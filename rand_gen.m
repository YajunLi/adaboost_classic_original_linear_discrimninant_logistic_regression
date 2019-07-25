function num = rand_gen(dis,val)
% dis is the distribution, n by one or one by n like [3 6 8 1 9 ...]
% val is the range of values like [1 2 3 4 5 ...]

% dimension of the data
n = size(dis,1);
% standardize distribution
s=sum(dis);
dis_prime = dis/s;
cum_dis = cumsum(dis_prime);
aaa = rand(1);
ind = find(cum_dis<=aaa);
num = val(ind(end));









