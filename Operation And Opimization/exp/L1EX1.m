m=3; n=8; A=10*rand(m,n); I=eye(m,m);
randIndex = randperm(size(A,2));
A(:,randIndex(1:m)) = I;
disp(A)

c = 0;

for i = 1: n-m
   if A(:, i:i+2) == I
      disp(i) 
      c = c+1;
   end
end

if c == 0
    disp('Not Exist');
end