function [xs,Bs,x_num]=BFS(A,b)
%BFS 给出LP问题的基本可行解及其基矩阵

xs = [];    %存储基本可行解(列向量)
Bs = {};    %存储可行基矩阵(cell)
x_num = 0;  %基本可行解个数

[m, n] = size(A);
if rank(A) < m
    error('不是LP问题');
end

% 计算基矩阵的最多个数
MaxB = factorial(n)/(factorial(m)*factorial(n-m));
colIdx = nchoosek(1:n,m);   %下标的全组合

% zero = zeros(n-m, 1);

for i = 1: MaxB
    tmpA = A(:, colIdx(i,:));
    if rank(tmpA) == m      % 得到基矩阵
        xb = inv(tmpA)*b;   % 得到基本解
        if xb >= 0          % 得到基本可行解
            x_num = x_num + 1;
            % x = [xb;zero];
            Bs{1, x_num} = tmpA;
            xs(colIdx(i,:), x_num) = xb;  %按对应位置重新排列
        end
    end
end
