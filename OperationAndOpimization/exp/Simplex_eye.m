function [x_opt, fx_opt, iter] = Simplex_eye(A, b, c)
% SIMPLEX_EYE 利用单纯形法解LP问题
%   max cx s.t. Ax=b x>=0
%   c:1*n行向量 A:m*n矩阵 b:m*1列向量
%   x_opt为最优解，fx_opt为最优函数值，iter为迭代次数

[m, n] = size(A);

% 寻找初始基可行解，初始化变量
idx_B = find_eye(A);           % 初始基变量，寻找单位矩阵
idx_N = setdiff(1:n, idx_B);    % 初始非基变量
iter = 0;                       % 迭代次数

while true % 迭代求解
    iter = iter + 1;
    B = A(idx_B);       % 初始可行基
    x0 = zeros(n, 1);   % 非基变量为0
    x0(idx_B) = b;      % 初始基变量
    cB = c(idx_B);      % 计算cB
    Sigma = zeros(1,n); % 基变量检验数为0
    % 计算检验数
    Sigma(idx_N) = c(idx_N) - cB * A(:,idx_N);   %计算非基变量检验数c_j - z_j
    % 判断最优解
    if ~any(Sigma > 0)  % 所有检验数小于等于0，可以达到最优解
        if any(Sigma(idx_N) == 0)   % 非基变量检验数为0，无穷多解
            disp('无穷解');
        end
        x_opt = x0;
        fx_opt = c * x_opt;     % 得到目标函数值
        return
    end
    % 确定换入变量k
    [~, idx_k] = max(Sigma);     % 选出最大检验数,确定入基变量k的索引
    % 判断无界解
    if all(A(:,idx_k) <= 0)      % 基变量对应系数小于0，无界解
       error('无界解'); 
    end
    % 确定换出变量l
    Theta = b ./ A(:,idx_k);
    Theta(Theta < 0) = Inf;     % 处理小于0的情况
    [~, fidx_l] = min(Theta);   % 选出大于0的最小Theta，获得换出变量索引
    % 用换入变量xk替换基变量中的换出变量xl
    idx_B(fidx_l) = idx_k;
    idx_N = setdiff(1:n, idx_B);% 新的非基变量
    % 高斯消元法确定新的A与b
    A(:,idx_N) = A(:,idx_B) \ A(:,idx_N);
    b = A(:,idx_B) \ b;
    A(:,idx_B) = eye(m,m);
end

end

