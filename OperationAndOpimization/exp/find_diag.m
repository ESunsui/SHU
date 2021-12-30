function [outputArg] = find_diag(A)
%FIND_DIAG 返回单位矩阵位置
outputArg = [];
r = rank(A);
[m, n] = size(A);
if m > r
    error('ERR');
end

for i = 1 : m
    for j = 1 : n
        sfx = zeros(m, 1);
        sfx(i, 1) = 1;
        if A(:, j) == sfx
            outputArg(i) = j;
            break;
        end
    end
end

end

