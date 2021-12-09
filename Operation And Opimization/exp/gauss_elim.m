function [x] = gauss_elim(A, b)
if rank(A) ~= length(A)
    disp("Not a full rank matrix.");
    return;
end
n = length(A);
D = [A,b];
for i=1:n
    ts = i+1;
    while D(i,i) == 0       %满秩，因而置换
        D([i,ts], :) = D([ts, i], :);
        ts = ts + 1;
    end
    for j = 1 : n
        D(j, :) = D(j, :) / D(j, j);
        for k = 1 : n
            if k ~= j
                D(k, :) = D(k, :) - D(k, j) * D(j, :);
            end
        end
    end
end

x = D(:, n+1);
end

