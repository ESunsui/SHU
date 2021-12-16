function [xs,Bs,x_num]=BFS(A,b)
%BFS ����LP����Ļ������н⼰�������

xs = [];    %�洢�������н�(������)
Bs = {};    %�洢���л�����(cell)
x_num = 0;  %�������н����

[m, n] = size(A);
if rank(A) < m
    error('����LP����');
end

% ����������������
MaxB = factorial(n)/(factorial(m)*factorial(n-m));
colIdx = nchoosek(1:n,m);   %�±��ȫ���

% zero = zeros(n-m, 1);

for i = 1: MaxB
    tmpA = A(:, colIdx(i,:)); 
    if rank(tmpA) == m      % �õ�������
        xb = inv(tmpA)*b;   % �õ�������
        if xb >= 0          % �õ��������н�
            x_num = x_num + 1;
            % x = [xb;zero];
            Bs{1, x_num} = tmpA;
            xs(colIdx(i,:), x_num) = xb;  %����Ӧλ����������
        end
    end
end


