function [x_opt, fx_opt, iter] = DSimplex_eye(A, b, c)
% DSIMPLEX_EYE ���ö�ż�����η���LP����
%   max cx s.t. Ax=b x>=0
%   c:1*n������ A:m*n���� b:m*1������
%   x_optΪ���Ž⣬fx_optΪ���ź���ֵ��iterΪ��������

[m, n] = size(A);

% Ѱ�ҳ�ʼ�����н⣬��ʼ������
idx_B = find_eye(A);           % ��ʼ��������Ѱ�ҵ�λ����
idx_N = setdiff(1:n, idx_B);    % ��ʼ�ǻ�����
iter = 0;                       % ��������

while true % �������
    iter = iter + 1;
    B = A(idx_B);       % ��ʼ���л�
    x0 = zeros(n, 1);   % �ǻ�����Ϊ0
    x0(idx_B) = b;      % ��ʼ������
    cB = c(idx_B);      % ����cB
    Sigma = zeros(1,n); % ������������Ϊ0
    % ���������
    Sigma(idx_N) = c(idx_N) - cB * A(:,idx_N);   %����ǻ�����������c_j - z_j
    % �ж����Ž�
    if ~any(x0(idx_B) <= 0)  % ���м�����С�ڵ���0�����Դﵽ���Ž�
        if any(x0(idx_B) == 0)   % �ǻ�����������Ϊ0��������
            disp('�����');
        end
        x_opt = x0;
        fx_opt = -c * x_opt;     % �õ�Ŀ�꺯��ֵ
        return
    end
    % ȷ����������l
    [~, idx_l] = min(x0(idx_B));     % ѡ����С�ĸ�ֵ������
    % ȷ���������k
    Theta = Sigma ./ A(idx_l,:);
    if all(Theta <= 0)
       error('�޽�'); 
    end
    Theta(Theta <= 0) = Inf;     % ����С��0�����
    [~, idx_k] = min(Theta);
    % �û������xk�滻�������еĻ�������xl
    idx_B(idx_l) = idx_k;
    idx_N = setdiff(1:n, idx_B);% �µķǻ�����
    % ��˹��Ԫ��ȷ���µ�A��b
    A(:,idx_N) = A(:,idx_B) \ A(:,idx_N);
    b = A(:,idx_B) \ b;
    A(:,idx_B) = eye(m,m);
end

end

