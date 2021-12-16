function [x_opt, fx_opt, iter] = Simplex_eye(A, b, c)
% SIMPLEX_EYE ���õ����η���LP����
%   max cx s.t. Ax=b x>=0
%   c:1*n������ A:m*n���� b:m*1������
%   x_optΪ���Ž⣬fx_optΪ���ź���ֵ��iterΪ��������

[m, n] = size(A);

% Ѱ�ҳ�ʼ�����н�
idx_B = find_diag(A);           % ��ʼ��������Ѱ�ҵ�λ����
idx_N = setdiff(1:n, idx_B);    % ��ʼ�ǻ�����
iter = 0;                       % ��������

while true % �������
    iter = iter + 1;
    B = A(idx_B);       % ��ʼ���л�
    x0 = zeros(n, 1);   % �ǻ�����Ϊ0
    x0(idx_B) = b;      % ��ʼ�����н�
    cB = c(idx_B);      % ����cB
    Sigma = zeros(1,n); % ������������Ϊ0
    % ���������
    Sigma(idx_N) = c(idx_N) - cB * A(:,idx_N);   %����ǻ�����������c_j - z_j
    % �ж����Ž�
    if ~any(Sigma > 0)  % ���м�����С�ڵ���0���ﵽ���Ž�
        x_opt = x0;
        fx_opt = c * x_opt;
        return
    end
    % ȷ���������k
    [~, idx_k] = max(Sigma);    % ѡ����������,ȷ���������k
    % �ж��޽��
    if all(A(:,idx_k) < 0)
       error('�޽��'); 
    end
    % ȷ����������l
    Theta = b ./ A(:,idx_k);
    Theta(Theta < 0) = Inf;     % ����С��0�����
    [~, fidx_l] = min(Theta);
    % �û������xk�滻�������еĻ�������xl
    l = idx_B(fidx_l);          % ����������������ʵ��������Ҫת��
    idx_B(idx_B == l) = idx_k;  % �������룬�µĻ�����
    idx_N = setdiff(1:n, idx_B);% �µķǻ�����
    % ȷ���µ�A��b
    A(:,idx_N) = A(:,idx_B) \ A(:,idx_N);
    b = A(:,idx_B) \ b;
    A(:,idx_B) = eye(m,m);
end

end

