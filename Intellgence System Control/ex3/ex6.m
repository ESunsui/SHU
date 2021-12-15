% Membership function
clear all;
close all;
M = 3;
if M == 1
% O
    x = 0 : 1 : 200;
    y = trapmf(x, [50 70 200 200]);
    plot(x, y, 'k');
    xlabel('x'); ylabel('y');
elseif M == 2
% Y
    x = 0 : 1: 200;
    y = trapmf(x, [0 0 25 70]);
    plot(x, y, 'k ');
    xlabel('x'); ylabel('y');
elseif M == 3
% W
    x = 0: 1 : 200;
    y = trapmf(x, [0 0 25 35]);
    plot(x, y, 'k');
    xlabel('x'); ylabel('y');
elseif M == 4
% V
    x = 0 : 1 : 200;
    y = trapmf(x, [20 30 50 70]) ;
    plot(x, y, 'k') ;
    xlabel('x'); ylabel('y');
end