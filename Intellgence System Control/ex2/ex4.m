%Expert PID Controller
clear all��
clc;
close all;
ts=0.001; %����ʱ��
sys=tf(5.235e005,[1,87.35,1.047e004,0]); %���ض��󴫵ݺ���
dsys=c2d(sys,ts,'z');%z�任����ɢ��
[num,den] = tfdata(dsys,'v');%�����ɢ��ģ�͵ķ��ӷ�ĸ����ʽϵ��
u_1=0; u_2=0; u_3=0;%���Ʊ���u��k-1����u��k-2����u��k-3����ֵ
y_1=0; y_2=0; y_3=0;%�������y(k-1����y��k-2����y��k-3����ֵ
x=[0,0,0]';%������������΢����
x2_1=0;
kp=0.6; ki=0.03 ;kd=0.01 ;
%���������ó��淽��������PID����ֵ
error_1=0; %���e��k-1)��ʼֵ
for k=1:1:500
    time(k)=k*ts;%��k����ʱ��
    r(k)=1.0; %����Ϊ��λ��Ծ�ź�
    u(k)=kp*x(1)+kd*x(2)+ki*x(3);%PID Controller
    %Expert control rule
    if abs(x(1))>0.8 %Rule1:����ر��ʱ�������ƣ��ֶζ�ֵ���ƣ�
        u(k)=0.45;
    elseif abs(x(1))>0.40
        u(k)=0.40;
    elseif abs(x(1))>0.20
        u(k)=0.12;
    elseif abs(x(1))>0.01
        u(k)=0.10;
    end
    if x(1)*x(2)>0 || (x(2)==0)%Rule2��������ֵ����ʱ
        if abs(x(1))>=0.05
            u(k)=u_1+2*kp*x(1);%���ϴ�ʱ���ƹ���
        else
            u(k)=u_1+0.4*kp*x(1);%����Сʱ���ƹ���
        end
    end
    if(x(1)*x(2)<0 & x(2)*x2_1>0) | (x(1)==0) %Rule3:������ֵ��Сʱ�����ֲ��� 
        u(k)=u(k); 
    end
    if x(1)*x(2)<0 & x(2)*x2_1<0 %Rule4:���ڼ�ֵ״̬ʱ 
        if abs(x(1))>=0.05 
            u(k)=u_1+2*kp*error_1;%��ֵ�ϴ�ʱ 
        else
            u(k)=u_1+0.6*kp*error_1;%��ֵ��Сʱ 
        end
    end
    if abs(x(1))<=0.001 %Rule5:���С��Ҫ��ʱ������Pl���Ƶ�����̬��� 
        u(k)=0.5*x(1)+0.010*x(3); 
    end
    %Restricting the output of controller 
    if u(k)>=10 
        u(k)=10; 
    end
    if u(k)<=-10 
        u(k)=-10; 
    end
    %����Ϊ��ɢ���ı��ض���ģ�� 
    y(k)=-den(2)*y_1-den(3)*y_2-den(4)*y_3+num(1)*u(k)+num(2)*u_1+num(3)*u_2+num(4)*u_3; 
    error(k)=r(k)-y(k); 
    %----------Return of parameters----------% 
    u_3=u_2;u_2=u_1;u_1=u(k);%���¿��Ʊ���
    y_3=y_2;y_2=y_1;y_1=y(k);%����������� 
    x(1)=error(k);%��������x��1��=e��k�� 
    x2_1=x(2);%�������仯�� 
    x(2)=(error(k)-error_1)/ts;%΢�ַ���x��2)=e(k)-e(k-1) 
    x(3)=x(3)+error(k)*ts;%���ַ�������3��=Ze(k) 
    error_1=error(k);%�������e��k-1):e��k-1)=e(k) 
end
figure(1); 
plot(time,r,'b',time,y,'r');
xlabel('time(s)'); ylabel('r,y');
figure(2);
plot(time,r-y,'r');
xlabel('time(s)');ylabel('error')
