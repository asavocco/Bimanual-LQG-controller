clear all
close all
clc

%-------------------------------------------------------------------------%
% based on the papers:
%
% 1) "A very fast time scale of human motor adaptation: Within movement 
% adjustments of internal representations during reaching", Crevecoeur
% 2020.
%
% 2) "Optimal Task-Dependent Changes of Bimanual Feedback Control and 
% Adaptation", Diedrichsen 2007.

%% %---PARAMETERS---% %% 

m              = 2.5;  % [kg]
k              = 0.1;  % [Nsm^-1]
tau            = 0.1;  % [s]
delta          = 0.01; % [s]  
theta          = 15;   % [N/(m/s)] - coeff pert force (F = +-L*dy/dt)
alpha          = [1000 1000 20 20 0 0];% [PosX, PosY, VelX, VelY, Fx, Fy]
learning_rates = [.1 .1];% [right left]
coeffQ         = 1;      % increase or decrease Q matrix during trials
time           = 0.6;    % [s] - experiment 1 - reaching the target
stab           = 0.01;   % [s] - experiment 1 - stabilization 
nStep          = round((time+stab)/delta)-1;
N              = round(time/delta);


% Protocol parameters

right_perturbation = 'CCW';     % CCW, CW or BASELINE (no FFs)
left_perturbation  = 'BASELINE';% CCW, CW or BASELINE (no FFs)
numoftrials        = 15;        % number of trials 
catch_trials       = 0;         % number of catch trials


%% %---SYSTEM CREATION---% %%

% STATE VECTOR REPRESENTATION:
% (1-6)  RIGHT HAND x, y, dx, dy, fx and fy
% (7-12) LEFT HAND  x, y, dx, dy, fx and fy

xinit  = [.06 0 0 0 0 0 -.06 0 0 0 0 0]';    % [right left]
xfinal = [.06 .15 0 0 0 0 -.06 .15 0 0 0 0]';% [right left] 

A = [0 0 1 0 0 0; 0 0 0 1 0 0;...
	 0 0 -k/m 0 1/m 0;...
	 0 0 0 -k/m 0 1/m; 0 0 0 0 -1/tau 0;...
	 0 0 0 0 0 -1/tau];
A = blkdiag(A,A);
	 
B = zeros(6,2);
B(5,1) = 1/tau;
B(6,2) = 1/tau;
B = blkdiag(B,B);

% Discrete time representation
ns = size(A,1);
nc = size(B,2);
nf = size(xfinal,1);

Ad = eye(ns) + delta * A; % Adiscrete = 1 + delta_t*Acontinuous 
Bd = delta * B;           % Bdiscrete = delta_t*Bcontinuous 

% Expend matrixes for the final target state vector
A = [Ad,zeros(ns,nf);zeros(nf,ns),eye(nf)];
B = [Bd;zeros(nf,nc)];

% Estimated matrix 
A_hat = A;


%% %---COST FUNCTION---% %%

Q = zeros(2*ns,2*ns,nStep+1);
R = repmat(10^-5*eye(nc), 1, 1, nStep);% Cost parameter for control action

% Position
p = zeros(2*ns,2*ns);
p(:,1) = [0.5 zeros(1,5) 0.5 zeros(1,5) -0.5 zeros(1,5) -0.5 zeros(1,5)]';
p(:,2) = [0 0.5 zeros(1,4) 0 0.5 zeros(1,4) 0 -0.5 zeros(1,4) 0 -0.5 zeros(1,4)]';
p(:,[7, 13, 19]) = p(:,1).*[ones(2*ns,1) -ones(2*ns,2)];
p(:,[8, 14, 20]) = p(:,2).*[ones(2*ns,1) -ones(2*ns,2)];

% Velocity
vx   = [0;0;1;0;0;0];
temp = vx*vx';
Vx   = blkdiag(temp,temp,zeros(ns,ns));
vy   = [0;0;0;1;0;0];
temp = vy*vy';
Vy   = blkdiag(temp,temp,zeros(ns,ns));

% Sum        
C = p + Vx + Vy;

% Cost
beta  = [1000 1000 0 0 0 0];
alpha = [alpha alpha beta beta];

for j = N+1:nStep+1
    
    for i = 1:2*ns
        
        temp = zeros(2*ns,2*ns);
        temp(:,i) = C(:,i);
        Q(:,:,j) = Q(:,:,end) + alpha(i)*temp;
        
    end
    
end

for t = 1:N
            Q(:,:,t) = (t/N)^3*Q(:,:,end);
end
%% %---LQG SOLVER---% %%

L = LQGsolver(A, B, Q, R, xinit);

%% %---SIMULATION---% %%

% Add perturbation (curl FFs) to the matrix A
switch right_perturbation
    case 'CCW'
        A(3,4) = -delta*(theta/m);
        A(4,3) = delta*(theta/m);
    case 'CW'
        A(3,4) = delta*(theta/m);
        A(4,3) = -delta*(theta/m);
    case 'BASELINE'
        A(3,4) = 0;
        A(4,3) = 0;
    otherwise
        error('The perturbation choice is incorrect !')
end

switch left_perturbation
    case 'CCW'
        A(9,10) = -delta*(theta/m);
        A(10,9) = delta*(theta/m);
    case 'CW'
        A(9,10) = delta*(theta/m);
        A(10,9) = -delta*(theta/m);
    case 'BASELINE'
        A(9,10) = 0;
        A(10,9) = 0;
    otherwise
        error('The perturbation choice is incorrect !')
end

% Initialization of simulation vectors
x             = zeros(2*ns,nStep+1,numoftrials);  % Initialize the state
x(1:ns,1,:)   = repmat(xinit, 1, 1,numoftrials);  % Initialize the estimated state
xhat          = x;                                % Initialize the state estiamte
control       = zeros(nc,nStep,numoftrials);      % Initialize control
avControl     = zeros(nc,nStep);                  % Average Control variable


% Random indexes for catch trials
catch_trials_idx = [];

if catch_trials ~= 0
    while length(catch_trials_idx) ~= catch_trials
        random = randi(numoftrials, 1, 1); 
        catch_trials_idx = [catch_trials_idx random];
        catch_trials_idx = unique(catch_trials_idx);
    end
end

for p = 1:numoftrials
    
    x(ns+1:end,1,p)    = xfinal;
    xhat(ns+1:end,1,p) = xfinal;
    Q                  = Q;% if coeffQ ~= 1, reset the Q matrix at each trial
    
    for k = 1:nStep-1
        
        A_old = A;% needed for catch trials
        
        if (~isempty(catch_trials_idx)) & (k == catch_trials_idx)
            A(3,4)  = 0;
            A(9,10) = 0;
            A(4,3)  = 0;
            A(10,9) = 0;
        end

        motorNoise   = mvnrnd(zeros(2*ns,1),(B*B'))'; % motor noise

        % Computation control vector and update optimal feedback gains
        u = -L(:,:,1)*x(:,k,p); % control variable
        Q = coeffQ*Q;           %Increase or decrease Q matrix
        L = LQGsolver(A_hat,B,Q(:,:,k+1:end),R(:,:,k+1:nStep),xhat(:,k,p));
        
        % Computation next state and next estimated state 
        xhat(:,k+1,p) = A_hat*x(:,k,p) + B*u;          % State Estimate
        x(:,k+1,p)    = A*x(:,k,p) + B*u + motorNoise; % dynamics

        % Update the A matrix
        
        eps           = x(1:ns/2,k+1,p)-xhat(1:ns/2,k+1,p);
        
        theta_up_R    = A_hat(3,4);
        dzhat_dL      = zeros(1,ns/2);
        dzhat_dL(1,3) = xhat(4,k+1,p);
        theta_up_R    = theta_up_R + learning_rates(1)*dzhat_dL*eps;
        A_hat(3,4)    = theta_up_R;
        
        theta_up_R    = A_hat(4,3);
        dzhat_dL      = zeros(1,ns/2);
        dzhat_dL(1,4) = xhat(3,k+1,p);
        theta_up_R    = theta_up_R + learning_rates(1)*dzhat_dL*eps;
        A_hat(4,3)    = theta_up_R;

        eps           = x(ns/2+1:ns,k+1,p)-xhat(ns/2+1:ns,k+1,p);
        
        theta_up_L    = A_hat(9,10);
        dzhat_dL      = zeros(1,ns/2);
        dzhat_dL(1,3) = xhat(10,k+1,p);
        theta_up_L    = theta_up_L + learning_rates(2)*dzhat_dL*eps;
        A_hat(9,10)   = theta_up_L;
        
        theta_up_L    = A_hat(10,9);
        dzhat_dL      = zeros(1,ns/2);
        dzhat_dL(1,4) = xhat(9,k+1,p);
        theta_up_L    = theta_up_L + learning_rates(2)*dzhat_dL*eps;
        A_hat(10,9)   = theta_up_L;

        control(:,k,p) = u;
        A = A_old;
end
    
    avControl = avControl + control(:,:,p)/numoftrials;
    
end

%% %---GRAPHS---% %%

x = x(:,1:N,:);

for trial = 1:numoftrials
    
    figure(1)
    
    % Position
    subplot(2,2,[1,2])
    midx = 0.5*(x(1,1:N,trial)+x(7,1:N,trial));
    midy = 0.5*(x(2,1:N,trial)+x(8,1:N,trial));
    plot(x(1,1:N,trial), x(2,1:N,trial)), hold on;
    plot(x(7,1:N,trial), x(8,1:N,trial));
    plot(midx, midy);
    plot(0,0,'ro','LineWidth',2);
    plot(0,.15,'ro','MarkerSize',10,'LineWidth',2);
    plot(0.06,0,'ro','LineWidth',2);
    plot(0.06,.15,'ro','MarkerSize',10,'LineWidth',2);
    plot(-0.06,0,'ro','LineWidth',2);
    plot(-0.06,.15,'ro','MarkerSize',10,'LineWidth',2);
    xlabel('x-coord [m]'); ylabel('y-coord [m]'); title(['LQG model - one cursor - trajectories'],'FontSize',14);
    axis([-(max(x(1,:,1)) + 0.04) (max(x(1,:,1)) + 0.04)  -0.01 0.16])

    % Control
    subplot(2,2,4)
    plot([.01:.01:(nStep)*.01],control(1,:,trial)), hold on;
    plot([.01:.01:(nStep)*.01],avControl(1,:),'k','Linewidth',2)
    xlabel('Time [s]'); ylabel('Control [Nm]'); title('Control Vector - Right','FontSize',14);
    %axis square

    subplot(2,2,3)
    plot([.01:.01:(nStep)*.01],control(3,:,trial)), hold on;
    plot([.01:.01:(nStep)*.01],avControl(3,:),'k','Linewidth',2)
    xlabel('Time [s]'); ylabel('Control [Nm]'); title('Control Vector - Left','FontSize',14);
    %axis square

    %input(' ');    

end

% Velocity profiles

figure(2)
subplot(131)
plot([.01:.01:(nStep)*.01], x(3,:,end), 'b');hold on;
plot([.01:.01:(nStep)*.01], x(9,:,end), 'r');hold off;
xlabel('Time [s]');
ylabel('X Velocity [m/s]');
legend('right', 'left');

subplot(132)
plot([.01:.01:(nStep)*.01], x(4,:,end), 'b');hold on;
plot([.01:.01:(nStep)*.01], x(10,:,end), 'r');hold off;
xlabel('Time [s]');
ylabel('Y Velocity [m/s]');
legend('right', 'left');

subplot(133)
plot([.01:.01:(nStep)*.01], sqrt(x(3,:,end).^2+x(4,:,end).^2), 'b');hold on;
plot([.01:.01:(nStep)*.01], sqrt(x(9,:,end).^2+x(10,:,end).^2), 'r');hold off;
xlabel('Time [s]');
ylabel('Velocity [m/s]');
legend('right', 'left');

% Force profiles

figure(3)
subplot(131)
plot([.01:.01:(nStep)*.01], x(5,:,end), 'b');hold on;
plot([.01:.01:(nStep)*.01], x(11,:,end), 'r');hold off;
xlabel('Time [s]');
ylabel('X Force [N]');
legend('right', 'left');

subplot(132)
plot([.01:.01:(nStep)*.01], x(6,:,end), 'b');hold on;
plot([.01:.01:(nStep)*.01], x(12,:,end), 'r');hold off;
xlabel('Time [s]');
ylabel('Y Force [N]');
legend('right', 'left');

subplot(133)
plot([.01:.01:(nStep)*.01], sqrt(x(5,:,end).^2+x(11,:,end).^2), 'b');hold on;
plot([.01:.01:(nStep)*.01], sqrt(x(6,:,end).^2+x(12,:,end).^2), 'r');hold off;
xlabel('Time [s]');
ylabel('Force [N]');
legend('right', 'left');

