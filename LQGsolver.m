function L = LQGsolver(A,B,Q,R,x0)
% Backwards recurrence for the optimal feedback gains

nStep = size(Q,3) - 1;
ns    = size(A,1);
nc    = size(B,2);

S  = Q(:,:,end); % initialize with the same dimension
L = zeros(nc,ns, nStep);
s = 0;

oXi = (B*B');         % Noise covariance matrix 

for k = nStep:-1:1 
    L(:,:,k) = (R(:,:,k)+B'*S*B)\B'*S*A;
    Sold = S;
    S = Q(:,:,k) + A'*Sold*(A-B*L(:,:,k));
    s = s + trace(Sold+oXi);  
end

end

