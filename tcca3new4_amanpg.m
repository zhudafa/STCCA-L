function [ Res] = tcca3new4_amanpg(A,B,C,S,num,option,maxiter,sp_type)
%min -Tr(X'A'*B*Y)+ tau1*||X||_1 + tau2 ||Y||_1,  ------ sp_type = 'l1'
% or min -Tr(X'A'*B*Y)+ tau1*||X||_{2,1} +tau2 ||Y||_{2,1},  -- sp_type='l21'

%  s.t. X'*A'*A*X=I_n,  Y'*B'*B*Y=In
%% manpg alternating
%parameters

tic;
n=option.n;  % number of column
p=option.p;  %dim A
q=option.q;  %dim B
L=option.L;  %dim C
m = size(A,1);%sample
tau1 = option.b1*sqrt((n+log(p))/m);
tau2 = option.b2*sqrt((n+log(q))/m);
tau3 = option.b3*sqrt((n+log(L))/m);
%fprintf('A-ManPG parameter tau1: %2.3f; \n', tau1);
%fprintf('A-ManPG parameter tau2: %2.3f; \n', tau2);
epsilon =option.epsilon;
maxiter =option.maxiter;
F=zeros(maxiter,1);
XRes1=zeros(maxiter,1);
YRes1=zeros(maxiter,1);
ZRes1=zeros(maxiter,1);
tol = option.tol;
inner_tol =  option.inner_tol;
if strcmp(sp_type,'l1') 
    h1=@(X) tau1*sum(sum(abs(X)));
    h2=@(X) tau2*sum(sum(abs(X)));
    h3=@(X) tau3*sum(sum(abs(X)));
    prox_func = @(b,lambda,r) proximal_l1(b,lambda,r);
end
if  strcmp(sp_type,'l21')
    h1=@(X) tau1*sum(vecnorm(X,2,2));
    h2=@(X) tau2*sum(vecnorm(X,2,2));
    h3=@(X) tau3*sum(vecnorm(X,2,2));
    prox_func = @(b,lambda,r) proximal_l21(b,lambda,r); 
end
inner_flag1=0;
%setduplicat_pduplicat(n);
Dn = sparse(DuplicationM(n));
pDn = (Dn'*Dn)\Dn';
t_min = 1e-4; % minimum stepsize
views={A,B,C};
   n_samples = size(views{1},1);     %cell琛屾暟
    n_views = length(views);          %cell鍒楁暟
    %Center each view
    for i=1:n_views
        views{i} = tensor(views{i} - repmat(mean(views{i} ), n_samples,1));   
    end
     %Calculate variances
    variances = cell(size(views));
    for i=1:n_views
        variances{i} =  (double(views{i})'*double(views{i}))/n_samples;
        variances{i} = variances{i} +  epsilon*ones(size(variances{i}));
    end
 %Calculate covariances
    covariances = [];
    for i=1:n_samples
       outer_product = views{1}(i,:);
       for j=2:length(views)
           outer_product = ttt(outer_product,views{j}(i,:));
       end
       if isempty(covariances)
           covariances = outer_product;
       else
           covariances = covariances+outer_product;
       end
    end   
    covariances = covariances / n_samples;
    
    CC = covariances;
    for i=1:length(variances)
        CC = ttm( CC,pinv(variances{i})^1/2,i);
    end 
%% set type
gamma = 1e-4/(m-1);
if svds(A,1,'smallest') < 1e-4
    M1 = (1-gamma)*variances{1} + gamma*eye(p); pdA = 0;
else
    M1 = variances{1}; pdA = 1;
end
if svds(B,1,'smallest') < 1e-4
    M2 = (1-gamma)*variances{2} + gamma*eye(q); pdB = 0;
else
    M2 = variances{2};  pdB =1;
end
if svds(C,1,'smallest') < 1e-4
    M3 = (1-gamma)*variances{3} + gamma*eye(q); pdC = 0;
else
    M3 = variances{3};  pdC =1;
end
if m > p/2;    typeA = 1;   else;  typeA=0;  end  % p^2 n
if m > q/2;    typeB = 1;   else;  typeB = 0;  end
if m > L/2;    typeC = 1;   else;  typeC = 0;  end
%% initial point
% X = option.X0;  Y= option.Y0;
X0 = option.X0; Y0 = option.Y0; Z0 = option.Z0;
X0 = X0(:,1:n);
Y0 = Y0(:,1:n);
Z0 = Z0(:,1:n);

X0 = retraction(X0,M1,A,typeA); 
Y0 = retraction(Y0,M2,B,typeB);
Z0 = retraction(Z0,M3,C,typeC);
%time_init = toc;
X = X0;   Y = Y0;  Z = Z0;
X2 = X;   Y2 = Y;  Z2=Z;
% fprintf('------------------result of initial_point :--------------------\n');
%
% [uhat, ~,~] = svd(X,0);  [vhat,~,~] = svd(Y,0);
% Init_lossu = norm(uhat * uhat'  - option.u_n * option.u_n', 'fro')^2;
% Init_lossv = norm(vhat * vhat'  - option.v_n * option.v_n', 'fro')^2;
% [~,~,Init_rho]  = canoncorr(option.Xtest * uhat, option.Ytest * vhat);
%
% fprintf('Canonical correlations on test data:  rho = %2.3f;  \n',Init_rho);
% fprintf('Projection U error: %2.3f; \n',Init_lossu);
% fprintf('Projection V error: %2.3f. \n',Init_lossv);
% fprintf('time: %.3f \n', time_init);
%%
la=diag(sum(S{1},2))-S{1}+eye(num)*eps;
lb=diag(sum(S{2},2))-S{2}+eye(num)*eps;
lc=diag(sum(S{3},2))-S{3}+eye(num)*eps;
AtlaA=A'*la*A;
BtlbB=B'*lb*B;
CtlcC=C'*lc*C;


if typeA ==1;    MX = M1*X;
else
    if pdA ==0
        MX = (1-gamma)* A'*(A*X)/(m-1) + gamma*X;
    else ; MX = A'*(A*X)/(m-1);
    end
end
if typeB ==1;    MY = M2*Y;
else
    if pdB ==0
        MY = (1-gamma)* B'*(B*Y)/(m-1) + gamma*Y;
    else ; MY = B'*(B*Y)/(m-1);
    end
end
if typeC ==1;    MZ = M3*Z;
else
    if pdC ==0
        MZ = (1-gamma)* C'*(C*Z)/(m-1) + gamma*Z;
    else ; MZ = C'*(C*Z)/(m-1);
    end
end
%F(1) = frob(double(ttm(CC,{X.',Y.',Z.'},[1,2,3]))) +h1(X) +h2(Y) +h3(Z) ;
F(1) = frob(double(ttm(CC,{X.',Y.',Z.'},[1,2,3]))) +trace(X.'*AtlaA*X)+trace(Y.'*BtlbB*Y)+trace(Z.'*CtlcC*Z)+h1(X) +h2(Y) +h3(Z) ;
num3 = zeros(maxiter,1);
num2 = zeros(maxiter,1); 
num1 = num2;
flag_maxiter = 0;% flag_linesearch =zeros(maxiter,1);
num_inex = 0; 
linesearch_num = 0;
alpha = 1;


%fX = frob(double(ttm(CC,{X.',Y.',Z.'},[1,2,3])))  + h1(X);
fX = frob(double(ttm(CC,{X.',Y.',Z.'},[1,2,3]))) +trace(X.'*AtlaA*X) + h1(X);
fY = frob(double(ttm(CC,{X.',Y.',Z.'},[1,2,3]))) +trace(Y.'*BtlbB*Y)+ h2(Y);
fZ = frob(double(ttm(CC,{X.',Y.',Z.'},[1,2,3]))) +trace(Z.'*CtlcC*Z)+ h3(Z);
for iter=2:maxiter
    %% update X
     ggx=ttm(CC,{Y.',Z.'},[2,3]);
     G_mode_1 = unfold(ggx, 1);
     G_mode_1=double(G_mode_1)/(m-1)^2;
     ggx1=ttm(CC,{X.',Y.',Z.'},[1,2,3]);
     G_mode1 = unfold(ggx1, 1);
     G_mode1=double(G_mode1)/(m-1)^3;
    gx = (G_mode1 * G_mode_1.')+AtlaA*X/(m-1);  pgx=gx;  % grad or projected gradient both okay
    %% subproblem
    if alpha < t_min || num_inex >10
        inner_tol = option.inner_tol/100; % if subproblem inexact, decrease the tol
    else
        inner_tol = option.inner_tol;
    end
    t1 = 1;
    if iter == 2
        [ PX,num1(iter),Lam1,~,in_flag1]=Semi_newton_matrix_l21(p,n,MX,t1,X-t1*pgx,tau1*t1,inner_tol,prox_func,zeros(n),Dn,pDn);
    else
        [ PX,num1(iter),Lam1,~ ,in_flag1]=Semi_newton_matrix_l21(p,n,MX,t1,X-t1*pgx,tau1*t1,inner_tol,prox_func,Lam1,Dn,pDn);
    end
    if in_flag1 == 1   % subprolem total iteration.
        inner_flag1 = 1 + inner_flag1;
    end
    alpha=1;
    DX = PX - X; %descent direction D
    X_temp  = retraction(PX,M1,A,typeA);
    %fX = sum(sum(X.*(AY))) + h1(X);
    f_trial=frob(double(ttm(CC,{X_temp.',Y.',Z.'},[1,2,3])))+trace(X_temp.'*AtlaA*X_temp);
    f_trialX=f_trial+ h1(X_temp) ;   normDsquared_X=norm(DX,'fro')^2; 
    %if  max( normDsquared_X,  normDsquared_Y) < tol
      %  fprintf('A-ManPG terminates: converged iteration:%4d\n', iter);
       % break;
    %end
    %% linesearch
    while f_trialX >= fX - 1e-4*alpha*normDsquared_X
        alpha=0.5*alpha;
        if alpha < t_min
            num_inex = num_inex+1;
            break;
        end
        PX=X+alpha*DX;   X_temp  = retraction(PX,M1,A,typeA);
        linesearch_num = linesearch_num +1;
        f_trial = frob(double(ttm(CC,{X_temp.',Y.',Z.'},[1,2,3])))+trace(X_temp.'*AtlaA*X_temp);    f_trialX = f_trial + h1(X_temp) ;
    end
    XRes1(iter)=norm(X-X_temp,'fro');
    X = X_temp;
    if typeA ==1;    MX = M1*X;
    else
        if pdA ==0
            MX = (1-gamma)* A'*(A*X)/(m-1) + gamma*X;
        else ; MX = A'*(A*X)/(m-1);
        end
    end
    X2=X;
     fX1 = frob(double(ttm(CC,{X2.',Y2.',Z2.'},[1,2,3]))) +trace(X2.'*AtlaA*X2) + h1(X2);
     diffnum1 = norm(fX-fX1);
      %diffnum=diffnum1 ;
 %if  diffnum1<tol;
      %fprintf('A-ManPG terminates: converged iteration:%4d\n', iter);
       % break;
 %end
  %%  update Y
    ggx=ttm(CC,{X.',Z.'},[1,3]);
     G_mode_2 = unfold(ggx, 2);
     G_mode_2=double(G_mode_2)/(m-1)^2;
     ggx1=ttm(CC,{X.',Y.',Z.'},[1,2,3]);
     G_mode1 = unfold(ggx1, 2);
     G_mode1=double(G_mode1)/(m-1)^3;
    gy = (G_mode1 * G_mode_2.')+BtlbB*Y/(m-1); 
    pgy=gy;  t2 =1;
    if iter == 2
        [ PY,num2(iter),Lam2,~,in_flag2] = Semi_newton_matrix_l21(q,n,MY,t2,Y-t2*pgy,tau2*t2,inner_tol,prox_func,zeros(n),Dn,pDn);
    else
        [ PY,num2(iter),Lam2,~ ,in_flag2] = Semi_newton_matrix_l21(q,n,MY,t2,Y-t2*pgy,tau2*t2,inner_tol,prox_func,Lam2,Dn,pDn);
    end
    alpha=1;     DY = PY - Y;%descent direction D
    Y_temp  = retraction(PY,M2,B,typeB);
    
    f_trial=frob(double(ttm(CC,{X.',Y_temp.',Z.'},[1,2,3])))+trace(Y_temp.'*BtlbB*Y_temp);
    f_trialY=f_trial+ h2(Y_temp) ;   normDsquared_Y = norm(DY,'fro')^2 ;
    %% linesearch
    while f_trialY >= fY -1e-4*alpha*normDsquared_Y
        alpha=0.5*alpha;
        if alpha < t_min
            num_inex = num_inex+1;
            break;
        end
        PY = Y + alpha*DY;    Y_temp  = retraction(PY,M2,B,typeB);
        linesearch_num = linesearch_num +1;
        f_trial = frob(double(ttm(CC,{X.',Y_temp.',Z.'},[1,2,3])))+trace(Y_temp.'*BtlbB*Y_temp);  f_trialY = f_trial+ h2(Y_temp) ;
    end
    YRes1(iter)=norm(Y-Y_temp,'fro');
    Y = Y_temp;
    if typeB ==1;    MY = M2*Y;
    else
        if pdB ==0
            MY = (1-gamma)* B'*(B*Y)/(m-1) + gamma*Y;
        else ; MY = B'*(B*Y)/(m-1);
        end
    end
     Y2=Y;
     fY1 = frob(double(ttm(CC,{X2.',Y2.',Z2.'},[1,2,3]))) +trace(Y2.'*BtlbB*Y2) + h2(Y2);
     diffnum2 = norm(fY-fY1);
      %diffnum=diffnum2 ;
 %if  diffnum2<tol;
     % fprintf('A-ManPG terminates: converged iteration:%4d\n', iter);
      %  break;
 %end
  %%  update Z
    ggz=ttm(CC,{X.',Y.'},[1,2]);
     G_mode_3 = unfold(ggz, 3);
     G_mode_3=double(G_mode_3)/(m-1)^2;
     ggx1=ttm(CC,{X.',Y.',Z.'},[1,2,3]);
     G_mode1 = unfold(ggx1, 3);
     G_mode1=double(G_mode1)/(m-1)^3;
    gz = (G_mode1* G_mode_3.')+CtlcC*Z/(m-1); 
    pgz=gz;  t3 =1;
    if iter == 2
        [ PZ,num3(iter),Lam3,~,in_flag3] = Semi_newton_matrix_l21(L,n,MZ,t3,Z-t3*pgz,tau3*t3,inner_tol,prox_func,zeros(n),Dn,pDn);
    else
        [ PZ,num3(iter),Lam3,~ ,in_flag3] = Semi_newton_matrix_l21(L,n,MZ,t3,Z-t3*pgz,tau3*t3,inner_tol,prox_func,Lam3,Dn,pDn);
    end
    alpha=1;     DZ = PZ - Z;%descent direction D
    Z_temp  = retraction(PZ,M3,C,typeC);
    f_trial=frob(double(ttm(CC,{X.',Y.',Z_temp.'},[1,2,3])))+trace(Z_temp.'*CtlcC*Z_temp);
    f_trialZ=f_trial+ h3(Z_temp) ;    normDsquared_Z = norm(DZ,'fro')^2 ;
    %% linesearch
    while f_trialZ >= fZ -1e-4*alpha*normDsquared_Z
        alpha=0.5*alpha;
        if alpha < t_min
            num_inex = num_inex+1;
            break;
        end
        PZ = Z + alpha*DZ;    Z_temp  = retraction(PZ,M3,C,typeC);
        linesearch_num = linesearch_num +1;
        f_trial =  frob(double(ttm(CC,{X.',Y.',Z_temp.'},[1,2,3])))+trace(Z_temp.'*CtlcC*Z_temp);  f_trialZ = f_trial+ h3(Z_temp) ;
    end
    ZRes1(iter)=norm(Z-Z_temp,'fro');
    Z = Z_temp;
    
    if typeC ==1;    MZ = M3*Z;
    else
        if pdC ==0
            MZ = (1-gamma)* C'*(C*Y)/(m-1) + gamma*Z;
        else ; MZ = C'*(C*Z)/(m-1);
        end
    end
     Z2=Z;
     fZ1 = frob(double(ttm(CC,{X2.',Y2.',Z2.'},[1,2,3]))) +trace(Z2.'*CtlcC*Z2) + h3(Z2);
     diffnum3 = norm(fZ-fZ1);
      %diffnum=diffnum3 ;
      diffnum=[diffnum1,diffnum2,diffnum3];
      mm=max(diffnum);
       fX = frob(double(ttm(CC,{X.',Y.',Z.'},[1,2,3])))+trace(X.'*AtlaA*X) + h1(X);
    F(iter) = fX +trace(Y.'*BtlbB*Y)+ h2(Y) +trace(Z.'*CtlcC*Z) + h3(Z);
    %fprintf('%4d, -th iteration,%d,XRes1=%d,YRes1=%d,ZRes1=%d\n', iter,(F(iter)-F(iter-1)),XRes1(iter),YRes1(iter),ZRes1(iter));
     %fprintf('%4d, -th iteration,%d,\n', iter-1,(F(iter)-F(iter-1)));
    %fprintf('%4d, -th iteration,%d,diffnum1=%d,diffnum2=%d,diffnum3=%d,diffnum4=%d,\n', iter,(F(iter)-F(iter-1))/F(iter),diffnum1,diffnum2,diffnum3,diffnum4);
 if  mm<tol;
      %fprintf('A-ManPG terminates: converged iteration:%4d\n', iter);
        break;
 end
   
    if iter ==maxiter
        flag_maxiter =1;
        fprintf('A-ManPG terminates: Achieved maximum iteration. \n');
    end
    
    
end
X((abs(X)<=1e-4))=0;
Y((abs(Y)<=1e-4))=0;
Z((abs(Z)<=1e-4))=0;

Res.time = toc;
Res.sparsityX = sum(sum(X~=0));%/(p*n);
Res.sparsityY = sum(sum(Y~=0));%/(q*n);
Res.sparsityZ = sum(sum(Z~=0));%/(q*n);
Res.Fval =  F(iter-1);
Res.X = X; 
Res.Y =Y;
Res.Z =Z;
Res.flag_maxiter =flag_maxiter;  
Res.iter = iter;
%Res.inner_total = sum(num1)+sum(num2)+sum(num3); 
%Res.linesearch_num = linesearch_num;


end