clc;close all; clear all; warning off
load('3Sources.mat');
label=y;
K=[1,2,3,4,5,6,7,8,9,10];
testRatio=0.3;
train_num=119;
test_num=50;
class1=6;
nExperiment=10;
corru=0.0;
%corru= [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
 for cc=1:length(corru) 
     corruption = corru(cc);%add noise
     corruption1 = 0;%列破损
    [DD1,N1] = size(X{1});
    [DD2,N2] = size(X{2});
    [DD3,N3] = size(X{3});
    
%%X{1}
a=zeros(size(X{1}));
corruption_mask = randperm( DD1*N1, round( corruption*DD1*N1 ) );
a(corruption_mask)=X{1}(corruption_mask);
a = imnoise(a,'gaussian');%高斯噪音
X{1}(corruption_mask)=a(corruption_mask);
X{1} = NormalizeFea(X{1}, 1);    %%% Normalization
%%X{2}
b=zeros(size(X{2}));
corruption_mask = randperm( DD2*N2, round( corruption*DD2*N2 ) );
b(corruption_mask)=X{2}(corruption_mask);
b = imnoise(b,'gaussian');%高斯噪音
X{2}(corruption_mask)=b(corruption_mask);
X{2} = NormalizeFea(X{2}, 1);    %%% Normalization
%%X{3}
c=zeros(size(X{3}));
corruption_mask = randperm( DD3*N3, round( corruption*DD3*N3 ) );
c(corruption_mask)=X{3}(corruption_mask);
c = imnoise(c,'gaussian');%高斯噪音
X{3}(corruption_mask)=c(corruption_mask);
X{3} = NormalizeFea(X{3}, 1);    %%% Normalization
for i1=2:2:20
[coeff, score]= pca(X{1});  % coeff閺勵垯瀵岄幋鎰瀻閸掑棝鍣洪敍灞藉祮閺嶉攱婀伴崡蹇旀煙瀹割喚鐓╅梼鐢垫畱閻楃懓绶涢崥鎴﹀櫤閿涙硞core閺勵垯瀵岄幋鎰瀻閿涘苯宓哾ata閸︺劋缍嗙紒瀵糕敄闂傚娈戦幎鏇炲閿涘奔绡冪亸杈ㄦЦ闂勫秶娣崥搴ｆ畱閺佺増宓侀敍宀?娣惔锕?鎷癲ata閻╃鎮撻敍宀冨閹娊妾风紒鏉戝煂k缂佽揪绱濋崣顏堟付鐟曚礁褰囬崜宄╅崚妤?宓嗛崣顖???
A = score(:, 1:i1);  % 闂勫秶娣崥搴ｆ畱閺佺増宓?
[coeff, score]= pca(X{2});
%size(score)
B = score(:, 1:i1); 
[coeff, score]= pca(X{3});
C = score(:, 1:i1); 

results_TCCALnew2=zeros(nExperiment, 4);

epsilon=[0.01];
lambda=[1];
%lambda=[1,0.1,0.01,0.001,0.0001,0.00001];
for i3=1:length(K)
   for i2=1:length(epsilon)
       for bb=1:length(lambda)
option.n=i1;option.p=i1;option.q=i1;option.L=i1;option.l=i1;
option.t1=1;option.t2=1;option.t3=1;
option.X0=rand(i1,i1);option.Y0=rand(i1,i1);option.Z0=rand(i1,i1);
option.maxiter=20;
option.epsilon=epsilon(i2);
option.inner_tol=10;option.tol=0.01; 
%option.b1=lambda(bb);option.b2=lambda(bb);option.b3=lambda(bb);

option.b1=0.35;option.b2=0.4;option.b3=0.5;
   for iExperiment = 1:nExperiment 
   
    %fprintf('=========================================================\n');
    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%% TCCA-Lnew2 %%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%
   [S2,num] = mulgraph3(A',B',C',K(i3),i1);
   tic
    [result_manpg_altnew2] = tcca3new4_amanpg(A,B,C,S2,num,option,20,'l21');
    A2new1=A*result_manpg_altnew2.X;
    B2new1=B*result_manpg_altnew2.Y;
     C2new1=C*result_manpg_altnew2.Z;
    Z_TCCALnew2=[A2new1,B2new1,C2new1];
    t_TCCALnew2=toc;
     [accr_TCCAL, f1score_TCCAL,~] = knnacc4(testRatio,train_num,test_num,Z_TCCALnew2,label,class1);
   
    dataformat_TCCAL = '%d-th experiment:  accr_TCCAL = %f, f1score_TCCAL = %f,time_TCCAL=%f\n';
    dataValue_TCCAL = [iExperiment, accr_TCCAL, f1score_TCCAL, t_TCCALnew2];
    results_TCCALnew2(iExperiment, :)=dataValue_TCCAL;


    end
    % output

    dataValue_TCCALnew2=mean(results_TCCALnew2, 1);
    STD_TCCALnew2=std(results_TCCALnew2(:,2));
    STD_TCCAf1new2=std(results_TCCALnew2(:,3));
    fprintf('\nAverage: Noise =%d,Dimension =%d,K =%d,epsilon =%d,lambda =%d:  accr_TCCALnew2 = %f, f1score_TCCAL  = %f,time_TCCALnew2=%f,stdacc_TCCALnew2=%f,stdf1_TCCAL=%f\n', corru(cc),i1,K(i3),epsilon(i2),lambda(bb),dataValue_TCCALnew2(2:end),STD_TCCALnew2,STD_TCCAf1new2);
end
end
end
end
 end
