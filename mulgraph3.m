 function [S,num] = mulgraph3(A,B,C,K,dim)
 
aa{1} = full((A.' - mean(A.', 2)) ./ repmat(std(A.', [], 2), 1, size(A.', 2)));
    aa{2} = full((B.' - mean(B.', 2)) ./ repmat(std(B.', [], 2), 1, size(B.', 2)));
    aa{3} = full((C.' - mean(C.', 2)) ./ repmat(std(C.', [], 2), 1, size(C.', 2)));
   

num=size(aa{1},1);
V=length(aa);
%c=length(unique(Y));

%K=[2];
if dim<2
    taa1=aa{1};
    aaa{1}=taa1(:,1:dim);
    taa2=aa{2};
    aaa{2}=taa2(:,1:dim);
    taa3=aa{3};
    aaa{3}=taa3(:,1:dim);
   
   % aaa = constructA_vd(aa, dim, K);
else
    aaa = constructA_vd(aa, 5, K);
    
end
%aaa
%num
%V
%K
%dim
    [S,~,~] =solverS(aaa,num,V,K,dim);
aa = pdist(A.', 'euclidean');
aa=squareform(aa);
bb = pdist(B.', 'euclidean');
bb=squareform(bb);
cc = pdist(C.', 'euclidean');
cc=squareform(cc);
