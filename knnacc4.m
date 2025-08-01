function [accr,f1,t] = knnacc4(testRatio,train_num,test_num,STX100,label,class)
  
   tic;
    STX100trainIndices = crossvalind('HoldOut', size(STX100, 1), testRatio);
    STX100testIndices = ~STX100trainIndices;
    STX100trainData = STX100(STX100trainIndices, :);
    STXtrainLabel = label(STX100trainIndices, :);
    STX100testData = STX100(STX100testIndices, :);
    STX100testLabel = label(STX100testIndices, :);
    %size(STX100trainData)
    %size(STXtrainLabel(1:train_num))
    %STX100trainData(1:2,1:10)
    %STXtrainLabel(1:train_num)
    knn_model = fitcknn(STX100trainData,STXtrainLabel(1:train_num),'NumNeighbors',class);
    result_KNN = predict(knn_model,STX100testData);
   STX100 = NormalizeFea(STX100, 1);
%X = normalizeColumn(X);        % 数据标准化
%label = result_KNN;            % 数据标签1-5，样本共5类
%Y = tsne(STX100testData);                  % 得到的矩阵为Nx2，N为N个样本，Y矩阵为320x2
%figure;
%gscatter(Y(:,1), Y(:,2),label);% 若无label输入，则画出的图没有色彩区分
    acc = 0.;
    for i = 1:test_num
        if result_KNN(i)==STX100testLabel(i)
           acc = acc+1;
        end
    end  
    accr  = (acc/test_num)*100;
    [f1,f1_c] = f1score(STX100testLabel,result_KNN);
    t=toc;
    
end
