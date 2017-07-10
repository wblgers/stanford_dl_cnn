function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1);
numImages = size(images,3);

%% Reshape parameters and setup gradient matrices

% 将unrolled的参数整理为卷积层和全连接层的权重矩阵和bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: 前向传播

%% 卷积层代码
% 卷积层输出的数据维度
convDim = imageDim-filterDim+1; 
% 均值池化输出的数据维度
outputDim = (convDim)/poolDim; 

% 初始化卷积特征提取后的输出数据
activations = zeros(convDim,convDim,numFilters,numImages);

% 初始化池化后的输出数据
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

% 卷积特征提取操作
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc); 
% 均值池化操作
activationsPooled = cnnPool(poolDim, activations);

% 将4-D特征矩阵转换为2-D的，作为softmax层的输入，以图像数numImages为列进行转换
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
% 将池化后的特征转换为二维矩阵，即隐层特征数hiddenSize×训练图像个数numImages，对其进行softmax概率计算

% 初始化numClasses*numImages的矩阵，用来存储每一个图像对应于每一个标签的概率
probs = zeros(numClasses,numImages);

% 计算hypothesis-h(x)
M = Wd*activationsPooled+repmat(bd,[1,numImages]); 
M = exp(M);
probs = bsxfun(@rdivide, M, sum(M));
%%======================================================================
%% Softmax层计算cost
% 根据训练样本的正确标记值和上一步得到的概率作为输入，计算交叉熵对象，并且保存在cost里
% cost初始化
cost = 0;

% 首先需要把labels弄成one-hot编码，即2-D矩阵numClasses×numImages
groundTruth = full(sparse(labels, 1:numImages, 1));

%按照公式计算，矩阵操作进行求和
cost = -1./numImages*groundTruth(:)'*log(probs(:));

% 如果当前只是做预测，那么只返回预测的标签，不再进行计算，在做test时会遇到

if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return
end
%======================================================================
%% 反向传播
% 从输出层将误差反向传播至softmax层和卷积&池化层，在每一层保存对应的误差项，用于
% 计算梯度值，完成梯度下降法的迭代。

% 网络结构: images--> convolvedFeatures--> activationsPooled--> probs

% 反向传播至softmax层的误差项
delta_d = -(groundTruth-probs); 

% 反向传播至池化层的误差项
delta_s = Wd'*delta_d;
delta_s = reshape(delta_s,outputDim,outputDim,numFilters,numImages);

% 反向传播至卷积层的误差项
delta_c = zeros(convDim,convDim,numFilters,numImages);
for i=1:numImages
    for j=1:numFilters
        delta_c(:,:,j,i) = (1./poolDim^2)*kron(delta_s(:,:,j,i), ones(poolDim));
    end
end

delta_c = activations.*(1-activations).*delta_c;

%% 梯度计算
% 用delta_d计算softmax层权重系数的梯度值
Wd_grad = (1./numImages)*delta_d*activationsPooled';
% 用delta_d计算softmax层bias项的梯度值，注意这里是要求和
bd_grad = (1./numImages)*sum(delta_d,2);

% 用delta_c计算卷积层权重系数和bias项的的梯度值
for i=1:numFilters
    Wc_i = zeros(filterDim,filterDim);
    for j=1:numImages  
        Wc_i = Wc_i+conv2(squeeze(images(:,:,j)),rot90(squeeze(delta_c(:,:,i,j)),2),'valid');
    end

    Wc_grad(:,:,i) = (1./numImages)*Wc_i;
    
    bc_i = delta_c(:,:,i,:);
    bc_i = bc_i(:);
    bc_grad(i) = sum(bc_i)/numImages;
end

%% 将梯度向量展开至列向量，作为minFunc的输入
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end