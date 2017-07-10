function theta = cnnInitParams(imageDim,filterDim,numFilters,...
                                poolDim,numClasses)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% 随机初始化参数
assert(filterDim < imageDim,'filterDim must be less that imageDim');

% Wc = 1e-1*randn(filterDim,filterDim,numFilters);

% 二维卷积后的图像维数
outDim = imageDim - filterDim + 1;

% outDim应该能被池化维度整除
assert(mod(outDim,poolDim)==0,...
       'poolDim must divide imageDim - filterDim + 1');

% 池化后的特征维数
outDim = outDim/poolDim;
% 卷积层池化后的特征维数
hiddenSize = outDim^2*numFilters;

% 不是很清楚r的取值
r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);

% 卷积层和全连接层的权重矩阵
Wc = 1e-1*randn(filterDim,filterDim,numFilters);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;

% 卷积层和全连接层的bias项
bc = zeros(numFilters, 1);
bd = zeros(numClasses, 1);

% 将所有的参数展开放在一个列向量
theta = [Wc(:) ; Wd(:) ; bc(:) ; bd(:)];

end

