clc;
clear;
close all;

% CSDN博客地址，欢迎指教！
% http://blog.csdn.net/wblgers1234/article/details/70921248

%%======================================================================
%% 第一步：初始化参数，载入训练数据

% 参数配置

% MNIST数据库图片的大小为28×28
imageDim = 28;

% 要分类的类别数
numClasses = 10;

%卷积层的特征提取模块的维数(滤波器维数)
filterDim = 9; 

% 特征提取滤波器的个数
numFilters = 20;

% 池化的维数，应该被imageDim-filterDim+1整除
poolDim = 2;

% 载入MNIST数据库的训练数据
addpath function/;

images = loadMNISTImages('../MNIST/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('../MNIST/train-labels-idx1-ubyte');
% 将分类标签0重新映射到10
labels(labels==0) = 10;

% 初始化参数
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);

%%======================================================================
%% Gradient Check

% 设置为false意味着不做gradient check
DEBUG=false;
% DEBUG=true;
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
    db_numFilters = 2;
    db_filterDim = 9;
    db_poolDim = 5;
    db_images = images(:,:,1:10);
    db_labels = labels(1:10);
    db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
                db_poolDim,numClasses);
    
    [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,...
                                db_filterDim,db_numFilters,db_poolDim);
    

    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
                                db_labels,numClasses,db_filterDim,...
                                db_numFilters,db_poolDim), db_theta);
 
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff); 
 
    assert(diff < 1e-9,...
        'Difference too large. Check your gradient computation again');
    
end;

%%======================================================================
%% 训练CNN网络

options.epochs = 3;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,...
                      numFilters,poolDim),theta,images,labels,options);

%%======================================================================
%% 测试CNN网络

% 载入MNIST数据库的测试集
testImages = loadMNISTImages('../MNIST/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('../MNIST/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10;

[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim,true);

acc = sum(preds==testLabels)/length(preds);

% 打印出测试集的分类准备率
fprintf('Accuracy is %f\n',acc);
