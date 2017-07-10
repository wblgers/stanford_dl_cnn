function [Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,...
                                 numFilters,poolDim,numClasses)
% 这个函数的作用是将unrolled的参数列向量重新整理成为每一层的权重矩阵和bias
%                            
% Parameters:
%  theta      -  unrolled parameter vectore
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  Wc      -  filterDim x filterDim x numFilters parameter matrix
%  Wd      -  numClasses x hiddenSize parameter matrix, hiddenSize is
%             calculated as numFilters*((imageDim-filterDim+1)/poolDim)^2 
%  bc      -  bias for convolution layer of size numFilters x 1
%  bd      -  bias for dense layer of size hiddenSize x 1

outDim = (imageDim - filterDim + 1)/poolDim;
hiddenSize = outDim^2*numFilters;

%% Reshape theta
indS = 1;
indE = filterDim^2*numFilters;
Wc = reshape(theta(indS:indE),filterDim,filterDim,numFilters);
indS = indE+1;
indE = indE+hiddenSize*numClasses;
Wd = reshape(theta(indS:indE),numClasses,hiddenSize);
indS = indE+1;
indE = indE+numFilters;
bc = theta(indS:indE);
bd = theta(indE+1:end);


end