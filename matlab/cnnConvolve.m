function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 30 seconds 
%   Convolving with 5000 images should take around 2 minutes
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% 用每一个特征提取权重（即二维滤波系数）对每幅图片进行处理卷积处理
for imageNum = 1:numImages
  for filterNum = 1:numFilters
    % 卷积操作后提取得到的特征矩阵
    convolvedImage = zeros(convDim, convDim);

    % 从权重矩阵中提取每一个二维的特征提取权重(filterDim * filterDim)
    filter = W(:,:,filterNum);
    % 对特征提取权重矩阵进行上下左右翻转
    filter_rot = rot90(squeeze(filter),2);
      
    % 提取图像
    im = squeeze(images(:, :, imageNum));

    % 二维卷积进行特征提取并加上bias偏差
    convolvedImage = conv2(im,filter_rot,'valid')+b(filterNum);

    % 用sigmod函数来得到激活值
    convolvedImage = sigmoid(convolvedImage);
    
    convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end
end