function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

% 对于每一副训练图像的每一个特征维，进行平均值池化
for imageNum = 1:numImages
  for filterNum = 1:numFilters
      tempFeature = convolvedFeatures(:,:,filterNum,imageNum);
      filter_rot = ones(poolDim,poolDim);
      tempPooledFeature = conv2(tempFeature,filter_rot,'valid')/(size(filter_rot,1)*size(filter_rot,2));
      
      pooledFeatures(:,:,filterNum,imageNum) = tempPooledFeature(1:poolDim:size(tempPooledFeature,1),1:poolDim:size(tempPooledFeature,2));
  end
  
end

end

