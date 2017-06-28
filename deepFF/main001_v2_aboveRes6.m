%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
%{
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

%}
clear
% close all
clc;
% cityscape - full input image
% input image       -- 1024x2048x3
% conv1 -- stride=2 -- 512x1024x64
% pool1 -- stride=2 -- 256x512x64
% res2a -- stride=1 -- 256x512x256
% res3a -- stride=2 -- 128x256x512
% res4a -- stride=2 -- 64x128x1024
% res5a -- stride=2 -- 32x64x2048

addpath(genpath('../libs'))
path_to_matconvnet = '../matconvnet';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
path_to_model = '../basemodels/';

% set GPU 
gpuId = 3;
gpuDevice(gpuId);
%% load imdb file
load('imdb_cityscapes.mat');
imdb.meta.classNum = imdb.num_classes;

idfactor = 10.^([1,2,3]-1);
labelDict = load('labelDictionary.mat');
validIndex = find(labelDict.ignoreInEval==0);
colorLabel = labelDict.colorLabel(validIndex,:);
colorID = sum(bsxfun(@times, colorLabel, idfactor), 2);
categoryName = labelDict.categoryName(validIndex);
classID = labelDict.classID(validIndex);
className = labelDict.className(validIndex);
hasInstances = labelDict.hasInstances(validIndex);
trainClassID = labelDict.trainClassID(validIndex);
classNum = length(trainClassID);

if ~exist('legendRGB.jpg', 'file')
    legendRGB = zeros(400,200,3);
    for i = 0:classNum-1
        legendRGB(1+i*20:i*20+20,:,1) = colorLabel(i+1,1);
        legendRGB(1+i*20:i*20+20,:,2) = colorLabel(i+1,2);
        legendRGB(1+i*20:i*20+20,:,3) = colorLabel(i+1,3);
    end
    figure(1000);
    imshow(uint8(legendRGB)); %title('legend');
    for i = 1:19
        text(20, i*20-10, className{i}, 'rotation', 00, 'color', 'white', 'fontSize', 12);
    end
    text(20, 20*20-10, 'void', 'rotation', 00, 'color', 'white', 'fontSize', 12)
    export_fig( 'legendRGB.jpg' );
end
legendRGB = imread('legendRGB.jpg');
%% sepcify model
modelName = 'initNet_cityscapes.mat';
netbasemodel = load( fullfile(path_to_model, modelName) );
netbasemodel = netbasemodel.netMat;
for i = 1:numel(netbasemodel.layers)            
    curLayerName = netbasemodel.layers(i).name;
    if ~isempty(strfind(curLayerName, 'bn'))
        %netbasemodel.layers(i).block = rmfield(netbasemodel.layers(i).block, 'usingGlobal');
        netbasemodel.layers(i).block.bnorm_moment_type_trn = 'global';
        netbasemodel.layers(i).block.bnorm_moment_type_tst = 'global';        
    end
end
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
RFinfo = netbasemodel.getVarReceptiveFields('data');
%% modify the basemodel to fit segmentation task
% netbasemodel.removeLayer('prob'); % remove layer
% netbasemodel.removeLayer('fc1000'); % remove layer
% netbasemodel.removeLayer('pool5'); % remove layer
layerList = {netbasemodel.layers.name};
paramList = {netbasemodel.params.name};
%% add objective function layer
scalingFactor = 1;
netbasemodel.meta.normalization.averageImage = reshape([123.68, 116.779,  103.939],[1,1,3]); % imagenet mean values
netbasemodel.meta.normalization.imageSize = [1024, 2048, 3, 1];
netbasemodel.meta.normalization.border = [304, 1328]; % 720x720
netbasemodel.meta.normalization.stepSize = [76, 83];
% netbasemodel.meta.normalization.border = [128, 1152]; % 896x896
% netbasemodel.meta.normalization.stepSize = [64, 384];
%% set learning rate for layers
netbasemodel.removeLayer('SoftMaxLayer'); % remove layer
netbasemodel.removeLayer('res7_interp'); % remove layer
netbasemodel.removeLayer('res7_conv'); % remove layer
netbasemodel.removeLayer('res6_dropout'); % remove layer
netbasemodel.removeLayer('res6_relu'); % remove layer
netbasemodel.removeLayer('res6_bn'); % remove layer
netbasemodel.removeLayer('res6_conv'); % remove layer
netbasemodel.removeLayer('concatLayer'); % remove layer
netbasemodel.removeLayer('pyramid_pool1'); % remove layer
netbasemodel.removeLayer('pyramid_pool1_conv'); % remove layer
netbasemodel.removeLayer('pyramid_pool1_bn'); % remove layer
netbasemodel.removeLayer('pyramid_pool1_relu'); % remove layer
netbasemodel.removeLayer('pyramid_pool1_interp'); % remove layer
netbasemodel.removeLayer('pyramid_pool2'); % remove layer
netbasemodel.removeLayer('pyramid_pool2_conv'); % remove layer
netbasemodel.removeLayer('pyramid_pool2_bn'); % remove layer
netbasemodel.removeLayer('pyramid_pool2_relu'); % remove layer
netbasemodel.removeLayer('pyramid_pool2_interp'); % remove layer
netbasemodel.removeLayer('pyramid_pool3'); % remove layer
netbasemodel.removeLayer('pyramid_pool3_conv'); % remove layer
netbasemodel.removeLayer('pyramid_pool3_bn'); % remove layer
netbasemodel.removeLayer('pyramid_pool3_relu'); % remove layer
netbasemodel.removeLayer('pyramid_pool3_interp'); % remove layer
netbasemodel.removeLayer('pyramid_pool6'); % remove layer
netbasemodel.removeLayer('pyramid_pool6_conv'); % remove layer
netbasemodel.removeLayer('pyramid_pool6_bn'); % remove layer
netbasemodel.removeLayer('pyramid_pool6_relu'); % remove layer
netbasemodel.removeLayer('pyramid_pool6_interp'); % remove layer

sName = 'res5_3_relu';
lName = 'res6_conv';
block = dagnn.Conv('size', [3 3 2048 512], 'hasBias', false, 'stride', 1, 'pad', 1, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f']});
sName = lName;
% ind = netbasemodel.getParamIndex([lName '_b']);
% netbasemodel.params(ind).value = zeros([1 basisFuncTimes*imdb.meta.classNum], 'single');
ind = netbasemodel.getParamIndex([lName '_f']);
weights = randn(3, 3, 2048, 512, 'single')*sqrt(2/512);
netbasemodel.params(ind).value = weights;


lName = 'res6_bn';
block = dagnn.BatchNorm('numChannels', 512);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_g'], [lName '_b'], [lName '_m']});
netbasemodel.layers(netbasemodel.getLayerIndex(lName)).block.bnorm_moment_type_trn = 'batch';
netbasemodel.layers(netbasemodel.getLayerIndex(lName)).block.bnorm_moment_type_tst = 'global';
pidx = netbasemodel.getParamIndex({[lName '_g'], [lName '_b'], [lName '_m']});
netbasemodel.params(pidx(1)).weightDecay = 1;
netbasemodel.params(pidx(2)).weightDecay = 1;
netbasemodel.params(pidx(3)).learningRate = 0.1;
netbasemodel.params(pidx(3)).trainMethod = 'average';
netbasemodel.params(pidx(1)).value = ones(512, 1, 'single'); % slope
netbasemodel.params(pidx(2)).value = zeros(512, 1, 'single');  % bias
netbasemodel.params(pidx(3)).value = zeros(512, 2, 'single'); % moments
sName = lName;

lName = 'res6_relu';
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;

lName = 'res6_dropout' ;
block = dagnn.DropOut('rate', 0.1);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;

block = dagnn.Conv('size', [1 1 512 imdb.meta.classNum], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
lName = 'res7_conv';
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
sName = lName;
ind = netbasemodel.getParamIndex([lName '_b']);
netbasemodel.params(ind).value = zeros([1 imdb.meta.classNum], 'single');
ind = netbasemodel.getParamIndex([lName '_f']);
weights = randn(1, 1, 512, imdb.meta.classNum, 'single')*sqrt(2/imdb.meta.classNum);
netbasemodel.params(ind).value = weights;


baseName = 'res7';
upsample_fac = 8;
filters = single(bilinear_u(upsample_fac*2, 19, 19));
crop = ones(1,4) * upsample_fac/2;
deconv_name = [baseName, '_interp'];
var_to_up_sample = [baseName, '_conv'];
netbasemodel.addLayer(deconv_name, ...
    dagnn.ConvTranspose('size', size(filters), ...
    'upsample', upsample_fac, ...
    'crop', crop, ...
    'opts', {'cudnn','nocudnn'}, ...
    'numGroups', 19, ...
    'hasBias', false), ...
    var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
ind = netbasemodel.getParamIndex([deconv_name  '_f']) ;
netbasemodel.params(ind).value = filters ;
netbasemodel.params(ind).learningRate = 0 ;
netbasemodel.params(ind).weightDecay = 1 ;

obj_name = sprintf('obj_div%d_seg', scalingFactor);
gt_name =  sprintf('gt_div%d_seg', scalingFactor);
input_name = 'res7_interp';
netbasemodel.addLayer(obj_name, ...
    SegmentationLossLogistic('loss', 'softmaxlog'), ... softmaxlog logistic
    {input_name, gt_name}, obj_name)

for i = 1:numel(netbasemodel.params)            
    netbasemodel.params(i).learningRate = 0;
end  
%% set learning rate for specific layers
for i = 317:numel(netbasemodel.layers)            
    for j = 1:length(netbasemodel.layers(i).paramIndexes)
        curInd = netbasemodel.layers(i).paramIndexes(j);
        if j == 3
            netbasemodel.params(curInd).learningRate = 0.01;
        else
            netbasemodel.params(curInd).learningRate = 1;
        end
    end    
end 

ind = netbasemodel.layers(netbasemodel.getLayerIndex('res6_conv')).paramIndexes;
netbasemodel.params(ind).learningRate = 10;

ind = netbasemodel.layers(netbasemodel.getLayerIndex('res6_bn')).paramIndexes;
netbasemodel.layers(netbasemodel.getLayerIndex('res6_bn')).block.bnorm_moment_type_trn = 'batch';
netbasemodel.layers(netbasemodel.getLayerIndex('res6_bn')).block.bnorm_moment_type_tst = 'global';
netbasemodel.params(ind(1)).learningRate = 10;
netbasemodel.params(ind(2)).learningRate = 10;
netbasemodel.params(ind(3)).learningRate = 0.1;

ind = netbasemodel.layers(netbasemodel.getLayerIndex('res7_conv')).paramIndexes;
netbasemodel.params(ind(1)).learningRate = 10;
netbasemodel.params(ind(2)).learningRate = 20;

for i = 1:numel(netbasemodel.params)            
    fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);   
end  
%% configure training environment
batchSize = 1;
totalEpoch = 150;
learningRate = 1:totalEpoch;
learningRate = (5e-5) * (1-learningRate/totalEpoch).^0.9;

weightDecay=0.0005; % weightDecay: usually use the default value

opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

opts.scalingFactor = scalingFactor;

opts.expDir = fullfile('./exp', 'main001_v2_aboveRes6');
if ~isdir(opts.expDir)
    mkdir(opts.expDir);
end

opts.numSubBatches = 1 ;
opts.continue = true ;
opts.gpus = gpuId ;
%gpuDevice(opts.train.gpus); % don't want clear the memory
opts.prefetch = false ;
opts.sync = false ; % for speed
opts.cudnn = true ; % for speed
opts.numEpochs = numel(opts.learningRate) ;
opts.learningRate = learningRate;

for i = 1:3
    curSetName = imdb.sets.name{i};
    curSetID = imdb.sets.id(i);
    curList = find(imdb.images.set==curSetID);
    opts.(curSetName) = curList(1:end);    
end

opts.checkpointFn = [];
mopts.classifyType = 'softmax';

rng(777);
bopts = netbasemodel.meta.normalization;
bopts.numThreads = 12;
bopts.imdb = imdb;
%% train
fn = getImgBatchWrapper(bopts);
opts.backPropDepth = 10; %inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
opts.backPropAboveLayerName = 'res6_conv';
% opts.backPropAboveLayerName = 'res5_1_projBranch';

trainfn = @cnnTrain;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', {sprintf('obj_div%d_seg', scalingFactor), 1}, opts);
%% leaving blank
%{
./evalPixelLevelSemanticLabeling.py noPSP_v3_standardConv2_softmax_net-epoch-17_predEval
classes          IoU      nIoU
--------------------------------
road          : 0.947      nan
sidewalk      : 0.684      nan
building      : 0.914      nan
wall          : 0.470      nan
fence         : 0.579      nan
pole          : 0.571      nan
traffic light : 0.659      nan
traffic sign  : 0.761      nan
vegetation    : 0.912      nan
terrain       : 0.518      nan
sky           : 0.934      nan
person        : 0.781    0.612
rider         : 0.567    0.422
car           : 0.912    0.842
truck         : 0.624    0.397
bus           : 0.763    0.584
train         : 0.609    0.470
motorcycle    : 0.523    0.368
bicycle       : 0.729    0.558
--------------------------------
Score Average : 0.708    0.532
--------------------------------


./evalPixelLevelSemanticLabeling.py noPSP_v3_standardConv2_softmax_net-epoch-27_predEval
classes          IoU      nIoU
--------------------------------
road          : 0.947      nan
sidewalk      : 0.686      nan
building      : 0.916      nan
wall          : 0.467      nan
fence         : 0.587      nan
pole          : 0.584      nan
traffic light : 0.669      nan
traffic sign  : 0.769      nan
vegetation    : 0.914      nan
terrain       : 0.525      nan
sky           : 0.936      nan
person        : 0.786    0.626
rider         : 0.578    0.432
car           : 0.915    0.849
truck         : 0.630    0.400
bus           : 0.771    0.590
train         : 0.620    0.473
motorcycle    : 0.536    0.379
bicycle       : 0.738    0.564
--------------------------------
Score Average : 0.714    0.539
--------------------------------

./evalPixelLevelSemanticLabeling.py noPSP_v3_standardConv2_softmax_net-epoch-29_predEval
classes          IoU      nIoU
--------------------------------
road          : 0.948      nan
sidewalk      : 0.687      nan
building      : 0.917      nan
wall          : 0.489      nan
fence         : 0.588      nan
pole          : 0.586      nan
traffic light : 0.673      nan
traffic sign  : 0.771      nan
vegetation    : 0.916      nan
terrain       : 0.526      nan
sky           : 0.936      nan
person        : 0.788    0.627
rider         : 0.580    0.433
car           : 0.915    0.854
truck         : 0.632    0.401
bus           : 0.772    0.593
train         : 0.628    0.477
motorcycle    : 0.538    0.379
bicycle       : 0.739    0.569
--------------------------------
Score Average : 0.717    0.542
--------------------------------



./evalPixelLevelSemanticLabeling.py noPSP_v3_standardConv2_softmax_net-epoch-35_predEval
classes          IoU      nIoU
--------------------------------
road          : 0.948      nan
sidewalk      : 0.689      nan
building      : 0.917      nan
wall          : 0.490      nan
fence         : 0.590      nan
pole          : 0.593      nan
traffic light : 0.678      nan
traffic sign  : 0.773      nan
vegetation    : 0.917      nan
terrain       : 0.529      nan
sky           : 0.938      nan
person        : 0.792    0.627
rider         : 0.586    0.435
car           : 0.916    0.858
truck         : 0.635    0.405
bus           : 0.773    0.590
train         : 0.633    0.477
motorcycle    : 0.545    0.385
bicycle       : 0.743    0.574
--------------------------------
Score Average : 0.720    0.544
--------------------------------



./evalPixelLevelSemanticLabeling.py noPSP_v3_standardConv2_softmax_net-epoch-55_predEval
classes          IoU      nIoU
--------------------------------
road          : 0.949      nan
sidewalk      : 0.692      nan
building      : 0.919      nan
wall          : 0.484      nan
fence         : 0.594      nan
pole          : 0.601      nan
traffic light : 0.689      nan
traffic sign  : 0.780      nan
vegetation    : 0.918      nan
terrain       : 0.534      nan
sky           : 0.941      nan
person        : 0.797    0.635
rider         : 0.597    0.450
car           : 0.919    0.858
truck         : 0.643    0.406
bus           : 0.781    0.604
train         : 0.647    0.484
motorcycle    : 0.553    0.392
bicycle       : 0.750    0.582
--------------------------------
Score Average : 0.726    0.551
--------------------------------



debug
./evalPixelLevelSemanticLabeling.py noPSP_v3_standardConv2_softmax_net-epoch-55_predEval
classes          IoU      nIoU
--------------------------------
road          : 0.980      nan
sidewalk      : 0.849      nan
building      : 0.916      nan
wall          : 0.475      nan
fence         : 0.596      nan
pole          : 0.598      nan
traffic light : 0.684      nan
traffic sign  : 0.780      nan
vegetation    : 0.918      nan
terrain       : 0.619      nan
sky           : 0.941      nan
person        : 0.803    0.635
rider         : 0.594    0.448
car           : 0.939    0.859
truck         : 0.631    0.398
bus           : 0.759    0.595
train         : 0.621    0.467
motorcycle    : 0.562    0.396
bicycle       : 0.755    0.582
--------------------------------
Score Average : 0.738    0.547
--------------------------------

%}
