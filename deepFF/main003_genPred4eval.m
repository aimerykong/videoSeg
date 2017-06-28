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
legendRGB = cat(2, legendRGB(1:201,:,:), legendRGB(202:end,:,:) );


mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);
%% read matconvnet model
% set GPU
gpuId = 2; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result


saveFolder = 'main002_v1_ftstdModelv3';
modelName = 'softmax_net-epoch-84.mat';
%% setup network%% setup network
netMat = load( fullfile('./exp', saveFolder, modelName) );
[a, b] = fileparts(modelName);
saveFolder = [saveFolder, '_', b];

netMat = netMat.net;
netMat = dagnn.DagNN.loadobj(netMat);


rmLayerName = 'obj_div1_seg';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer
    layerTop = sprintf('SoftMaxLayer');
    netMat.addLayer(layerTop, dagnn.SoftMax(),sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

netMat.move('gpu');
netMat.mode = 'test' ;
netMat.conserveMemory = 1;
%% test res2
setName = 'val';
path_to_folder = fullfile('/home/skong2/data/MarkovCNN/dataset/cityscapes_imgFine/', setName);
% path_to_folder = '/mnt/data2/skong/MarkovCNN/dataset/cityscapes_imgFine/test/';
% path_to_folder = '/mnt/data2/skong/MarkovCNN/dataset/cityscapes_imgFine/train/aachen';
% path_to_folder = '/mnt/data2/skong/MarkovCNN/dataset/cityscapes_imgFine/val/frankfurt';
% path_to_folder = '/mnt/data2/skong/MarkovCNN/dataset/cityscapes_imgFine/test/berlin/';

saveFolder4eval_FFpath = [ strrep(saveFolder,'/','')  '_FFpath_predEval']; % '/home/skong2/mySSD/'
if ~isdir(saveFolder4eval_FFpath)
    mkdir(saveFolder4eval_FFpath);
end


imgCount = 0;
cityList = dir(path_to_folder);
cityList = cityList(3:end);
for cityIdx = 1:length(cityList)
    imgList = dir(fullfile(path_to_folder, cityList(cityIdx).name, '*png'));
    for testImgIdx = 1:length(imgList)
        curImgName = fullfile(path_to_folder, cityList(cityIdx).name, imgList(testImgIdx).name);
        
        imOrg = imread(curImgName); % read image
        imFeed = single(imOrg); % convert to single precision
        imFeed = bsxfun(@minus, imFeed, mean_rgb);       
        %%
        imgCount = imgCount + 1;
        fprintf('NO.%03d image-%03d %s ... \n', imgCount, testImgIdx, imgList(testImgIdx).name);
        
        inputs = {'data', gpuArray(imFeed)};
        netMat.eval(inputs) ;
        
        
        SoftMaxLayer = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer')).outputIndexes).value);
        [probMapFF, predMapFF] = max(SoftMaxLayer,[],3);
        [output_softmaxFF, evalLabelMapFF] = index2RGBlabel(predMapFF-1, colorLabel, classID);        
%         figure; imshow(uint8(output_softmaxFF)); title('output_softmaxFF');
        
        %% save results
        [~, curImgName, ~] = fileparts(curImgName);
        
        a = strfind(curImgName, '_');
        curImgName = curImgName(1:a(3)-1);
        
        imwrite(uint8(evalLabelMapFF), sprintf('%s/%s_predEval.png', saveFolder4eval_FFpath, curImgName) );
    end
end
%% leaving blank
%{

classes          IoU      nIoU
--------------------------------
road          : 0.983      nan
sidewalk      : 0.861      nan
building      : 0.927      nan
wall          : 0.545      nan
fence         : 0.631      nan
pole          : 0.648      nan
traffic light : 0.717      nan
traffic sign  : 0.801      nan
vegetation    : 0.926      nan
terrain       : 0.629      nan
sky           : 0.946      nan
person        : 0.823    0.666
rider         : 0.623    0.484
car           : 0.949    0.876
truck         : 0.693    0.432
bus           : 0.821    0.629
train         : 0.743    0.551
motorcycle    : 0.614    0.436
bicycle       : 0.779    0.613
--------------------------------
Score Average : 0.772    0.586
--------------------------------


categories       IoU      nIoU
--------------------------------
flat          : 0.986      nan
nature        : 0.929      nan
object        : 0.717      nan
sky           : 0.946      nan
construction  : 0.934      nan
human         : 0.837    0.694
vehicle       : 0.940    0.855
--------------------------------
Score Average : 0.898    0.775
--------------------------------

%}
