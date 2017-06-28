function [stats, prof] = cityscapes_processEpoch(net, state, scalingFactor, opts, mode, totalEpoch)
% -------------------------------------------------------------------------

%% initialize empty momentum
if strcmp(mode,'train')
    state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

%% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net.move('gpu') ;
    if strcmp(mode,'train')
        state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
    end
end
if numGpus > 1
    mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
    mmap = [] ;
end

%% profile
if opts.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

subset = state.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;

start = tic ;
for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d/%03d: %3d/%3d:', mode, state.epoch, totalEpoch, ...
        fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
%         scalingFactor = log2(scalingFactor)-2;
%         scalingFactor = 2^(scalingFactor);
        
        [image, arrayGT_class, arrayGT_id, arrayEdge, arrayMask, arrayGT_color, imo, dataMat, arrayGT_depthID, arrayGT_depthClass] = state.getBatch(batch, mode, scalingFactor) ;         
        %% double check the data
        %{
        labelDict = load('labelDictionary.mat');
        validIndex = find(labelDict.ignoreInEval==0);
        colorLabel = labelDict.colorLabel(validIndex,:);
        classID = labelDict.classID(validIndex);
        [~, classID4scoreMap] = max(arrayGT_class, [], 3);
        [ accumulateScore, evalLabelMap ] = index2RGBlabel(classID4scoreMap-1, colorLabel, classID);
        accumulateScore = bsxfun(@times, accumulateScore, arrayMask);
        
        evalLabelMap = evalLabelMap.*arrayMask;
        classID4scoreMap = classID4scoreMap .* arrayMask;
        
        figure(2); 
        subplot(1,2,1); imshow(uint8(arrayGT_color)); 
        subplot(1,2,2); imshow(uint8(accumulateScore));
        
        figure(3); 
        subplot(1,3,1); imagesc(arrayGT_id); axis off image;
        subplot(1,3,2); imagesc(classID4scoreMap); axis off image;
        subplot(1,3,3); imagesc(evalLabelMap); axis off image;
        
        norm(classID4scoreMap(:)-arrayGT_id(:), 'fro')
        norm(accumulateScore(:)-arrayGT_color(:), 'fro')                
        %}
        %% train/test        
%         figure(1); subplot(2,3,1); imagesc(uint8(imo)); axis off image;
        if opts.withDepth 
            arrayGT_depthClass = imresize(arrayGT_depthClass, 1/8);
            A = sum(arrayGT_depthClass, 3);
            arrayGT_depthClass = bsxfun(@rdivide, arrayGT_depthClass, (A+(A==0)) );
            inputs = {'depthClass', gpuArray(arrayGT_depthClass)};
        else
            inputs = {};
        end
        
        if ~isnan(net.getLayerIndex('depthEstSoftmaxloss'))
            inputs{end+1} = 'depthID';
            inputs{end+1} = gpuArray(arrayGT_depthID);
        end
        
        
        if numGpus >= 1
            image = gpuArray(image) ;
            inputs{end+1} = 'data';
            inputs{end+1} = image;            
            for scale_i = 1:length(scalingFactor)
                curScaleFactor = scalingFactor(scale_i);
                gt_name =  sprintf('gt_div%d_seg', curScaleFactor);
                inputs{end+1} = gt_name;
                if strcmp(net.layers(end).block.loss, 'softmaxlog') || strcmp(net.layers(end).block.loss, 'softmaxlogbootstrap')
                    inputs{end+1} = gpuArray(arrayGT_id);
                else
                    inputs{end+1} = gpuArray(arrayGT_class);
                end
            end 
        else
%             inputs = {'data', image};
            inputs = {'data', image, 'depthClass', arrayGT_depthClass};
            for scale_i = 1:length(scalingFactor)
                curScaleFactor = scalingFactor(scale_i);
                gt_name =  sprintf('gt_div%d_seg', curScaleFactor);
                inputs{end+1} = gt_name;
                if strcmp(net.layers(end).block.loss, 'softmaxlog')|| strcmp(net.layers(end).block.loss, 'softmaxlogbootstrap')
                    inputs{end+1} = gpuArray(arrayGT_id);
                else
                    inputs{end+1} = gpuArray(arrayGT_class);
                end
            end 
        end
        
        if opts.prefetch
            if s == opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            state.getBatch(nextBatch, mode, scalingFactor) ;
%             state.getBatch(state.imdb, nextBatch) ;
        end
        
        if strcmp(mode, 'train')
            net.mode = 'normal' ;
            net.accumulateParamDers = (s ~= 1) ;            
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
%             net.vars(outputIdx).precious = 1;            
%             net.eval(inputs, opts.derOutputs) ;
            net.eval(inputs, opts.derOutputs, 'backPropAboveLayerName', opts.backPropAboveLayerName) ;
           %{
            img = inputs{2};
            img = bsxfun(@plus, img, net.meta.normalization.averageImage);
            
            labelDict = load('labelDictionary.mat');
            validIndex = find(labelDict.ignoreInEval==0);
            colorLabel = labelDict.colorLabel(validIndex,:);       
            
            pred = gather(net.vars(outputIdx).value);
            [~, pred] = max(pred, [], 3);
            pred = index2RGBlabel(pred-1, colorLabel);
            
            gt = inputs{4};
%             [~, gt] = max(gt, [], 3);
            gt = index2RGBlabel(gt-1, colorLabel);
            
            figure; 
            subplot(1,3,1); imshow(uint8(img)); title('image');
            subplot(1,3,2); imshow(uint8(gt)); title('gt');
            subplot(1,3,3); imshow(uint8(pred)); title('pred');
            disp('done');
            %}
        else
            net.mode = 'test' ;
%             net.mode = 'normal' ;
%             net.conserveMemory = 0;
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
%             net.vars(outputIdx).precious = 1;
            net.eval(inputs) ;
            %{
            img = inputs{2};
            img = bsxfun(@plus, img, net.meta.normalization.averageImage);
            
            labelDict = load('labelDictionary.mat');
            validIndex = find(labelDict.ignoreInEval==0);
            colorLabel = labelDict.colorLabel(validIndex,:);       
            
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
            pred = gather(net.vars(outputIdx).value);
            [~, pred] = max(pred, [], 3);
            pred = index2RGBlabel(pred-1, colorLabel);
            
            gt = inputs{4};
%             [~, gt] = max(gt, [], 3);
            gt = index2RGBlabel(gt-1, colorLabel);
            
            figure; 
            subplot(1,3,1); imshow(uint8(img)); title('image');
            subplot(1,3,2); imshow(uint8(gt)); title('gt');
            subplot(1,3,3); imshow(uint8(pred)); title('pred');
            disp('done');
            %}
        end
    end
    
    % accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            write_gradients(mmap, net) ;
            labBarrier() ;
        end
        state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
    end
    
    % get statistics
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats = opts.extractStatsFn(net) ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == opts.batchSize + 1
        % compensate for the first iteration, which is an outlier
        adjustTime = 2*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s:', f) ;
        fprintf(' %.4f', stats.(f)) ;
    end
    fprintf('\n') ;
end

if ~isempty(mmap)
    unmap_gradients(mmap) ;
end

if opts.profile
    if numGpus <= 1
        prof = profile('info') ;
        profile off ;
    else
        prof = mpiprofile('info');
        mpiprofile off ;
    end
else
    prof = [] ;
end

net.reset() ;
net.move('cpu') ;
