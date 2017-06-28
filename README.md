# videoSeg

MatConvNet is used in our project, some functions are changed. So it might be required to re-compile. Useful commands are --

```python
LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab 

path_to_matconvnet = '../matconvnet';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

```

The basse model is trained as a deep feed-forward network, as seen in folder 'deepFF'. Simply running 'main002_v1_ftstdModelv3.m' will give you an idea how it trains. Running 'main003_genPred4eval.m' will save the results of validation set for evaluation. Here is to fast test the cityscapes script -- 
You should download the model from the shared google drive. You may also want to link your dataset here (not sure? Just see errors if you get some:)
```
cd ./deepFF/cityscapesscripts/evaluation/
./evalPixelLevelSemanticLabeling.py main002_v1_ftstdModelv3_softmax_net-epoch-84_FFpath_predEval/
```


Shu Kong @ UCI
06/28/2017
