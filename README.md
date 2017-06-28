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

Here is to fast test the cityscapes script -- 
You may want to link your dataset here (not sure? Just see errors if you get some:)
```
cd videoSeg/deepFF/cityscapesscripts/evaluation/
./evalPixelLevelSemanticLabeling.py main002_v1_ftstdModelv3_softmax_net-epoch-84_FFpath_predEval/
```


Shu Kong @ UCI
06/28/2017
