### Faster R-CNN 和 VOC 数据集

#### 制作VOC数据

##### 修改文件名

因为VOC文件名都是使用6位数字,为了适应代码,所以需要格式化文件名

文件改名脚本:

```bash
#!/bin/bash
index=1000000;
for f in *.png;
do
    mv "$f" ${index#1}.png
    ((index++))
done
```

VOC代码处理的是`jpg`格式的文件,所以还需要把`png`转换成`jpg`:

```bash
bash -c 'for image in *.png; do convert "$image" "${image%.png}.jpg"; echo “image $image converted to ${image%.png}.jpg ”; done'
```

看了pascal_voc.py代码,可以把代码的`jpg`拼接改成`png`,这样可以不做上一步.

##### 标记图片:

依然使用`labelImg`工具,生成对应的`xml`文件.

然后把

1. 所有图片放到__/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages__中
2. 所有`xml`文件放到__/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations__中
3. 修改__tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main__下的4个文件:
   * test.txt	测试图片名称,数量占总图片数量的50%
   * trainval.txt    训练/验证图片名称,数量占总图片数量的50%
   * train.txt    训练图片名称,数量占 训练/验证(上一条) 总数的50%
   * val.txt    验证图片名称,数量占 训练/验证总数的50%

最后修改__tf-faster-rcnn/lib/datasets/pascal_voc.py__,把`self._classes`定义的类别填入我们自己要识别的类别.

到此,自己的VOC数据集就可以使用了.

#### Faster RCNN使用

##### 安装

```bash
git clone https://github.com/endernewton/tf-faster-rcnn.git
```

##### 修改配置,使支持CPU选项

* tf-faster-rcnn/lib/model/nms_wrapper.py 修改/注释以下行:

  ```python
  #from nms.gpu_nms import gpu_nms
  def nms(dets, thresh, force_cpu=False):
     #Dispatch to either CPU or GPU NMS implementations.

      if dets.shape[0] == 0:
          return []
      if cfg.USE_GPU_NMS and not force_cpu:
          #return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
          return cpu_nms(dets, thresh)
      else:
          return cpu_nms(dets, thresh)
  ```

* tf-faster-rcnn/lib/model/config.py 修改代码为:

  ```python
  __C.USE_GPU_NMS = False
  ```

* tf-faster-rcnn/lib/setup.py 注释下面代码:

  ```python
  CUDA = locate_cuda()
  self.src_extensions.append('.cu')
  Extension('nms.gpu_nms',
          ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
          library_dirs=[CUDA['lib64']],
          libraries=['cudart'],
          language='c++',
          runtime_library_dirs=[CUDA['lib64']],
          # this syntax is specific to this build system
          # we're only going to use certain compiler args with nvcc and not with gcc
          # the implementation of this trick is in customize_compiler() below
          extra_compile_args={'gcc': ["-Wno-unused-function"],
                              'nvcc': ['-arch=sm_52',
                                       '--ptxas-options=-v',
                                       '-c',
                                       '--compiler-options',
                                       "'-fPIC'"]},
          include_dirs = [numpy_include, CUDA['include']]
  ```

##### 编译Cython

```bash
cd lib
make clean
make
cd ..
```

##### 安装python coco API

```bash
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```

##### 下载预训练模型

```bash
./data/scripts/fetch_faster_rcnn_models.sh
```

##### 测试

```bash
./tools/demo.py
```

会对__/data/demo/__目录下的图片进行识别.如果需要识别其他图片,修改demo.py里的im_names列表.

##### 训练自己的数据

下载预训练的模型,目前支持`VGG16`和`Resnet V1`

```bash
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../..

mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..
```

训练/验证

```bash
./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# GPU_ID is the GPU you want to test on
# NET in {vgg16, res50, res101, res152} is the network arch to use
# DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
# Examples:
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/train_faster_rcnn.sh 1 coco res101

./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# GPU_ID is the GPU you want to test on
# NET in {vgg16, res50, res101, res152} is the network arch to use
# DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
# Examples:
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/test_faster_rcnn.sh 1 coco res101
```

每次训练前需要把

output和data/cache目录删掉

训练次数需要在训练脚本中修改.它的训练不像SSD可以随时中断,然后接着之前的训练.需要一次训练好才会生成文件.后续需要添加

训练次数在

train_faster_rcnn.sh 和test_faster_rcnn.sh 里修改



##### 错误解决:

运行test的时候:

```bash
Loading model check point from output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_300.ckpt
Loaded.
im_detect: 1/35 3.285s 0.001s
im_detect: 2/35 3.619s 0.001s
im_detect: 3/35 4.309s 0.001s
im_detect: 4/35 4.635s 0.001s
im_detect: 5/35 4.665s 0.001s
im_detect: 6/35 4.506s 0.001s
im_detect: 7/35 4.561s 0.001s
im_detect: 8/35 4.713s 0.001s
im_detect: 9/35 4.776s 0.001s
im_detect: 10/35 4.750s 0.001s
im_detect: 11/35 4.787s 0.001s
im_detect: 12/35 4.761s 0.001s
im_detect: 13/35 4.721s 0.001s
im_detect: 14/35 4.788s 0.001s
im_detect: 15/35 4.861s 0.001s
im_detect: 16/35 4.831s 0.001s
im_detect: 17/35 4.848s 0.001s
im_detect: 18/35 4.793s 0.001s
im_detect: 19/35 4.791s 0.001s
im_detect: 20/35 4.755s 0.001s
im_detect: 21/35 4.795s 0.001s
im_detect: 22/35 4.829s 0.001s
im_detect: 23/35 4.873s 0.001s
im_detect: 24/35 4.822s 0.001s
im_detect: 25/35 4.801s 0.001s
im_detect: 26/35 4.787s 0.001s
im_detect: 27/35 4.758s 0.001s
im_detect: 28/35 4.747s 0.001s
im_detect: 29/35 4.717s 0.001s
im_detect: 30/35 4.725s 0.001s
im_detect: 31/35 4.754s 0.001s
im_detect: 32/35 4.759s 0.001s
im_detect: 33/35 4.775s 0.001s
im_detect: 34/35 4.786s 0.001s
im_detect: 35/35 4.796s 0.001s
Evaluating detections
Writing knife VOC results file
Traceback (most recent call last):
  File "./tools/test_net.py", line 120, in <module>
    test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)
  File "/home/wow/Github/tf-faster-rcnn/tools/../lib/model/test.py", line 192, in test_net
    imdb.evaluate_detections(all_boxes, output_dir)
  File "/home/wow/Github/tf-faster-rcnn/tools/../lib/datasets/pascal_voc.py", line 278, in evaluate_detections
    self._write_voc_results_file(all_boxes)
  File "/home/wow/Github/tf-faster-rcnn/tools/../lib/datasets/pascal_voc.py", line 205, in _write_voc_results_file
    with open(filename, 'wt') as f:
IOError: [Errno 2] No such file or directory: '/home/wow/Github/tf-faster-rcnn/data/VOCdevkit2007/results/VOC2007/Main/comp4_c412401c-8e25-43ab-ba7f-e65b9089897c_det_test_knife.txt'
Command exited with non-zero status 1
590.78user 10.24system 2:51.10elapsed 351%CPU (0avgtext+0avgdata 2778584maxresident)k

```

查看pascal_voc.py代码,发现这里```with open(filename, 'wt') as f```代码应该会创建文件,只是因为没有results目录,所以手动创建tf-faster-rcnn/data/VOCdevkit2007/results/VOC2007/Main/目录即可.



测试通过后进行对图片的识别,识别依然使用`./tools/demo.py`:

```bash
./tools/demo.py --net=vgg16 --dataset=pascal_voc
```

得到如下错误:

```bash
2017-12-26 17:04:58.521558: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
begin to restore
Traceback (most recent call last):
  File "./tools/demo.py", line 141, in <module>
    saver.restore(sess, tfmodel)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/training/saver.py", line 1666, in restore
    {self.saver_def.filename_tensor_name: save_path})
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 889, in run
    run_metadata_ptr)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1120, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1317, in _do_run
    options, run_metadata)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1336, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [84] rhs shape= [16]
	 [[Node: save/Assign = Assign[T=DT_FLOAT, _class=["loc:@vgg_16/bbox_pred/biases"], use_locking=true, validate_shape=true, _device="/job:localhost/replica:0/task:0/device:CPU:0"](vgg_16/bbox_pred/biases, save/RestoreV2)]]

Caused by op u'save/Assign', defined at:
  File "./tools/demo.py", line 139, in <module>
    saver = tf.train.Saver()
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/training/saver.py", line 1218, in __init__
    self.build()
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/training/saver.py", line 1227, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/training/saver.py", line 1263, in _build
    build_save=build_save, build_restore=build_restore)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/training/saver.py", line 751, in _build_internal
    restore_sequentially, reshape)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/training/saver.py", line 439, in _AddRestoreOps
    assign_ops.append(saveable.restore(tensors, shapes))
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/training/saver.py", line 160, in restore
    self.op.get_shape().is_fully_defined())
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/ops/state_ops.py", line 276, in assign
    validate_shape=validate_shape)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/ops/gen_state_ops.py", line 57, in assign
    use_locking=use_locking, name=name)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 2956, in create_op
    op_def=op_def)
  File "/home/wow/ML/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 1470, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): Assign requires shapes of both tensors to match. lhs shape= [84] rhs shape= [16]
	 [[Node: save/Assign = Assign[T=DT_FLOAT, _class=["loc:@vgg_16/bbox_pred/biases"], use_locking=true, validate_shape=true, _device="/job:localhost/replica:0/task:0/device:CPU:0"](vgg_16/bbox_pred/biases, save/RestoreV2)]]
```

这个错误推测是之前训练的cache没有清空导致模型数据不匹配.删除/data/cache和/output,重新训练.

又遇到类似的错误:

```bash
tensorflow.python.framework.errors_impl.InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [4096,21] rhs shape= [4096,4]
```

这次的shape和上一次的不一样.通过阅读demo.py代码,发现这一行定义了21:

```python
    net.create_architecture("TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
```

把21改成4,成功运行.



Reference:

https://github.com/smallcorgi/Faster-RCNN_TF

https://github.com/endernewton/tf-faster-rcnn