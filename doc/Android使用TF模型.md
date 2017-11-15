以下代码来自于TensorFlowObjectDetectionAPIModel.java

Android调用Tensorflow模型主要通过一个类：TensorFlowInferenceInterface
通过传入assetManager(要从asset读pb文件)，和modelFilename(模型名)实例化这个类

```java
d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
```

有了这个实例就可以调用TF相关的方法

```java
	//获取graph实例
    final Graph g = d.inferenceInterface.graph();

    d.inputName = "image_tensor";
    final Operation inputOp = g.operation(d.inputName);
    if (inputOp == null) {
      throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
    }
    ...
    final Operation outputOp1 = g.operation("detection_scores");
    if (outputOp1 == null) {
      throw new RuntimeException("Failed to find output Node 'detection_scores'");
    }

```

上面是我截取的一部分代码，简单介绍一下：

Graph是TF中的图，图是由operation和tensor构成，operation可以看做是图里面的节点，tensor就是连接节点的线。所以要进行对operation进行操作就必须有一个Graph对象。

```java
d.inputName = "image_tensor";
final Operation inputOp = g.operation(d.inputName);
```

这里给一个inputName赋值image_tensor，这个值我开始以为是operation需要命名所以任意给了一个标识名，方便后面查找，但发现这个值是不能改的，改了会出错。从代码可以看到，对于所有的operation对象都会有一个非空判断，因为这个op是和模型中训练时候生成的图对应的，获取实例的时候接口会去模型中查找这个节点，也就是这个op。所以使用模型的时候，必须要知道这个模型的输入输出节点。

为什么是输入输出节点，因为训练模型生成的图是很大的，我用代码(我放在Tests目录下了)把ssd_mobilenet_v1_android_export.pb模型所有op打出来，发现一共有5000多个，所以说这个图的中间节点有非常多。而有用的，目前从代码来看，就是一个输入节点（输入图像的tensor），4个输出节点（输出：分类，准确度分数，识别物体在图片中的位置用于画框，和num_detections）。所以单纯地使用模型，我认为知道模型这几个节点就可以了。

> 这里推荐一篇文章[TensorFlow固定图的权重并储存为Protocol Buffers](https://www.ouyangsong.com/2017/05/23/tensorflow-freeze-model-protocolbuffers/)
讲的是Tensorflow保存的模型中都由哪些东西组成的。

知道这几个节点的名称，就可以实例化这些节点，然后就对节点操作：

```java
	bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    //????
    for (int i = 0; i < intValues.length; ++i) {
      byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
      byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
      byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
    }

    // Copy the input data into TensorFlow.
    //给inputname节点operation的tensor赋值  feed里有一个Tensor.create  创建张量
    inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);

    // Run the inference call.
    // 运行output operations
    inferenceInterface.run(outputNames, logStats);

    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");
    outputLocations = new float[MAX_RESULTS * 4];
    outputScores = new float[MAX_RESULTS];
    outputClasses = new float[MAX_RESULTS];
    outputNumDetections = new float[1];
    // 从tensor的operation中取值
    inferenceInterface.fetch(outputNames[0], outputLocations);
    inferenceInterface.fetch(outputNames[1], outputScores);
    inferenceInterface.fetch(outputNames[2], outputClasses);
    inferenceInterface.fetch(outputNames[3], outputNumDetections);
```

上面代码有几个方法：
首先是通过getPixels把图片转换成数组，其实就是张量，也就是Tensor，Tensor的形式就是这样任意维度的数组，可以看做是矩阵
之后它对这个数组做了一次处理，这里对图像数据的处理我没看明白。。

然后，使用feed方法把tensor传给operation，参数里inputName其实就是用来定位operation的。数据传给input，后面只要对output做一次处理：inferenceInterface.run(outputNames, logStats);这里第一个参数outputNames是一个数组，包含了所有用来output的operation的名称。
最最后，通过inferenceInterface.fetch方法获取每个output operation输出的结果。

这里还有一点，为什么run方法是作用在output operation的？
是因为，tensorflow生成graph后，不会直接运行，因为Graph会有很多条通路，只有在对输出的operation进行run之后，graph才会从output operation开始，反向查找运行的前置条件，只到完成通路才会执行。也就是说：Graph的很多通路不一定都会执行。

最后再提一下label文件，因为label是和图像对应的，资源文件中也有记录着所有训练labels的文件，那么它用在哪？

```java
// Find the best detections.
    final PriorityQueue<Recognition> pq =
        new PriorityQueue<Recognition>(
            1,
            new Comparator<Recognition>() {
              @Override
              public int compare(final Recognition lhs, final Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });

    // Scale them back to the input size.
    for (int i = 0; i < outputScores.length; ++i) {
      final RectF detection =
          new RectF(
              outputLocations[4 * i + 1] * inputSize,
              outputLocations[4 * i] * inputSize,
              outputLocations[4 * i + 3] * inputSize,
              outputLocations[4 * i + 2] * inputSize);
      pq.add(new Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], detection));
```

label用在最后一行的 labels.get((int) outputClasses[i])
labels就是保存文件中所有label的数组，outputClasses就是上个代码段中output输出的内容。
这个代码段只是把输出结果保存成Recognition对象，然后按照outputScore进行排序，最可能的值排最前面输出。所以我是这么理解的：label数据在模型中就已经存在了，因为pb文件不仅存储了graph，还存储了训练过程的信息。labels文件对我们来说就是为了获得结果。


总结
1. 使用inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);实例化TF入口类
2. 通过TF入口实例化graph,Graph g = d.inferenceInterface.graph();
3. 用g.operation(name)检查输入输出的operation是否存在
4. 把输入数据转换成数组(Tensor)形式，比如图片：bitmap.getPixels(intValues...)
5. 把输入数据喂给输入operation  inferenceInterface.feed()
6. run输出operations inferenceInterface.run()
7. 用fetch获取结果inferenceInterface.fetch()