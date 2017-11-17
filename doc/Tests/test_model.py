from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
    file_name = "./test.jpg"
    model_file = "./ssd_mobilenet_v1_android_export.pb"
    label_file = "./coco_labels_list.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "image_tensor"
    output_layer = "detection_classes"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    # print(input_operation.outputs)
    # print(input_operation.outputs[0])
    output_operation = graph.get_operation_by_name(output_name);
    # print(output_operation.outputs)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i], results[i])



    # with tf.Graph().as_default():
    #     output_graph_def = tf.GraphDef()
    #
    # with open("./ssd_mobilenet_v1_android_export.pb", "rb") as f:
    #     output_graph_def.ParseFromString(f.read())
    #     _ = tf.import_graph_def(output_graph_def, name="")
    #
    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #
    #     op_input = sess.graph.get_operation_by_name("image_tensor")
    #     if not (op_input is None):
    #         print(op_input)
    #
    #     op_output1 = sess.graph.get_operation_by_name("detection_scores")
    #     if not (op_output1 is None):
    #         print(op_output1)
    #
    #     op_output2 = sess.graph.get_operation_by_name("detection_boxes")
    #     if not (op_output2 is None):
    #         print(op_output2)
    #
    #     op_output3 = sess.graph.get_operation_by_name("detection_classes")
    #     if not (op_output3 is None):
    #         print(op_output3)
    #
    #     img = io.imread("./test.jpg")
    #     img = transform.resize(img, (224, 224, 3))
    #
    #     print(sess.run(op_input, feed_dict={Placeholder:img}))
    #     print(sess.run(op_output3))