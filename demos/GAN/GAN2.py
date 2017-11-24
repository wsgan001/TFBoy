# generative network
# use multi-layer percepton to generate time sequence from random noise
# input tensor must be in shape of (batch_size, self.seq_len)
def generator(self, inputTensor):
    with tf.name_scope('G_net'):
        gInputTensor = tf.identity(inputTensor, name='input')
        # Multilayer percepton implementation
        numNodesInEachLayer = 10
        numLayers = 3

        previous_output_tensor = gInputTensor
        for layerIdx in range(numLayers):
            activation, z = self.fullConnectedLayer(previous_output_tensor, numNodesInEachLayer, layerIdx)
            previous_output_tensor = activation

        g_logit = z
        g_logit = tf.identity(g_logit, 'g_logit')
        return g_logit


