import tensorflow as tf;

from ..sa_net_arch import AbstractCNNArch;
from ..sa_net_arch_utilities import CNNArchUtils;
from ..sa_net_optimizer import OptimizerTypes, CNNOptimizer;
from ..sa_net_data_provider import AbstractDataProvider;
import os;
import numpy as np;

class ClassifierTesterExternalInput:
    def __init__(self, cnn_arch:AbstractCNNArch, session_config, output_dir, output_ext, kwargs):
        # predefined list of arguments

        self.cnn_arch = cnn_arch;
        #self.cost_func = cost_func;
        #self.test_data_provider = test_data_provider;
        if(session_config == None):
            self.session_config = tf.ConfigProto();
            self.session_config.gpu_options.allow_growth = True
        else:
            self.session_config = session_config;
        self.output_dir = output_dir;
        self.output_ext = output_ext;

        self.init = tf.global_variables_initializer();

    def init_model(self, do_init, do_restore):
        self.sess = tf.Session(config=self.session_config);
        with self.sess.as_default():
            if(do_init):
                self.sess.run(tf.global_variables_initializer());
                #sess.run(self.init);
            if(do_restore):
                self.cnn_arch.restore_model(self.sess);


    def predict(self, inputs):
        with self.sess.as_default():
            batch_x = inputs;
            if (batch_x is None):
                return None;
            batch_x = tf.image.resize_images(batch_x, (self.cnn_arch.input_img_height, self.cnn_arch.input_img_width));
            batch_x = batch_x.eval()

            batch_y = self.sess.run([self.cnn_arch.logits] \
                , feed_dict={self.cnn_arch.input_x: batch_x \
                    , self.cnn_arch.isTest: True \
                });
            #print(np.array(batch_y).shape)
            #print(np.array(batch_y))
            #print(np.array(batch_y)[...,-1])
            batch_y_sig = self.sigmoid(np.array(batch_y)[...,-1]);
            return batch_y_sig;

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))