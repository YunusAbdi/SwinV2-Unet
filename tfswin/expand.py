import math
import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.norm import LayerNorm
from tensorflow.nn import depth_to_space


@register_keras_serializable(package='TFSwinV2')
class PatchExpanding(layers.Layer):
    def __init__(self, swin_v2=False, name = '', upsample_rate=2,return_vector=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.return_vector = return_vector
        self.swin_v2 = swin_v2
        self.upsample_rate = upsample_rate


    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[-1]
        self.H = input_shape[1]
        self.W = input_shape[2]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')
        
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels})
        
        # noinspection PyAttributeOutsideInit
        self.norm = LayerNorm(name='norm')
        # Linear transformations that doubles the channels 
        self.linear_trans1 = layers.Conv2D(self.upsample_rate*self.channels, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(self.name))
        # 
        #self.linear_trans2 = layers.Conv2D(self.upsample_rate*self.channels, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        self.prefix = self.name


        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        _, H, W, C = inputs.get_shape().as_list()
        x = self.linear_trans1(inputs)
        
        x = depth_to_space(x, self.upsample_rate, data_format='NHWC', name='{}_d_to_space'.format(self.prefix))
        
        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1, H*W*self.upsample_rate*self.upsample_rate, C//2))

        return x

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        def _scale(value):
            return None if value is None else math.ceil(value * 2)

        return input_shape[0], _scale(input_shape[1]), _scale(input_shape[2]), self.channels / 2

    def get_config(self):
        config = super().get_config()
        config.update({'swin_v2': self.swin_v2,
        'name': self.name, 
        'return_vector':self.return_vector,
        'upsample_rate':self.upsample_rate
        })

        return config
