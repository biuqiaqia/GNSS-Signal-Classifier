# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for N-D convolutional LSTM layers."""


# import tensorflow.compat.v2 as tf

# from tf_keras import activations
# from tf_keras import backend
# from tf_keras import constraints
# from tf_keras import initializers
# from tf_keras import regularizers
# from tf_keras.engine import base_layer
# from tf_keras.layers.rnn.base_conv_rnn import ConvRNN
# from tf_keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
# from tf_keras.utils import conv_utils

# ConvRNN 不存在于公开 API，需手动实现或继承 RNN
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import RNN


class ConvLSTMCell(Layer):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rank = rank
        if self.rank > 3:
            raise ValueError(
                f"Rank {rank} convolutions are not currently "
                f"implemented. Received: rank={rank}"
            )
        
        self.filters = filters
        self.kernel_size = self._normalize_tuple(kernel_size, rank, "kernel_size")

        self.strides = self._normalize_tuple(strides, rank, "strides", allow_zero=True)
        
        self.padding = self._normalize_padding(padding)
        
        self.data_format = self._normalize_data_format(data_format)
        
        self.dilation_rate = self._normalize_tuple(dilation_rate, rank, "dilation_rate")
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.state_size = (self.filters, self.filters)

    def _normalize_tuple(self, value, n, name, allow_zero=False):
        if isinstance(value, (list, tuple)):
            if len(value) != n:
                raise ValueError(
                    f"The `{name}` must be a tuple of {n} integers. Received: {value}"
                )
            return tuple(value)
        elif allow_zero and value == 0:
            return (0,) * n
        else:
            return (value,) * n

    def _normalize_padding(self, padding):
        if isinstance(padding, str):
            padding = padding.lower()
            if padding not in {"valid", "same"}:
                raise ValueError(
                    f"Invalid padding: '{padding}'. Must be one of 'valid' or 'same'."
                )
            return padding
        else:
            raise TypeError("Padding must be a string.")

    def _normalize_data_format(self, data_format):
        if data_format is None:
            data_format = tf.keras.backend.image_data_format()
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(
                f"Invalid data_format: '{data_format}'. Must be one of 'channels_last' or 'channels_first'."
            )
        return data_format

    def build(self, input_shape):
            super().build(input_shape)
            if self.data_format == "channels_first":
                channel_axis = 1
            else:
                channel_axis = -1
            if input_shape[channel_axis] is None:
                raise ValueError(
                    "The channel dimension of the inputs (last axis) should be "
                    "defined. Found None. Full input shape received: "
                    f"input_shape={input_shape}"
                )
            input_dim = input_shape[channel_axis]
            self.kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
            recurrent_kernel_shape = self.kernel_size + (
                self.filters,
                self.filters * 4,
            )

            self.kernel = self.add_weight(
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
            self.recurrent_kernel = self.add_weight(
                shape=recurrent_kernel_shape,
                initializer=self.recurrent_initializer,
                name="recurrent_kernel",
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint,
            )

            if self.use_bias:
                if self.unit_forget_bias:

                    def bias_initializer(_, *args, **kwargs):
                        return backend.concatenate(
                            [
                                self.bias_initializer(
                                    (self.filters,), *args, **kwargs
                                ),
                                initializers.get("ones")(
                                    (self.filters,), *args, **kwargs
                                ),
                                self.bias_initializer(
                                    (self.filters * 2,), *args, **kwargs
                                ),
                            ]
                        )

                else:
                    bias_initializer = self.bias_initializer
                self.bias = self.add_weight(
                    shape=(self.filters * 4,),
                    name="bias",
                    initializer=bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )
            else:
                self.bias = None
    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # 应用掩码（如果存在）
        if mask is not None:
            mask = tf.cast(mask, h_tm1.dtype)
            mask = tf.expand_dims(mask, axis=-1)  # 扩展维度匹配 h_tm1 形状
            h_tm1 = h_tm1 * mask
            c_tm1 = c_tm1 * mask

        # dropout matrices for input units
        if training:
            if 0 < self.dropout < 1.0:
                dp_mask = tf.nn.dropout(tf.ones_like(inputs), rate=self.dropout)
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            if 0 < self.recurrent_dropout < 1.0:
                rec_dp_mask = tf.nn.dropout(tf.ones_like(h_tm1), rate=self.recurrent_dropout)
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        # 卷积计算
        x_i = self.input_conv(inputs_i, self.kernel_i, self.bias_i, padding=self.padding)
        x_f = self.input_conv(inputs_f, self.kernel_f, self.bias_f, padding=self.padding)
        x_c = self.input_conv(inputs_c, self.kernel_c, self.bias_c, padding=self.padding)
        x_o = self.input_conv(inputs_o, self.kernel_o, self.bias_o, padding=self.padding)

        h_i = self.recurrent_conv(h_tm1_i, self.recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, self.recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, self.recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, self.recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        return h, [h, c]
    
class ConvLSTM(RNN):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        dropout=0.0,
        recurrent_dropout=0.0,
        **kwargs
    ):
        cell = ConvLSTMCell(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get("dtype"),
            name="conv_lstm_cell"
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs
        )
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state
        )

    def get_config(self):
        config = {
            "filters": self.cell.filters,
            "kernel_size": self.cell.kernel_size,
            "strides": self.cell.strides,
            "padding": self.cell.padding,
            "data_format": self.cell.data_format,
            "dilation_rate": self.cell.dilation_rate,
            "activation": tf.keras.activations.serialize(self.cell.activation),
            "recurrent_activation": tf.keras.activations.serialize(self.cell.recurrent_activation),
            "use_bias": self.cell.use_bias,
            "kernel_initializer": tf.keras.initializers.serialize(self.cell.kernel_initializer),
            "recurrent_initializer": tf.keras.initializers.serialize(self.cell.recurrent_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.cell.bias_initializer),
            "unit_forget_bias": self.cell.unit_forget_bias,
            "kernel_regularizer": tf.keras.regularizers.serialize(self.cell.kernel_regularizer),
            "recurrent_regularizer": tf.keras.regularizers.serialize(self.cell.recurrent_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.cell.bias_regularizer),
            "activity_regularizer": tf.keras.regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self.cell.kernel_constraint),
            "recurrent_constraint": tf.keras.constraints.serialize(self.cell.recurrent_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self.cell.bias_constraint),
            "dropout": self.cell.dropout,
            "recurrent_dropout": self.cell.recurrent_dropout,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, inputs_shape):
        super().build(inputs_shape)

    def compute_output_shape(self, inputs_shape):
        return super().compute_output_shape(inputs_shape)
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

    @property
    def _keras_api_names_v1(self):
        return []

    @property
    def _keras_api_names(self):
        return []
    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    
    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(
                self.recurrent_activation
            ),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "unit_forget_bias": self.unit_forget_bias,
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# masked_convLSTM2D_module.py 的最后
class MaskedConvLSTM2D(ConvLSTM):
    def __init__(self, **kwargs):
        super().__init__(rank=2, **kwargs)