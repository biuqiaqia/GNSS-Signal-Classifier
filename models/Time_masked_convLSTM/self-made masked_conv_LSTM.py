# self-made masked_conv_LSTM
# this is a try as the downloaded model is hard to use
# hope I can make it work
import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend

class ConvLSTMCell(Layer):
    def __init__(
        self,
        rank,  # 卷积的维度(1D, 2D或3D)
        filters,  # 输出空间的维度(卷积滤波器的数量)
        kernel_size,  # 卷积窗口的大小
        strides=1,  # 卷积的步长
        padding="valid",  # 填充方式("valid"或"same")
        data_format=None,  # 数据格式("channels_last"或"channels_first")
        dilation_rate=1,  # 膨胀卷积的膨胀率
        activation="tanh",  # 激活函数
        recurrent_activation="hard_sigmoid",  # 循环层的激活函数
        use_bias=True,  # 是否使用偏置向量
        kernel_initializer="glorot_uniform",  # 卷积核的初始化器
        recurrent_initializer="orthogonal",  # 循环核的初始化器
        bias_initializer="zeros",  # 偏置向量的初始化器
        unit_forget_bias=True,  # 是否在遗忘门偏置上加1
        kernel_regularizer=None,  # 卷积核的正则化函数
        recurrent_regularizer=None,  # 循环核的正则化函数
        bias_regularizer=None,  # 偏置向量的正则化函数
        kernel_constraint=None,  # 卷积核的约束函数
        recurrent_constraint=None,  # 循环核的约束函数
        bias_constraint=None,  # 偏置向量的约束函数
        dropout=0.0,  # 输入的dropout率
        recurrent_dropout=0.0,  # 循环状态的dropout率
        **kwargs,  # 其他关键字参数
    ):
        super().__init__(**kwargs)  # 调用父类(Layer)的初始化方法
        self.rank = rank
        if self.rank > 3:
            raise ValueError(f"Rank {rank} convolutions are not currently implemented. Received: rank = {rank}")
        self.filters = filters
        self.kernel_size = self._normalize_tuple(kernel_size,rank,"kernel_size")
        self.strides = self._normalize_tuple(strides, rank, "strides", allow_zero=True)
        self.padding = self._normalize_padding(padding)
        self.data_format = self._normalize_data_format(data_format) # channels_last,channels_first
        self.dilation_rate = self._normalize_tuple(dilation_rate, rank,"dilation_rate")
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias
        self.kernel_regularizer = regularizers.get(kernel_regularizer)  # 卷积核的正则化
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)  # 循环核的正则化
        self.bias_regularizer = regularizers.get(bias_regularizer)  # 偏置的正则化
        self.kernel_constraint = constraints.get(kernel_constraint)  # 卷积核的约束
        self.recurrent_constraint = constraints.get(recurrent_constraint)  # 循环核的约束
        self.bias_constraint = constraints.get(bias_constraint)  # 偏置的约束
        self.dropout = min(1.0, max(0.0, dropout))  # 输入dropout率（默认0）
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))  # 循环dropout率（默认0）
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
            data_format = backend.image_data_format()
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
            raise ValueError("The channel dimension of the inputs should be defined. Found None.")
        input_dim = input_shape[channel_axis]
        
        self.kernel_shape = self.kernel_size + (input_dim, self.filters * 4)

        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)
       
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

    def call(self, inputs, states, training=None, mask=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # # 应用掩码（如果存在）
        # if mask is not None:
        #     mask = tf.cast(mask, h_tm1.dtype)
        #     mask = tf.expand_dims(mask, axis=-1)  # 扩展维度匹配 h_tm1 形状
        #     h_tm1 = h_tm1 * mask
        #     c_tm1 = c_tm1 * mask

        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4
        )

        if 0 < self.dropout < 1.0:
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
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        (kernel_i, kernel_f, kernel_c, kernel_o) = tf.split(
            self.kernel, 4, axis=self.rank + 1
        )
        (
            recurrent_kernel_i,
            recurrent_kernel_f,
            recurrent_kernel_c,
            recurrent_kernel_o,
        ) = tf.split(self.recurrent_kernel, 4, axis=self.rank + 1)

        if self.use_bias:
            bias_i, bias_f, bias_c, bias_o = tf.split(self.bias, 4)
        else:
            bias_i, bias_f, bias_c, bias_o = None, None, None, None

        x_i = self.input_conv(inputs_i, kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(inputs_f, kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(inputs_c, kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(inputs_o, kernel_o, bias_o, padding=self.padding)
        h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)
        return h, [h, c]
    
    @property
    def _conv_func(self):
        if self.rank == 1:
            return backend.conv1d
        if self.rank == 2:
            return backend.conv2d
        if self.rank == 3:
            return backend.conv3d

    def input_conv(self, x, w, b=None, padding="valid"):
        conv_out = self._conv_func(
            x,
            w,
            strides=self.strides,
            padding=padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if b is not None:
            conv_out = backend.bias_add(
                conv_out, b, data_format=self.data_format
            )
        return conv_out

    def recurrent_conv(self, x, w):
        strides = self._normalize_tuple(
            1, self.rank, "strides", allow_zero=True
        )
        conv_out = self._conv_func(
            x,
            w, 
            strides=strides, 
            padding="same", 
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        return conv_out

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
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        recurrent_activation="hard_sigmoid",
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
        **kwargs,
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
            name="conv_lstm_cell",
            dtype=kwargs.get("dtype"),
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(
            inputs, mask=mask, training=training, initial_state=initial_state
        )
    
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