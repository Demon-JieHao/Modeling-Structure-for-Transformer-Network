# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
              dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    return tf.where(copy_cond, output, new_output)

def _gru_encoder(cell, inputs, initial_state, tgt_len, state=None, dtype=None):
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype
    next_state={}
    if state is not None:
        inputs = tf.concat([state["key"], inputs], axis=1)
        next_state["key"] = inputs
        next_state["value"] = inputs
        #inputs1 = inputs1[:, -1:, :]
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state1):
        inp_t = input_ta.read(t)
        cell_output, new_state = cell(inp_t, state1)
        cell_output = _copy_through(t, tgt_len, zero_output, cell_output)
        new_state = _copy_through(t, tgt_len, state1, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    if state is not None:
#        return all_output, next_state
        out = all_output[:, -1:, :]
        outputs = {"outputs": out, "state":next_state}
        return outputs
    return {"outputs": all_output}

def gru_encoder(cell, inputs, sequence_length, initial_state, dtype=None):
    # Assume that the underlying cell is GRUCell-like
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]

    zero_output = tf.zeros([batch, output_size], dtype)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

    def loop_func(t, out_ta, state):
        inp_t = input_ta.read(t)
        cell_output, new_state = cell(inp_t, state)
        cell_output = _copy_through(t, sequence_length, zero_output,
                                    cell_output)
        new_state = _copy_through(t, sequence_length, state, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])

    return all_output, final_state


def transformer_encoder(inputs, bias, sequence_length, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        inputs_bw = tf.reverse_sequence(inputs, sequence_length,
                                        batch_axis=0, seq_axis=1)
        cell_fw = layers.rnn_cell.ONLSTMCell(params.hidden_size)
        cell_bw = layers.rnn_cell.ONLSTMCell(params.hidden_size)
        if params.use_variational_dropout:
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                    cell_fw,
                    input_keep_prob=1.0 - params.dropout,
                    output_keep_prob=1.0 - params.dropout,
                    state_keep_prob=1.0 - params.dropout,
                    variational_recurrent=True,
                    input_size=params.hidden_size,
                    dtype=tf.float32
                    )
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                    cell_bw,
                    input_keep_prob=1.0 - params.dropout,
                    output_keep_prob=1.0 - params.dropout,
                    state_keep_prob=1.0 - params.dropout,
                    variational_recurrent=True,
                    input_size=params.hidden_size,
                    dtype=tf.float32
                    )

        for layer in range(3):
            with tf.variable_scope("layer_rnn_%d" % layer):            
                with tf.variable_scope("forward"):
                    output_fw, state_fw = gru_encoder(cell_fw, x,
                                                       sequence_length, None,
                                                       dtype=dtype)

                with tf.variable_scope("backward"):
                    output_bw, state_bw = gru_encoder(cell_bw, tf.reverse_sequence(x, sequence_length,
                                            batch_axis=0, seq_axis=1),sequence_length, None, dtype=dtype)
                    output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                                    batch_axis=0, seq_axis=1)
                
                y = tf.concat([output_fw, output_bw], axis=-1)
                y = layers.nn.linear(y, params.hidden_size, False, False)
                x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                x = _layer_process(x, params.layer_postprocess)
        
        onlstm_output = x 

        for layer in range(3):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)
        outputs = _residual_fn(outputs, onlstm_output, 1.0-params.residual_dropout)
        outputs = _layer_process(outputs, params.layer_postprocess)

        return outputs

def transformer_decoder(inputs, memory, bias, mem_bias, sequence_length, params, state=None,
                        dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        cell_fw = layers.rnn_cell.ONLSTMCell(params.hidden_size)
        if params.use_variational_dropout:
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                    cell_fw,
                    input_keep_prob=1.0 - params.dropout,
                    output_keep_prob=1.0 - params.dropout,
                    state_keep_prob=1.0 - params.dropout,
                    variational_recurrent=True,
                    input_size=params.hidden_size,
                    dtype=tf.float32
                    )
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                if layer<3:
                    with tf.variable_scope("on-lstm"):
                        with tf.variable_scope("forward"):
                            outputs = _gru_encoder(cell_fw, x, None, sequence_length, state=layer_state, dtype=dtype) 
                            if layer_state is not None:
                                next_state[layer_name] = outputs["state"]
                        y = layers.nn.linear(outputs["outputs"], params.hidden_size, False, False)
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess)
                else:
                    with tf.variable_scope("self_attention"):
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            None,
                            bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            state=layer_state
                        )

                        if layer_state is not None:
                            next_state[layer_name] = y["state"]

                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                if layer==2: onlstm_output = x

        outputs = _layer_process(x, params.layer_preprocess)
        if layer==5:
            outputs = _residual_fn(outputs, onlstm_output, 1.0-params.residual_dropout)
            outputs = _layer_process(outputs, params.layer_postprocess)

        if state is not None:
            return outputs, next_state

        return outputs


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    # id => embedding
    # src_seq: [batch, max_src_length]
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)

    # Preparing encoder
    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, src_len, params)

    return encoder_output


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    # id => embedding
    # tgt_seq: [batch, max_tgt_length]
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder and decoder input
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode != "infer":
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias, tgt_len,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias, tgt_len,
                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    # [batch, length, channel] => [batch * length, vocab_size]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    labels = features["target"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


def model_graph(features, mode, params):
    encoder_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output
    }
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            layer_preprocess="layer_norm",
            layer_postprocess="none",
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            use_variational_dropout=True,
            dropout=0.0
        )

        return params
