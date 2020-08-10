# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.transformer
import thumt.models.transformer_onlstm_decoder
import thumt.models.transformer_joint_encdec

def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return thumt.models.rnnsearch.RNNsearch
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "transformer_onlstm_decoder":
	   return thumt.models.transformer_onlstm_decoder.Transformer
    elif name == "transformer_joint_encdec":
        return thumt.models.transformer_joint_encdec.Transformer
    else:
        raise LookupError("Unknown model %s" % name)
