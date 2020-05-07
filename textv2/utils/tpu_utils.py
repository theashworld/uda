# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def construct_scalar_host_call(
    metric_dict,
    model_dir,
    prefix="",
    reduce_fn=None):

  metric_names = list(metric_dict.keys())

  def host_call_fn(global_step, *args):
    step = global_step[0]
    with tf.compat.v2.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf.compat.v2.summary.record_if(True):
        for i, name in enumerate(metric_names):
          if reduce_fn is None:
            scalar = args[i][0]
          else:
            scalar = reduce_fn(args[i])
          with tf.compat.v2.summary.record_if(lambda: tf.math.equal(0, step % 1)):
            tf.compat.v2.summary.scalar(name=prefix + name, data=scalar, step=step)

        return tf.compat.v1.summary.all_v2_summary_ops()

  global_step_tensor = tf.reshape(tf.compat.v1.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(metric_dict[key], [-1]) for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors

