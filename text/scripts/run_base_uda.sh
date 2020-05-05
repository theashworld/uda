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
gsutil -m rsync -r ./data/proc_data/ $GS_DIR/data/proc_data
gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m rsync -r ./pretrained_models $GS_DIR/pretrained_models
python main.py \
  --tpu_name=$TPU_NAME \
  --do_train=True \
  --do_eval=True \
  --sup_train_data_dir=$GS_DIR/data/proc_data/IMDB/train_20 \
  --unsup_data_dir=$GS_DIR/data/proc_data/IMDB/unsup \
  --eval_data_dir=$GS_DIR/data/proc_data/IMDB/dev \
  --bert_config_file=gs://bewgle-data/pretrained_models/bert_base/bert_config.json \
  --vocab_file=gs://bewgle-data/pretrained_models/bert_base/vocab.txt \
  --init_checkpoint=gs://bewgle-data/pretrained_models/bert_base/bert_model.ckpt \
  --task_name=IMDB \
  --model_dir=$GS_DIR/ckpt/base_uda \
  --num_train_steps=3000 \
  --learning_rate=2e-05 \
  --num_warmup_steps=1000 \
  --unsup_ratio=3 \
  --tsa=linear_schedule \
  --aug_ops=bt-0.9 \
  --aug_copy=1 \
  --uda_coeff=1 \
  $@
