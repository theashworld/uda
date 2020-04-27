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
#!/bin/bash


'''
input_file: The file to be back translated. We assume that each paragraph is in
a separate line
'''
echo "*** spliting paragraph ***"
# install nltk
python split_paragraphs.py \
  --input_file=${INPUT_FILE} \
  --output_file=${INPUT_FILE}_split.txt \
  --doc_len_file=${INPUT_FILE}_doclen.json

input_file=gs://bewgle-data/${INPUT_FILE}

'''
sampling_temp: The sampling temperature for translation. See README.md for more
details.
'''
sampling_temp=0.8


# Dirs
data_dir=gs://bewgle-data/back_trans_data
doc_len_dir=${data_dir}/doc_len
forward_src_dir=${data_dir}/forward_src
forward_gen_dir=${data_dir}/forward_gen
backward_gen_dir=${data_dir}/backward_gen
para_dir=${data_dir}/paraphrase

gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m rsync -r checkpoints gs://bewgle-data/
gsutil -m cp ${INPUT_FILE} gs://bewgle-data/
gsutil -m cp ${INPUT_FILE}_split.txt ${forward_src_dir}/
gsutil -m cp ${INPUT_FILE}_doclen.json ${forward_src_dir}/

#mkdir -p ${data_dir}
#mkdir -p ${forward_src_dir}
#mkdir -p ${forward_gen_dir}
#mkdir -p ${backward_gen_dir}
#mkdir -p ${doc_len_dir}
#mkdir -p ${para_dir}

echo "*** forward translation ***"
t2t-decoder \
  --problem=translate_enfr_wmt32k \
  --model=transformer \
  --hparams_set=transformer_big \
  --hparams="sampling_method=random,sampling_temp=${sampling_temp}" \
  --decode_hparams="beam_size=1,batch_size=1024" \
  --checkpoint_path=gs://bewgle-data/enfr/model.ckpt-500000 \
  --output_dir=gs://bewgle-data/tmp/t2t \
  --decode_from_file=${forward_src_dir}/${INPUT_FILE}_split.txt \
  --decode_to_file=${forward_gen_dir}/${INPUT_FILE}_split.txt \
  --data_dir=gs://bewgle-data/ \
  --cloud_tpu_name=$TPU_NAME \
  --use_tpu  &> /dev/null

echo "*** backward translation ***"
t2t-decoder \
  --problem=translate_enfr_wmt32k_rev \
  --model=transformer \
  --hparams_set=transformer_big \
  --hparams="sampling_method=random,sampling_temp=${sampling_temp}" \
  --decode_hparams="beam_size=1,batch_size=1024,alpha=0" \
  --checkpoint_path=gs://bewgle-data/fren/model.ckpt-500000 \
  --output_dir=/tmp/t2t \
  --decode_from_file=${forward_gen_dir}/${INPUT_FILE}_split.txt \
  --decode_to_file=${backward_gen_dir}/${INPUT_FILE}_split.txt \
  --data_dir=gs://bewgle-data/ \
  --cloud_tpu_name=$TPU_NAME \
  --use_tpu &> /dev/null

gsutil -m rsync ${backward_gen_dir}/ .

echo "*** transform sentences back into paragraphs***"
python sent_to_paragraph.py \
  --input_file=${INPUT_FILE}_split.txt \
  --doc_len_file=${INPUT_FILE}_doclen.json \
  --output_file=${INPUT_FILE}_out

