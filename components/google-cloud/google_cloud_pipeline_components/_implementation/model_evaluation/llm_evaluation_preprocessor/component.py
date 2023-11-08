# Copyright 2023 The Kubeflow Authors. All Rights Reserved.
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
"""LLM Eval Preprocessor Component for Converting Eval Dataset Format."""

from typing import List

from google_cloud_pipeline_components._implementation.model_evaluation import version
from kfp import dsl


# pylint: disable=g-import-not-at-top, g-doc-args
# pytype: disable=invalid-annotation
@dsl.component(base_image=version.LLM_EVAL_IMAGE_TAG)
def evaluation_dataset_preprocessor(
    gcs_source_uris: List[str],
    output_dirs: dsl.OutputPath(list),
):
  # fmt: off
  """Preprocesses Eval Dataset format.

  This component adds a `prompt` field for running Batch Prediction component on
  the eval dataset. This component is used in LLM Evaluation pipelines.

  Args:
      gcs_source_uris: a list of GCS URIs of the input eval dataset.

  Returns:
      output_dirs: a list GCS directories where the output files will be stored. Should be generated by the pipeline.
  """
  # fmt: on

  # KFP component dependencies must be imported in function body.
  from etils import epath
  import json
  import logging
  import sys

  # pylint: disable=invalid-name
  GCS_ROOT_PREFIX = '/gcs/'
  GS_PREFIX = 'gs://'
  INPUT_TEXT_KEY = 'input_text'
  DATA_KEY = 'data'
  PROMPT_KEY = 'prompt'
  # pylint: enable=invalid-name

  def preprocess_file(dataset_file_uri: str, output_path: epath.Path):
    input_path = epath.Path(dataset_file_uri)
    output_file_path = output_path.parent / input_path.name

    num_entries = 0

    content = ''
    for dataset_line in input_path.read_text().splitlines():
      dataset_line = dataset_line.strip()
      if not dataset_line:
        continue
      json_instance = json.loads(dataset_line)
      # Check if file has one JSON training example per line, or has an object
      # whose data field contains all the examples.
      for data_obj in json_instance.get(DATA_KEY, [json_instance]):
        if PROMPT_KEY not in data_obj and INPUT_TEXT_KEY in data_obj:
          data_obj[PROMPT_KEY] = data_obj[INPUT_TEXT_KEY]
        content += json.dumps(data_obj, ensure_ascii=False) + '\n'
        num_entries += 1
    output_file_path.write_text(content)
    if 0 == num_entries:
      raise ValueError(
          'Dataset is inaccessible or contains no valid entries. Please'
          ' ensure your pipeline can access the dataset and that it is'
          f' properly formatted. Dataset URI: {dataset_file_uri}'
      )
    output_dirs.append(
        str(output_file_path).replace(GCS_ROOT_PREFIX, GS_PREFIX, 1)
    )

  output_path = epath.Path(output_dirs)
  try:
    if not gcs_source_uris:
      output_path.write_text(json.dumps([]))

    output_dirs = []
    for gcs_source_uri in gcs_source_uris:
      logging.info('Processing Input gcs_source_uri: %s', gcs_source_uri)
      preprocess_file(gcs_source_uri, output_path)

    logging.info('Pipeline Output output_dirs: %s', output_dirs)
    output_path.write_text(json.dumps(output_dirs))
  except Exception as e:  # pylint: disable=broad-exception-caught
    if isinstance(e, ValueError):
      raise
    logging.exception(str(e))
    sys.exit(13)


# pytype: enable=invalid-annotation
# pylint: enable=g-import-not-at-top, g-doc-args
