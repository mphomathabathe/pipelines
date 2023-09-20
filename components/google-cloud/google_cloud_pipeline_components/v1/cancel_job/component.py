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

from google_cloud_pipeline_components import _image
from kfp.dsl import container_component
from kfp.dsl import ContainerSpec
from kfp.dsl import PipelineTaskFinalStatus


@container_component
def cancel_job(
    pipeline_task_status: PipelineTaskFinalStatus,
    gcp_resources: str = 'default',
):
  # fmt: off
  """Cancels a job.

  Currently this component only support cancellation on V1 CustomJob resource.

  To use this component, first create a component that outputs a JSON formatted gcp_resources proto, then pass it to the cancel component.

  dataflow_python_op = gcpc.experimental.dataflow.LaunchPythonOp(
      python_file_path = ...
  )

  dataflow_cancel_op = gcpc.experimental.cancel_job.CancelJobOp(
      gcp_resources = dataflow_python_op.outputs["gcp_resources"]
  )

  For details on how to create a Json serialized gcp_resources proto as output, see
  https://github.com/kubeflow/pipelines/tree/master/components/google-cloud/google_cloud_pipeline_components/experimental/proto

  Args:
    pipeline_task_status (PipelineTaskFinalStatus):
        Task status of the outer pipeline
        For details, see https://kubeflow-pipelines.readthedocs.io/en/2.0.0b6/source/dsl.html#kfp.dsl.PipelineTaskFinalStatus
      gcp_resources (str):
        Serialized JSON of gcp_resources proto, indicating the resource to cancel by this component
        For details, see https://github.com/kubeflow/pipelines/tree/master/components/google-cloud/google_cloud_pipeline_components/experimental/proto

  """
  # fmt: on
  return ContainerSpec(
      image=_image.GCPC_IMAGE_TAG,
      command=[
          'python3',
          '-u',
          '-m',
          'google_cloud_pipeline_components.container.v1.cancel_job.launcher',
      ],
      args=[
          '--pipeline_task_status',
          pipeline_task_status,
          '--gcp_resources',
          gcp_resources,
      ],
  )
