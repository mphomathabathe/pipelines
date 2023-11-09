"""Graph Component for uploading and deployig llm model ."""

import json
from typing import NamedTuple, Optional

from google_cloud_pipeline_components import _placeholders
from google_cloud_pipeline_components._implementation.llm import deploy_llm_model
from google_cloud_pipeline_components._implementation.llm import function_based
from google_cloud_pipeline_components._implementation.llm import upload_llm_model
import kfp

PipelineOutput = NamedTuple(
    'Outputs', model_resource_name=str, endpoint_resource_name=str
)


@kfp.dsl.pipeline(
    name='llm-model-handler-graph',
    description='Uploads llm model and deploys it to an endpoint.',
)
def pipeline(
    output_adapter_path: str,
    large_model_reference: str,
    model_display_name: Optional[str] = None,
    deploy_model: bool = True,
) -> PipelineOutput:
  # fmt: off
  """Trains a reward model.

  Args:
    output_adapter_path: Path to the trained model adapter if LoRA tuning was used.
    large_model_reference: Name of the base model. Supported values are `text-bison@001`, `t5-small`, `t5-large`, `t5-xl` and `t5-xxl`. `text-bison@001` and `t5-small` are supported in `us-central1` and `europe-west4`. `t5-large`, `t5-xl` and `t5-xxl` are only supported in `europe-west4`.
    model_display_name: Name of the fine-tuned model shown in the Model Registry. If not provided, a default name will be created.
    deploy_model: Whether to deploy the model to an endpoint in `us-central1`. Default is True.

  Returns:
    model_resource_name: Path to the model uploaded to the Model Registry. This will be an empty string if the model was not deployed.
    endpoint_resource_name: Path the Online Prediction Endpoint. This will be an empty string if the model was not deployed. reward_model_output_path: Path to the trained reward model.
  """
  # fmt: on
  upload_location = 'us-central1'
  adapter_artifact = kfp.dsl.importer(
      artifact_uri=output_adapter_path,
      artifact_class=kfp.dsl.Artifact,
  ).set_display_name('Import Tuned Adapter')

  regional_endpoint = function_based.resolve_regional_endpoint(
      upload_location=upload_location
  ).set_display_name('Resolve Regional Endpoint')

  display_name = function_based.resolve_model_display_name(
      large_model_reference=large_model_reference,
      model_display_name=model_display_name,
  ).set_display_name('Resolve Model Display Name')

  reference_model_metadata = function_based.resolve_reference_model_metadata(
      large_model_reference=large_model_reference,
  ).set_display_name('Resolve Model Metadata')

  upload_model = function_based.resolve_upload_model(
      large_model_reference=reference_model_metadata.outputs[
          'large_model_reference'
      ]
  ).set_display_name('Resolve Upload Model')
  upload_task = (
      upload_llm_model.upload_llm_model(
          project=_placeholders.PROJECT_ID_PLACEHOLDER,
          location=upload_location,
          regional_endpoint=regional_endpoint.output,
          artifact_uri=adapter_artifact.output,
          model_display_name=display_name.output,
          model_reference_name='text-bison@001',
          upload_model=upload_model.output,
      )
      .set_env_variable(
          name='VERTEX_AI_PIPELINES_RUN_LABELS',
          value=json.dumps({'tune-type': 'rlhf'}),
      )
      .set_display_name('Upload Model')
  )
  deploy_model = function_based.resolve_deploy_model(
      deploy_model=deploy_model,
      large_model_reference=reference_model_metadata.outputs[
          'large_model_reference'
      ],
  ).set_display_name('Resolve Deploy Model')
  deploy_task = deploy_llm_model.create_endpoint_and_deploy_model(
      project=_placeholders.PROJECT_ID_PLACEHOLDER,
      location=upload_location,
      model_resource_name=upload_task.outputs['model_resource_name'],
      display_name=display_name.output,
      regional_endpoint=regional_endpoint.output,
      deploy_model=deploy_model.output,
  ).set_display_name('Deploy Model')
  return PipelineOutput(
      model_resource_name=upload_task.outputs['model_resource_name'],
      endpoint_resource_name=deploy_task.outputs['endpoint_resource_name'],
  )
