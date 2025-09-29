#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import json, secrets, os, time
import mlflow
from mlops import ModelDeployment

class ModelReDeployment():
    """
    Class to manage the model deployment of the xgboost model
    """

    def __init__(self, projectId, username):
        self.client = cmlapi.default_client()
        self.projectId = projectId
        self.username = username


    def createModelBuild(self, projectId, modelVersionId, modelCreationId, runtimeId, cpu, mem, replicas):
        """
        Method to create a Model build
        """


        # Create Model Build
        CreateModelBuildRequest = {
                                    "registered_model_version_id": modelVersionId,
                                    "runtime_identifier": runtimeId,
                                    "comment": "invoking model build",
                                    "model_id": modelCreationId,
                                    "cpu": cpu,
                                    "mem": mem,
                                    "replicas": replicas
                                  }

        try:
            # Create a model build.
            api_response = self.client.create_model_build(CreateModelBuildRequest, projectId, modelCreationId)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_build: %s\n" % e)

        return api_response


    def createModelDeployment(self, modelBuildId, projectId, modelCreationId):
        """
        Method to deploy a model build
        """

        CreateModelDeploymentRequest = {
          "cpu" : "2",
          "memory" : "4"
        }

        try:
            # Create a model deployment.
            api_response = self.client.create_model_deployment(CreateModelDeploymentRequest, projectId, modelCreationId, modelBuildId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_deployment: %s\n" % e)

        return api_response


    def listRuntimes(self):
        """
        Method to list available runtimes
        """
        search_filter = {"kernel": "Python 3.9", "edition": "Standard", "editor": "Workbench"}
        # str | Search filter is an optional HTTP parameter to filter results by.
        # Supported search filter keys are: [\"image_identifier\", \"editor\", \"kernel\", \"edition\", \"description\", \"full_version\"].
        # For example:   search_filter = {\"kernel\":\"Python 3.7\",\"editor\":\"JupyterLab\"},. (optional)
        search = json.dumps(search_filter)
        try:
            # List the available runtimes, optionally filtered, sorted, and paginated.
            api_response = self.client.list_runtimes(search_filter=search, page_size=1000)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->list_runtimes: %s\n" % e)

        return api_response


    def registerModelFromExperimentRun(self, modelName, experimentId, experimentRunId, modelPath):
        """
        Method to register a model from an Experiment Run
        This is an alternative to the mlflow method to register a model via the register_model parameter in the log_model method
        Input: requires an experiment run
        Output:
        """

        CreateRegisteredModelRequest = {
                                        "project_id": os.environ['CDSW_PROJECT_ID'],
                                        "experiment_id" : experimentId,
                                        "run_id": experimentRunId,
                                        "model_name": modelName,
                                        "model_path": modelPath
                                       }

        try:
            # Register a model.
            api_response = self.client.create_registered_model(CreateRegisteredModelRequest)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_registered_model: %s\n" % e)

        return api_response



    def get_latest_deployment_details(self, model_name):
        """
        Given a APIv2 client object and Model Name, use APIv2 to retrieve details about the latest/current deployment.
        This function only works for models deployed within the current project.
        """

        project_id = os.environ["CDSW_PROJECT_ID"]

        # gather model details
        models = (
            self.client.list_models(project_id=project_id, async_req=True, page_size = 50)
            .get()
            .to_dict()
        )
        model_info = [
            model for model in models["models"] if model["name"] == model_name
        ][0]

        model_id = model_info["id"]
        model_crn = model_info["crn"]

        # gather latest build details
        builds = (
            self.client.list_model_builds(
                project_id=project_id, model_id=model_id, async_req=True, page_size = 50
            )
            .get()
            .to_dict()
        )
        build_info = builds["model_builds"][-1]  # most recent build

        build_id = build_info["id"]

        # gather latest deployment details
        deployments = (
            self.client.list_model_deployments(
                project_id=project_id,
                model_id=model_id,
                build_id=build_id,
                async_req=True,
                page_size = 50
            )
            .get()
            .to_dict()
        )
        deployment_info = deployments["model_deployments"][-1]  # most recent deployment

        model_deployment_crn = deployment_info["crn"]

        return {
            "model_name": model_name,
            "model_id": model_id,
            "model_crn": model_crn,
            "latest_build_id": build_id,
            "latest_deployment_crn": model_deployment_crn,
        }

USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = os.environ["DBNAME_PREFIX"]+"_"+USERNAME
CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]
projectId = os.environ['CDSW_PROJECT_ID']

# SET MLFLOW EXPERIMENT NAME
experimentName = "xgb-telco-{0}".format(USERNAME)

experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
runsDf = mlflow.search_runs(experimentId, run_view_type=1)

experimentId = runsDf.iloc[-1]['experiment_id']
experimentRunId = runsDf.iloc[-1]['run_id']

modelPath = "artifacts"
modelName = "CellTwrFail-CLF-" + USERNAME

deployment = ModelReDeployment(projectId, USERNAME)
getLatestDeploymentResponse = deployment.get_latest_deployment_details(modelName)

registeredModelResponse = deployment.registerModelFromExperimentRun(modelName, experimentId, experimentRunId, modelPath)

modelId = registeredModelResponse.model_id
modelVersionId = registeredModelResponse.model_versions[0].model_version_id

registeredModelResponse.model_versions[0].model_version_id

modelCreationId = getLatestDeploymentResponse["model_id"]

cpu = 2
mem = 4
replicas = 1

runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5" #Modify as needed

createModelBuildResponse = deployment.createModelBuild(projectId, modelVersionId, modelCreationId, runtimeId, cpu, mem, replicas)
modelBuildId = createModelBuildResponse.id

deployment.createModelDeployment(modelBuildId, projectId, modelCreationId)
