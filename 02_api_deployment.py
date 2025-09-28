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


client = cmlapi.default_client()
client.list_projects()

projectId = os.environ['CDSW_PROJECT_ID']
username = os.environ["PROJECT_OWNER"]
experimentName = "xgb-telco-{0}".format(username)

experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
runsDf = mlflow.search_runs(experimentId, run_view_type=1)

experimentId = runsDf.iloc[-1]['experiment_id']
experimentRunId = runsDf.iloc[-1]['run_id']

deployment = ModelDeployment(client, projectId, username, experimentName, experimentId)

modelPath = "artifacts"
modelName = "CellTwrFail-CLF-" + username

# HOLD FOR A MOMENT AND THEN RUN THE FOLLOWING
registeredModelResponse = deployment.registerModelFromExperimentRun(modelName, experimentId, experimentRunId, modelPath)

modelId = registeredModelResponse.model_id
modelVersionId = registeredModelResponse.model_versions[0].model_version_id

registeredModelResponse.model_versions[0].model_version_id
createModelResponse = deployment.createModel(projectId, modelName, modelId)
modelCreationId = createModelResponse.id

runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.06.1-b5" #Modify as needed

createModelBuildResponse = deployment.createModelBuild(projectId, modelVersionId, modelCreationId, runtimeId)
modelBuildId = createModelBuildResponse.id

deployment.createModelDeployment(modelBuildId, projectId, modelCreationId)

## NOW TRY A REQUEST WITH THIS PAYLOAD!
#{"dataframe_split": {"columns": ["iot_signal_1", "iot_signal_2", "iot_signal_3", "iot_signal_4", "signal_score"], "data":[[35.5, 200.5, 30.5, 14.5, 23.00]]}}
