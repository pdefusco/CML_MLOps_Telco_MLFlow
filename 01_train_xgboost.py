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

import os, warnings, sys, logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import mlflow.sklearn
from xgboost import XGBClassifier
from datetime import date
import cml.data_v1 as cmldata
import pyspark.pandas as ps


USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = os.environ["DBNAME_PREFIX"]+"_"+USERNAME
CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]

DATE = date.today()
EXPERIMENT_NAME = "xgb-telco-{0}".format(USERNAME)

mlflow.set_experiment(EXPERIMENT_NAME)

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

df_from_sql = ps.read_table('{0}.TELCO_CELL_TOWERS_{1}.snapshots'.format(DBNAME, USERNAME))
df_from_sql = df_from_sql[["iot_signal_1", "iot_signal_2", "iot_signal_3", "iot_signal_4", "signal_score", "cell_tower_failure"]]
df = df_from_sql.to_pandas()

test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(df.drop("cell_tower_failure", axis=1), df["cell_tower_failure"], test_size=test_size)

with mlflow.start_run():

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # Step 1: cambiar test_size linea 69 y recorrer
    # Step 2: cambiar linea 74, agregar linea 97, y recorrer
      # linea 75: model = XGBClassifier(use_label_encoder=False, max_depth=4, eval_metric="logloss")
      # linea 97: mlflow.log_param("max_depth", 4)
    # Step 3: cambiar linea 74 y 97, agregar linea 98, y recorrer
      # linea 75: model = XGBClassifier(use_label_encoder=False, max_depth=2, max_leaf_nodes=5, eval_metric="logloss")
      # linea 97: mlflow.log_param("max_depth", 2)
      # linea 98: mlflow.log_param("max_leaf_nodes", 5)

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Test Size: %.2f%%" % (test_size * 100.0))

    mlflow.log_param("accuracy", accuracy)
    mlflow.log_param("test_size", test_size)

    # Step 2:
    # Step 3:

    mlflow.xgboost.log_model(model, artifact_path="artifacts")#, registered_model_name="my_xgboost_model"

def getLatestExperimentInfo(experimentName):
    """
    Method to capture the latest Experiment Id and Run ID for the provided experimentName
    """
    experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
    runsDf = mlflow.search_runs(experimentId, run_view_type=1)
    experimentId = runsDf.iloc[-1]['experiment_id']
    experimentRunId = runsDf.iloc[-1]['run_id']

    return experimentId, experimentRunId

experimentId, experimentRunId = getLatestExperimentInfo(EXPERIMENT_NAME)

#Replace Experiment Run ID here:
run = mlflow.get_run(experimentRunId)

pd.DataFrame(data=[run.data.params], index=["Value"]).T
pd.DataFrame(data=[run.data.metrics], index=["Value"]).T

client = mlflow.tracking.MlflowClient()
client.list_artifacts(run_id=run.info.run_id)
