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

# SET USER VARIABLES
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "TELCO_MLOPS_"+USERNAME
STORAGE = "s3a://go01-demo"
CONNECTION_NAME = "go01-aw-dl"

# SET MLFLOW EXPERIMENT NAME
EXPERIMENT_NAME = "xgb-telco-{0}".format(USERNAME)
mlflow.set_experiment(EXPERIMENT_NAME)

# CREATE SPARK SESSION WITH DATA CONNECTIONS
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# READ LATEST ICEBERG METADATA
snapshot_id = spark.read.format("iceberg").load('{0}.TELCO_CELL_TOWERS_{1}.snapshots'.format(DBNAME, USERNAME)).select("snapshot_id").tail(1)[0][0]
committed_at = spark.read.format("iceberg").load('{0}.TELCO_CELL_TOWERS_{1}.snapshots'.format(DBNAME, USERNAME)).select("committed_at").tail(1)[0][0].strftime('%m/%d/%Y')
parent_id = spark.read.format("iceberg").load('{0}.TELCO_CELL_TOWERS_{1}.snapshots'.format(DBNAME, USERNAME)).select("parent_id").tail(1)[0][0]

incReadDf = spark.read\
    .format("iceberg")\
    .option("start-snapshot-id", parent_id)\
    .option("end-snapshot-id", snapshot_id)\
    .load("{0}.TELCO_CELL_TOWERS_{1}".format(DBNAME, USERNAME))

incReadDf = incReadDf[["iot_signal_1", "iot_signal_2", "iot_signal_3", "iot_signal_4", "cell_tower_failure"]]
df = incReadDf.toPandas()

# SET MLFLOW TAGS
tags = {
  "iceberg_snapshot_id": snapshot_id,
  "iceberg_snapshot_committed_at": committed_at,
  "iceberg_parent_id": parent_id,
  "row_count": df.count()
}

# TRAIN TEST SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(df.drop("cell_tower_failure", axis=1), df["cell_tower_failure"], test_size=0.3)

# MLFLOW EXPERIMENT RUN
with mlflow.start_run():

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Recall: %.2f%%" % (recall * 100.0))

    mlflow.log_param("accuracy", accuracy)
    mlflow.log_param("recall", recall)
    mlflow.xgboost.log_model(model, artifact_path="artifacts")#, registered_model_name="my_xgboost_model"
    mlflow.set_tags(tags)

mlflow.end_run()

# MLFLOW CLIENT EXPERIMENT METADATA
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
