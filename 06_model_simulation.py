# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2024
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
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
# ###########################################################################

import cdsw
import time, os, random, json, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmlbootstrap import CMLBootstrap
from pyspark.sql import SparkSession
from sklearn.metrics import classification_report
import cmlapi
from src.api import ApiUtility
import cml.data_v1 as cmldata
from utils import BankDataGen
import datetime

#---------------------------------------------------
#               CREATE BATCH DATA
#---------------------------------------------------

# SET USER VARIABLES
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "BNK_MLOPS_HOL_{}".format(USERNAME)
STORAGE = "s3a://eng-ml-weekly"
CONNECTION_NAME = "eng-ml-int-env-aws-dl"

# Instantiate BankDataGen class
dg = BankDataGen(USERNAME, DBNAME, STORAGE, CONNECTION_NAME)

# Create CML Spark Connection
spark = dg.createSparkConnection()

# Create Banking Transactions DF
sparkDf = dg.dataGen(spark)

df = sparkDf.toPandas()

# You can access all models with API V2
client = cmlapi.default_client()
project_id = os.environ["CDSW_PROJECT_ID"]
client.list_models(project_id)

# You can use an APIV2-based utility to access the latest model's metadata. For more, explore the src folder
apiUtil = ApiUtility()

modelName = "FraudCLF-" + username

Model_AccessKey = apiUtil.get_latest_deployment_details(model_name=modelName)["model_access_key"]
Deployment_CRN = apiUtil.get_latest_deployment_details(model_name=modelName)["latest_deployment_crn"]

#{"dataframe_split": {"columns": ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance", "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance", "longitude", "latitude", "transaction_amount"], "data":[[35.5, 20000.5, 3900.5, 14000.5, 2944.5, 3400.5, 12000.5, 29000.5, 1300.5, 15000.5, 10000.5, 2000.5, 90.5, 120.5]]}}

def submitRequest(Model_AccessKey):
    """
    Method to create and send a synthetic request to Time Series Query Model
    """

    record = '{"dataframe_split": {"columns": ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance", "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance", "longitude", "latitude", "transaction_amount"]}}'
    randomInts = [[random.uniform(1.01,500.01) for i in range(14)]]
    data = json.loads(record)
    data["dataframe_split"]["data"] = randomInts
    response = cdsw.call_model(Model_AccessKey, data)

    return response

response_labels_sample = []
percent_counter = 0
percent_max = len(df)

# This will randomly return True for input and increases the likelihood of returning
# true based on `percent`
def bnkFraud(percent):
    if random.random() < percent:
        return 1
    else:
        return 0

for i in range(1000):
  print("Added {} records".format(percent_counter)) if (
      percent_counter % 25 == 0
  ) else None
  percent_counter += 1
  response = submitRequest(Model_AccessKey)
  response_labels_sample.append(
        {
            "uuid": response["response"]["uuid"],
            "response_label": response["response"]["prediction"],
            "final_label": bnkFraud(percent_counter / percent_max),
            "timestamp_ms": int(round(time.time() * 1000)),
        }
    )


# The "ground truth" loop adds the updated actual label value and an accuracy measure
# every 100 calls to the model.
for index, vals in enumerate(response_labels_sample):
    print("Update {} records".format(index)) if (index % 50 == 0) else None
    cdsw.track_delayed_metrics({"final_label": vals["final_label"]}, vals["uuid"])
    if index % 1000 == 0:
        start_timestamp_ms = vals["timestamp_ms"]
        final_labels = []
        response_labels = []
    final_labels.append(vals["final_label"])
    response_labels.append(vals["response_label"][0])
    if index % 100 == 99:
        print("Adding accuracy metric")
        end_timestamp_ms = vals["timestamp_ms"]
        accuracy = classification_report(
            final_labels, response_labels, output_dict=True
        )["accuracy"]
        cdsw.track_aggregate_metrics(
            {"accuracy": accuracy},
            start_timestamp_ms,
            end_timestamp_ms,
            model_deployment_crn=Deployment_CRN,
        )
