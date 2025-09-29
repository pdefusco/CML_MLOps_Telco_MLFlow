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

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType
from pyspark.sql import SparkSession
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import cml.data_v1 as cmldata
from pyspark.sql import functions as F
from pyspark.sql.functions import when, rand


class TelcoDataGen:

    '''Class to Generate Telco Data'''

    def __init__(self, username, dbname, connectionName):
        self.username = username
        self.dbname = dbname
        self.connectionName = connectionName


    def telcoDataGen(self, spark, shuffle_partitions_requested = 1, partitions_requested = 1, data_rows = 1440):
        """
        Method to create IoT data in Spark Df
        """

        manufacturers = ["New World Corp", "AI Inc.", "Hot Data Ltd"]

        iotDataSpec = (
            dg.DataGenerator(spark, name="device_data_set", rows=data_rows, partitions=partitions_requested)
            .withIdOutput()
            .withColumn("internal_device_id", "long", minValue=0x1000000000000, uniqueValues=int(data_rows/36), omit=True, baseColumnType="hash")
            .withColumn("device_id", "string", format="0x%013x", baseColumn="internal_device_id")
            .withColumn("manufacturer", "string", values=manufacturers, baseColumn="internal_device_id", )
            .withColumn("model_ser", "integer", minValue=1, maxValue=11, baseColumn="device_id", baseColumnType="hash", omit=True, )
            .withColumn("event_type", "string", values=["battery 10%", "battery 5%", "device error", "system malfunction"], random=True)
            .withColumn("longitude", "float", expr="rand()*rand()+10 + -93.6295")
            .withColumn("latitude", "float", expr="rand()*rand()+10 + 41.5949")
            .withColumn("iot_signal_1", "integer", minValue=1, maxValue=10, random=True)
            .withColumn("iot_signal_3", "integer", minValue=50, maxValue=55, random=True)
            .withColumn("iot_signal_4", "integer", minValue=100, maxValue=107, random=True)
            .withColumn("cell_tower_failure", "string", values=["0", "1"], weights=[6, 4], random=True)
        )

        df = iotDataSpec.build()
        df = df.withColumn("cell_tower_failure", df["cell_tower_failure"].cast(IntegerType()))
        df = df.withColumn(
                    "signal_score",
                    F.when(rand() < 0.20, rand())  # 20% of the time, just random noise
                    .otherwise(F.col("cell_tower_failure") * F.col("iot_signal_1") * F.col("iot_signal_3"))
                )
        df = df.withColumn("signal_score", F.col("signal_score").cast(FloatType()))

        return df


    def createSparkConnection(self):
        """
        Method to create a Spark Connection using CML Data Connections
        """

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark


    def saveFileToCloud(self, df):
        """
        Method to save credit card transactions df as csv in cloud storage
        """

        df.write.format("csv").mode('overwrite').save(self.storage + "/telco_demo/" + self.username)


    def addCorrelatedColumn(self, dataGenDf):
        """
        Method to create a column iot_signal_2 with value that is correlated to value of iot_signal_1 column
        """

        from pyspark.sql.functions import udf
        import random

        def addColUdf(val):
          return (val)+random.randint(0, 2)

        udf_column = udf(addColUdf, IntegerType())
        dataGenDf = dataGenDf.withColumn('iot_signal_2', udf_column('iot_signal_1'))

        return dataGenDf


    def createDatabase(self, spark):
        """
        Method to create database before data generated is saved to new database and table
        """

        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def createOrReplace(self, df):
        """
        Method to create or append data to the TELCO_CELL_TOWERS table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        try:
            df.writeTo("{0}.TELCO_CELL_TOWERS_{1}".format(self.dbname, self.username))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.TELCO_CELL_TOWERS_{1}".format(self.dbname, self.username))\
                .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()


    def validateTable(self, spark):
        """
        Method to validate creation of table
        """
        print("SHOW TABLES FROM '{}'".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()


def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = os.environ["DBNAME_PREFIX"]+"_"+USERNAME
    CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]

    # Instantiate BankDataGen class
    dg = TelcoDataGen(USERNAME, DBNAME, CONNECTION_NAME)

    # Create CML Spark Connection
    spark = dg.createSparkConnection()

    # Create Banking Transactions DF
    df = dg.telcoDataGen(spark)
    df = dg.addCorrelatedColumn(df)

    # Create Spark Database
    dg.createDatabase(spark)

    # Create Iceberg Table in Database
    dg.createOrReplace(df)

    # Validate Iceberg Table in Database
    dg.validateTable(spark)


if __name__ == '__main__':
    main()
