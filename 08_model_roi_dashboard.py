#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
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

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import os
import cml.data_v1 as cmldata
import pyspark.pandas as ps
import os
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# SET USER VARIABLES
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = os.environ["DBNAME_PREFIX"]+"_"+USERNAME
CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]

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

# ----------------------------
# Data prep and model training
# ----------------------------
incReadDf = incReadDf[["iot_signal_1", "iot_signal_2", "iot_signal_3", "iot_signal_4", "signal_score", "cell_tower_failure"]]
df = incReadDf.toPandas()

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("cell_tower_failure", axis=1),
    df["cell_tower_failure"],
    test_size=0.3,
    random_state=42
)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_proba = model.predict_proba(X_test)[:, 1]

# ----------------------------
# Initialize Dash app
# ----------------------------
app = dash.Dash(__name__)

# ----------------------------
# Layout
# ----------------------------
app.layout = html.Div([
    html.H2("Interactive Model ROI Dashboard",
            style={"font-size": "32px", "font-family": "Arial, sans-serif"}),
    html.Div([
        html.P(
            "Use this tool to explore how adjusting the decision threshold affects model predictions and financial outcomes. "
            "Modify the threshold and the input variables below to see the impact on True Positives, False Positives, "
            "False Negatives, and Net Revenue.",
            style={"font-size": "18px", "font-family": "Arial, sans-serif", "color": "gray"}
        ),
        html.P([
            html.Span("Net Revenue = TP Revenue + TN Revenue - Penalty for FP - Opportunity cost for FN."),
            html.Br(),
            html.Span("TP Revenue = TP * Revenue1, TN Revenue = TN * Revenue0,"),
            html.Br(),
            html.Span("FP Penalty = FP * PenaltyFP, FN Opportunity Cost = FN * Revenue1.")
        ], style={"font-size": "18px", "font-family": "'Segoe UI', 'Helvetica Neue', sans-serif", "color": "black"})
    ], style={"margin-bottom": "30px"}),

    # Inputs at the top
    html.Div([
        html.Div([
            html.Label("Financial Revenue (Actual Target=1):", style={"font-size": "18px"}),
            dcc.Input(id="revenue-class-1", type="number", value=100, step=10, style={"font-size": "16px"})
        ], style={"margin-right": "20px"}),

        html.Div([
            html.Label("Financial Revenue (Actual Target=0):", style={"font-size": "18px"}),
            dcc.Input(id="revenue-class-0", type="number", value=10, step=10, style={"font-size": "16px"})
        ], style={"margin-right": "20px"}),

        html.Div([
            html.Label("Penalty per False Positive:", style={"font-size": "18px"}),
            dcc.Input(id="penalty-fp", type="number", value=50, step=10, style={"font-size": "16px"})
        ])
    ], style={"display": "flex", "margin-bottom": "30px"}),

    # Confusion matrix and threshold slider side by side
    html.Div([
        # Confusion matrix
        html.Div([
            dcc.Graph(id="confusion-matrix-heatmap")
        ], style={"width": "70%"}),

        # Vertical slider
        html.Div([
            html.Label("Decision Threshold:", style={"font-size": "18px"}),
            dcc.Slider(
                id="threshold-slider",
                min=0.0, max=1.0, step=0.01, value=0.5,
                marks={0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1.0"},
                tooltip={"placement": "left", "always_visible": True},
                vertical=True,
                verticalHeight=400
            )
        ], style={"width": "10%", "margin-left": "30px", "margin-top": "50px"})
    ], style={"display": "flex"}),

    # Accuracy and revenue breakdown below
    html.Div([
        html.Div(id="accuracy-text", style={"margin-top": "20px", "font-size": "20px", "font-family": "Arial, sans-serif"}),
        html.Div(id="breakdown-text", style={"margin-top": "10px", "font-size": "18px", "font-family": "Arial, sans-serif", "color": "darkblue"})
    ])
], style={"font-family": "'Segoe UI', 'Helvetica Neue', sans-serif"})

# ----------------------------
# Callback
# ----------------------------
@app.callback(
    [Output("accuracy-text", "children"),
     Output("breakdown-text", "children"),
     Output("confusion-matrix-heatmap", "figure")],
    [Input("threshold-slider", "value"),
     Input("revenue-class-1", "value"),
     Input("revenue-class-0", "value"),
     Input("penalty-fp", "value")]
)
def update_outputs(threshold, revenue_1, revenue_0, penalty_fp):
    # Handle None inputs by providing default values
    revenue_1 = revenue_1 if revenue_1 is not None else 0
    revenue_0 = revenue_0 if revenue_0 is not None else 0
    penalty_fp = penalty_fp if penalty_fp is not None else 0

    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Revenue calculation
    revenue_tp = tp * revenue_1
    revenue_tn = tn * revenue_0
    total_fp_penalty = fp * penalty_fp
    total_fn_opp_cost = fn * revenue_1
    net_revenue = revenue_tp + revenue_tn - total_fp_penalty - total_fn_opp_cost

    # Confusion matrix DataFrame
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    # Heatmap with readable labels
    fig = px.imshow(cm_df, color_continuous_scale="Blues", title="Confusion Matrix")
    fig.update_layout(title_x=0.5)  # center the title

    annotations = []
    for i, row in enumerate(cm_df.index):
        for j, col in enumerate(cm_df.columns):
            if i == 0 and j == 0:
                label = f"TN={cm_df.iloc[i,j]}"
            elif i == 0 and j == 1:
                label = f"FP={cm_df.iloc[i,j]}"
            elif i == 1 and j == 0:
                label = f"FN={cm_df.iloc[i,j]}"
            else:
                label = f"TP={cm_df.iloc[i,j]}"
            # White text if background is dark, black otherwise
            text_color = "white" if cm_df.iloc[i,j] > cm_df.values.max()/2 else "black"
            annotations.append(
                dict(x=j, y=i, text=label, showarrow=False, font=dict(size=16, color=text_color))
            )
    fig.update_layout(annotations=annotations)

    acc_text = f"Accuracy at threshold {threshold:.2f}: {accuracy:.3f} | Net Revenue = ${net_revenue:,.2f}"
    breakdown_text = (
        f"TP Revenue = ${revenue_tp:,.2f} | TN Revenue = ${revenue_tn:,.2f} | "
        f"FP Penalty = ${total_fp_penalty:,.2f} | FN Opportunity Cost = ${total_fn_opp_cost:,.2f} | "
        f"Net Revenue = ${net_revenue:,.2f}"
    )

    return acc_text, breakdown_text, fig

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))
