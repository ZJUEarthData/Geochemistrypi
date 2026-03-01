import os
import dash
import flask
import pandas as pd
import numpy as np
import requests  

from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

from .data.data_readiness import read_data

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FAKE_DATABASE_DIR = os.path.join(CURRENT_DIR, "fake_database")
os.makedirs(FAKE_DATABASE_DIR, exist_ok=True)

USER_XLSX = os.path.join(FAKE_DATABASE_DIR, "user_data.xlsx")
USER_CSV = os.path.join(FAKE_DATABASE_DIR, "user_data.csv")

def _load_user_df() -> pd.DataFrame:
    if os.path.exists(USER_CSV):
        return pd.read_csv(USER_CSV)
    if os.path.exists(USER_XLSX):
        return pd.read_excel(USER_XLSX)
    return pd.DataFrame()

def dash_pipeline(requests_pathname_prefix: str = "/dash/") -> dash.Dash:
    """主 Dash 应用"""
    req_prefix = requests_pathname_prefix or "/dash/"
    if not req_prefix.startswith("/"):
        req_prefix = "/" + req_prefix
    if not req_prefix.endswith("/"):
        req_prefix += "/"

    server = flask.Flask(__name__)
    server.secret_key = os.environ.get("secret_key", "secret")

    @server.route("/ping")
    def _ping():
        return "pong"

    app = dash.Dash(
        __name__,
        server=server,
        requests_pathname_prefix=req_prefix,
        suppress_callback_exceptions=True,
    )

    data_regression = read_data("Data_Regression.xlsx")
    data_classification = read_data("Data_Classification.xlsx")
    
    user_option = (
        [{"label": "User's Uploaded Data", "value": "user_data"}]
        if (os.path.exists(USER_CSV) or os.path.exists(USER_XLSX))
        else []
    )

    app.layout = html.Div(
        [
            html.H1("Geochemistry π - Web ML System"),
            html.H2("Part 1: Data Loading"),

            dcc.Dropdown(
                id="dataset-dropdown",
                options=user_option + [
                    {"label": "Built-in Data For Classification", "value": "data_classification"},
                ],
                value="user_data" if user_option else None,
                placeholder="Select a dataset",
                clearable=True,
            ),

            dash_table.DataTable(id="data-table", columns=[], data=[], page_size=10),

            html.Hr(),
            html.H3("Part 2: Multi-Classification Training"),

            html.Div("1. Select Target (label) column:"),
            dcc.Dropdown(id="target-col-dd", placeholder="Select the label column"),

            html.Div("2. Select Machine Learning Model:"),
            dcc.Dropdown(
                id="model-dd", 
                options=[
                    {"label": "XGBoost", "value": "XGBoost"},
                    {"label": "Random Forest", "value": "Random Forest"},
                    {"label": "Decision Tree", "value": "Decision Tree"},
                    {"label": "Support Vector Machine", "value": "Support Vector Machine"}
                ],
                value="Random Forest",
                placeholder="Select a model"
            ),

            html.Div(style={"height": 20}),
            html.Button(" Run Training via API", id="train-btn", n_clicks=0, style={"fontSize": "18px", "padding": "10px", "backgroundColor": "#4CAF50", "color": "white"}),
            
            html.Div(style={"height": 20}),
            html.Div(id="train-metrics", style={"padding": "20px", "border": "1px solid #ccc", "backgroundColor": "#f9f9f9"}),
        ],
        style={"maxWidth": "900px", "margin": "40px auto", "fontFamily": "Arial, sans-serif"},
    )

    def _get_df(selected_dataset: str) -> pd.DataFrame:
        if selected_dataset == "user_data":
            return _load_user_df()
        if selected_dataset == "data_classification":
            return data_classification
        return pd.DataFrame()

    @app.callback(
        [Output("data-table", "columns"), Output("data-table", "data")],
        Input("dataset-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_table(selected_dataset):
        df = _get_df(selected_dataset)
        columns = [{"name": col, "id": col} for col in df.columns]
        data = df.to_dict("records")
        return columns, data

    @app.callback(
        Output("target-col-dd", "options"),
        Input("dataset-dropdown", "value"),
    )
    def populate_target_options(selected_dataset):
        df = _get_df(selected_dataset)
        if df.empty:
            return []
        opts = [{"label": c, "value": c} for c in df.columns]
        return opts

    @app.callback(
        Output("train-metrics", "children"),
        Input("train-btn", "n_clicks"),
        State("target-col-dd", "value"),
        State("model-dd", "value"),
        prevent_initial_call=True,
    )
    def train_model_via_api(n_clicks, target_col, model_name):
        if n_clicks == 0:
            return ""
        if not target_col:
            return html.Div(" Please select a target column first.", style={"color": "red"})
        if not model_name:
            return html.Div(" Please select a model.", style={"color": "red"})

        payload = {
            "dataset_id": 1, 
            "target_column": target_col,
            "model_name": model_name,
            "label_mapping": {
                "type": "quantile",
                "num_classes": 4,
                "labels": ["Level_1", "Level_2", "Level_3", "Level_4"]
            }
        }

        try:
            response = requests.post("http://127.0.0.1:8000/data-mining/run-classification", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return html.Div([
                    html.H4(" Backend Execution Successful!", style={"color": "green"}),
                    html.P(f"Message: {result.get('message', 'Done')}"),
                    html.P(f"Status: {result.get('status', 'Success')}"),
                    html.P("Model artifacts and predictions have been successfully saved to the 'output' directory.", style={"fontWeight": "bold"})
                ])
            elif response.status_code == 401:
                return html.Div(" Unauthorized: Please login via Swagger or update your API Token.", style={"color": "red"})
            else:
                return html.Div(f" Backend Error {response.status_code}: {response.text}", style={"color": "red"})
                
        except Exception as e:
            return html.Div(f" Connection Error: Ensure FastAPI server is running. Details: {str(e)}", style={"color": "red"})

    return app