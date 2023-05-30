import os

import dash
import flask
import pandas as pd
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

from .data.data_readiness import read_data

# Mock the database
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FAKE_DATABASE_DIR = os.path.join(CURRENT_DIR, "fake_database")
os.makedirs(FAKE_DATABASE_DIR, exist_ok=True)


def dash_pipeline(requests_pathname_prefix: str = None) -> dash.Dash:
    """The web applicatioin for Geochemistry π."""

    server = flask.Flask(__name__)
    server.secret_key = os.environ.get("secret_key", "secret")

    app = dash.Dash(__name__, server=server, requests_pathname_prefix=requests_pathname_prefix)

    # Built-in datasets, load in advance to decrease I/O cost
    data_regression = read_data("Data_Regression.xlsx")
    data_classification = read_data("Data_Classification.xlsx")
    data_clustering = read_data("Data_Clustering.xlsx")
    data_decomposition = read_data("Data_Decomposition.xlsx")

    user_data_path = os.path.join(FAKE_DATABASE_DIR, "user_data.xlsx")

    app.layout = html.Div(
        [
            html.H1(children="Geochemistry π"),
            html.H2(children="Part 1: Data Loading"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[
                    {"label": "User's Uploaded Data", "value": "user_data"},
                    {"label": "Built-in Data For Regression", "value": "data_regression"},
                    {"label": "Built-in Data For Classification", "value": "data_classification"},
                    {"label": "Built-in Data For Clustering", "value": "data_clustering"},
                    {"label": "Built-in Data For Decomposition", "value": "data_decomposition"},
                ],
                value=None,
                placeholder="Select a dataset",
            ),
            dash_table.DataTable(
                id="data-table",
                columns=[],
                data=[],
                page_size=10,
            ),
            html.Button("Toggle", id="toggle-button"),
            html.Div(id="content-div", children="Content to be hidden or shown"),
        ]
    )

    @app.callback(
        [Output("data-table", "columns"), Output("data-table", "data")],
        [Input("dataset-dropdown", "value")],
    )
    def update_table(selected_dataset):
        """Update the table based on the selected dataset."""
        df = pd.DataFrame()
        if selected_dataset == "user_data":
            df = pd.read_excel(user_data_path)
        elif selected_dataset == "data_regression":
            df = data_regression
        elif selected_dataset == "data_classification":
            df = data_classification
        elif selected_dataset == "data_clustering":
            df = data_clustering
        elif selected_dataset == "data_decomposition":
            df = data_decomposition
        columns = [{"name": col, "id": col} for col in df.columns]
        data = df.to_dict("records")
        return columns, data

    @app.callback(Output("content-div", "style"), [Input("toggle-button", "n_clicks")])
    def toggle_div_visibility(n_clicks):
        if n_clicks and n_clicks % 2 == 1:
            return {"display": "none"}  # Hide the div
        else:
            return {"display": "block"}  # Show the div

    return app
