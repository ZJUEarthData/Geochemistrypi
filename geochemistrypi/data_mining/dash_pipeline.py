import os

import dash
import flask
import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html
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
            # 新增：回归功能界面
            html.H2(children="Part 2: Regression Analysis"),
            html.Div(
                [
                    html.Label("Select X variables (features):"),
                    dcc.Dropdown(
                        id="x-variables-dropdown",
                        multi=True,
                        placeholder="Select X variables...",
                    ),
                ],
                style={"margin": "10px"},
            ),
            html.Div(
                [
                    html.Label("Select Y variables (targets) - Supports multiple Y columns:"),
                    dcc.Dropdown(
                        id="y-variables-dropdown",
                        multi=True,
                        placeholder="Select Y variables (can select multiple)...",
                    ),
                ],
                style={"margin": "10px"},
            ),
            html.Div(
                [
                    html.Label("Select Regression Model:"),
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[
                            {"label": "Linear Regression", "value": "Linear Regression"},
                            {"label": "Random Forest", "value": "Random Forest"},
                            {"label": "XGBoost", "value": "XGBoost"},
                            {"label": "Support Vector Machine", "value": "Support Vector Machine"},
                            {"label": "Decision Tree", "value": "Decision Tree"},
                            {"label": "Gradient Boosting", "value": "Gradient Boosting"},
                            {"label": "Lasso Regression", "value": "Lasso Regression"},
                            {"label": "Ridge Regression", "value": "Ridge Regression"},
                            {"label": "Elastic Net", "value": "Elastic Net"},
                            {"label": "K-Nearest Neighbors", "value": "K-Nearest Neighbors"},
                            {"label": "SGD Regression", "value": "SGD Regression"},
                            {"label": "BayesianRidge Regression", "value": "BayesianRidge Regression"},
                            {"label": "Multi-layer Perceptron", "value": "Multi-layer Perceptron"},
                            {"label": "Polynomial Regression", "value": "Polynomial Regression"},
                            {"label": "Extra-Trees", "value": "Extra-Trees"},
                        ],
                        value="Linear Regression",
                        placeholder="Select a regression model...",
                    ),
                ],
                style={"margin": "10px"},
            ),
            html.Button("Run Regression", id="run-regression-button", n_clicks=0),
            html.Div(id="regression-results"),
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

    # New: Updated variable selection dropdown
    @app.callback(
        [Output("x-variables-dropdown", "options"), Output("y-variables-dropdown", "options")],
        [Input("dataset-dropdown", "value")],
    )
    def update_variable_options(selected_dataset):
        """Update variable options based on the selected dataset."""
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

        options = [{"label": col, "value": col} for col in df.columns]
        return options, options

    # New: Run regression analysis
    @app.callback(
        Output("regression-results", "children"),
        [Input("run-regression-button", "n_clicks")],
        [State("dataset-dropdown", "value"), State("x-variables-dropdown", "value"), State("y-variables-dropdown", "value"), State("model-dropdown", "value")],
    )
    def run_regression(n_clicks, selected_dataset, x_vars, y_vars, model_name):
        """Run regression analysis with selected variables."""
        if n_clicks == 0 or not all([selected_dataset, x_vars, y_vars, model_name]):
            return "Please select dataset, X variables, Y variables, and model."

        try:
            # Get data
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

            # Prepare X and Y data
            X = df[x_vars]
            y = df[y_vars]

            # Check data
            if X.empty or y.empty:
                return "Error: Selected variables contain no data."

            # Show data info
            result_text = f"""
            <h3>Regression Analysis Results</h3>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Number of X Variables:</strong> {len(x_vars)} ({', '.join(x_vars)})</p>
            <p><strong>Number of Y Variables:</strong> {len(y_vars)} ({', '.join(y_vars)})</p>
            <p><strong>Number of Samples:</strong> {len(X)}</p>
            <p><strong>Support Multiple Y Columns:</strong> {'Yes' if len(y_vars) > 1 else 'No'}</p>
            """

            # You can add the actual regression analysis code here
            # Since environment variables and output paths need to be set, only basic information is shown here
            result_text += "<p><em>Note: Complete regression analysis requires setting environment variables and output paths. Please use the CLI version for full analysis.</em></p>"

            return html.Div([html.Div(result_text, dangerouslySetInnerHTML={"__html": result_text})])

        except Exception as e:
            return f"Error: {str(e)}"

    @app.callback(Output("content-div", "style"), [Input("toggle-button", "n_clicks")])
    def toggle_div_visibility(n_clicks):
        if n_clicks and n_clicks % 2 == 1:
            return {"display": "none"}  # Hide the div
        else:
            return {"display": "block"}  # Show the div

    return app
