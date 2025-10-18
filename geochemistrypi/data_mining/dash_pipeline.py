import os
import dash
import flask
import pandas as pd
import numpy as np

from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

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
        routes_pathname_prefix="/",
        suppress_callback_exceptions=True,
    )

    data_regression = read_data("Data_Regression.xlsx")
    data_classification = read_data("Data_Classification.xlsx")
    data_clustering = read_data("Data_Clustering.xlsx")
    data_decomposition = read_data("Data_Decomposition.xlsx")

    user_option = (
        [{"label": "User's Uploaded Data", "value": "user_data"}]
        if (os.path.exists(USER_CSV) or os.path.exists(USER_XLSX))
        else []
    )

    app.layout = html.Div(
        [
            html.H1("Geochemistry π"),
            html.H2("Part 1: Data Loading"),

            dcc.Dropdown(
                id="dataset-dropdown",
                options=user_option
                + [
                    {"label": "Built-in Data For Regression", "value": "data_regression"},
                    {"label": "Built-in Data For Classification", "value": "data_classification"},
                    {"label": "Built-in Data For Clustering", "value": "data_clustering"},
                    {"label": "Built-in Data For Decomposition", "value": "data_decomposition"},
                ],
                value="user_data" if user_option else None,
                placeholder="Select a dataset",
                clearable=True,
            ),

            dash_table.DataTable(id="data-table", columns=[], data=[], page_size=10),

            html.Hr(),
            html.H3("Part 2: Classification (Custom N-Class)"),

            html.Div("Target (label) column:"),
            dcc.Dropdown(id="target-col-dd", placeholder="Select the label column"),

            html.Div("Select target classes (optional):"),
            dcc.Dropdown(id="class-filter-dd", multi=True, placeholder="Select subset of label values"),

            html.Div(style={"height": 8}),
            html.Button("Train (Logistic Regression)", id="train-btn"),
            html.Div(style={"height": 8}),
            html.Div(id="train-metrics"),
            dcc.Graph(id="conf-matrix"),
        ],
        style={"maxWidth": "95%", "margin": "0 auto"},
    )

    def _get_df(selected_dataset: str) -> pd.DataFrame:
        if selected_dataset == "user_data":
            return _load_user_df()
        if selected_dataset == "data_regression":
            return data_regression
        if selected_dataset == "data_classification":
            return data_classification
        if selected_dataset == "data_clustering":
            return data_clustering
        if selected_dataset == "data_decomposition":
            return data_decomposition
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
        opts = []
        for c in df.columns:
            s = df[c].dropna()
            nunique = s.nunique()
            if not np.issubdtype(s.to_numpy().dtype, np.number) or nunique <= 20:
                opts.append({"label": f"{c} (unique={nunique})", "value": c})
        return opts

    @app.callback(
        Output("class-filter-dd", "options"),
        Input("target-col-dd", "value"),
        State("dataset-dropdown", "value"),
    )
    def update_class_options(target_col, dataset_key):
        df = _get_df(dataset_key)
        if not target_col or target_col not in df.columns:
            return []
        classes = sorted(df[target_col].dropna().unique())
        return [{"label": str(c), "value": c} for c in classes]

    @app.callback(
        Output("train-metrics", "children"),
        Output("conf-matrix", "figure"),
        Input("train-btn", "n_clicks"),
        State("dataset-dropdown", "value"),
        State("target-col-dd", "value"),
        State("class-filter-dd", "value"),
        prevent_initial_call=True,
    )
    def train_model(n_clicks, dataset_key, target_col, class_filter):
        if not dataset_key:
            return "Please select a dataset first.", {}
        df = _get_df(dataset_key)
        if df.empty:
            return "Dataset is empty.", {}
        if not target_col or target_col not in df.columns:
            return "Please select a valid target column.", {}

        if class_filter:
            df = df[df[target_col].isin(class_filter)]
            if len(df[target_col].unique()) < 2:
                return "Need at least 2 unique classes to train.", {}

        df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        X = X.select_dtypes(include=["number"])
        X = X.fillna(X.mean()) 
        X = X.dropna(axis=1, how="all")  
        mask = X.notna().all(axis=1)
        X, y = X[mask], y[mask]

        if X.empty:
            return "No numeric features available after preprocessing.", {}

        try:
            strat = y if y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat
            )
        except ValueError as e:
            return f"Split error: {e}", {}

        clf = LogisticRegression(max_iter=1000, multi_class="auto")
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            return f"Training error: {e}", {}

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        labels = sorted(pd.unique(y))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig = px.imshow(
            cm,
            text_auto=True,
            x=[str(i) for i in labels],
            y=[str(i) for i in labels],
            labels=dict(x="Predicted", y="True", color="Count"),
            title="Confusion Matrix",
        )
        return f"Accuracy: {acc:.4f} | Classes: {labels}", fig

    return app
