# Standard Imports
import plotly.express as px
import dash_bootstrap_components as dbc

from dash_labs.plugins import register_page
from dash import dcc, html, Input, Output, callback
from .side_bar import sidebar
from app_dataframe import df_dpto_day, df_dpto_month, df_dpto_year

register_page(__name__)

# ======================== Plotly Graphs

def layout():
    return dbc.Row([dbc.Col('would map', width=12),
                    ])

