from dash import html
from dash_labs.plugins import register_page

register_page(
    __name__,
    path="/",
    top_nav=True,
    order=2
    )

layout = html.Div("Presentación y contexto")
