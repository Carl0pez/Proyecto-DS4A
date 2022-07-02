from dash import html
from dash_labs.plugins import register_page

register_page(
    __name__,
    top_nav=True,
    order=5
    )


layout = html.Div("Página modelo")
