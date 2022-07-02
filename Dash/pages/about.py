from dash import html
from dash_labs.plugins import register_page
import dash_bootstrap_components as dbc

register_page(
    __name__,
    top_nav=True,
    order=6
    )

texto1 = open("assets/texto1.txt", mode='r')
texto2 = open("assets/texto2.txt", mode='r')
texto3 = open("assets/texto3.txt", mode='r')
texto4 = open("assets/texto4.txt", mode='r')
texto5 = open("assets/texto5.txt", mode='r')

imagen1 = '/assets/vonNeumann.gif'
imagen2 = '/assets/nash.jpg'
imagen3 = '/assets/dantzig.jpg'
imagen4 = '/assets/turing.jpg'
imagen5 = '/assets/gauss.jpg'

layout = dbc.Row([
    dbc.Col('Equipo:', width=2),
    dbc.Col(html.Img(src=imagen1, height="120px"), width=2),
    dbc.Col(html.Img(src=imagen2, height="120px"), width=2),
    dbc.Col(html.Img(src=imagen3, height="120px"), width=2),
    dbc.Col(html.Img(src=imagen4, height="120px"), width=2),
    dbc.Col(html.Img(src=imagen5, height="120px"), width=2),
    dbc.Col('', width=2),
    dbc.Col(texto1.read(), width=2),
    dbc.Col(texto2.read(), width=2),
    dbc.Col(texto3.read(), width=2),
    dbc.Col(texto4.read(), width=2),
    dbc.Col(texto5.read(), width=2),
    ])

#html.Div([texto,])