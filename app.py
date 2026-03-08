"""NeuralPipe v0.2 — Dash application entry point."""
import threading
import webbrowser

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)
server = app.server

_nav = dbc.NavbarSimple(
    brand="NeuralPipe",
    brand_href="/",
    color="dark",
    dark=True,
    className="mb-0 px-3",
    children=[
        dbc.NavItem(dbc.NavLink("Route",   href="/",        active="exact")),
        dbc.NavItem(dbc.NavLink("Results", href="/results", active="exact")),
    ],
)

app.layout = html.Div([
    _nav,
    # Shared JSON store — persists across pages in the browser session.
    # Schema: {cloud_path, cloud_bmin, cloud_bmax, cloud_preview_pts,
    #          start, end, candidates}
    dcc.Store(id="store", storage_type="session"),
    dash.page_container,
])

if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8050")).start()
    app.run(debug=True, port=8050, use_reloader=False)
