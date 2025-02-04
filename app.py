import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Read and Prepare CSV Data
# -----------------------------------------------------------------------------
df = pd.read_csv("investor_data.csv")

# Convert 'Announced Date' to datetime (assumes format dd/mm/yyyy)
df['Announced Date'] = pd.to_datetime(df['Announced Date'], format='%d/%m/%Y', errors='coerce')
df['Investor Names'] = df['Investor Names'].fillna('')

# Determine the minimum and maximum year in the dataset
min_year = int(df['Announced Date'].dt.year.min())
max_year = int(df['Announced Date'].dt.year.max())

# Generate a sorted list of all unique investors (for the dropdown)
all_investors = set()
for names in df['Investor Names'].dropna():
    for name in names.split(','):
        all_investors.add(name.strip())
all_investors = sorted(list(all_investors))

# -----------------------------------------------------------------------------
# 2. Function to Create Co-Investment Pairs
# -----------------------------------------------------------------------------
def create_co_investment_pairs(investors):
    """
    Creates all possible (Investor A, Investor B) pairs from a list of investors.
    """
    pairs = []
    for i in range(len(investors)):
        for j in range(i + 1, len(investors)):
            a = investors[i].strip()
            b = investors[j].strip()
            if a and b:
                pairs.append((a, b))
    return pairs

# -----------------------------------------------------------------------------
# 3. Function to Generate the Network Figure with Extra Information, Filtering,
#    and Variable Node Sizes (scaled by number of funding rounds)
# -----------------------------------------------------------------------------
def generate_network_figure(start_year, end_year, min_degree, min_rounds, investor_filter):
    # Filter data by the selected time range
    df_filtered = df[
        (df['Announced Date'].dt.year >= start_year) &
        (df['Announced Date'].dt.year <= end_year)
    ]
    # Falls ein spezifischer Investor ausgewählt wurde, filtere nach diesem Investor
    if investor_filter != "All":
        df_filtered = df_filtered[df_filtered['Investor Names'].str.contains(investor_filter, case=False, na=False)]
    
    # --- Aggregate extra information for each investor ---
    investor_info = {}
    for _, row in df_filtered.iterrows():
        # Hole den Rohwert und behandle NaN, "n/a", "na" oder leere Strings als 0.0
        val = row['Money Raised (in USD)']
        if pd.isna(val) or (isinstance(val, str) and val.strip().lower() in ["n/a", "na", ""]):
            money = 0.0
        else:
            try:
                money = float(val)
            except (ValueError, TypeError):
                money = 0.0
        
        if row['Investor Names']:
            # Split the comma-separated investor names
            investors = [inv.strip() for inv in row['Investor Names'].split(',')]
            for inv in investors:
                if inv not in investor_info:
                    investor_info[inv] = {"rounds": 0, "money": 0.0, "transactions": []}
                investor_info[inv]["rounds"] += 1
                investor_info[inv]["money"] += money
                transaction_detail = (
                    f"{row.get('Transaction Name', 'n/a')} | "
                    f"{row.get('Organization Name', 'n/a')} | "
                    f"{row.get('Funding Stage', 'n/a')}"
                )
                investor_info[inv]["transactions"].append(transaction_detail)
    
    # --- Create the Co-Investment Edges ---
    co_investments = []
    for _, row in df_filtered.iterrows():
        if row['Investor Names']:
            investors = [inv.strip() for inv in row['Investor Names'].split(',')]
            co_investments.extend(create_co_investment_pairs(investors))
    
    # Build the network graph using NetworkX
    G = nx.Graph()
    G.add_edges_from(co_investments)
    
    # Remove nodes with a degree lower than the specified minimum
    nodes_to_remove_degree = [n for n, deg in G.degree() if deg < min_degree]
    G.remove_nodes_from(nodes_to_remove_degree)
    
    # Remove nodes (investors) with less funding rounds than min_rounds
    nodes_to_remove_rounds = [n for n in list(G.nodes()) 
                              if investor_info.get(n, {"rounds": 0})["rounds"] < min_rounds]
    G.remove_nodes_from(nodes_to_remove_rounds)
    
    # Falls keine Knoten übrig sind, liefere eine leere Figure mit einem Hinweis
    if len(G.nodes()) == 0:
        fig = go.Figure()
        fig.update_layout(title="No investors meet the minimum requirements for this time period.")
        return fig
    
    # Compute degree values for color scaling
    node_degrees = dict(G.degree())
    deg_min = min(node_degrees.values()) if node_degrees else 0
    deg_max = max(node_degrees.values()) if node_degrees else 1

    # Compute a 3D spring layout for the graph
    pos_3d = nx.spring_layout(G, dim=3, k=0.2, iterations=30, seed=42)
    
    # --- Prepare Edge Coordinates for Plotly ---
    x_edges, y_edges, z_edges = [], [], []
    for u, v in G.edges():
        x_edges.extend([pos_3d[u][0], pos_3d[v][0], None])
        y_edges.extend([pos_3d[u][1], pos_3d[v][1], None])
        z_edges.extend([pos_3d[u][2], pos_3d[v][2], None])
    
    edge_trace = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(color='grey', width=2),
        hoverinfo='none'
    )
    
    # --- Prepare Node Coordinates and Additional Information ---
    all_nodes = list(G.nodes())
    x_nodes = [pos_3d[n][0] for n in all_nodes]
    y_nodes = [pos_3d[n][1] for n in all_nodes]
    z_nodes = [pos_3d[n][2] for n in all_nodes]
    
    hover_texts = []
    for n in all_nodes:
        text = f"<b>{n}</b><br>Co-Investors: {node_degrees[n]}"
        details = investor_info.get(n)
        if details:
            text += f"<br>Funding Rounds: {details['rounds']}"
            text += f"<br>Total Raised (USD): ${details['money']:,.0f}"
            transactions = list(dict.fromkeys(details['transactions']))
            if len(transactions) > 3:
                transactions = transactions[:3] + ["..."]
            text += f"<br>Transactions: {'; '.join(transactions)}"
        hover_texts.append(text)
    
    # --- Variable Node Sizes ---
    sizes = []
    for n in all_nodes:
        rounds = investor_info.get(n, {"rounds": 1})["rounds"]
        sizes.append(5 + 2 * rounds)
    
    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers+text',
        text=all_nodes,
        textposition='top center',
        hovertemplate='%{customdata}',
        customdata=hover_texts,
        marker=dict(
            size=sizes,
            color=[node_degrees[n] for n in all_nodes],
            colorscale='Viridis',
            cmin=deg_min,
            cmax=deg_max,
            showscale=False,
            colorbar=dict(
                title=dict(text='Co-Investors', font=dict(color='white')),
                tickfont=dict(color='white')
            )
        )
    )
    
    # Build the final figure with a dark theme
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Co-Investment Network {start_year}-{end_year} – Degree-based (min. Co-Investors: {min_degree}, min. Rounds: {min_rounds})",
        title_font_color='white',
        paper_bgcolor='white',
        plot_bgcolor='black',
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showbackground=False, color='white'),
            yaxis=dict(showgrid=False, zeroline=False, showbackground=False, color='white'),
            zaxis=dict(showgrid=False, zeroline=False, showbackground=False, color='white'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        margin=dict(l=0, r=0, b=0, t=100)
    )
    
    return fig

# -----------------------------------------------------------------------------
# 4. Dash App Layout (Full-Screen Dashboard with Smaller Filters)
# -----------------------------------------------------------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

investor_options = [{'label': 'All Investors', 'value': 'All'}] + \
                   [{'label': inv, 'value': inv} for inv in all_investors]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Co-Investment Network Dashboard",
                        className="text-center mb-2",
                        style={"marginTop": "20px", "fontSize": "1.5em", "color": "#822723"}), width=10),
        dbc.Col(
            dbc.Button("View on GitHub", href="https://github.com/your-repo", target="_blank",
                       style={"backgroundColor": "#24292e", "color": "white", "border": "none", "marginRight": "+10px"}),
            width=2, className="d-flex justify-content-end align-items-center"
        )
    ]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Parameters", style={"fontSize": "0.9em", "padding": "0.5em", "fontFamily": "Futura, sans-serif"}),
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            dbc.Label("Investor Perspective:", style={"fontSize": "0.8em", "fontFamily": "Futura, sans-serif"}),
                                            dcc.Dropdown(
                                                id="investor-perspective",
                                                options=investor_options,
                                                value="TechVisionFund", 
                                                clearable=False,
                                                style={"fontSize": "0.8em", "height": "30px", "fontFamily": "Futura, sans-serif"}
                                            )
                                        ],
                                        className="mb-2"
                                    ),
                                    html.Div(
                                        [
                                            dbc.Label("Start Year:", style={"fontSize": "0.8em", "fontFamily": "Futura, sans-serif"}),
                                            dbc.Input(
                                                id="start-year",
                                                type="number",
                                                value=2020,  
                                                min=min_year,
                                                max=max_year,
                                                step=1,
                                                style={"fontSize": "0.8em", "height": "30px", "fontFamily": "Futura, sans-serif"}
                                            )
                                        ],
                                        className="mb-2"
                                    ),
                                    html.Div(
                                        [
                                            dbc.Label("End Year:", style={"fontSize": "0.8em", "fontFamily": "Futura, sans-serif"}),
                                            dbc.Input(
                                                id="end-year",
                                                type="number",
                                                value=2025, 
                                                min=min_year,
                                                max=max_year,
                                                step=1,
                                                style={"fontSize": "0.8em", "height": "30px", "fontFamily": "Futura, sans-serif"}
                                            )
                                        ],
                                        className="mb-2"
                                    ),
                                    html.Div(
                                        [
                                            dbc.Label("Minimum Co-Investors:", style={"fontSize": "0.8em", "fontFamily": "Futura, sans-serif"}),
                                            dbc.Input(
                                                id="min-degree",
                                                type="number",
                                                value=1, 
                                                min=0,
                                                step=1,
                                                style={"fontSize": "0.8em", "height": "30px", "fontFamily": "Futura, sans-serif"}
                                            )
                                        ],
                                        className="mb-2"
                                    ),
                                    html.Div(
                                        [
                                            dbc.Label("Minimum Funding Rounds:", style={"fontSize": "0.8em", "fontFamily": "Futura, sans-serif"}),
                                            dbc.Input(
                                                id="min-rounds",
                                                type="number",
                                                value=1,  # Default-Wert auf 1 gesetzt
                                                min=0,
                                                step=1,
                                                style={"fontSize": "0.8em", "height": "30px", "fontFamily": "Futura, sans-serif"}
                                            )
                                        ],
                                        className="mb-2"
                                    ),
                                    dbc.Button(
                                        "Update Dashboard",
                                        id="update-button",
                                        color="light",
                                        className="w-100",
                                        style={
                                            "fontSize": "0.8em",
                                            "padding": "0.3em",
                                            "backgroundColor": "#822723",
                                            "borderColor": "#822723",
                                            "color": "white",
                                            "fontFamily": "Futura, sans-serif"
                                        }
                                    )
                                ]
                            )
                        ],
                        style={"fontSize": "0.8em", "padding": "0.5em", "fontFamily": "Futura, sans-serif"}
                    ),
                    width=2
                ),
                dbc.Col(
                    dcc.Loading(
                        id="loading-graph",
                        type="default",
                        children=dcc.Graph(
                            id="network-graph",
                            style={"height": "calc(100vh - 100px)"}
                        )
                    ),
                    width=10
                )
            ],
            className="g-0"
        ),
# Footer with Logo-Link
        dbc.Row(
            dbc.Col(
                html.Div(
                    html.A(
                        html.Img(src="assets/tvf.png", style={"height": "30px", "marginTop": "-15px"}),  
                        href="https://tvf.vc",  
                        target="_blank"
                    ),
                    style={"textAlign": "center", "padding": "10px"}
                )
            )
        )
    ],
    fluid=True,
    style={"padding": "0", "margin": "0", "width": "100vw", "height": "100vh", "fontFamily": "Futura, sans-serif"}
)

# -----------------------------------------------------------------------------
# 5. Callback to Update the Graph
# -----------------------------------------------------------------------------
@app.callback(
    Output("network-graph", "figure"),
    Input("update-button", "n_clicks"),
    State("investor-perspective", "value"),
    State("start-year", "value"),
    State("end-year", "value"),
    State("min-degree", "value"),
    State("min-rounds", "value")
)
def update_graph(n_clicks, investor_perspective, start_year, end_year, min_degree, min_rounds):
    if None in (start_year, end_year, min_degree, min_rounds):
        return go.Figure()
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    return generate_network_figure(start_year, end_year, min_degree, min_rounds, investor_perspective)

# -----------------------------------------------------------------------------
# 6. Run the Dash App
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)
