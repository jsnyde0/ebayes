import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from django.conf import settings

def plot_sales_vs_predictor(date, sales, predictor, currencies):
    """
    Function to plot Sales vs a predictor (e.g., Facebook Ad Spend).
    If the currencies are the same, it will plot on a single axis, otherwise on two.
    
    Args:
        date (pd.Series or pd.DataFrame): Time series data for the x-axis (date).
        sales (pd.Series): Sales data for the y-axis.
        predictor (pd.Series): Predictor data (e.g., Facebook Ad Spend) for the y-axis.
        currencies (dict): Dictionary containing currency information for each Series.
    """
    if len(date) != len(sales) or len(date) != len(predictor):
        raise ValueError("Date, sales, and predictor must have the same length")

    if currencies[sales.name] == currencies[predictor.name]:
        return plot_sales_vs_predictor_single_axis(date, sales, predictor, currencies)
    else:
        return plot_sales_vs_predictor_double_axis(date, sales, predictor, currencies)

def plot_sales_vs_predictor_double_axis(date, sales, predictor, currencies):
    """
    Function to plot Sales vs a predictor (e.g., Facebook Ad Spend) with two y-axes.
    
    Args:
        date (pd.Series or pd.DataFrame): Time series data for the x-axis (date).
        sales (pd.Series): Sales data for the y-axis.
        predictor (pd.Series): Predictor data (e.g., Facebook Ad Spend) for the y-axis.
        currencies (dict): Dictionary containing currency information for each Series.
    """
    fig = go.Figure()

    grid_color = settings.DAISYUI_COLORS['base-300']
    green = settings.DAISYUI_COLORS['success']
    red = settings.DAISYUI_COLORS['error']

    # Add sales trace
    fig.add_trace(
        go.Scatter(
            x=date,
            y=sales,
            name=sales.name,
            mode='lines+markers',
            line=dict(color=green),
            marker=dict(color=green),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=date,
            y=predictor,
            name=predictor.name,
            mode='lines+markers',
            line=dict(color=red),
            marker=dict(color=red),
            yaxis='y2'
        )
    )

    fig.update_layout(
        title=f'{sales.name} vs {predictor.name}',
        xaxis=dict(
            title='Date',
            gridcolor='rgba(80, 80, 80, 0.2)',  # Dark grey with some transparency
            zerolinecolor='rgba(80, 80, 80, 0.2)'
        ),
        yaxis=dict(
            title=f'{sales.name} ({currencies[sales.name]})',
            titlefont=dict(color=green),
            tickfont=dict(color=green),
            gridcolor=grid_color,  # Dark grey with some transparency
            zerolinecolor=grid_color,
            range=[0, max(sales) * 1.1]
        ),
        yaxis2=dict(
            title=f'{predictor.name} ({currencies.get(predictor.name) or "#"})',
            titlefont=dict(color=red),
            tickfont=dict(color=red),
            gridcolor=grid_color,  # Dark grey with some transparency
            zerolinecolor=grid_color,
            overlaying='y',
            side='right',
            range=[0, max(predictor) * 1.1]
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Overall figure background
        plot_bgcolor='rgba(0,0,0,0)'    # Plot area background
    )

    # Convert the Plotly figure to HTML for embedding in the template
    graph_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    return graph_html


def plot_sales_vs_predictor_single_axis(date, sales, predictor, currencies):
    """
    Function to plot Sales vs a predictor (e.g., Facebook Ad Spend) with a single y-axis.
    
    Args:
        date (pd.Series or pd.DataFrame): Time series data for the x-axis (date).
        sales (pd.Series): Sales data for the y-axis.
        predictor (pd.Series): Predictor data (e.g., Facebook Ad Spend) for the y-axis.
        currencies (dict): Dictionary containing currency information for each Series.
    
    Returns:
        str: Plotly figure rendered as HTML.
    """

    # Combine the data into a single DataFrame
    df = sales.to_frame(name=sales.name).assign(Date=date)
    df[predictor.name] = predictor
    sales_currency = currencies[sales.name]
    # predictor_currency = currencies[predictor.name]

    green = 'rgba(76, 175, 80, 1)'
    red = 'rgba(215, 90, 90, 1)'

    # Create the Plotly line chart
    fig = px.line(
        df, 
        x='Date', 
        y=[sales.name, predictor.name], 
        title=f'{sales.name} vs {predictor.name}',
        labels={'value': f'Value ({sales_currency})', 'variable': 'Metric'},
        markers=True,
        color_discrete_map={
            sales.name: green,
            predictor.name: red
        }
    )

    # Set background to transparent
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Overall figure background
        plot_bgcolor='rgba(0,0,0,0)'    # Plot area background
    )

    # Convert the Plotly figure to HTML for embedding in the template
    graph_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    return graph_html