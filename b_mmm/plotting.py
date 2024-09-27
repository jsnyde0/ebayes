import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from django.conf import settings

COLORS = {
    'grid': settings.DAISYUI_COLORS['base-300'],
    'text': settings.DAISYUI_COLORS['base-content'],
    'sales': settings.DAISYUI_COLORS['success'],
    'predictor': settings.DAISYUI_COLORS['error'],
    'transparent': 'rgba(0,0,0,0)',
}

def plot_sales_vs_predictor(date, sales, predictor, currencies):
    """
    Plot Sales vs a predictor (e.g., Facebook Ad Spend).
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
    """Plot Sales vs a predictor with two y-axes."""
    fig = go.Figure()

    # Add sales trace
    fig.add_trace(
        go.Scatter(
            x=date,
            y=sales,
            name=sales.name,
            mode='lines+markers',
            line=dict(color=COLORS['sales']),
            marker=dict(color=COLORS['sales']),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=date,
            y=predictor,
            name=predictor.name,
            mode='lines+markers',
            line=dict(color=COLORS['predictor']),
            marker=dict(color=COLORS['predictor']),
            yaxis='y2'
        )
    )

    fig.update_layout(
        title=f'{sales.name} vs {predictor.name}',
        xaxis=dict(
            title=dict(text='Date', font=dict(color=COLORS['text'])),
            tickfont=dict(color=COLORS['text']),
            gridcolor=COLORS['grid'],  # Dark grey with some transparency
            zerolinecolor=COLORS['grid']
        ),
        yaxis=dict(
            title=f'{sales.name} ({currencies[sales.name]})',
            titlefont=dict(color=COLORS['sales']),
            tickfont=dict(color=COLORS['sales']),
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            range=[0, max(sales) * 1.1]
        ),
        yaxis2=dict(
            title=f'{predictor.name} ({currencies.get(predictor.name) or "#"})',
            titlefont=dict(color=COLORS['predictor']),
            tickfont=dict(color=COLORS['predictor']),
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            overlaying='y',
            side='right',
            range=[0, max(predictor) * 1.1]
        ),
        legend=dict(font=dict(color=COLORS['text'])),
        paper_bgcolor=COLORS['transparent'],
        plot_bgcolor=COLORS['transparent']
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def plot_sales_vs_predictor_single_axis(date, sales, predictor, currencies):
    """Plot Sales vs a predictor with a single y-axis."""
    df = sales.to_frame(name=sales.name).assign(Date=date)
    df[predictor.name] = predictor
    sales_currency = currencies[sales.name]

    fig = px.line(
        df, 
        x='Date', 
        y=[sales.name, predictor.name], 
        title=f'{sales.name} vs {predictor.name}',
        labels={'value': f'Value ({sales_currency})', 'variable': ''},
        markers=True,
        color_discrete_map={
            sales.name: COLORS['sales'],
            predictor.name: COLORS['predictor']
        }
    )

    fig.update_layout(
        paper_bgcolor=COLORS['transparent'],
        plot_bgcolor=COLORS['transparent'],
        xaxis=dict(
            title=dict(text='Date', font=dict(color=COLORS['text'])),
            tickfont=dict(color=COLORS['text']),
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid']
        ),
        yaxis=dict(
            title=dict(text=f'Value ({sales_currency})', font=dict(color=COLORS['text'])),
            tickfont=dict(color=COLORS['text']),
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid']
        ),
        legend=dict(font=dict(color=COLORS['text'])),
        title=dict(font=dict(color=COLORS['text']))
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs=False)