import plotly.express as px
import plotly.io as pio
from django.conf import settings
    
def plot_sales_vs_predictor(date, sales, predictor):
    """
    Function to plot Sales vs a predictor (e.g., Facebook Ad Spend).
    
    Args:
        date (pd.Series or pd.DataFrame): Time series data for the x-axis (date).
        sales (pd.Series): Sales data for the y-axis.
        predictor (pd.Series): Predictor data (e.g., Facebook Ad Spend) for the y-axis.
    
    Returns:
        str: Plotly figure rendered as HTML.
    """
    if len(date) != len(sales) or len(date) != len(predictor):
        raise ValueError("Date, sales, and predictor must have the same length")

    # Combine the data into a single DataFrame
    df = sales.to_frame(name=sales.name).assign(Date=date)
    df[predictor.name] = predictor

    muted_green = 'rgba(76, 175, 80, 1)'  # More muted green, fully opaque
    muted_red = 'rgba(215, 90, 90, 1)'

    # Create the Plotly line chart
    fig = px.line(
        df, 
        x='Date', 
        y=[sales.name, predictor.name], 
        title=f'{sales.name} vs {predictor.name}',
        labels={'value': 'Amount (USD)', 'variable': 'Metric'},
        markers=True,
        color_discrete_map={
            sales.name: muted_green,
            predictor.name: muted_red
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