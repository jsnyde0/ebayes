import plotly.express as px
import plotly.io as pio

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
    df = sales.to_frame(name='Sales').assign(Date=date, Predictor=predictor)

    # Create the Plotly line chart
    fig = px.line(
        df, 
        x='Date', 
        y=['Sales', 'Predictor'], 
        title=f'Sales and {predictor.name} Over Time',
        labels={'value': 'Amount (USD)', 'variable': 'Metric'},
        markers=True  # Add markers to the lines
    )

    # Convert the Plotly figure to HTML for embedding in the template
    graph_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    return graph_html