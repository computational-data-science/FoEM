import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

def slide_style(fig, title, yaxis_title, xaxis_title="Time (daily data)"):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(size=18),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    return fig

def simulate_ar1(phi, n=150, seed=42):
    np.random.seed(seed)
    epsilon = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i-1] + epsilon[i]
    return y

def main():
    st.sidebar.title("Navigation")
    # Define the tabs
    tabs = ["Intuition Unit Root", "Overview", "Identify Components of Non-Stationarity", 
        "Dealing with Non-Stationarity", 
        "Outlook: What did we gain?", "What did we learn?"]
    selected_tab = st.sidebar.radio("", tabs)

    # Load the selected tab
    if selected_tab == "Intuition Unit Root":
        load_home()
    elif selected_tab == "Overview":
        wiederholung()
    elif selected_tab == "Identify Components of Non-Stationarity":
        load_analysis()
    elif selected_tab == "Dealing with Non-Stationarity":
        load_milch_data()
    elif selected_tab == "Outlook: What did we gain?":
        load_milch_forecast()

def load_home():
    st.title("Intuition Unit Root")

    st.markdown("""
        We know **White Noise** $X_t = Z_t$ and a **Random Walk** $X_t = X_{t-1} + Z_t$
        """)

    st.markdown("""Reality: Influence of $X_{t-1}$ on $X_t$ often dampend:
            """)

    # Display AR(1) formula
    st.latex(r"X_t = \phi \cdot X_{t-1} + Z_t")

    # Create two columns for the two number inputs
    col1, col2 = st.columns(2)

    # Place the number input for phi in the first column
    phi = col1.number_input("Phi value:", min_value=-2.0, max_value=2.0, value=0.0, step=0.05)

    # Place the number input for seed in the second column
    #seed = col2.number_input("Zufallszustand", min_value=1, max_value=9999, value=42, step=1)
    seed = col2.slider('Random state:', min_value=0, max_value=100, value=42, step=1)

    # Compute and plot the AR(1) process using the given phi and seed
    y = simulate_ar1(phi, seed=seed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode='lines', name=f"phi={phi:.2f}"))
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=len(y),
                y0=0,
                y1=0,
                line=dict(color="black", width=2)
            )
        ]
    )

    st.plotly_chart(slide_style(fig, "Example process for different phi values", yaxis_title="Werte", xaxis_title="Zeit"))

def load_analysis():
    st.title("Components of Non-Stationarity in Time Series")

    col_seed, col_unit_root, col_trend, col_seasonality, col_variance = st.columns(5)

    seed = col_seed.slider('Random state:', min_value=0, max_value=100, value=42, step=1)
    np.random.seed(seed)

    show_unit_root = col_unit_root.checkbox('1')
    show_trend = col_trend.checkbox('2')
    show_seasonality = col_seasonality.checkbox('3')
    show_variance = col_variance.checkbox('4')

    # Parameters
    n = 120  # number of data points

    # Time Series with Unit Root (More Pronounced)
    epsilon = np.random.normal(scale=2.0, size=n)
    epsilon[0] = 0  # Ensure the first element is 0
    ts_unit_root = np.cumsum(epsilon)
    
    # Time Series with Linear Trend
    alpha, beta = 0, 0.5
    ts_trend = alpha + beta * np.arange(n) + np.random.normal(size=n)

    # Time Series with Seasonality
    seasonal_amplitude = np.array([0, -2, -5, -8, -7, -3, 2, 5, 10, 8, 5, 3])
    ts_seasonality = np.tile(seasonal_amplitude, n // 12) + np.random.normal(size=n)

    # Time Series with Changing Variance (ARCH(1))
    alpha0, alpha1 = 1.0, 1.5
    h = np.zeros(n)
    epsilon_arch = np.random.normal(size=n)
    ts_changing_variance = np.zeros(n)
    for t in range(1, n):
        h[t] = alpha0 + alpha1 * epsilon_arch[t-1]**2
        epsilon_arch[t] = np.random.normal(scale=np.sqrt(h[t]))
        ts_changing_variance[t] = epsilon_arch[t]

    # Combine all the series
    combined_ts = np.random.normal(size=n)+ ts_unit_root + ts_trend + ts_seasonality + ts_changing_variance

    if show_unit_root:
        combined_ts -= ts_unit_root

    if show_trend:
        combined_ts -= ts_trend

    if show_seasonality:
        combined_ts -= ts_seasonality

    if show_variance:
        combined_ts -= ts_changing_variance

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=combined_ts, mode='lines', name='Time Series', line=dict(width=2, color="blue")))

    if show_unit_root:
        fig.add_trace(go.Scatter(y=ts_unit_root, mode='lines', name='Unit Root', line=dict(color='grey')))

    if show_trend:
        fig.add_trace(go.Scatter(y=ts_trend, mode='lines', name='Trend', line=dict(color='grey')))

    if show_seasonality:
        fig.add_trace(go.Scatter(y=ts_seasonality, mode='lines', name='Seasonality', line=dict(color='grey')))

    if show_variance:
        fig.add_trace(go.Scatter(y=ts_changing_variance, mode='lines', name='Heteroscedasticity', line=dict(color='grey')))

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=len(combined_ts),
                y0=0,
                y1=0,
                line=dict(color="black", width=2)
            )
        ])
    st.plotly_chart(slide_style(fig, "Example: Time series of a made up process", yaxis_title="Values", xaxis_title="Time"))

def load_settings():
    st.title("Components of Non-Stationarity in Time Series")

    # Create columns for the seed and checkboxes
    col_seed, col_unit_root, col_trend, col_seasonality, col_variance = st.columns(5)

    # User controls
    seed = col_seed.slider('Random state:', min_value=0, max_value=100, value=42, step=1)
    np.random.seed(seed)

    show_unit_root = col_unit_root.checkbox('Show Unit Root Component', False)
    show_trend = col_trend.checkbox('Show Trend Component', False)
    show_seasonality = col_seasonality.checkbox('Show Seasonal Component', False)
    show_variance = col_variance.checkbox('Shoe Heteroscedastic Component', False)

    # Parameters
    n = 120  # number of data points

    # Time Series with Unit Root (More Pronounced)
    epsilon = np.random.normal(scale=2.0, size=n)
    epsilon[0] = 0  # Ensure the first element is 0
    ts_unit_root = np.cumsum(epsilon)

    # Time Series with Linear Trend
    alpha, beta = 0, 0.5
    ts_trend = alpha + beta * np.arange(n) + np.random.normal(size=n)

    # Time Series with Seasonality
    seasonal_amplitude = np.array([0, -2, -5, -8, -7, -3, 2, 5, 10, 8, 5, 3])
    ts_seasonality = np.tile(seasonal_amplitude, n // 12) + np.random.normal(size=n)

    # Time Series with Changing Variance (ARCH(1))
    alpha0, alpha1 = 1.0, 1.5
    h = np.zeros(n)
    epsilon_arch = np.random.normal(size=n)
    ts_changing_variance = np.zeros(n)
    for t in range(1, n):
        h[t] = alpha0 + alpha1 * epsilon_arch[t-1]**2
        epsilon_arch[t] = np.random.normal(scale=np.sqrt(h[t]))
        ts_changing_variance[t] = epsilon_arch[t]

    # New combined series based on user's checkbox selections
    combined_ts2 = np.zeros(n)
    if show_unit_root:
        combined_ts2 += ts_unit_root
    if show_trend:
        combined_ts2 += ts_trend
    if show_seasonality:
        combined_ts2 += ts_seasonality
    if show_variance:
        combined_ts2 += ts_changing_variance

    # Plot new combined series
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=combined_ts2, mode='lines', name='Time series', line=dict(width=2)))
    fig2.update_layout(
        title="Example: Non-stationary time series",
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=len(combined_ts2),
                y0=0,
                y1=0,
                line=dict(color="black", width=2)
            )
        ]
    )
    st.plotly_chart(fig2)

def load_milch_data():
    st.title("Dealing with Non-Stationarity")
    
    # Read in data
    filtered_data = pd.read_csv("milch_daten.csv", parse_dates=True, index_col=0)
    
    # Display the original plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Mkg'], mode='lines+markers', name='Mkg'))
    st.plotly_chart(slide_style(fig, "Milk from cow Loretta", yaxis_title="Milk in kg"))
    
    st.write("What do we do?")

    # Checkboxes for transformations
    col1, col2 = st.columns(2)
    detrend = col1.checkbox("De-trend")
    first_diff_check = col2.checkbox("First differences")

    # Create plots based on checkboxes
    if detrend or first_diff_check:
        # Convert datetime to ordinal numbers
        time_ordinal = filtered_data.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        milchmenge = filtered_data['Mkg'].values.reshape(-1, 1)

        # Create linear regression model for detrending
        model = LinearRegression()
        model.fit(time_ordinal, milchmenge)

        # Predict using the linear model
        trend = model.predict(time_ordinal)

        # Calculate residuals
        residuals_detrend = milchmenge - trend

        # Calculate first differences
        first_diff = np.diff(milchmenge, axis=0)

        # Plot residuals and first differences
        fig_resid = go.Figure()
        
        if detrend:
            fig_resid.add_trace(go.Scatter(x=filtered_data.index, y=residuals_detrend.flatten(),
                    mode='lines',
                    name='De-trended Data'))
        
        if first_diff_check:
            fig_resid.add_trace(go.Scatter(x=filtered_data.index[1:], y=first_diff.flatten(),
                    mode='lines',
                    name='First Differences'))
        
        fig_resid.update_layout(shapes=[dict(type="line", x0=filtered_data.index[0], x1=filtered_data.index[-1], y0=0, y1=0, line=dict(color="black", width=2))])

        st.plotly_chart(slide_style(fig_resid, "Residues after Dealing with Non-Stationarity", yaxis_title="Residuen"))

def load_milch_forecast():
    st.title("Outlook: Predictions")
    st.write("Predictions - with and without dealing with non-stationarity")
    # Read in data
    filtered_data = pd.read_csv("milch_daten.csv", parse_dates=True, index_col=0)
    
    # Checkboxes for predictions
    #regular_prediction = st.checkbox("Vorhersage ohne Umgang mit Nicht-Stationarität")
    #detrend_prediction = st.checkbox("Vorhersage nach De-Trending")
    #first_diff_prediction = st.checkbox("Vorhersage nach erstem Differenzieren")
    
    # Convert datetime to ordinal numbers
    time_ordinal = filtered_data.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    milchmenge = filtered_data['Mkg'].values

    # Depending on checkboxes, run the ARIMA models
    #if regular_prediction:
        # AR(1) model for regular data
    model_regular = ARIMA(milchmenge[:-21], order=(1,0,0))
    model_regular_fit = model_regular.fit()
    forecast_regular = model_regular_fit.forecast(steps=21)
    
    #if detrend_prediction:
        # De-trending
    model = LinearRegression()
    model.fit(time_ordinal[:-21], milchmenge[:-21])
    trend = model.predict(time_ordinal)
    residuals_detrend = milchmenge - trend

        # AR(1) model for de-trended data
    model_detrend = ARIMA(residuals_detrend[:-21], order=(1,0,0))
    model_detrend_fit = model_detrend.fit()
    forecast_detrend = model_detrend_fit.forecast(steps=21) + trend[-21:]
        
    #if first_diff_prediction:
    first_diff = np.diff(milchmenge)
    # AR(1) model for first differences
    model_diff = ARIMA(first_diff[:-21], order=(1,0,0))
    model_diff_fit = model_diff.fit()
    forecast_diff = np.r_[milchmenge[-22], model_diff_fit.forecast(steps=21)].cumsum()
    
    # Now plot the predictions using Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=filtered_data.index[:-21], y=milchmenge[:], mode='lines', name='Training data'))
    fig.add_trace(go.Scatter(x=filtered_data.index[-21:], y=milchmenge[-21:], mode='lines', name='Testdata (true values)'))
    
    #if regular_prediction:
    fig.add_trace(go.Scatter(x=filtered_data.index[-21:], y=forecast_regular, mode='lines', name='Predication with dealing with non-stationarity', visible='legendonly'))
    
    #if detrend_prediction:
    fig.add_trace(go.Scatter(x=filtered_data.index[-21:], y=forecast_detrend, mode='lines', name='Prediction after de-trending', visible='legendonly'))
        
    #if first_diff_prediction:
    fig.add_trace(go.Scatter(x=filtered_data.index[-21:], y=forecast_diff, mode='lines', name='Prediction after taking first differences', visible='legendonly'))

    fig = slide_style(fig, "Simple predictions for Loretta's milk", "Milk in kg")  # Assuming you've defined slide_style previously
    fig.update_layout(legend=dict(y=-0.3, orientation="h"), height=600)
    
    st.plotly_chart(fig)

def wiederholung():
    st.title("Overview")
    
    st.markdown("""
    We distinguish between these components / causes of non-stationarity
    - Existence of a **trend** in the time series (e.g. linear, non-linear, rising, falling, ...),
    - **Seasonal patterns** that are repeated at regular intervals,
    - **Heteroscedasticity**, where the variance can change over time,
    - Presence of a **unit root**.
    """)

if __name__ == "__main__":
    main()
