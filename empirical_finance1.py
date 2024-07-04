import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.title("Failure of OLS on Nonstationary Series")
# Add a selectbox to the sidebar:
n_sim = st.sidebar.selectbox("Simulation Trials?", (1000, 100, 10000, 10_000))
st.sidebar.write("Simulation Trials ", n_sim)
n_time = st.sidebar.selectbox("Time Periods?", (600, 200, 500, 800))
st.sidebar.write("Total Time Steps ", n_time)
mu = st.sidebar.slider("mu", 0.0, 100.0, 0.5)
sigma = st.sidebar.slider("sigma", 0.0, 100.0, 4.0)
st.sidebar.write(f"Set simulated distribution with mu", mu, "% sigma", sigma, "%")


"This article demonstrate the difference between time series model (non-stationary) versus random norm series and how it impact the modeling results."
st.caption(
    "*** PS. I built this project mainly for learning streamlit. This project feels like a good starter project where I can check functionality and provide some interaction with users while gaining hands-on experience***"
)
st.divider()  # 👈 Draws a horizontal rule
st.subheader("Simulation Setup")
st.write(
    "Simulate T time series observations each of of the following two return series N times"
)

st.latex(
    r"""
    r_{1, t} = \mu + \sigma \epsilon_{1, t} \\
    r_{2, t} = \mu + \sigma \epsilon_{2, t}
    """
)

"For each of the N time-series, regress:"
st.latex(
    r"""
    r_{1, t} = \alpha + \beta r_{2, t} + \epsilon_{t}
    """
)


"Next, construct price series based on each return using:"
st.latex(
    r"""
    p_{1, t} = p_{1, t-1} + r_{1, t} \\
    p_{2, t} = p_{2, t-1} + r_{2, t}
    """
)

st.markdown("And we set $p_{1, 0}, p_{2,0}$ = 0")
"Similarly, we repeat the regression exercise"
st.latex(
    r"""
    r_{1, t} = \alpha + \beta r_{2, t} + \epsilon_{t}
    """
)


def regression(X, y):
    model = LinearRegression().fit(X, y)
    beta = model.coef_

    pred_y = model.predict(X)
    var_e = sum((y - pred_y) ** 2) / (X.shape[0] - X.shape[1])
    var_b = var_e * (np.linalg.inv(np.dot(X.T, X)).diagonal())
    std_b = np.sqrt(var_b)
    ts_b = beta / std_b
    return beta, ts_b, std_b


def pricing_simulation(N, T, mu, sigma):
    mu = mu / 100  # 0.5%
    sigma = sigma / 100  # 4%
    r1 = np.zeros((N, T))
    r2 = np.zeros((N, T))
    p1 = np.zeros((N, T + 1))
    p2 = np.zeros((N, T + 1))
    beta_ret = np.zeros(N)
    beta_prc = np.zeros(N)
    t_ret = np.zeros(N)
    t_prc = np.zeros(N)
    std_b_ret = np.zeros(N)
    std_b_prc = np.zeros(N)

    for i in range(N):
        epsilon1 = np.random.normal(0, 1, T)
        epsilon2 = np.random.normal(0, 1, T)
        r1[i] = mu + sigma * epsilon1
        r2[i] = mu + sigma * epsilon2
        # Compute the price series
        for t in range(1, T + 1):
            p1[i, t] = p1[i, t - 1] + r1[i, t - 1]
            p2[i, t] = p2[i, t - 1] + r2[i, t - 1]

    # Perform regression for each time series and store the beta values
    for i in range(N):
        beta_ri, t_ri, std_b_ri = regression(r2[i].reshape(-1, 1), r1[i])
        beta_pi, t_pi, std_b_pi = regression(p2[i, 1:].reshape(-1, 1), p1[i, 1:])
        beta_ret[i] = beta_ri[0]
        beta_prc[i] = beta_pi[0]
        t_ret[i] = t_ri[0]
        t_prc[i] = t_pi[0]
        std_b_ret = std_b_ri[0]
        std_b_prc = std_b_pi[0]
    return beta_ret, beta_prc, t_ret, t_prc, std_b_ret, std_b_prc


beta_ret, beta_prc, t_ret, t_prc, std_b_ret, std_b_prc = pricing_simulation(
    n_sim, n_time, mu, sigma
)

## not neccesarily needed
# fig = make_subplots(rows=1, cols=2)

# trace0 = go.Histogram(histfunc="count", x=beta_ret)
# trace1 = go.Histogram(histfunc="count", x=t_ret, xbins=dict(size=0.1))
# fig.append_trace(trace0, 1, 1)
# fig.append_trace(trace1, 1, 2)
# fig.update_layout(coloraxis=dict(colorscale="Bluered_r"), showlegend=False)
# st.plotly_chart(fig, use_container_width=True)

##====== Ploting ====
st.subheader("Stationary Results")
hist_data = [beta_ret, t_ret]
group_labels = ["beta coefficient", "t-stats"]
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])
st.plotly_chart(fig, use_container_width=True)


dataframe = pd.DataFrame(
    np.array([np.mean(beta_ret), np.mean(std_b_ret)]).reshape(1, 2),
    columns=["Mean Beta", "Mean Standard Error"],
)
st.dataframe(dataframe)


st.divider()  # 👈 Draws a horizontal rule

st.subheader("Nonstationary Results")
hist_data = [beta_prc, t_prc]
group_labels = ["beta coefficient", "t-stats"]
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])
st.plotly_chart(fig, use_container_width=True)

dataframe = pd.DataFrame(
    np.array([np.mean(beta_prc), np.mean(std_b_prc)]).reshape(1, 2),
    columns=["Mean Beta", "Mean Standard Error"],
)
st.dataframe(dataframe)

st.divider()
st.subheader("Summary")
"From the simulation, we observe that for regular series, the OLS results are acceptable. Although we cannot reject the null hypothesis of beta = 0, the estimates remain within reasonable bounds."

"However, when we examine the results from non-stationary series using OLS, interestingly (or not really?) we encounter significant issues. The estimated beta is far from zero, and the associated t-statistics are strikingly inaccurate, deviating significantly from expected values."

"This clearly indicates a failure of OLS in the presence of non-stationary data. The primary difference in the second case is the non-stationarity of the series, which fundamentally disrupts the reliability of OLS estimates. Now we can easily see why simple regression model is not applicable for time series data"
