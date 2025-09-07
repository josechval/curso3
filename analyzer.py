import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.stats import skew, kurtosis

# This script performs a VaR-based portfolio optimization for the "High Tech" ETF,
# analyzes its performance, and generates the required data and plots for the investor's prospectus.

# -------------------------------------------------------------------------------------
# Phase 1: Setup and Data Collection with a Robust Method
# -------------------------------------------------------------------------------------

print("Phase 1: Setting up environment and collecting data...")

# New, verified list of 50 assets for the "High Tech" ETF and the S&P 500 benchmark
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'AMD', 'CRM',
    'ADBE', 'PYPL', 'INTC', 'CSCO', 'CMCSA', 'QCOM', 'TXN', 'ORCL', 'SAP', 'SHOP',
    'NOW', 'SNOW', 'MDB', 'SQ', 'DOCU', 'OKTA', 'UBER', 'LYFT', 'SPOT', 'ROKU',
    'WDAY', 'TEAM', 'CRWD', 'S', 'BABA', 'JD', 'BIDU', 'V', 'MA', 'AVGO',
    'KO', 'JNJ', 'PG', 'HD', 'MCD', 'DIS', 'NKE', 'BAC', 'JPM', 'XOM'
]
benchmarks = ['SPY']
all_tickers = tickers + benchmarks

# Set the timeframe for at least 10 years of daily observations
start_date = '2015-09-06'
end_date = '2025-09-06'

# New, robust data retrieval function
def fetch_data_robustly(tickers, start, end):
    """Fetches adjusted close price data for each ticker individually."""
    data_list = []
    print("\nStarting robust data download for each ticker...")
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}...")
            df = yf.download(ticker, start=start, end=end, auto_adjust=False)
            if not df.empty and 'Adj Close' in df.columns:
                 adj_close = df['Adj Close']
                 adj_close.name = ticker
                 data_list.append(adj_close)
            else:
                print(f"Warning: No or incomplete data found for {ticker}. Skipping.")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}. Skipping.")

    if not data_list:
        print("\nAll data downloads failed. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Concatenate all individual DataFrames
    data = pd.concat(data_list, axis=1)

    print("\nData retrieved successfully for all available tickers.")
    return data

data = fetch_data_robustly(all_tickers, start_date, end_date)

if data.empty:
    print("\nCould not retrieve data from any source. Please check your tickers and internet connection.")
    exit()

# -------------------------------------------------------------------------------------
# Phase 2: Data Preprocessing
# -------------------------------------------------------------------------------------

print("\nPhase 2: Preprocessing data...")

# Handle missing values by forward-filling and dropping columns with no data
data = data.ffill().dropna(axis=1)
tickers = [t for t in tickers if t in data.columns]
all_tickers = data.columns.tolist()

# Calculate the daily logarithmic yields
log_returns = np.log(data / data.shift(1)).dropna()

# -------------------------------------------------------------------------------------
# Phase 3: Portfolio Optimization (VaR Minimization)
# -------------------------------------------------------------------------------------

print("\nPhase 3: Performing VaR-based portfolio optimization...")

# Define the VaR calculation function (Historical Simulation)
def calculate_portfolio_var(weights, returns, alpha=0.05):
    """Calculates the portfolio's Value at Risk at a given confidence level."""
    portfolio_returns = returns.dot(weights)
    return np.percentile(portfolio_returns, alpha * 100)

# Define the objective function for minimization
def objective_function(weights, returns, alpha=0.05):
    """Returns the positive VaR for minimization."""
    return calculate_portfolio_var(weights, returns, alpha)

# Set the constraints for the optimization
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights equals 1
    {'type': 'ineq', 'fun': lambda x: x}  # No short selling (all weights >= 0)
]

# Loop to generate at least three possible portfolios
optimized_portfolios = []
num_portfolios = 3

print(f"Generating {num_portfolios} optimized portfolios...")

for i in range(num_portfolios):
    # Get initial weights by generating a random set of weights that sums to 1
    initial_weights = np.random.uniform(0, 1, len(tickers))
    initial_weights /= np.sum(initial_weights)

    # Run the optimization process with the COBYLA method
    result = sco.minimize(
        fun=objective_function,
        x0=initial_weights,
        args=(log_returns[tickers],),
        method='COBYLA',
        constraints=constraints
    )

    if result.success:
        portfolio_returns = log_returns[tickers].dot(result.x)
        final_return = portfolio_returns.mean() * 252
        optimized_portfolios.append({
            'weights': result.x,
            'var': result.fun,
            'return': final_return,
            'status': 'Success'
        })
        print(f"\nOptimization for Portfolio {i+1} completed successfully.")
    else:
        optimized_portfolios.append({'status': 'Failed'})
        print(f"\nOptimization for Portfolio {i+1} failed: {result.message}")

# -------------------------------------------------------------------------------------
# Phase 4: Output and Visualization
# -------------------------------------------------------------------------------------

print("\nPhase 4: Generating plots and Excel files...")

# Select the first optimized portfolio for detailed analysis
if not optimized_portfolios:
    print("No optimized portfolios were found. Exiting analysis.")
    exit()

best_portfolio = optimized_portfolios[0]
weights = best_portfolio['weights']
portfolio_returns = log_returns[tickers].dot(weights)
spy_returns = log_returns['SPY']

# Create the Excel file
with pd.ExcelWriter('High_Tech_ETF_Report.xlsx') as writer:
    # 1. Price Data
    data.to_excel(writer, sheet_name='Price Data')

    # 2. Asset Allocations
    alloc_df = pd.DataFrame(index=tickers)
    for i, p in enumerate(optimized_portfolios):
        if p['status'] == 'Success':
            alloc_df[f'Portfolio_{i+1}'] = p['weights']
    alloc_df.to_excel(writer, sheet_name='Asset Allocations')

    # 3. Prospectus Metrics Data
    # Calculate daily metrics for the full period
    rolling_var_95 = portfolio_returns.rolling(window=252).apply(lambda x: np.percentile(x, 5), raw=True).dropna()
    rolling_var_99 = portfolio_returns.rolling(window=252).apply(lambda x: np.percentile(x, 1), raw=True).dropna()

    def calculate_es(returns, alpha=0.05):
        var = np.percentile(returns, alpha * 100)
        expected_shortfall = returns[returns <= var].mean()
        return expected_shortfall

    rolling_es_95 = portfolio_returns.rolling(window=252).apply(lambda x: calculate_es(x, alpha=0.05), raw=True).dropna()
    rolling_es_99 = portfolio_returns.rolling(window=252).apply(lambda x: calculate_es(x, alpha=0.01), raw=True).dropna()

    metrics_df = pd.DataFrame({
        'ETF Returns': portfolio_returns,
        'SPY Returns': spy_returns,
        'Daily VaR (95%)': rolling_var_95,
        'Daily VaR (99%)': rolling_var_99,
        'Daily ES (95%)': rolling_es_95,
        'Daily ES (99%)': rolling_es_99
    })
    metrics_df.to_excel(writer, sheet_name='Daily Metrics')

    # Calculate monthly rolling metrics
    monthly_returns = portfolio_returns.resample('M').sum()
    monthly_spy_returns = spy_returns.resample('M').sum()
    risk_free_rate = 0.02 / 12  # Assuming 2% annual risk-free rate
    rolling_sharpe = (monthly_returns.rolling(window=12).mean() - risk_free_rate) / monthly_returns.rolling(window=12).std() * np.sqrt(12)
    rolling_beta = monthly_returns.rolling(window=12).cov(monthly_spy_returns) / monthly_spy_returns.rolling(window=12).var()

    monthly_metrics_df = pd.DataFrame({
        'ETF Monthly Returns': monthly_returns,
        'SPY Monthly Returns': monthly_spy_returns,
        'Rolling Sharpe Ratio': rolling_sharpe,
        'Rolling Beta': rolling_beta
    })
    monthly_metrics_df.to_excel(writer, sheet_name='Monthly Metrics')

    # 4. Descriptive Statistics & Summary Table
    # Filter for the last 5 years
    five_years_ago = pd.to_datetime(end_date) - pd.DateOffset(years=5)
    recent_returns = portfolio_returns[portfolio_returns.index >= five_years_ago]

    last_var_95 = rolling_var_95.iloc[-1] if not rolling_var_95.empty else np.nan
    min_var_95 = rolling_var_95.min() if not rolling_var_95.empty else np.nan
    max_var_95 = rolling_var_95.max() if not rolling_var_95.empty else np.nan

    mean_ret = recent_returns.mean() * 252
    std_dev = recent_returns.std() * np.sqrt(252)
    portfolio_skew = skew(recent_returns)
    portfolio_kurt = kurtosis(recent_returns)

    summary_df = pd.DataFrame({
        'Metric': ['Last VaR (95%)', 'Min VaR (95%)', 'Max VaR (95%)', 'Annualized Return (last 5 yrs)', 'Annualized Std. Dev. (last 5 yrs)', 'Skewness (last 5 yrs)', 'Kurtosis (last 5 yrs)'],
        'Value': [last_var_95, min_var_95, max_var_95, mean_ret, std_dev, portfolio_skew, portfolio_kurt]
    })
    summary_df.to_excel(writer, sheet_name='Summary Metrics', index=False)

    print("\nExcel file 'High_Tech_ETF_Report.xlsx' has been created successfully.")

# -------------------------------------------------------------------------------------
# Phase 5: Plotting
# -------------------------------------------------------------------------------------

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Historical VaR and Expected Shortfall Plot
fig, ax1 = plt.subplots()
ax1.plot(rolling_var_95, label='VaR (95%)', color='red', linestyle='--')
ax1.plot(rolling_var_99, label='VaR (99%)', color='darkred', linestyle='--')
ax1.plot(rolling_es_95, label='ES (95%)', color='blue', linestyle='-')
ax1.plot(rolling_es_99, label='ES (99%)', color='darkblue', linestyle='-')
ax1.set_title('Historical VaR and Expected Shortfall (10 Years)', fontsize=16)
ax1.set_xlabel('Date')
ax1.set_ylabel('Loss (%)')
ax1.legend()
plt.tight_layout()
plt.savefig('VaR_ES_Graph.png')
plt.show()

# Rolling Sharpe Ratio Plot
plt.figure()
rolling_sharpe.plot(title='Rolling 12-Month Sharpe Ratio (10 Years)')
plt.ylabel('Sharpe Ratio')
plt.xlabel('Date')
plt.axhline(0, color='gray', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('Sharpe_Ratio_Graph.png')
plt.show()

# Rolling Beta Plot
plt.figure()
rolling_beta.plot(title='Rolling 12-Month Beta vs. S&P 500 (10 Years)')
plt.ylabel('Beta')
plt.xlabel('Date')
plt.axhline(1, color='gray', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.savefig('Rolling_Beta_Graph.png')
plt.show()

# Cumulative Returns Comparison Plot
cumulative_returns_etf = np.exp(portfolio_returns.cumsum())
cumulative_returns_spy = np.exp(spy_returns.cumsum())
plt.figure()
cumulative_returns_etf.plot(label='High Tech ETF')
cumulative_returns_spy.plot(label='S&P 500 Buy-and-Hold')
plt.title('Cumulative Returns: ETF vs. S&P 500 (10 Years)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Buy_and_Hold_Comparison.png')
plt.show()

print("\nScript execution complete. Check your directory for the Excel file and saved plots. ✅")
