import numpy as np
import pandas as pd
from typing import Dict, Tuple

class Portfolio:
    def __init__(
        self, 
        prices: pd.DataFrame, 
        weights: Dict[str, float]
    ):
        self.prices = prices.loc[:, list(weights.keys())]
        self.weights = weights
        self.returns = self.prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        # Convert weights to numpy array in the same order as price columns
        self.weights_arr = np.array([weights.get(ticker, 0) for ticker in prices.columns])
        self.weights_arr = self.weights_arr / np.sum(self.weights_arr)  # Normalize weights
        
        # Calculate portfolio returns
        self.portfolio_returns = self._calculate_portfolio_returns()
        
        # Calculate cumulative returns
        self.cumulative_returns = self._calculate_cumulative_returns()
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        return self.returns.dot(self.weights_arr)
    
    def _calculate_cumulative_returns(self) -> pd.Series:
        return (1 + self.portfolio_returns).cumprod() - 1
    
    def get_annualized_return(self) -> float:
        # Assume 252 trading days in a year
        return self.portfolio_returns.mean() * 252 * 100
    
    def get_annualized_volatility(self) -> float:
        # Assume 252 trading days in a year
        return self.portfolio_returns.std() * np.sqrt(252) * 100
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        # Convert annual risk-free rate to daily
        daily_rf = risk_free_rate / 252
        excess = self.portfolio_returns - daily_rf
        std = excess.std(ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0
        return np.sqrt(252) * excess.mean() / std
    
    def get_max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        # Calculate the cumulative wealth index
        wealth_index = (1 + self.portfolio_returns).cumprod()
        
        # Calculate previous peaks
        previous_peaks = wealth_index.cummax()
        
        # Calculate drawdowns
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        # Find the maximum drawdown
        max_dd = drawdowns.min()
        
        # Find the peak and trough dates
        trough_date = drawdowns.idxmin()
        peak_date = previous_peaks.loc[:trough_date].idxmax()
        
        return max_dd * 100, peak_date, trough_date
    
    def get_performance_summary(self) -> Dict[str, float]:
        annualized_return = self.get_annualized_return()
        annualized_volatility = self.get_annualized_volatility()
        sharpe_ratio = self.get_sharpe_ratio()
        max_drawdown, peak_date, trough_date = self.get_max_drawdown()
        
        total_return = self.cumulative_returns.iloc[-1] * 100
        
        return {
            "總回報 (%)": total_return,
            "年化報酬 (%)": annualized_return,
            "年化波動率 (%)": annualized_volatility,
            "夏普比率": sharpe_ratio,
            "最大回撤 (%)": max_drawdown,
            "最大回撤峰值日期": peak_date.strftime('%Y-%m-%d'),
            "最大回撤谷值日期": trough_date.strftime('%Y-%m-%d')
        }

    def compare_to_benchmark(self, benchmark_returns: pd.Series) -> Dict[str, float]:
        # Calculate benchmark cumulative returns
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
        
        # Calculate benchmark annualized return and volatility
        bench_annual_return = benchmark_returns.mean() * 252 * 100
        bench_annual_vol = benchmark_returns.std() * np.sqrt(252) * 100
        
        # Calculate benchmark Sharpe ratio
        daily_rf = (1 + 0.02) ** (1/252) - 1  # Assuming 2% risk-free rate
        bench_excess_return = benchmark_returns - daily_rf
        bench_sharpe = (bench_excess_return.mean() / bench_excess_return.std()) * np.sqrt(252)
        
        # Calculate tracking error
        tracking_error = (self.portfolio_returns - benchmark_returns).std() * np.sqrt(252) * 100
        
        # Calculate information ratio
        information_ratio = (self.get_annualized_return() - bench_annual_return) / tracking_error
        
        # Calculate benchmark max drawdown
        bench_wealth_index = (1 + benchmark_returns).cumprod()
        bench_previous_peaks = bench_wealth_index.cummax()
        bench_drawdowns = (bench_wealth_index - bench_previous_peaks) / bench_previous_peaks
        bench_max_dd = bench_drawdowns.min() * 100
        
        # Calculate beta
        cov_matrix = pd.concat([self.portfolio_returns, benchmark_returns], axis=1).cov()
        beta = cov_matrix.iloc[0, 1] / benchmark_returns.var()
        
        # Calculate alpha (Jensen's Alpha)
        expected_return = daily_rf * 252 + beta * (bench_annual_return/100 - daily_rf * 252)
        alpha = self.get_annualized_return()/100 - expected_return
        
        return {
            "投資組合總回報 (%)": self.cumulative_returns.iloc[-1] * 100,
            "基準總回報 (%)": benchmark_cum_returns.iloc[-1] * 100,
            "投資組合年化報酬 (%)": self.get_annualized_return(),
            "基準年化報酬 (%)": bench_annual_return,
            "投資組合夏普比率": self.get_sharpe_ratio(),
            "基準夏普比率": bench_sharpe,
            "投資組合最大回撤 (%)": self.get_max_drawdown()[0],
            "基準最大回撤 (%)": bench_max_dd,
            "Alpha (%)": alpha * 100,
            "Beta": beta,
            "追蹤誤差 (%)": tracking_error,
            "資訊比率": information_ratio
        }