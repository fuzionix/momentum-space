import pandas as pd
from typing import Dict, Optional

def format_cumulative_returns_data(
    portfolio_cumulative_returns: pd.Series,
    benchmark_cumulative_returns: Optional[pd.Series] = None,
    period: str = "5y"
) -> pd.DataFrame:
    """
    Format cumulative returns data for tabular display with period-appropriate date sampling.
    
    Args:
        portfolio_cumulative_returns (pd.Series): Cumulative returns of the portfolio.
        benchmark_cumulative_returns (Optional[pd.Series]): Cumulative returns of the benchmark.
        period (str): Time period of the data ('1mo', '3mo', '6mo', '1y', '3y', '5y', '10y', 'max').
    
    Returns:
        output (pd.DataFrame): DataFrame with formatted cumulative returns data.
    """
    # Convert to percentage and round
    portfolio_pct = (portfolio_cumulative_returns * 100).round(2)
    
    # Create dataframe with portfolio data
    data = pd.DataFrame({
        '日期': portfolio_pct.index,
        '投資組合 (%)': portfolio_pct.values
    })
    
    # Add benchmark data if provided
    if benchmark_cumulative_returns is not None:
        # Align benchmark returns with portfolio dates
        aligned_benchmark = benchmark_cumulative_returns.reindex(
            portfolio_cumulative_returns.index, method='ffill'
        )
        benchmark_pct = (aligned_benchmark * 100).round(2)
        data['基準 (%)'] = benchmark_pct.values
    
    # Determine appropriate resampling frequency based on period
    if period in ['1mo', '3mo']:
        resample_rule = 'W'
        date_format = '%m-%d'
    elif period in ['6mo', '1y']:
        resample_rule = '2W'
        date_format = '%Y-%m-%d'
    elif period in ['3y', '5y']:
        resample_rule = 'ME'
        date_format = '%Y-%m'
    else:
        resample_rule = 'QE'
        date_format = '%Y-%m'
    
    # Resample to reduce data points for better display
    resampled_data = data.set_index('日期').resample(resample_rule).last().reset_index()
    resampled_data['日期'] = resampled_data['日期'].dt.strftime(date_format)

    return resampled_data


def format_drawdowns_data(
    portfolio_returns: pd.Series,
    period: str = "5y"
) -> pd.DataFrame:
    """
    Format drawdowns data for tabular display with period-appropriate date sampling.
    
    Args:
        portfolio_returns (pd.Series): Returns of the portfolio.
        period (str): Time period of the data.
    
    Returns:
        output (pd.DataFrame): DataFrame with drawdown information.
    """
    # Calculate wealth index
    wealth_index = (1 + portfolio_returns).cumprod()
    
    # Calculate previous peaks
    previous_peaks = wealth_index.cummax()
    
    # Calculate drawdowns
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # Convert to percentage and round
    drawdowns_pct = (drawdowns * 100).round(2)
    
    # Create dataframe
    data = pd.DataFrame({
        '日期': drawdowns_pct.index,
        '最大回撤 (%)': drawdowns_pct.values
    })
    
    # Determine appropriate resampling frequency based on period
    if period in ['1mo', '3mo']:
        resample_rule = 'W'
        date_format = '%m-%d'
    elif period in ['6mo', '1y']:
        resample_rule = '2W'
        date_format = '%Y-%m-%d'
    elif period in ['3y', '5y']:
        resample_rule = 'ME'
        date_format = '%Y-%m'
    else:
        resample_rule = 'QE'
        date_format = '%Y-%m'
    
    # Resample to reduce data points
    resampled_data = data.set_index('日期').resample(resample_rule).min().reset_index()
    resampled_data['日期'] = resampled_data['日期'].dt.strftime(date_format)

    return resampled_data


def format_returns_distribution_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    period: str = "5y",
) -> Dict[str, pd.DataFrame]:
    """
    Format returns distribution data for tabular display.
    
    Args:
        portfolio_returns (pd.Series): Returns of the portfolio.
        benchmark_returns (Optional[pd.Series]): Returns of the benchmark.
        period (str): Time period of the data ('1mo', '3mo', '6mo', '1y', '3y', '5y', '10y', 'max').
    
    Returns:
        output (Dict[str, pd.DataFrame]): Dictionary with statistical summary DataFrames.
    """
    # Calculate portfolio return statistics
    portfolio_stats = {
        '平均 (%)': portfolio_returns.mean() * 100,
        '中位數 (%)': portfolio_returns.median() * 100,
        '最小 (%)': portfolio_returns.min() * 100,
        '最大 (%)': portfolio_returns.max() * 100,
        '標準差 (%)': portfolio_returns.std() * 100,
        '偏度 (Skew)': portfolio_returns.skew(),
        '峰度 (Kurtosis)': portfolio_returns.kurtosis()
    }
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        '統計量': list(portfolio_stats.keys()),
        '投資組合 (%)': [round(val, 2) for val in portfolio_stats.values()]
    })
    
    # Add benchmark data if provided
    if benchmark_returns is not None:
        benchmark_stats = {
            '平均 (%)': benchmark_returns.mean() * 100,
            '中位數 (%)': benchmark_returns.median() * 100,
            '最小 (%)': benchmark_returns.min() * 100,
            '最大 (%)': benchmark_returns.max() * 100,
            '標準差 (%)': benchmark_returns.std() * 100,
            '偏度 (Skew)': benchmark_returns.skew(),
            '峰度 (Kurtosis)': benchmark_returns.kurtosis()
        }
        summary_df['基準 (%)'] = [round(val, 2) for val in benchmark_stats.values()]
    
    # Determine appropriate resampling for monthly returns based on period
    if period in ['1mo', '3mo']:
        resample_rule = 'W'
        label = '週'
    elif period in ['6mo', '1y', '3y']:
        resample_rule = 'ME'
        label = '月份'
    else:
        resample_rule = 'QE'
        label = '季度'
    
    # Create periodic returns dataframe
    periodic_returns = portfolio_returns.resample(resample_rule).apply(
        lambda x: (1 + x).prod() - 1
    ).to_frame('投資組合 (%)') * 100
    
    if resample_rule == 'W':
        periodic_returns.index = periodic_returns.index.strftime('%Y-%m-%d')
    elif resample_rule == 'ME':
        periodic_returns.index = periodic_returns.index.strftime('%Y-%m')
    else:
        periodic_returns.index = periodic_returns.index.strftime('%Y-%m')
    
    periodic_returns = periodic_returns.round(2)
    
    # Add benchmark periodic returns if provided
    if benchmark_returns is not None:
        benchmark_periodic = benchmark_returns.resample(resample_rule).apply(
            lambda x: (1 + x).prod() - 1
        ).to_frame('基準 (%)') * 100
        
        if resample_rule == 'W':
            benchmark_periodic.index = benchmark_periodic.index.strftime('%Y-%m-%d')
        elif resample_rule == 'ME':
            benchmark_periodic.index = benchmark_periodic.index.strftime('%Y-%m')
        else:
            benchmark_periodic.index = benchmark_periodic.index.strftime('%Y-%m')
        
        benchmark_periodic = benchmark_periodic.round(2)
        
        # Join portfolio and benchmark periodic returns
        periodic_returns = periodic_returns.join(benchmark_periodic)
    
    # Reset index to make the period a column
    periodic_returns = periodic_returns.reset_index()
    periodic_returns.columns = [label] + list(periodic_returns.columns[1:])
    
    return {
        'summary': summary_df,
        'periodic_returns': periodic_returns,
        'period_type': label
    }


def format_weight_allocation_data(
    weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Format portfolio weight allocation data for tabular display.
    
    Args:
        weights (Dict[str, float]): Dictionary mapping tickers to weights.
    
    Returns:
        output (pd.DataFrame): DataFrame with weight allocation information.
    """
    # Convert weights to percentages
    weights_pct = {ticker: weight * 100 for ticker, weight in weights.items()}
    
    # Sort weights from highest to lowest
    sorted_weights = sorted(weights_pct.items(), key=lambda x: x[1], reverse=True)
    
    # Create dataframe
    data = pd.DataFrame(sorted_weights, columns=['資產', '權重 (%)'])
    data['權重 (%)'] = data['權重 (%)'].round(2)

    return data


def format_performance_metrics(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Format performance metrics for tabular display.
    
    Args:
        metrics (Dict[str, float]): Dictionary of performance metrics.
    
    Returns:
        output (pd.DataFrame): DataFrame with formatted performance metrics.
    """
    # Create dataframe
    data = pd.DataFrame({
        '指標': list(metrics.keys()),
        '數值': list(metrics.values())
    })
    
    # For metrics that are percentages or numeric values, round them
    data['數值'] = data['數值'].apply(
        lambda x: round(x, 2) if isinstance(x, (int, float)) else x
    )
    
    return data


def format_benchmark_comparison(comparison: Dict[str, float]) -> pd.DataFrame:
    """
    Format benchmark comparison data for tabular display.
    
    Args:
        comparison (Dict[str, float]): Dictionary of benchmark comparison metrics.
    
    Returns:
        output (pd.DataFrame): DataFrame with formatted benchmark comparison data.
    """
    # Organize data into portfolio vs benchmark structure
    portfolio_metrics = {
        '總回報 (%)': comparison['投資組合總回報 (%)'],
        '年化回報 (%)': comparison['投資組合年化報酬 (%)'],
        '夏普比率': comparison['投資組合夏普比率'],
        '最大回撤 (%)': comparison['投資組合最大回撤 (%)']
    }
    
    benchmark_metrics = {
        '總回報 (%)': comparison['基準總回報 (%)'],
        '年化回報 (%)': comparison['基準年化報酬 (%)'],
        '夏普比率': comparison['基準夏普比率'],
        '最大回撤 (%)': comparison['基準最大回撤 (%)']
    }
    
    # Create comparison dataframe
    data = pd.DataFrame({
        '指標': list(portfolio_metrics.keys()),
        '投資組合': [round(val, 2) if isinstance(val, (int, float)) else val for val in portfolio_metrics.values()],
        '基準': [round(val, 2) if isinstance(val, (int, float)) else val for val in benchmark_metrics.values()]
    })
    
    # Add relative metrics
    relative_metrics = {
        '超額報酬 (%)': comparison['Alpha (%)'],
        'Beta': comparison['Beta'],
        '追蹤誤差 (%)': comparison['追蹤誤差 (%)'],
        '資訊比率': comparison['資訊比率']
    }
    
    relative_df = pd.DataFrame({
        '指標': list(relative_metrics.keys()),
        '數值': [round(val, 2) if isinstance(val, (int, float)) else val for val in relative_metrics.values()]
    })
    
    return {'comparison': data, 'relative': relative_df}

def prepare_cumulative_returns_plot(cum_table: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare long-form DataFrame for LinePlot of cumulative returns.
    Expected table columns: ['日期', '投資組合 (%)', optional '基準 (%)']
    """
    value_cols = [col for col in cum_table.columns if col.endswith("(%)")]
    # Melt to long form with 類別 and 回報 (%)
    plot_df = cum_table.melt(id_vars=["日期"], value_vars=value_cols, var_name="類別", value_name="回報 (%)")
    return plot_df


def prepare_drawdowns_plot(drawdowns_table: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for LinePlot of drawdowns.
    Expected columns: ['日期', '最大回撤 (%)']
    """
    return drawdowns_table.copy()

def prepare_weight_allocation_plot(weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for BarPlot of weight allocation.
    Expected columns: ['資產','權重 (%)']
    """
    return weights_df.copy()