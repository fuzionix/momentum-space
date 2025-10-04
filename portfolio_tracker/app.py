"""
Portfolio Performance Tracker - Main Application

An interactive web application to analyze and visualize portfolio performance metrics.
"""

import gradio as gr

from utils.data_fetcher import fetch_stock_data, fetch_benchmark_data, validate_tickers
from utils.portfolio import Portfolio
from utils.visualizations import (
    format_cumulative_returns_data,
    format_drawdowns_data,
    format_returns_distribution_data,
    format_weight_allocation_data,
    format_performance_metrics,
    format_benchmark_comparison
)

BENCHMARK_OPTIONS = {
    "標普 500": "^GSPC",
    "道瓊斯": "^DJI",
    "納斯達克": "^IXIC",
    "羅素 2000": "^RUT"
}

DATE_RANGE_OPTIONS = {
    "一個月": "1mo",
    "三個月": "3mo",
    "六個月": "6mo",
    "一年": "1y",
    "三年": "3y",
    "五年": "5y",
    "十年": "10y",
    "最大": "max"
}

def process_portfolio(
    tickers_input: str,
    weights_input: str,
    date_range: str,
    benchmark: str
):
    """
    Process portfolio data and generate analysis.
    
    Args:
        tickers_input (str): Comma-separated string of ticker symbols.
        weights_input (str): Comma-separated string of weights (should sum to 100).
        date_range (str): Date range to analyze.
        benchmark (str): Benchmark to compare against.
    
    Returns:
        output (tuple): Multiple return values including plots and performance metrics.
    """
    # Process tickers and weights inputs
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    weights_str = weights_input.split(',')
    
    # Validate inputs
    if len(tickers) != len(weights_str):
        return (None, None, None, None, None, 
                f"Error: Number of tickers ({len(tickers)}) does not match number of weights ({len(weights_str)})")
    
    try:
        weights_list = [float(w.strip()) for w in weights_str]
        # Convert percentage weights to decimals (e.g., 40 -> 0.4)
        weights_list = [w/100 for w in weights_list]
    except ValueError:
        return (None, None, None, None, None, 
                "Error: Weights must be numeric values")
    
    # Check if weights sum to approximately 1
    if not 0.99 <= sum(weights_list) <= 1.01:
        return (None, None, None, None, None, 
                f"Error: Weights must sum to approximately 100% (current sum: {sum(weights_list)*100:.1f}%)")
    
    # Create weights dictionary
    weights = dict(zip(tickers, weights_list))
    
    # Validate tickers
    valid_tickers, invalid_tickers = validate_tickers(tickers)
    if invalid_tickers:
        return (None, None, None, None, None, 
                f"Error: The following tickers are invalid: {', '.join(invalid_tickers)}")
    
    # Fetch stock data
    try:
        period = DATE_RANGE_OPTIONS[date_range]
        stock_data = fetch_stock_data(valid_tickers, period=period)
        benchmark_data = fetch_benchmark_data(BENCHMARK_OPTIONS[benchmark], period=period)
    except Exception as e:
        return (None, None, None, None, None, f"Error fetching data: {str(e)}")
    
    # Create portfolio
    portfolio = Portfolio(stock_data, weights)
    
    # Calculate benchmark returns
    benchmark_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
    
    # Generate tables
    cumulative_returns_table = format_cumulative_returns_data(
        portfolio.cumulative_returns, 
        benchmark_cumulative_returns
    )
    
    drawdowns_table = format_drawdowns_data(portfolio.portfolio_returns)
    
    returns_distribution_data = format_returns_distribution_data(
        portfolio.portfolio_returns, 
        benchmark_returns
    )
    
    weight_allocation_table = format_weight_allocation_data(weights)
    
    # Get performance metrics
    portfolio_metrics = portfolio.get_performance_summary()
    metrics_table = format_performance_metrics(portfolio_metrics)
    
    benchmark_comparison = portfolio.compare_to_benchmark(benchmark_returns)
    comparison_tables = format_benchmark_comparison(benchmark_comparison)
    
    return (
        cumulative_returns_table,
        drawdowns_table,
        returns_distribution_data['summary'],
        returns_distribution_data['monthly_returns'],
        weight_allocation_table,
        metrics_table,
        comparison_tables['comparison'],
        comparison_tables['relative'],
        f"Successfully analyzed portfolio of {len(valid_tickers)} assets"
    )


def create_ui():
    """
    Create the Gradio interface.
    
    Returns:
        output (gr.Interface): Gradio interface object.
    """
    with gr.Blocks(title="投資組合分析") as app:
        gr.Markdown("# 投資組合分析")
        gr.Markdown(
            """
            輸入股票代碼及其權重，選擇分析的時間範圍和基準指數以生成投資組合的績效分析報告。
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                tickers_input = gr.Textbox(
                    label="股票代碼",
                    placeholder="AAPL, MSFT, GOOGL, AMZN",
                    info="輸入以逗號分隔的股票代碼"
                )
                
                weights_input = gr.Textbox(
                    label="投資組合權重 (%)",
                    placeholder="25, 25, 25, 25",
                    info="輸入以百分比表示的權重（總和為 100）"
                )
                
            with gr.Column(scale=1):
                date_range = gr.Dropdown(
                    choices=list(DATE_RANGE_OPTIONS.keys()),
                    value="五年",
                    label="時間範圍"
                )
                
                benchmark = gr.Dropdown(
                    choices=list(BENCHMARK_OPTIONS.keys()),
                    value="標普 500",
                    label="基準"
                )

        analyze_btn = gr.Button("分析投資組合", variant="primary")

        output_message = gr.Textbox(label="狀態")
        summary_output = gr.Markdown(label="分析摘要")

        with gr.Tabs():
            with gr.TabItem("累積回報"):
                cumulative_returns_table = gr.DataFrame()
                
            with gr.TabItem("最大回撤"):
                drawdowns_table = gr.DataFrame()
                
            with gr.TabItem("報酬分佈"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 報酬分佈摘要")
                        returns_stats_table = gr.DataFrame()
                    with gr.Column():
                        gr.Markdown("### 每月報酬")
                        monthly_returns_table = gr.DataFrame()
                
            with gr.TabItem("投資組合配置"):
                weight_allocation_table = gr.DataFrame()
                
            with gr.TabItem("績效指標"):
                metrics_table = gr.DataFrame()

            with gr.TabItem("基準比較"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 投資組合 vs 基準")
                        comparison_table = gr.DataFrame()
                    with gr.Column():
                        gr.Markdown("### 相對績效")
                        relative_metrics_table = gr.DataFrame()
        
        analyze_btn.click(
            fn=process_portfolio,
            inputs=[tickers_input, weights_input, date_range, benchmark],
            outputs=[
                cumulative_returns_table, 
                drawdowns_table, 
                returns_stats_table,
                monthly_returns_table, 
                weight_allocation_table, 
                metrics_table,
                comparison_table,
                relative_metrics_table,
                output_message
            ]
        )
        
        # Example inputs
        examples = gr.Examples(
            examples=[
                ["AAPL, MSFT, GOOGL, AMZN", "25, 25, 25, 25", "五年", "標普 500"],
                ["SPY, QQQ, IWM", "50, 30, 20", "三年", "道瓊斯"],
                ["AAPL, BRK-B, KO, JNJ, PG", "20, 30, 20, 15, 15", "十年", "納斯達克"],
            ],
            inputs=[tickers_input, weights_input, date_range, benchmark]
        )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()