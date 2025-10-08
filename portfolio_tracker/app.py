import os
import warnings
import gradio as gr

from dotenv import load_dotenv
from utils.data_fetcher import fetch_stock_data, fetch_benchmark_data, validate_tickers
from utils.input_validator import validate_inputs, InputValidationError
from utils.portfolio import Portfolio
from utils.visualizations import (
    format_cumulative_returns_data,
    format_drawdowns_data,
    format_returns_distribution_data,
    format_weight_allocation_data,
    format_performance_metrics,
    format_benchmark_comparison,
    prepare_cumulative_returns_plot,
    prepare_drawdowns_plot,
    prepare_weight_allocation_plot,
    prepare_contribution_allocation_plot,
)

warnings.simplefilter(action='ignore', category=FutureWarning)

load_dotenv()

BENCHMARK_OPTIONS = {
    "標普 500": "^GSPC",
    "道瓊斯": "^DJI",
    "納斯達克": "^IXIC",
    "羅素 2000": "^RUT",
    "歐洲 Stoxx 600": "^STOXX",
    "恆生指數": "^HSI",
    "台灣加權指數": "^TWII",
    "日經 225": "^N225",
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
    # Step 1: Validate inputs and tickers
    try:
        tickers, weights = validate_inputs(tickers_input, weights_input)
    except InputValidationError as e:
        return (
            None, None, None, None, None, None, None,
            None, None, None, None, None, None,
            f"❌ 輸入錯誤：{e.message}"
        )
    
    weights = dict(zip(tickers, weights))
    
    valid_tickers, invalid_tickers = validate_tickers(tickers)
    if invalid_tickers:
        return (
            None, None, None, None, None, None, None,
            None, None, None, None, None, None,
            f"❌ 輸入錯誤：無效的股票代碼 - {', '.join(invalid_tickers)}"
        )
    
    # Step 2: Fetch stock data
    try:
        period = DATE_RANGE_OPTIONS[date_range]
        stock_data = fetch_stock_data(valid_tickers, period=period)
        benchmark_data = fetch_benchmark_data(BENCHMARK_OPTIONS[benchmark], period=period)
    except Exception as e:
        return (
            None, None, None, None, None, None, None,
            None, None, None, None, None, None,
            f"❌ 獲取數據時出錯：{str(e)}"
        )
    
    # Step 3: Generate formatted data for visualizations and metrics
    portfolio = Portfolio(stock_data, weights)
    
    benchmark_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
    
    cumulative_returns_table = format_cumulative_returns_data(
        portfolio.cumulative_returns, 
        benchmark_cumulative_returns,
        period=period
    )
    
    drawdowns_table = format_drawdowns_data(
        portfolio.portfolio_returns,
        period=period
    )
    
    returns_distribution_data = format_returns_distribution_data(
        portfolio.portfolio_returns, 
        benchmark_returns,
        period=period
    )
    
    weight_allocation_table = format_weight_allocation_data(weights, stock_data)
    
    portfolio_metrics = portfolio.get_performance_summary()
    metrics_table = format_performance_metrics(portfolio_metrics)
    
    benchmark_comparison = portfolio.compare_to_benchmark(benchmark_returns)
    comparison_tables = format_benchmark_comparison(benchmark_comparison)

    # Step 4: Prepare plot data from formatted tables
    cumulative_returns_plot_df = prepare_cumulative_returns_plot(cumulative_returns_table)
    drawdowns_plot_df = prepare_drawdowns_plot(drawdowns_table)
    weight_allocation_plot_df = prepare_weight_allocation_plot(weight_allocation_table)
    contribution_allocation_plot_df = prepare_contribution_allocation_plot(weight_allocation_table)
    
    return (
        cumulative_returns_plot_df,
        cumulative_returns_table,
        drawdowns_plot_df,
        drawdowns_table,
        returns_distribution_data['histogram'],
        returns_distribution_data['summary'],
        returns_distribution_data['periodic_returns'],
        weight_allocation_plot_df,
        weight_allocation_table, 
        contribution_allocation_plot_df,
        metrics_table,
        comparison_tables['comparison'],
        comparison_tables['relative'],
        f"✅ 載入成功：分析涵蓋 ({len(valid_tickers)}) 支股票"
    )


def create_ui():
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

        with gr.Tabs():
            with gr.TabItem("累積回報"):
                cumulative_returns_plot = gr.LinePlot(
                    label="累積回報圖",
                    x="日期",
                    y="回報 (%)",
                    color="類別",
                    overlay=True,
                    title="累積回報 (%)",
                    height=300,
                    x_label_angle=90,
                    sort='x',
                    color_map={"投資組合 (%)": "#f97316", "基準 (%)": "#28170b"},
                    show_fullscreen_button=True,
                    show_export_button=True,
                )
                cumulative_returns_table = gr.DataFrame(label="累積回報（每月）")
                
            with gr.TabItem("最大回撤"):
                drawdowns_plot = gr.BarPlot(
                    label="最大回撤圖",
                    x="日期",
                    y="最大回撤 (%)",
                    title="最大回撤 (%)",
                    height=300,
                    x_label_angle=90,
                    sort='x',
                    show_fullscreen_button=True,
                    show_export_button=True,
                )
                drawdowns_table = gr.DataFrame(label="最大回撤（每月最小值）")
                
            with gr.TabItem("報酬分佈"):
                returns_histogram_plot = gr.BarPlot(
                    label="日報酬分佈圖",
                    x="回報區間 (%)",
                    y="頻率",
                    height=300,
                    show_fullscreen_button=True,
                    show_export_button=True,
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 報酬分佈摘要")
                        returns_stats_table = gr.DataFrame()
                    with gr.Column():
                        gr.Markdown("### 每月報酬")
                        monthly_returns_table = gr.DataFrame()
                
            with gr.TabItem("投資組合配置"):
                with gr.Row():
                    with gr.Column():
                        weight_allocation_plot = gr.BarPlot(
                            label="投資組合配置圖",
                            x="資產",
                            y="權重 (%)",
                            y_lim=(0, 100),
                            color_map={"權重 (%)": "#f97316"},
                            height=300,
                        )
                    with gr.Column():
                        contribution_allocation_plot = gr.BarPlot(
                            label="投資組合貢獻圖",
                            x="資產",
                            y="貢獻 (%)",
                            y_lim=(0, 100),
                            color_map={"貢獻 (%)": "#3b82f6"},
                            height=300,
                        )
                weight_allocation_table = gr.DataFrame(label="投資組合配置（表格）")
                
            with gr.TabItem("績效指標"):
                metrics_table = gr.DataFrame()
                gr.Markdown("> **投資組合：** 投資組合在所選期間的總回報率。")

            with gr.TabItem("基準比較"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 投資組合 vs 基準")
                        comparison_table = gr.DataFrame()
                    with gr.Column():
                        gr.Markdown("### 相對績效")
                        relative_metrics_table = gr.DataFrame()
                gr.Markdown(
                """
                > - **追蹤誤差**：投資組合與基準回報的標準差。
                > - **資訊比率**：超額報酬與追蹤誤差的比率。
                """
                )
        
        analyze_btn.click(
            fn=process_portfolio,
            inputs=[tickers_input, weights_input, date_range, benchmark],
            outputs=[
                cumulative_returns_plot,
                cumulative_returns_table,
                drawdowns_plot,
                drawdowns_table,
                returns_histogram_plot,
                returns_stats_table,
                monthly_returns_table, 
                weight_allocation_plot,
                weight_allocation_table, 
                contribution_allocation_plot,
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
                ["SPY, QQQ, IWM", "50, 30, 20", "五年", "標普 500"],
                ["AAPL, BRK-B, KO, JNJ, PG", "20, 30, 20, 15, 15", "五年", "標普 500"],
            ],
            inputs=[tickers_input, weights_input, date_range, benchmark]
        )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    if os.environ.get("HF_SPACE", "false") == "true":
        app.launch()
    else:
        server_ip = os.environ.get("SERVER_IP", "127.0.0.1")
        server_port = int(os.environ.get("SERVER_PORT", "7860"))
        app.launch(server_name=server_ip, server_port=server_port, share=True)