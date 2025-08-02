# visualization.py
import vectorbt as vbt
import plotly.express as px

def plot_portfolio(pf: vbt.Portfolio, title="ML Strategy Backtest", show=True):
    fig = pf.plot(title=title)
    fig.update_layout(
        title=dict(y=0.95),
        legend=dict(yanchor="top", y=0.99),
        margin=dict(t=80)
    )
    if show:
        fig.show()
    return fig


def plot_trade_return_vs_duration(df_trades, show=True):
    print(df_trades['duration'].describe())
    print(df_trades['duration'].value_counts())
    
    if df_trades.empty or 'duration' not in df_trades.columns or 'return' not in df_trades.columns:
        print("Trade data must include 'duration' and 'return' columns.")
        return

    fig = px.scatter(
        df_trades,
        x="duration",
        y="return",
        hover_data=df_trades.columns,
        title="Trade Return vs Holding Duration"
    )
    fig.update_layout(
        xaxis_title="Holding Duration (days)",
        yaxis_title="Trade Return",
        title=dict(y=0.95),
        margin=dict(t=80)
    )
    if show:
        fig.show()
    return fig