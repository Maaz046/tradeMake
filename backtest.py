# backtest.py
import vectorbt as vbt
import pandas as pd
from utils import display_trade_log
from visualization import plot_portfolio, plot_trade_return_vs_duration, plot_price_with_volatility


def run_backtest(df: pd.DataFrame, df_trades: pd.DataFrame, title="ML Strategy Backtest"):
    pf = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=df['entry_signal'],
        exits=df['exit_signal'],
        freq='1D',
        slippage=0.001,     # 0.1% slippage 
        fees=0.001          # 0.1% trading fees
    )
    
    # Display results
    print(pf.stats())
    plot_price_with_volatility(df,pf)
    plot_portfolio(pf)
    # plot_trade_return_vs_duration(df_trades)
    display_trade_log(pf)


    return pf

if __name__ == "__main__":
    from data import fetch_okx_data
    from features import add_technical_indicators
    from signals import generate_signals

    df = fetch_okx_data(symbol='TON/USDT', timeframe='1d')
    df = add_technical_indicators(df)
    df = generate_signals(df)

    from analysis import compute_trade_durations_returns
    df_trades = compute_trade_durations_returns(df)
    run_backtest(df, df_trades)