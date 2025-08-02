# backtest.py
import vectorbt as vbt
import pandas as pd
from utils import plot_portfolio, display_trade_log


def run_backtest(df: pd.DataFrame, title="ML Strategy Backtest"):
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
    plot_portfolio(pf)
    display_trade_log(pf)


    return pf

if __name__ == "__main__":
    from data import fetch_okx_data
    from features import add_technical_indicators
    from signals import generate_signals

    df = fetch_okx_data(symbol='TON/USDT', timeframe='1d')
    df = add_technical_indicators(df)
    df = generate_signals(df)

    run_backtest(df)