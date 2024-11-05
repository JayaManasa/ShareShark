import numpy as np
from typing import List, Dict


class TradingStrategy:
    def __init__(self, prices: List[float]):
        self.prices = np.array(prices)
        self.signals = []

    def calculate_sma(self, period: int) -> np.array:
        """Calculate Simple Moving Average"""
        return np.convolve(self.prices, np.ones(period) / period, mode='valid')

    def calculate_rsi(self, period: int = 14) -> np.array:
        """Calculate Relative Strength Index"""
        delta = np.diff(self.prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')

        rs = avg_gain / (avg_loss + 1e-10)  # Adding small number to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def find_trade_signals(self,
                           rsi_period: int = 14,
                           rsi_oversold: float = 30,
                           rsi_overbought: float = 70,
                           min_profit: float = 0.2,
                           stop_loss: float = 0.15) -> List[Dict]:
        """Find trading signals based on RSI and price action"""

        rsi = self.calculate_rsi(rsi_period)
        sma_5 = self.calculate_sma(5)
        sma_10 = self.calculate_sma(10)

        signals = []
        in_position = False
        entry_price = 0
        entry_index = 0

        # We need to offset our index due to the indicators calculation
        offset = max(rsi_period, 10)

        for i in range(offset, len(self.prices)):
            current_price = self.prices[i]
            current_rsi = rsi[i - offset]

            # Buy conditions
            if not in_position:
                # Check for oversold condition and price momentum
                if (current_rsi < rsi_oversold and
                        sma_5[i - offset] > sma_10[i - offset] and
                        self.prices[i] > self.prices[i - 1]):
                    entry_price = current_price
                    entry_index = i
                    in_position = True
                    signals.append({
                        'type': 'BUY',
                        'price': current_price,
                        'index': i,
                        'rsi': current_rsi
                    })

            # Sell conditions
            else:
                profit = current_price - entry_price
                loss = entry_price - current_price

                # Check for overbought condition or profit target or stop loss
                if (current_rsi > rsi_overbought or
                        profit >= min_profit or
                        loss >= stop_loss):
                    in_position = False
                    signals.append({
                        'type': 'SELL',
                        'price': current_price,
                        'index': i,
                        'rsi': current_rsi,
                        'profit': profit
                    })

        return signals

    def analyze_trades(self) -> None:
        """Analyze and print trade results"""
        signals = self.find_trade_signals()
        total_profit = 0
        trades = []

        for i in range(0, len(signals) - 1, 2):
            if i + 1 < len(signals):
                buy = signals[i]
                sell = signals[i + 1]
                profit = sell['price'] - buy['price']
                trades.append({
                    'buy_price': buy['price'],
                    'sell_price': sell['price'],
                    'profit': profit,
                    'buy_index': buy['index'],
                    'sell_index': sell['index']
                })
                total_profit += profit

        # Print results
        print(f"\nTotal number of trades: {len(trades)}")
        print(f"Total profit: {total_profit:.2f} Rs")
        print("\nDetailed Trade Analysis:")
        for i, trade in enumerate(trades, 1):
            print(f"\nTrade {i}:")
            print(f"Buy Price: {trade['buy_price']}")
            print(f"Sell Price: {trade['sell_price']}")
            print(f"Profit: {trade['profit']:.2f}")
            print(f"Duration: {trade['sell_index'] - trade['buy_index']} ticks")


test_data_4 = [936.1, 936.0, 936.1, 936.2, 936.3, 936.1, 936.3, 935.75, 935.85, 936.1, 936.0, 935.85, 936.2, 936.1, 936.2, 936.1, 936.25, 936.15, 936.1, 936.2, 936.3, 935.9, 936.3, 936.5, 936.4, 936.5, 936.7, 936.6, 936.85, 936.7, 936.8, 936.75, 936.6, 936.65, 936.4, 936.35, 936.65, 936.35, 936.4, 936.65, 936.6, 936.5, 936.65, 936.5, 936.65, 936.7, 936.8, 936.9, 936.95, 936.9, 937.1, 937.2, 937.15, 937.2, 937.15, 937.2, 936.9, 937.2, 937.1, 937.15, 937.2, 937.0, 937.2, 937.0, 937.2, 937.0, 937.2, 937.1, 937.0, 937.2, 937.15, 937.1, 937.15, 936.9, 937.2, 937.1, 937.2, 937.25, 937.2, 937.35, 937.4, 937.45, 937.4, 937.1, 937.25, 937.35, 937.4, 937.45, 937.4, 937.3, 937.4, 937.1, 937.2, 937.3, 937.1, 937.4, 937.45, 937.3, 937.4, 937.1, 937.3, 937.4, 937.1, 937.45, 937.3, 937.4, 937.3, 937.4, 937.3, 937.05, 937.3, 937.2, 937.5, 937.2, 937.1, 937.2, 936.85, 937.25, 937.0, 937.3, 937.25, 937.0, 937.3, 937.35, 937.25, 937.35, 937.1, 937.2, 937.25, 937.2, 937.0, 937.25, 937.3, 937.45, 937.1, 937.45, 937.1, 937.35, 937.4, 937.1, 937.35, 937.45, 937.5, 937.35, 937.6, 937.45, 937.5, 937.65, 937.35, 937.5, 937.65, 937.7, 937.65, 937.7, 937.65, 937.7, 937.15, 937.1, 937.45, 937.5, 937.3, 937.55, 937.5, 937.55, 937.5, 937.55, 937.6, 937.55, 937.65, 937.55, 937.5, 937.65, 937.4, 937.7, 937.6, 937.55, 937.45, 937.65, 937.4, 937.65, 937.7, 937.55, 937.65, 937.7, 937.65, 937.7, 937.75, 937.8, 937.75, 937.8, 937.75, 937.55, 937.8, 937.7, 936.9, 937.0, 937.1, 937.0, 937.25, 937.3, 937.25, 937.2, 937.0, 937.2, 937.1, 937.2, 937.1, 937.2, 937.25, 937.15, 937.0, 937.1, 937.0, 937.15, 937.2, 937.0, 937.15, 937.1, 937.15, 937.0, 937.15, 937.0, 937.15, 936.9, 937.1, 936.9, 936.95, 936.8, 936.95, 937.0, 936.6, 936.95, 936.6, 936.9, 936.55, 936.85, 936.55, 936.9, 936.6, 936.9, 936.6, 936.9, 936.5, 936.7, 936.65, 936.45, 936.3, 936.7, 936.65, 936.3, 936.6, 936.55, 936.3, 936.55, 936.6, 936.3, 936.6, 936.65, 936.3, 936.65, 936.3, 936.65, 936.3, 936.65, 936.6, 936.3, 936.55, 936.3, 936.55, 936.3, 936.2, 936.5, 936.2, 936.45, 936.15, 936.45, 936.3, 936.45, 936.3, 936.45, 936.2, 936.15, 936.25, 936.3, 936.2, 936.3, 936.4, 936.15, 936.3, 936.4, 936.3, 936.4, 936.15, 936.4, 936.3, 936.4, 936.5, 936.4, 936.3, 936.4, 936.3, 936.1, 936.3, 936.35, 936.3, 936.35, 936.3, 936.35, 936.4, 936.45, 936.35, 936.5, 936.2, 936.45, 936.5, 936.45, 936.4, 936.45, 936.5, 936.35, 936.5, 936.35, 936.5, 936.35, 936.5, 936.55, 936.6, 936.5, 936.4, 936.65, 936.55, 936.65, 936.55, 936.65, 936.6, 936.8, 936.55, 936.65, 936.7, 936.8, 936.65]

# Usage

strategy = TradingStrategy(test_data_4)
strategy.find_trade_signals(
    rsi_period=5,          # Shorter RSI for faster signals
    rsi_oversold=20,        # Less strict oversold condition
    rsi_overbought=65,      # Less strict overbought condition
    min_profit=0.15,        # Lower profit target for more trades
    stop_loss=0.1           # Tighter stop loss
)
strategy.analyze_trades()
