import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List, Tuple, Optional
import warnings

class TechnicalIndicators:
    def __init__(self, config: Dict):
        """
        Initialize TechnicalIndicators with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing indicator parameters
        """
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the indicators module"""
        self.logger = logging.getLogger(__name__)
        
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data (pd.DataFrame): Price data with 'close' column
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        try:
            return pd.Series(talib.RSI(data['close'].values, timeperiod=period), index=data.index)
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=data.index)
            
    def calculate_macd(self, data: pd.DataFrame, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data (pd.DataFrame): Price data with 'close' column
            fast_period (int): Fast period
            slow_period (int): Slow period
            signal_period (int): Signal period
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, Signal line, and Histogram
        """
        try:
            macd, signal, hist = talib.MACD(
                data['close'].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            return (
                pd.Series(macd, index=data.index),
                pd.Series(signal, index=data.index),
                pd.Series(hist, index=data.index)
            )
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(index=data.index), pd.Series(index=data.index), pd.Series(index=data.index)
            
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, 
                                num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data (pd.DataFrame): Price data with 'close' column
            period (int): Moving average period
            num_std (float): Number of standard deviations
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Upper band, Middle band, Lower band
        """
        try:
            upper, middle, lower = talib.BBANDS(
                data['close'].values,
                timeperiod=period,
                nbdevup=num_std,
                nbdevdn=num_std
            )
            return (
                pd.Series(upper, index=data.index),
                pd.Series(middle, index=data.index),
                pd.Series(lower, index=data.index)
            )
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.Series(index=data.index), pd.Series(index=data.index), pd.Series(index=data.index)
            
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            data (pd.DataFrame): Price data with 'high', 'low', 'close' columns
            period (int): ATR period
            
        Returns:
            pd.Series: ATR values
        """
        try:
            return pd.Series(
                talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period),
                index=data.index
            )
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(index=data.index)
            
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with all indicators added
        """
        try:
            # Get parameters from config
            rsi_period = self.config.get('rsi_period', 14)
            macd_fast = self.config.get('macd_fast', 12)
            macd_slow = self.config.get('macd_slow', 26)
            macd_signal = self.config.get('macd_signal', 9)
            bb_period = self.config.get('bb_period', 20)
            bb_std = self.config.get('bb_std', 2.0)
            atr_period = self.config.get('atr_period', 14)
            
            # Calculate indicators
            data['rsi'] = self.calculate_rsi(data, rsi_period)
            macd, signal, hist = self.calculate_macd(data, macd_fast, macd_slow, macd_signal)
            data['macd'] = macd
            data['macd_signal'] = signal
            data['macd_hist'] = hist
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data, bb_period, bb_std)
            data['bb_upper'] = bb_upper
            data['bb_middle'] = bb_middle
            data['bb_lower'] = bb_lower
            
            data['atr'] = self.calculate_atr(data, atr_period)
            
            # Add moving averages
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            data['sma_200'] = data['close'].rolling(window=200).mean()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return data
            
    def get_trading_signals(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Generate trading signals based on technical indicators
        
        Args:
            data (pd.DataFrame): Data with calculated indicators
            
        Returns:
            Tuple[pd.DataFrame, float]: DataFrame with signals and overall signal strength
        """
        try:
            signals = pd.DataFrame(index=data.index)
            
            # RSI signals
            signals['rsi_signal'] = 0
            signals.loc[data['rsi'] < 30, 'rsi_signal'] = 1  # Oversold
            signals.loc[data['rsi'] > 70, 'rsi_signal'] = -1  # Overbought
            
            # MACD signals
            signals['macd_signal'] = 0
            signals.loc[data['macd'] > data['macd_signal'], 'macd_signal'] = 1
            signals.loc[data['macd'] < data['macd_signal'], 'macd_signal'] = -1
            
            # Bollinger Bands signals
            signals['bb_signal'] = 0
            signals.loc[data['close'] < data['bb_lower'], 'bb_signal'] = 1  # Price below lower band
            signals.loc[data['close'] > data['bb_upper'], 'bb_signal'] = -1  # Price above upper band
            
            # Moving Average signals
            signals['ma_signal'] = 0
            signals.loc[data['sma_20'] > data['sma_50'], 'ma_signal'] = 1
            signals.loc[data['sma_20'] < data['sma_50'], 'ma_signal'] = -1
            
            # Calculate overall signal strength
            signal_strength = signals.mean(axis=1)
            
            # Add signal strength to signals DataFrame
            signals['signal_strength'] = signal_strength
            
            return signals, signal_strength.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            return pd.DataFrame(index=data.index), 0.0

    def calculate_stochastic(self, data, k_window=14, d_window=3):
        stoch = ta.momentum.StochasticOscillator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=k_window,
            smooth_window=d_window
        )
        return pd.DataFrame({
            'stoch_k': stoch.stoch(),
            'stoch_d': stoch.stoch_signal()
        })

    def calculate_ichimoku(self, data):
        ichimoku = ta.trend.IchimokuIndicator(
            high=data['high'],
            low=data['low']
        )
        return pd.DataFrame({
            'tenkan_sen': ichimoku.ichimoku_conversion_line(),
            'kijun_sen': ichimoku.ichimoku_base_line(),
            'senkou_span_a': ichimoku.ichimoku_a(),
            'senkou_span_b': ichimoku.ichimoku_b()
        })

    def calculate_volume_indicators(self, data):
        # Volume indicators
        obv = ta.volume.OnBalanceVolumeIndicator(close=data['close'], volume=data['volume'])
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume']
        )
        return pd.DataFrame({
            'obv': obv.on_balance_volume(),
            'cmf': cmf.chaikin_money_flow()
        })

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade múltiples medias móviles con diferentes períodos"""
        periods = [7, 14, 21, 50, 100, 200]
        for period in periods:
            df[f'sma_{period}'] = ta.trend.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = ta.trend.EMA(df['close'], timeperiod=period)
            df[f'wma_{period}'] = ta.trend.WMA(df['close'], timeperiod=period)
        return df

    def add_ichimoku_cloud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade el indicador Ichimoku Cloud"""
        high_prices = df['high']
        low_prices = df['low']
        
        # Períodos
        conversion_period = 9
        base_period = 26
        span_b_period = 52
        displacement = 26
        
        # Cálculos
        conversion_line = (high_prices.rolling(window=conversion_period).max() + 
                         low_prices.rolling(window=conversion_period).min()) / 2
        base_line = (high_prices.rolling(window=base_period).max() + 
                    low_prices.rolling(window=base_period).min()) / 2
        
        span_a = ((conversion_line + base_line) / 2).shift(displacement)
        span_b = ((high_prices.rolling(window=span_b_period).max() + 
                  low_prices.rolling(window=span_b_period).min()) / 2).shift(displacement)
        
        df['ichimoku_span_a'] = span_a
        df['ichimoku_span_b'] = span_b
        df['ichimoku_base'] = base_line
        df['ichimoku_conversion'] = conversion_line
        
        return df

    def add_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade el indicador Supertrend"""
        atr_period = 10
        atr_multiplier = 3.0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Cálculo ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        
        # Cálculo Supertrend
        hl2 = (high + low) / 2
        upperband = hl2 + (atr_multiplier * atr)
        lowerband = hl2 - (atr_multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(1, len(df)):
            if close[i] > upperband[i-1]:
                direction[i] = 1
            elif close[i] < lowerband[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            if direction[i] == 1:
                supertrend[i] = max(lowerband[i], supertrend[i-1])
            else:
                supertrend[i] = min(upperband[i], supertrend[i-1])
        
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        
        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade indicadores de volumen"""
        # Volume Weighted Average Price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # On Balance Volume (OBV)
        df['obv'] = ta.volume.OBV(df['close'], df['volume'])
        
        # Chaikin Money Flow (CMF)
        df['cmf'] = ta.volume.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume Rate of Change
        df['vroc'] = df['volume'].pct_change(periods=14) * 100
        
        return df

    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade patrones de velas japonesas"""
        patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
            'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
            'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
            'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
            'CDLXSIDEGAP3METHODS'
        ]
        
        for pattern in patterns:
            df[pattern.lower()] = getattr(ta.volume, pattern)(
                df['open'], df['high'], df['low'], df['close']
            )
        
        return df

    def get_support_resistance(self, data, window=20):
        """Calculate support and resistance levels using multiple methods"""
        # Pivot points
        high = data['high'].rolling(window=window, center=True).max()
        low = data['low'].rolling(window=window, center=True).min()
        close = data['close'].rolling(window=window, center=True).mean()
        
        # Fibonacci levels
        diff = high - low
        fib_0 = low
        fib_0_236 = low + 0.236 * diff
        fib_0_382 = low + 0.382 * diff
        fib_0_5 = low + 0.5 * diff
        fib_0_618 = low + 0.618 * diff
        fib_0_786 = low + 0.786 * diff
        fib_1 = high
        
        return pd.DataFrame({
            'support': low,
            'resistance': high,
            'pivot': close,
            'fib_0': fib_0,
            'fib_0_236': fib_0_236,
            'fib_0_382': fib_0_382,
            'fib_0_5': fib_0_5,
            'fib_0_618': fib_0_618,
            'fib_0_786': fib_0_786,
            'fib_1': fib_1
        })

    def get_trend_strength(self, data, window=14):
        """Calculate trend strength using multiple indicators"""
        # ADX
        adx = ta.trend.ADXIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=window
        )
        
        # Aroon
        aroon = ta.trend.AroonIndicator(
            high=data['high'],
            low=data['low'],
            window=window
        )
        
        # CCI
        cci = ta.trend.CCIIndicator(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=window
        )
        
        return pd.DataFrame({
            'adx': adx.adx(),
            'di_plus': adx.adx_pos(),
            'di_minus': adx.adx_neg(),
            'aroon_up': aroon.aroon_up(),
            'aroon_down': aroon.aroon_down(),
            'cci': cci.cci()
        }) 