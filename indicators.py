import pandas as pd
import numpy as np
import ta
from config import INDICATORS_CONFIG
from typing import Dict, List, Tuple
import logging

class TechnicalIndicators:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_rsi(self, data):
        return ta.momentum.RSIIndicator(
            close=data['close'],
            window=self.config['rsi_period']
        ).rsi()

    def calculate_macd(self, data):
        macd = ta.trend.MACD(
            close=data['close'],
            window_slow=self.config['macd_slow'],
            window_fast=self.config['macd_fast'],
            window_sign=self.config['macd_signal']
        )
        return pd.DataFrame({
            'macd': macd.macd(),
            'macd_signal': macd.macd_signal(),
            'macd_diff': macd.macd_diff()
        })

    def calculate_atr(self, data):
        return ta.volatility.AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.config['atr_period']
        ).average_true_range()

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        bb = ta.volatility.BollingerBands(
            close=data['close'],
            window=window,
            window_dev=num_std
        )
        return pd.DataFrame({
            'bb_high': bb.bollinger_hband(),
            'bb_mid': bb.bollinger_mavg(),
            'bb_low': bb.bollinger_lband(),
            'bb_width': bb.bollinger_wband()
        })

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

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores técnicos"""
        try:
            # Indicadores de tendencia
            df = self.add_moving_averages(df)
            df = self.add_ichimoku_cloud(df)
            df = self.add_supertrend(df)
            
            # Indicadores de momentum
            df = self.add_rsi(df)
            df = self.add_stochastic(df)
            df = self.add_macd(df)
            df = self.add_cci(df)
            
            # Indicadores de volatilidad
            df = self.add_bollinger_bands(df)
            df = self.add_atr(df)
            df = self.add_keltner_channels(df)
            
            # Indicadores de volumen
            df = self.add_volume_indicators(df)
            
            # Indicadores de patrones
            df = self.add_candlestick_patterns(df)
            
            # Limpieza de datos
            df = df.dropna()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculando indicadores: {str(e)}")
            raise

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

    def get_trading_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Genera señales de trading basadas en múltiples indicadores"""
        signals = pd.DataFrame(index=df.index)
        signal_strength = {}
        
        # Señales de tendencia
        signals['trend_signal'] = self._get_trend_signals(df)
        signal_strength['trend'] = self._calculate_signal_strength(signals['trend_signal'])
        
        # Señales de momentum
        signals['momentum_signal'] = self._get_momentum_signals(df)
        signal_strength['momentum'] = self._calculate_signal_strength(signals['momentum_signal'])
        
        # Señales de volatilidad
        signals['volatility_signal'] = self._get_volatility_signals(df)
        signal_strength['volatility'] = self._calculate_signal_strength(signals['volatility_signal'])
        
        # Señales de volumen
        signals['volume_signal'] = self._get_volume_signals(df)
        signal_strength['volume'] = self._calculate_signal_strength(signals['volume_signal'])
        
        # Señales de patrones
        signals['pattern_signal'] = self._get_pattern_signals(df)
        signal_strength['pattern'] = self._calculate_signal_strength(signals['pattern_signal'])
        
        # Señal final combinada
        signals['final_signal'] = self._combine_signals(signals, signal_strength)
        
        return signals, signal_strength

    def _get_trend_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genera señales basadas en tendencia"""
        signals = pd.Series(0, index=df.index)
        
        # Ichimoku Cloud
        signals += np.where(df['close'] > df['ichimoku_span_a'], 1, 0)
        signals += np.where(df['close'] > df['ichimoku_span_b'], 1, 0)
        signals -= np.where(df['close'] < df['ichimoku_span_a'], 1, 0)
        signals -= np.where(df['close'] < df['ichimoku_span_b'], 1, 0)
        
        # Supertrend
        signals += np.where(df['supertrend_direction'] == 1, 1, 0)
        signals -= np.where(df['supertrend_direction'] == -1, 1, 0)
        
        # Moving Averages
        signals += np.where(df['close'] > df['sma_50'], 1, 0)
        signals += np.where(df['close'] > df['sma_200'], 1, 0)
        signals -= np.where(df['close'] < df['sma_50'], 1, 0)
        signals -= np.where(df['close'] < df['sma_200'], 1, 0)
        
        return signals

    def _get_momentum_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genera señales basadas en momentum"""
        signals = pd.Series(0, index=df.index)
        
        # RSI
        signals += np.where(df['rsi'] < 30, 1, 0)
        signals -= np.where(df['rsi'] > 70, 1, 0)
        
        # Stochastic
        signals += np.where((df['stoch_k'] < 20) & (df['stoch_d'] < 20), 1, 0)
        signals -= np.where((df['stoch_k'] > 80) & (df['stoch_d'] > 80), 1, 0)
        
        # MACD
        signals += np.where(df['macd'] > df['macd_signal'], 1, 0)
        signals -= np.where(df['macd'] < df['macd_signal'], 1, 0)
        
        return signals

    def _get_volatility_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genera señales basadas en volatilidad"""
        signals = pd.Series(0, index=df.index)
        
        # Bollinger Bands
        signals += np.where(df['close'] < df['bb_low'], 1, 0)
        signals -= np.where(df['close'] > df['bb_high'], 1, 0)
        
        # Keltner Channels
        signals += np.where(df['close'] < df['kc_lower'], 1, 0)
        signals -= np.where(df['close'] > df['kc_upper'], 1, 0)
        
        return signals

    def _get_volume_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genera señales basadas en volumen"""
        signals = pd.Series(0, index=df.index)
        
        # VWAP
        signals += np.where(df['close'] > df['vwap'], 1, 0)
        signals -= np.where(df['close'] < df['vwap'], 1, 0)
        
        # OBV
        signals += np.where(df['obv'].diff() > 0, 1, 0)
        signals -= np.where(df['obv'].diff() < 0, 1, 0)
        
        # CMF
        signals += np.where(df['cmf'] > 0, 1, 0)
        signals -= np.where(df['cmf'] < 0, 1, 0)
        
        return signals

    def _get_pattern_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genera señales basadas en patrones de velas"""
        signals = pd.Series(0, index=df.index)
        
        # Patrones alcistas
        bullish_patterns = ['cdlhammer', 'cdlmorningstar', 'cdlengulfing', 'cdlharami']
        for pattern in bullish_patterns:
            signals += np.where(df[pattern] > 0, 1, 0)
        
        # Patrones bajistas
        bearish_patterns = ['cdlshootingstar', 'cdleveningstar', 'cdlengulfing', 'cdlharami']
        for pattern in bearish_patterns:
            signals -= np.where(df[pattern] < 0, 1, 0)
        
        return signals

    def _calculate_signal_strength(self, signal: pd.Series) -> float:
        """Calcula la fuerza de la señal"""
        return abs(signal.mean())

    def _combine_signals(self, signals: pd.DataFrame, signal_strength: Dict) -> pd.Series:
        """Combina todas las señales con pesos basados en su fuerza"""
        total_strength = sum(signal_strength.values())
        weights = {k: v/total_strength for k, v in signal_strength.items()}
        
        final_signal = pd.Series(0, index=signals.index)
        for signal_type, weight in weights.items():
            final_signal += signals[f'{signal_type}_signal'] * weight
        
        return final_signal

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