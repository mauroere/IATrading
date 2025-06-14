import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import requests
from datetime import datetime, timedelta
import ta  # Usar la librería ta en vez de talib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ccxt
import json

class MarketAnalysis:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchange = ccxt.binance({
            'apiKey': config['binance_api_key'],
            'secret': config['binance_api_secret']
        })
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5)

    def analyze_market_conditions(self, symbol: str) -> Dict:
        """Analiza condiciones generales del mercado"""
        try:
            # Obtener datos del mercado
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Análisis técnico
            technical_analysis = self._perform_technical_analysis(df)
            
            # Análisis de volumen
            volume_analysis = self._analyze_volume(df)
            
            # Análisis de sentimiento
            sentiment_analysis = self._analyze_sentiment(symbol)
            
            # Análisis de flujo de órdenes
            order_flow = self._analyze_order_flow(symbol)
            
            # Análisis de correlación
            correlation_analysis = self._analyze_correlation(symbol)
            
            return {
                'technical': technical_analysis,
                'volume': volume_analysis,
                'sentiment': sentiment_analysis,
                'order_flow': order_flow,
                'correlation': correlation_analysis,
                'market_condition': self._determine_market_condition(
                    technical_analysis,
                    volume_analysis,
                    sentiment_analysis,
                    order_flow
                )
            }
        except Exception as e:
            self.logger.error(f"Error en análisis de mercado: {str(e)}")
            return {}

    def _perform_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Realiza análisis técnico completo"""
        try:
            # Tendencia
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
            
            # Momentum
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Volatilidad
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            df['atr'] = atr.average_true_range()
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Patrones
            # Determinar condiciones
            trend = 'bullish' if df['close'].iloc[-1] > df['sma_200'].iloc[-1] else 'bearish'
            momentum = 'strong' if df['rsi'].iloc[-1] > 70 or df['rsi'].iloc[-1] < 30 else 'neutral'
            volatility = 'high' if df['atr'].iloc[-1] > df['atr'].mean() * 1.5 else 'normal'
            
            return {
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility
            }
        except Exception as e:
            self.logger.error(f"Error en análisis técnico: {str(e)}")
            return {}

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analiza patrones de volumen"""
        try:
            # Volumen relativo
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma']
            
            # OBV
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            df['obv'] = obv.on_balance_volume()
            
            # Chaikin Money Flow
            cmf = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'])
            df['cmf'] = cmf.chaikin_money_flow()
            
            # Análisis de divergencia
            price_high = df['close'].rolling(window=20).max()
            obv_high = df['obv'].rolling(window=20).max()
            
            volume_trend = 'increasing' if df['relative_volume'].iloc[-1] > 1.5 else 'decreasing'
            money_flow = 'positive' if df['cmf'].iloc[-1] > 0 else 'negative'
            
            return {
                'volume_trend': volume_trend,
                'money_flow': money_flow,
                'relative_volume': df['relative_volume'].iloc[-1],
                'obv_trend': 'up' if df['obv'].diff().iloc[-1] > 0 else 'down'
            }
        except Exception as e:
            self.logger.error(f"Error en análisis de volumen: {str(e)}")
            return {}

    def _analyze_sentiment(self, symbol: str) -> Dict:
        """Analiza sentimiento del mercado"""
        try:
            # Obtener datos de redes sociales y noticias
            social_sentiment = self._get_social_sentiment(symbol)
            news_sentiment = self._get_news_sentiment(symbol)
            
            # Obtener datos de miedo y codicia
            fear_greed = self._get_fear_greed_index()
            
            # Calcular sentimiento general
            sentiment_score = (
                social_sentiment['score'] * 0.4 +
                news_sentiment['score'] * 0.4 +
                fear_greed['score'] * 0.2
            )
            
            return {
                'overall_sentiment': 'bullish' if sentiment_score > 0.6 else 'bearish' if sentiment_score < 0.4 else 'neutral',
                'sentiment_score': sentiment_score,
                'social_sentiment': social_sentiment,
                'news_sentiment': news_sentiment,
                'fear_greed': fear_greed
            }
        except Exception as e:
            self.logger.error(f"Error en análisis de sentimiento: {str(e)}")
            return {}

    def _analyze_order_flow(self, symbol: str) -> Dict:
        """Analiza flujo de órdenes"""
        try:
            # Obtener libro de órdenes
            order_book = self.exchange.fetch_order_book(symbol)
            
            # Análisis de profundidad
            bid_depth = sum(bid[1] for bid in order_book['bids'][:10])
            ask_depth = sum(ask[1] for ask in order_book['asks'][:10])
            
            # Análisis de presión de compra/venta
            buy_pressure = bid_depth / (bid_depth + ask_depth)
            
            # Análisis de spread
            spread = (order_book['asks'][0][0] - order_book['bids'][0][0]) / order_book['bids'][0][0]
            
            return {
                'buy_pressure': buy_pressure,
                'spread': spread,
                'depth_ratio': bid_depth / ask_depth if ask_depth > 0 else float('inf'),
                'order_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth)
            }
        except Exception as e:
            self.logger.error(f"Error en análisis de flujo de órdenes: {str(e)}")
            return {}

    def _analyze_correlation(self, symbol: str) -> Dict:
        """Analiza correlación con otros activos"""
        try:
            # Obtener datos de otros activos
            correlated_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            correlations = {}
            
            for corr_symbol in correlated_symbols:
                if corr_symbol != symbol:
                    corr_data = self.exchange.fetch_ohlcv(corr_symbol, '1h', limit=100)
                    corr_df = pd.DataFrame(corr_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    corr_df['timestamp'] = pd.to_datetime(corr_df['timestamp'], unit='ms')
                    corr_df.set_index('timestamp', inplace=True)
                    
                    correlation = corr_df['close'].corr(pd.Series(self.exchange.fetch_ohlcv(symbol, '1h', limit=100))[:, 4])
                    correlations[corr_symbol] = correlation
            
            return {
                'correlations': correlations,
                'highest_correlation': max(correlations.items(), key=lambda x: abs(x[1])),
                'lowest_correlation': min(correlations.items(), key=lambda x: abs(x[1]))
            }
        except Exception as e:
            self.logger.error(f"Error en análisis de correlación: {str(e)}")
            return {}

    def _determine_market_condition(self, technical: Dict, volume: Dict, 
                                  sentiment: Dict, order_flow: Dict) -> str:
        """Determina la condición general del mercado"""
        try:
            # Ponderar diferentes factores
            technical_weight = 0.4
            volume_weight = 0.2
            sentiment_weight = 0.2
            order_flow_weight = 0.2
            
            # Calcular score técnico
            technical_score = 1 if technical.get('trend') == 'bullish' else 0
            technical_score += 0.5 if technical.get('momentum') == 'strong' else 0
            
            # Calcular score de volumen
            volume_score = 1 if volume.get('volume_trend') == 'increasing' else 0
            volume_score += 0.5 if volume.get('money_flow') == 'positive' else 0
            
            # Calcular score de sentimiento
            sentiment_score = 1 if sentiment.get('overall_sentiment') == 'bullish' else 0
            
            # Calcular score de flujo de órdenes
            order_flow_score = 1 if order_flow.get('buy_pressure', 0) > 0.6 else 0
            
            # Calcular score final
            final_score = (
                technical_score * technical_weight +
                volume_score * volume_weight +
                sentiment_score * sentiment_weight +
                order_flow_score * order_flow_weight
            )
            
            # Determinar condición
            if final_score > 0.7:
                return 'strong_bullish'
            elif final_score > 0.5:
                return 'bullish'
            elif final_score > 0.3:
                return 'neutral'
            elif final_score > 0.2:
                return 'bearish'
            else:
                return 'strong_bearish'
        except Exception as e:
            self.logger.error(f"Error determinando condición de mercado: {str(e)}")
            return 'unknown'

    def _get_social_sentiment(self, symbol: str) -> Dict:
        """Obtiene sentimiento de redes sociales"""
        # Implementar integración con APIs de redes sociales
        return {'score': 0.5, 'source': 'social_media'}

    def _get_news_sentiment(self, symbol: str) -> Dict:
        """Obtiene sentimiento de noticias"""
        # Implementar integración con APIs de noticias
        return {'score': 0.5, 'source': 'news'}

    def _get_fear_greed_index(self) -> Dict:
        """Obtiene índice de miedo y codicia"""
        # Implementar integración con API de Fear & Greed Index
        return {'score': 0.5, 'source': 'fear_greed_index'} 