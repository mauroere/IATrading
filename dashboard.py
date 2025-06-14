import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import json
import logging
from typing import Dict, List, Tuple
import ccxt
import os
from config import TRADING_CONFIG, API_CONFIG, INDICATORS_CONFIG
from database import DatabaseHandler
from notifications import TelegramNotifier
from indicators import TechnicalIndicators
from market_analysis import MarketAnalysis
from ml_model import MLModel
from optimizer import BayesianOptimizer
from backtester import Backtester
from monitor import SystemMonitor

# Importar componentes del sistema
from profit_calculator import ProfitCalculator

class TradingDashboard:
    def __init__(self):
        """Inicializa el dashboard y sus componentes."""
        self.config = TRADING_CONFIG
        self.api_config = API_CONFIG
        self.indicators_config = INDICATORS_CONFIG
        self.setup_logging()
        self.load_config()
        self.initialize_components()
        
    def setup_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            filename='dashboard.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        """Carga la configuración del sistema."""
        try:
            with open('config.yaml', 'r') as file:
                self.config = yaml.safe_load(file)
            # Ensure indicators_config is set from YAML or fallback to INDICATORS_CONFIG
            if 'indicators_config' in self.config:
                self.indicators_config = self.config['indicators_config']
            else:
                self.indicators_config = INDICATORS_CONFIG
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
            self.indicators_config = INDICATORS_CONFIG
            raise
    
    def initialize_components(self):
        """Inicializa todos los componentes del sistema."""
        try:
            # Initialize exchange
            self.exchange = ccxt.binance({
                'apiKey': self.api_config['binance_api_key'],
                'secret': self.api_config['binance_api_secret'],
                'enableRateLimit': True
            })
            
            # Initialize database
            self.db = DatabaseHandler()
            
            # Initialize notifier
            self.notifier = TelegramNotifier()
            
            # Initialize indicators with config (always pass config)
            self.indicators = TechnicalIndicators(self.indicators_config)
            
            # Initialize market analysis
            self.market_analysis = MarketAnalysis(self.config)
            
            # Initialize ML model
            self.ml_model = MLModel(self.config)
            
            self.profit_calculator = ProfitCalculator()
            self.risk_manager = RiskManager()
            self.optimizer = BayesianOptimizer()
            self.backtester = Backtester()
            self.monitor = SystemMonitor()
        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {e}")
            raise
    
    def run(self):
        """Ejecuta el dashboard."""
        st.set_page_config(
            page_title="Sistema de Trading",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("Sistema de Trading Integrado")
        
        # Sidebar para navegación
        st.sidebar.title("Navegación")
        page = st.sidebar.radio(
            "Seleccionar página",
            ["Dashboard", "Análisis Técnico", "Gestión de Riesgo", 
             "Machine Learning", "Optimización", "Backtesting", 
             "Calculadora de Ganancias", "Monitoreo", "Análisis de Mercado", 
             "Predicciones ML", "Historial de Trading", "Estado del Sistema"]
        )
        
        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Análisis Técnico":
            self.show_technical_analysis()
        elif page == "Gestión de Riesgo":
            self.show_risk_management()
        elif page == "Machine Learning":
            self.show_ml_analysis()
        elif page == "Optimización":
            self.show_optimization()
        elif page == "Backtesting":
            self.show_backtesting()
        elif page == "Calculadora de Ganancias":
            self.show_profit_calculator()
        elif page == "Monitoreo":
            self.show_monitoring()
        elif page == "Análisis de Mercado":
            self.show_market_analysis()
        elif page == "Predicciones ML":
            self.show_ml_predictions()
        elif page == "Historial de Trading":
            self.show_trading_history()
        elif page == "Estado del Sistema":
            self.show_system_status()
    
    def show_dashboard(self):
        """Muestra el dashboard principal."""
        # Resumen del sistema
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Rendimiento Total",
                f"{self.monitor.get_total_performance():.2f}%",
                f"{self.monitor.get_daily_performance():.2f}%"
            )
        
        with col2:
            st.metric(
                "Operaciones Totales",
                self.monitor.get_total_trades(),
                self.monitor.get_daily_trades()
            )
        
        with col3:
            st.metric(
                "Tasa de Éxito",
                f"{self.monitor.get_win_rate():.2f}%",
                f"{self.monitor.get_daily_win_rate():.2f}%"
            )
        
        # Gráfico de rendimiento
        performance_data = self.monitor.get_performance_history()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=performance_data.index,
            y=performance_data['performance'],
            name="Rendimiento"
        ))
        st.plotly_chart(fig)
        
        # Últimas operaciones
        st.subheader("Últimas Operaciones")
        trades = self.monitor.get_recent_trades()
        st.dataframe(trades)
    
    def show_technical_analysis(self):
        """Muestra el análisis técnico."""
        st.header("Análisis Técnico")
        
        # Selección de par y timeframe
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox(
                "Par de Trading",
                self.config['trading_pairs']
            )
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                self.config['timeframes']
            )
        
        # Calcular indicadores
        indicators = self.indicators.calculate_all_indicators(self._get_market_data(symbol, timeframe))
        
        # Mostrar gráfico
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Gráfico de precios
        fig.add_trace(
            go.Candlestick(
                x=indicators.index,
                open=indicators['open'],
                high=indicators['high'],
                low=indicators['low'],
                close=indicators['close'],
                name="Precio"
            ),
            row=1, col=1
        )
        
        # Indicadores
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators['sma_20'],
                name="SMA 20"
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators['rsi'],
                name="RSI"
            ),
            row=2, col=1
        )
        
        st.plotly_chart(fig)
        
        # Señales de trading
        st.subheader("Señales de Trading")
        signals, signal_strength = self.indicators.get_trading_signals(indicators)
        st.dataframe(signals)
    
    def show_risk_management(self):
        """Muestra la gestión de riesgo."""
        st.header("Gestión de Riesgo")
        
        # Configuración de riesgo
        col1, col2 = st.columns(2)
        with col1:
            max_position_size = st.slider(
                "Tamaño Máximo de Posición (%)",
                1, 100, 5
            )
        with col2:
            max_drawdown = st.slider(
                "Drawdown Máximo (%)",
                1, 50, 20
            )
        
        # Actualizar configuración
        self.risk_manager.update_config({
            'max_position_size': max_position_size / 100,
            'max_drawdown': max_drawdown / 100
        })
        
        # Mostrar métricas de riesgo
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Exposición Total",
                f"{risk_metrics['total_exposure']:.2f}%"
            )
        with col2:
            st.metric(
                "Drawdown Actual",
                f"{risk_metrics['current_drawdown']:.2f}%"
            )
        with col3:
            st.metric(
                "Riesgo por Operación",
                f"{risk_metrics['risk_per_trade']:.2f}%"
            )
    
    def show_ml_analysis(self):
        """Muestra el análisis de machine learning."""
        st.header("Análisis de Machine Learning")
        
        # Estado del modelo
        model_status = self.ml_model.get_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Precisión del Modelo",
                f"{model_status['accuracy']:.2f}%"
            )
        with col2:
            st.metric(
                "Última Actualización",
                model_status['last_update']
            )
        
        # Predicciones
        st.subheader("Predicciones")
        predictions = self.ml_model.get_predictions()
        st.dataframe(predictions)
        
        # Gráfico de predicciones vs real
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions['actual'],
            name="Real"
        ))
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions['predicted'],
            name="Predicción"
        ))
        st.plotly_chart(fig)
    
    def show_optimization(self):
        """Muestra la optimización de parámetros."""
        st.header("Optimización de Parámetros")
        
        # Configuración de optimización
        col1, col2 = st.columns(2)
        with col1:
            optimization_method = st.selectbox(
                "Método de Optimización",
                ["Bayesiano", "Grid Search", "Random Search"]
            )
        with col2:
            metric = st.selectbox(
                "Métrica a Optimizar",
                ["Sharpe Ratio", "Sortino Ratio", "Win Rate"]
            )
        
        if st.button("Iniciar Optimización"):
            with st.spinner("Optimizando..."):
                results = self.optimizer.optimize(
                    method=optimization_method,
                    metric=metric
                )
                
                st.subheader("Resultados de Optimización")
                st.json(results)
    
    def show_backtesting(self):
        """Muestra el backtesting."""
        st.header("Backtesting")
        
        # Configuración de backtesting
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Fecha Inicial",
                datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "Fecha Final",
                datetime.now()
            )
        
        if st.button("Ejecutar Backtesting"):
            with st.spinner("Ejecutando backtesting..."):
                results = self.backtester.run(
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Mostrar resultados
                st.subheader("Resultados del Backtesting")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Rendimiento Total",
                        f"{results['total_return']:.2f}%"
                    )
                with col2:
                    st.metric(
                        "Sharpe Ratio",
                        f"{results['sharpe_ratio']:.2f}"
                    )
                with col3:
                    st.metric(
                        "Tasa de Éxito",
                        f"{results['win_rate']:.2f}%"
                    )
                
                # Gráfico de equity
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['equity_curve'].index,
                    y=results['equity_curve'].values,
                    name="Equity"
                ))
                st.plotly_chart(fig)
    
    def show_profit_calculator(self):
        """Muestra la calculadora de ganancias."""
        st.header("Calculadora de Ganancias")
        
        # Selección de estrategia
        strategy = st.selectbox(
            "Estrategia",
            ["Scalping", "Grid Trading", "Martingala", "DCA"]
        )
        
        # Parámetros comunes
        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input(
                "Capital Inicial",
                min_value=100,
                max_value=1000000,
                value=1000
            )
        with col2:
            days = st.number_input(
                "Días a Simular",
                min_value=1,
                max_value=365,
                value=30
            )
        
        # Parámetros específicos por estrategia
        if strategy == "Scalping":
            col1, col2 = st.columns(2)
            with col1:
                trade_size = st.slider(
                    "Tamaño de Operación (%)",
                    1, 100, 2
                )
                win_rate = st.slider(
                    "Tasa de Éxito (%)",
                    1, 100, 60
                )
            with col2:
                risk_reward = st.slider(
                    "Ratio Riesgo/Beneficio",
                    0.1, 5.0, 1.5
                )
                trades_per_day = st.number_input(
                    "Operaciones por Día",
                    min_value=1,
                    max_value=100,
                    value=10
                )
            
            if st.button("Calcular"):
                results = self.profit_calculator.calculate_scalping_profit(
                    initial_capital=initial_capital,
                    trade_size=trade_size/100,
                    win_rate=win_rate/100,
                    risk_reward=risk_reward,
                    trades_per_day=trades_per_day,
                    days=days
                )
                st.json(results)
        
        elif strategy == "Grid Trading":
            col1, col2 = st.columns(2)
            with col1:
                grid_levels = st.number_input(
                    "Niveles de Grid",
                    min_value=2,
                    max_value=100,
                    value=10
                )
                grid_spacing = st.slider(
                    "Espaciado entre Niveles (%)",
                    0.1, 10.0, 1.0
                )
            with col2:
                position_size = st.number_input(
                    "Tamaño de Posición",
                    min_value=0.01,
                    max_value=10.0,
                    value=0.1
                )
                price_range = st.slider(
                    "Rango de Precios",
                    1000, 100000,
                    (30000, 35000)
                )
            
            if st.button("Calcular"):
                results = self.profit_calculator.calculate_grid_trading_profit(
                    initial_capital=initial_capital,
                    grid_levels=grid_levels,
                    grid_spacing=grid_spacing/100,
                    position_size=position_size,
                    price_range=price_range,
                    days=days
                )
                st.json(results)
        
        elif strategy == "Martingala":
            col1, col2 = st.columns(2)
            with col1:
                base_bet = st.number_input(
                    "Apuesta Base",
                    min_value=1,
                    max_value=1000,
                    value=10
                )
                multiplier = st.slider(
                    "Multiplicador",
                    1.1, 3.0, 2.0
                )
            with col2:
                max_steps = st.number_input(
                    "Máximo de Pasos",
                    min_value=1,
                    max_value=10,
                    value=5
                )
                win_rate = st.slider(
                    "Tasa de Éxito (%)",
                    1, 100, 50
                )
            
            if st.button("Calcular"):
                results = self.profit_calculator.calculate_martingale_profit(
                    initial_capital=initial_capital,
                    base_bet=base_bet,
                    multiplier=multiplier,
                    max_steps=max_steps,
                    win_rate=win_rate/100,
                    trades_per_day=20,
                    days=days
                )
                st.json(results)
        
        elif strategy == "DCA":
            col1, col2 = st.columns(2)
            with col1:
                base_investment = st.number_input(
                    "Inversión Base",
                    min_value=10,
                    max_value=10000,
                    value=100
                )
                dca_steps = st.number_input(
                    "Pasos de DCA",
                    min_value=2,
                    max_value=20,
                    value=5
                )
            with col2:
                dca_multiplier = st.slider(
                    "Multiplicador de Inversión",
                    1.1, 3.0, 1.5
                )
                price_decrease = st.slider(
                    "Disminución de Precio por Paso (%)",
                    1, 20, 5
                )
            
            if st.button("Calcular"):
                results = self.profit_calculator.calculate_dca_profit(
                    initial_capital=initial_capital,
                    base_investment=base_investment,
                    dca_steps=dca_steps,
                    dca_multiplier=dca_multiplier,
                    price_decrease=price_decrease/100,
                    days=days
                )
                st.json(results)
    
    def show_monitoring(self):
        """Muestra el monitoreo del sistema."""
        st.header("Monitoreo del Sistema")
        
        # Métricas del sistema
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "CPU",
                f"{self.monitor.get_cpu_usage():.1f}%"
            )
        with col2:
            st.metric(
                "Memoria",
                f"{self.monitor.get_memory_usage():.1f}%"
            )
        with col3:
            st.metric(
                "Latencia",
                f"{self.monitor.get_latency():.0f}ms"
            )
        
        # Gráfico de métricas
        metrics_data = self.monitor.get_metrics_history()
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        
        fig.add_trace(
            go.Scatter(
                x=metrics_data.index,
                y=metrics_data['cpu'],
                name="CPU"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=metrics_data.index,
                y=metrics_data['memory'],
                name="Memoria"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=metrics_data.index,
                y=metrics_data['latency'],
                name="Latencia"
            ),
            row=3, col=1
        )
        
        st.plotly_chart(fig)
        
        # Alertas
        st.subheader("Alertas del Sistema")
        alerts = self.monitor.get_alerts()
        st.dataframe(alerts)

    def show_market_analysis(self):
        """Muestra el análisis de mercado"""
        st.header("Análisis de Mercado")
        
        symbol = st.selectbox(
            "Par de Trading",
            self.config['trading_pairs'],
            key="market_analysis"
        )
        
        analysis = self.market_analysis.analyze_market_conditions(symbol)
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Análisis Técnico")
            st.write(analysis['technical'])
            
            st.subheader("Análisis de Volumen")
            st.write(analysis['volume'])
            
        with col2:
            st.subheader("Análisis de Sentimiento")
            st.write(analysis['sentiment'])
            
            st.subheader("Flujo de Órdenes")
            st.write(analysis['order_flow'])
            
        st.subheader("Condición General del Mercado")
        st.write(f"Estado: {analysis['market_condition']}")
        
    def show_ml_predictions(self):
        """Muestra predicciones del modelo ML"""
        st.header("Predicciones ML")
        
        symbol = st.selectbox(
            "Par de Trading",
            self.config['trading_pairs'],
            key="ml_predictions"
        )
        
        # Obtener datos
        df = self._get_market_data(symbol, '1h')
        
        # Preparar características
        features = self.ml_model.prepare_features(df)
        
        # Obtener predicción
        prediction = self.ml_model.predict(features)
        
        # Mostrar resultados
        st.subheader("Predicción")
        st.write(f"Señal: {'Compra' if prediction == 1 else 'Venta' if prediction == -1 else 'Neutral'}")
        
        # Mostrar probabilidades
        probabilities = self.ml_model.predict_proba(features)
        st.write("Probabilidades:", probabilities)
        
    def show_trading_history(self):
        """Muestra historial de trading"""
        st.header("Historial de Trading")
        
        # Obtener trades
        trades = self.db.get_trades()
        
        # Mostrar tabla
        st.dataframe(trades)
        
        # Mostrar estadísticas
        st.subheader("Estadísticas")
        stats = self.db.get_daily_stats()
        st.dataframe(stats)
        
    def show_system_status(self):
        """Muestra estado del sistema"""
        st.header("Estado del Sistema")
        
        # Verificar conexiones
        st.subheader("Conexiones")
        self._verify_connections()
        
        # Mostrar logs
        st.subheader("Logs")
        logs = self.db.get_logs()
        st.dataframe(logs)
        
    def _verify_connections(self):
        """Verifica conexiones con APIs"""
        try:
            # Verificar Binance
            self.exchange.fetch_ticker(self.config['trading_pairs'][0])
            st.success("✅ Conexión con Binance establecida")
        except Exception as e:
            st.error(f"❌ Error en conexión con Binance: {str(e)}")
            
        try:
            # Verificar Telegram
            self.notifier.send_test_message()
            st.success("✅ Conexión con Telegram establecida")
        except Exception as e:
            st.error(f"❌ Error en conexión con Telegram: {str(e)}")
            
    def _get_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Obtiene datos del mercado"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            st.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()

def main():
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 