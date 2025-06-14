import os
import sys
import logging
import yaml
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import ccxt
import time
import threading
import queue
import signal
import psutil

# Importar módulos del sistema
from indicators import TechnicalIndicators
from risk_manager import RiskManager
from ml_model import MLModel
from market_analysis import MarketAnalysis
from optimizer import BayesianOptimizer
from backtester import Backtester
from monitor import SystemMonitor

class SystemTester:
    def __init__(self):
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('system_test.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar componentes
        self._initialize_components()
        
        # Cola para comunicación entre hilos
        self.message_queue = queue.Queue()
        
        # Flag para control de ejecución
        self.running = True
        
    def _load_config(self) -> dict:
        """Carga configuración del sistema"""
        try:
            # Cargar variables de entorno
            load_dotenv()
            
            # Cargar configuración YAML
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Agregar variables de entorno
            config['binance_api_key'] = os.getenv('BINANCE_API_KEY')
            config['binance_api_secret'] = os.getenv('BINANCE_API_SECRET')
            config['telegram_bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN')
            config['telegram_chat_id'] = os.getenv('TELEGRAM_CHAT_ID')
            
            return config
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {str(e)}")
            sys.exit(1)
    
    def _initialize_components(self):
        """Inicializa componentes del sistema"""
        try:
            # Inicializar exchange
            self.exchange = ccxt.binance({
                'apiKey': self.config['binance_api_key'],
                'secret': self.config['binance_api_secret']
            })
            
            # Inicializar componentes
            self.indicators = TechnicalIndicators(self.config)
            self.risk_manager = RiskManager(self.config)
            self.ml_model = MLModel(self.config)
            self.market_analysis = MarketAnalysis(self.config)
            self.optimizer = BayesianOptimizer(self.config)
            self.backtester = Backtester(self.config)
            self.monitor = SystemMonitor(self.config)
            
            self.logger.info("Componentes inicializados correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {str(e)}")
            sys.exit(1)
    
    def test_market_analysis(self):
        """Prueba el análisis de mercado"""
        try:
            self.logger.info("Iniciando prueba de análisis de mercado...")
            
            # Obtener datos de mercado
            symbol = self.config['trading_pairs'][0]
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Realizar análisis
            market_conditions = self.market_analysis.analyze_market_conditions(symbol)
            
            # Verificar resultados
            assert market_conditions, "Análisis de mercado falló"
            assert 'technical' in market_conditions, "Falta análisis técnico"
            assert 'volume' in market_conditions, "Falta análisis de volumen"
            assert 'sentiment' in market_conditions, "Falta análisis de sentimiento"
            assert 'order_flow' in market_conditions, "Falta análisis de flujo de órdenes"
            
            self.logger.info("Prueba de análisis de mercado completada exitosamente")
            return True
        except Exception as e:
            self.logger.error(f"Error en prueba de análisis de mercado: {str(e)}")
            return False
    
    def test_indicators(self):
        """Prueba los indicadores técnicos"""
        try:
            self.logger.info("Iniciando prueba de indicadores...")
            
            # Obtener datos
            symbol = self.config['trading_pairs'][0]
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calcular indicadores
            df = self.indicators.calculate_all_indicators(df)
            
            # Verificar resultados
            required_indicators = [
                'sma_20', 'sma_50', 'sma_200',
                'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_middle', 'bb_lower',
                'atr', 'supertrend', 'ichimoku_tenkan',
                'ichimoku_kijun', 'ichimoku_senkou_b',
                'volume_ma', 'obv', 'cmf'
            ]
            
            for indicator in required_indicators:
                assert indicator in df.columns, f"Falta indicador: {indicator}"
            
            # Probar señales
            signals = self.indicators.get_trading_signals(df)
            assert 'signal' in signals, "Falta señal de trading"
            assert 'signal_strength' in signals, "Falta fuerza de señal"
            
            self.logger.info("Prueba de indicadores completada exitosamente")
            return True
        except Exception as e:
            self.logger.error(f"Error en prueba de indicadores: {str(e)}")
            return False
    
    def test_risk_management(self):
        """Prueba la gestión de riesgo"""
        try:
            self.logger.info("Iniciando prueba de gestión de riesgo...")
            
            # Probar cálculo de tamaño de posición
            balance = 1000
            price = 100
            atr = 2
            signal_strength = 0.8
            
            position_size = self.risk_manager.calculate_position_size(
                balance, price, atr, signal_strength
            )
            assert 0 < position_size <= balance, "Tamaño de posición inválido"
            
            # Probar cálculo de stop loss y take profit
            stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
                price, atr, signal_strength
            )
            assert stop_loss < price, "Stop loss inválido"
            assert take_profit > price, "Take profit inválido"
            
            # Probar validación de trade
            trade_valid = self.risk_manager.validate_trade(
                price, position_size, signal_strength
            )
            assert isinstance(trade_valid, bool), "Validación de trade inválida"
            
            self.logger.info("Prueba de gestión de riesgo completada exitosamente")
            return True
        except Exception as e:
            self.logger.error(f"Error en prueba de gestión de riesgo: {str(e)}")
            return False
    
    def test_ml_model(self):
        """Prueba el modelo de machine learning"""
        try:
            self.logger.info("Iniciando prueba de modelo ML...")
            
            # Obtener datos
            symbol = self.config['trading_pairs'][0]
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Preparar features
            features = self.ml_model.prepare_features(df)
            assert not features.empty, "Features vacías"
            
            # Entrenar modelo
            self.ml_model.train(features)
            
            # Probar predicciones
            predictions = self.ml_model.predict(features)
            assert len(predictions) == len(features), "Error en predicciones"
            
            # Verificar métricas
            metrics = self.ml_model.get_performance_metrics()
            assert 'accuracy' in metrics, "Falta métrica de accuracy"
            assert 'precision' in metrics, "Falta métrica de precision"
            assert 'recall' in metrics, "Falta métrica de recall"
            
            self.logger.info("Prueba de modelo ML completada exitosamente")
            return True
        except Exception as e:
            self.logger.error(f"Error en prueba de modelo ML: {str(e)}")
            return False
    
    def test_optimizer(self):
        """Prueba el optimizador de parámetros"""
        try:
            self.logger.info("Iniciando prueba de optimizador...")
            
            # Función de evaluación
            def evaluation_function(params):
                return np.random.random()  # Simulación de evaluación
            
            # Ejecutar optimización
            best_params = self.optimizer.optimize(
                evaluation_function,
                n_iterations=10
            )
            
            assert best_params, "Optimización falló"
            assert all(param in best_params for param in self.optimizer.param_bounds), "Faltan parámetros"
            
            # Verificar reporte
            report = self.optimizer.get_optimization_report()
            assert report, "Reporte de optimización falló"
            
            self.logger.info("Prueba de optimizador completada exitosamente")
            return True
        except Exception as e:
            self.logger.error(f"Error en prueba de optimizador: {str(e)}")
            return False
    
    def test_backtester(self):
        """Prueba el backtester"""
        try:
            self.logger.info("Iniciando prueba de backtester...")
            
            # Obtener datos
            symbol = self.config['trading_pairs'][0]
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ejecutar backtest
            results = self.backtester.run_backtest(df, self.config)
            
            assert results, "Backtest falló"
            assert 'trades' in results, "Faltan trades"
            assert 'metrics' in results, "Faltan métricas"
            
            # Verificar métricas
            metrics = results['metrics']
            assert 'win_rate' in metrics, "Falta win rate"
            assert 'profit_factor' in metrics, "Falta profit factor"
            assert 'sharpe_ratio' in metrics, "Falta Sharpe ratio"
            
            self.logger.info("Prueba de backtester completada exitosamente")
            return True
        except Exception as e:
            self.logger.error(f"Error en prueba de backtester: {str(e)}")
            return False
    
    def test_monitor(self):
        """Prueba el sistema de monitoreo"""
        try:
            self.logger.info("Iniciando prueba de monitoreo...")
            
            # Iniciar monitoreo
            self.monitor.start_monitoring()
            
            # Esperar algunas métricas
            time.sleep(5)
            
            # Verificar métricas del sistema
            system_metrics = self.monitor._get_system_metrics_history()
            assert not system_metrics.empty, "Faltan métricas del sistema"
            
            # Verificar métricas de trading
            trading_metrics = self.monitor._get_trading_metrics_history()
            assert not trading_metrics.empty, "Faltan métricas de trading"
            
            # Generar reporte
            self.monitor.generate_report('monitor_report.png')
            assert os.path.exists('monitor_report.png'), "Reporte no generado"
            
            self.logger.info("Prueba de monitoreo completada exitosamente")
            return True
        except Exception as e:
            self.logger.error(f"Error en prueba de monitoreo: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas"""
        try:
            self.logger.info("Iniciando pruebas del sistema...")
            
            # Lista de pruebas
            tests = [
                self.test_market_analysis,
                self.test_indicators,
                self.test_risk_management,
                self.test_ml_model,
                self.test_optimizer,
                self.test_backtester,
                self.test_monitor
            ]
            
            # Ejecutar pruebas
            results = {}
            for test in tests:
                test_name = test.__name__
                self.logger.info(f"Ejecutando prueba: {test_name}")
                results[test_name] = test()
            
            # Generar reporte
            self._generate_test_report(results)
            
            # Verificar resultados
            all_passed = all(results.values())
            if all_passed:
                self.logger.info("Todas las pruebas completadas exitosamente")
            else:
                self.logger.error("Algunas pruebas fallaron")
            
            return all_passed
        except Exception as e:
            self.logger.error(f"Error ejecutando pruebas: {str(e)}")
            return False
    
    def _generate_test_report(self, results: dict):
        """Genera reporte de pruebas"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'summary': {
                    'total_tests': len(results),
                    'passed_tests': sum(results.values()),
                    'failed_tests': len(results) - sum(results.values())
                }
            }
            
            # Guardar reporte
            with open('test_report.json', 'w') as f:
                json.dump(report, f, indent=4)
            
            self.logger.info("Reporte de pruebas generado exitosamente")
        except Exception as e:
            self.logger.error(f"Error generando reporte de pruebas: {str(e)}")
    
    def cleanup(self):
        """Limpia recursos"""
        try:
            self.running = False
            self.monitor.close()
            self.logger.info("Recursos liberados correctamente")
        except Exception as e:
            self.logger.error(f"Error limpiando recursos: {str(e)}")

def main():
    # Configurar manejador de señales
    def signal_handler(signum, frame):
        print("\nDeteniendo pruebas...")
        tester.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Ejecutar pruebas
    tester = SystemTester()
    success = tester.run_all_tests()
    
    # Limpiar recursos
    tester.cleanup()
    
    # Salir con código de estado apropiado
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 