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
import requests
import telegram
from typing import Dict, List, Tuple

# Importar módulos del sistema
from indicators import TechnicalIndicators
from risk_manager import RiskManager
from ml_model import MLModel
from market_analysis import MarketAnalysis
from optimizer import BayesianOptimizer
from backtester import Backtester
from monitor import SystemMonitor

class SystemVerifier:
    def __init__(self):
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('system_verification.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar componentes
        self._initialize_components()
        
        # Resultados de verificación
        self.verification_results = {
            'components': {},
            'dependencies': {},
            'connections': {},
            'performance': {},
            'security': {}
        }
    
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
            
            # Inicializar bot de Telegram
            self.telegram_bot = telegram.Bot(token=self.config['telegram_bot_token'])
            
            self.logger.info("Componentes inicializados correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {str(e)}")
            sys.exit(1)
    
    def verify_system(self):
        """Verifica la funcionalidad completa del sistema"""
        try:
            self.logger.info("Iniciando verificación del sistema...")
            
            # Verificar componentes
            self._verify_components()
            
            # Verificar dependencias
            self._verify_dependencies()
            
            # Verificar conexiones
            self._verify_connections()
            
            # Verificar rendimiento
            self._verify_performance()
            
            # Verificar seguridad
            self._verify_security()
            
            # Generar reporte
            self._generate_verification_report()
            
            # Verificar resultados
            all_passed = all(
                all(category.values())
                for category in self.verification_results.values()
            )
            
            if all_passed:
                self.logger.info("Verificación completada exitosamente")
            else:
                self.logger.error("Algunas verificaciones fallaron")
            
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Error en verificación del sistema: {str(e)}")
            return False
    
    def _verify_components(self):
        """Verifica componentes del sistema"""
        try:
            # Verificar indicadores
            self.verification_results['components']['indicators'] = self._verify_indicators()
            
            # Verificar gestión de riesgo
            self.verification_results['components']['risk_manager'] = self._verify_risk_manager()
            
            # Verificar modelo ML
            self.verification_results['components']['ml_model'] = self._verify_ml_model()
            
            # Verificar análisis de mercado
            self.verification_results['components']['market_analysis'] = self._verify_market_analysis()
            
            # Verificar optimizador
            self.verification_results['components']['optimizer'] = self._verify_optimizer()
            
            # Verificar backtester
            self.verification_results['components']['backtester'] = self._verify_backtester()
            
            # Verificar monitor
            self.verification_results['components']['monitor'] = self._verify_monitor()
            
        except Exception as e:
            self.logger.error(f"Error verificando componentes: {str(e)}")
    
    def _verify_dependencies(self):
        """Verifica dependencias del sistema"""
        try:
            # Verificar Python
            self.verification_results['dependencies']['python'] = self._verify_python_version()
            
            # Verificar paquetes
            self.verification_results['dependencies']['packages'] = self._verify_packages()
            
            # Verificar archivos
            self.verification_results['dependencies']['files'] = self._verify_files()
            
            # Verificar permisos
            self.verification_results['dependencies']['permissions'] = self._verify_permissions()
            
        except Exception as e:
            self.logger.error(f"Error verificando dependencias: {str(e)}")
    
    def _verify_connections(self):
        """Verifica conexiones del sistema"""
        try:
            # Verificar Binance
            self.verification_results['connections']['binance'] = self._verify_binance_connection()
            
            # Verificar Telegram
            self.verification_results['connections']['telegram'] = self._verify_telegram_connection()
            
            # Verificar base de datos
            self.verification_results['connections']['database'] = self._verify_database_connection()
            
            # Verificar red
            self.verification_results['connections']['network'] = self._verify_network_connection()
            
        except Exception as e:
            self.logger.error(f"Error verificando conexiones: {str(e)}")
    
    def _verify_performance(self):
        """Verifica rendimiento del sistema"""
        try:
            # Verificar CPU
            self.verification_results['performance']['cpu'] = self._verify_cpu_performance()
            
            # Verificar memoria
            self.verification_results['performance']['memory'] = self._verify_memory_performance()
            
            # Verificar disco
            self.verification_results['performance']['disk'] = self._verify_disk_performance()
            
            # Verificar red
            self.verification_results['performance']['network'] = self._verify_network_performance()
            
        except Exception as e:
            self.logger.error(f"Error verificando rendimiento: {str(e)}")
    
    def _verify_security(self):
        """Verifica seguridad del sistema"""
        try:
            # Verificar API keys
            self.verification_results['security']['api_keys'] = self._verify_api_keys()
            
            # Verificar permisos
            self.verification_results['security']['permissions'] = self._verify_security_permissions()
            
            # Verificar logs
            self.verification_results['security']['logs'] = self._verify_logs()
            
            # Verificar backups
            self.verification_results['security']['backups'] = self._verify_backups()
            
        except Exception as e:
            self.logger.error(f"Error verificando seguridad: {str(e)}")
    
    def _verify_indicators(self) -> bool:
        """Verifica indicadores técnicos"""
        try:
            # Obtener datos
            symbol = self.config['trading_pairs'][0]
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calcular indicadores
            df = self.indicators.calculate_all_indicators(df)
            
            # Verificar indicadores requeridos
            required_indicators = [
                'sma_20', 'sma_50', 'sma_200',
                'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_middle', 'bb_lower',
                'atr', 'supertrend', 'ichimoku_tenkan',
                'ichimoku_kijun', 'ichimoku_senkou_b',
                'volume_ma', 'obv', 'cmf'
            ]
            
            return all(indicator in df.columns for indicator in required_indicators)
        except Exception as e:
            self.logger.error(f"Error verificando indicadores: {str(e)}")
            return False
    
    def _verify_risk_manager(self) -> bool:
        """Verifica gestión de riesgo"""
        try:
            # Probar cálculo de tamaño de posición
            balance = 1000
            price = 100
            atr = 2
            signal_strength = 0.8
            
            position_size = self.risk_manager.calculate_position_size(
                balance, price, atr, signal_strength
            )
            
            # Probar cálculo de stop loss y take profit
            stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
                price, atr, signal_strength
            )
            
            # Probar validación de trade
            trade_valid = self.risk_manager.validate_trade(
                price, position_size, signal_strength
            )
            
            return all([
                0 < position_size <= balance,
                stop_loss < price,
                take_profit > price,
                isinstance(trade_valid, bool)
            ])
        except Exception as e:
            self.logger.error(f"Error verificando gestión de riesgo: {str(e)}")
            return False
    
    def _verify_ml_model(self) -> bool:
        """Verifica modelo de machine learning"""
        try:
            # Obtener datos
            symbol = self.config['trading_pairs'][0]
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Preparar features
            features = self.ml_model.prepare_features(df)
            
            # Entrenar modelo
            self.ml_model.train(features)
            
            # Probar predicciones
            predictions = self.ml_model.predict(features)
            
            # Verificar métricas
            metrics = self.ml_model.get_performance_metrics()
            
            return all([
                not features.empty,
                len(predictions) == len(features),
                'accuracy' in metrics,
                'precision' in metrics,
                'recall' in metrics
            ])
        except Exception as e:
            self.logger.error(f"Error verificando modelo ML: {str(e)}")
            return False
    
    def _verify_market_analysis(self) -> bool:
        """Verifica análisis de mercado"""
        try:
            # Analizar condiciones de mercado
            symbol = self.config['trading_pairs'][0]
            market_conditions = self.market_analysis.analyze_market_conditions(symbol)
            
            return all([
                'technical' in market_conditions,
                'volume' in market_conditions,
                'sentiment' in market_conditions,
                'order_flow' in market_conditions,
                'correlation' in market_conditions,
                'market_condition' in market_conditions
            ])
        except Exception as e:
            self.logger.error(f"Error verificando análisis de mercado: {str(e)}")
            return False
    
    def _verify_optimizer(self) -> bool:
        """Verifica optimizador"""
        try:
            # Función de evaluación
            def evaluation_function(params):
                return np.random.random()
            
            # Ejecutar optimización
            best_params = self.optimizer.optimize(
                evaluation_function,
                n_iterations=10
            )
            
            # Verificar reporte
            report = self.optimizer.get_optimization_report()
            
            return all([
                best_params is not None,
                all(param in best_params for param in self.optimizer.param_bounds),
                report is not None
            ])
        except Exception as e:
            self.logger.error(f"Error verificando optimizador: {str(e)}")
            return False
    
    def _verify_backtester(self) -> bool:
        """Verifica backtester"""
        try:
            # Obtener datos
            symbol = self.config['trading_pairs'][0]
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ejecutar backtest
            results = self.backtester.run_backtest(df, self.config)
            
            # Generar gráficos
            self.backtester.plot_results('verification_backtest.png')
            
            return all([
                results is not None,
                'trades' in results,
                'metrics' in results,
                os.path.exists('verification_backtest.png')
            ])
        except Exception as e:
            self.logger.error(f"Error verificando backtester: {str(e)}")
            return False
    
    def _verify_monitor(self) -> bool:
        """Verifica sistema de monitoreo"""
        try:
            # Iniciar monitoreo
            self.monitor.start_monitoring()
            
            # Esperar algunas métricas
            time.sleep(5)
            
            # Verificar métricas
            system_metrics = self.monitor._get_system_metrics_history()
            trading_metrics = self.monitor._get_trading_metrics_history()
            
            # Generar reporte
            self.monitor.generate_report('verification_monitor.png')
            
            return all([
                not system_metrics.empty,
                not trading_metrics.empty,
                os.path.exists('verification_monitor.png')
            ])
        except Exception as e:
            self.logger.error(f"Error verificando monitor: {str(e)}")
            return False
    
    def _verify_python_version(self) -> bool:
        """Verifica versión de Python"""
        try:
            required_version = (3, 8)
            current_version = sys.version_info[:2]
            return current_version >= required_version
        except Exception as e:
            self.logger.error(f"Error verificando versión de Python: {str(e)}")
            return False
    
    def _verify_packages(self) -> bool:
        """Verifica paquetes requeridos"""
        try:
            required_packages = [
                'pandas', 'numpy', 'ccxt', 'python-telegram-bot',
                'scikit-learn', 'matplotlib', 'seaborn', 'ta-lib',
                'schedule', 'psutil', 'requests', 'pyyaml', 'python-dotenv'
            ]
            
            import pkg_resources
            installed_packages = {pkg.key for pkg in pkg_resources.working_set}
            
            return all(pkg in installed_packages for pkg in required_packages)
        except Exception as e:
            self.logger.error(f"Error verificando paquetes: {str(e)}")
            return False
    
    def _verify_files(self) -> bool:
        """Verifica archivos requeridos"""
        try:
            required_files = [
                'config.yaml',
                '.env',
                'indicators.py',
                'risk_manager.py',
                'ml_model.py',
                'market_analysis.py',
                'optimizer.py',
                'backtester.py',
                'monitor.py',
                'run_system.py'
            ]
            
            return all(os.path.exists(file) for file in required_files)
        except Exception as e:
            self.logger.error(f"Error verificando archivos: {str(e)}")
            return False
    
    def _verify_permissions(self) -> bool:
        """Verifica permisos de archivos"""
        try:
            required_files = [
                'config.yaml',
                '.env',
                'trading_system.log',
                'system_verification.log'
            ]
            
            return all(os.access(file, os.R_OK | os.W_OK) for file in required_files)
        except Exception as e:
            self.logger.error(f"Error verificando permisos: {str(e)}")
            return False
    
    def _verify_binance_connection(self) -> bool:
        """Verifica conexión con Binance"""
        try:
            # Probar conexión
            self.exchange.fetch_ticker(self.config['trading_pairs'][0])
            return True
        except Exception as e:
            self.logger.error(f"Error verificando conexión con Binance: {str(e)}")
            return False
    
    def _verify_telegram_connection(self) -> bool:
        """Verifica conexión con Telegram"""
        try:
            # Probar conexión
            self.telegram_bot.get_me()
            return True
        except Exception as e:
            self.logger.error(f"Error verificando conexión con Telegram: {str(e)}")
            return False
    
    def _verify_database_connection(self) -> bool:
        """Verifica conexión con base de datos"""
        try:
            # Probar conexión
            cursor = self.monitor.db_connection.cursor()
            cursor.execute('SELECT 1')
            return True
        except Exception as e:
            self.logger.error(f"Error verificando conexión con base de datos: {str(e)}")
            return False
    
    def _verify_network_connection(self) -> bool:
        """Verifica conexión de red"""
        try:
            # Probar conexión
            requests.get('https://api.binance.com/api/v3/ping', timeout=5)
            return True
        except Exception as e:
            self.logger.error(f"Error verificando conexión de red: {str(e)}")
            return False
    
    def _verify_cpu_performance(self) -> bool:
        """Verifica rendimiento de CPU"""
        try:
            # Medir uso de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 80
        except Exception as e:
            self.logger.error(f"Error verificando rendimiento de CPU: {str(e)}")
            return False
    
    def _verify_memory_performance(self) -> bool:
        """Verifica rendimiento de memoria"""
        try:
            # Medir uso de memoria
            memory = psutil.virtual_memory()
            return memory.percent < 80
        except Exception as e:
            self.logger.error(f"Error verificando rendimiento de memoria: {str(e)}")
            return False
    
    def _verify_disk_performance(self) -> bool:
        """Verifica rendimiento de disco"""
        try:
            # Medir uso de disco
            disk = psutil.disk_usage('/')
            return disk.percent < 80
        except Exception as e:
            self.logger.error(f"Error verificando rendimiento de disco: {str(e)}")
            return False
    
    def _verify_network_performance(self) -> bool:
        """Verifica rendimiento de red"""
        try:
            # Medir latencia
            start_time = time.time()
            requests.get('https://api.binance.com/api/v3/ping')
            latency = (time.time() - start_time) * 1000
            
            return latency < 1000
        except Exception as e:
            self.logger.error(f"Error verificando rendimiento de red: {str(e)}")
            return False
    
    def _verify_api_keys(self) -> bool:
        """Verifica API keys"""
        try:
            return all([
                self.config['binance_api_key'],
                self.config['binance_api_secret'],
                self.config['telegram_bot_token'],
                self.config['telegram_chat_id']
            ])
        except Exception as e:
            self.logger.error(f"Error verificando API keys: {str(e)}")
            return False
    
    def _verify_security_permissions(self) -> bool:
        """Verifica permisos de seguridad"""
        try:
            # Verificar permisos de archivos sensibles
            sensitive_files = ['.env', 'config.yaml']
            return all(os.stat(file).st_mode & 0o600 == 0o600 for file in sensitive_files)
        except Exception as e:
            self.logger.error(f"Error verificando permisos de seguridad: {str(e)}")
            return False
    
    def _verify_logs(self) -> bool:
        """Verifica logs"""
        try:
            # Verificar archivos de log
            log_files = ['trading_system.log', 'system_verification.log']
            return all(os.path.exists(file) for file in log_files)
        except Exception as e:
            self.logger.error(f"Error verificando logs: {str(e)}")
            return False
    
    def _verify_backups(self) -> bool:
        """Verifica backups"""
        try:
            # Verificar directorio de backups
            backup_dir = 'backups'
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            # Crear backup de prueba
            backup_file = os.path.join(backup_dir, f'backup_test_{int(time.time())}.json')
            with open(backup_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            # Verificar backup
            return os.path.exists(backup_file)
        except Exception as e:
            self.logger.error(f"Error verificando backups: {str(e)}")
            return False
    
    def _generate_verification_report(self):
        """Genera reporte de verificación"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'results': self.verification_results,
                'summary': {
                    'total_checks': sum(len(category) for category in self.verification_results.values()),
                    'passed_checks': sum(
                        sum(1 for check in category.values() if check)
                        for category in self.verification_results.values()
                    ),
                    'failed_checks': sum(
                        sum(1 for check in category.values() if not check)
                        for category in self.verification_results.values()
                    )
                }
            }
            
            # Guardar reporte
            with open('verification_report.json', 'w') as f:
                json.dump(report, f, indent=4)
            
            # Enviar notificación
            self._send_verification_notification(report)
            
            self.logger.info("Reporte de verificación generado exitosamente")
        except Exception as e:
            self.logger.error(f"Error generando reporte de verificación: {str(e)}")
    
    def _send_verification_notification(self, report: Dict):
        """Envía notificación de verificación"""
        try:
            summary = report['summary']
            message = (
                f"🔍 Reporte de Verificación del Sistema\n\n"
                f"Total de verificaciones: {summary['total_checks']}\n"
                f"Verificaciones exitosas: {summary['passed_checks']}\n"
                f"Verificaciones fallidas: {summary['failed_checks']}\n\n"
                f"Estado: {'✅ OK' if summary['failed_checks'] == 0 else '❌ ERROR'}"
            )
            
            self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=message
            )
        except Exception as e:
            self.logger.error(f"Error enviando notificación de verificación: {str(e)}")
    
    def cleanup(self):
        """Limpia recursos"""
        try:
            self.monitor.close()
            self.logger.info("Recursos liberados correctamente")
        except Exception as e:
            self.logger.error(f"Error limpiando recursos: {str(e)}")

def main():
    # Configurar manejador de señales
    def signal_handler(signum, frame):
        print("\nDeteniendo verificación...")
        verifier.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Ejecutar verificación
    verifier = SystemVerifier()
    success = verifier.verify_system()
    
    # Limpiar recursos
    verifier.cleanup()
    
    # Salir con código de estado apropiado
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 