import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import psutil
import os
import json
import requests
import telegram
from threading import Thread, Lock
import time
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from contextlib import contextmanager
import queue
import asyncio

class SystemMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.telegram_bot = telegram.Bot(token=config['telegram_bot_token'])
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.metrics_history = []
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.db_path = config.get('logs_db', 'trading_logs.db')
        self.db_pool = queue.Queue(maxsize=5)
        self._initialize_db_pool()
        self.metrics_lock = Lock()
        self.stop_event = threading.Event()

    def _initialize_db_pool(self):
        """Initialize database connection pool"""
        for _ in range(self.db_pool.maxsize):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=5000')
            self.db_pool.put(conn)

    @contextmanager
    def _get_db_connection(self):
        """Thread-safe database connection getter"""
        conn = self.db_pool.get()
        try:
            yield conn
        finally:
            self.db_pool.put(conn)

    def _initialize_alert_thresholds(self) -> Dict:
        """Initialize alert thresholds"""
        return {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'disk_usage': 80.0,
            'network_latency': 1000,
            'error_rate': 0.05,
            'drawdown': 0.15,
            'profit_loss': -0.1,
            'trade_frequency': 0.5,
            'api_errors': 5,
            'memory_leak': 100,
            'response_time': 2000,
            'balance_change': 0.1,
            'volatility': 0.05,
            'spread': 0.002
        }

    def start_monitoring(self):
        """Start system monitoring"""
        try:
            # Start monitoring threads
            Thread(target=self._monitor_system_metrics, daemon=True).start()
            Thread(target=self._monitor_trading_metrics, daemon=True).start()
            Thread(target=self._monitor_anomalies, daemon=True).start()
            Thread(target=self._cleanup_old_data, daemon=True).start()
            
            self.logger.info("Monitoring system started")
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")

    def stop_monitoring(self):
        """Stop monitoring system"""
        self.stop_event.set()
        self._close_db_connections()

    def _close_db_connections(self):
        """Close all database connections"""
        while not self.db_pool.empty():
            conn = self.db_pool.get()
            conn.close()

    def _monitor_system_metrics(self):
        """Monitor system metrics"""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'network_latency': self._measure_network_latency(),
                    'error_rate': self._calculate_error_rate()
                }
                
                # Save metrics
                self._save_system_metrics(metrics)
                
                # Check alerts
                self._check_system_alerts(metrics)
                
                # Update history
                with self.metrics_lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Error monitoring system metrics: {str(e)}")
                time.sleep(60)

    def _monitor_trading_metrics(self):
        """Monitorea métricas de trading"""
        while not self.stop_event.is_set():
            try:
                with self._get_db_connection() as conn:
                    # Recolectar métricas
                    metrics = {
                        'timestamp': datetime.now(),
                        'profit_loss': self._calculate_profit_loss(),
                        'drawdown': self._calculate_drawdown(),
                        'trade_count': self._get_trade_count(),
                        'api_errors': self._get_api_errors(),
                        'concurrent_trades': self._get_concurrent_trades(),
                        'balance': self._get_account_balance(),
                        'volatility': self._calculate_volatility(),
                        'spread': self._calculate_spread()
                    }
                    
                    # Guardar métricas
                    self._save_trading_metrics(metrics)
                    
                    # Verificar alertas
                    self._check_trading_alerts(metrics)
                
                time.sleep(60)  # Actualizar cada minuto
            except Exception as e:
                self.logger.error(f"Error monitoreando métricas de trading: {str(e)}")
                time.sleep(60)

    def _monitor_anomalies(self):
        """Monitorea anomalías en las métricas"""
        while not self.stop_event.is_set():
            try:
                # Obtener datos históricos
                system_metrics = self._get_system_metrics_history()
                trading_metrics = self._get_trading_metrics_history()
                
                if len(system_metrics) > 0:
                    # Detectar anomalías en métricas del sistema
                    system_anomalies = self._detect_anomalies(system_metrics)
                    for anomaly in system_anomalies:
                        asyncio.run(self._send_alert('system_anomaly', f"Anomalía detectada en métricas del sistema: {anomaly}", 'high'))
                
                if len(trading_metrics) > 0:
                    # Detectar anomalías en métricas de trading
                    trading_anomalies = self._detect_anomalies(trading_metrics)
                    for anomaly in trading_anomalies:
                        asyncio.run(self._send_alert('trading_anomaly', f"Anomalía detectada en métricas de trading: {anomaly}", 'high'))
                
                time.sleep(300)  # Verificar cada 5 minutos
            except Exception as e:
                self.logger.error(f"Error monitoreando anomalías: {str(e)}")
                time.sleep(300)

    def _cleanup_old_data(self):
        """Limpia datos antiguos de la base de datos"""
        while not self.stop_event.is_set():
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Eliminar datos más antiguos de 30 días
                    cutoff_date = datetime.now() - timedelta(days=30)
                    
                    cursor.execute('''
                        DELETE FROM system_metrics 
                        WHERE timestamp < ?
                    ''', (cutoff_date,))
                    
                    cursor.execute('''
                        DELETE FROM trading_metrics 
                        WHERE timestamp < ?
                    ''', (cutoff_date,))
                    
                    cursor.execute('''
                        DELETE FROM alerts 
                        WHERE timestamp < ? AND resolved = 1
                    ''', (cutoff_date,))
                
                time.sleep(3600)  # Limpiar cada hora
            except Exception as e:
                self.logger.error(f"Error limpiando datos antiguos: {str(e)}")
                time.sleep(3600)

    def _measure_network_latency(self) -> float:
        """Mide latencia de red"""
        try:
            start_time = time.time()
            requests.get('https://api.binance.com/api/v3/ping')
            return (time.time() - start_time) * 1000  # Convertir a ms
        except Exception as e:
            self.logger.error(f"Error midiendo latencia: {str(e)}")
            return float('inf')

    def _calculate_error_rate(self) -> float:
        """Calculate error rate with proper error handling"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM alerts 
                    WHERE timestamp > datetime('now', '-1 hour')
                    AND severity = 'error'
                ''')
                error_count = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT COUNT(*) FROM trading_metrics 
                    WHERE timestamp > datetime('now', '-1 hour')
                ''')
                total_operations = cursor.fetchone()[0]
                
                return error_count / total_operations if total_operations > 0 else 0
        except Exception as e:
            self.logger.error(f"Error calculating error rate: {str(e)}")
            return 0

    def _calculate_profit_loss(self) -> float:
        """Calcula ganancia/pérdida"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT profit_loss FROM trading_metrics 
                    ORDER BY timestamp DESC LIMIT 1
                ''')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error calculando P&L: {str(e)}")
            return 0

    def _calculate_drawdown(self) -> float:
        """Calcula drawdown actual"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT MAX(profit_loss) - MIN(profit_loss) 
                    FROM trading_metrics 
                    WHERE timestamp > datetime('now', '-1 day')
                ''')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error calculando drawdown: {str(e)}")
            return 0

    def _get_trade_count(self) -> int:
        """Obtiene número de trades en la última hora"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT trade_count FROM trading_metrics 
                    WHERE timestamp > datetime('now', '-1 hour')
                    ORDER BY timestamp DESC LIMIT 1
                ''')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error obteniendo conteo de trades: {str(e)}")
            return 0

    def _get_api_errors(self) -> int:
        """Obtiene número de errores de API"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM alerts 
                    WHERE timestamp > datetime('now', '-1 hour')
                    AND alert_type = 'api_error'
                ''')
                result = cursor.fetchone()
                return result[0]
        except Exception as e:
            self.logger.error(f"Error obteniendo errores de API: {str(e)}")
            return 0

    def _get_concurrent_trades(self) -> int:
        """Obtiene número de trades concurrentes"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT concurrent_trades FROM trading_metrics 
                    ORDER BY timestamp DESC LIMIT 1
                ''')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error obteniendo trades concurrentes: {str(e)}")
            return 0

    def _get_account_balance(self) -> float:
        """Obtiene balance de la cuenta"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT balance FROM trading_metrics 
                    ORDER BY timestamp DESC LIMIT 1
                ''')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error obteniendo balance: {str(e)}")
            return 0

    def _calculate_volatility(self) -> float:
        """Calcula volatilidad actual"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT profit_loss FROM trading_metrics 
                    WHERE timestamp > datetime('now', '-1 day')
                ''')
                results = cursor.fetchall()
                if results:
                    returns = pd.Series([r[0] for r in results])
                    return returns.std()
                return 0
        except Exception as e:
            self.logger.error(f"Error calculando volatilidad: {str(e)}")
            return 0

    def _calculate_spread(self) -> float:
        """Calcula spread actual"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT spread FROM trading_metrics 
                    ORDER BY timestamp DESC LIMIT 1
                ''')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error calculando spread: {str(e)}")
            return 0

    def _save_system_metrics(self, metrics: Dict):
        """Save system metrics with proper transaction handling"""
        try:
            with self._get_db_connection() as conn:
                with conn:  # This creates a transaction
                    conn.execute('''
                        INSERT INTO system_metrics 
                        (timestamp, cpu_usage, memory_usage, disk_usage, network_latency, error_rate)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics['timestamp'],
                        metrics['cpu_usage'],
                        metrics['memory_usage'],
                        metrics['disk_usage'],
                        metrics['network_latency'],
                        metrics['error_rate']
                    ))
        except Exception as e:
            self.logger.error(f"Error saving system metrics: {str(e)}")

    def _save_trading_metrics(self, metrics: Dict):
        """Guarda métricas de trading"""
        try:
            with self._get_db_connection() as conn:
                with conn:
                    conn.execute('''
                        INSERT INTO trading_metrics 
                        (timestamp, profit_loss, drawdown, trade_count, api_errors, 
                        concurrent_trades, balance, volatility, spread)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics['timestamp'],
                        metrics['profit_loss'],
                        metrics['drawdown'],
                        metrics['trade_count'],
                        metrics['api_errors'],
                        metrics['concurrent_trades'],
                        metrics['balance'],
                        metrics['volatility'],
                        metrics['spread']
                    ))
        except Exception as e:
            self.logger.error(f"Error guardando métricas de trading: {str(e)}")

    def _check_system_alerts(self, metrics: Dict):
        """Verifica alertas del sistema"""
        try:
            # CPU
            if metrics['cpu_usage'] > self.alert_thresholds['cpu_usage']:
                asyncio.run(self._send_alert('high_cpu', f"Uso de CPU alto: {metrics['cpu_usage']}%", 'warning'))
            
            # Memoria
            if metrics['memory_usage'] > self.alert_thresholds['memory_usage']:
                asyncio.run(self._send_alert('high_memory', f"Uso de memoria alto: {metrics['memory_usage']}%", 'warning'))
            
            # Disco
            if metrics['disk_usage'] > self.alert_thresholds['disk_usage']:
                asyncio.run(self._send_alert('high_disk', f"Uso de disco alto: {metrics['disk_usage']}%", 'warning'))
            
            # Latencia
            if metrics['network_latency'] > self.alert_thresholds['network_latency']:
                asyncio.run(self._send_alert('high_latency', f"Latencia de red alta: {metrics['network_latency']}ms", 'warning'))
            
            # Tasa de errores
            if metrics['error_rate'] > self.alert_thresholds['error_rate']:
                asyncio.run(self._send_alert('high_error_rate', f"Tasa de errores alta: {metrics['error_rate']*100}%", 'error'))
        except Exception as e:
            self.logger.error(f"Error verificando alertas del sistema: {str(e)}")

    def _check_trading_alerts(self, metrics: Dict):
        """Verifica alertas de trading"""
        try:
            # Drawdown
            if metrics['drawdown'] > self.alert_thresholds['drawdown']:
                asyncio.run(self._send_alert('high_drawdown', f"Drawdown alto: {metrics['drawdown']*100}%", 'error'))
            
            # Pérdida
            if metrics['profit_loss'] < self.alert_thresholds['profit_loss']:
                asyncio.run(self._send_alert('high_loss', f"Pérdida alta: {metrics['profit_loss']*100}%", 'error'))
            
            # Frecuencia de trades
            if metrics['trade_count'] > self.alert_thresholds['trade_frequency']:
                asyncio.run(self._send_alert('high_trade_frequency', f"Frecuencia de trades alta: {metrics['trade_count']} trades/hora", 'warning'))
            
            # Errores de API
            if metrics['api_errors'] > self.alert_thresholds['api_errors']:
                asyncio.run(self._send_alert('high_api_errors', f"Errores de API altos: {metrics['api_errors']} errores", 'error'))
            
            # Trades concurrentes
            if metrics['concurrent_trades'] > self.alert_thresholds['concurrent_trades']:
                asyncio.run(self._send_alert('high_concurrent_trades', f"Trades concurrentes altos: {metrics['concurrent_trades']} trades", 'warning'))
            
            # Cambio de balance
            if abs(metrics['balance'] - self._get_account_balance()) > self.alert_thresholds['balance_change']:
                asyncio.run(self._send_alert('balance_change', f"Cambio significativo en balance: {metrics['balance']}", 'warning'))
            
            # Volatilidad
            if metrics['volatility'] > self.alert_thresholds['volatility']:
                asyncio.run(self._send_alert('high_volatility', f"Volatilidad alta: {metrics['volatility']*100}%", 'warning'))
            
            # Spread
            if metrics['spread'] > self.alert_thresholds['spread']:
                asyncio.run(self._send_alert('high_spread', f"Spread alto: {metrics['spread']*100}%", 'warning'))
        except Exception as e:
            self.logger.error(f"Error verificando alertas de trading: {str(e)}")

    def _detect_anomalies(self, data: pd.DataFrame) -> List[str]:
        """Detecta anomalías en los datos"""
        try:
            # Preparar datos
            X = data.select_dtypes(include=[np.number]).values
            
            # Entrenar detector de anomalías
            self.anomaly_detector.fit(X)
            
            # Predecir anomalías
            predictions = self.anomaly_detector.predict(X)
            
            # Identificar anomalías
            anomalies = []
            for i, pred in enumerate(predictions):
                if pred == -1:  # Anomalía detectada
                    anomalies.append(f"Anomalía en {data.index[i]}: {data.iloc[i].to_dict()}")
            
            return anomalies
        except Exception as e:
            self.logger.error(f"Error detectando anomalías: {str(e)}")
            return []

    async def _send_alert(self, alert_type: str, message: str, severity: str = 'warning'):
        """Send alert asynchronously"""
        try:
            await self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=f"🚨 {severity.upper()}\n{message}"
            )
            
            # Log alert in database
            with self._get_db_connection() as conn:
                with conn:
                    conn.execute('''
                        INSERT INTO alerts (timestamp, alert_type, message, severity)
                        VALUES (?, ?, ?, ?)
                    ''', (datetime.now(), alert_type, message, severity))
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")

    def _get_system_metrics_history(self) -> pd.DataFrame:
        """Get system metrics history with proper error handling"""
        try:
            with self._get_db_connection() as conn:
                return pd.read_sql_query('''
                    SELECT * FROM system_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                ''', conn)
        except Exception as e:
            self.logger.error(f"Error getting system metrics history: {str(e)}")
            return pd.DataFrame()

    def _get_trading_metrics_history(self) -> pd.DataFrame:
        """Obtiene historial de métricas de trading"""
        try:
            with self._get_db_connection() as conn:
                return pd.read_sql_query('''
                    SELECT * FROM trading_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                ''', conn)
        except Exception as e:
            self.logger.error(f"Error obteniendo historial de métricas de trading: {str(e)}")
            return pd.DataFrame()

    def generate_report(self, save_path: str = None):
        """Generate monitoring report"""
        try:
            # Get data
            system_metrics = self._get_system_metrics_history()
            trading_metrics = self._get_trading_metrics_history()
            
            # Create figure
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
            
            # Plot resource usage
            axes[0, 0].plot(system_metrics['timestamp'], system_metrics['cpu_usage'], label='CPU')
            axes[0, 0].plot(system_metrics['timestamp'], system_metrics['memory_usage'], label='Memory')
            axes[0, 0].set_title('Resource Usage')
            axes[0, 0].legend()
            
            # Plot performance
            axes[0, 1].plot(trading_metrics['timestamp'], trading_metrics['profit_loss'])
            axes[0, 1].set_title('Performance')
            
            # Plot drawdown
            axes[1, 0].plot(trading_metrics['timestamp'], trading_metrics['drawdown'])
            axes[1, 0].set_title('Drawdown')
            
            # Plot trades
            axes[1, 1].plot(trading_metrics['timestamp'], trading_metrics['trade_count'])
            axes[1, 1].set_title('Trades per Hour')
            
            # Plot error rate
            axes[2, 0].plot(system_metrics['timestamp'], system_metrics['error_rate'])
            axes[2, 0].set_title('Error Rate')
            
            # Plot volatility
            axes[2, 1].plot(trading_metrics['timestamp'], trading_metrics['volatility'])
            axes[2, 1].set_title('Volatility')
            
            plt.tight_layout()
            
            # Save report
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")

    def close(self):
        """Cierra conexiones y recursos"""
        try:
            if self.db_path:
                os.remove(self.db_path)
        except Exception as e:
            self.logger.error(f"Error cerrando conexiones: {str(e)}") 