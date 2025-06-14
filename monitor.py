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
from threading import Thread
import time
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

class SystemMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.telegram_bot = telegram.Bot(token=config['telegram_bot_token'])
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.metrics_history = []
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.db_connection = self._initialize_database()
        
    def _initialize_alert_thresholds(self) -> Dict:
        """Inicializa umbrales de alerta"""
        return {
            'cpu_usage': 80.0,  # Porcentaje
            'memory_usage': 80.0,  # Porcentaje
            'disk_usage': 80.0,  # Porcentaje
            'network_latency': 1000,  # ms
            'error_rate': 0.05,  # Porcentaje
            'drawdown': 0.15,  # Porcentaje
            'profit_loss': -0.1,  # Porcentaje
            'trade_frequency': 0.5,  # Trades por hora
            'api_errors': 5,  # Número de errores
            'memory_leak': 100,  # MB por hora
            'response_time': 2000,  # ms
            'concurrent_trades': 3,  # Número de trades
            'balance_change': 0.1,  # Porcentaje
            'volatility': 0.05,  # Porcentaje
            'spread': 0.002  # Porcentaje
        }
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Inicializa base de datos para métricas"""
        try:
            conn = sqlite3.connect('monitoring.db')
            cursor = conn.cursor()
            
            # Crear tablas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp DATETIME PRIMARY KEY,
                    cpu_usage FLOAT,
                    memory_usage FLOAT,
                    disk_usage FLOAT,
                    network_latency FLOAT,
                    error_rate FLOAT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    timestamp DATETIME PRIMARY KEY,
                    profit_loss FLOAT,
                    drawdown FLOAT,
                    trade_count INTEGER,
                    api_errors INTEGER,
                    concurrent_trades INTEGER,
                    balance FLOAT,
                    volatility FLOAT,
                    spread FLOAT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    timestamp DATETIME PRIMARY KEY,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    resolved BOOLEAN
                )
            ''')
            
            conn.commit()
            return conn
        except Exception as e:
            self.logger.error(f"Error inicializando base de datos: {str(e)}")
            return None
    
    def start_monitoring(self):
        """Inicia el monitoreo del sistema"""
        try:
            # Iniciar hilos de monitoreo
            Thread(target=self._monitor_system_metrics, daemon=True).start()
            Thread(target=self._monitor_trading_metrics, daemon=True).start()
            Thread(target=self._monitor_anomalies, daemon=True).start()
            Thread(target=self._cleanup_old_data, daemon=True).start()
            
            self.logger.info("Sistema de monitoreo iniciado")
        except Exception as e:
            self.logger.error(f"Error iniciando monitoreo: {str(e)}")
    
    def _monitor_system_metrics(self):
        """Monitorea métricas del sistema"""
        while True:
            try:
                # Recolectar métricas
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'network_latency': self._measure_network_latency(),
                    'error_rate': self._calculate_error_rate()
                }
                
                # Guardar métricas
                self._save_system_metrics(metrics)
                
                # Verificar alertas
                self._check_system_alerts(metrics)
                
                # Actualizar historial
                self.metrics_history.append(metrics)
                
                time.sleep(60)  # Actualizar cada minuto
            except Exception as e:
                self.logger.error(f"Error monitoreando métricas del sistema: {str(e)}")
                time.sleep(60)
    
    def _monitor_trading_metrics(self):
        """Monitorea métricas de trading"""
        while True:
            try:
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
        while True:
            try:
                # Obtener datos históricos
                system_metrics = self._get_system_metrics_history()
                trading_metrics = self._get_trading_metrics_history()
                
                if len(system_metrics) > 0:
                    # Detectar anomalías en métricas del sistema
                    system_anomalies = self._detect_anomalies(system_metrics)
                    for anomaly in system_anomalies:
                        self._send_alert(
                            'system_anomaly',
                            f"Anomalía detectada en métricas del sistema: {anomaly}",
                            'high'
                        )
                
                if len(trading_metrics) > 0:
                    # Detectar anomalías en métricas de trading
                    trading_anomalies = self._detect_anomalies(trading_metrics)
                    for anomaly in trading_anomalies:
                        self._send_alert(
                            'trading_anomaly',
                            f"Anomalía detectada en métricas de trading: {anomaly}",
                            'high'
                        )
                
                time.sleep(300)  # Verificar cada 5 minutos
            except Exception as e:
                self.logger.error(f"Error monitoreando anomalías: {str(e)}")
                time.sleep(300)
    
    def _cleanup_old_data(self):
        """Limpia datos antiguos de la base de datos"""
        while True:
            try:
                cursor = self.db_connection.cursor()
                
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
                
                self.db_connection.commit()
                
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
        """Calcula tasa de errores"""
        try:
            cursor = self.db_connection.cursor()
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
            self.logger.error(f"Error calculando tasa de errores: {str(e)}")
            return 0
    
    def _calculate_profit_loss(self) -> float:
        """Calcula ganancia/pérdida"""
        try:
            cursor = self.db_connection.cursor()
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
            cursor = self.db_connection.cursor()
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
            cursor = self.db_connection.cursor()
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
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM alerts 
                WHERE timestamp > datetime('now', '-1 hour')
                AND alert_type = 'api_error'
            ''')
            return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Error obteniendo errores de API: {str(e)}")
            return 0
    
    def _get_concurrent_trades(self) -> int:
        """Obtiene número de trades concurrentes"""
        try:
            cursor = self.db_connection.cursor()
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
            cursor = self.db_connection.cursor()
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
            cursor = self.db_connection.cursor()
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
            cursor = self.db_connection.cursor()
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
        """Guarda métricas del sistema"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
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
            self.db_connection.commit()
        except Exception as e:
            self.logger.error(f"Error guardando métricas del sistema: {str(e)}")
    
    def _save_trading_metrics(self, metrics: Dict):
        """Guarda métricas de trading"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
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
            self.db_connection.commit()
        except Exception as e:
            self.logger.error(f"Error guardando métricas de trading: {str(e)}")
    
    def _check_system_alerts(self, metrics: Dict):
        """Verifica alertas del sistema"""
        try:
            # CPU
            if metrics['cpu_usage'] > self.alert_thresholds['cpu_usage']:
                self._send_alert(
                    'high_cpu',
                    f"Uso de CPU alto: {metrics['cpu_usage']}%",
                    'warning'
                )
            
            # Memoria
            if metrics['memory_usage'] > self.alert_thresholds['memory_usage']:
                self._send_alert(
                    'high_memory',
                    f"Uso de memoria alto: {metrics['memory_usage']}%",
                    'warning'
                )
            
            # Disco
            if metrics['disk_usage'] > self.alert_thresholds['disk_usage']:
                self._send_alert(
                    'high_disk',
                    f"Uso de disco alto: {metrics['disk_usage']}%",
                    'warning'
                )
            
            # Latencia
            if metrics['network_latency'] > self.alert_thresholds['network_latency']:
                self._send_alert(
                    'high_latency',
                    f"Latencia de red alta: {metrics['network_latency']}ms",
                    'warning'
                )
            
            # Tasa de errores
            if metrics['error_rate'] > self.alert_thresholds['error_rate']:
                self._send_alert(
                    'high_error_rate',
                    f"Tasa de errores alta: {metrics['error_rate']*100}%",
                    'error'
                )
        except Exception as e:
            self.logger.error(f"Error verificando alertas del sistema: {str(e)}")
    
    def _check_trading_alerts(self, metrics: Dict):
        """Verifica alertas de trading"""
        try:
            # Drawdown
            if metrics['drawdown'] > self.alert_thresholds['drawdown']:
                self._send_alert(
                    'high_drawdown',
                    f"Drawdown alto: {metrics['drawdown']*100}%",
                    'error'
                )
            
            # Pérdida
            if metrics['profit_loss'] < self.alert_thresholds['profit_loss']:
                self._send_alert(
                    'high_loss',
                    f"Pérdida alta: {metrics['profit_loss']*100}%",
                    'error'
                )
            
            # Frecuencia de trades
            if metrics['trade_count'] > self.alert_thresholds['trade_frequency']:
                self._send_alert(
                    'high_trade_frequency',
                    f"Frecuencia de trades alta: {metrics['trade_count']} trades/hora",
                    'warning'
                )
            
            # Errores de API
            if metrics['api_errors'] > self.alert_thresholds['api_errors']:
                self._send_alert(
                    'high_api_errors',
                    f"Errores de API altos: {metrics['api_errors']} errores",
                    'error'
                )
            
            # Trades concurrentes
            if metrics['concurrent_trades'] > self.alert_thresholds['concurrent_trades']:
                self._send_alert(
                    'high_concurrent_trades',
                    f"Trades concurrentes altos: {metrics['concurrent_trades']} trades",
                    'warning'
                )
            
            # Cambio de balance
            if abs(metrics['balance'] - self._get_account_balance()) > self.alert_thresholds['balance_change']:
                self._send_alert(
                    'balance_change',
                    f"Cambio significativo en balance: {metrics['balance']}",
                    'warning'
                )
            
            # Volatilidad
            if metrics['volatility'] > self.alert_thresholds['volatility']:
                self._send_alert(
                    'high_volatility',
                    f"Volatilidad alta: {metrics['volatility']*100}%",
                    'warning'
                )
            
            # Spread
            if metrics['spread'] > self.alert_thresholds['spread']:
                self._send_alert(
                    'high_spread',
                    f"Spread alto: {metrics['spread']*100}%",
                    'warning'
                )
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
    
    def _send_alert(self, alert_type: str, message: str, severity: str):
        """Envía alerta por Telegram"""
        try:
            # Guardar alerta en base de datos
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, message, severity, resolved)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now(), alert_type, message, severity, False))
            self.db_connection.commit()
            
            # Enviar mensaje por Telegram
            self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=f"🚨 {severity.upper()}\n{message}"
            )
        except Exception as e:
            self.logger.error(f"Error enviando alerta: {str(e)}")
    
    def _get_system_metrics_history(self) -> pd.DataFrame:
        """Obtiene historial de métricas del sistema"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT * FROM system_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''')
            columns = [description[0] for description in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            self.logger.error(f"Error obteniendo historial de métricas del sistema: {str(e)}")
            return pd.DataFrame()
    
    def _get_trading_metrics_history(self) -> pd.DataFrame:
        """Obtiene historial de métricas de trading"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                SELECT * FROM trading_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''')
            columns = [description[0] for description in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            self.logger.error(f"Error obteniendo historial de métricas de trading: {str(e)}")
            return pd.DataFrame()
    
    def generate_report(self, save_path: str = None):
        """Genera reporte de monitoreo"""
        try:
            # Obtener datos
            system_metrics = self._get_system_metrics_history()
            trading_metrics = self._get_trading_metrics_history()
            
            # Crear figura
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
            
            # Gráfico de uso de recursos
            axes[0, 0].plot(system_metrics['timestamp'], system_metrics['cpu_usage'], label='CPU')
            axes[0, 0].plot(system_metrics['timestamp'], system_metrics['memory_usage'], label='Memoria')
            axes[0, 0].set_title('Uso de Recursos')
            axes[0, 0].legend()
            
            # Gráfico de rendimiento
            axes[0, 1].plot(trading_metrics['timestamp'], trading_metrics['profit_loss'])
            axes[0, 1].set_title('Rendimiento')
            
            # Gráfico de drawdown
            axes[1, 0].plot(trading_metrics['timestamp'], trading_metrics['drawdown'])
            axes[1, 0].set_title('Drawdown')
            
            # Gráfico de trades
            axes[1, 1].plot(trading_metrics['timestamp'], trading_metrics['trade_count'])
            axes[1, 1].set_title('Trades por Hora')
            
            # Gráfico de errores
            axes[2, 0].plot(system_metrics['timestamp'], system_metrics['error_rate'])
            axes[2, 0].set_title('Tasa de Errores')
            
            # Gráfico de volatilidad
            axes[2, 1].plot(trading_metrics['timestamp'], trading_metrics['volatility'])
            axes[2, 1].set_title('Volatilidad')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Error generando reporte: {str(e)}")
    
    def close(self):
        """Cierra conexiones y recursos"""
        try:
            if self.db_connection:
                self.db_connection.close()
        except Exception as e:
            self.logger.error(f"Error cerrando conexiones: {str(e)}") 