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
import schedule
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

class TradingSystem:
    def __init__(self):
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log'),
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
        
        # Estado del sistema
        self.system_state = {
            'active_trades': {},
            'daily_stats': {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit_loss': 0.0
            },
            'last_update': datetime.now()
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
    
    def start(self):
        """Inicia el sistema de trading"""
        try:
            self.logger.info("Iniciando sistema de trading...")
            
            # Iniciar monitoreo
            self.monitor.start_monitoring()
            
            # Programar tareas
            self._schedule_tasks()
            
            # Iniciar hilos principales
            Thread(target=self._market_analysis_thread, daemon=True).start()
            Thread(target=self._trading_thread, daemon=True).start()
            Thread(target=self._optimization_thread, daemon=True).start()
            
            # Mantener el hilo principal vivo
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error iniciando sistema: {str(e)}")
            self.cleanup()
            sys.exit(1)
    
    def _schedule_tasks(self):
        """Programa tareas periódicas"""
        try:
            # Reset diario de estadísticas
            schedule.every().day.at("00:00").do(self._reset_daily_stats)
            
            # Optimización semanal
            schedule.every().monday.at("02:00").do(self._run_weekly_optimization)
            
            # Backtesting mensual
            schedule.every().day.at("03:00").do(self._run_monthly_backtest)
            
            # Limpieza de datos
            schedule.every().day.at("04:00").do(self._cleanup_old_data)
            
            # Iniciar scheduler en un hilo separado
            Thread(target=self._run_scheduler, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Error programando tareas: {str(e)}")
    
    def _run_scheduler(self):
        """Ejecuta el scheduler"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def _market_analysis_thread(self):
        """Hilo de análisis de mercado"""
        while self.running:
            try:
                for symbol in self.config['trading_pairs']:
                    # Analizar condiciones de mercado
                    market_conditions = self.market_analysis.analyze_market_conditions(symbol)
                    
                    # Actualizar estado
                    self.system_state['market_conditions'] = market_conditions
                    
                    # Enviar a cola de mensajes
                    self.message_queue.put({
                        'type': 'market_analysis',
                        'symbol': symbol,
                        'data': market_conditions
                    })
                
                time.sleep(60)  # Actualizar cada minuto
            except Exception as e:
                self.logger.error(f"Error en análisis de mercado: {str(e)}")
                time.sleep(60)
    
    def _trading_thread(self):
        """Hilo de trading"""
        while self.running:
            try:
                # Procesar mensajes de la cola
                while not self.message_queue.empty():
                    message = self.message_queue.get()
                    
                    if message['type'] == 'market_analysis':
                        self._process_trading_signals(
                            message['symbol'],
                            message['data']
                        )
                
                # Verificar trades activos
                self._check_active_trades()
                
                time.sleep(1)  # Verificar cada segundo
            except Exception as e:
                self.logger.error(f"Error en hilo de trading: {str(e)}")
                time.sleep(1)
    
    def _optimization_thread(self):
        """Hilo de optimización"""
        while self.running:
            try:
                # Verificar si es necesario optimizar
                if self._should_optimize():
                    self._run_optimization()
                
                time.sleep(3600)  # Verificar cada hora
            except Exception as e:
                self.logger.error(f"Error en hilo de optimización: {str(e)}")
                time.sleep(3600)
    
    def _process_trading_signals(self, symbol: str, market_conditions: Dict):
        """Procesa señales de trading"""
        try:
            # Verificar si ya hay un trade activo para este símbolo
            if symbol in self.system_state['active_trades']:
                return
            
            # Verificar límites diarios
            if self.system_state['daily_stats']['trades'] >= self.config['max_daily_trades']:
                return
            
            # Obtener datos OHLCV
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calcular indicadores
            df = self.indicators.calculate_all_indicators(df)
            
            # Obtener señales
            signals = self.indicators.get_trading_signals(df)
            
            # Verificar condiciones de mercado
            if market_conditions['market_condition'] in ['strong_bullish', 'bullish']:
                if signals['signal'] > 0:  # Señal de compra
                    self._execute_trade(symbol, 'buy', df)
            elif market_conditions['market_condition'] in ['strong_bearish', 'bearish']:
                if signals['signal'] < 0:  # Señal de venta
                    self._execute_trade(symbol, 'sell', df)
            
        except Exception as e:
            self.logger.error(f"Error procesando señales: {str(e)}")
    
    def _execute_trade(self, symbol: str, side: str, df: pd.DataFrame):
        """Ejecuta una operación de trading"""
        try:
            # Obtener precio actual
            current_price = df['close'].iloc[-1]
            
            # Calcular tamaño de posición
            balance = self._get_account_balance()
            position_size = self.risk_manager.calculate_position_size(
                balance,
                current_price,
                df['atr'].iloc[-1],
                abs(df['signal_strength'].iloc[-1])
            )
            
            # Calcular stop loss y take profit
            stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
                current_price,
                df['atr'].iloc[-1],
                abs(df['signal_strength'].iloc[-1])
            )
            
            # Verificar trade
            if not self.risk_manager.validate_trade(
                current_price,
                position_size,
                abs(df['signal_strength'].iloc[-1])
            ):
                return
            
            # Ejecutar orden
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=position_size,
                price=current_price
            )
            
            # Actualizar estado
            self.system_state['active_trades'][symbol] = {
                'order_id': order['id'],
                'side': side,
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now()
            }
            
            # Actualizar estadísticas
            self.system_state['daily_stats']['trades'] += 1
            
            # Enviar notificación
            self._send_trade_notification(symbol, side, current_price, position_size)
            
        except Exception as e:
            self.logger.error(f"Error ejecutando trade: {str(e)}")
    
    def _check_active_trades(self):
        """Verifica trades activos"""
        try:
            for symbol, trade in list(self.system_state['active_trades'].items()):
                # Obtener precio actual
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Calcular P&L
                pnl = (current_price - trade['entry_price']) * trade['size']
                if trade['side'] == 'sell':
                    pnl = -pnl
                
                # Verificar stop loss
                if pnl <= -trade['stop_loss']:
                    self._close_trade(symbol, current_price, 'stop_loss')
                
                # Verificar take profit
                elif pnl >= trade['take_profit']:
                    self._close_trade(symbol, current_price, 'take_profit')
                
        except Exception as e:
            self.logger.error(f"Error verificando trades activos: {str(e)}")
    
    def _close_trade(self, symbol: str, current_price: float, reason: str):
        """Cierra una operación de trading"""
        try:
            trade = self.system_state['active_trades'][symbol]
            
            # Ejecutar orden de cierre
            self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side='sell' if trade['side'] == 'buy' else 'buy',
                amount=trade['size'],
                price=current_price
            )
            
            # Calcular P&L
            pnl = (current_price - trade['entry_price']) * trade['size']
            if trade['side'] == 'sell':
                pnl = -pnl
            
            # Actualizar estadísticas
            self.system_state['daily_stats']['profit_loss'] += pnl
            if pnl > 0:
                self.system_state['daily_stats']['wins'] += 1
            else:
                self.system_state['daily_stats']['losses'] += 1
            
            # Enviar notificación
            self._send_trade_close_notification(symbol, current_price, pnl, reason)
            
            # Eliminar trade activo
            del self.system_state['active_trades'][symbol]
            
        except Exception as e:
            self.logger.error(f"Error cerrando trade: {str(e)}")
    
    def _get_account_balance(self) -> float:
        """Obtiene balance de la cuenta"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            self.logger.error(f"Error obteniendo balance: {str(e)}")
            return 0.0
    
    def _should_optimize(self) -> bool:
        """Determina si es necesario optimizar"""
        try:
            # Verificar rendimiento reciente
            recent_trades = self._get_recent_trades()
            if not recent_trades:
                return True
            
            # Calcular métricas
            win_rate = len([t for t in recent_trades if t['pnl'] > 0]) / len(recent_trades)
            profit_factor = abs(
                sum(t['pnl'] for t in recent_trades if t['pnl'] > 0) /
                sum(t['pnl'] for t in recent_trades if t['pnl'] < 0)
            ) if sum(t['pnl'] for t in recent_trades if t['pnl'] < 0) != 0 else float('inf')
            
            # Optimizar si el rendimiento es bajo
            return win_rate < 0.5 or profit_factor < 1.5
            
        except Exception as e:
            self.logger.error(f"Error verificando necesidad de optimización: {str(e)}")
            return False
    
    def _run_optimization(self):
        """Ejecuta optimización de parámetros"""
        try:
            self.logger.info("Iniciando optimización...")
            
            # Obtener datos históricos
            data = self._get_historical_data()
            
            # Función de evaluación
            def evaluation_function(params):
                # Actualizar configuración
                self.config.update(params)
                
                # Ejecutar backtest
                results = self.backtester.run_backtest(data, self.config)
                
                # Calcular score
                if not results or 'metrics' not in results:
                    return float('-inf')
                
                metrics = results['metrics']
                return (
                    metrics['win_rate'] * 0.4 +
                    metrics['profit_factor'] * 0.3 +
                    metrics['sharpe_ratio'] * 0.3
                )
            
            # Ejecutar optimización
            best_params = self.optimizer.optimize(
                evaluation_function,
                n_iterations=50
            )
            
            # Actualizar configuración
            if best_params:
                self.config.update(best_params)
                self._save_config()
                
                # Notificar
                self._send_optimization_notification(best_params)
            
            self.logger.info("Optimización completada")
            
        except Exception as e:
            self.logger.error(f"Error en optimización: {str(e)}")
    
    def _get_historical_data(self) -> pd.DataFrame:
        """Obtiene datos históricos"""
        try:
            data = {}
            for symbol in self.config['trading_pairs']:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=1000)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                data[symbol] = df
            return data
        except Exception as e:
            self.logger.error(f"Error obteniendo datos históricos: {str(e)}")
            return {}
    
    def _get_recent_trades(self) -> List[Dict]:
        """Obtiene trades recientes"""
        try:
            cursor = self.monitor.db_connection.cursor()
            cursor.execute('''
                SELECT * FROM trading_metrics 
                WHERE timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
            ''')
            return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error obteniendo trades recientes: {str(e)}")
            return []
    
    def _reset_daily_stats(self):
        """Resetea estadísticas diarias"""
        try:
            self.system_state['daily_stats'] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit_loss': 0.0
            }
            self.logger.info("Estadísticas diarias reseteadas")
        except Exception as e:
            self.logger.error(f"Error reseteando estadísticas: {str(e)}")
    
    def _run_weekly_optimization(self):
        """Ejecuta optimización semanal"""
        try:
            self._run_optimization()
        except Exception as e:
            self.logger.error(f"Error en optimización semanal: {str(e)}")
    
    def _run_monthly_backtest(self):
        """Ejecuta backtesting mensual"""
        try:
            data = self._get_historical_data()
            results = self.backtester.run_backtest(data, self.config)
            
            # Generar reporte
            self.backtester.plot_results('monthly_backtest.png')
            
            # Enviar notificación
            self._send_backtest_notification(results)
            
        except Exception as e:
            self.logger.error(f"Error en backtesting mensual: {str(e)}")
    
    def _cleanup_old_data(self):
        """Limpia datos antiguos"""
        try:
            # Limpiar datos de monitoreo
            self.monitor._cleanup_old_data()
            
            # Limpiar datos de backtesting
            if os.path.exists('backtest_results'):
                for file in os.listdir('backtest_results'):
                    if file.endswith('.json'):
                        file_path = os.path.join('backtest_results', file)
                        if os.path.getmtime(file_path) < time.time() - 30 * 24 * 3600:  # 30 días
                            os.remove(file_path)
            
        except Exception as e:
            self.logger.error(f"Error limpiando datos antiguos: {str(e)}")
    
    def _save_config(self):
        """Guarda configuración actualizada"""
        try:
            with open('config.yaml', 'w') as f:
                yaml.dump(self.config, f)
        except Exception as e:
            self.logger.error(f"Error guardando configuración: {str(e)}")
    
    def _send_trade_notification(self, symbol: str, side: str, price: float, size: float):
        """Envía notificación de trade"""
        try:
            message = (
                f"🔄 Nueva operación\n"
                f"Símbolo: {symbol}\n"
                f"Lado: {side.upper()}\n"
                f"Precio: {price:.8f}\n"
                f"Tamaño: {size:.8f}\n"
                f"Total: {price * size:.8f} USDT"
            )
            
            self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=message
            )
        except Exception as e:
            self.logger.error(f"Error enviando notificación de trade: {str(e)}")
    
    def _send_trade_close_notification(self, symbol: str, price: float, pnl: float, reason: str):
        """Envía notificación de cierre de trade"""
        try:
            message = (
                f"✅ Operación cerrada\n"
                f"Símbolo: {symbol}\n"
                f"Precio: {price:.8f}\n"
                f"P&L: {pnl:.8f} USDT\n"
                f"Razón: {reason}"
            )
            
            self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=message
            )
        except Exception as e:
            self.logger.error(f"Error enviando notificación de cierre: {str(e)}")
    
    def _send_optimization_notification(self, params: Dict):
        """Envía notificación de optimización"""
        try:
            message = (
                f"🔄 Optimización completada\n"
                f"Nuevos parámetros:\n"
                + "\n".join(f"{k}: {v}" for k, v in params.items())
            )
            
            self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=message
            )
        except Exception as e:
            self.logger.error(f"Error enviando notificación de optimización: {str(e)}")
    
    def _send_backtest_notification(self, results: Dict):
        """Envía notificación de backtesting"""
        try:
            metrics = results['metrics']
            message = (
                f"📊 Reporte de Backtesting\n"
                f"Win Rate: {metrics['win_rate']*100:.2f}%\n"
                f"Profit Factor: {metrics['profit_factor']:.2f}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Retorno: {metrics['return']:.2f}%\n"
                f"Drawdown Máximo: {metrics['max_drawdown']*100:.2f}%"
            )
            
            self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=message
            )
            
            # Enviar gráfico
            with open('monthly_backtest.png', 'rb') as f:
                self.telegram_bot.send_photo(
                    chat_id=self.config['telegram_chat_id'],
                    photo=f
                )
        except Exception as e:
            self.logger.error(f"Error enviando notificación de backtesting: {str(e)}")
    
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
        print("\nDeteniendo sistema...")
        system.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Iniciar sistema
    system = TradingSystem()
    system.start()

if __name__ == "__main__":
    main() 