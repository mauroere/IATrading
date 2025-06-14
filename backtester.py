import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import os
from indicators import TechnicalIndicators
from risk_manager import RiskManager
from ml_model import MLModel

class Backtester:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators(config)
        self.risk_manager = RiskManager(config)
        self.ml_model = MLModel(config)
        self.results = []
        self.metrics = {}
        
    def run_backtest(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Ejecuta backtesting con los parámetros dados"""
        try:
            # Inicializar resultados
            self.results = []
            initial_balance = self.config['initial_balance']
            balance = initial_balance
            position = None
            trades = []
            
            # Calcular indicadores
            df = self.indicators.calculate_all_indicators(data)
            
            # Iterar sobre cada período
            for i in range(len(df)):
                current_data = df.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                
                # Obtener señales
                signals = self.indicators.get_trading_signals(current_data)
                
                # Si hay posición abierta, verificar stop loss y take profit
                if position:
                    # Calcular P&L
                    pnl = (current_price - position['entry_price']) * position['size']
                    if position['side'] == 'sell':
                        pnl = -pnl
                    
                    # Verificar stop loss
                    if pnl <= -position['stop_loss']:
                        balance += pnl
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_data.index[-1],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'size': position['size'],
                            'side': position['side'],
                            'pnl': pnl,
                            'exit_reason': 'stop_loss'
                        })
                        position = None
                    
                    # Verificar take profit
                    elif pnl >= position['take_profit']:
                        balance += pnl
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_data.index[-1],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'size': position['size'],
                            'side': position['side'],
                            'pnl': pnl,
                            'exit_reason': 'take_profit'
                        })
                        position = None
                
                # Si no hay posición y hay señal, abrir nueva posición
                elif signals['signal'] != 0:
                    # Calcular tamaño de posición
                    position_size = self.risk_manager.calculate_position_size(
                        balance,
                        current_price,
                        current_data['atr'].iloc[-1],
                        signals['signal_strength']
                    )
                    
                    # Calcular stop loss y take profit
                    stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
                        current_price,
                        current_data['atr'].iloc[-1],
                        signals['signal_strength']
                    )
                    
                    position = {
                        'entry_time': current_data.index[-1],
                        'entry_price': current_price,
                        'size': position_size,
                        'side': 'buy' if signals['signal'] > 0 else 'sell',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
            
            # Cerrar posición final si existe
            if position:
                pnl = (current_price - position['entry_price']) * position['size']
                if position['side'] == 'sell':
                    pnl = -pnl
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_data.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'size': position['size'],
                    'side': position['side'],
                    'pnl': pnl,
                    'exit_reason': 'end_of_backtest'
                })
            
            # Calcular métricas
            self.metrics = self._calculate_metrics(trades, initial_balance)
            
            return {
                'trades': trades,
                'metrics': self.metrics,
                'final_balance': balance,
                'return': (balance - initial_balance) / initial_balance * 100
            }
        except Exception as e:
            self.logger.error(f"Error en backtesting: {str(e)}")
            return {}
    
    def run_walk_forward_optimization(self, data: pd.DataFrame, 
                                    train_size: int = 1000,
                                    test_size: int = 200,
                                    step_size: int = 50) -> Dict:
        """Ejecuta optimización walk-forward"""
        try:
            results = []
            total_periods = len(data) - train_size - test_size
            
            for i in range(0, total_periods, step_size):
                # Dividir datos
                train_data = data.iloc[i:i+train_size]
                test_data = data.iloc[i+train_size:i+train_size+test_size]
                
                # Entrenar modelo
                self.ml_model.train(train_data)
                
                # Ejecutar backtest en período de prueba
                backtest_result = self.run_backtest(test_data, self.config)
                
                results.append({
                    'period': i,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'metrics': backtest_result['metrics']
                })
            
            # Calcular métricas agregadas
            aggregated_metrics = self._aggregate_walk_forward_metrics(results)
            
            return {
                'period_results': results,
                'aggregated_metrics': aggregated_metrics
            }
        except Exception as e:
            self.logger.error(f"Error en optimización walk-forward: {str(e)}")
            return {}
    
    def _calculate_metrics(self, trades: List[Dict], initial_balance: float) -> Dict:
        """Calcula métricas de rendimiento"""
        try:
            if not trades:
                return {}
            
            # Convertir trades a DataFrame
            df = pd.DataFrame(trades)
            
            # Métricas básicas
            total_trades = len(trades)
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] <= 0])
            
            # Métricas de rendimiento
            total_pnl = df['pnl'].sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Métricas de riesgo
            max_drawdown = self._calculate_max_drawdown(df['pnl'].cumsum())
            sharpe_ratio = self._calculate_sharpe_ratio(df['pnl'])
            
            # Métricas de tiempo
            avg_trade_duration = (df['exit_time'] - df['entry_time']).mean()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_trade_duration': avg_trade_duration,
                'return': (total_pnl / initial_balance) * 100
            }
        except Exception as e:
            self.logger.error(f"Error calculando métricas: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calcula el máximo drawdown"""
        try:
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns - rolling_max
            return abs(drawdowns.min())
        except Exception as e:
            self.logger.error(f"Error calculando max drawdown: {str(e)}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calcula el ratio de Sharpe"""
        try:
            if len(returns) < 2:
                return 0.0
            return np.sqrt(252) * returns.mean() / returns.std()
        except Exception as e:
            self.logger.error(f"Error calculando Sharpe ratio: {str(e)}")
            return 0.0
    
    def _aggregate_walk_forward_metrics(self, results: List[Dict]) -> Dict:
        """Agrega métricas de optimización walk-forward"""
        try:
            if not results:
                return {}
            
            # Extraer métricas de cada período
            metrics_df = pd.DataFrame([r['metrics'] for r in results])
            
            # Calcular estadísticas
            aggregated = {
                'mean_win_rate': metrics_df['win_rate'].mean(),
                'mean_profit_factor': metrics_df['profit_factor'].mean(),
                'mean_sharpe_ratio': metrics_df['sharpe_ratio'].mean(),
                'mean_return': metrics_df['return'].mean(),
                'std_return': metrics_df['return'].std(),
                'max_drawdown': metrics_df['max_drawdown'].max(),
                'total_trades': metrics_df['total_trades'].sum(),
                'consistency_score': self._calculate_consistency_score(metrics_df)
            }
            
            return aggregated
        except Exception as e:
            self.logger.error(f"Error agregando métricas: {str(e)}")
            return {}
    
    def _calculate_consistency_score(self, metrics_df: pd.DataFrame) -> float:
        """Calcula score de consistencia"""
        try:
            # Calcular consistencia en diferentes métricas
            win_rate_consistency = 1 - metrics_df['win_rate'].std()
            return_consistency = 1 - metrics_df['return'].std() / metrics_df['return'].mean()
            profit_factor_consistency = 1 - metrics_df['profit_factor'].std() / metrics_df['profit_factor'].mean()
            
            # Promedio ponderado
            return (
                win_rate_consistency * 0.4 +
                return_consistency * 0.4 +
                profit_factor_consistency * 0.2
            )
        except Exception as e:
            self.logger.error(f"Error calculando consistencia: {str(e)}")
            return 0.0
    
    def plot_results(self, save_path: str = None):
        """Genera gráficos de resultados"""
        try:
            if not self.results:
                return
            
            # Crear figura con subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
            
            # Gráfico de equity
            equity_curve = pd.Series([t['pnl'] for t in self.results]).cumsum()
            axes[0].plot(equity_curve.index, equity_curve.values)
            axes[0].set_title('Curva de Equity')
            axes[0].set_xlabel('Tiempo')
            axes[0].set_ylabel('Equity')
            
            # Gráfico de drawdown
            drawdown = self._calculate_drawdown_series(equity_curve)
            axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            axes[1].set_title('Drawdown')
            axes[1].set_xlabel('Tiempo')
            axes[1].set_ylabel('Drawdown %')
            
            # Gráfico de distribución de retornos
            returns = pd.Series([t['pnl'] for t in self.results])
            sns.histplot(returns, ax=axes[2], bins=50)
            axes[2].set_title('Distribución de Retornos')
            axes[2].set_xlabel('Retorno')
            axes[2].set_ylabel('Frecuencia')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Error generando gráficos: {str(e)}")
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calcula serie de drawdown"""
        try:
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max * 100
            return drawdown
        except Exception as e:
            self.logger.error(f"Error calculando serie de drawdown: {str(e)}")
            return pd.Series()
    
    def save_results(self, filepath: str):
        """Guarda resultados del backtest"""
        try:
            results = {
                'trades': self.results,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {str(e)}")
    
    def load_results(self, filepath: str):
        """Carga resultados del backtest"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    results = json.load(f)
                
                self.results = results['trades']
                self.metrics = results['metrics']
        except Exception as e:
            self.logger.error(f"Error cargando resultados: {str(e)}") 