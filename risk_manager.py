import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.trade_history = []
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0.0
        }
        self.reset_daily_stats()

    def reset_daily_stats(self):
        """Reinicia las estadísticas diarias"""
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0.0
        }

    def calculate_position_size(self, account_balance: float, current_price: float, 
                              atr: float, signal_strength: float) -> float:
        """Calcula el tamaño de la posición usando múltiples métodos"""
        try:
            method = self.config['position_sizing_method']
            
            if method == 'fixed':
                return self._fixed_position_sizing(account_balance)
            elif method == 'kelly':
                return self._kelly_criterion(account_balance)
            elif method == 'optimal_f':
                return self._optimal_f_position_sizing(account_balance)
            else:
                return self._adaptive_position_sizing(account_balance, atr, signal_strength)
        except Exception as e:
            self.logger.error(f"Error calculando tamaño de posición: {str(e)}")
            return 0.0

    def _fixed_position_sizing(self, account_balance: float) -> float:
        """Tamaño de posición fijo basado en porcentaje de la cuenta"""
        return account_balance * self.config['position_size']

    def _kelly_criterion(self, account_balance: float) -> float:
        """Cálculo del criterio de Kelly"""
        win_rate = self.daily_stats['wins'] / max(1, self.daily_stats['trades'])
        avg_win = self.daily_stats['profit_loss'] / max(1, self.daily_stats['wins'])
        avg_loss = abs(self.daily_stats['profit_loss'] / max(1, self.daily_stats['losses']))
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = max(0, min(kelly, 0.5))  # Limitar al 50% máximo
        
        return account_balance * kelly * self.config['kelly_fraction']

    def _optimal_f_position_sizing(self, account_balance: float) -> float:
        """Cálculo de posición usando Optimal F"""
        if not self.trade_history:
            return self._fixed_position_sizing(account_balance)
        
        returns = [trade['profit_loss'] for trade in self.trade_history]
        f_values = np.linspace(0.01, 1.0, 100)
        optimal_f = 0.01
        
        max_geometric_mean = 0
        for f in f_values:
            geometric_mean = np.prod([1 + f * r for r in returns]) ** (1/len(returns))
            if geometric_mean > max_geometric_mean:
                max_geometric_mean = geometric_mean
                optimal_f = f
        
        return account_balance * optimal_f

    def _adaptive_position_sizing(self, account_balance: float, atr: float, 
                                signal_strength: float) -> float:
        """Tamaño de posición adaptativo basado en volatilidad y fuerza de señal"""
        base_size = self._fixed_position_sizing(account_balance)
        
        # Ajustar por volatilidad
        volatility_factor = 1 / (1 + atr/100)  # Reducir tamaño en alta volatilidad
        
        # Ajustar por fuerza de señal
        signal_factor = 0.5 + abs(signal_strength)  # Aumentar tamaño con señales fuertes
        
        # Ajustar por drawdown
        drawdown_factor = 1 - (self.daily_stats['losses'] / max(1, self.daily_stats['trades']))
        
        return base_size * volatility_factor * signal_factor * drawdown_factor

    def calculate_stop_loss_take_profit(self, entry_price: float, atr: float, 
                                      signal_strength: float) -> Tuple[float, float]:
        """Calcula niveles de stop loss y take profit dinámicos"""
        try:
            # Calcular stop loss basado en ATR
            atr_multiplier = 2.0 - abs(signal_strength)  # Ajustar por fuerza de señal
            stop_loss_distance = atr * atr_multiplier
            
            # Calcular take profit basado en ratio riesgo/beneficio
            risk_reward_ratio = self.config['min_risk_reward_ratio']
            take_profit_distance = stop_loss_distance * risk_reward_ratio
            
            # Ajustar por volatilidad del mercado
            volatility_factor = 1 + (atr / entry_price)
            stop_loss_distance *= volatility_factor
            take_profit_distance *= volatility_factor
            
            # Calcular niveles finales
            stop_loss = entry_price - stop_loss_distance
            take_profit = entry_price + take_profit_distance
            
            return stop_loss, take_profit
        except Exception as e:
            self.logger.error(f"Error calculando SL/TP: {str(e)}")
            return entry_price * 0.99, entry_price * 1.01

    def validate_trade(self, signal_strength: float, current_price: float, 
                      atr: float) -> bool:
        """Valida si una operación cumple con los criterios de riesgo"""
        try:
            # Verificar límite diario de operaciones
            if self.daily_stats['trades'] >= self.config['max_daily_trades']:
                self.logger.warning("Límite diario de operaciones alcanzado")
                return False
            
            # Verificar drawdown máximo
            if self.daily_stats['profit_loss'] < -self.config['max_daily_loss']:
                self.logger.warning("Drawdown máximo diario alcanzado")
                return False
            
            # Verificar volatilidad
            if atr > self.config['max_atr']:
                self.logger.warning("Volatilidad demasiado alta")
                return False
            
            # Verificar fuerza de señal
            if abs(signal_strength) < self.config['min_signal_strength']:
                self.logger.warning("Señal demasiado débil")
                return False
            
            # Verificar operaciones concurrentes
            active_trades = len([t for t in self.trade_history if t['status'] == 'active'])
            if active_trades >= self.config['max_concurrent_trades']:
                self.logger.warning("Máximo de operaciones concurrentes alcanzado")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validando operación: {str(e)}")
            return False

    def update_trade_history(self, trade_data: Dict):
        """Actualiza el historial de operaciones"""
        try:
            self.trade_history.append(trade_data)
            
            # Actualizar estadísticas diarias
            if trade_data['status'] == 'closed':
                self.daily_stats['trades'] += 1
                self.daily_stats['profit_loss'] += trade_data['profit_loss']
                
                if trade_data['profit_loss'] > 0:
                    self.daily_stats['wins'] += 1
                else:
                    self.daily_stats['losses'] += 1
            
            # Limpiar historial antiguo
            self._clean_old_trades()
        except Exception as e:
            self.logger.error(f"Error actualizando historial: {str(e)}")

    def _clean_old_trades(self):
        """Limpia operaciones antiguas del historial"""
        cutoff_date = datetime.now() - timedelta(days=30)
        self.trade_history = [
            trade for trade in self.trade_history 
            if trade['timestamp'] > cutoff_date
        ]

    def get_risk_metrics(self) -> Dict:
        """Obtiene métricas de riesgo actuales"""
        try:
            if not self.trade_history:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'avg_trade': 0.0
                }
            
            # Calcular métricas
            wins = len([t for t in self.trade_history if t['profit_loss'] > 0])
            losses = len([t for t in self.trade_history if t['profit_loss'] < 0])
            total_trades = len(self.trade_history)
            
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            total_profit = sum([t['profit_loss'] for t in self.trade_history if t['profit_loss'] > 0])
            total_loss = abs(sum([t['profit_loss'] for t in self.trade_history if t['profit_loss'] < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            returns = [t['profit_loss'] for t in self.trade_history]
            sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
            
            cumulative_returns = np.cumsum(returns)
            max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
            
            avg_trade = np.mean(returns)
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade': avg_trade
            }
        except Exception as e:
            self.logger.error(f"Error calculando métricas de riesgo: {str(e)}")
            return {} 