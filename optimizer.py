import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class BayesianOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        self.param_bounds = self._initialize_param_bounds()
        self.history = []
        self.best_params = None
        self.best_score = float('-inf')
        
    def _initialize_param_bounds(self) -> Dict:
        """Inicializa los límites de los parámetros a optimizar"""
        return {
            'rsi_period': (10, 30),
            'rsi_overbought': (65, 85),
            'rsi_oversold': (15, 35),
            'macd_fast': (8, 20),
            'macd_slow': (20, 40),
            'macd_signal': (5, 15),
            'bb_period': (15, 30),
            'bb_std': (1.5, 3.0),
            'atr_period': (10, 30),
            'supertrend_period': (5, 15),
            'supertrend_multiplier': (1.5, 4.0),
            'ichimoku_tenkan': (5, 15),
            'ichimoku_kijun': (15, 30),
            'ichimoku_senkou_b': (30, 60),
            'volume_ma_period': (10, 30),
            'stop_loss_atr': (1.0, 4.0),
            'take_profit_atr': (1.5, 6.0),
            'max_position_size': (0.01, 0.1),
            'max_daily_trades': (5, 20),
            'max_drawdown': (0.05, 0.2)
        }
    
    def optimize(self, evaluation_function, n_iterations: int = 50) -> Dict:
        """Optimiza los parámetros usando Bayesian Optimization"""
        try:
            for i in range(n_iterations):
                # Obtener siguiente punto a evaluar
                next_params = self._get_next_point()
                
                # Evaluar el punto
                score = evaluation_function(next_params)
                
                # Actualizar historial
                self.history.append({
                    'params': next_params,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Actualizar mejor resultado
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = next_params
                
                # Actualizar modelo GP
                X = np.array([list(p.values()) for p in self._get_param_values()])
                y = np.array([h['score'] for h in self.history])
                self.gp.fit(X, y)
                
                # Guardar progreso
                self._save_progress()
                
                self.logger.info(f"Iteración {i+1}/{n_iterations} - Score: {score:.4f}")
            
            return self.best_params
        except Exception as e:
            self.logger.error(f"Error en optimización: {str(e)}")
            return None
    
    def _get_next_point(self) -> Dict:
        """Obtiene el siguiente punto a evaluar usando UCB (Upper Confidence Bound)"""
        try:
            # Generar puntos aleatorios
            n_samples = 1000
            param_samples = self._generate_random_samples(n_samples)
            
            # Calcular predicción y varianza
            X = np.array([list(p.values()) for p in param_samples])
            y_pred, y_std = self.gp.predict(X, return_std=True)
            
            # Calcular UCB
            kappa = 2.0  # Factor de exploración
            ucb = y_pred + kappa * y_std
            
            # Seleccionar mejor punto
            best_idx = np.argmax(ucb)
            return param_samples[best_idx]
        except Exception as e:
            self.logger.error(f"Error obteniendo siguiente punto: {str(e)}")
            return self._generate_random_samples(1)[0]
    
    def _generate_random_samples(self, n_samples: int) -> List[Dict]:
        """Genera muestras aleatorias de parámetros"""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in self.param_bounds.items():
                if isinstance(min_val, int):
                    sample[param] = np.random.randint(min_val, max_val + 1)
                else:
                    sample[param] = np.random.uniform(min_val, max_val)
            samples.append(sample)
        return samples
    
    def _get_param_values(self) -> List[Dict]:
        """Obtiene valores de parámetros del historial"""
        return [h['params'] for h in self.history]
    
    def _save_progress(self):
        """Guarda el progreso de la optimización"""
        try:
            progress = {
                'history': self.history,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'param_bounds': self.param_bounds
            }
            
            with open('optimization_progress.json', 'w') as f:
                json.dump(progress, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error guardando progreso: {str(e)}")
    
    def load_progress(self):
        """Carga el progreso de la optimización"""
        try:
            if os.path.exists('optimization_progress.json'):
                with open('optimization_progress.json', 'r') as f:
                    progress = json.load(f)
                
                self.history = progress['history']
                self.best_params = progress['best_params']
                self.best_score = progress['best_score']
                self.param_bounds = progress['param_bounds']
                
                # Reconstruir modelo GP
                X = np.array([list(p.values()) for p in self._get_param_values()])
                y = np.array([h['score'] for h in self.history])
                self.gp.fit(X, y)
        except Exception as e:
            self.logger.error(f"Error cargando progreso: {str(e)}")
    
    def get_optimization_report(self) -> Dict:
        """Genera un reporte de la optimización"""
        try:
            if not self.history:
                return {}
            
            # Convertir historial a DataFrame
            df = pd.DataFrame(self.history)
            
            # Calcular estadísticas
            stats = {
                'best_score': self.best_score,
                'best_params': self.best_params,
                'total_iterations': len(self.history),
                'score_mean': df['score'].mean(),
                'score_std': df['score'].std(),
                'score_min': df['score'].min(),
                'score_max': df['score'].max(),
                'optimization_time': (
                    datetime.fromisoformat(df['timestamp'].iloc[-1]) -
                    datetime.fromisoformat(df['timestamp'].iloc[0])
                ).total_seconds()
            }
            
            # Análisis de convergencia
            stats['convergence'] = {
                'iterations_to_best': df['score'].argmax() + 1,
                'improvement_rate': (
                    (df['score'].max() - df['score'].iloc[0]) /
                    df['score'].iloc[0] * 100
                )
            }
            
            return stats
        except Exception as e:
            self.logger.error(f"Error generando reporte: {str(e)}")
            return {} 