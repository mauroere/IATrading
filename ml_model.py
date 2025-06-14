import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from datetime import datetime, timedelta

class MLModel:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.last_training_time = None
        self.performance_metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara las características para el modelo"""
        try:
            # Seleccionar características
            features = self._select_features(df)
            
            # Crear características técnicas
            features = self._create_technical_features(features)
            
            # Crear características de mercado
            features = self._create_market_features(features)
            
            # Crear características de tiempo
            features = self._create_time_features(features)
            
            # Crear características de volatilidad
            features = self._create_volatility_features(features)
            
            # Crear características de volumen
            features = self._create_volume_features(features)
            
            # Crear características de patrones
            features = self._create_pattern_features(features)
            
            # Crear target
            target = self._create_target(features)
            
            # Limpiar datos
            features = features.dropna()
            target = target[features.index]
            
            return features, target
        except Exception as e:
            self.logger.error(f"Error preparando características: {str(e)}")
            raise

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selecciona las características relevantes"""
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_upper', 'bb_middle', 'bb_lower',
            'stoch_k', 'stoch_d',
            'atr', 'vwap', 'obv', 'cmf'
        ]
        return df[feature_columns].copy()

    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características técnicas adicionales"""
        # Retornos
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(4) - 1
        df['rate_of_change'] = df['close'].pct_change(periods=10)
        
        # Volatilidad
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Tendencia
        df['trend'] = df['close'].rolling(window=20).mean() / df['close'] - 1
        
        return df

    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de mercado"""
        # Spread
        df['spread'] = (df['high'] - df['low']) / df['close']
        
        # Rango de precio
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Fuerza relativa
        df['relative_strength'] = df['close'] / df['close'].rolling(window=20).mean() - 1
        
        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de tiempo"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Patrones de tiempo
        df['is_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 13) & (df['hour'] <= 17)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
        
        return df

    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de volatilidad"""
        # Volatilidad histórica
        df['historical_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volatilidad implícita (usando ATR)
        df['implied_volatility'] = df['atr'] / df['close'] * np.sqrt(252)
        
        # Ratio de volatilidad
        df['volatility_ratio'] = df['historical_volatility'] / df['implied_volatility']
        
        return df

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de volumen"""
        # Volumen relativo
        df['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Cambio en volumen
        df['volume_change'] = df['volume'].pct_change()
        
        # Volumen ponderado
        df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        return df

    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de patrones"""
        # Patrones de velas
        df['doji'] = abs(df['open'] - df['close']) / (df['high'] - df['low'])
        df['hammer'] = ((df['high'] - df['low']) > 3 * (df['open'] - df['close'])) & \
                      ((df['close'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6)
        
        # Patrones de precio
        df['higher_highs'] = df['high'] > df['high'].shift(1)
        df['lower_lows'] = df['low'] < df['low'].shift(1)
        
        return df

    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Crea la variable objetivo"""
        # Retorno futuro
        future_returns = df['close'].shift(-1) / df['close'] - 1
        
        # Clasificación binaria
        target = (future_returns > self.config['min_return_threshold']).astype(int)
        
        return target

    def train(self, features: pd.DataFrame, target: pd.Series):
        """Entrena el modelo"""
        try:
            # Dividir datos en entrenamiento y validación
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Preparar datos
            X = self.scaler.fit_transform(features)
            y = target.values
            
            # Configurar modelo
            self.model = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=7,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                scale_pos_weight=1,
                random_state=42
            )
            
            # Entrenar modelo
            self.model.fit(
                X, y,
                eval_set=[(X, y)],
                eval_metric='auc',
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Calcular importancia de características
            self.feature_importance = dict(zip(
                features.columns,
                self.model.feature_importances_
            ))
            
            # Calcular métricas de rendimiento
            self._calculate_performance_metrics(X, y)
            
            # Actualizar tiempo de último entrenamiento
            self.last_training_time = datetime.now()
            
            # Guardar modelo
            self._save_model()
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo: {str(e)}")
            raise

    def _calculate_performance_metrics(self, X: np.ndarray, y: np.ndarray):
        """Calcula métricas de rendimiento del modelo"""
        try:
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            self.performance_metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'auc': self.model.evals_result()['validation_0']['auc'][-1]
            }
        except Exception as e:
            self.logger.error(f"Error calculando métricas: {str(e)}")

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones"""
        try:
            # Verificar si el modelo necesita reentrenamiento
            if self._needs_retraining():
                self.logger.warning("Modelo necesita reentrenamiento")
                return np.zeros(len(features)), np.zeros(len(features))
            
            # Preparar datos
            X = self.scaler.transform(features)
            
            # Realizar predicciones
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            return predictions, probabilities
        except Exception as e:
            self.logger.error(f"Error realizando predicciones: {str(e)}")
            return np.zeros(len(features)), np.zeros(len(features))

    def _needs_retraining(self) -> bool:
        """Verifica si el modelo necesita reentrenamiento"""
        if self.last_training_time is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return hours_since_training >= self.config['retrain_interval']

    def _save_model(self):
        """Guarda el modelo y el scaler"""
        try:
            joblib.dump(self.model, 'model/xgboost_model.joblib')
            joblib.dump(self.scaler, 'model/scaler.joblib')
        except Exception as e:
            self.logger.error(f"Error guardando modelo: {str(e)}")

    def load_model(self):
        """Carga el modelo y el scaler"""
        try:
            self.model = joblib.load('model/xgboost_model.joblib')
            self.scaler = joblib.load('model/scaler.joblib')
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {str(e)}")

    def get_feature_importance(self) -> Dict:
        """Obtiene la importancia de las características"""
        return self.feature_importance

    def get_performance_metrics(self) -> Dict:
        """Obtiene las métricas de rendimiento"""
        return self.performance_metrics 