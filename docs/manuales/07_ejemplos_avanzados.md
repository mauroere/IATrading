# Ejemplos Avanzados del Sistema de Trading

## 1. Trading Multi-Estrategia

### Objetivo
Combinar múltiples estrategias de trading con ponderación dinámica basada en rendimiento.

### Configuración
```yaml
trading_pairs:
  - "BTC/USDT"
  - "ETH/USDT"

timeframes:
  - "1h"
  - "4h"
  - "1d"

strategies:
  trend_following:
    weight: 0.4
    indicators:
      sma: [20, 50, 200]
      macd: [12, 26, 9]
      atr: 14
  
  mean_reversion:
    weight: 0.3
    indicators:
      bollinger: [20, 2.0]
      rsi: [14, 30, 70]
      stoch: [14, 3, 3]
  
  breakout:
    weight: 0.3
    indicators:
      ichimoku: [9, 26, 52]
      volume: [20, true]
      atr: 14

risk_management:
  position_sizing:
    method: "adaptive"
    max_position_size: 0.05
    strategy_weights: true
  
  stop_loss:
    method: "atr"
    atr_multiplier: 2.0
  
  take_profit:
    method: "atr"
    atr_multiplier: 3.0
```

### Ejemplo de Código
```python
class MultiStrategyTrader:
    def __init__(self, config):
        self.config = config
        self.strategies = {}
        self.performance = {}
        self.initialize_strategies()
    
    def initialize_strategies(self):
        # Inicializar cada estrategia
        self.strategies['trend_following'] = TrendFollowingStrategy(
            self.config['strategies']['trend_following']
        )
        self.strategies['mean_reversion'] = MeanReversionStrategy(
            self.config['strategies']['mean_reversion']
        )
        self.strategies['breakout'] = BreakoutStrategy(
            self.config['strategies']['breakout']
        )
    
    def update_weights(self):
        # Actualizar pesos basados en rendimiento
        total_performance = sum(self.performance.values())
        for strategy in self.strategies:
            self.strategies[strategy].weight = (
                self.performance[strategy] / total_performance
            )
    
    def analyze_market(self, df):
        signals = []
        weights = []
        
        # Obtener señales de cada estrategia
        for name, strategy in self.strategies.items():
            strategy_signals = strategy.analyze(df)
            if strategy_signals:
                signals.extend(strategy_signals)
                weights.extend([strategy.weight] * len(strategy_signals))
        
        # Combinar señales con pesos
        if signals:
            weighted_signals = []
            for signal, weight in zip(signals, weights):
                weighted_signals.append({
                    'signal': signal,
                    'weight': weight,
                    'strength': signal['strength'] * weight
                })
            
            # Agrupar señales por tipo
            grouped_signals = self._group_signals(weighted_signals)
            
            # Generar señal final
            final_signal = self._generate_final_signal(grouped_signals)
            return final_signal
        
        return None
    
    def _group_signals(self, weighted_signals):
        grouped = {'BUY': [], 'SELL': []}
        for signal in weighted_signals:
            grouped[signal['signal']['type']].append(signal)
        return grouped
    
    def _generate_final_signal(self, grouped_signals):
        buy_strength = sum(s['strength'] for s in grouped_signals['BUY'])
        sell_strength = sum(s['strength'] for s in grouped_signals['SELL'])
        
        if buy_strength > sell_strength and buy_strength > 0.6:
            return {
                'type': 'BUY',
                'strength': buy_strength,
                'stop_loss': self._calculate_stop_loss('BUY'),
                'take_profit': self._calculate_take_profit('BUY')
            }
        elif sell_strength > buy_strength and sell_strength > 0.6:
            return {
                'type': 'SELL',
                'strength': sell_strength,
                'stop_loss': self._calculate_stop_loss('SELL'),
                'take_profit': self._calculate_take_profit('SELL')
            }
        
        return None
```

## 2. Análisis de Sentimiento Avanzado

### Objetivo
Integrar análisis de sentimiento de redes sociales y noticias con análisis técnico.

### Configuración
```yaml
sentiment_analysis:
  sources:
    twitter:
      enabled: true
      keywords: ["bitcoin", "crypto", "btc", "eth"]
      min_followers: 1000
      time_window: 24h
    
    news:
      enabled: true
      sources: ["coindesk", "cointelegraph", "bloomberg"]
      time_window: 24h
    
    reddit:
      enabled: true
      subreddits: ["bitcoin", "cryptocurrency", "ethereum"]
      min_score: 10
      time_window: 24h

  analysis:
    method: "vader"
    min_confidence: 0.7
    weight: 0.3

technical_analysis:
  weight: 0.7
  indicators:
    sma: [20, 50, 200]
    rsi: [14, 30, 70]
    macd: [12, 26, 9]
```

### Ejemplo de Código
```python
class SentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.sources = {}
        self.initialize_sources()
    
    def initialize_sources(self):
        if self.config['sentiment_analysis']['sources']['twitter']['enabled']:
            self.sources['twitter'] = TwitterAnalyzer(
                self.config['sentiment_analysis']['sources']['twitter']
            )
        
        if self.config['sentiment_analysis']['sources']['news']['enabled']:
            self.sources['news'] = NewsAnalyzer(
                self.config['sentiment_analysis']['sources']['news']
            )
        
        if self.config['sentiment_analysis']['sources']['reddit']['enabled']:
            self.sources['reddit'] = RedditAnalyzer(
                self.config['sentiment_analysis']['sources']['reddit']
            )
    
    def analyze_sentiment(self):
        sentiment_scores = {}
        
        # Analizar cada fuente
        for source_name, source in self.sources.items():
            sentiment_scores[source_name] = source.analyze()
        
        # Combinar resultados
        combined_sentiment = self._combine_sentiment_scores(sentiment_scores)
        
        return combined_sentiment
    
    def _combine_sentiment_scores(self, sentiment_scores):
        total_score = 0
        total_weight = 0
        
        for source, score in sentiment_scores.items():
            weight = self.config['sentiment_analysis']['sources'][source]['weight']
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0

class MarketAnalyzer:
    def __init__(self, config):
        self.config = config
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.technical_analyzer = TechnicalAnalyzer(
            config['technical_analysis']
        )
    
    def analyze_market(self, df):
        # Análisis técnico
        technical_signal = self.technical_analyzer.analyze(df)
        
        # Análisis de sentimiento
        sentiment_score = self.sentiment_analyzer.analyze_sentiment()
        
        # Combinar análisis
        final_signal = self._combine_analysis(
            technical_signal,
            sentiment_score
        )
        
        return final_signal
    
    def _combine_analysis(self, technical_signal, sentiment_score):
        if not technical_signal:
            return None
        
        # Ajustar señal técnica con sentimiento
        sentiment_weight = self.config['sentiment_analysis']['analysis']['weight']
        technical_weight = self.config['technical_analysis']['weight']
        
        adjusted_strength = (
            technical_signal['strength'] * technical_weight +
            sentiment_score * sentiment_weight
        ) / (technical_weight + sentiment_weight)
        
        return {
            'type': technical_signal['type'],
            'strength': adjusted_strength,
            'stop_loss': technical_signal['stop_loss'],
            'take_profit': technical_signal['take_profit'],
            'sentiment_score': sentiment_score
        }
```

## 3. Optimización Avanzada de Parámetros

### Objetivo
Implementar optimización bayesiana con validación cruzada y walk-forward analysis.

### Configuración
```yaml
optimization:
  method: "bayesian"
  parameters:
    sma_periods: [10, 200]
    rsi_period: [7, 21]
    macd_fast: [8, 21]
    macd_slow: [21, 55]
    atr_period: [7, 21]
    atr_multiplier: [1.5, 3.0]
  
  validation:
    method: "walk_forward"
    train_size: 1000
    test_size: 200
    steps: 5
  
  metrics:
    - "sharpe_ratio"
    - "sortino_ratio"
    - "max_drawdown"
    - "win_rate"
    - "profit_factor"
  
  constraints:
    min_trades: 50
    max_drawdown: 0.2
    min_sharpe: 0.5
```

### Ejemplo de Código
```python
class BayesianOptimizer:
    def __init__(self, config):
        self.config = config
        self.parameter_space = self._create_parameter_space()
        self.best_params = None
        self.best_score = float('-inf')
    
    def _create_parameter_space(self):
        space = {}
        for param, (min_val, max_val) in self.config['optimization']['parameters'].items():
            space[param] = (min_val, max_val)
        return space
    
    def optimize(self, strategy, data):
        # Inicializar optimizador bayesiano
        optimizer = BayesianOptimization(
            f=self._objective_function,
            pbounds=self.parameter_space,
            random_state=42
        )
        
        # Ejecutar optimización
        optimizer.maximize(
            init_points=5,
            n_iter=20
        )
        
        self.best_params = optimizer.max['params']
        self.best_score = optimizer.max['target']
        
        return self.best_params
    
    def _objective_function(self, **params):
        # Validación walk-forward
        scores = []
        
        for i in range(self.config['optimization']['validation']['steps']):
            # Dividir datos
            train_data = self._get_train_data(i)
            test_data = self._get_test_data(i)
            
            # Entrenar y evaluar
            strategy.set_parameters(params)
            strategy.train(train_data)
            score = self._evaluate_strategy(strategy, test_data)
            scores.append(score)
        
        # Calcular score final
        final_score = np.mean(scores)
        
        # Verificar restricciones
        if not self._check_constraints(strategy, test_data):
            return float('-inf')
        
        return final_score
    
    def _evaluate_strategy(self, strategy, data):
        results = strategy.backtest(data)
        
        # Calcular métricas
        metrics = {}
        for metric in self.config['optimization']['metrics']:
            metrics[metric] = self._calculate_metric(metric, results)
        
        # Combinar métricas
        final_score = self._combine_metrics(metrics)
        
        return final_score
    
    def _check_constraints(self, strategy, data):
        results = strategy.backtest(data)
        
        if len(results['trades']) < self.config['optimization']['constraints']['min_trades']:
            return False
        
        if results['max_drawdown'] > self.config['optimization']['constraints']['max_drawdown']:
            return False
        
        if results['sharpe_ratio'] < self.config['optimization']['constraints']['min_sharpe']:
            return False
        
        return True
```

## 4. Gestión de Riesgo Avanzada

### Objetivo
Implementar gestión de riesgo dinámica con correlación de activos y ajuste de posición.

### Configuración
```yaml
risk_management:
  portfolio:
    max_correlation: 0.7
    max_drawdown: 0.2
    target_volatility: 0.15
  
  position_sizing:
    method: "kelly"
    max_position_size: 0.1
    min_position_size: 0.01
    correlation_adjustment: true
  
  stop_loss:
    method: "dynamic"
    atr_multiplier: 2.0
    volatility_adjustment: true
    correlation_adjustment: true
  
  take_profit:
    method: "dynamic"
    atr_multiplier: 3.0
    volatility_adjustment: true
    correlation_adjustment: true
  
  hedging:
    enabled: true
    method: "correlation"
    threshold: 0.8
```

### Ejemplo de Código
```python
class AdvancedRiskManager:
    def __init__(self, config):
        self.config = config
        self.portfolio = {}
        self.correlations = {}
        self.volatilities = {}
    
    def update_portfolio(self, positions):
        self.portfolio = positions
        self._update_correlations()
        self._update_volatilities()
    
    def _update_correlations(self):
        # Calcular correlaciones entre activos
        prices = pd.DataFrame()
        for asset, position in self.portfolio.items():
            prices[asset] = position['price_history']
        
        self.correlations = prices.corr()
    
    def _update_volatilities(self):
        # Calcular volatilidades
        for asset, position in self.portfolio.items():
            returns = pd.Series(position['price_history']).pct_change()
            self.volatilities[asset] = returns.std() * np.sqrt(252)
    
    def calculate_position_size(self, asset, signal):
        # Calcular tamaño base con Kelly
        kelly_fraction = self._calculate_kelly_fraction(asset)
        
        # Ajustar por correlación
        correlation_adjustment = self._calculate_correlation_adjustment(asset)
        
        # Ajustar por volatilidad
        volatility_adjustment = self._calculate_volatility_adjustment(asset)
        
        # Calcular tamaño final
        position_size = (
            kelly_fraction *
            correlation_adjustment *
            volatility_adjustment
        )
        
        # Aplicar límites
        position_size = max(
            min(position_size, self.config['risk_management']['position_sizing']['max_position_size']),
            self.config['risk_management']['position_sizing']['min_position_size']
        )
        
        return position_size
    
    def calculate_stop_loss(self, asset, entry_price, position_size):
        # Calcular ATR
        atr = self._calculate_atr(asset)
        
        # Ajustar por volatilidad
        volatility_adjustment = self._calculate_volatility_adjustment(asset)
        
        # Ajustar por correlación
        correlation_adjustment = self._calculate_correlation_adjustment(asset)
        
        # Calcular stop loss
        stop_loss_distance = (
            atr *
            self.config['risk_management']['stop_loss']['atr_multiplier'] *
            volatility_adjustment *
            correlation_adjustment
        )
        
        return entry_price - stop_loss_distance
    
    def calculate_take_profit(self, asset, entry_price, position_size):
        # Calcular ATR
        atr = self._calculate_atr(asset)
        
        # Ajustar por volatilidad
        volatility_adjustment = self._calculate_volatility_adjustment(asset)
        
        # Ajustar por correlación
        correlation_adjustment = self._calculate_correlation_adjustment(asset)
        
        # Calcular take profit
        take_profit_distance = (
            atr *
            self.config['risk_management']['take_profit']['atr_multiplier'] *
            volatility_adjustment *
            correlation_adjustment
        )
        
        return entry_price + take_profit_distance
    
    def check_hedging_needs(self, asset):
        if not self.config['risk_management']['hedging']['enabled']:
            return None
        
        # Buscar activos correlacionados
        correlated_assets = self._find_correlated_assets(asset)
        
        if correlated_assets:
            return self._calculate_hedge_ratio(asset, correlated_assets[0])
        
        return None
```

## 5. Monitoreo Avanzado del Sistema

### Objetivo
Implementar monitoreo en tiempo real con alertas y análisis predictivo.

### Configuración
```yaml
monitoring:
  system:
    metrics:
      - "cpu_usage"
      - "memory_usage"
      - "disk_usage"
      - "network_latency"
      - "api_errors"
    
    thresholds:
      cpu_usage: 80
      memory_usage: 85
      disk_usage: 90
      network_latency: 1000
      api_errors: 5
  
  trading:
    metrics:
      - "win_rate"
      - "profit_factor"
      - "sharpe_ratio"
      - "max_drawdown"
      - "position_correlation"
    
    thresholds:
      win_rate: 0.5
      profit_factor: 1.5
      sharpe_ratio: 1.0
      max_drawdown: 0.2
      position_correlation: 0.7
  
  alerts:
    channels:
      - "telegram"
      - "email"
      - "slack"
    
    severity:
      - "info"
      - "warning"
      - "error"
      - "critical"
```

### Ejemplo de Código
```python
class AdvancedSystemMonitor:
    def __init__(self, config):
        self.config = config
        self.metrics_history = {}
        self.alerts = []
        self.initialize_monitoring()
    
    def initialize_monitoring(self):
        # Inicializar monitoreo de sistema
        self.system_monitor = SystemMonitor(
            self.config['monitoring']['system']
        )
        
        # Inicializar monitoreo de trading
        self.trading_monitor = TradingMonitor(
            self.config['monitoring']['trading']
        )
        
        # Inicializar alertas
        self.alert_manager = AlertManager(
            self.config['monitoring']['alerts']
        )
    
    def monitor(self):
        # Monitorear sistema
        system_metrics = self.system_monitor.collect_metrics()
        self._check_system_metrics(system_metrics)
        
        # Monitorear trading
        trading_metrics = self.trading_monitor.collect_metrics()
        self._check_trading_metrics(trading_metrics)
        
        # Actualizar historial
        self._update_metrics_history(system_metrics, trading_metrics)
        
        # Analizar tendencias
        self._analyze_trends()
    
    def _check_system_metrics(self, metrics):
        for metric, value in metrics.items():
            threshold = self.config['monitoring']['system']['thresholds'][metric]
            
            if value > threshold:
                self.alert_manager.send_alert(
                    f"System {metric} above threshold: {value} > {threshold}",
                    "warning"
                )
    
    def _check_trading_metrics(self, metrics):
        for metric, value in metrics.items():
            threshold = self.config['monitoring']['trading']['thresholds'][metric]
            
            if value < threshold:
                self.alert_manager.send_alert(
                    f"Trading {metric} below threshold: {value} < {threshold}",
                    "warning"
                )
    
    def _update_metrics_history(self, system_metrics, trading_metrics):
        timestamp = pd.Timestamp.now()
        
        if 'system' not in self.metrics_history:
            self.metrics_history['system'] = pd.DataFrame()
        
        if 'trading' not in self.metrics_history:
            self.metrics_history['trading'] = pd.DataFrame()
        
        self.metrics_history['system'].loc[timestamp] = system_metrics
        self.metrics_history['trading'].loc[timestamp] = trading_metrics
    
    def _analyze_trends(self):
        # Analizar tendencias en métricas del sistema
        system_trends = self._analyze_system_trends()
        
        # Analizar tendencias en métricas de trading
        trading_trends = self._analyze_trading_trends()
        
        # Generar alertas si es necesario
        self._check_trend_alerts(system_trends, trading_trends)
    
    def generate_report(self):
        report = {
            'system': {
                'current_metrics': self.system_monitor.get_current_metrics(),
                'trends': self._analyze_system_trends(),
                'alerts': self.alert_manager.get_system_alerts()
            },
            'trading': {
                'current_metrics': self.trading_monitor.get_current_metrics(),
                'trends': self._analyze_trading_trends(),
                'alerts': self.alert_manager.get_trading_alerts()
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
```

## Siguientes Pasos

1. Consultar la guía de solución de problemas en `docs/manuales/08_solucion_problemas.md`
2. Revisar las mejores prácticas en `docs/manuales/05_mejores_practicas.md`
3. Explorar casos de uso adicionales en `docs/manuales/06_casos_uso.md` 