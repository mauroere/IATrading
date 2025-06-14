# Configuración Avanzada del Sistema de Trading

## Estructura de Configuración

El sistema utiliza un archivo `config.yaml` para su configuración. Aquí está la estructura completa con todas las opciones disponibles:

```yaml
# Configuración General
trading_pairs:
  - "BTC/USDT"
  - "ETH/USDT"
  - "BNB/USDT"

timeframes:
  - "1h"
  - "4h"
  - "1d"

# Configuración de Indicadores Técnicos
indicators:
  # Medias Móviles
  sma:
    periods: [20, 50, 200]
    weights: [0.4, 0.3, 0.3]
  
  # RSI
  rsi:
    period: 14
    overbought: 70
    oversold: 30
  
  # MACD
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  
  # Bollinger Bands
  bollinger:
    period: 20
    std_dev: 2.0
  
  # ATR
  atr:
    period: 14
  
  # Supertrend
  supertrend:
    period: 10
    multiplier: 3.0
  
  # Ichimoku
  ichimoku:
    tenkan_period: 9
    kijun_period: 26
    senkou_b_period: 52
    displacement: 26
  
  # Volumen
  volume:
    ma_period: 20
    obv_enabled: true
    cmf_period: 20

# Configuración de Gestión de Riesgo
risk_management:
  # Tamaño de Posición
  position_sizing:
    method: "adaptive"  # fixed, kelly, optimal_f, adaptive
    max_position_size: 0.1
    kelly_fraction: 0.5
    optimal_f_lookback: 100
  
  # Stop Loss y Take Profit
  stop_loss:
    method: "atr"  # fixed, atr, volatility
    atr_multiplier: 2.0
    fixed_percentage: 0.02
  
  take_profit:
    method: "atr"  # fixed, atr, volatility
    atr_multiplier: 3.0
    fixed_percentage: 0.03
  
  # Límites de Trading
  limits:
    max_daily_trades: 5
    max_concurrent_trades: 3
    max_drawdown: 0.05
    min_win_rate: 0.5
  
  # Validación de Trades
  validation:
    min_signal_strength: 0.7
    min_volume: 1000000
    max_spread: 0.001
    min_volatility: 0.001
    max_volatility: 0.05

# Configuración del Modelo de Machine Learning
ml_model:
  # Entrenamiento
  training:
    retraining_interval: 24  # horas
    min_training_samples: 1000
    validation_split: 0.2
    test_split: 0.1
  
  # Features
  features:
    technical: true
    market: true
    time: true
    volatility: true
    volume: true
    pattern: true
  
  # Modelo
  model:
    type: "xgboost"  # xgboost, lightgbm, catboost
    params:
      max_depth: 6
      learning_rate: 0.1
      n_estimators: 100
      subsample: 0.8
      colsample_bytree: 0.8
  
  # Feature Selection
  feature_selection:
    method: "importance"  # importance, correlation, pca
    importance_threshold: 0.01
    correlation_threshold: 0.7
    pca_components: 10

# Configuración de Análisis de Mercado
market_analysis:
  # Análisis Técnico
  technical:
    weight: 0.4
    indicators:
      trend: true
      momentum: true
      volatility: true
      volume: true
      pattern: true
  
  # Análisis de Volumen
  volume:
    weight: 0.2
    metrics:
      relative_volume: true
      obv: true
      cmf: true
  
  # Análisis de Sentimiento
  sentiment:
    weight: 0.2
    sources:
      social_media: true
      news: true
      fear_greed: true
  
  # Análisis de Order Flow
  order_flow:
    weight: 0.2
    metrics:
      buy_pressure: true
      sell_pressure: true
      spread: true

# Configuración del Optimizador
optimizer:
  # Método de Optimización
  method: "bayesian"  # bayesian, grid, random
  n_iterations: 100
  n_random_starts: 10
  
  # Parámetros a Optimizar
  parameters:
    rsi_period: [7, 21]
    macd_fast: [8, 16]
    macd_slow: [21, 34]
    bb_period: [10, 30]
    atr_period: [7, 21]
    supertrend_period: [7, 21]
    supertrend_multiplier: [2.0, 4.0]
  
  # Métricas de Optimización
  metrics:
    primary: "sharpe_ratio"
    secondary: ["win_rate", "profit_factor"]
    constraints:
      min_win_rate: 0.5
      min_profit_factor: 1.5
      max_drawdown: 0.1

# Configuración del Backtester
backtester:
  # Período de Backtesting
  period:
    start: "2023-01-01"
    end: "2024-01-01"
  
  # Walk-Forward Optimization
  walk_forward:
    enabled: true
    window_size: 30  # días
    step_size: 7  # días
  
  # Métricas
  metrics:
    returns: true
    drawdown: true
    win_rate: true
    profit_factor: true
    sharpe_ratio: true
    sortino_ratio: true
    calmar_ratio: true
  
  # Visualización
  visualization:
    equity_curve: true
    drawdown: true
    returns_distribution: true
    monthly_returns: true

# Configuración del Monitor
monitor:
  # Intervalos de Monitoreo
  intervals:
    system_metrics: 60  # segundos
    trading_metrics: 300  # segundos
    alerts: 60  # segundos
  
  # Métricas del Sistema
  system_metrics:
    cpu: true
    memory: true
    disk: true
    network: true
    errors: true
  
  # Métricas de Trading
  trading_metrics:
    trades: true
    performance: true
    risk: true
    signals: true
  
  # Alertas
  alerts:
    telegram: true
    email: false
    thresholds:
      cpu_usage: 80
      memory_usage: 80
      disk_usage: 80
      network_latency: 1000
      error_rate: 0.01
  
  # Base de Datos
  database:
    type: "sqlite"
    path: "monitoring.db"
    retention_days: 30

## 1. Configuración del Dashboard

### 1.1 Configuración General
```yaml
# config.yaml
dashboard:
  title: "Sistema de Trading"
  theme: "dark"
  layout: "wide"
  refresh_interval: 60  # segundos
  
  # Configuración de páginas
  pages:
    - name: "Dashboard"
      enabled: true
    - name: "Análisis Técnico"
      enabled: true
    - name: "Gestión de Riesgo"
      enabled: true
    - name: "Machine Learning"
      enabled: true
    - name: "Optimización"
      enabled: true
    - name: "Backtesting"
      enabled: true
    - name: "Calculadora de Ganancias"
      enabled: true
    - name: "Monitoreo"
      enabled: true
```

### 1.2 Configuración de Gráficos
```yaml
dashboard:
  charts:
    candlestick:
      colors:
        up: "#26a69a"
        down: "#ef5350"
      volume: true
      indicators: true
    
    performance:
      type: "line"
      colors:
        profit: "#26a69a"
        loss: "#ef5350"
      show_ma: true
      ma_period: 20
```

### 1.3 Configuración de Alertas
```yaml
dashboard:
  alerts:
    enabled: true
    channels:
      - type: "telegram"
        bot_token: "${TELEGRAM_BOT_TOKEN}"
        chat_id: "${TELEGRAM_CHAT_ID}"
      - type: "email"
        smtp_server: "smtp.gmail.com"
        smtp_port: 587
        sender_email: "${ALERT_EMAIL}"
        sender_password: "${ALERT_EMAIL_PASSWORD}"
    
    conditions:
      - type: "price"
        threshold: 0.05  # 5% de cambio
      - type: "volume"
        threshold: 2.0   # 2x volumen normal
      - type: "indicator"
        name: "RSI"
        threshold: 70
```

## 2. Configuración de la Calculadora de Ganancias

### 2.1 Configuración de Estrategias
```yaml
profit_calculator:
  strategies:
    scalping:
      enabled: true
      default_params:
        trade_size: 0.02
        win_rate: 0.6
        risk_reward: 1.5
        trades_per_day: 10
    
    grid_trading:
      enabled: true
      default_params:
        grid_levels: 10
        grid_spacing: 0.01
        position_size: 0.1
    
    martingale:
      enabled: true
      default_params:
        base_bet: 10
        multiplier: 2.0
        max_steps: 5
        win_rate: 0.5
    
    dca:
      enabled: true
      default_params:
        base_investment: 100
        dca_steps: 5
        dca_multiplier: 1.5
        price_decrease: 0.05
```

### 2.2 Configuración de Cálculos
```yaml
profit_calculator:
  calculations:
    fees:
      maker: 0.001  # 0.1%
      taker: 0.002  # 0.2%
    
    slippage:
      default: 0.001  # 0.1%
      max: 0.01      # 1%
    
    risk_management:
      max_position_size: 0.1  # 10% del capital
      max_drawdown: 0.2      # 20% máximo drawdown
      stop_loss: 0.02        # 2% stop loss
```

## 3. Configuración de Integración

### 3.1 Configuración de Componentes
```yaml
integration:
  components:
    indicators:
      enabled: true
      update_interval: 60  # segundos
      cache_size: 1000     # número de velas
    
    risk_manager:
      enabled: true
      update_interval: 30  # segundos
      max_positions: 10
    
    ml_model:
      enabled: true
      update_interval: 3600  # 1 hora
      retrain_interval: 86400  # 24 horas
    
    optimizer:
      enabled: true
      update_interval: 3600  # 1 hora
      max_iterations: 100
    
    backtester:
      enabled: true
      update_interval: 3600  # 1 hora
      max_history: 365  # días
    
    monitor:
      enabled: true
      update_interval: 60  # segundos
      metrics_retention: 7  # días
```

### 3.2 Configuración de Comunicación
```yaml
integration:
  communication:
    api:
      enabled: true
      port: 8000
      rate_limit: 100  # requests por minuto
    
    websocket:
      enabled: true
      port: 8001
      ping_interval: 30  # segundos
    
    database:
      type: "postgresql"
      host: "localhost"
      port: 5432
      name: "trading_db"
      user: "${DB_USER}"
      password: "${DB_PASSWORD}"
```

## 4. Configuración de Monitoreo

### 4.1 Métricas del Sistema
```yaml
monitoring:
  system:
    cpu:
      enabled: true
      threshold: 80  # %
      check_interval: 60  # segundos
    
    memory:
      enabled: true
      threshold: 80  # %
      check_interval: 60  # segundos
    
    disk:
      enabled: true
      threshold: 80  # %
      check_interval: 300  # segundos
    
    network:
      enabled: true
      latency_threshold: 1000  # ms
      check_interval: 30  # segundos
```

### 4.2 Métricas de Trading
```yaml
monitoring:
  trading:
    performance:
      enabled: true
      metrics:
        - "total_return"
        - "sharpe_ratio"
        - "sortino_ratio"
        - "max_drawdown"
      update_interval: 300  # segundos
    
    risk:
      enabled: true
      metrics:
        - "exposure"
        - "var"
        - "beta"
        - "correlation"
      update_interval: 60  # segundos
    
    operations:
      enabled: true
      metrics:
        - "win_rate"
        - "profit_factor"
        - "avg_trade"
        - "total_trades"
      update_interval: 60  # segundos
```

## 5. Configuración de Seguridad

### 5.1 Autenticación
```yaml
security:
  authentication:
    enabled: true
    method: "jwt"
    token_expiry: 3600  # segundos
    refresh_token_expiry: 604800  # 7 días
    
    users:
      - username: "${ADMIN_USER}"
        password: "${ADMIN_PASSWORD}"
        role: "admin"
      - username: "${TRADER_USER}"
        password: "${TRADER_PASSWORD}"
        role: "trader"
```

### 5.2 Autorización
```yaml
security:
  authorization:
    roles:
      admin:
        permissions:
          - "read:all"
          - "write:all"
          - "delete:all"
      
      trader:
        permissions:
          - "read:trading"
          - "write:trading"
          - "read:analysis"
          - "read:monitoring"
```

## 6. Siguientes Pasos

1. Revisar las mejores prácticas en `docs/manuales/05_mejores_practicas.md`
2. Explorar casos de uso en `docs/manuales/06_casos_uso.md`
3. Revisar ejemplos avanzados en `docs/manuales/07_ejemplos_avanzados.md`
4. Consultar la guía de solución de problemas en `docs/manuales/08_solucion_problemas.md` 