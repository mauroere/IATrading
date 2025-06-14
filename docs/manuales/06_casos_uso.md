# Casos de Uso del Sistema de Trading

## 1. Trading de Tendencia

### Objetivo
Identificar y operar tendencias de mercado usando múltiples timeframes y confirmación de volumen.

### Configuración
```yaml
trading_pairs:
  - "BTC/USDT"
  - "ETH/USDT"

timeframes:
  - "4h"
  - "1d"

indicators:
  sma:
    periods: [20, 50, 200]
    weights: [0.4, 0.3, 0.3]
  
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  
  volume:
    ma_period: 20
    obv_enabled: true

risk_management:
  position_sizing:
    method: "adaptive"
    max_position_size: 0.05
  
  stop_loss:
    method: "atr"
    atr_multiplier: 2.0
  
  take_profit:
    method: "atr"
    atr_multiplier: 3.0
```

### Estrategia
1. Identificar tendencia en timeframe diario
2. Buscar entradas en timeframe de 4 horas
3. Confirmar con volumen y momentum
4. Usar stop loss dinámico basado en ATR
5. Trailing stop para maximizar ganancias

### Ejemplo de Código
```python
def identify_trend(df):
    # Calcular medias móviles
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Determinar tendencia
    if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1]:
        return 'UPTREND'
    elif df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1] < df['sma_200'].iloc[-1]:
        return 'DOWNTREND'
    else:
        return 'SIDEWAYS'

def find_entry_signals(df):
    signals = []
    
    # Calcular MACD
    macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    
    # Calcular volumen
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['obv'] = (df['volume'] * (df['close'] - df['open'])).cumsum()
    
    for i in range(1, len(df)):
        # Señal de compra
        if (macd.iloc[i] > signal.iloc[i] and 
            macd.iloc[i-1] <= signal.iloc[i-1] and
            df['volume'].iloc[i] > df['volume_ma'].iloc[i] and
            df['obv'].iloc[i] > df['obv'].iloc[i-1]):
            signals.append(('BUY', df.index[i]))
        
        # Señal de venta
        elif (macd.iloc[i] < signal.iloc[i] and 
              macd.iloc[i-1] >= signal.iloc[i-1] and
              df['volume'].iloc[i] > df['volume_ma'].iloc[i] and
              df['obv'].iloc[i] < df['obv'].iloc[i-1]):
            signals.append(('SELL', df.index[i]))
    
    return signals
```

## 2. Trading de Rango

### Objetivo
Operar en mercados laterales usando soporte y resistencia, y osciladores.

### Configuración
```yaml
trading_pairs:
  - "BTC/USDT"
  - "ETH/USDT"

timeframes:
  - "1h"
  - "4h"

indicators:
  bollinger:
    period: 20
    std_dev: 2.0
  
  rsi:
    period: 14
    overbought: 70
    oversold: 30
  
  atr:
    period: 14

risk_management:
  position_sizing:
    method: "fixed"
    max_position_size: 0.03
  
  stop_loss:
    method: "fixed"
    fixed_percentage: 0.01
  
  take_profit:
    method: "fixed"
    fixed_percentage: 0.02
```

### Estrategia
1. Identificar rangos de trading
2. Usar Bollinger Bands para niveles
3. Confirmar con RSI
4. Stop loss y take profit fijos
5. Salir en reversión de tendencia

### Ejemplo de Código
```python
def identify_range(df):
    # Calcular Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Calcular RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Identificar rango
    range_high = df['bb_upper'].iloc[-20:].mean()
    range_low = df['bb_lower'].iloc[-20:].mean()
    range_size = (range_high - range_low) / range_low
    
    return range_size < 0.05  # 5% de rango

def find_range_signals(df):
    signals = []
    
    for i in range(1, len(df)):
        # Señal de compra
        if (df['close'].iloc[i] < df['bb_lower'].iloc[i] and
            df['rsi'].iloc[i] < 30):
            signals.append(('BUY', df.index[i]))
        
        # Señal de venta
        elif (df['close'].iloc[i] > df['bb_upper'].iloc[i] and
              df['rsi'].iloc[i] > 70):
            signals.append(('SELL', df.index[i]))
    
    return signals
```

## 3. Trading de Breakout

### Objetivo
Identificar y operar rupturas de niveles importantes con confirmación de volumen.

### Configuración
```yaml
trading_pairs:
  - "BTC/USDT"
  - "ETH/USDT"

timeframes:
  - "1h"
  - "4h"

indicators:
  atr:
    period: 14
  
  volume:
    ma_period: 20
    obv_enabled: true
  
  ichimoku:
    tenkan_period: 9
    kijun_period: 26
    senkou_b_period: 52

risk_management:
  position_sizing:
    method: "adaptive"
    max_position_size: 0.04
  
  stop_loss:
    method: "atr"
    atr_multiplier: 1.5
  
  take_profit:
    method: "atr"
    atr_multiplier: 2.5
```

### Estrategia
1. Identificar niveles de soporte/resistencia
2. Esperar consolidación
3. Confirmar ruptura con volumen
4. Usar stop loss cercano
5. Take profit basado en ATR

### Ejemplo de Código
```python
def identify_breakout_levels(df):
    # Calcular niveles de soporte y resistencia
    highs = df['high'].rolling(window=20).max()
    lows = df['low'].rolling(window=20).min()
    
    # Calcular Ichimoku
    df['tenkan'] = (df['high'].rolling(window=9).max() + 
                    df['low'].rolling(window=9).min()) / 2
    df['kijun'] = (df['high'].rolling(window=26).max() + 
                   df['low'].rolling(window=26).min()) / 2
    
    # Identificar niveles
    levels = []
    for i in range(20, len(df)):
        if df['close'].iloc[i] > highs.iloc[i-1]:
            levels.append(('RESISTANCE', df.index[i], df['close'].iloc[i]))
        elif df['close'].iloc[i] < lows.iloc[i-1]:
            levels.append(('SUPPORT', df.index[i], df['close'].iloc[i]))
    
    return levels

def find_breakout_signals(df, levels):
    signals = []
    
    # Calcular volumen
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['obv'] = (df['volume'] * (df['close'] - df['open'])).cumsum()
    
    for level in levels:
        level_type, level_time, level_price = level
        
        # Buscar rupturas
        for i in range(1, len(df)):
            if df.index[i] > level_time:
                # Ruptura alcista
                if (level_type == 'RESISTANCE' and
                    df['close'].iloc[i] > level_price and
                    df['volume'].iloc[i] > df['volume_ma'].iloc[i] * 1.5 and
                    df['obv'].iloc[i] > df['obv'].iloc[i-1]):
                    signals.append(('BUY', df.index[i]))
                
                # Ruptura bajista
                elif (level_type == 'SUPPORT' and
                      df['close'].iloc[i] < level_price and
                      df['volume'].iloc[i] > df['volume_ma'].iloc[i] * 1.5 and
                      df['obv'].iloc[i] < df['obv'].iloc[i-1]):
                    signals.append(('SELL', df.index[i]))
    
    return signals
```

## 4. Trading de Momentum

### Objetivo
Operar movimientos de precio basados en momentum y volumen.

### Configuración
```yaml
trading_pairs:
  - "BTC/USDT"
  - "ETH/USDT"

timeframes:
  - "15m"
  - "1h"

indicators:
  rsi:
    period: 14
    overbought: 70
    oversold: 30
  
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  
  volume:
    ma_period: 20
    cmf_period: 20

risk_management:
  position_sizing:
    method: "adaptive"
    max_position_size: 0.03
  
  stop_loss:
    method: "atr"
    atr_multiplier: 1.5
  
  take_profit:
    method: "atr"
    atr_multiplier: 2.0
```

### Estrategia
1. Identificar momentum con RSI
2. Confirmar con MACD
3. Validar con volumen
4. Stop loss dinámico
5. Take profit escalonado

### Ejemplo de Código
```python
def identify_momentum(df):
    # Calcular RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calcular MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['signal'] = df['macd'].ewm(span=9).mean()
    
    # Calcular CMF
    df['cmf'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['cmf'] = df['cmf'] * df['volume']
    df['cmf'] = df['cmf'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    return df

def find_momentum_signals(df):
    signals = []
    
    for i in range(1, len(df)):
        # Señal de compra
        if (df['rsi'].iloc[i] < 30 and
            df['macd'].iloc[i] > df['signal'].iloc[i] and
            df['cmf'].iloc[i] > 0):
            signals.append(('BUY', df.index[i]))
        
        # Señal de venta
        elif (df['rsi'].iloc[i] > 70 and
              df['macd'].iloc[i] < df['signal'].iloc[i] and
              df['cmf'].iloc[i] < 0):
            signals.append(('SELL', df.index[i]))
    
    return signals
```

## 5. Trading de Patrones

### Objetivo
Identificar y operar patrones de velas japonesas con confirmación técnica.

### Configuración
```yaml
trading_pairs:
  - "BTC/USDT"
  - "ETH/USDT"

timeframes:
  - "1h"
  - "4h"

indicators:
  atr:
    period: 14
  
  volume:
    ma_period: 20
    obv_enabled: true
  
  ichimoku:
    tenkan_period: 9
    kijun_period: 26
    senkou_b_period: 52

risk_management:
  position_sizing:
    method: "adaptive"
    max_position_size: 0.03
  
  stop_loss:
    method: "atr"
    atr_multiplier: 1.5
  
  take_profit:
    method: "atr"
    atr_multiplier: 2.0
```

### Estrategia
1. Identificar patrones de velas
2. Confirmar con Ichimoku
3. Validar con volumen
4. Stop loss basado en ATR
5. Take profit dinámico

### Ejemplo de Código
```python
def identify_candlestick_patterns(df):
    patterns = []
    
    for i in range(2, len(df)):
        # Doji
        if abs(df['close'].iloc[i] - df['open'].iloc[i]) <= 0.1 * (df['high'].iloc[i] - df['low'].iloc[i]):
            patterns.append(('DOJI', df.index[i]))
        
        # Engulfing alcista
        elif (df['close'].iloc[i] > df['open'].iloc[i] and
              df['close'].iloc[i-1] < df['open'].iloc[i-1] and
              df['close'].iloc[i] > df['open'].iloc[i-1] and
              df['open'].iloc[i] < df['close'].iloc[i-1]):
            patterns.append(('BULLISH_ENGULFING', df.index[i]))
        
        # Engulfing bajista
        elif (df['close'].iloc[i] < df['open'].iloc[i] and
              df['close'].iloc[i-1] > df['open'].iloc[i-1] and
              df['close'].iloc[i] < df['open'].iloc[i-1] and
              df['open'].iloc[i] > df['close'].iloc[i-1]):
            patterns.append(('BEARISH_ENGULFING', df.index[i]))
    
    return patterns

def find_pattern_signals(df, patterns):
    signals = []
    
    # Calcular Ichimoku
    df['tenkan'] = (df['high'].rolling(window=9).max() + 
                    df['low'].rolling(window=9).min()) / 2
    df['kijun'] = (df['high'].rolling(window=26).max() + 
                   df['low'].rolling(window=26).min()) / 2
    
    # Calcular volumen
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['obv'] = (df['volume'] * (df['close'] - df['open'])).cumsum()
    
    for pattern in patterns:
        pattern_type, pattern_time = pattern
        
        # Buscar señales
        for i in range(1, len(df)):
            if df.index[i] == pattern_time:
                # Señal de compra
                if (pattern_type in ['DOJI', 'BULLISH_ENGULFING'] and
                    df['close'].iloc[i] > df['tenkan'].iloc[i] and
                    df['volume'].iloc[i] > df['volume_ma'].iloc[i] and
                    df['obv'].iloc[i] > df['obv'].iloc[i-1]):
                    signals.append(('BUY', df.index[i]))
                
                # Señal de venta
                elif (pattern_type in ['DOJI', 'BEARISH_ENGULFING'] and
                      df['close'].iloc[i] < df['tenkan'].iloc[i] and
                      df['volume'].iloc[i] > df['volume_ma'].iloc[i] and
                      df['obv'].iloc[i] < df['obv'].iloc[i-1]):
                    signals.append(('SELL', df.index[i]))
    
    return signals
```

## Siguientes Pasos

1. Explorar ejemplos avanzados en `docs/manuales/07_ejemplos_avanzados.md`
2. Consultar la guía de solución de problemas en `docs/manuales/08_solucion_problemas.md`
3. Revisar las mejores prácticas en `docs/manuales/05_mejores_practicas.md` 