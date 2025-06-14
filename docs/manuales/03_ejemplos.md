# Ejemplos Prácticos del Sistema de Trading

## Ejemplo 1: Análisis Técnico Básico

### Objetivo
Analizar el par BTC/USDT usando indicadores técnicos básicos y generar señales de trading.

### Código
```python
import pandas as pd
import ccxt
from indicators import TechnicalIndicators
from datetime import datetime, timedelta

# Configuración
config = {
    'trading_pairs': ['BTC/USDT'],
    'timeframes': ['1h']
}

# Inicializar exchange
exchange = ccxt.binance({
    'apiKey': 'tu_api_key',
    'secret': 'tu_api_secret'
})

# Obtener datos históricos
symbol = 'BTC/USDT'
timeframe = '1h'
since = exchange.parse8601((datetime.now() - timedelta(days=30)).isoformat())
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)

# Convertir a DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Inicializar indicadores
indicators = TechnicalIndicators(config)

# Calcular indicadores
df = indicators.calculate_all_indicators(df)

# Obtener señales
signals = indicators.get_trading_signals(df)

# Mostrar últimas señales
print("\nÚltimas señales de trading:")
print(signals.tail())
```

### Resultado Esperado
```
Últimas señales de trading:
timestamp           signal  strength
2024-03-13 10:00:00   BUY    0.85
2024-03-13 11:00:00   HOLD   0.50
2024-03-13 12:00:00   SELL   0.75
```

## Ejemplo 2: Backtesting con Optimización

### Objetivo
Realizar un backtest del sistema con optimización de parámetros.

### Código
```python
import pandas as pd
from backtester import Backtester
from optimizer import BayesianOptimizer
import yaml

# Cargar configuración
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Cargar datos
df = pd.read_csv('data/btc_usdt_1h.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Inicializar componentes
backtester = Backtester(config)
optimizer = BayesianOptimizer(config)

# Función de evaluación
def evaluation_function(params):
    # Actualizar configuración con nuevos parámetros
    test_config = config.copy()
    test_config.update(params)
    
    # Ejecutar backtest
    results = backtester.run_backtest(df, test_config)
    
    # Retornar métrica de optimización (Sharpe Ratio)
    return results['metrics']['sharpe_ratio']

# Ejecutar optimización
best_params = optimizer.optimize(evaluation_function, n_iterations=50)

# Ejecutar backtest final con mejores parámetros
final_config = config.copy()
final_config.update(best_params)
final_results = backtester.run_backtest(df, final_config)

# Mostrar resultados
print("\nMejores parámetros encontrados:")
print(best_params)

print("\nResultados del backtest:")
print(f"Total Trades: {final_results['trades']}")
print(f"Win Rate: {final_results['metrics']['win_rate']:.2%}")
print(f"Profit Factor: {final_results['metrics']['profit_factor']:.2f}")
print(f"Sharpe Ratio: {final_results['metrics']['sharpe_ratio']:.2f}")

# Generar gráficos
backtester.plot_results('optimized_backtest_results.png')
```

### Resultado Esperado
```
Mejores parámetros encontrados:
{
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'bb_period': 20,
    'atr_period': 14
}

Resultados del backtest:
Total Trades: 156
Win Rate: 58.33%
Profit Factor: 1.85
Sharpe Ratio: 2.15
```

## Ejemplo 3: Sistema de Trading en Tiempo Real

### Objetivo
Implementar un sistema de trading en tiempo real con gestión de riesgo.

### Código
```python
import ccxt
import time
from datetime import datetime
from indicators import TechnicalIndicators
from risk_manager import RiskManager
from market_analysis import MarketAnalysis
import yaml
import telegram

# Cargar configuración
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inicializar componentes
exchange = ccxt.binance({
    'apiKey': config['binance_api_key'],
    'secret': config['binance_api_secret']
})

indicators = TechnicalIndicators(config)
risk_manager = RiskManager(config)
market_analysis = MarketAnalysis(config)
telegram_bot = telegram.Bot(token=config['telegram_bot_token'])

def send_telegram_message(message):
    telegram_bot.send_message(
        chat_id=config['telegram_chat_id'],
        text=message
    )

def process_trading_signals(symbol):
    try:
        # Obtener datos
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calcular indicadores
        df = indicators.calculate_all_indicators(df)
        
        # Obtener señales
        signals = indicators.get_trading_signals(df)
        latest_signal = signals.iloc[-1]
        
        # Analizar mercado
        market_conditions = market_analysis.analyze_market_conditions(symbol)
        
        if latest_signal['signal'] != 'HOLD':
            # Obtener balance
            balance = float(exchange.fetch_balance()['USDT']['free'])
            
            # Obtener precio actual
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calcular ATR
            atr = df['atr'].iloc[-1]
            
            # Validar trade
            if risk_manager.validate_trade(current_price, balance, latest_signal['strength']):
                # Calcular tamaño de posición
                position_size = risk_manager.calculate_position_size(
                    balance=balance,
                    price=current_price,
                    atr=atr,
                    signal_strength=latest_signal['strength']
                )
                
                # Calcular stop loss y take profit
                stop_loss, take_profit = risk_manager.calculate_stop_loss_take_profit(
                    price=current_price,
                    atr=atr,
                    signal_strength=latest_signal['strength']
                )
                
                # Enviar alerta
                message = (
                    f"🔔 Nueva Señal de Trading\n\n"
                    f"Par: {symbol}\n"
                    f"Señal: {latest_signal['signal']}\n"
                    f"Fuerza: {latest_signal['strength']:.2f}\n"
                    f"Precio: {current_price:.2f}\n"
                    f"Stop Loss: {stop_loss:.2f}\n"
                    f"Take Profit: {take_profit:.2f}\n"
                    f"Tamaño: {position_size:.4f}\n\n"
                    f"Condición de Mercado: {market_conditions['market_condition']}"
                )
                send_telegram_message(message)
                
    except Exception as e:
        error_message = f"❌ Error procesando señales: {str(e)}"
        send_telegram_message(error_message)

def main():
    send_telegram_message("🚀 Sistema de Trading iniciado")
    
    while True:
        try:
            for symbol in config['trading_pairs']:
                process_trading_signals(symbol)
            
            # Esperar 1 hora
            time.sleep(3600)
            
        except Exception as e:
            error_message = f"❌ Error en el bucle principal: {str(e)}"
            send_telegram_message(error_message)
            time.sleep(60)

if __name__ == "__main__":
    main()
```

### Resultado Esperado
```
🚀 Sistema de Trading iniciado

🔔 Nueva Señal de Trading

Par: BTC/USDT
Señal: BUY
Fuerza: 0.85
Precio: 50000.00
Stop Loss: 49000.00
Take Profit: 52000.00
Tamaño: 0.1000

Condición de Mercado: BULLISH
```

## Ejemplo 4: Análisis de Mercado Avanzado

### Objetivo
Realizar un análisis completo de mercado incluyendo análisis técnico, volumen, sentimiento y order flow.

### Código
```python
import pandas as pd
from market_analysis import MarketAnalysis
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar configuración
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inicializar análisis de mercado
market_analysis = MarketAnalysis(config)

# Analizar mercado
symbol = 'BTC/USDT'
conditions = market_analysis.analyze_market_conditions(symbol)

# Crear visualización
plt.figure(figsize=(15, 10))

# Gráfico de condiciones técnicas
plt.subplot(2, 2, 1)
technical_data = pd.Series(conditions['technical'])
technical_data.plot(kind='bar')
plt.title('Análisis Técnico')
plt.xticks(rotation=45)

# Gráfico de volumen
plt.subplot(2, 2, 2)
volume_data = pd.Series(conditions['volume'])
volume_data.plot(kind='bar')
plt.title('Análisis de Volumen')
plt.xticks(rotation=45)

# Gráfico de sentimiento
plt.subplot(2, 2, 3)
sentiment_data = pd.Series(conditions['sentiment'])
sentiment_data.plot(kind='bar')
plt.title('Análisis de Sentimiento')
plt.xticks(rotation=45)

# Gráfico de order flow
plt.subplot(2, 2, 4)
order_flow_data = pd.Series(conditions['order_flow'])
order_flow_data.plot(kind='bar')
plt.title('Análisis de Order Flow')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('market_analysis.png')
plt.close()

# Mostrar resumen
print("\nResumen de Análisis de Mercado:")
print(f"Condición General: {conditions['market_condition']}")
print(f"Score Técnico: {conditions['technical']['score']:.2f}")
print(f"Score de Volumen: {conditions['volume']['score']:.2f}")
print(f"Score de Sentimiento: {conditions['sentiment']['score']:.2f}")
print(f"Score de Order Flow: {conditions['order_flow']['score']:.2f}")
```

### Resultado Esperado
```
Resumen de Análisis de Mercado:
Condición General: BULLISH
Score Técnico: 0.75
Score de Volumen: 0.65
Score de Sentimiento: 0.80
Score de Order Flow: 0.70
```

## Ejemplo 5: Monitoreo del Sistema

### Objetivo
Implementar un sistema de monitoreo completo con alertas y reportes.

### Código
```python
from monitor import SystemMonitor
import yaml
import time
from datetime import datetime

# Cargar configuración
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inicializar monitor
monitor = SystemMonitor(config)

# Iniciar monitoreo
monitor.start_monitoring()

# Esperar algunas métricas
time.sleep(300)  # 5 minutos

# Generar reporte
report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
monitor.generate_report(f'monitoring_report_{report_time}.png')

# Obtener métricas
system_metrics = monitor._get_system_metrics_history()
trading_metrics = monitor._get_trading_metrics_history()

# Mostrar resumen
print("\nResumen de Monitoreo:")
print("\nMétricas del Sistema:")
print(f"CPU: {system_metrics['cpu_usage'].mean():.1f}%")
print(f"Memoria: {system_metrics['memory_usage'].mean():.1f}%")
print(f"Disco: {system_metrics['disk_usage'].mean():.1f}%")
print(f"Latencia: {system_metrics['network_latency'].mean():.1f}ms")

print("\nMétricas de Trading:")
print(f"Trades Totales: {trading_metrics['total_trades'].sum()}")
print(f"Win Rate: {trading_metrics['win_rate'].mean():.1%}")
print(f"Profit Factor: {trading_metrics['profit_factor'].mean():.2f}")
print(f"Sharpe Ratio: {trading_metrics['sharpe_ratio'].mean():.2f}")

# Detener monitoreo
monitor.close()
```

### Resultado Esperado
```
Resumen de Monitoreo:

Métricas del Sistema:
CPU: 45.2%
Memoria: 60.5%
Disco: 35.8%
Latencia: 150.3ms

Métricas de Trading:
Trades Totales: 156
Win Rate: 58.3%
Profit Factor: 1.85
Sharpe Ratio: 2.15
```

## 1. Uso del Dashboard

### 1.1 Iniciar el Dashboard
```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar el dashboard
streamlit run dashboard.py
```

### 1.2 Navegación Básica
```python
# Ejemplo de navegación entre secciones
dashboard = TradingDashboard()
dashboard.run()  # Inicia el dashboard con todas las secciones
```

### 1.3 Personalización del Dashboard
```python
# Ejemplo de configuración personalizada
config = {
    'trading_pairs': ['BTC/USDT', 'ETH/USDT'],
    'timeframes': ['1h', '4h', '1d'],
    'indicators': {
        'sma': [20, 50, 200],
        'rsi': 14,
        'macd': {'fast': 12, 'slow': 26, 'signal': 9}
    }
}

# Guardar configuración
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
```

## 2. Calculadora de Ganancias

### 2.1 Scalping
```python
# Ejemplo de cálculo de ganancias con scalping
calculator = ProfitCalculator()

results = calculator.calculate_scalping_profit(
    initial_capital=1000,
    trade_size=0.02,  # 2% del capital
    win_rate=0.6,     # 60% de operaciones ganadoras
    risk_reward=1.5,  # Ratio riesgo/beneficio
    trades_per_day=10,
    days=30
)

print(f"Ganancia total: {results['total_profit']:.2f} USDT")
print(f"ROI: {results['roi']:.2f}%")
print(f"Ganancia diaria promedio: {results['daily_profit']:.2f} USDT")
```

### 2.2 Grid Trading
```python
# Ejemplo de cálculo de ganancias con grid trading
results = calculator.calculate_grid_trading_profit(
    initial_capital=1000,
    grid_levels=10,
    grid_spacing=0.01,  # 1% entre niveles
    position_size=0.1,
    price_range=(30000, 35000),
    days=30
)

print(f"Ganancia total: {results['total_profit']:.2f} USDT")
print(f"ROI: {results['roi']:.2f}%")
print(f"Operaciones totales: {results['total_trades']}")
```

### 2.3 Martingala
```python
# Ejemplo de cálculo de ganancias con martingala
results = calculator.calculate_martingale_profit(
    initial_capital=1000,
    base_bet=10,
    multiplier=2.0,
    max_steps=5,
    win_rate=0.5,
    trades_per_day=20,
    days=30
)

print(f"Ganancia total: {results['total_profit']:.2f} USDT")
print(f"ROI: {results['roi']:.2f}%")
print(f"Riesgo máximo: {results['max_risk']:.2f} USDT")
```

### 2.4 DCA (Dollar Cost Averaging)
```python
# Ejemplo de cálculo de ganancias con DCA
results = calculator.calculate_dca_profit(
    initial_capital=1000,
    base_investment=100,
    dca_steps=5,
    dca_multiplier=1.5,
    price_decrease=0.05,  # 5% de disminución por paso
    days=30
)

print(f"Ganancia total: {results['total_profit']:.2f} USDT")
print(f"ROI: {results['roi']:.2f}%")
print(f"Precio promedio de entrada: {results['avg_entry_price']:.2f} USDT")
```

## 3. Integración de Componentes

### 3.1 Análisis Técnico con Dashboard
```python
# Ejemplo de análisis técnico integrado
dashboard = TradingDashboard()

# Obtener datos y calcular indicadores
symbol = 'BTC/USDT'
timeframe = '1h'
indicators = dashboard.indicators.calculate_all(symbol, timeframe)

# Mostrar gráfico en el dashboard
dashboard.show_technical_analysis()
```

### 3.2 Gestión de Riesgo Integrada
```python
# Ejemplo de gestión de riesgo integrada
dashboard = TradingDashboard()

# Configurar límites de riesgo
dashboard.risk_manager.update_config({
    'max_position_size': 0.05,  # 5% del capital
    'max_drawdown': 0.20,      # 20% máximo drawdown
    'risk_per_trade': 0.01     # 1% de riesgo por operación
})

# Obtener métricas de riesgo
risk_metrics = dashboard.risk_manager.get_risk_metrics()
print(f"Exposición total: {risk_metrics['total_exposure']:.2f}%")
```

### 3.3 Monitoreo del Sistema
```python
# Ejemplo de monitoreo integrado
dashboard = TradingDashboard()

# Obtener métricas del sistema
cpu_usage = dashboard.monitor.get_cpu_usage()
memory_usage = dashboard.monitor.get_memory_usage()
latency = dashboard.monitor.get_latency()

print(f"CPU: {cpu_usage:.1f}%")
print(f"Memoria: {memory_usage:.1f}%")
print(f"Latencia: {latency:.0f}ms")
```

## 4. Ejemplos de Optimización

### 4.1 Optimización de Parámetros
```python
# Ejemplo de optimización integrada
dashboard = TradingDashboard()

# Configurar y ejecutar optimización
results = dashboard.optimizer.optimize(
    method="Bayesiano",
    metric="Sharpe Ratio"
)

print("Mejores parámetros encontrados:")
print(json.dumps(results['best_params'], indent=2))
```

### 4.2 Backtesting Integrado
```python
# Ejemplo de backtesting integrado
dashboard = TradingDashboard()

# Ejecutar backtesting
results = dashboard.backtester.run(
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now()
)

print(f"Rendimiento total: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Tasa de éxito: {results['win_rate']:.2f}%")
```

## 5. Siguientes Pasos

1. Revisar la configuración avanzada en `docs/manuales/04_configuracion_avanzada.md`
2. Consultar las mejores prácticas en `docs/manuales/05_mejores_practicas.md`
3. Explorar casos de uso en `docs/manuales/06_casos_uso.md`
4. Revisar ejemplos avanzados en `docs/manuales/07_ejemplos_avanzados.md`
5. Consultar la guía de solución de problemas en `docs/manuales/08_solucion_problemas.md` 