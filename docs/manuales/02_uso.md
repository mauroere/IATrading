# Manual de Uso del Sistema de Trading

## Estructura del Sistema

El sistema está compuesto por varios módulos principales:

1. **Indicadores Técnicos** (`indicators.py`)
2. **Gestión de Riesgo** (`risk_manager.py`)
3. **Modelo de Machine Learning** (`ml_model.py`)
4. **Análisis de Mercado** (`market_analysis.py`)
5. **Optimizador** (`optimizer.py`)
6. **Backtester** (`backtester.py`)
7. **Monitor** (`monitor.py`)

## Inicio del Sistema

### 1. Verificación Inicial
Antes de iniciar el sistema, es recomendable ejecutar la verificación:
```bash
python verify_system.py
```

### 2. Inicio del Sistema
Para iniciar el sistema:
```bash
python run_system.py
```

## Componentes Principales

### 1. Indicadores Técnicos

#### Uso Básico
```python
from indicators import TechnicalIndicators

# Inicializar
indicators = TechnicalIndicators(config)

# Calcular indicadores
df = indicators.calculate_all_indicators(df)

# Obtener señales
signals = indicators.get_trading_signals(df)
```

#### Ejemplo de Uso
```python
import pandas as pd
from indicators import TechnicalIndicators

# Cargar datos
df = pd.read_csv('data/btc_usdt_1h.csv')

# Inicializar indicadores
indicators = TechnicalIndicators(config)

# Calcular indicadores
df = indicators.calculate_all_indicators(df)

# Obtener señales
signals = indicators.get_trading_signals(df)

# Mostrar señales
print(signals)
```

### 2. Gestión de Riesgo

#### Uso Básico
```python
from risk_manager import RiskManager

# Inicializar
risk_manager = RiskManager(config)

# Calcular tamaño de posición
position_size = risk_manager.calculate_position_size(
    balance=1000,
    price=50000,
    atr=1000,
    signal_strength=0.8
)

# Calcular stop loss y take profit
stop_loss, take_profit = risk_manager.calculate_stop_loss_take_profit(
    price=50000,
    atr=1000,
    signal_strength=0.8
)
```

#### Ejemplo de Uso
```python
from risk_manager import RiskManager

# Inicializar
risk_manager = RiskManager(config)

# Validar trade
trade_valid = risk_manager.validate_trade(
    price=50000,
    position_size=0.1,
    signal_strength=0.8
)

if trade_valid:
    # Calcular niveles
    stop_loss, take_profit = risk_manager.calculate_stop_loss_take_profit(
        price=50000,
        atr=1000,
        signal_strength=0.8
    )
    
    print(f"Stop Loss: {stop_loss}")
    print(f"Take Profit: {take_profit}")
```

### 3. Modelo de Machine Learning

#### Uso Básico
```python
from ml_model import MLModel

# Inicializar
ml_model = MLModel(config)

# Preparar features
features = ml_model.prepare_features(df)

# Entrenar modelo
ml_model.train(features)

# Hacer predicciones
predictions = ml_model.predict(features)
```

#### Ejemplo de Uso
```python
from ml_model import MLModel
import pandas as pd

# Cargar datos
df = pd.read_csv('data/btc_usdt_1h.csv')

# Inicializar modelo
ml_model = MLModel(config)

# Preparar features
features = ml_model.prepare_features(df)

# Entrenar modelo
ml_model.train(features)

# Obtener métricas
metrics = ml_model.get_performance_metrics()
print(metrics)

# Hacer predicciones
predictions = ml_model.predict(features)
print(predictions)
```

### 4. Análisis de Mercado

#### Uso Básico
```python
from market_analysis import MarketAnalysis

# Inicializar
market_analysis = MarketAnalysis(config)

# Analizar condiciones de mercado
conditions = market_analysis.analyze_market_conditions('BTC/USDT')
```

#### Ejemplo de Uso
```python
from market_analysis import MarketAnalysis

# Inicializar
market_analysis = MarketAnalysis(config)

# Analizar mercado
conditions = market_analysis.analyze_market_conditions('BTC/USDT')

# Mostrar resultados
print("Condición de Mercado:", conditions['market_condition'])
print("Análisis Técnico:", conditions['technical'])
print("Análisis de Volumen:", conditions['volume'])
print("Sentimiento:", conditions['sentiment'])
```

### 5. Optimizador

#### Uso Básico
```python
from optimizer import BayesianOptimizer

# Inicializar
optimizer = BayesianOptimizer(config)

# Función de evaluación
def evaluation_function(params):
    # Implementar lógica de evaluación
    return score

# Ejecutar optimización
best_params = optimizer.optimize(evaluation_function, n_iterations=100)
```

#### Ejemplo de Uso
```python
from optimizer import BayesianOptimizer
from backtester import Backtester

# Inicializar
optimizer = BayesianOptimizer(config)
backtester = Backtester(config)

# Función de evaluación
def evaluation_function(params):
    # Ejecutar backtest con parámetros
    results = backtester.run_backtest(df, params)
    return results['metrics']['sharpe_ratio']

# Ejecutar optimización
best_params = optimizer.optimize(evaluation_function, n_iterations=100)

# Obtener reporte
report = optimizer.get_optimization_report()
print(report)
```

### 6. Backtester

#### Uso Básico
```python
from backtester import Backtester

# Inicializar
backtester = Backtester(config)

# Ejecutar backtest
results = backtester.run_backtest(df, config)
```

#### Ejemplo de Uso
```python
from backtester import Backtester
import pandas as pd

# Cargar datos
df = pd.read_csv('data/btc_usdt_1h.csv')

# Inicializar backtester
backtester = Backtester(config)

# Ejecutar backtest
results = backtester.run_backtest(df, config)

# Mostrar resultados
print("Total Trades:", results['trades'])
print("Win Rate:", results['metrics']['win_rate'])
print("Profit Factor:", results['metrics']['profit_factor'])
print("Sharpe Ratio:", results['metrics']['sharpe_ratio'])

# Generar gráficos
backtester.plot_results('backtest_results.png')
```

### 7. Monitor

#### Uso Básico
```python
from monitor import SystemMonitor

# Inicializar
monitor = SystemMonitor(config)

# Iniciar monitoreo
monitor.start_monitoring()
```

#### Ejemplo de Uso
```python
from monitor import SystemMonitor
import time

# Inicializar monitor
monitor = SystemMonitor(config)

# Iniciar monitoreo
monitor.start_monitoring()

# Esperar algunas métricas
time.sleep(60)

# Generar reporte
monitor.generate_report('monitoring_report.png')

# Detener monitoreo
monitor.close()
```

## Monitoreo y Alertas

### Alertas vía Telegram
El sistema envía alertas automáticas a través de Telegram para:
- Señales de trading
- Errores del sistema
- Métricas de rendimiento
- Resultados de optimización
- Reportes de backtesting

### Logs del Sistema
Los logs se guardan en:
- `trading_system.log`: Logs generales del sistema
- `system_verification.log`: Logs de verificación
- `backtest_results.json`: Resultados de backtesting
- `optimization_report.json`: Reportes de optimización

## Mantenimiento

### Limpieza de Datos
El sistema realiza limpieza automática de:
- Logs antiguos
- Datos de backtesting
- Resultados de optimización
- Métricas de monitoreo

### Actualización del Sistema
Para actualizar el sistema:
1. Hacer backup de la configuración
2. Actualizar el código
3. Verificar la instalación
4. Reiniciar el sistema

## Siguientes Pasos

1. Revisar los ejemplos prácticos en `docs/manuales/03_ejemplos.md`
2. Explorar la configuración avanzada en `docs/manuales/04_configuracion_avanzada.md`
3. Consultar las mejores prácticas en `docs/manuales/05_mejores_practicas.md`

## 1. Dashboard Principal

El dashboard es la interfaz central del sistema que integra todas las funcionalidades. Para iniciarlo:

```bash
streamlit run dashboard.py
```

### 1.1 Navegación
El dashboard está organizado en las siguientes secciones:
- **Dashboard**: Vista general del sistema
- **Análisis Técnico**: Indicadores y señales
- **Gestión de Riesgo**: Control de exposición y drawdown
- **Machine Learning**: Predicciones y estado del modelo
- **Optimización**: Ajuste de parámetros
- **Backtesting**: Pruebas históricas
- **Calculadora de Ganancias**: Simulación de estrategias
- **Monitoreo**: Estado del sistema

### 1.2 Dashboard Principal
Muestra:
- Rendimiento total y diario
- Operaciones totales y diarias
- Tasa de éxito
- Gráfico de rendimiento histórico
- Últimas operaciones

### 1.3 Análisis Técnico
Permite:
- Seleccionar par y timeframe
- Ver gráficos de velas con indicadores
- Analizar señales de trading
- Configurar indicadores técnicos

### 1.4 Gestión de Riesgo
Opciones:
- Ajustar tamaño máximo de posición
- Controlar drawdown
- Ver métricas de riesgo
- Monitorear exposición

### 1.5 Machine Learning
Funcionalidades:
- Ver estado del modelo
- Analizar predicciones
- Monitorear precisión
- Visualizar resultados

### 1.6 Optimización
Herramientas:
- Seleccionar método de optimización
- Elegir métricas a optimizar
- Ver resultados
- Aplicar parámetros optimizados

### 1.7 Backtesting
Capacidades:
- Seleccionar período de prueba
- Ver métricas de rendimiento
- Analizar equity curve
- Exportar resultados

### 1.8 Calculadora de Ganancias
Estrategias disponibles:
- Scalping
- Grid Trading
- Martingala
- DCA (Dollar Cost Averaging)

### 1.9 Monitoreo
Métricas:
- Uso de CPU y memoria
- Latencia
- Alertas del sistema
- Estado general

## 2. Calculadora de Ganancias

### 2.1 Scalping
```python
results = profit_calculator.calculate_scalping_profit(
    initial_capital=1000,
    trade_size=0.02,  # 2% del capital
    win_rate=0.6,     # 60% de operaciones ganadoras
    risk_reward=1.5,  # Ratio riesgo/beneficio
    trades_per_day=10,
    days=30
)
```

### 2.2 Grid Trading
```python
results = profit_calculator.calculate_grid_trading_profit(
    initial_capital=1000,
    grid_levels=10,
    grid_spacing=0.01,  # 1% entre niveles
    position_size=0.1,
    price_range=(30000, 35000),
    days=30
)
```

### 2.3 Martingala
```python
results = profit_calculator.calculate_martingale_profit(
    initial_capital=1000,
    base_bet=10,
    multiplier=2.0,
    max_steps=5,
    win_rate=0.5,
    trades_per_day=20,
    days=30
)
```

### 2.4 DCA (Dollar Cost Averaging)
```python
results = profit_calculator.calculate_dca_profit(
    initial_capital=1000,
    base_investment=100,
    dca_steps=5,
    dca_multiplier=1.5,
    price_decrease=0.05,  # 5% de disminución por paso
    days=30
)
```

## 3. Integración de Componentes

### 3.1 Flujo de Datos
1. Los datos de mercado se obtienen a través de la API
2. Se procesan con indicadores técnicos
3. El modelo de ML genera predicciones
4. El gestor de riesgo valida las operaciones
5. Las señales se envían al ejecutor
6. Los resultados se registran y monitorean

### 3.2 Monitoreo en Tiempo Real
- Métricas de rendimiento
- Estado del sistema
- Alertas y notificaciones
- Logs de operaciones

### 3.3 Gestión de Riesgo Integrada
- Control de exposición
- Stop loss dinámico
- Take profit adaptativo
- Diversificación de estrategias

## 4. Mejores Prácticas

### 4.1 Uso del Dashboard
- Monitorear regularmente el rendimiento
- Revisar alertas del sistema
- Ajustar parámetros según resultados
- Mantener un registro de cambios

### 4.2 Gestión de Riesgo
- No exceder límites de exposición
- Mantener diversificación
- Ajustar tamaño de posición
- Monitorear drawdown

### 4.3 Optimización
- Realizar backtesting antes de cambios
- Validar resultados en diferentes períodos
- Mantener registro de optimizaciones
- No sobre-optimizar

### 4.4 Monitoreo
- Revisar métricas diariamente
- Responder a alertas
- Mantener logs actualizados
- Realizar backups regulares

## 5. Solución de Problemas

### 5.1 Problemas Comunes
1. **Dashboard no inicia**
   - Verificar instalación de dependencias
   - Comprobar puerto disponible
   - Revisar logs de error

2. **Cálculos incorrectos**
   - Verificar datos de entrada
   - Comprobar configuración
   - Revisar logs de operaciones

3. **Problemas de rendimiento**
   - Optimizar cálculos
   - Reducir frecuencia de actualización
   - Ajustar parámetros de monitoreo

### 5.2 Contacto y Soporte
- Revisar documentación
- Consultar logs del sistema
- Contactar soporte técnico
- Reportar problemas

## 6. Siguientes Pasos

1. Revisar la configuración avanzada en `docs/manuales/04_configuracion_avanzada.md`
2. Consultar las mejores prácticas en `docs/manuales/05_mejores_practicas.md`
3. Explorar casos de uso en `docs/manuales/06_casos_uso.md`
4. Revisar ejemplos avanzados en `docs/manuales/07_ejemplos_avanzados.md`
5. Consultar la guía de solución de problemas en `docs/manuales/08_solucion_problemas.md` 