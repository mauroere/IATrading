# Manual de Instalación del Sistema de Trading

## Requisitos Previos

### Sistema Operativo
- Windows 10/11
- Linux (Ubuntu 20.04 o superior)
- macOS 10.15 o superior

### Python
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Dependencias del Sistema
- Git
- Compilador C/C++ (para TA-Lib)
- Visual Studio Build Tools (Windows)

## Pasos de Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/IATrading.git
cd IATrading
```

### 2. Crear Entorno Virtual
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

#### 3.1 Instalar TA-Lib

**Windows:**
1. Descargar el instalador de TA-Lib desde [aquí](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
2. Instalar el archivo .whl correspondiente a tu versión de Python:
```bash
pip install TA_Lib‑0.4.24‑cp38‑cp38‑win_amd64.whl
```

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

**macOS:**
```bash
brew install ta-lib
```

#### 3.2 Instalar Dependencias de Python
```bash
pip install -r requirements.txt
```

### 4. Configuración del Sistema

#### 4.1 Crear Archivo de Configuración
Crear archivo `.env` en la raíz del proyecto:
```env
BINANCE_API_KEY=tu_api_key
BINANCE_API_SECRET=tu_api_secret
TELEGRAM_BOT_TOKEN=tu_bot_token
TELEGRAM_CHAT_ID=tu_chat_id
```

#### 4.2 Configurar Parámetros
Editar `config.yaml` con tus preferencias:
```yaml
trading_pairs:
  - "BTC/USDT"
  - "ETH/USDT"
  - "BNB/USDT"

timeframes:
  - "1h"
  - "4h"
  - "1d"

risk_management:
  max_position_size: 0.1
  max_daily_trades: 5
  max_drawdown: 0.05
  stop_loss_atr_multiplier: 2.0
  take_profit_atr_multiplier: 3.0

ml_model:
  retraining_interval: 24
  min_training_samples: 1000
  validation_split: 0.2
  feature_importance_threshold: 0.01

monitoring:
  metrics_interval: 60
  alert_thresholds:
    cpu_usage: 80
    memory_usage: 80
    disk_usage: 80
    network_latency: 1000
    error_rate: 0.01
```

### 5. Verificar la Instalación

Ejecutar el script de verificación:
```bash
python verify_system.py
```

Si todo está correcto, verás un mensaje de éxito y un reporte detallado en `verification_report.json`.

## Solución de Problemas Comunes

### Error al instalar TA-Lib
Si encuentras problemas al instalar TA-Lib, asegúrate de:
1. Tener instalado el compilador C/C++
2. Tener las variables de entorno correctamente configuradas
3. Usar la versión correcta del archivo .whl para tu versión de Python

### Error de Conexión con Binance
Si hay problemas de conexión:
1. Verificar que las API keys sean correctas
2. Comprobar la conexión a internet
3. Verificar que no haya restricciones de IP

### Error de Memoria
Si el sistema consume demasiada memoria:
1. Reducir el número de pares de trading
2. Aumentar el intervalo de monitoreo
3. Limitar el número de indicadores calculados

## Siguientes Pasos

Una vez completada la instalación, puedes:
1. Revisar el manual de uso en `docs/manuales/02_uso.md`
2. Consultar los ejemplos prácticos en `docs/manuales/03_ejemplos.md`
3. Explorar la configuración avanzada en `docs/manuales/04_configuracion_avanzada.md` 