# Guía de Solución de Problemas

## 1. Problemas de Instalación

### Error al instalar TA-Lib
**Síntoma**: Error durante la instalación de TA-Lib con pip.

**Solución**:
1. Windows:
   ```bash
   # Descargar e instalar el binario de TA-Lib
   pip install --index-url=https://pypi.org/simple/ TA-Lib
   ```

2. Linux:
   ```bash
   # Instalar dependencias
   sudo apt-get update
   sudo apt-get install ta-lib
   
   # Instalar Python wrapper
   pip install TA-Lib
   ```

3. macOS:
   ```bash
   # Instalar con Homebrew
   brew install ta-lib
   
   # Instalar Python wrapper
   pip install TA-Lib
   ```

### Error de dependencias faltantes
**Síntoma**: Error al instalar paquetes Python.

**Solución**:
1. Actualizar pip:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Instalar dependencias del sistema:
   ```bash
   # Windows
   pip install -r requirements.txt
   
   # Linux/macOS
   sudo pip install -r requirements.txt
   ```

## 2. Problemas de Conexión

### Error de conexión con Binance
**Síntoma**: No se puede conectar a la API de Binance.

**Solución**:
1. Verificar credenciales:
   ```python
   # Verificar en config.yaml
   exchange:
     name: "binance"
     api_key: "tu_api_key"
     api_secret: "tu_api_secret"
   ```

2. Verificar límites de API:
   - Revisar límites de peso en Binance
   - Implementar rate limiting
   - Usar IP permitida

3. Verificar red:
   ```bash
   # Probar conexión
   ping api.binance.com
   ```

### Error de timeout
**Síntoma**: Timeout al realizar operaciones.

**Solución**:
1. Ajustar timeouts:
   ```python
   # En config.yaml
   exchange:
     timeout: 30000  # 30 segundos
     retry_on_timeout: true
     max_retries: 3
   ```

2. Implementar reintentos:
   ```python
   def make_request(self, func, *args, **kwargs):
       for attempt in range(self.max_retries):
           try:
               return func(*args, **kwargs)
           except TimeoutError:
               if attempt == self.max_retries - 1:
                   raise
               time.sleep(1)
   ```

## 3. Problemas de Rendimiento

### Alto uso de CPU
**Síntoma**: El sistema consume demasiados recursos.

**Solución**:
1. Optimizar cálculos:
   ```python
   # Usar numpy para cálculos vectorizados
   import numpy as np
   
   def calculate_indicators(df):
       # Vectorizar cálculos
       df['sma'] = np.convolve(df['close'], np.ones(20)/20, mode='valid')
   ```

2. Reducir frecuencia de actualización:
   ```python
   # En config.yaml
   system:
     update_interval: 60  # segundos
     max_concurrent_tasks: 4
   ```

3. Implementar caché:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def calculate_expensive_metric(data):
       # Cálculos costosos
       pass
   ```

### Alto uso de memoria
**Síntoma**: El sistema consume demasiada memoria.

**Solución**:
1. Limitar historial:
   ```python
   # En config.yaml
   data:
     max_history_size: 1000
     cleanup_interval: 3600  # 1 hora
   ```

2. Liberar memoria:
   ```python
   def cleanup_old_data(self):
       # Eliminar datos antiguos
       self.data = self.data.tail(self.max_history_size)
       gc.collect()
   ```

3. Usar generadores:
   ```python
   def process_data(self):
       for chunk in pd.read_csv('data.csv', chunksize=1000):
           # Procesar por chunks
           yield process_chunk(chunk)
   ```

## 4. Problemas de Trading

### Errores de ejecución de órdenes
**Síntoma**: Las órdenes no se ejecutan correctamente.

**Solución**:
1. Verificar balance:
   ```python
   def check_balance(self, symbol, amount):
       balance = self.exchange.fetch_balance()
       if balance[symbol]['free'] < amount:
           raise InsufficientFunds(f"Fondos insuficientes para {symbol}")
   ```

2. Verificar límites de orden:
   ```python
   def validate_order(self, symbol, amount, price):
       limits = self.exchange.market(symbol)['limits']
       if amount < limits['amount']['min']:
           raise InvalidOrder(f"Cantidad mínima: {limits['amount']['min']}")
   ```

3. Implementar reintentos:
   ```python
   def place_order(self, symbol, type, side, amount, price=None):
       for attempt in range(3):
           try:
               return self.exchange.create_order(
                   symbol, type, side, amount, price
               )
           except Exception as e:
               if attempt == 2:
                   raise
               time.sleep(1)
   ```

### Errores de cálculo de indicadores
**Síntoma**: Los indicadores técnicos dan resultados incorrectos.

**Solución**:
1. Verificar datos de entrada:
   ```python
   def validate_data(self, df):
       # Verificar valores nulos
       if df.isnull().any().any():
           raise ValueError("Datos contienen valores nulos")
       
       # Verificar orden temporal
       if not df.index.is_monotonic_increasing:
           raise ValueError("Datos no están ordenados temporalmente")
   ```

2. Implementar validación de resultados:
   ```python
   def validate_indicator(self, indicator, value):
       if not isinstance(value, (int, float)):
           raise ValueError(f"Valor inválido para {indicator}")
       if np.isnan(value) or np.isinf(value):
           raise ValueError(f"Valor no numérico para {indicator}")
   ```

3. Usar bibliotecas probadas:
   ```python
   import talib
   
   def calculate_rsi(self, data):
       return talib.RSI(data, timeperiod=14)
   ```

## 5. Problemas de Optimización

### Errores en optimización de parámetros
**Síntoma**: La optimización no converge o da resultados incorrectos.

**Solución**:
1. Verificar espacio de parámetros:
   ```python
   def validate_parameter_space(self, params):
       for param, (min_val, max_val) in params.items():
           if min_val >= max_val:
               raise ValueError(f"Rango inválido para {param}")
   ```

2. Implementar validación de resultados:
   ```python
   def validate_optimization_result(self, result):
       if result['score'] < self.min_score:
           raise ValueError("Score de optimización muy bajo")
       if result['iterations'] < self.min_iterations:
           raise ValueError("Muy pocas iteraciones")
   ```

3. Usar múltiples métricas:
   ```python
   def evaluate_strategy(self, params):
       metrics = {
           'sharpe': self.calculate_sharpe(),
           'sortino': self.calculate_sortino(),
           'drawdown': self.calculate_drawdown()
       }
       return self.combine_metrics(metrics)
   ```

### Errores de backtesting
**Síntoma**: El backtesting da resultados inconsistentes.

**Solución**:
1. Verificar datos históricos:
   ```python
   def validate_historical_data(self, data):
       # Verificar gaps
       gaps = data.index.to_series().diff() > pd.Timedelta('1h')
       if gaps.any():
           raise ValueError("Datos históricos contienen gaps")
   ```

2. Implementar validación de trades:
   ```python
   def validate_trade(self, trade):
       if trade['entry_price'] <= 0:
           raise ValueError("Precio de entrada inválido")
       if trade['exit_price'] <= 0:
           raise ValueError("Precio de salida inválido")
   ```

3. Usar walk-forward analysis:
   ```python
   def walk_forward_analysis(self, data):
       results = []
       for i in range(0, len(data), self.step_size):
           train = data[i:i+self.train_size]
           test = data[i+self.train_size:i+self.train_size+self.test_size]
           results.append(self.evaluate_period(train, test))
       return self.aggregate_results(results)
   ```

## 6. Problemas de Monitoreo

### Errores en el sistema de alertas
**Síntoma**: Las alertas no se envían o son incorrectas.

**Solución**:
1. Verificar configuración de alertas:
   ```python
   def validate_alert_config(self):
       if not self.telegram_token:
           raise ValueError("Token de Telegram no configurado")
       if not self.telegram_chat_id:
           raise ValueError("Chat ID de Telegram no configurado")
   ```

2. Implementar reintentos:
   ```python
   def send_alert(self, message, severity):
       for attempt in range(3):
           try:
               self.telegram.send_message(
                   chat_id=self.chat_id,
                   text=f"{severity}: {message}"
               )
               break
           except Exception as e:
               if attempt == 2:
                   self.log_error(f"Error enviando alerta: {e}")
   ```

3. Logging de alertas:
   ```python
   def log_alert(self, message, severity):
       self.logger.info(f"Alerta {severity}: {message}")
       self.alert_history.append({
           'timestamp': pd.Timestamp.now(),
           'message': message,
           'severity': severity
       })
   ```

### Errores en el sistema de logging
**Síntoma**: Los logs no se generan o son incompletos.

**Solución**:
1. Configurar logging:
   ```python
   def setup_logging(self):
       logging.basicConfig(
           level=logging.INFO,
           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
           handlers=[
               logging.FileHandler('trading.log'),
               logging.StreamHandler()
           ]
       )
   ```

2. Implementar rotación de logs:
   ```python
   from logging.handlers import RotatingFileHandler
   
   def setup_rotating_logs(self):
       handler = RotatingFileHandler(
           'trading.log',
           maxBytes=1024*1024,  # 1MB
           backupCount=5
       )
       logging.getLogger().addHandler(handler)
   ```

3. Logging estructurado:
   ```python
   def log_trade(self, trade):
       self.logger.info({
           'event': 'trade',
           'symbol': trade['symbol'],
           'side': trade['side'],
           'price': trade['price'],
           'amount': trade['amount']
       })
   ```

## 7. Problemas de Seguridad

### Errores de autenticación
**Síntoma**: Problemas con las credenciales de API.

**Solución**:
1. Verificar credenciales:
   ```python
   def validate_credentials(self):
       try:
           self.exchange.fetch_balance()
       except Exception as e:
           raise AuthenticationError(f"Error de autenticación: {e}")
   ```

2. Implementar rotación de claves:
   ```python
   def rotate_api_keys(self):
       # Generar nuevas claves
       new_key = self.generate_api_key()
       
       # Actualizar configuración
       self.update_config(new_key)
       
       # Verificar nueva clave
       self.validate_credentials()
   ```

3. Seguridad de claves:
   ```python
   def secure_api_key(self, key):
       # Encriptar clave
       encrypted = self.encrypt(key)
       
       # Guardar de forma segura
       self.save_encrypted_key(encrypted)
   ```

### Errores de permisos
**Síntoma**: Problemas con los permisos de archivos o directorios.

**Solución**:
1. Verificar permisos:
   ```python
   def check_permissions(self):
       required_dirs = ['data', 'logs', 'config']
       for dir in required_dirs:
           if not os.access(dir, os.W_OK):
               raise PermissionError(f"Sin permisos de escritura en {dir}")
   ```

2. Establecer permisos:
   ```python
   def set_permissions(self):
       for dir in ['data', 'logs', 'config']:
           os.chmod(dir, 0o755)
   ```

3. Verificar usuario:
   ```python
   def check_user(self):
       if os.geteuid() == 0:
           raise SecurityError("No ejecutar como root")
   ```

## 8. Problemas de Red

### Errores de conexión
**Síntoma**: Problemas de conectividad con el exchange.

**Solución**:
1. Verificar conexión:
   ```python
   def check_connection(self):
       try:
           self.exchange.fetch_time()
       except Exception as e:
           raise ConnectionError(f"Error de conexión: {e}")
   ```

2. Implementar reconexión:
   ```python
   def reconnect(self):
       for attempt in range(3):
           try:
               self.exchange.load_markets()
               break
           except Exception as e:
               if attempt == 2:
                   raise
               time.sleep(5)
   ```

3. Monitorear latencia:
   ```python
   def monitor_latency(self):
       start = time.time()
       self.exchange.fetch_time()
       latency = time.time() - start
       
       if latency > self.max_latency:
           self.log_warning(f"Alta latencia: {latency}s")
   ```

### Errores de proxy
**Síntoma**: Problemas al usar proxy.

**Solución**:
1. Configurar proxy:
   ```python
   def setup_proxy(self):
       self.exchange.proxies = {
           'http': 'http://proxy:port',
           'https': 'https://proxy:port'
       }
   ```

2. Verificar proxy:
   ```python
   def verify_proxy(self):
       try:
           requests.get('https://api.binance.com', proxies=self.proxies)
       except Exception as e:
           raise ProxyError(f"Error de proxy: {e}")
   ```

3. Rotar proxies:
   ```python
   def rotate_proxy(self):
       self.current_proxy = next(self.proxy_pool)
       self.setup_proxy()
   ```

## Siguientes Pasos

1. Revisar las mejores prácticas en `docs/manuales/05_mejores_practicas.md`
2. Explorar casos de uso en `docs/manuales/06_casos_uso.md`
3. Consultar ejemplos avanzados en `docs/manuales/07_ejemplos_avanzados.md` 