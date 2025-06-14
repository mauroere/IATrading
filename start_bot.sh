#!/bin/bash

# Activar el entorno virtual
source venv/bin/activate

# Crear directorios necesarios si no existen
mkdir -p logs
mkdir -p data
mkdir -p models

# Iniciar el bot en modo daemon usando screen
screen -dmS trading_bot python main.py

# Verificar que el bot está corriendo
sleep 5
if pgrep -f "python main.py" > /dev/null
then
    echo "Trading bot iniciado correctamente"
else
    echo "Error al iniciar el trading bot"
    exit 1
fi

# Iniciar el dashboard en modo daemon
screen -dmS dashboard streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0

# Verificar que el dashboard está corriendo
sleep 5
if pgrep -f "streamlit run dashboard.py" > /dev/null
then
    echo "Dashboard iniciado correctamente"
else
    echo "Error al iniciar el dashboard"
    exit 1
fi 