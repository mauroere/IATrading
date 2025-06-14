#!/bin/bash

# Función para verificar si un proceso está corriendo
check_process() {
    if pgrep -f "$1" > /dev/null
    then
        return 0
    else
        return 1
    fi
}

# Función para reiniciar un proceso
restart_process() {
    local process_name=$1
    local screen_name=$2
    local command=$3

    echo "Reiniciando $process_name..."
    screen -X -S $screen_name quit
    sleep 2
    screen -dmS $screen_name $command
    sleep 5
}

# Bucle infinito de monitoreo
while true
do
    # Verificar el bot de trading
    if ! check_process "python main.py"
    then
        echo "$(date): Bot de trading no está corriendo. Reiniciando..."
        restart_process "trading_bot" "trading_bot" "python main.py"
    fi

    # Verificar el dashboard
    if ! check_process "streamlit run dashboard.py"
    then
        echo "$(date): Dashboard no está corriendo. Reiniciando..."
        restart_process "dashboard" "dashboard" "streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"
    fi

    # Verificar el uso de memoria
    memory_usage=$(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2 }')
    if (( $(echo "$memory_usage > 90" | bc -l) ))
    then
        echo "$(date): Uso de memoria alto ($memory_usage). Reiniciando servicios..."
        restart_process "trading_bot" "trading_bot" "python main.py"
        restart_process "dashboard" "dashboard" "streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"
    fi

    # Verificar el uso de CPU
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    if (( $(echo "$cpu_usage > 90" | bc -l) ))
    then
        echo "$(date): Uso de CPU alto ($cpu_usage%). Reiniciando servicios..."
        restart_process "trading_bot" "trading_bot" "python main.py"
        restart_process "dashboard" "dashboard" "streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"
    fi

    # Esperar 5 minutos antes de la siguiente verificación
    sleep 300
done 