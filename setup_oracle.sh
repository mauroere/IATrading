#!/bin/bash

# Actualizar el sistema
sudo apt update
sudo apt upgrade -y

# Instalar dependencias
sudo apt install -y python3-pip python3-venv screen bc

# Crear directorio del proyecto
mkdir -p ~/trading-bot
cd ~/trading-bot

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias de Python
pip install -r requirements.txt

# Dar permisos de ejecución a los scripts
chmod +x start_bot.sh
chmod +x monitor_bot.sh

# Configurar el servicio systemd
sudo cp trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Iniciar el monitor en segundo plano
screen -dmS monitor ./monitor_bot.sh

echo "Instalación completada. El bot está corriendo y monitoreado."
echo "Puedes verificar el estado con: sudo systemctl status trading-bot"
echo "Para ver los logs: sudo journalctl -u trading-bot -f" 