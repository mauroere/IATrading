import os
from binance.client import Client
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_binance_connection():
    """Prueba la conexión con Binance usando las credenciales del .env"""
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        # Obtener credenciales
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("❌ Faltan credenciales de Binance en el archivo .env")
            return
        
        # Crear cliente de Binance
        client = Client(api_key, api_secret)
        
        # Probar conexión obteniendo el tiempo del servidor
        server_time = client.get_server_time()
        logger.info(f"✅ Conexión exitosa con Binance. Server Time: {server_time}")
        
        # Obtener balance de la cuenta
        account = client.get_account()
        balances = [b for b in account['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
        
        logger.info("\nBalances disponibles:")
        for balance in balances:
            logger.info(f"{balance['asset']}: Libre={balance['free']}, Bloqueado={balance['locked']}")
            
    except Exception as e:
        logger.error(f"❌ Error al conectar con Binance: {str(e)}")

if __name__ == "__main__":
    test_binance_connection() 