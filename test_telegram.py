import os
from telegram import Bot
import asyncio

async def test_telegram_connection():
    """Prueba la conexión con Telegram y envía un mensaje de prueba."""
    try:
        # Credenciales de Telegram
        bot_token = "7287954720:AAHOGOg8HySy6XExLxsBEWSdnQ9GcuPRWUI"
        chat_id = "7955937529"
        
        # Crear instancia del bot
        bot = Bot(token=bot_token)
        
        # Enviar mensaje de prueba
        message = "🔄 Prueba de conexión del Sistema de Trading\n\n✅ Conexión exitosa con Telegram"
        await bot.send_message(chat_id=chat_id, text=message)
        
        print("✅ Mensaje de prueba enviado exitosamente")
        
    except Exception as e:
        print(f"❌ Error al conectar con Telegram: {str(e)}")

if __name__ == "__main__":
    # Ejecutar la prueba
    asyncio.run(test_telegram_connection()) 