import asyncio
import logging
from trading_bot import TradingBot
from dashboard import Dashboard
import streamlit as st
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_dashboard():
    dashboard = Dashboard()
    dashboard.run()

def run_bot():
    bot = TradingBot()
    asyncio.run(bot.run())

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )

    # Check if required environment variables are set
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return

    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    # Run the dashboard in the main thread
    run_dashboard()

if __name__ == "__main__":
    main() 