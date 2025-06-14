import telegram
from config import API_CONFIG
import logging

class TelegramNotifier:
    def __init__(self):
        self.bot_token = API_CONFIG['telegram_bot_token']
        self.chat_id = API_CONFIG['telegram_chat_id']
        self.bot = telegram.Bot(token=self.bot_token)
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

    async def send_message(self, message):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logging.error(f"Error sending Telegram message: {str(e)}")

    async def send_trade_notification(self, trade_data):
        message = (
            f"🔔 Trade Notification\n"
            f"Symbol: {trade_data['symbol']}\n"
            f"Action: {trade_data['side']}\n"
            f"Price: {trade_data['price']}\n"
            f"Amount: {trade_data['amount']}\n"
            f"Total: {trade_data['total']}\n"
            f"Status: {trade_data['status']}"
        )
        await self.send_message(message)

    async def send_error_notification(self, error_message):
        message = f"⚠️ Error Alert\n{error_message}"
        await self.send_message(message)

    async def send_emergency_stop_notification(self, reason):
        message = f"🚨 EMERGENCY STOP\nReason: {reason}"
        await self.send_message(message)

    async def send_daily_summary(self, summary_data):
        message = (
            f"📊 Daily Trading Summary\n"
            f"Total Trades: {summary_data['total_trades']}\n"
            f"Winning Trades: {summary_data['winning_trades']}\n"
            f"Losing Trades: {summary_data['losing_trades']}\n"
            f"Total P/L: {summary_data['total_profit_loss']:.2f} USDT"
        )
        await self.send_message(message)

    def start(self):
        pass  # No longer needed

    def stop(self):
        pass  # No longer needed 