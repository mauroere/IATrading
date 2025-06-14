import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import TRADING_CONFIG, API_CONFIG
from database import DatabaseHandler
from notifications import TelegramNotifier
from indicators import TechnicalIndicators
import xgboost as xgb
import joblib

class TradingBot:
    def __init__(self):
        self.config = TRADING_CONFIG
        self.exchange = ccxt.binance({
            'apiKey': API_CONFIG['binance_api_key'],
            'secret': API_CONFIG['binance_api_secret'],
            'enableRateLimit': True
        })
        self.db = DatabaseHandler()
        self.notifier = TelegramNotifier()
        self.indicators = TechnicalIndicators()
        self.model = self._load_model()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            filename='trading_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_model(self):
        try:
            return xgb.XGBClassifier()
            # model.load_model(MODEL_CONFIG['model_path'])
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None

    async def fetch_market_data(self, timeframe='1m', limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.config['symbol'],
                timeframe=timeframe,
                limit=limit
            )
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Error fetching market data: {str(e)}")
            await self.notifier.send_error_notification(f"Error fetching market data: {str(e)}")
            return None

    def prepare_features(self, data):
        signals = self.indicators.get_trading_signals(data)
        support_resistance = self.indicators.get_support_resistance(data)
        trend_strength = self.indicators.get_trend_strength(data)
        
        features = pd.concat([
            signals,
            support_resistance,
            trend_strength
        ], axis=1)
        
        return features.fillna(0)

    async def execute_trade(self, side, amount):
        try:
            order = self.exchange.create_order(
                symbol=self.config['symbol'],
                type='market',
                side=side,
                amount=amount
            )
            
            trade_data = {
                'symbol': self.config['symbol'],
                'side': side,
                'price': order['price'],
                'amount': amount,
                'total': order['cost'],
                'status': 'completed'
            }
            
            self.db.log_trade(**trade_data)
            await self.notifier.send_trade_notification(trade_data)
            
            return order
        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            logging.error(error_msg)
            await self.notifier.send_error_notification(error_msg)
            return None

    async def check_emergency_stop(self, current_balance, initial_balance):
        if (initial_balance - current_balance) / initial_balance > self.config['emergency_stop_loss']:
            await self.notifier.send_emergency_stop_notification(
                f"Emergency stop triggered. Loss exceeded {self.config['emergency_stop_loss']*100}%"
            )
            return True
        return False

    async def run(self):
        initial_balance = self.exchange.fetch_balance()['USDT']['free']
        daily_trades = 0
        last_trade_date = datetime.now().date()

        while True:
            try:
                # Reset daily trades counter if it's a new day
                current_date = datetime.now().date()
                if current_date > last_trade_date:
                    daily_trades = 0
                    last_trade_date = current_date
                    
                    # Send daily summary
                    summary = self.db.get_daily_stats().iloc[0]
                    await self.notifier.send_daily_summary(summary)

                # Check if we've reached daily trade limit
                if daily_trades >= self.config['max_daily_trades']:
                    logging.info("Daily trade limit reached")
                    continue

                # Fetch and process market data
                market_data = await self.fetch_market_data()
                if market_data is None:
                    continue

                # Prepare features and get prediction
                features = self.prepare_features(market_data)
                prediction = self.model.predict(features.iloc[-1:])[0]

                # Get current balance
                current_balance = self.exchange.fetch_balance()['USDT']['free']

                # Check emergency stop
                if await self.check_emergency_stop(current_balance, initial_balance):
                    break

                # Execute trade based on prediction
                if prediction == 1:  # Buy signal
                    amount = (current_balance * self.config['position_size']) / market_data['close'].iloc[-1]
                    await self.execute_trade('buy', amount)
                    daily_trades += 1
                elif prediction == -1:  # Sell signal
                    amount = (current_balance * self.config['position_size'])
                    await self.execute_trade('sell', amount)
                    daily_trades += 1

            except Exception as e:
                error_msg = f"Error in main loop: {str(e)}"
                logging.error(error_msg)
                await self.notifier.send_error_notification(error_msg)
                continue

    def stop(self):
        self.db.close()
        self.notifier.stop() 