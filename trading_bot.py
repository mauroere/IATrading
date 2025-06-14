import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import TRADING_CONFIG, API_CONFIG, INDICATORS_CONFIG
from database import DatabaseHandler
from notifications import TelegramNotifier
from indicators import TechnicalIndicators
import xgboost as xgb
import joblib
import asyncio
from typing import Optional, Dict, Any, List
from risk_manager import RiskManager
from ml_model import MLModel

class TradingBot:
    def __init__(self):
        """Initialize the trading bot with configuration and components."""
        self.config = TRADING_CONFIG
        self.api_config = API_CONFIG
        self.indicators_config = INDICATORS_CONFIG
        self.exchange = ccxt.binance({
            'apiKey': API_CONFIG['binance_api_key'],
            'secret': API_CONFIG['binance_api_secret'],
            'enableRateLimit': True
        })
        self.db = DatabaseHandler()
        self.notifier = TelegramNotifier()
        self.indicators = TechnicalIndicators(INDICATORS_CONFIG)
        self.model = self._load_model()
        self.risk_manager = RiskManager()
        self.ml_model = MLModel()
        self.setup_logging()
        self.stop_event = asyncio.Event()

    def setup_logging(self):
        """Configure logging for the trading bot."""
        logging.basicConfig(
            filename='trading_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        try:
            return xgb.XGBClassifier()
            # model.load_model(MODEL_CONFIG['model_path'])
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None

    async def fetch_market_data(self, timeframe='1m', limit=100) -> Optional[pd.DataFrame]:
        try:
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv,
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

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            signals = self.indicators.get_trading_signals(data)
            support_resistance = self.indicators.get_support_resistance(data)
            trend_strength = self.indicators.get_trend_strength(data)
            
            features = pd.concat([
                signals,
                support_resistance,
                trend_strength
            ], axis=1)
            
            return features.fillna(0)
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    async def execute_trade(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """
        Execute a trade on the exchange.
        
        Args:
            symbol (str): Trading pair symbol
            side (str): 'buy' or 'sell'
            amount (float): Amount to trade
            price (float): Price to trade at
            
        Returns:
            Dict: Trade execution details
        """
        try:
            # Check risk limits
            if not self.risk_manager.check_trade_risk(symbol, side, amount, price):
                self.logger.warning(f"Trade rejected by risk manager: {symbol} {side} {amount} @ {price}")
                return None
                
            # Execute trade
            order = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
            
            # Log trade
            self.db.log_trade({
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'order_id': order['id'],
                'timestamp': datetime.now().isoformat()
            })
            
            # Send notification
            await self.notifier.send_trade_notification(order)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            await self.notifier.send_error_notification(f"Trade execution error: {str(e)}")
            return None

    async def analyze_market(self, symbol: str) -> Dict:
        """
        Analyze market conditions for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict: Market analysis results
        """
        try:
            # Get market data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate indicators
            df = self.indicators.calculate_all_indicators(df)
            
            # Get trading signals
            signals, signal_strength = self.indicators.get_trading_signals(df)
            
            # Get ML prediction
            features = self.ml_model.prepare_features(df)
            prediction = self.ml_model.predict(features)
            
            return {
                'signals': signals,
                'signal_strength': signal_strength,
                'ml_prediction': prediction,
                'indicators': df
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            return None

    async def check_emergency_stop(self, current_balance: float, initial_balance: float) -> bool:
        if (initial_balance - current_balance) / initial_balance > self.config['emergency_stop_loss']:
            await self.notifier.send_emergency_stop_notification(
                f"Emergency stop triggered. Loss exceeded {self.config['emergency_stop_loss']*100}%"
            )
            return True
        return False

    async def run(self):
        """Main trading loop."""
        try:
            initial_balance = await asyncio.to_thread(
                lambda: self.exchange.fetch_balance()['USDT']['free']
            )
            daily_trades = 0
            last_trade_date = datetime.now().date()

            while not self.stop_event.is_set():
                try:
                    # Reset daily trades counter if it's a new day
                    current_date = datetime.now().date()
                    if current_date > last_trade_date:
                        daily_trades = 0
                        last_trade_date = current_date
                        
                        # Send daily summary
                        summary = await asyncio.to_thread(self.db.get_daily_stats)
                        await self.notifier.send_daily_summary(summary.iloc[0])

                    # Check if we've reached daily trade limit
                    if daily_trades >= self.config['max_daily_trades']:
                        logging.info("Daily trade limit reached")
                        await asyncio.sleep(60)
                        continue

                    # Fetch and process market data
                    market_data = await self.fetch_market_data()
                    if market_data is None:
                        await asyncio.sleep(60)
                        continue

                    # Prepare features and get prediction
                    features = self.prepare_features(market_data)
                    if features.empty:
                        await asyncio.sleep(60)
                        continue

                    prediction = self.model.predict(features.iloc[-1:])[0]

                    # Get current balance
                    current_balance = await asyncio.to_thread(
                        lambda: self.exchange.fetch_balance()['USDT']['free']
                    )

                    # Check emergency stop
                    if await self.check_emergency_stop(current_balance, initial_balance):
                        break

                    # Execute trade based on prediction
                    if prediction == 1:  # Buy signal
                        amount = (current_balance * self.config['position_size']) / market_data['close'].iloc[-1]
                        await self.execute_trade(self.config['symbol'], 'buy', amount, market_data['close'].iloc[-1])
                        daily_trades += 1
                    elif prediction == -1:  # Sell signal
                        amount = (current_balance * self.config['position_size'])
                        await self.execute_trade(self.config['symbol'], 'sell', amount, market_data['close'].iloc[-1])
                        daily_trades += 1

                    await asyncio.sleep(60)  # Wait before next iteration

                except Exception as e:
                    error_msg = f"Error in main loop: {str(e)}"
                    logging.error(error_msg)
                    await self.notifier.send_error_notification(error_msg)
                    await asyncio.sleep(60)

        except Exception as e:
            error_msg = f"Fatal error in trading bot: {str(e)}"
            logging.error(error_msg)
            await self.notifier.send_error_notification(error_msg)
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading bot gracefully"""
        self.stop_event.set()
        self.db.close()
        await self.notifier.stop()

def main():
    """Main entry point for the trading bot."""
    bot = TradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        asyncio.run(bot.stop())
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        asyncio.run(bot.stop())
        
if __name__ == "__main__":
    main() 