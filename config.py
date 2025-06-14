import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading Configuration
TRADING_CONFIG = {
    'symbol': 'USDT/USDT',  # Trading pair
    'max_daily_trades': 10,  # Maximum trades per day
    'emergency_stop_loss': 0.15,  # 15% emergency stop loss
    'position_size': 0.05,  # 5% of capital per trade
    'min_profit_threshold': 0.002,  # 0.2% minimum profit threshold
    'max_position_size': 0.1,  # Maximum position size (10% of capital)
    'max_margin_usage': 0.8,  # Maximum margin usage (80% of capital)
    'max_position_exposure': 0.3,  # Maximum exposure per position (30% of capital)
    'max_daily_drawdown': 0.05,  # Maximum daily drawdown (5%)
    'max_volatility': 0.03,  # Maximum volatility threshold (3%)
    'atr_stop_loss_multiplier': 2.0,  # ATR multiplier for stop loss
    'atr_take_profit_multiplier': 3.0,  # ATR multiplier for take profit
}

# Technical Indicators Configuration
INDICATORS_CONFIG = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_period': 14,
    'bb_period': 20,
    'bb_std': 2,
    'stoch_k_period': 14,
    'stoch_d_period': 3,
    'ichimoku_tenkan': 9,
    'ichimoku_kijun': 26,
    'ichimoku_senkou_b': 52,
}

# API Configuration
API_CONFIG = {
    'binance_api_key': os.getenv('BINANCE_API_KEY'),
    'binance_api_secret': os.getenv('BINANCE_API_SECRET'),
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
}

# Database Configuration
DB_CONFIG = {
    'trades_db': 'trades.db',
    'logs_db': 'trading_logs.db',
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_file': 'trading_bot.log',
    'log_level': 'INFO',
}

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'models/',
    'training_data_path': 'data/training_data.csv',
    'prediction_threshold': 0.7,
    'retrain_interval': 24,  # Hours
    'min_training_samples': 1000,
    'feature_importance_threshold': 0.01,
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'port': 8501,
    'host': '0.0.0.0',
    'debug': False,
    'theme': 'dark',
    'refresh_interval': 5,  # Seconds
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_concurrent_trades': 3,
    'max_correlation': 0.7,
    'min_risk_reward_ratio': 2.0,
    'max_slippage': 0.001,
    'position_sizing_method': 'kelly',  # Options: 'fixed', 'kelly', 'optimal_f'
    'kelly_fraction': 0.5,  # Fraction of Kelly criterion to use
    'optimal_f_risk': 0.02,  # Risk per trade for optimal f
}

# Performance Monitoring Configuration
MONITORING_CONFIG = {
    'metrics_update_interval': 60,  # Seconds
    'alert_thresholds': {
        'drawdown': 0.05,
        'win_rate': 0.4,
        'profit_factor': 1.5,
        'sharpe_ratio': 1.0,
    },
    'performance_window': 30,  # Days
} 