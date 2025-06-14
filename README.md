# Automated Trading Bot with AI

This is an automated trading bot that uses machine learning and technical indicators to trade USDT on Binance Spot. The bot includes a real-time dashboard for monitoring trades and performance metrics.

## Features

- Automated trading using XGBoost model
- Technical indicators (RSI, MACD, ATR)
- Real-time monitoring dashboard
- Telegram notifications
- Emergency stop loss
- Daily trade limits
- Performance tracking and analytics
- SQLite database for trade history
- Logging system

## Prerequisites

- Python 3.8+
- Oracle Cloud Free Tier account
- Binance account with API access
- Telegram bot token

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Usage

1. Start the bot and dashboard:
```bash
python main.py
```

2. Access the dashboard at `http://localhost:8501`

## Oracle Cloud Setup

1. Create a new VM instance in Oracle Cloud Free Tier
2. Install required packages:
```bash
sudo apt update
sudo apt install python3-pip python3-venv
```

3. Set up the environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Create a systemd service:
```bash
sudo nano /etc/systemd/system/trading-bot.service
```

Add the following content:
```ini
[Unit]
Description=Trading Bot Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/path/to/trading-bot
Environment="PATH=/path/to/trading-bot/venv/bin"
ExecStart=/path/to/trading-bot/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

5. Start the service:
```bash
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

## Security Features

- Emergency stop loss (15% of capital)
- Daily trade limits (10 trades)
- Connection validation
- Error handling and logging
- Secure API key storage

## Monitoring

The dashboard provides:
- Real-time trade monitoring
- Performance metrics
- Balance history
- Trading statistics
- Recent logs

## Telegram Notifications

The bot sends notifications for:
- Trade executions
- Errors and warnings
- Emergency stops
- Daily summaries

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

Trading cryptocurrencies involves significant risk. This bot is for educational purposes only. Use at your own risk. 