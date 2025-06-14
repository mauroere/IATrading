import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_CONFIG

class DatabaseHandler:
    def __init__(self):
        self.trades_conn = sqlite3.connect(DB_CONFIG['trades_db'])
        self.logs_conn = sqlite3.connect(DB_CONFIG['logs_db'])
        self._create_tables()

    def _create_tables(self):
        # Create trades table
        self.trades_conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                side TEXT,
                price REAL,
                amount REAL,
                total REAL,
                profit_loss REAL,
                status TEXT
            )
        ''')

        # Create logs table
        self.logs_conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                level TEXT,
                message TEXT
            )
        ''')

        self.trades_conn.commit()
        self.logs_conn.commit()

    def log_trade(self, symbol, side, price, amount, total, profit_loss=None, status='completed'):
        query = '''
            INSERT INTO trades (timestamp, symbol, side, price, amount, total, profit_loss, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        self.trades_conn.execute(query, (
            datetime.now(),
            symbol,
            side,
            price,
            amount,
            total,
            profit_loss,
            status
        ))
        self.trades_conn.commit()

    def log_event(self, level, message):
        query = '''
            INSERT INTO logs (timestamp, level, message)
            VALUES (?, ?, ?)
        '''
        self.logs_conn.execute(query, (datetime.now(), level, message))
        self.logs_conn.commit()

    def get_trades(self, limit=100):
        query = f'''
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT {limit}
        '''
        return pd.read_sql_query(query, self.trades_conn)

    def get_logs(self, limit=100):
        query = f'''
            SELECT * FROM logs
            ORDER BY timestamp DESC
            LIMIT {limit}
        '''
        return pd.read_sql_query(query, self.logs_conn)

    def get_daily_stats(self):
        query = '''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_trades,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(profit_loss) as total_profit_loss
            FROM trades
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        '''
        return pd.read_sql_query(query, self.trades_conn)

    def close(self):
        self.trades_conn.close()
        self.logs_conn.close() 