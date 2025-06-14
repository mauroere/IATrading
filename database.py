import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_CONFIG
import threading
from contextlib import contextmanager
import queue
import logging

class DatabaseHandler:
    def __init__(self, max_connections=5):
        self.trades_db_path = DB_CONFIG['trades_db']
        self.logs_db_path = DB_CONFIG['logs_db']
        self.trades_pool = queue.Queue(maxsize=max_connections)
        self.logs_pool = queue.Queue(maxsize=max_connections)
        self._initialize_pools()
        self._create_tables()
        self.logger = logging.getLogger(__name__)

    def _initialize_pools(self):
        """Initialize connection pools"""
        for _ in range(self.trades_pool.maxsize):
            conn = sqlite3.connect(self.trades_db_path, check_same_thread=False)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=5000')
            self.trades_pool.put(conn)
        
        for _ in range(self.logs_pool.maxsize):
            conn = sqlite3.connect(self.logs_db_path, check_same_thread=False)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=5000')
            self.logs_pool.put(conn)

    @contextmanager
    def _get_trades_conn(self):
        """Thread-safe connection getter for trades database"""
        conn = self.trades_pool.get()
        try:
            yield conn
        finally:
            self.trades_pool.put(conn)

    @contextmanager
    def _get_logs_conn(self):
        """Thread-safe connection getter for logs database"""
        conn = self.logs_pool.get()
        try:
            yield conn
        finally:
            self.logs_pool.put(conn)

    def _create_tables(self):
        """Create necessary tables with proper error handling"""
        try:
            with self._get_trades_conn() as conn:
                conn.execute('''
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
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error creating trades table: {str(e)}")
            raise

        try:
            with self._get_logs_conn() as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        level TEXT,
                        message TEXT
                    )
                ''')
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error creating logs table: {str(e)}")
            raise

    def log_trade(self, symbol, side, price, amount, total, profit_loss=None, status='completed'):
        """Log a trade with proper transaction handling"""
        query = '''
            INSERT INTO trades (timestamp, symbol, side, price, amount, total, profit_loss, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        try:
            with self._get_trades_conn() as conn:
                with conn:  # This creates a transaction
                    conn.execute(query, (
                        datetime.now(),
                        symbol,
                        side,
                        price,
                        amount,
                        total,
                        profit_loss,
                        status
                    ))
        except Exception as e:
            self.logger.error(f"Error logging trade: {str(e)}")
            raise

    def log_event(self, level, message):
        """Log an event with proper transaction handling"""
        query = '''
            INSERT INTO logs (timestamp, level, message)
            VALUES (?, ?, ?)
        '''
        try:
            with self._get_logs_conn() as conn:
                with conn:  # This creates a transaction
                    conn.execute(query, (datetime.now(), level, message))
        except Exception as e:
            self.logger.error(f"Error logging event: {str(e)}")
            raise

    def get_trades(self, limit=100):
        """Get trades with proper error handling"""
        query = f'''
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT {limit}
        '''
        try:
            with self._get_trades_conn() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            self.logger.error(f"Error getting trades: {str(e)}")
            return pd.DataFrame()

    def get_logs(self, limit=100):
        """Get logs with proper error handling"""
        query = f'''
            SELECT * FROM logs
            ORDER BY timestamp DESC
            LIMIT {limit}
        '''
        try:
            with self._get_logs_conn() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            self.logger.error(f"Error getting logs: {str(e)}")
            return pd.DataFrame()

    def get_daily_stats(self):
        """Get daily statistics with proper error handling"""
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
            LIMIT 1
        '''
        try:
            with self._get_trades_conn() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            self.logger.error(f"Error getting daily stats: {str(e)}")
            return pd.DataFrame()

    def close(self):
        """Close all database connections"""
        while not self.trades_pool.empty():
            conn = self.trades_pool.get()
            conn.close()
        
        while not self.logs_pool.empty():
            conn = self.logs_pool.get()
            conn.close() 