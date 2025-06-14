import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml
import logging
from datetime import datetime, timedelta

class ProfitCalculator:
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Inicializa el calculador de ganancias.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Carga la configuración desde el archivo YAML."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Error cargando configuración: {e}")
    
    def setup_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('profit_calculator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ProfitCalculator')
    
    def calculate_scalping_profit(
        self,
        initial_capital: float,
        trade_size: float,
        win_rate: float,
        risk_reward: float,
        trades_per_day: int,
        days: int
    ) -> Dict:
        """
        Calcula ganancias potenciales para estrategia de scalping.
        
        Args:
            initial_capital: Capital inicial
            trade_size: Tamaño de cada operación (% del capital)
            win_rate: Tasa de éxito (%)
            risk_reward: Ratio riesgo/beneficio
            trades_per_day: Número de operaciones por día
            days: Número de días a simular
        
        Returns:
            Dict con resultados del cálculo
        """
        try:
            daily_trades = trades_per_day
            total_trades = daily_trades * days
            
            # Calcular ganancia/pérdida por operación
            win_amount = trade_size * risk_reward
            loss_amount = trade_size
            
            # Calcular resultados
            wins = int(total_trades * win_rate)
            losses = total_trades - wins
            
            total_profit = (wins * win_amount) - (losses * loss_amount)
            final_capital = initial_capital * (1 + total_profit)
            
            return {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_profit': total_profit,
                'total_trades': total_trades,
                'winning_trades': wins,
                'losing_trades': losses,
                'win_rate': win_rate,
                'daily_profit': total_profit / days,
                'roi': (total_profit / initial_capital) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error en cálculo de scalping: {e}")
            raise
    
    def calculate_grid_trading_profit(
        self,
        initial_capital: float,
        grid_levels: int,
        grid_spacing: float,
        position_size: float,
        price_range: Tuple[float, float],
        days: int
    ) -> Dict:
        """
        Calcula ganancias potenciales para estrategia de grid trading.
        
        Args:
            initial_capital: Capital inicial
            grid_levels: Número de niveles de grid
            grid_spacing: Espaciado entre niveles (%)
            position_size: Tamaño de cada posición
            price_range: Rango de precios (min, max)
            days: Número de días a simular
        
        Returns:
            Dict con resultados del cálculo
        """
        try:
            min_price, max_price = price_range
            price_step = (max_price - min_price) / grid_levels
            
            # Calcular niveles de grid
            grid_prices = [min_price + (i * price_step) for i in range(grid_levels + 1)]
            
            # Simular operaciones
            total_profit = 0
            positions = []
            
            for i in range(len(grid_prices) - 1):
                buy_price = grid_prices[i]
                sell_price = grid_prices[i + 1]
                
                # Calcular ganancia por nivel
                profit = (sell_price - buy_price) * position_size
                total_profit += profit
                
                positions.append({
                    'level': i + 1,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'profit': profit
                })
            
            final_capital = initial_capital + total_profit
            
            return {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_profit': total_profit,
                'grid_levels': grid_levels,
                'positions': positions,
                'daily_profit': total_profit / days,
                'roi': (total_profit / initial_capital) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error en cálculo de grid trading: {e}")
            raise
    
    def calculate_martingale_profit(
        self,
        initial_capital: float,
        base_bet: float,
        multiplier: float,
        max_steps: int,
        win_rate: float,
        trades_per_day: int,
        days: int
    ) -> Dict:
        """
        Calcula ganancias potenciales para estrategia de martingala.
        
        Args:
            initial_capital: Capital inicial
            base_bet: Apuesta base
            multiplier: Multiplicador de apuesta
            max_steps: Máximo número de pasos
            win_rate: Tasa de éxito (%)
            trades_per_day: Número de operaciones por día
            days: Número de días a simular
        
        Returns:
            Dict con resultados del cálculo
        """
        try:
            daily_trades = trades_per_day
            total_trades = daily_trades * days
            
            # Simular operaciones
            current_bet = base_bet
            total_profit = 0
            consecutive_losses = 0
            
            for _ in range(total_trades):
                if np.random.random() < win_rate:
                    # Ganancia
                    total_profit += current_bet
                    current_bet = base_bet
                    consecutive_losses = 0
                else:
                    # Pérdida
                    total_profit -= current_bet
                    if consecutive_losses < max_steps:
                        current_bet *= multiplier
                    consecutive_losses += 1
            
            final_capital = initial_capital + total_profit
            
            return {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_profit': total_profit,
                'total_trades': total_trades,
                'max_consecutive_losses': consecutive_losses,
                'daily_profit': total_profit / days,
                'roi': (total_profit / initial_capital) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error en cálculo de martingala: {e}")
            raise
    
    def calculate_dca_profit(
        self,
        initial_capital: float,
        base_investment: float,
        dca_steps: int,
        dca_multiplier: float,
        price_decrease: float,
        days: int
    ) -> Dict:
        """
        Calcula ganancias potenciales para estrategia de DCA (Dollar Cost Averaging).
        
        Args:
            initial_capital: Capital inicial
            base_investment: Inversión base
            dca_steps: Número de pasos de DCA
            dca_multiplier: Multiplicador de inversión
            price_decrease: Disminución de precio por paso (%)
            days: Número de días a simular
        
        Returns:
            Dict con resultados del cálculo
        """
        try:
            total_investment = 0
            total_coins = 0
            current_price = 100  # Precio inicial arbitrario
            
            # Simular compras DCA
            for step in range(dca_steps):
                investment = base_investment * (dca_multiplier ** step)
                price = current_price * (1 - (price_decrease * step))
                
                coins = investment / price
                total_investment += investment
                total_coins += coins
            
            # Calcular valor final
            final_price = current_price * (1 - (price_decrease * (dca_steps - 1)))
            final_value = total_coins * final_price
            total_profit = final_value - total_investment
            
            return {
                'initial_capital': initial_capital,
                'total_investment': total_investment,
                'final_value': final_value,
                'total_profit': total_profit,
                'total_coins': total_coins,
                'average_entry': total_investment / total_coins,
                'roi': (total_profit / total_investment) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error en cálculo de DCA: {e}")
            raise
    
    def generate_report(self, results: Dict, strategy: str) -> str:
        """
        Genera un reporte formateado de los resultados.
        
        Args:
            results: Diccionario con resultados
            strategy: Nombre de la estrategia
        
        Returns:
            String con el reporte formateado
        """
        report = f"\n=== Reporte de {strategy} ===\n"
        
        for key, value in results.items():
            if isinstance(value, float):
                report += f"{key}: {value:.2f}\n"
            else:
                report += f"{key}: {value}\n"
        
        return report

def main():
    # Ejemplo de uso
    calculator = ProfitCalculator()
    
    # Calcular ganancias para diferentes estrategias
    scalping_results = calculator.calculate_scalping_profit(
        initial_capital=1000,
        trade_size=0.02,  # 2% del capital
        win_rate=0.6,     # 60% de éxito
        risk_reward=1.5,  # Ratio 1:1.5
        trades_per_day=10,
        days=30
    )
    
    grid_results = calculator.calculate_grid_trading_profit(
        initial_capital=1000,
        grid_levels=10,
        grid_spacing=0.01,  # 1% entre niveles
        position_size=0.1,  # 0.1 BTC por nivel
        price_range=(30000, 35000),
        days=30
    )
    
    martingale_results = calculator.calculate_martingale_profit(
        initial_capital=1000,
        base_bet=10,
        multiplier=2,
        max_steps=5,
        win_rate=0.5,
        trades_per_day=20,
        days=30
    )
    
    dca_results = calculator.calculate_dca_profit(
        initial_capital=1000,
        base_investment=100,
        dca_steps=5,
        dca_multiplier=1.5,
        price_decrease=0.05,  # 5% de disminución por paso
        days=30
    )
    
    # Generar reportes
    print(calculator.generate_report(scalping_results, "Scalping"))
    print(calculator.generate_report(grid_results, "Grid Trading"))
    print(calculator.generate_report(martingale_results, "Martingala"))
    print(calculator.generate_report(dca_results, "DCA"))

if __name__ == "__main__":
    main() 