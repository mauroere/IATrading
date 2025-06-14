import streamlit as st
import json
import os
from config import TRADING_CONFIG, INDICATORS_CONFIG, RISK_CONFIG, MODEL_CONFIG
import yaml
from dotenv import load_dotenv

class DashboardConfig:
    def __init__(self):
        self.config_file = 'config.yaml'
        self.env_file = '.env'
        self.load_config()
        self.load_env()

    def load_env(self):
        """Cargar variables de entorno"""
        load_dotenv()
        self.api_keys = {
            'binance_api_key': os.getenv('BINANCE_API_KEY', ''),
            'binance_api_secret': os.getenv('BINANCE_API_SECRET', ''),
            'telegram_token': os.getenv('TELEGRAM_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
        }

    def save_env(self, api_keys):
        """Guardar variables de entorno"""
        with open(self.env_file, 'w') as f:
            for key, value in api_keys.items():
                f.write(f"{key.upper()}={value}\n")
        self.api_keys = api_keys

    def load_config(self):
        """Cargar configuración desde archivo YAML"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'trading': TRADING_CONFIG,
                'indicators': INDICATORS_CONFIG,
                'risk': RISK_CONFIG,
                'model': MODEL_CONFIG
            }
            self.save_config()

    def save_config(self):
        """Guardar configuración en archivo YAML"""
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    def render_api_config(self):
        """Renderizar sección de configuración de APIs"""
        st.subheader("Configuración de APIs")
        
        # Binance API
        st.write("### Binance API")
        binance_api_key = st.text_input(
            "Binance API Key",
            value=self.api_keys['binance_api_key'],
            type="password"
        )
        binance_api_secret = st.text_input(
            "Binance API Secret",
            value=self.api_keys['binance_api_secret'],
            type="password"
        )

        # Telegram API
        st.write("### Telegram API")
        telegram_token = st.text_input(
            "Telegram Bot Token",
            value=self.api_keys['telegram_token'],
            type="password"
        )
        telegram_chat_id = st.text_input(
            "Telegram Chat ID",
            value=self.api_keys['telegram_chat_id'],
            type="password"
        )

        # Botón para guardar APIs
        if st.button("Guardar Configuración de APIs"):
            new_api_keys = {
                'binance_api_key': binance_api_key,
                'binance_api_secret': binance_api_secret,
                'telegram_token': telegram_token,
                'telegram_chat_id': telegram_chat_id
            }
            self.save_env(new_api_keys)
            st.success("Configuración de APIs guardada exitosamente!")
            
            # Verificar conexiones
            if st.button("Verificar Conexiones"):
                self.verify_connections()

    def verify_connections(self):
        """Verificar conexiones con APIs"""
        try:
            # Verificar Binance
            from binance.client import Client
            client = Client(self.api_keys['binance_api_key'], self.api_keys['binance_api_secret'])
            client.get_account()
            st.success("✅ Conexión con Binance establecida correctamente")
        except Exception as e:
            st.error(f"❌ Error en conexión con Binance: {str(e)}")

        try:
            # Verificar Telegram
            import requests
            url = f"https://api.telegram.org/bot{self.api_keys['telegram_token']}/getMe"
            response = requests.get(url)
            if response.status_code == 200:
                st.success("✅ Conexión con Telegram establecida correctamente")
            else:
                st.error("❌ Error en conexión con Telegram")
        except Exception as e:
            st.error(f"❌ Error en conexión con Telegram: {str(e)}")

    def render_config_page(self):
        """Renderizar página de configuración en el dashboard"""
        st.title("Configuración del Bot")

        # Pestañas para diferentes secciones de configuración
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "APIs", "Trading", "Indicadores", "Gestión de Riesgo", "Modelo ML"
        ])

        with tab1:
            self.render_api_config()

        with tab2:
            st.subheader("Configuración de Trading")
            self.config['trading']['symbol'] = st.text_input(
                "Par de Trading",
                value=self.config['trading']['symbol']
            )
            self.config['trading']['max_daily_trades'] = st.number_input(
                "Máximo de Operaciones Diarias",
                min_value=1,
                max_value=100,
                value=self.config['trading']['max_daily_trades']
            )
            self.config['trading']['position_size'] = st.slider(
                "Tamaño de Posición (%)",
                min_value=0.01,
                max_value=1.0,
                value=self.config['trading']['position_size'],
                step=0.01
            )
            self.config['trading']['emergency_stop_loss'] = st.slider(
                "Stop Loss de Emergencia (%)",
                min_value=0.01,
                max_value=0.5,
                value=self.config['trading']['emergency_stop_loss'],
                step=0.01
            )

        with tab3:
            st.subheader("Configuración de Indicadores")
            col1, col2 = st.columns(2)
            
            with col1:
                self.config['indicators']['rsi_period'] = st.number_input(
                    "Período RSI",
                    min_value=2,
                    max_value=50,
                    value=self.config['indicators']['rsi_period']
                )
                self.config['indicators']['macd_fast'] = st.number_input(
                    "MACD Fast",
                    min_value=2,
                    max_value=50,
                    value=self.config['indicators']['macd_fast']
                )
                self.config['indicators']['macd_slow'] = st.number_input(
                    "MACD Slow",
                    min_value=2,
                    max_value=50,
                    value=self.config['indicators']['macd_slow']
                )

            with col2:
                self.config['indicators']['bb_period'] = st.number_input(
                    "Período Bollinger Bands",
                    min_value=2,
                    max_value=50,
                    value=self.config['indicators']['bb_period']
                )
                self.config['indicators']['stoch_k_period'] = st.number_input(
                    "Período Stochastic K",
                    min_value=2,
                    max_value=50,
                    value=self.config['indicators']['stoch_k_period']
                )
                self.config['indicators']['atr_period'] = st.number_input(
                    "Período ATR",
                    min_value=2,
                    max_value=50,
                    value=self.config['indicators']['atr_period']
                )

        with tab4:
            st.subheader("Gestión de Riesgo")
            col1, col2 = st.columns(2)
            
            with col1:
                self.config['risk']['max_concurrent_trades'] = st.number_input(
                    "Máximo de Operaciones Concurrentes",
                    min_value=1,
                    max_value=10,
                    value=self.config['risk']['max_concurrent_trades']
                )
                self.config['risk']['min_risk_reward_ratio'] = st.number_input(
                    "Ratio Mínimo Riesgo/Beneficio",
                    min_value=1.0,
                    max_value=5.0,
                    value=self.config['risk']['min_risk_reward_ratio'],
                    step=0.1
                )

            with col2:
                self.config['risk']['position_sizing_method'] = st.selectbox(
                    "Método de Tamaño de Posición",
                    options=['fixed', 'kelly', 'optimal_f'],
                    index=['fixed', 'kelly', 'optimal_f'].index(
                        self.config['risk']['position_sizing_method']
                    )
                )
                self.config['risk']['kelly_fraction'] = st.slider(
                    "Fracción de Kelly",
                    min_value=0.1,
                    max_value=1.0,
                    value=self.config['risk']['kelly_fraction'],
                    step=0.1
                )

        with tab5:
            st.subheader("Configuración del Modelo ML")
            self.config['model']['prediction_threshold'] = st.slider(
                "Umbral de Predicción",
                min_value=0.5,
                max_value=0.95,
                value=self.config['model']['prediction_threshold'],
                step=0.01
            )
            self.config['model']['retrain_interval'] = st.number_input(
                "Intervalo de Reentrenamiento (horas)",
                min_value=1,
                max_value=168,
                value=self.config['model']['retrain_interval']
            )
            self.config['model']['min_training_samples'] = st.number_input(
                "Mínimo de Muestras para Entrenamiento",
                min_value=100,
                max_value=10000,
                value=self.config['model']['min_training_samples'],
                step=100
            )

        # Botón para guardar configuración
        if st.button("Guardar Configuración"):
            self.save_config()
            st.success("Configuración guardada exitosamente!")
            
            # Reiniciar el bot con la nueva configuración
            if st.button("Reiniciar Bot con Nueva Configuración"):
                os.system("sudo systemctl restart trading-bot")
                st.success("Bot reiniciado con la nueva configuración!")

    def get_config(self):
        """Obtener la configuración actual"""
        return self.config 