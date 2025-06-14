# Usar una imagen base de Python
FROM python:3.10-bullseye

# Instala dependencias del sistema necesarias para compilar TA-Lib y otras librerías
RUN apt-get update --fix-missing && \
    apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    wget \
    curl \
    libtool \
    autoconf \
    automake \
    pkg-config \
    git \
    libta-lib0 \
    libta-lib0-dev \
    && rm -rf /var/lib/apt/lists/*

# Descargar, compilar e instalar TA-Lib (opcional, pero robusto)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Exportar rutas para el linker y compilador
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV TA_LIBRARY_PATH="/usr/local/lib"
ENV TA_INCLUDE_PATH="/usr/local/include"

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Exponer el puerto para Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación (ajusta según el servicio)
CMD streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0 