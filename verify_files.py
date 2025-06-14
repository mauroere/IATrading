import os

REQUIRED_FILES = [
    'config.yaml',
    'requirements.txt',
    'Dockerfile',
    'indicators.py',
    'risk_manager.py',
    'ml_model.py',
    'market_analysis.py',
    'optimizer.py',
    'backtester.py',
    'monitor.py',
    'run_system.py',
    'dashboard.py',
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missing:
    print("❌ Faltan los siguientes archivos críticos:")
    for f in missing:
        print(f" - {f}")
    exit(1)
else:
    print("✅ Todos los archivos críticos existen y están listos para el deploy.") 