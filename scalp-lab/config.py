# Конфиг бэктест-лаборатории
PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
    "DOTUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "ARBUSDT",
    "OPUSDT", "SUIUSDT", "PEPEUSDT", "FILUSDT", "APTUSDT",
]

# days: глубина истории; train_days/test_days: окно walk-forward
TFS = {
    "1m":  {"days": 180, "train_days": 20, "test_days": 5},
    "5m":  {"days": 365, "train_days": 30, "test_days": 7},
    "15m": {"days": 365, "train_days": 45, "test_days": 14},
}

# taker fee фьючерсов Binance (0.05%) + оценочное проскальзывание
FEE = 0.0005
SLIP = 0.0002
