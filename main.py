from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

# ===============================
# Configuration and Constants
# ===============================

# ใช้ Pathlib เพื่อจัดการเส้นทางไฟล์ให้เป็น Relative Path และใช้งานได้ทุก OS
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Weather_DATASET.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True) # สร้างโฟลเดอร์ models ถ้ายังไม่มี

MODEL_1H_PATH = MODEL_DIR / "model_1h.joblib"
MODEL_3H_PATH = MODEL_DIR / "model_3h.joblib"
MODEL_24H_PATH = MODEL_DIR / "model_24h.joblib"

# PM2.5 Thresholds (มาตรฐานไทย: ต่ำกว่า 25, 25-50, 50-90, 90+)
PM25_MODERATE_THRESHOLD = 25
PM25_HIGH_THRESHOLD = 50
PM25_CRITICAL_THRESHOLD = 100 
# (ปรับ 90 เป็น 100 เพื่อให้ดูเป็นเกณฑ์สากลหรือปรับตามเกณฑ์ที่ใช้จริงได้)

app = FastAPI()
templates = Jinja2Templates(directory="templates") # ชี้ไปที่โฟลเดอร์ 'templates'

# Global variables (สำหรับเก็บข้อมูลที่ประมวลผลแล้วและโมเดล)
df_processed = None
X_features = None
model_1h = None
model_3h = None
model_24h = None

def air_risk_level(pm: float) -> str:
    """กำหนดระดับความเสี่ยงตามค่า PM2.5"""
    if pm < PM25_MODERATE_THRESHOLD:
        return "Low"
    elif pm < PM25_HIGH_THRESHOLD:
        return "Moderate"
    elif pm < PM25_CRITICAL_THRESHOLD:
        return "High"
    else:
        return "Critical"

def train_model(X_train: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """ฟังก์ชันสำหรับฝึกโมเดล RandomForest"""
    split_idx = int(len(y) * 0.8)
    y_train = y.iloc[:split_idx]
    
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# ===============================
# Data Loading and Model Setup (Startup Event)
# ===============================

def setup_data_and_models():
    """โหลดข้อมูล, ทำ Feature Engineering, และ Train/Load Models"""
    global df_processed, X_features, model_1h, model_3h, model_24h

    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at: {DATA_PATH}. Please check the file path.")
        raise
        
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Feature Engineering
    df = df.drop(columns=["aqi", "pm10"], errors='ignore') # เพิ่ม errors='ignore' เพื่อความปลอดภัย
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday
    df["day"] = df["date"].dt.day
    
    # Lag Features
    lags = [1, 2, 3, 6, 12, 24]
    for lag in lags:
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag)

    # Targets
    df["target_1h"] = df["pm2_5"].shift(-1)
    df["target_3h"] = df["pm2_5"].shift(-3)
    df["target_24h"] = df["pm2_5"].shift(-24)

    df_processed = df.dropna().reset_index(drop=True)

    feature_cols = ["temperature_c", "humidity_percent", "wind_speed_kmph", "hour", "weekday", "day"] + \
                   [f"pm2_5_lag_{lag}" for lag in lags]
                   
    X_features = df_processed[feature_cols]
    split_idx = int(len(df_processed) * 0.8)
    X_train = X_features.iloc[:split_idx]


    # 3. Train or Load Models
    if not all([MODEL_1H_PATH.exists(), MODEL_3H_PATH.exists(), MODEL_24H_PATH.exists()]):
        print("Training new models and saving to disk...")
        model_1h = train_model(X_train, df_processed["target_1h"])
        model_3h = train_model(X_train, df_processed["target_3h"])
        model_24h = train_model(X_train, df_processed["target_24h"])
        
        joblib.dump(model_1h, MODEL_1H_PATH)
        joblib.dump(model_3h, MODEL_3H_PATH)
        joblib.dump(model_24h, MODEL_24H_PATH)
    else:
        print("Loading pre-trained models from disk...")
        model_1h = joblib.load(MODEL_1H_PATH)
        model_3h = joblib.load(MODEL_3H_PATH)
        model_24h = joblib.load(MODEL_24H_PATH)


@app.on_event("startup")
async def startup_event():
    """รันฟังก์ชัน setup_data_and_models เมื่อแอปพลิเคชันเริ่มต้น"""
    setup_data_and_models()

# ===============================
# Home Endpoint
# ===============================
@app.get("/")
def home(request: Request):
    """แสดงผลข้อมูลล่าสุดและการพยากรณ์"""
    if X_features is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "System setup failed. Check server logs."})

    # เตรียม Features ล่าสุดสำหรับการพยากรณ์ (ใช้ .to_frame().T เพื่อคงโครงสร้าง 2D)
    latest_features = X_features.iloc[-1].to_frame().T

    # Prediction
    pred_1h = model_1h.predict(latest_features)[0]
    pred_3h = model_3h.predict(latest_features)[0]
    pred_24h = model_24h.predict(latest_features)[0]

    # เตรียมข้อมูลสำหรับตาราง
    table_data = []
    last_rows = df_processed.tail(10) # แสดง 10 แถวล่าสุด (ข้อมูลที่ใช้ในการพยากรณ์)
    
    # Note: เราใช้ค่าพยากรณ์ชุดเดียวกัน (จากข้อมูลล่าสุด) สำหรับทุกแถวในตาราง
    # หากต้องการพยากรณ์ล่วงหน้าสำหรับทุกๆ แถวในตารางจะต้องรัน model.predict ซ้ำ
    for i, row in last_rows.iterrows():
        table_data.append({
            "date": row["date"].strftime("%Y-%m-%d %H:%M"), # จัด format วันที่
            "temperature_c": round(row["temperature_c"], 1),
            "humidity_percent": round(row["humidity_percent"], 1),
            "wind_speed_kmph": round(row["wind_speed_kmph"], 1),
            "pm2_5": round(row["pm2_5"], 2),
            "pred_1h": pred_1h, # ใช้ค่าพยากรณ์ล่าสุด
            "pred_3h": pred_3h,
            "pred_24h": pred_24h,
            "airrisk": air_risk_level(row["pm2_5"])
        })
        
    # เตรียมข้อมูลสรุปการพยากรณ์
    latest_pred = {
        "pred_1h": pred_1h,
        "pred_3h": pred_3h,
        "pred_24h": pred_24h,
        "airrisk_1h": air_risk_level(pred_1h),
        "airrisk_3h": air_risk_level(pred_3h),
        "airrisk_24h": air_risk_level(pred_24h)
    }

    return templates.TemplateResponse(
        "home.html",
        {"request": request, "table_data": table_data, "latest_pred": latest_pred}
    )

# วิธีรัน (จาก Terminal/Command Prompt): uvicorn main:app --reload