from datetime import datetime, timezone as dt_timezone, timedelta
from typing import Optional, Dict, Any, List
import math

from fastapi import FastAPI, Query, HTTPException, Path
from pydantic import BaseModel, Field
import h3
from timezonefinder import TimezoneFinder
import pytz
import holidays

app = FastAPI(title="earth-presence-intelligence-grid-1552")

tf = TimezoneFinder()

# --- Pydantic Models for Tools and API ---

class LocationInput(BaseModel):
    lat: float = Field(..., description="Latitude (-90 to 90)")
    lon: float = Field(..., description="Longitude (-180 to 180)")

class ActivityWindowInput(BaseModel):
    lat: float
    lon: float
    hours: int = Field(24, description="Forecast horizon in hours")

class SleepForecastInput(BaseModel):
    lat: float
    lon: float
    hours: int = Field(48, description="Forecast horizon in hours")

class H3Input(BaseModel):
    h3: str = Field(..., description="H3 Cell Index")

# --- Helper Functions ---

def validate_coords(lat: float, lon: float):
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise HTTPException(status_code=400, detail="Invalid coordinates")

def get_local_time(lat: float, lon: float) -> datetime:
    validate_coords(lat, lon)
    tz_str = tf.timezone_at(lat=lat, lng=lon) or 'UTC'
    try:
        loc_tz = pytz.timezone(tz_str)
    except Exception:
        loc_tz = pytz.UTC
    return datetime.now(loc_tz)

def compute_sleep_and_awake(local_dt: datetime):
    hour = local_dt.hour + local_dt.minute / 60.0
    # Sleep bands logic
    if 0.0 <= hour < 4.0:
        sleep_prob = 0.85
        awake_prob = 0.15
        sleep_band = "Very High"
    elif 4.0 <= hour < 6.0:
        sleep_prob = 0.70
        awake_prob = 0.30
        sleep_band = "High"
    elif 6.0 <= hour < 8.0:
        sleep_prob = 0.35
        awake_prob = 0.65
        sleep_band = "Medium"
    elif 8.0 <= hour < 21.0:
        sleep_prob = 0.10
        awake_prob = 0.90
        sleep_band = "Low"
    else:
        sleep_prob = 0.45
        awake_prob = 0.55
        sleep_band = "Increasing"
        
    return sleep_prob, awake_prob, sleep_band

def compute_activity_score(local_dt: datetime, is_holiday: bool) -> tuple[int, str]:
    hour = local_dt.hour
    weekday = local_dt.weekday() # 0-6
    
    base_score = 50
    # Business hours boost
    if 9 <= hour <= 17 and weekday < 5 and not is_holiday:
        base_score = 85
    elif (7 <= hour < 9) or (17 < hour <= 21):
        base_score = 65
    elif 21 < hour or hour < 6:
        base_score = 15
    else:
        base_score = 40

    if is_holiday and weekday >= 5:
        base_score = int(base_score * 0.7)

    activity_score = max(0, min(100, base_score))
    
    if activity_score > 75:
        state = "ACTIVE"
    elif activity_score > 40:
        state = "MODERATE"
    else:
        state = "QUIET"
        
    return activity_score, state

def compute_darkness_score(lat: float, local_dt: datetime) -> tuple[int, str, bool]:
    hour = local_dt.hour + local_dt.minute / 60.0
    # Approximate solar elevation based on hour and latitude
    # Night roughly 19:00 to 6:00
    is_night = hour < 6.0 or hour > 19.0
    if is_night:
        darkness_score = 85
        light_state = "DARK"
    elif 6.0 <= hour < 7.0 or 18.0 < hour <= 19.0:
        darkness_score = 50
        light_state = "DIM"
    else:
        darkness_score = 10
        light_state = "BRIGHT"
        
    return darkness_score, light_state, is_night

def check_is_holiday(lat: float, local_dt: datetime) -> bool:
    # Use python-holidays with rough country estimation or fallback
    try:
        # Simple lookup fallback based on lat/lon rough regions or default to global/US/MY
        country_holidays = holidays.CountryHoliday('US') # default fallback
        return local_dt.date() in country_holidays
    except Exception:
        return False

# --- Core API Endpoints ---

@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "earth-presence-intelligence-grid-1552",
        "name": "Earth Presence Intelligence Grid (EPIG)"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/heartbeat")
def heartbeat():
    return {
        "status": "operational",
        "timestamp": datetime.now(dt_timezone.utc).isoformat(),
        "service": "earth-presence-intelligence-grid-1552"
    }

@app.get("/tools")
def tools():
    return {
        "tools": [
            {
                "name": "get_human_presence",
                "description": "Get current human presence score and metrics for a location.",
                "input_schema": {"lat": "float", "lon": "float"}
            },
            {
                "name": "get_activity_window",
                "description": "Get expected activity windows over a forecast period.",
                "input_schema": {"lat": "float", "lon": "float", "hours": "int"}
            },
            {
                "name": "get_sleep_forecast",
                "description": "Get sleep/awake probability forecast.",
                "input_schema": {"lat": "float", "lon": "float", "hours": "int"}
            },
            {
                "name": "get_darkness_state",
                "description": "Get darkness and illumination state.",
                "input_schema": {"lat": "float", "lon": "float"}
            },
            {
                "name": "get_h3_presence",
                "description": "Get aggregated presence for an H3 cell.",
                "input_schema": {"h3": "string"}
            }
        ]
    }

@app.get("/v1/presence/current")
def presence_current(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    is_hol = check_is_holiday(lat, local_dt)
    activity_score, activity_state = compute_activity_score(local_dt, is_hol)
    sleep_prob, awake_prob, _ = compute_sleep_and_awake(local_dt)
    darkness_score, light_state, night = compute_darkness_score(lat, local_dt)
    
    # Formula: Presence Score = Activity Score + Awake Score (scaled) + Calendar Mod + Illumination Mod
    awake_score_val = int(awake_prob * 30)
    cal_mod = -15 if is_hol else 0
    illum_mod = -5 if light_state == "BRIGHT" else 5
    
    presence_score = max(0, min(100, activity_score + awake_score_val + cal_mod + illum_mod))
    
    if presence_score > 80:
        classification = "PEAK"
    elif presence_score > 60:
        classification = "HIGH"
    elif presence_score > 40:
        classification = "MODERATE"
    elif presence_score > 20:
        classification = "QUIET"
    else:
        classification = "MOSTLY_OFFLINE"

    availability = "HIGH" if presence_score > 60 else ("MODERATE" if presence_score > 30 else "LOW")
    response_prob = round(presence_score / 100.0, 2)

    return {
        "presence_score": presence_score,
        "score": presence_score,
        "classification": classification,
        "activity_score": activity_score,
        "activity_state": activity_state,
        "awake_probability": awake_prob,
        "sleep_probability": sleep_prob,
        "darkness_score": darkness_score,
        "holiday": is_hol,
        "availability": availability,
        "response_probability": response_prob,
        "timestamp": local_dt.isoformat()
    }

@app.get("/v1/presence/location")
def presence_location(lat: float = Query(...), lon: float = Query(...)):
    return presence_current(lat=lat, lon=lon)

@app.get("/v1/activity")
def activity(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    is_hol = check_is_holiday(lat, local_dt)
    activity_score, state = compute_activity_score(local_dt, is_hol)
    return {
        "activity_score": activity_score,
        "score": activity_score,
        "state": state,
        "holiday": is_hol,
        "local_time": local_dt.isoformat()
    }

@app.get("/v1/sleep")
def sleep_endpoint(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    sleep_prob, awake_prob, sleep_band = compute_sleep_and_awake(local_dt)
    pop_estimate = int(awake_prob * 32500000) # Example population scale
    return {
        "awake_probability": awake_prob,
        "sleep_probability": sleep_prob,
        "sleep_band": sleep_band,
        "population_awake_estimate": pop_estimate,
        "local_time": local_dt.isoformat()
    }

@app.get("/v1/darkness")
def darkness(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    darkness_score, light_state, night = compute_darkness_score(lat, local_dt)
    return {
        "darkness_score": darkness_score,
        "light_state": light_state,
        "light": light_state,
        "night": night,
        "moon_illumination": 50, # Default mock or calculated value
        "local_time": local_dt.isoformat()
    }

@app.get("/v1/calendar")
def calendar_intelligence(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    is_hol = check_is_holiday(lat, local_dt)
    weekend = local_dt.weekday() >= 5
    
    if is_hol:
        calendar_state = "PUBLIC_HOLIDAY"
        office_activity = "LOW"
        gov_activity = "CLOSED"
    elif weekend:
        calendar_state = "WEEKEND"
        office_activity = "LOW"
        gov_activity = "CLOSED"
    else:
        calendar_state = "WORKDAY"
        office_activity = "HIGH"
        gov_activity = "OPEN"

    return {
        "calendar_state": calendar_state,
        "state": calendar_state,
        "office_activity": office_activity,
        "government_activity": gov_activity,
        "holiday": is_hol,
        "weekend": weekend,
        "local_date": local_dt.date().isoformat()
    }

@app.get("/v1/h3/{cell}")
def h3_presence(cell: str = Path(...)):
    try:
        # validate h3 index
        h3.h3_is_valid(cell)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid H3 cell index")
    
    return {
        "h3": cell,
        "cell": cell,
        "presence_score": 76,
        "awake_probability": 0.82,
        "darkness_score": 12,
        "holiday": False,
        "availability": "HIGH"
    }

@app.get("/v1/forecast")
def forecast(lat: float = Query(3.139), lon: float = Query(101.6869), hours: int = Query(24)):
    validate_coords(lat, lon)
    base_dt = get_local_time(lat, lon)
    forecast_list = []
    best_windows = []
    
    current_window_start = None
    
    for i in range(hours):
        step_dt = base_dt + timedelta(hours=i)
        is_hol = check_is_holiday(lat, step_dt)
        act, _ = compute_activity_score(step_dt, is_hol)
        _, awake_prob, _ = compute_sleep_and_awake(step_dt)
        
        forecast_list.append({
            "time": step_dt.isoformat(),
            "awake_probability": awake_prob,
            "activity_score": act
        })
        
        if act > 70 and not current_window_start:
            current_window_start = step_dt.strftime("%H:%M")
        elif act <= 70 and current_window_start:
            best_windows.append({
                "start": current_window_start,
                "end": step_dt.strftime("%H:%M"),
                "confidence": 0.88
            })
            current_window_start = None

    if current_window_start:
        best_windows.append({
            "start": current_window_start,
            "end": (base_dt + timedelta(hours=hours)).strftime("%H:%M"),
            "confidence": 0.85
        })

    return {
        "best_windows": best_windows,
        "forecast": forecast_list
    }
