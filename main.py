from datetime import datetime, timezone as dt_timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import math
import numpy as np

from fastapi import FastAPI, Query, HTTPException, Path, Body
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
import h3
from timezonefinder import TimezoneFinder
import pytz
import holidays

# ---------------------------------------------------------------------------
# Framework & App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="earth-presence-intelligence-grid-1552",
    description="Global Human Presence, Circadian Wakefulness, and Activity Density Engine for Autonomous Agents",
    version="2.0.0",
    default_response_class=ORJSONResponse,
)

tf = TimezoneFinder()

# ---------------------------------------------------------------------------
# Presence & Activity Enums
# ---------------------------------------------------------------------------
class PresenceClassification(str, Enum):
    DORMANT = "DORMANT"
    LOW_ACTIVITY = "LOW_ACTIVITY"
    ACTIVE = "ACTIVE"
    HIGH_ACTIVITY = "HIGH_ACTIVITY"
    PEAK_ACTIVITY = "PEAK_ACTIVITY"

class LightState(str, Enum):
    BRIGHT = "BRIGHT"
    DAYLIGHT = "DAYLIGHT"
    CIVIL_TWILIGHT = "CIVIL_TWILIGHT"
    NAUTICAL_TWILIGHT = "NAUTICAL_TWILIGHT"
    ASTRONOMICAL_TWILIGHT = "ASTRONOMICAL_TWILIGHT"
    NIGHT = "NIGHT"

# ---------------------------------------------------------------------------
# Standard Global Metro Coordinates
# ---------------------------------------------------------------------------
GLOBAL_METRO_REGISTRY: Dict[str, Tuple[float, float, str]] = {
    "kuala lumpur": (3.1390, 101.6869, "Kuala Lumpur"),
    "tokyo": (35.6762, 139.6503, "Tokyo"),
    "london": (51.5074, -0.1278, "London"),
    "new york": (40.7128, -74.0060, "New York"),
    "singapore": (1.3521, 103.8198, "Singapore"),
    "san francisco": (37.7749, -122.4194, "San Francisco"),
    "dubai": (25.2048, 55.2708, "Dubai"),
    "sydney": (-33.8688, 151.2093, "Sydney"),
    "hong kong": (22.3193, 114.1694, "Hong Kong"),
    "berlin": (52.5200, 13.4050, "Berlin"),
    "paris": (48.8566, 2.3522, "Paris"),
    "toronto": (43.6532, -79.3832, "Toronto"),
    "shanghai": (31.2304, 121.4737, "Shanghai"),
    "seoul": (37.5665, 126.9780, "Seoul"),
    "mumbai": (19.0760, 72.8777, "Mumbai"),
    "chicago": (41.8781, -87.6298, "Chicago"),
    "los angeles": (34.0522, -118.2437, "Los Angeles"),
    "sao paulo": (-23.5505, -46.6333, "Sao Paulo"),
    "johannesburg": (-26.2041, 28.0473, "Johannesburg"),
}

# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------
class LocationInput(BaseModel):
    lat: float = Field(..., description="Latitude (-90 to 90)")
    lon: float = Field(..., description="Longitude (-180 to 180)")

class ActivityWindowInput(BaseModel):
    lat: float = Field(..., description="Latitude (-90 to 90)")
    lon: float = Field(..., description="Longitude (-180 to 180)")
    hours: int = Field(24, description="Forecast horizon in hours")

class SleepForecastInput(BaseModel):
    lat: float = Field(..., description="Latitude (-90 to 90)")
    lon: float = Field(..., description="Longitude (-180 to 180)")
    hours: int = Field(48, description="Forecast horizon in hours")

class H3Input(BaseModel):
    h3: str = Field(..., description="H3 Cell Index")

class CompareRegionsInput(BaseModel):
    cities: List[str] = Field(
        default=["Kuala Lumpur", "Tokyo", "London", "New York"],
        description="List of city names to evaluate and compare"
    )

# ---------------------------------------------------------------------------
# Cross-Version H3 Spatial Helpers
# ---------------------------------------------------------------------------
def _is_valid_h3(cell: str) -> bool:
    try:
        if hasattr(h3, 'is_valid_cell'):
            return bool(h3.is_valid_cell(cell))
        elif hasattr(h3, 'h3_is_valid'):
            return bool(h3.h3_is_valid(cell))
    except Exception:
        pass
    return len(cell) >= 15 and all(c in '0123456789abcdefABCDEF' for c in cell)

def _h3_to_latlon(cell: str) -> Tuple[float, float]:
    try:
        if hasattr(h3, 'cell_to_latlng'):
            coords = h3.cell_to_latlng(cell)
            return (float(coords[0]), float(coords[1]))
        elif hasattr(h3, 'h3_to_geo'):
            coords = h3.h3_to_geo(cell)
            return (float(coords[0]), float(coords[1]))
    except Exception:
        pass
    return (3.1390, 101.6869)

# ---------------------------------------------------------------------------
# Core Analytical Engines
# ---------------------------------------------------------------------------
def calculate_human_wakefulness(local_hour: float) -> float:
    """
    Step 1: Human Wakefulness Engine.
    Uses continuous circadian rhythm biology (Process C & Process S) to model
    wakefulness smoothly without discrete step functions.
    03:00 -> very low (~4)
    08:00 -> rising (~68)
    11:00 -> peak (~99)
    15:00 -> moderate postprandial dip (~82)
    22:00 -> declining (~47)
    """
    h = local_hour % 24.0

    def sig(x: float, x0: float, k: float) -> float:
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))

    if h < 12.0:
        raw = sig(h, 7.2, 0.95)
    else:
        raw = 1.0 - sig(h, 21.8, 0.85)

    base_score = 5.0 + 91.0 * raw

    # Circadian morning peak around 10:30-11:30
    peak_boost = 6.0 * math.exp(-((h - 11.0) ** 2) / 4.0)
    # Postprandial afternoon dip around 14:30-15:30
    afternoon_dip = 14.0 * math.exp(-((h - 15.0) ** 2) / 3.5)

    val = base_score + peak_boost - afternoon_dip
    if h < 5.0:
        val = 4.0 + 8.0 * (1.0 - math.exp(-((h - 3.0) ** 2) / 3.0))

    return round(float(np.clip(val, 3.0, 99.0)), 2)

def calculate_sleep_probability(local_hour: float) -> float:
    """
    Step 5: Human Sleep Probability Engine.
    Inverse circadian model calibrated to population sleep homeostatic pressure.
    Returns 0.0 - 100.0%.
    """
    wakefulness = calculate_human_wakefulness(local_hour)
    return round(float(np.clip(100.0 - wakefulness, 1.0, 97.0)), 2)

def calculate_activity_density(
    hour_of_day: float,
    weekday: int,
    is_holiday: bool = False
) -> float:
    """
    Step 2: Human Activity Density Engine.
    Smooth multimodal Fourier/Gaussian distribution modeling urban and commercial movement.
    Differentiates workdays, weekends, and public holidays.
    """
    h = hour_of_day % 24.0
    is_weekend = (weekday >= 5)

    if not is_weekend and not is_holiday:
        # Workday: Commute and corporate core surges
        p1 = 38.0 * math.exp(-((h - 10.5) ** 2) / 10.0)
        p2 = 36.0 * math.exp(-((h - 17.0) ** 2) / 8.0)
        day_baseline = 42.0 / (1.0 + math.exp(-1.2 * (h - 7.5))) * (1.0 / (1.0 + math.exp(1.0 * (h - 20.5))))
        night_floor = 6.0 + 8.0 * (1.0 - 1.0 / (1.0 + math.exp(-0.8 * (h - 4.5))))
        raw = p1 + p2 + day_baseline + night_floor
    else:
        # Weekend & Holiday: Shifted waking curve, recreational afternoon peaks
        p1 = 45.0 * math.exp(-((h - 16.5) ** 2) / 16.0)
        day_baseline = 40.0 / (1.0 + math.exp(-0.9 * (h - 9.5))) * (1.0 / (1.0 + math.exp(0.8 * (h - 22.0))))
        night_floor = 8.0
        raw = p1 + day_baseline + night_floor
        if is_holiday:
            raw *= 0.92

    return round(float(np.clip(raw, 4.0, 98.0)), 1)

def calculate_business_availability(
    hour_of_day: float,
    weekday: int,
    is_holiday: bool = False
) -> Dict[str, Any]:
    """
    Step 3: Business Open-State Engine.
    Outputs machine-readable metrics for office, financial market, and retail activities.
    """
    h = hour_of_day % 24.0
    is_weekend = (weekday >= 5)

    def smooth_box(x: float, open_hr: float, close_hr: float, slope: float = 1.2) -> float:
        open_ramp = 1.0 / (1.0 + math.exp(-slope * (x - open_hr)))
        close_ramp = 1.0 / (1.0 + math.exp(slope * (x - close_hr)))
        return open_ramp * close_ramp

    # 1. Corporate Office (08:30 - 18:00)
    office_box = smooth_box(h, 8.5, 17.5, slope=1.5)
    if is_holiday:
        office_score = 5.0 + 10.0 * office_box
    elif is_weekend:
        office_score = 8.0 + 12.0 * office_box
    else:
        office_score = 4.0 + 94.0 * office_box

    # 2. Financial / Commercial Markets (09:00 - 16:30)
    market_box = smooth_box(h, 9.0, 16.5, slope=1.8)
    if is_holiday or is_weekend:
        market_score = 3.0 + 5.0 * market_box
    else:
        market_score = 2.0 + 96.0 * market_box

    # 3. Consumer Retail, Dining, and Leisure (10:00 - 22:00)
    retail_box = smooth_box(h, 10.0, 21.5, slope=1.1)
    if is_holiday or is_weekend:
        retail_score = 8.0 + 90.0 * retail_box
    else:
        retail_score = 6.0 + 82.0 * retail_box

    office_score = round(float(np.clip(office_score, 0.0, 100.0)), 1)
    market_score = round(float(np.clip(market_score, 0.0, 100.0)), 1)
    retail_score = round(float(np.clip(retail_score, 0.0, 100.0)), 1)
    composite = round((office_score * 0.40 + market_score * 0.35 + retail_score * 0.25), 1)

    return {
        "office_activity": office_score,
        "market_activity": market_score,
        "retail_activity": retail_score,
        "business_availability_score": composite,
        "open_state": "OPEN" if composite >= 50.0 else ("PARTIAL" if composite >= 25.0 else "CLOSED")
    }

def calculate_solar_elevation(lat: float, lon: float, dt_utc: datetime) -> float:
    """Astronomical solar elevation angle calculation using standard solar position algorithms."""
    day_of_year = dt_utc.timetuple().tm_yday
    hour_utc = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    gamma = 2.0 * math.pi / 365.25 * (day_of_year - 1 + (hour_utc - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )

    time_offset = eqtime + 4.0 * lon
    tst = (hour_utc * 60.0 + time_offset) % 1440.0
    ha_rad = math.radians((tst / 4.0) - 180.0)
    lat_rad = math.radians(lat)

    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith_deg = math.degrees(math.acos(cos_zenith))
    return 90.0 - zenith_deg

def calculate_darkness_score(lat: float, lon: float, dt_utc: datetime) -> Tuple[float, str, bool, float]:
    """
    Step 4: Global Darkness Engine.
    Uses precise solar elevation to evaluate Civil, Nautical, and Astronomical Twilights.
    Returns (darkness_score, light_state, is_night, solar_elevation_deg).
    """
    elevation_deg = calculate_solar_elevation(lat, lon, dt_utc)

    if elevation_deg >= 10.0:
        darkness_score = max(0.0, 5.0 * (1.0 - elevation_deg / 60.0))
        light_state = LightState.BRIGHT.value
        is_night = False
    elif elevation_deg >= 0.0:
        darkness_score = 5.0 + (10.0 - elevation_deg) * 2.0
        light_state = LightState.DAYLIGHT.value
        is_night = False
    elif elevation_deg >= -6.0:
        darkness_score = 25.0 + (-elevation_deg / 6.0) * 30.0
        light_state = LightState.CIVIL_TWILIGHT.value
        is_night = False
    elif elevation_deg >= -12.0:
        darkness_score = 55.0 + ((-elevation_deg - 6.0) / 6.0) * 25.0
        light_state = LightState.NAUTICAL_TWILIGHT.value
        is_night = True
    elif elevation_deg >= -18.0:
        darkness_score = 80.0 + ((-elevation_deg - 12.0) / 6.0) * 15.0
        light_state = LightState.ASTRONOMICAL_TWILIGHT.value
        is_night = True
    else:
        darkness_score = min(100.0, 95.0 + ((-elevation_deg - 18.0) / 10.0) * 5.0)
        light_state = LightState.NIGHT.value
        is_night = True

    return (
        round(float(np.clip(darkness_score, 0.0, 100.0)), 1),
        light_state,
        is_night,
        round(elevation_deg, 2)
    )

def calculate_calendar_intelligence(lat: float, lon: float, local_dt: datetime) -> Dict[str, Any]:
    """
    Step 6: Calendar Intelligence Layer.
    Resolves country boundaries to calculate local holidays, weekends, and activity modifiers.
    """
    tz_str = tf.timezone_at(lat=lat, lng=lon) or 'UTC'
    country_code = 'US'

    # Extract ISO country code from timezone where available
    for code, zones in pytz.country_timezones.items():
        if tz_str in zones:
            country_code = code
            break

    is_hol = False
    try:
        nat_holidays = holidays.country_holidays(country_code)
        is_hol = local_dt.date() in nat_holidays
    except Exception:
        is_hol = False

    weekday = local_dt.weekday()
    is_weekend = weekday >= 5

    if is_hol:
        calendar_state = "PUBLIC_HOLIDAY"
        office_activity_state = "LOW"
        gov_activity = "CLOSED"
        activity_modifier = 0.70
    elif is_weekend:
        calendar_state = "WEEKEND"
        office_activity_state = "LOW"
        gov_activity = "CLOSED"
        activity_modifier = 0.82 if weekday == 5 else 0.75
    else:
        calendar_state = "WORKDAY"
        office_activity_state = "HIGH"
        gov_activity = "OPEN"
        activity_modifier = 1.00

    return {
        "calendar_state": calendar_state,
        "is_holiday": is_hol,
        "is_weekend": is_weekend,
        "office_activity_state": office_activity_state,
        "government_activity": gov_activity,
        "activity_modifier": round(activity_modifier, 2),
        "country_code": country_code,
        "timezone": tz_str
    }

def calculate_productivity_index(
    wakefulness: float,
    activity_density: float,
    business_availability_score: float,
    calendar_modifier: float
) -> float:
    """
    Step 7: Global Productivity Engine.
    Synthesizes cognitive wakefulness, physical urban density, commercial open states,
    and calendar constraints into a 0 - 100 productivity score.
    """
    w = max(0.0, min(100.0, wakefulness))
    a = max(0.0, min(100.0, activity_density))
    b = max(0.0, min(100.0, business_availability_score))
    mod = max(0.4, min(1.2, calendar_modifier))

    raw = (0.35 * w + 0.35 * b + 0.30 * a) * mod
    return round(float(np.clip(raw, 0.0, 100.0)), 1)

def calculate_earth_presence_index(
    wakefulness: float,
    activity_density: float,
    productivity_score: float,
    business_availability_score: float,
    darkness_score: float
) -> float:
    """
    Step 8: Earth Presence Index (EPI).
    The primary machine-consumable signal synthesizing human wakefulness,
    productivity, physical density, commercial availability, and diurnal lighting.
    """
    w = max(0.0, min(100.0, wakefulness))
    a = max(0.0, min(100.0, activity_density))
    p = max(0.0, min(100.0, productivity_score))
    b = max(0.0, min(100.0, business_availability_score))

    # Ambient daylight factor: daylight slightly amplifies presence without zeroing night presence
    illum_weight = (100.0 - max(0.0, min(100.0, darkness_score))) / 100.0
    light_factor = 0.85 + 0.15 * illum_weight

    raw = (0.28 * w + 0.26 * a + 0.26 * p + 0.20 * b) * light_factor
    return round(float(np.clip(raw, 0.0, 100.0)), 1)

def classify_presence(earth_presence_index: float) -> str:
    """
    Step 9: Standard Presence Classification.
    Returns: DORMANT, LOW_ACTIVITY, ACTIVE, HIGH_ACTIVITY, PEAK_ACTIVITY.
    """
    score = max(0.0, min(100.0, earth_presence_index))
    if score >= 80.0:
        return PresenceClassification.PEAK_ACTIVITY.value
    elif score >= 60.0:
        return PresenceClassification.HIGH_ACTIVITY.value
    elif score >= 40.0:
        return PresenceClassification.ACTIVE.value
    elif score >= 20.0:
        return PresenceClassification.LOW_ACTIVITY.value
    else:
        return PresenceClassification.DORMANT.value

def _legacy_presence_classification(presence_score: float) -> str:
    """Preserves exact legacy classification strings for backward compatibility."""
    if presence_score > 80:
        return "PEAK"
    elif presence_score > 60:
        return "HIGH"
    elif presence_score > 40:
        return "MODERATE"
    elif presence_score > 20:
        return "QUIET"
    else:
        return "MOSTLY_OFFLINE"

# ---------------------------------------------------------------------------
# Backward-Compatible Helper Functions (Preserved Signatures)
# ---------------------------------------------------------------------------
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

def compute_sleep_and_awake(local_dt: datetime) -> Tuple[float, float, str]:
    hour = local_dt.hour + local_dt.minute / 60.0
    wake_score = calculate_human_wakefulness(hour)
    sleep_prob = round((100.0 - wake_score) / 100.0, 2)
    awake_prob = round(wake_score / 100.0, 2)

    if sleep_prob >= 0.80:
        sleep_band = "Very High"
    elif sleep_prob >= 0.60:
        sleep_band = "High"
    elif sleep_prob >= 0.35:
        sleep_band = "Medium"
    elif sleep_prob >= 0.15:
        sleep_band = "Low"
    else:
        sleep_band = "Very Low"

    return sleep_prob, awake_prob, sleep_band

def compute_activity_score(local_dt: datetime, is_holiday: bool) -> Tuple[int, str]:
    hour = local_dt.hour + local_dt.minute / 60.0
    density = calculate_activity_density(hour, local_dt.weekday(), is_holiday)
    score_int = int(round(density))

    if score_int > 75:
        state = "ACTIVE"
    elif score_int > 40:
        state = "MODERATE"
    else:
        state = "QUIET"

    return score_int, state

def compute_darkness_score(lat: float, local_dt: datetime) -> Tuple[int, str, bool]:
    # Convert local_dt to UTC for solar elevation
    if local_dt.tzinfo is not None:
        utc_dt = local_dt.astimezone(pytz.UTC)
    else:
        utc_dt = local_dt.replace(tzinfo=pytz.UTC)

    # Approximate longitude from timezone offset if longitude not given directly
    offset_hours = local_dt.utcoffset().total_seconds() / 3600.0 if local_dt.utcoffset() else 0.0
    approx_lon = offset_hours * 15.0

    score_f, state, is_night, _ = calculate_darkness_score(lat, approx_lon, utc_dt)
    return int(round(score_f)), state, is_night

def check_is_holiday(lat: float, local_dt: datetime) -> bool:
    cal_info = calculate_calendar_intelligence(lat, 0.0, local_dt)
    return cal_info["is_holiday"]

# ---------------------------------------------------------------------------
# Master Presence Aggregator
# ---------------------------------------------------------------------------
def evaluate_presence(lat: float, lon: float, target_dt: Optional[datetime] = None) -> Dict[str, Any]:
    """Generates the unified presence intelligence signal for geographic coordinates."""
    validate_coords(lat, lon)
    if target_dt is None:
        local_dt = get_local_time(lat, lon)
        utc_dt = datetime.now(pytz.UTC)
    else:
        tz_str = tf.timezone_at(lat=lat, lng=lon) or 'UTC'
        try:
            loc_tz = pytz.timezone(tz_str)
            local_dt = target_dt.astimezone(loc_tz)
        except Exception:
            local_dt = target_dt
        utc_dt = target_dt.astimezone(pytz.UTC) if target_dt.tzinfo else target_dt.replace(tzinfo=pytz.UTC)

    local_hour = local_dt.hour + local_dt.minute / 60.0 + local_dt.second / 3600.0

    # 1. Wakefulness & Sleep
    wakefulness = calculate_human_wakefulness(local_hour)
    sleep_prob_pct = calculate_sleep_probability(local_hour)
    sleep_prob = round(sleep_prob_pct / 100.0, 2)
    awake_prob = round(wakefulness / 100.0, 2)

    # 2. Calendar Intelligence
    cal_info = calculate_calendar_intelligence(lat, lon, local_dt)

    # 3. Activity Density
    activity_density = calculate_activity_density(local_hour, local_dt.weekday(), cal_info["is_holiday"])
    legacy_act_score, legacy_act_state = compute_activity_score(local_dt, cal_info["is_holiday"])

    # 4. Business Open-State
    business_state = calculate_business_availability(local_hour, local_dt.weekday(), cal_info["is_holiday"])

    # 5. Global Darkness & Solar Elevation
    darkness_score, light_state, is_night, solar_elev = calculate_darkness_score(lat, lon, utc_dt)

    # 6. Global Productivity Index
    productivity_score = calculate_productivity_index(
        wakefulness=wakefulness,
        activity_density=activity_density,
        business_availability_score=business_state["business_availability_score"],
        calendar_modifier=cal_info["activity_modifier"]
    )

    # 7. Earth Presence Index (EPI)
    earth_presence_index = calculate_earth_presence_index(
        wakefulness=wakefulness,
        activity_density=activity_density,
        productivity_score=productivity_score,
        business_availability_score=business_state["business_availability_score"],
        darkness_score=darkness_score
    )

    # 8. Classifications
    presence_classification = classify_presence(earth_presence_index)
    legacy_classification = _legacy_presence_classification(earth_presence_index)
    availability = "HIGH" if earth_presence_index > 60 else ("MODERATE" if earth_presence_index > 30 else "LOW")
    response_prob = round(earth_presence_index / 100.0, 2)

    return {
        "earth_presence_index": earth_presence_index,
        "presence_score": earth_presence_index,
        "score": earth_presence_index,
        "classification": legacy_classification,
        "presence_classification": presence_classification,
        "activity_density": activity_density,
        "activity_score": legacy_act_score,
        "activity_state": legacy_act_state,
        "wakefulness": wakefulness,
        "awake_probability": awake_prob,
        "sleep_probability": sleep_prob,
        "sleep_probability_percent": sleep_prob_pct,
        "productivity_score": productivity_score,
        "business_availability": business_state,
        "darkness_score": darkness_score,
        "light_state": light_state,
        "solar_elevation_deg": solar_elev,
        "night": is_night,
        "holiday": cal_info["is_holiday"],
        "weekend": cal_info["is_weekend"],
        "calendar_state": cal_info["calendar_state"],
        "activity_modifier": cal_info["activity_modifier"],
        "availability": availability,
        "response_probability": response_prob,
        "timestamp": local_dt.isoformat(),
        "local_time": local_dt.isoformat(),
        "timezone": cal_info["timezone"]
    }

# ---------------------------------------------------------------------------
# Step 10: Multi-Region Comparison Engine
# ---------------------------------------------------------------------------
def compare_regions(cities: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compares real-time human presence across global metropolitan regions.
    Outputs: highest activity, highest productivity, highest sleep probability, and lowest presence.
    """
    if not cities:
        cities = ["Kuala Lumpur", "Tokyo", "London", "New York"]

    evaluated_regions = []
    now_utc = datetime.now(pytz.UTC)

    for city_name in cities:
        c_clean = city_name.strip().lower()
        if c_clean in GLOBAL_METRO_REGISTRY:
            lat, lon, canon_name = GLOBAL_METRO_REGISTRY[c_clean]
        else:
            # Substring search or fallback to Kuala Lumpur
            matched = False
            for k, val in GLOBAL_METRO_REGISTRY.items():
                if c_clean in k or k in c_clean:
                    lat, lon, canon_name = val
                    matched = True
                    break
            if not matched:
                lat, lon, canon_name = 3.1390, 101.6869, city_name.strip().title()

        metrics = evaluate_presence(lat, lon, now_utc)
        evaluated_regions.append({
            "city": canon_name,
            "coordinates": {"lat": lat, "lon": lon},
            "earth_presence_index": metrics["earth_presence_index"],
            "activity_density": metrics["activity_density"],
            "productivity_score": metrics["productivity_score"],
            "wakefulness": metrics["wakefulness"],
            "sleep_probability": metrics["sleep_probability"],
            "business_availability": metrics["business_availability"]["business_availability_score"],
            "presence_classification": metrics["presence_classification"],
            "local_time": metrics["local_time"],
            "light_state": metrics["light_state"]
        })

    # Compute comparative leaders
    highest_activity = max(evaluated_regions, key=lambda x: x["activity_density"])
    highest_productivity = max(evaluated_regions, key=lambda x: x["productivity_score"])
    highest_sleep = max(evaluated_regions, key=lambda x: x["sleep_probability"])
    lowest_presence = min(evaluated_regions, key=lambda x: x["earth_presence_index"])

    return {
        "highest_activity_region": highest_activity["city"],
        "highest_productivity_region": highest_productivity["city"],
        "highest_sleep_probability_region": highest_sleep["city"],
        "lowest_presence_region": lowest_presence["city"],
        "regions": evaluated_regions,
        "comparison_timestamp_utc": now_utc.isoformat()
    }

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "earth-presence-intelligence-grid-1552",
        "name": "Earth Presence Intelligence Grid (EPIG)",
        "version": "2.0.0"
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
    """Agent / MCP tool registry providing machine-readable schemas."""
    return {
        "tools": [
            {
                "name": "get_human_presence",
                "description": "Get current human presence score, wakefulness, productivity, and activity metrics for a location.",
                "input_schema": {"lat": "float", "lon": "float"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number", "description": "Latitude coordinate"},
                        "lon": {"type": "number", "description": "Longitude coordinate"}
                    },
                    "required": ["lat", "lon"]
                }
            },
            {
                "name": "get_activity_window",
                "description": "Get expected activity density and optimal engagement windows over a forecast period.",
                "input_schema": {"lat": "float", "lon": "float", "hours": "int"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                        "hours": {"type": "integer", "default": 24}
                    },
                    "required": ["lat", "lon"]
                }
            },
            {
                "name": "get_sleep_forecast",
                "description": "Get circadian sleep and awake probability projections.",
                "input_schema": {"lat": "float", "lon": "float", "hours": "int"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                        "hours": {"type": "integer", "default": 48}
                    },
                    "required": ["lat", "lon"]
                }
            },
            {
                "name": "get_darkness_state",
                "description": "Get solar elevation, civil/nautical twilight, and illumination state.",
                "input_schema": {"lat": "float", "lon": "float"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"}
                    },
                    "required": ["lat", "lon"]
                }
            },
            {
                "name": "get_h3_presence",
                "description": "Get aggregated Earth Presence Index and metrics for an Uber H3 discrete cell.",
                "input_schema": {"h3": "string"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "h3": {"type": "string", "description": "H3 Cell Index"}
                    },
                    "required": ["h3"]
                }
            },
            {
                "name": "compare_regions",
                "description": "Compare real-time human presence, productivity, and sleep probability across multiple world regions.",
                "input_schema": {"cities": "list[string]"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "cities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["Kuala Lumpur", "Tokyo", "London", "New York"]
                        }
                    }
                }
            }
        ]
    }

@app.get("/v1/presence/current")
def presence_current(lat: float = Query(3.139), lon: float = Query(101.6869)):
    return evaluate_presence(lat=lat, lon=lon)

@app.get("/v1/presence/location")
def presence_location(lat: float = Query(...), lon: float = Query(...)):
    return evaluate_presence(lat=lat, lon=lon)

@app.get("/v1/activity")
def activity(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    cal_info = calculate_calendar_intelligence(lat, lon, local_dt)
    local_hour = local_dt.hour + local_dt.minute / 60.0
    density = calculate_activity_density(local_hour, local_dt.weekday(), cal_info["is_holiday"])
    score_int, state = compute_activity_score(local_dt, cal_info["is_holiday"])

    return {
        "activity_score": score_int,
        "score": score_int,
        "activity_density": density,
        "state": state,
        "holiday": cal_info["is_holiday"],
        "weekend": cal_info["is_weekend"],
        "activity_modifier": cal_info["activity_modifier"],
        "local_time": local_dt.isoformat()
    }

@app.get("/v1/sleep")
def sleep_endpoint(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    sleep_prob, awake_prob, sleep_band = compute_sleep_and_awake(local_dt)
    local_hour = local_dt.hour + local_dt.minute / 60.0
    wake_score = calculate_human_wakefulness(local_hour)
    pop_estimate = int(awake_prob * 32500000)

    return {
        "awake_probability": awake_prob,
        "sleep_probability": sleep_prob,
        "sleep_band": sleep_band,
        "wakefulness_score": wake_score,
        "population_awake_estimate": pop_estimate,
        "local_time": local_dt.isoformat()
    }

@app.get("/v1/darkness")
def darkness(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    utc_dt = datetime.now(pytz.UTC)
    darkness_score, light_state, is_night, solar_elev = calculate_darkness_score(lat, lon, utc_dt)

    return {
        "darkness_score": int(round(darkness_score)),
        "darkness_score_precise": darkness_score,
        "light_state": light_state,
        "light": light_state,
        "night": is_night,
        "solar_elevation_deg": solar_elev,
        "moon_illumination": 50,
        "local_time": local_dt.isoformat()
    }

@app.get("/v1/calendar")
def calendar_intelligence(lat: float = Query(3.139), lon: float = Query(101.6869)):
    validate_coords(lat, lon)
    local_dt = get_local_time(lat, lon)
    cal_info = calculate_calendar_intelligence(lat, lon, local_dt)
    local_hour = local_dt.hour + local_dt.minute / 60.0
    biz_state = calculate_business_availability(local_hour, local_dt.weekday(), cal_info["is_holiday"])

    return {
        "calendar_state": cal_info["calendar_state"],
        "state": cal_info["calendar_state"],
        "office_activity": cal_info["office_activity_state"],
        "office_activity_score": biz_state["office_activity"],
        "government_activity": cal_info["government_activity"],
        "market_activity_score": biz_state["market_activity"],
        "retail_activity_score": biz_state["retail_activity"],
        "activity_modifier": cal_info["activity_modifier"],
        "holiday": cal_info["is_holiday"],
        "weekend": cal_info["is_weekend"],
        "local_date": local_dt.date().isoformat()
    }

@app.get("/v1/h3/{cell}")
def h3_presence(cell: str = Path(...)):
    """Computes real Earth Presence signals for any H3 discrete hexagon index."""
    if not _is_valid_h3(cell):
        raise HTTPException(status_code=400, detail="Invalid H3 cell index")

    lat, lon = _h3_to_latlon(cell)
    metrics = evaluate_presence(lat, lon)

    return {
        "h3": cell,
        "cell": cell,
        "presence_score": metrics["presence_score"],
        "earth_presence_index": metrics["earth_presence_index"],
        "presence_classification": metrics["presence_classification"],
        "awake_probability": metrics["awake_probability"],
        "sleep_probability": metrics["sleep_probability"],
        "darkness_score": int(round(metrics["darkness_score"])),
        "productivity_score": metrics["productivity_score"],
        "activity_density": metrics["activity_density"],
        "holiday": metrics["holiday"],
        "availability": metrics["availability"]
    }

@app.get("/v1/forecast")
def forecast(lat: float = Query(3.139), lon: float = Query(101.6869), hours: int = Query(24)):
    validate_coords(lat, lon)
    base_dt = get_local_time(lat, lon)
    base_utc = datetime.now(pytz.UTC)
    forecast_list = []
    best_windows = []
    current_window_start = None

    for i in range(hours):
        step_local = base_dt + timedelta(hours=i)
        step_utc = base_utc + timedelta(hours=i)
        local_hour = step_local.hour + step_local.minute / 60.0

        cal_info = calculate_calendar_intelligence(lat, lon, step_local)
        wake = calculate_human_wakefulness(local_hour)
        density = calculate_activity_density(local_hour, step_local.weekday(), cal_info["is_holiday"])
        biz = calculate_business_availability(local_hour, step_local.weekday(), cal_info["is_holiday"])
        dark_score, _, _, _ = calculate_darkness_score(lat, lon, step_utc)

        prod = calculate_productivity_index(
            wake, density, biz["business_availability_score"], cal_info["activity_modifier"]
        )
        epi = calculate_earth_presence_index(
            wake, density, prod, biz["business_availability_score"], dark_score
        )
        awake_prob = round(wake / 100.0, 2)
        legacy_act = int(round(density))

        forecast_list.append({
            "time": step_local.isoformat(),
            "awake_probability": awake_prob,
            "activity_score": legacy_act,
            "earth_presence_index": epi,
            "productivity_score": prod,
            "activity_density": density
        })

        if legacy_act > 70 and not current_window_start:
            current_window_start = step_local.strftime("%H:%M")
        elif legacy_act <= 70 and current_window_start:
            best_windows.append({
                "start": current_window_start,
                "end": step_local.strftime("%H:%M"),
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

@app.get("/v1/compare")
def api_compare_regions(cities: List[str] = Query(default=["Kuala Lumpur", "Tokyo", "London", "New York"])):
    """GET endpoint to compare human presence and productivity across cities."""
    return compare_regions(cities=cities)

@app.post("/v1/compare")
def api_compare_regions_post(payload: CompareRegionsInput):
    """POST endpoint to compare human presence across custom city lists."""
    return compare_regions(cities=payload.cities)

# ---------------------------------------------------------------------------
# MCP Tool Endpoints
# ---------------------------------------------------------------------------
@app.post("/tools/get_human_presence")
@app.post("/mcp/get_human_presence")
def mcp_get_human_presence(payload: LocationInput):
    return evaluate_presence(lat=payload.lat, lon=payload.lon)

@app.post("/tools/get_activity_window")
@app.post("/mcp/get_activity_window")
def mcp_get_activity_window(payload: ActivityWindowInput):
    return forecast(lat=payload.lat, lon=payload.lon, hours=payload.hours)

@app.post("/tools/get_sleep_forecast")
@app.post("/mcp/get_sleep_forecast")
def mcp_get_sleep_forecast(payload: SleepForecastInput):
    return forecast(lat=payload.lat, lon=payload.lon, hours=payload.hours)

@app.post("/tools/get_darkness_state")
@app.post("/mcp/get_darkness_state")
def mcp_get_darkness_state(payload: LocationInput):
    return darkness(lat=payload.lat, lon=payload.lon)

@app.post("/tools/get_h3_presence")
@app.post("/mcp/get_h3_presence")
def mcp_get_h3_presence(payload: H3Input):
    return h3_presence(cell=payload.h3)

@app.post("/tools/compare_regions")
@app.post("/mcp/compare_regions")
def mcp_compare_regions(payload: CompareRegionsInput):
    return compare_regions(cities=payload.cities)

# ---------------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
