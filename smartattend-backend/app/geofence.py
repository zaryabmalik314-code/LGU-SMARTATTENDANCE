"""
Campus boundary geofencing.
Reuses your 16-point polygon idea from SCEMS, plus GPS-noise handling.
"""
import math
from typing import List, Tuple
from .schemas import GPSReading

# Real campus boundary — Lahore Garrison University, DHA Phase 6, Lahore.
# Traced from actual satellite imagery of the campus perimeter wall (21 points).
# ORIGINAL — do not modify.
CAMPUS_BOUNDARY: List[Tuple[float, float]] = [
    (31.463801, 74.441493),
    (31.464043, 74.441724),
    (31.464234, 74.441990),
    (31.464446, 74.442265),
    (31.464764, 74.442671),
    (31.464844, 74.442842),
    (31.464454, 74.443324),
    (31.464066, 74.443752),
    (31.463730, 74.444093),
    (31.463483, 74.444375),
    (31.463255, 74.444610),
    (31.463003, 74.444890),
    (31.462979, 74.444712),
    (31.463154, 74.444091),
    (31.463268, 74.443655),
    (31.463371, 74.443291),
    (31.463398, 74.442963),
    (31.463580, 74.442352),
    (31.463656, 74.442015),
    (31.463745, 74.441712),
]

# Home geofence (dev/test only — a 50m octagon around Zaryab's house) has been
# REMOVED for the production rollout. Only the real campus boundary is allowed.
# If you ever need to test from another location again, add a boundary here and
# include it in ALLOWED_BOUNDARIES below — never ship that to real faculty.

# All allowed zones — point must be inside at least one.
ALLOWED_BOUNDARIES = [CAMPUS_BOUNDARY]

MAX_ACCEPTABLE_ACCURACY_M = 100.0  # reject readings noisier than this
BOUNDARY_BUFFER_M = 50.0  # treat points within this distance of edge as "inside" too


def pick_best_reading(readings: List[GPSReading]) -> GPSReading:
    """Pick lowest-accuracy-value (most precise) reading from a batch."""
    if not readings:
        raise ValueError("No GPS readings provided")
    return min(readings, key=lambda r: r.accuracy)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in meters between two lat/lng points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def point_in_polygon(lat: float, lng: float, polygon: List[Tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon check."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lng_i = polygon[i]
        lat_j, lng_j = polygon[j]
        intersect = ((lng_i > lng) != (lng_j > lng)) and (
            lat < (lat_j - lat_i) * (lng - lng_i) / (lng_j - lng_i + 1e-15) + lat_i
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def is_allowed_location(lat: float, lng: float) -> bool:
    """Returns True if point is inside any of the allowed boundaries."""
    return any(point_in_polygon(lat, lng, boundary) for boundary in ALLOWED_BOUNDARIES)


def distance_to_polygon_edge_m(lat: float, lng: float, polygon: List[Tuple[float, float]]) -> float:
    """Shortest distance from point to any polygon edge, in meters."""
    min_dist = float("inf")
    n = len(polygon)
    for i in range(n):
        lat1, lng1 = polygon[i]
        lat2, lng2 = polygon[(i + 1) % n]
        for lat_p, lng_p in [(lat1, lng1), (lat2, lng2), ((lat1 + lat2) / 2, (lng1 + lng2) / 2)]:
            d = haversine_m(lat, lng, lat_p, lng_p)
            min_dist = min(min_dist, d)
    return min_dist


def nearest_boundary_distance_m(lat: float, lng: float) -> float:
    """Shortest distance to any allowed boundary edge."""
    return min(
        distance_to_polygon_edge_m(lat, lng, boundary)
        for boundary in ALLOWED_BOUNDARIES
    )


def check_location(reading: GPSReading) -> dict:
    """
    Returns dict: {allowed: bool, reason: str, distance_to_boundary_m: float}
    Logic:
      1. Reject if GPS accuracy too poor to trust.
      2. Accept if strictly inside any allowed polygon.
      3. If outside but within BOUNDARY_BUFFER_M of any edge, accept (accounts for GPS drift).
      4. Otherwise reject.
    """
    if reading.accuracy > MAX_ACCEPTABLE_ACCURACY_M:
        return {
            "allowed": False,
            "reason": f"gps_too_noisy ({reading.accuracy:.0f}m > {MAX_ACCEPTABLE_ACCURACY_M:.0f}m)",
            "distance_to_boundary_m": None,
        }

    inside = is_allowed_location(reading.latitude, reading.longitude)
    dist = nearest_boundary_distance_m(reading.latitude, reading.longitude)

    if inside:
        return {"allowed": True, "reason": "inside_boundary", "distance_to_boundary_m": dist}

    if dist <= BOUNDARY_BUFFER_M:
        return {"allowed": True, "reason": "within_buffer_zone", "distance_to_boundary_m": dist}

    return {"allowed": False, "reason": "outside_boundary", "distance_to_boundary_m": dist}


# Anti-GPS-spoofing: impossible-movement detection.
MAX_PLAUSIBLE_SPEED_KMH = 160.0
MIN_MINUTES_TO_CHECK = 1.0
MAX_HOURS_SINCE_LAST_TO_CHECK = 12.0


def check_impossible_movement(
    prev_lat: float, prev_lng: float, prev_time, new_lat: float, new_lng: float, new_time
) -> dict:
    """
    Returns {"flagged": bool, "reason": str | None, "speed_kmh": float | None}
    based on implied travel speed between two recorded points.
    """
    elapsed_hours = (new_time - prev_time).total_seconds() / 3600.0
    elapsed_minutes = elapsed_hours * 60

    if elapsed_minutes < MIN_MINUTES_TO_CHECK or elapsed_hours > MAX_HOURS_SINCE_LAST_TO_CHECK:
        return {"flagged": False, "reason": None, "speed_kmh": None}

    distance_km = haversine_m(prev_lat, prev_lng, new_lat, new_lng) / 1000.0
    speed_kmh = distance_km / elapsed_hours

    if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
        return {
            "flagged": True,
            "reason": (
                f"implied speed {speed_kmh:.0f} km/h over {distance_km:.1f}km "
                f"in {elapsed_minutes:.1f}min exceeds plausible travel speed"
            ),
            "speed_kmh": speed_kmh,
        }

    return {"flagged": False, "reason": None, "speed_kmh": speed_kmh}


GPS_SPOOF_MIN_READINGS = 3
GPS_SPOOF_JITTER_FLOOR = 0.0000008  # ~0.09m — real GPS always drifts more than this


def check_gps_spoofing(readings: List["GPSReading"]) -> dict:
    """
    Analyzes multi-reading GPS consistency + metadata signals.
    Real GPS drifts 1-15m between samples even standing still.
    Mock location apps produce frozen coordinates and missing sensor data.
    Returns {"spoofed": bool, "reason": str|None}.
    """
    if len(readings) < GPS_SPOOF_MIN_READINGS:
        return {"spoofed": False, "reason": None}

    if any(getattr(r, "is_mock", None) is True for r in readings):
        return {"spoofed": True, "reason": "mock_provider_flag"}

    lats = [r.latitude for r in readings]
    lngs = [r.longitude for r in readings]
    accs = [r.accuracy for r in readings]

    lat_mean = sum(lats) / len(lats)
    lng_mean = sum(lngs) / len(lngs)
    lat_std = math.sqrt(sum((x - lat_mean) ** 2 for x in lats) / len(lats))
    lng_std = math.sqrt(sum((x - lng_mean) ** 2 for x in lngs) / len(lngs))

    if lat_std < GPS_SPOOF_JITTER_FLOOR and lng_std < GPS_SPOOF_JITTER_FLOOR:
        return {"spoofed": True, "reason": "gps_coordinates_frozen"}

    unique_accs = len(set(round(a, 1) for a in accs))
    if unique_accs == 1 and lat_std < GPS_SPOOF_JITTER_FLOOR * 10 and lng_std < GPS_SPOOF_JITTER_FLOOR * 10:
        return {"spoofed": True, "reason": "gps_accuracy_frozen"}

    all_alt_null = all(getattr(r, "altitude", None) is None for r in readings)
    all_spd_null = all(getattr(r, "speed", None) is None for r in readings)
    if all_alt_null and all_spd_null and unique_accs == 1 and lat_std < GPS_SPOOF_JITTER_FLOOR and lng_std < GPS_SPOOF_JITTER_FLOOR:
        return {"spoofed": True, "reason": "gps_metadata_absent"}

    return {"spoofed": False, "reason": None}
