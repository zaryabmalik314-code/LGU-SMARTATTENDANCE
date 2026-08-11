"""
Campus boundary geofencing.
Reuses your 16-point polygon idea from SCEMS, plus GPS-noise handling.
"""
import math
from typing import List, Tuple
from .schemas import GPSReading

# Real campus boundary — Lahore Garrison University, DHA Phase 6, Lahore.
# ORIGINAL 21-point trace of the perimeter wall (from satellite imagery).
# Kept here for reference — CAMPUS_BOUNDARY below is derived from it.
CAMPUS_BOUNDARY_ORIGINAL: List[Tuple[float, float]] = [
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

# ACTIVE boundary: the original trace expanded outward ~50 m and smoothed to
# 120 points, so faculty in any department (incl. buildings just past the tight
# wall-trace) fall inside. The extra points don't add area — the ~50 m outward
# offset does; the density just keeps the enlarged edge smooth. Regenerate with
# the same offset+interpolation if the original trace ever changes.
CAMPUS_BOUNDARY: List[Tuple[float, float]] = [
    (31.463813, 74.440967),
    (31.463868, 74.441008),
    (31.463923, 74.441048),
    (31.463978, 74.441089),
    (31.464034, 74.44113),
    (31.464089, 74.44117),
    (31.464144, 74.441211),
    (31.464191, 74.441262),
    (31.464239, 74.441312),
    (31.464286, 74.441363),
    (31.464333, 74.441414),
    (31.464381, 74.441464),
    (31.464428, 74.441515),
    (31.464482, 74.441575),
    (31.464535, 74.441635),
    (31.464589, 74.441695),
    (31.464642, 74.441756),
    (31.464696, 74.441816),
    (31.464749, 74.441876),
    (31.464821, 74.441976),
    (31.464893, 74.442076),
    (31.464965, 74.442175),
    (31.465037, 74.442275),
    (31.465109, 74.442375),
    (31.465181, 74.442475),
    (31.465198, 74.442516),
    (31.465214, 74.442557),
    (31.465231, 74.442598),
    (31.465248, 74.442639),
    (31.465264, 74.44268),
    (31.465281, 74.442721),
    (31.465216, 74.442841),
    (31.465151, 74.442961),
    (31.465086, 74.443081),
    (31.465022, 74.4432),
    (31.464957, 74.44332),
    (31.464892, 74.44344),
    (31.464792, 74.443568),
    (31.464692, 74.443696),
    (31.464591, 74.443824),
    (31.464491, 74.443951),
    (31.464391, 74.444079),
    (31.464291, 74.444207),
    (31.464194, 74.444276),
    (31.464098, 74.444344),
    (31.464001, 74.444413),
    (31.463904, 74.444482),
    (31.463808, 74.44455),
    (31.463711, 74.444619),
    (31.463654, 74.444663),
    (31.463596, 74.444707),
    (31.463539, 74.444751),
    (31.463482, 74.444796),
    (31.463424, 74.44484),
    (31.463367, 74.444884),
    (31.46332, 74.44492),
    (31.463273, 74.444955),
    (31.463227, 74.444991),
    (31.46318, 74.445027),
    (31.463133, 74.445062),
    (31.463086, 74.445098),
    (31.463038, 74.445142),
    (31.46299, 74.445185),
    (31.462943, 74.445228),
    (31.462895, 74.445272),
    (31.462847, 74.445315),
    (31.462799, 74.445359),
    (31.462791, 74.445327),
    (31.462783, 74.445295),
    (31.462775, 74.445262),
    (31.462768, 74.44523),
    (31.46276, 74.445198),
    (31.462752, 74.445166),
    (31.462774, 74.445057),
    (31.462796, 74.444948),
    (31.462818, 74.444839),
    (31.46284, 74.44473),
    (31.462862, 74.444621),
    (31.462884, 74.444512),
    (31.462892, 74.444427),
    (31.4629, 74.444343),
    (31.462908, 74.444258),
    (31.462916, 74.444173),
    (31.462924, 74.444089),
    (31.462932, 74.444004),
    (31.462934, 74.443912),
    (31.462936, 74.443821),
    (31.462938, 74.443729),
    (31.46294, 74.443637),
    (31.462942, 74.443546),
    (31.462944, 74.443454),
    (31.462951, 74.443339),
    (31.462957, 74.443223),
    (31.462964, 74.443107),
    (31.46297, 74.442992),
    (31.462976, 74.442877),
    (31.462983, 74.442761),
    (31.463063, 74.442608),
    (31.463143, 74.442455),
    (31.463223, 74.442303),
    (31.463302, 74.44215),
    (31.463382, 74.441997),
    (31.463462, 74.441844),
    (31.463486, 74.441785),
    (31.46351, 74.441727),
    (31.463534, 74.441668),
    (31.463558, 74.441609),
    (31.463582, 74.441551),
    (31.463606, 74.441492),
    (31.463628, 74.441441),
    (31.46365, 74.44139),
    (31.463672, 74.441339),
    (31.463694, 74.441287),
    (31.463716, 74.441236),
    (31.463738, 74.441185),
    (31.46375, 74.441149),
    (31.463763, 74.441112),
    (31.463775, 74.441076),
    (31.463788, 74.44104),
    (31.4638, 74.441003),
]

# Home geofence (dev/test only — a 50m octagon around Zaryab's house) has been
# REMOVED for the production rollout. Only the real campus boundary is allowed.
# If you ever need to test from another location again, add a boundary here and
# include it in ALLOWED_BOUNDARIES below — never ship that to real faculty.

# All allowed zones — point must be inside at least one.
ALLOWED_BOUNDARIES = [CAMPUS_BOUNDARY]

MAX_ACCEPTABLE_ACCURACY_M = 100.0  # reject readings noisier than this
# The polygon itself now carries the campus coverage margin (expanded ~50 m),
# so this buffer only absorbs everyday GPS jitter near the edge. The frontend
# (index.html GEO_BUFFER_M) mirrors this exact value so the app's gate never
# disagrees with the server.
BOUNDARY_BUFFER_M = 30.0


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
