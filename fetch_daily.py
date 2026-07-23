# Anvil Pipeline: 日次でUSGS + ISCから新規地震イベントを取得するスクリプト
#
# 設計:
# - USGSとISC、両方を取得する(README方針: マージしない、event_idにソース接頭辞を付与)
# - 型の揺れ(時刻フォーマット混在、数値/文字列混入)に強い変換関数を必ず通す
# - 前回実行日時を state.json に記録し、次回はその続きから取得する(差分取得)

import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta

STATE_FILE = "state.json"
OUTPUT_DIR = "data"

# ============================================================
# 型の揺れに強い変換関数
# ============================================================
def safe_parse_time(series):
    """時刻列を、フォーマットの揺れ(小数秒の有無等)に関わらず統一的にパースする"""
    return pd.to_datetime(series, errors='coerce', utc=True, format='mixed')

def safe_numeric(series):
    """数値であるべき列を、文字列混入があっても数値に統一する(変換不能はNaN)"""
    return pd.to_numeric(series, errors='coerce')

def safe_string(series):
    """文字列であるべき列を、型を強制的に統一する"""
    return series.astype(str).where(series.notna(), None)


# ============================================================
# OMNI2(宇宙天気)取得 — CDAWeb HAPI経由
# 恭介さんの既存ノートブック(2026-07-18作成)で実証済みのロジックをそのまま移植
# ============================================================
PARAM_MAP = {
    'ABS_B1800': 'IMF_B', 'BX_GSE1800': 'BX_GSE', 'BY_GSM1800': 'BY_GSM', 'BZ_GSM1800': 'BZ_GSM',
    'THETA_AV1800': 'B_LAT_GSE', 'PHI_AV1800': 'B_LON_GSE',
    'T1800': 'SW_TEMP', 'N1800': 'SW_DENSITY', 'V1800': 'SW_SPEED',
    'THETA-V1800': 'FLOW_LATITUDE', 'PHI-V1800': 'FLOW_LONGITUDE',
    'Ratio1800': 'ALPHA_PROTON_RATIO', 'Pressure1800': 'FLOW_PRESSURE',
    'E1800': 'EY', 'Beta1800': 'PLASMA_BETA',
    'Mach_num1800': 'ALFVEN_MACH', 'Mgs_mach_num1800': 'MAGNETOSONIC_MACH',
    'SIGMA-B1800': 'RMS_B', 'SIGMA-ABS_B1800': 'RMS_VECTOR',
    'SIGMA-Bx1800': 'RMS_BX', 'SIGMA-By1800': 'RMS_BY', 'SIGMA-Bz1800': 'RMS_BZ',
    'SIGMA-T1800': 'SIGMA_T', 'SIGMA-N1800': 'SIGMA_N', 'SIGMA-V1800': 'SIGMA_V',
    'SIGMA-THETA-V1800': 'SIGMA_FLOW_LATITUDE', 'SIGMA-PHI-V1800': 'SIGMA_FLOW_LONGITUDE',
    'SIGMA-ratio1800': 'SIGMA_RATIO',
    'R1800': 'SUNSPOT', 'F10_INDEX1800': 'F107', 'KP1800': 'KP',
    'DST1800': 'DST', 'AE1800': 'AE', 'AL_INDEX1800': 'AL', 'AU_INDEX1800': 'AU',
    'AP_INDEX1800': 'AP', 'PC_N_INDEX1800': 'PC',
    'Solar_Lyman_alpha1800': 'LYMAN_ALPHA', 'Proton_QI1800': 'QI',
}

def fetch_omni(start_time, end_time):
    """CDAWeb HAPIから、差分期間ぶんのOMNI2データを取得し、日次平均に集計する"""
    from hapiclient import hapi
    params = ','.join(PARAM_MAP.keys())

    data, meta = hapi(
        'https://cdaweb.gsfc.nasa.gov/hapi', 'OMNI2_H0_MRG1HR',
        params,
        start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        logging=False,
    )

    df = pd.DataFrame(data)
    if len(df) == 0:
        return df

    df['Time'] = pd.to_datetime(df['Time'].str.decode('utf-8'), utc=True)

    # fill値(欠測を表す特殊値)をNaNに変換
    for p in meta['parameters'][1:]:
        f = p.get('fill')
        if f is not None and p['name'] in df.columns:
            v = df[p['name']].astype(float)
            df[p['name']] = v.mask(np.isclose(v, float(f)))

    df = df.rename(columns=PARAM_MAP)
    daily = df.set_index('Time').resample('D').mean().reset_index()
    daily['KP'] = daily['KP'] / 10  # OMNI2のKpは10倍値で格納されているため
    return daily


# ============================================================
# ASTRO(月・太陽)計算 — skyfieldによる決定論的計算(API通信不要)
# 恭介さんの既存ノートブック(2026-07-18)のcompute_astro()をそのまま移植
# ============================================================
def compute_astro(event_ids, dt_series):
    """発生時刻ちょうどの天文値5種を一括計算する"""
    from skyfield.api import load
    from skyfield import almanac

    eph = load('de421.bsp')
    ts = load.timescale()
    earth, moon, sun = eph['earth'], eph['moon'], eph['sun']

    t = ts.from_datetimes(dt_series.dt.to_pydatetime().tolist())
    e = earth.at(t)
    m = e.observe(moon).apparent()
    s = e.observe(sun).apparent()

    phase_deg = almanac.moon_phase(eph, t).degrees
    illum = m.fraction_illuminated(sun) * 100
    moon_dist = m.distance().km
    sun_dist = s.distance().au
    _, sun_dec, _ = s.radec()

    def phase_name(deg):
        if deg < 22.5 or deg >= 337.5: return 'New Moon'
        elif deg < 67.5: return 'Waxing Crescent'
        elif deg < 112.5: return 'First Quarter'
        elif deg < 157.5: return 'Waxing Gibbous'
        elif deg < 202.5: return 'Full Moon'
        elif deg < 247.5: return 'Waning Gibbous'
        elif deg < 292.5: return 'Last Quarter'
        else: return 'Waning Crescent'

    astro = pd.DataFrame({
        'event_id': event_ids.values,
        'MOON_PHASE_DEG': phase_deg,
        'MOON_ILLUMINATION': illum,
        'MOON_DISTANCE_KM': moon_dist,
        'EARTH_SUN_DISTANCE_AU': sun_dist,
        'SOLAR_DECLINATION_DEG': sun_dec.degrees,
    })
    astro['MOON_PHASE'] = astro['MOON_PHASE_DEG'].apply(phase_name)
    return astro


# ============================================================
# 前回実行日時の管理
# ============================================================
def load_last_run():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        return datetime.fromisoformat(state['last_run'])
    # 初回は昨日からとする
    return datetime.now(timezone.utc) - timedelta(days=1)

def save_last_run(dt):
    with open(STATE_FILE, 'w') as f:
        json.dump({'last_run': dt.isoformat()}, f)


# ============================================================
# USGS取得(FDSN Event, GeoJSON形式)
# ============================================================
def fetch_usgs(start_time, end_time):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 2.5,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    rows = []
    for f in features:
        props = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [None, None, None])
        rows.append({
            "event_id": f"USGS_{props.get('ids','').strip(',').split(',')[0].lstrip('us') or f.get('id')}",
            "time": pd.to_datetime(props.get("time"), unit="ms", utc=True, errors='coerce'),
            "latitude": coords[1],
            "longitude": coords[0],
            "depth_km": coords[2],
            "mag": props.get("mag"),
            "mag_type": props.get("magType"),
            "place": props.get("place"),
            "status": props.get("status"),
            "source_catalog": "USGS",
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["latitude"] = safe_numeric(df["latitude"])
    df["longitude"] = safe_numeric(df["longitude"])
    df["depth_km"] = safe_numeric(df["depth_km"])
    df["mag"] = safe_numeric(df["mag"])
    return df


# ============================================================
# ISC取得(FDSN Event, text形式)
# ============================================================
def fetch_isc(start_time, end_time):
    url = "https://www.isc.ac.uk/fdsnws/event/1/query"
    params = {
        "format": "text",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 2.5,
        "catalog": "ISC",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    lines = [l for l in resp.text.splitlines() if l.strip() and not l.startswith("#EventID")]
    rows = []
    # 公式FDSN text形式の列順(IRIS/GeoNet/ISC共通仕様で確認済み):
    # EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName
    for line in lines:
        parts = line.split("|")
        if len(parts) < 13:
            continue
        rows.append({
            "event_id": f"ISC_{parts[0].strip()}",
            "time": parts[1].strip(),
            "latitude": parts[2].strip(),
            "longitude": parts[3].strip(),
            "depth_km": parts[4].strip(),
            "Author": parts[5].strip(),
            "Catalog": parts[6].strip(),
            "Contributor": parts[7].strip(),
            "ContributorID": parts[8].strip(),
            "mag_type": parts[9].strip(),
            "mag": parts[10].strip(),
            "MagAuthor": parts[11].strip(),
            "place": parts[12].strip(),
            "source_catalog": "ISC",
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["time"] = safe_parse_time(df["time"])
    df["latitude"] = safe_numeric(df["latitude"])
    df["longitude"] = safe_numeric(df["longitude"])
    df["depth_km"] = safe_numeric(df["depth_km"])
    df["mag"] = safe_numeric(df["mag"])
    return df


# ============================================================
# メイン処理
# ============================================================
def main():
    last_run = load_last_run()
    now = datetime.now(timezone.utc)
    print(f"取得期間: {last_run.isoformat()} 〜 {now.isoformat()}")

    print("USGSを取得中...")
    usgs_df = fetch_usgs(last_run, now)
    print(f"  USGS新規: {len(usgs_df)} 件")

    print("ISCを取得中...")
    try:
        isc_df = fetch_isc(last_run, now)
        print(f"  ISC新規: {len(isc_df)} 件")
    except Exception as e:
        print(f"  ISC取得エラー(スキップして続行): {e}")
        isc_df = pd.DataFrame()

    combined = pd.concat([usgs_df, isc_df], ignore_index=True)

    if len(combined) == 0:
        print("新規イベントはありませんでした。")
    else:
        print("ASTRO(月・太陽)を計算中...")
        try:
            astro_df = compute_astro(combined['event_id'], combined['time'])
            combined = combined.merge(astro_df, on='event_id', how='left')
            print(f"  ASTRO計算完了: {len(astro_df)} 件")
        except Exception as e:
            print(f"  ASTRO計算エラー(スキップして続行): {e}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"anvil_daily_{now.strftime('%Y%m%d')}.parquet")
        combined.to_parquet(out_path, index=False)
        print(f"保存: {out_path}({len(combined)} 件)")

    print("OMNI2(宇宙天気)を取得中...")
    try:
        omni_df = fetch_omni(last_run, now)
        print(f"  OMNI2新規: {len(omni_df)} 日分")
        if len(omni_df) > 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            omni_out_path = os.path.join(OUTPUT_DIR, f"omni_daily_{now.strftime('%Y%m%d')}.parquet")
            omni_df.to_parquet(omni_out_path, index=False)
            print(f"保存: {omni_out_path}")
    except Exception as e:
        print(f"  OMNI2取得エラー(スキップして続行): {e}")

    save_last_run(now)
    print("完了")


if __name__ == "__main__":
    main()
