# Anvil Pipeline: 日次でUSGS + ISCから新規地震イベントを取得するスクリプト
#
# 設計:
# - USGSとISC、両方を取得する(README方針: マージしない、event_idにソース接頭辞を付与)
# - 型の揺れ(時刻フォーマット混在、数値/文字列混入)に強い変換関数を必ず通す
# - 前回実行日時を state.json に記録し、次回はその続きから取得する(差分取得)

import requests
import pandas as pd
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
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"anvil_daily_{now.strftime('%Y%m%d')}.parquet")
        combined.to_parquet(out_path, index=False)
        print(f"保存: {out_path}({len(combined)} 件)")

    save_last_run(now)
    print("完了")


if __name__ == "__main__":
    main()
