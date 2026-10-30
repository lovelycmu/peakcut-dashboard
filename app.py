import os
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 설정값
# ============================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py와 같은 폴더의 CSV를 시간순으로 합쳐서 사용
UNIT = "kW"
WINTER_MONTHS = [12, 1, 2]
SUMMER_MONTHS = [7, 8, 9]
BLOCK = 15                 # 수요전력 산정 단위 (분)
# 피크 한계값 P_max = 전년도 여름(7~9월)·겨울(12~2월) 15분 평균 최대 수요
#   (한전 요금적용전력은 직전 12개월 중 여름·겨울철 최대수요로 정해지므로 두 계절을 한 '요금 연도'로 묶는다)
#   요금 연도 Y = Y년 7~9월 + Y년 12월 ~ Y+1년 2월
P_WARN_KW = 3096.0         # P_warn 고정값 (kW). 예상 수요가 이 값 이상 -> WARNING (유연부하 제어 시작)
WARN_RATIO = 0.90          # P_WARN_KW = None 이면 P_warn = 한계값 x 90%
# P_peak 범위 (팀 문서 '인수인계_P값_설정방식.md' 5절, 2024 겨울 = 전년도 기준 값. 그래프 표시용)
P_PEAKMAX_KW = 3107.0      # P_peak max: 최대이익의 90% 이상이 유지되는 얕은 쪽 경계
P_PEAKMIN_KW = 3101.5      # P_peak min: 이익이 급락(절벽)하기 직전의 깊은 쪽 경계
EMERG_RATIO = 1.00         # 한계값 100% 이상 -> EMERGENCY (유연 + 탄력 제어)
SMOOTH_MIN = 3             # 예상 수요 계산에 쓰는 최근 전력 평균 (분) - 1분 출렁임 완화
WINDOW_MIN = 60            # 1분 그래프에 보여줄 최근 분 수
WINDOW_BLOCKS = 8          # 15분 막대 그래프에 보여줄 최근 구간 수
FEE_WON_PER_KW_YEAR = 112523  # 기본요금 8,320원/kW·월 x 12 x 1.127 (팀 문서 7절)

RIGID = "[1] 경직성부하"
ELASTIC = "[2] 탄력부하"
FLEX = "[3] 유연부하"
SHORT = {RIGID: "경직", ELASTIC: "탄력", FLEX: "유연"}  # 관리자 확정 표에서 쓰는 짧은 이름
LONG = {v: k for k, v in SHORT.items()}
FAN_PUMP = ("Pump", "Fan", "Blower", "HVAC", "Chiller", "Condenser", "Cooler", "Collector")  # 상사법칙 표시 대상

SEASON_ALL = "여름+겨울 통합"
SEASON_SUMMER = "여름만 (7~9월)"
SEASON_WINTER = "겨울만 (12~2월)"

MODE_GREEDY = "우선순위대로 필요한 만큼만"
MODE_ALL = "대상 전부 일괄 제어"
FLEX_ONOFF = "ON/OFF 차단"
FLEX_SPEED = "속도 제어 (인버터, 최소 가동률까지 감속)"
ACTION_TEXT = {  # 조치 종류 -> (제어 상태, 제어 사유)
    "OFF_W": ("OFF (AI 자동)", "Warning: P_warn 초과 - 유연부하 차단"),
    "OFF": ("OFF (AI 자동)", "Emergency: 한계값 초과 예상 - 유연부하 차단"),
    "SPD_W": ("SPEED (AI 자동)", "Warning: P_warn 초과 - 유연부하 감속"),
    "SPD": ("SPEED (AI 자동)", "Emergency: 한계값 초과 예상 - 유연부하 감속"),
    "SHIFT": ("TIME_SHIFT (AI 자동)", "Emergency: 15분 지연 운전 (15분 뒤 복귀)"),
}


# ============================================================
# 1. 데이터 로드 / 부하 분류 (오프라인 전처리와 동일한 DBSCAN + K-Means)
# ============================================================
def p_warn(limit):
    """WARNING 기준값 (kW): 고정값이 있으면 고정값, 없으면 한계값의 90%."""
    return P_WARN_KW if P_WARN_KW is not None else WARN_RATIO * limit


def read_csv_auto(path):
    for enc in ("utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 읽을 수 없습니다 (utf-8, cp949 모두 실패)")


def read_factory(paths):
    """공장 CSV 여러 개(연도·계절별 파일)를 시간순으로 합친다. 같은 시각이 겹치면 앞 파일 값 사용.

    - 파일마다 컬럼이 조금 달라도 된다 (예: 여름 파일에만 유형별 합계 컬럼이 있음). 모든 파일에 있는 컬럼만 쓴다.
    - 유형별 합계 컬럼이 있는 파일에서는 설비 라벨을 먼저 읽어 둔다.
    """
    dfs = [read_csv_auto(p) for p in paths]
    labels = {}
    for p, d in zip(paths, dfs):
        if "Timestamp" not in d.columns:
            raise ValueError(f"{os.path.basename(p)}: Timestamp 컬럼이 없습니다")
        labels.update(file_labels(d, [c for c in d.columns[1:] if pd.api.types.is_numeric_dtype(d[c])]))
    common = [c for c in dfs[0].columns if all(c in d.columns for d in dfs)]
    df = pd.concat([d[common] for d in dfs], ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)
    return df, labels


def parse_schema(df):
    """(시각, 설비 컬럼, 총합 검증값) 반환. 이름에 합/Sum/Total이 들어간 컬럼은 설비가 아닌 합계로 본다."""
    ts = df["Timestamp"]
    skip = ("합", "Sum", "Total")
    cols = [
        c for c in df.columns[1:]
        if pd.api.types.is_numeric_dtype(df[c]) and not any(k in c for k in skip)
    ]
    totals = [c for c in df.columns if "Total" in c and pd.api.types.is_numeric_dtype(df[c])]
    ref_total = df[totals[0]] if totals else None
    return ts, cols, ref_total


def load_features(X):
    """설비별 행동 특징 (X: 1분 전력, 행=시각, 열=설비)."""
    tot = X.mean(axis=0).sum() or 1e-9
    rows = []
    for j in range(X.shape[1]):
        x = X[:, j]
        m = x.mean()
        hi = np.percentile(x, 99.5)
        s = pd.Series(x)
        r = (s - s.rolling(61, center=True, min_periods=1).mean()).to_numpy() # 운영 스케줄(60분 추세) 제거
        r = r - r.mean()
        f = np.fft.rfft(r, n=2 * len(r))
        ac = np.fft.irfft(f * np.conj(f))[:61]                                  # 자기상관 (지연 0~60분)
        rows.append({
            "Mean_Power": m,
            "Share": m / tot,
            "On_Ratio": float((x > 0.1 * hi).mean()) if hi > 0 else 0.0,
            "Periodicity": float(ac[3:].max() / ac[0]) if ac[0] > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def classify_loads(df_all, cols):
    """설비별 특징 -> DBSCAN(이상 설비 검토 표시) -> K-Means(k=3) -> 경직/탄력/유연 이름 붙이기.

    - 부하 비중(Share)       : 전체 평균 부하 중 이 설비 비율 (log로 군집화)
    - 가동률(On_Ratio)       : 최대치의 10% 이상으로 켜져 있는 시간 비율
    - 주기성(Periodicity)    : 60분 추세를 뺀 뒤 3~60분 지연 자기상관의 최댓값 (주기적으로 켜졌다 꺼지는 정도)
    - DBSCAN 이상 설비는 가장 가까운 K-Means 군집에 넣고 '검토 필요'로 표시 (관리자가 확정)
    - 이름: 부하 비중이 가장 큰 군집 = 경직, 나머지 중 주기성이 큰 군집 = 탄력, 남은 군집 = 유연
    """
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler

    f = load_features(np.nan_to_num(df_all[cols].to_numpy(dtype=float)))
    f.index = cols
    f["Max_Power"] = df_all[cols].groupby(np.arange(len(df_all)) // BLOCK).mean().max()  # 15분 평균 최대
    f["Log_Share"] = np.log10(f["Share"].clip(lower=1e-5))
    x_cols = ["Log_Share", "On_Ratio", "Periodicity"]

    z = StandardScaler().fit_transform(f[x_cols])
    noise = DBSCAN(eps=1.2, min_samples=2).fit_predict(z) == -1
    fit_rows = ~noise if (~noise).sum() >= 3 else np.ones(len(cols), dtype=bool)
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(z[fit_rows])
    f["Cluster"] = km.predict(z)  # 이상 설비도 가장 가까운 군집 중심으로 배정
    f["Review"] = noise

    g = f.groupby("Cluster")[["Log_Share", "Periodicity"]].mean()
    rigid_id = g["Log_Share"].idxmax()
    rest = g.drop(rigid_id)
    elastic_id = rest["Periodicity"].idxmax()
    label_map = {rigid_id: RIGID, elastic_id: ELASTIC}
    label_map.update({k: FLEX for k in rest.index if k != elastic_id})
    f["Load_Type"] = f["Cluster"].map(label_map)
    return f


def file_labels(df, cols):
    """파일에 유형별 합계 컬럼(예: '경직부하 b~f셀의 합(Rigid_Sum)')이 있으면 설비별 라벨로 읽는다.

    엑셀 열 범위(b~f)의 설비 합이 그 합계 컬럼과 실제로 같을 때만 라벨로 인정한다.
    """
    kinds = {"Rigid": RIGID, "Elastic": ELASTIC, "Flexible": FLEX}
    out = {}
    for c in df.columns:
        m = re.search(r"\b([a-zA-Z])\s*~\s*([a-zA-Z])", c)
        kind = next((v for k, v in kinds.items() if k in c), None)
        if not (m and kind):
            continue
        a, b = (ord(ch.lower()) - ord("a") for ch in m.groups())
        members = [df.columns[i] for i in range(a, b + 1) if i < len(df.columns)]
        if not members or any(x not in cols for x in members):
            continue
        err = float((df[members].sum(axis=1) - df[c]).abs().max())
        if err <= 1e-6 * max(1.0, float(df[c].abs().max())):
            out.update({x: kind for x in members})
    return out


def season_info(ts):
    """시각별 (계절 이름, 요금 연도). 여름·겨울이 아니면 계절 이름은 ''.

    요금 연도 Y = Y년 7~9월(여름) + Y년 12월 ~ Y+1년 2월(겨울)
    """
    month = ts.dt.month.values
    year = ts.dt.year.values
    name = np.where(np.isin(month, SUMMER_MONTHS), "여름", np.where(np.isin(month, WINTER_MONTHS), "겨울", ""))
    fee_year = np.where(month <= 2, year - 1, year)
    return name, fee_year


def block_max(P, ts):
    """1분 전력(행=시각) -> 15분 평균 최대값과 그 시각."""
    blk = pd.Series(P.sum(axis=1), index=pd.DatetimeIndex(ts)).resample(f"{BLOCK}min").mean().dropna()
    return float(blk.max()), blk.idxmax()


@st.cache_resource(show_spinner="데이터 로딩 및 부하 분류 중...")
def load_bundle(paths, season_key=SEASON_ALL):
    df, labels = read_factory(paths)
    ts, cols, ref_total = parse_schema(df)

    # 부하 분류 (여름·겨울 전체 기간 데이터로 수행)
    features = classify_loads(df, cols)
    types = features["Load_Type"].to_dict()

    # 총합 검증: 설비 합 == 파일의 총합 컬럼(Total_kW) 인지
    check_err = None
    if ref_total is not None:
        check_err = float((df[cols].sum(axis=1) - ref_total).abs().max())

    # 관제 대상 = 가장 최근 요금 연도의 여름+겨울 (실제 날짜 그대로). 비교 기준 = 그 전 요금 연도
    sname, fee_year = season_info(ts)
    want = {SEASON_ALL: ("여름", "겨울"), SEASON_SUMMER: ("여름",), SEASON_WINTER: ("겨울",)}[season_key]
    in_season = np.isin(sname, want)
    if in_season.any():
        target = int(fee_year[in_season].max())
        mask = in_season & (fee_year == target)
        prev = in_season & (fee_year == target - 1)
        parts = []
        for nm in want:
            m = mask & (sname == nm)
            if m.any():
                parts.append(f"{nm} {ts[m].min().strftime('%Y-%m')}~{ts[m].max().strftime('%Y-%m')}")
        season = " + ".join(parts)
        prev_name = f"전년도({target - 1} 요금 연도) {'·'.join(want)}"
    else:
        mask = np.ones(len(ts), dtype=bool)
        prev = np.zeros(len(ts), dtype=bool)
        season = "전체 기간"
        prev_name = "전년도 같은 계절"
    tsw = pd.DatetimeIndex(ts[mask])
    P = np.nan_to_num(df.loc[mask, cols].to_numpy(dtype=float))
    season_of = sname[mask]

    mib = np.asarray(tsw.minute % BLOCK, dtype=int)      # 15분 구간 내 경과 분 (0~14)
    gaps = np.diff(tsw.values).astype("timedelta64[s]").astype(int) != 60
    ends = np.append(np.flatnonzero(gaps), len(tsw) - 1)  # 연속 구간(계절) 마지막 인덱스

    raw = P.sum(axis=1)
    blk = pd.Series(raw, index=tsw).resample(f"{BLOCK}min").mean().dropna()
    peak_demand = float(blk.max())
    P_prev = np.nan_to_num(df.loc[prev, cols].to_numpy(dtype=float))
    ts_prev = pd.DatetimeIndex(ts[prev])
    if prev.any():  # 피크 한계값 = 전년도 (여름·겨울) 15분 평균 최대 수요
        limit, limit_at = block_max(P_prev, ts_prev)
        limit_src = f"{prev_name} 15분 최대 수요 ({limit_at.strftime('%Y-%m-%d %H:%M')})"
        limit_fallback = False
    else:
        limit = peak_demand
        limit_src = f"{prev_name} 데이터 없음 → 이번 {season} 15분 최대 수요로 임시 대체"
        limit_fallback = True

    # 시연용 자동 시점: 피크 30분 전 / 평일 오후 평균 수준 구간
    peak_start = tsw.get_loc(blk.idxmax())
    peak_idx = max(0, peak_start - 2 * BLOCK)
    peak_idx -= int(mib[peak_idx])
    safe = blk[blk < p_warn(limit)]  # 정상 시나리오는 P_warn 아래(SAFE) 구간에서 고름
    cand = safe[(safe.index.hour >= 13) & (safe.index.hour <= 16) & (safe.index.day_of_week < 5)]
    if cand.empty:
        cand = safe if not safe.empty else blk
    normal_ts = (cand - cand.median()).abs().idxmin()
    normal_idx = int(tsw.get_loc(normal_ts))

    return {
        "ts": tsw, "P": P, "cols": cols, "types": types, "features": features,
        "mib": mib, "ends": ends, "limit": limit, "peak_demand": peak_demand,
        "check_err": check_err, "season": season, "labels": labels, "season_of": season_of,
        "limit_src": limit_src, "limit_fallback": limit_fallback,
        "P_prev": P_prev, "ts_prev": ts_prev,
        "files": [os.path.basename(p) for p in paths],
        "ref_name": None if ref_total is None else str(ref_total.name),
        "peak_idx": int(peak_idx), "normal_idx": normal_idx,
        "dates": sorted({d.strftime("%Y-%m-%d") for d in tsw.normalize().unique()}),
    }


# ============================================================
# 2. 제어 에이전트 (15분 구간 수요전력 기준 2단계 제어)
# ============================================================
def simulate(P, mib, order, flex_set, cap, limit, greedy=True, flex_min=None, warn=None):
    """1분마다 '이번 15분 구간의 예상 수요전력'을 계산해 제어한다.

    - 예상 수요 = (구간 내 지금까지 제어 후 전력 합 + 최근 3분 평균 전력 x 남은 분) / 15
    - SAFE      : 예상 수요 < P_warn(고정 3,096kW) -> 개입 없음
    - WARNING   : 예상 수요 >= P_warn -> 유연부하 차단(또는 감속)
    - EMERGENCY : 예상 수요 >= 한계값 -> 유연부하 차단(또는 감속) + 탄력부하 15분 지연(Time-Shift)
    - 지연된 탄력부하는 15분 뒤에 다시 더해진다 (리바운드 반영)
    - order   : 자동제어 대상 설비 인덱스 (우선순위 순, 사용자가 체크 해제한 설비는 빠짐)
    - cap     : 설비별 수동 가동률 (0~1). 0 = 수동 OFF, 1 = 정격 그대로
    - greedy  : True  -> 1순위부터 하나씩 조치, 예상 수요가 P_warn 미만이 되면 멈춤 (기본)
                False -> 대상 설비 전부에 조치
    - flex_min: None -> 유연부하 ON/OFF 차단 / 0~1 -> 속도 제어: 최소 가동률까지 필요한 만큼만 감속
    - warn    : P_warn (kW). None이면 p_warn(limit)
    """
    n = P.shape[0]
    p_eff = P * np.asarray(cap, dtype=float)[None, :]
    raw = P.sum(axis=1)
    base0 = p_eff.sum(axis=1)
    smooth = pd.Series(base0).rolling(SMOOTH_MIN, min_periods=1).mean().to_numpy()
    target = p_warn(limit) if warn is None else warn
    emerg = EMERG_RATIO * limit
    actions = []  # 마지막 시점에 실행된 조치 (설비 인덱스, 종류, 감축량)
    p_list = p_eff.tolist()  # 파이썬 반복문에서 numpy 원소 접근보다 빠름

    ctrl = np.zeros(n)
    status = np.zeros(n, dtype=int)
    d_dec = np.zeros(n)
    proj_raw = np.zeros(n)
    proj_ctrl = np.zeros(n)
    cut = np.zeros(n)
    rebound = np.zeros(n + BLOCK + 1)

    prev_ctrl = 0.0
    prev_raw = 0.0
    for t in range(n):
        k = int(mib[t])
        if k == 0:
            prev_ctrl = 0.0
            prev_raw = 0.0
        remain = BLOCK - k
        base = base0[t] + rebound[t]
        level = smooth[t] + rebound[t]  # 예상에 쓰는 현재 수준 (최근 3분 평균)
        d = (prev_ctrl + level * remain) / BLOCK
        d_dec[t] = d
        st_now = 2 if d >= emerg else (1 if d >= target else 0)
        status[t] = st_now
        reduction = 0.0
        actions = []
        if st_now:
            row = p_list[t]
            for i in order:
                p = row[i]
                if p <= 0:
                    continue
                if i in flex_set:
                    if flex_min is None:
                        r = p
                        kind = "OFF" if st_now == 2 else "OFF_W"
                    else:
                        r = p * (1.0 - flex_min)  # 최소 가동률까지만 감속 가능
                        if greedy:  # 예상 수요가 P_warn 바로 아래가 되는 만큼만 감속
                            need = level - reduction - (target * BLOCK - prev_ctrl) / remain + 0.01
                            r = min(r, max(need, 0.0))
                        kind = "SPD" if st_now == 2 else "SPD_W"
                elif st_now == 2:
                    r = p
                    kind = "SHIFT"
                    rebound[t + BLOCK] += p
                else:
                    continue  # WARNING 단계에서는 탄력부하를 건드리지 않음
                reduction += r
                actions.append((i, kind, r))
                if greedy and (prev_ctrl + (level - reduction) * remain) / BLOCK < target:
                    break
        c = base - reduction
        ctrl[t] = c
        cut[t] = raw[t] - c
        proj_raw[t] = (prev_raw + raw[t] * remain) / BLOCK
        proj_ctrl[t] = (prev_ctrl + c * remain) / BLOCK
        prev_ctrl += c
        prev_raw += raw[t]

    return {
        "raw": raw, "ctrl": ctrl, "status": status, "d_dec": d_dec,
        "proj_raw": proj_raw, "proj_ctrl": proj_ctrl, "cut": cut,
        "rebound": rebound[:n], "actions": actions,
    }


def block_series(res, mib, ts, n_blocks):
    """구간별 (지금까지의) 평균 수요전력: 제어 전 vs 제어 후."""
    n = len(res["raw"])
    starts = [i for i in range(n) if mib[i] == 0][-n_blocks:]
    labels, raw_m, ctrl_m = [], [], []
    for s in starts:
        e = min(s + BLOCK, n)
        labels.append(ts[s].strftime("%m-%d %H:%M"))
        raw_m.append(float(res["raw"][s:e].mean()))
        ctrl_m.append(float(res["ctrl"][s:e].mean()))
    return labels, raw_m, ctrl_m


def season_summary(bundle, order, flex_set, greedy, flex_min=None, P=None, limit=None, warn=None):
    """대상 기간 전체를 한 번에 시뮬레이션해 요약 통계를 반환 (수동 조작 없음).

    P, limit, warn을 주면 그 값으로 계산 (부하 구성비 시나리오용).
    """
    to_kwh = 1 / 60  # 1분 kW 값 합 -> kWh
    P = bundle["P"] if P is None else P
    limit = bundle["limit"] if limit is None else limit
    mib, ends, season_of = bundle["mib"], bundle["ends"], bundle["season_of"]
    cap = np.ones(P.shape[1])
    raw_b, ctrl_b, stat_b, name_b = [], [], [], []
    energy_raw = energy_cut = 0.0
    s = 0
    for e in ends:
        seg_P = P[s:e + 1]
        res = simulate(seg_P, mib[s:e + 1], order, flex_set, cap, limit, greedy, flex_min, warn)
        nb = len(seg_P) // BLOCK
        m = nb * BLOCK
        raw_b.append(res["raw"][:m].reshape(nb, BLOCK).mean(axis=1))
        ctrl_b.append(res["ctrl"][:m].reshape(nb, BLOCK).mean(axis=1))
        stat_b.append(res["status"][:m].reshape(nb, BLOCK).max(axis=1))
        name_b.append(season_of[s:s + m:BLOCK])
        energy_raw += res["raw"].sum()
        energy_cut += res["cut"].sum()
        s = e + 1
    raw_b = np.concatenate(raw_b)
    ctrl_b = np.concatenate(ctrl_b)
    stat_b = np.concatenate(stat_b)
    name_b = np.concatenate(name_b)
    by_season = {}
    for nm in ("여름", "겨울"):
        m = name_b == nm
        if m.any():
            by_season[nm] = (float(raw_b[m].max()), float(ctrl_b[m].max()))
    return {
        "blocks": int(len(raw_b)),
        "raw_max": float(raw_b.max()), "ctrl_max": float(ctrl_b.max()),
        "over_raw": int((raw_b > limit).sum()), "over_ctrl": int((ctrl_b > limit).sum()),
        "warn_blocks": int((stat_b >= 1).sum()), "emerg_blocks": int((stat_b >= 2).sum()),
        "energy_raw_kwh": energy_raw * to_kwh, "energy_cut_kwh": energy_cut * to_kwh,
        "by_season": by_season,
        "saving_won": (float(raw_b.max()) - float(ctrl_b.max())) * FEE_WON_PER_KW_YEAR,
    }


# ============================================================
# 2-1. 부하 구성비 시나리오 (경직/탄력/유연 비율 바꾸기)
# ============================================================
def type_shares(P, names, types):
    """유형별 에너지 비율 (평균 전력 기준)."""
    m = P.mean(axis=0)
    tot = m.sum() or 1e-9
    return {t: float(sum(m[i] for i, c in enumerate(names) if types[c] == t) / tot) for t in (RIGID, ELASTIC, FLEX)}


def type_scale(P, names, types, shares):
    """유형별 계수 k = 목표 비율 / 현재 비율. 총 에너지(평균 전력)는 그대로, 시간 패턴도 그대로.

    예: 유연 15% -> 30%면 유연부하 전부 x2.0, 나머지 유형은 줄여서 합계를 맞춘다.
    """
    cur = type_shares(P, names, types)
    k = {t: (shares[t] / cur[t] if cur[t] > 0 else 0.0) for t in cur}
    return np.array([k[types[c]] for c in names]), cur


def ratio_scenario(bundle, names, types, order, flex_set, greedy, flex_min, shares):
    """구성비를 바꾼 가상 공장으로 P_max·P_warn을 다시 정하고 겨울·여름 전체를 시뮬레이션.

    - 전년도 데이터도 같은 계수로 바꿔 P_max를 다시 계산
    - P_warn은 현재 설정의 'P_max 대비 비율'을 그대로 유지
    """
    kv, _ = type_scale(bundle["P"], names, types, shares)
    P = bundle["P"] * kv[None, :]
    if len(bundle["P_prev"]):
        limit, _ = block_max(bundle["P_prev"] * kv[None, :], bundle["ts_prev"])
    else:
        limit, _ = block_max(P, bundle["ts"])
    warn = limit * p_warn(bundle["limit"]) / bundle["limit"]
    r = season_summary(bundle, order, flex_set, greedy, flex_min, P=P, limit=limit, warn=warn)
    r["limit"], r["warn"] = limit, warn
    # 감축 여력: 매 순간 유연부하를 최소 가동률까지 줄였을 때의 15분 최대 (탄력 지연 제외, 상한 참고값)
    fmin = 0.0 if flex_min is None else flex_min
    sub = np.array([(1 - fmin) if i in flex_set and i in order else 0.0 for i in range(len(names))])
    floor = P - P * sub[None, :]
    r["floor_max"], _ = block_max(floor, bundle["ts"])
    return r


def depth_sweep(bundle, names, types, order, flex_set, greedy, flex_min, shares, step=25.0, depth=400.0):
    """구성비를 바꾼 가상 공장에서 P_warn을 P_max부터 step kW씩 낮추며 제어 후 최대 15분 수요를 구한다.

    반환: [(감축 목표 x, P_warn, 제어 후 최대)], 무제어 최대. 제어 후 최대가 가장 낮은 점 = 감축 여력 한계.
    """
    kv, _ = type_scale(bundle["P"], names, types, shares)
    P = bundle["P"] * kv[None, :]
    if len(bundle["P_prev"]):
        limit, _ = block_max(bundle["P_prev"] * kv[None, :], bundle["ts_prev"])
    else:
        limit, _ = block_max(P, bundle["ts"])
    out, raw_max = [], None
    for x in np.arange(0.0, depth + step / 2, step):
        r = season_summary(bundle, order, flex_set, greedy, flex_min, P=P, limit=limit, warn=limit - x)
        raw_max = r["raw_max"]
        out.append((float(x), limit - float(x), r["ctrl_max"]))
    return out, raw_max, limit


# ============================================================
# 3. 대시보드
# ============================================================
def main():
    st.set_page_config(page_title="Peak-Cut 실시간 관제 대시보드", page_icon="⚡", layout="wide")

    # ---- 세션 상태 ----
    ss = st.session_state
    if "is_running" not in ss:
        ss.is_running = False
    if "current_step" not in ss:
        ss.current_step = 0
    if "anchor" not in ss:
        ss.anchor = None

    # ---- 사이드바: 데이터 파일 선택 (여러 개면 시간순으로 합침) ----
    st.sidebar.header("📂 데이터")
    files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv"))
    if not files:
        st.error(f"❌ app.py와 같은 폴더({DATA_DIR})에 CSV 파일이 없습니다.")
        st.stop()
    picked = st.sidebar.multiselect("데이터 파일 (여러 개 선택 시 합쳐서 사용)", files, default=files,
                                    key="data_files")
    if not picked:
        st.warning("데이터 파일을 하나 이상 선택하세요.")
        st.stop()
    season_key = st.sidebar.radio("관제 계절", [SEASON_ALL, SEASON_SUMMER, SEASON_WINTER], key="season_key",
                                  help="통합: 한전 요금적용전력처럼 여름·겨울 중 가장 큰 15분 수요를 P_max로 사용")
    data_file = "|".join(picked) + "|" + season_key  # 위젯 key 구분용
    if ss.get("loaded_file") != data_file:  # 파일·계절이 바뀌면 재생 상태 초기화
        ss.loaded_file = data_file
        ss.anchor = None
        ss.current_step = 0
        ss.is_running = False

    try:
        bundle = load_bundle(tuple(os.path.join(DATA_DIR, f) for f in picked), season_key)
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        st.stop()

    # ---- 링크로 시연 장면 바로 열기: 주소 끝에 ?demo=peak (선택: &step=분) ----
    qp = st.query_params
    if qp.get("demo") == "peak" and ss.get("demo_applied") != data_file:
        ss.demo_applied = data_file
        ss["scenario"] = "피크 구간 (자동)"
        ss.anchor = bundle["peak_idx"]
        ss.current_step = int(qp.get("step", 35))

    ts, P, names, ai_types = bundle["ts"], bundle["P"], bundle["cols"], bundle["types"]
    mib, ends, limit, unit = bundle["mib"], bundle["ends"], bundle["limit"], UNIT
    season, feats = bundle["season"], bundle["features"]

    st.title("⚡ Peak-Cut 실시간 관제 & 통합 제어 대시보드")
    st.caption((  # '~'는 마크다운 취소선이 되므로 이스케이프 (파일명은 사이드바에 표시)
        f"관제 기간: {season} (1분 데이터) · 설비 {len(names)}개 · "
        f"단위 {unit} · 수요전력은 한전 방식(15분 평균) 기준으로 판단"
    ).replace("~", "\\~"))

    st.sidebar.success(f"✅ 파일 {len(bundle['files'])}개 · 관제 {len(ts):,}분 · 설비 {len(names)}개")
    if bundle["check_err"] is not None:
        st.sidebar.caption(
            f"총합 검증: 설비 합 = {bundle['ref_name']} (최대 오차 {bundle['check_err']:.1f} {unit})"
        )

    st.sidebar.header("☀️❄️ 관제 시점 선택")
    scenario = st.sidebar.radio(
        "시연 시나리오",
        ["직접 선택", "정상 구간 (자동)", "피크 구간 (자동)"],
        key="scenario",
    )
    if scenario.startswith("정상"):
        sel_idx = bundle["normal_idx"]
    elif scenario.startswith("피크"):
        sel_idx = bundle["peak_idx"]
    else:
        sel_date = st.sidebar.selectbox("시작 날짜", bundle["dates"], key=f"sel_date_{data_file}")
        time_opts = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
        sel_time = st.sidebar.selectbox("시작 시간 (15분 단위)", time_opts, index=56, key="sel_time")
        target = pd.Timestamp(f"{sel_date} {sel_time}")
        pos = int(ts.get_indexer([target])[0])
        if pos < 0:
            pos = int(min(ts.searchsorted(target), len(ts) - 1))
        sel_idx = pos
    sel_idx -= int(mib[sel_idx])  # 15분 구간 시작으로 맞춤
    st.sidebar.caption(f"선택 시점: {ts[sel_idx].strftime('%Y-%m-%d %H:%M')}")

    st.sidebar.markdown("---")
    st.sidebar.header("🕹️ 관제 목표 및 속도")
    st.sidebar.metric("🔒 피크 한계값 P_max (전년도 15분 최대)", f"{limit:,.1f} {unit}")
    pw = p_warn(limit)
    pw_note = "고정값" if P_WARN_KW is not None else "한계값의 90%"
    st.sidebar.caption((f"P_warn = {pw:,.0f} {unit} ({pw_note}, 한계값의 {pw / limit * 100:.1f}%) · "
                        f"한계값 기준: {bundle['limit_src']} · "
                        f"P_peak 범위 {P_PEAKMIN_KW:,.1f} ~ {P_PEAKMAX_KW:,.1f} {unit} (팀 문서 2024 겨울 값)"
                        ).replace("~", "\\~"))
    if bundle["limit_fallback"]:
        st.sidebar.warning("전년도 데이터가 없어 이번 기간 최대값을 임시 한계값으로 사용 중입니다.")
    speed = st.sidebar.slider("재생 속도 (초/분)", min_value=0.05, max_value=2.0, value=0.2)

    b1, b2 = st.sidebar.columns(2)
    b3, b4 = st.sidebar.columns(2)
    if b1.button("▶️ 재생 시작"):
        ss.anchor = sel_idx
        ss.current_step = 0
        ss.is_running = True
    if b2.button("⏸️ 일시 정지"):
        ss.is_running = False
    start_idx = ss.anchor if ss.anchor is not None else sel_idx
    max_step = int(ends[np.searchsorted(ends, start_idx)] - start_idx)
    if b3.button("⏩ +15분"):
        ss.current_step = min(ss.current_step + BLOCK, max_step)
    if b4.button("⏹️ 초기화"):
        ss.anchor = None
        ss.current_step = 0
        ss.is_running = False
        start_idx = sel_idx
        max_step = int(ends[np.searchsorted(ends, start_idx)] - start_idx)

    # ---- 사이드바: 부하 유형 확정 + 자동제어 우선순위/대상 선택 ----
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 부하 유형 확정 · 자동제어 우선순위")
    by_type = {t: [i for i, c in enumerate(names) if ai_types[c] == t] for t in (FLEX, ELASTIC, RIGID)}
    by_peak = lambda i: -feats.loc[names[i], "Max_Power"]
    cand = [i for t in (FLEX, ELASTIC, RIGID) for i in sorted(by_type[t], key=by_peak)]  # 기본: 유연 -> 탄력 -> 경직, 각 15분 최대전력 큰 순
    with st.sidebar.expander("📋 유형 확정 · 순위 · 자동제어 대상", expanded=False):
        mode = st.radio(
            "제어 방식", [MODE_GREEDY, MODE_ALL], key="ctrl_mode",
            help="우선순위대로: 1순위부터 하나씩 제어하고, 예상 수요가 P_warn 아래로 내려가면 멈춥니다. "
                 "일괄: 경고 시 유연부하 전부, 비상 시 유연+탄력 전부 제어합니다.",
        )
        df_pri = pd.DataFrame({
            "자동제어": [True] * len(cand),
            "순위": list(range(1, len(cand) + 1)),
            "설비": [names[i] for i in cand],
            "유형": [SHORT[ai_types[names[i]]] for i in cand],
            "검토": ["⚠" if feats.loc[names[i], "Review"] else "" for i in cand],
        })
        edited = st.data_editor(
            df_pri, key=f"priority_table_{data_file}", hide_index=True, use_container_width=True,
            disabled=["설비", "검토"],
            column_config={
                "자동제어": st.column_config.CheckboxColumn(
                    "자동제어", width="small", help="체크 해제하면 AI가 제어하지 않음"
                ),
                "순위": st.column_config.NumberColumn(
                    "순위", width="small", min_value=1, max_value=len(cand), step=1,
                    help="숫자가 작을수록 먼저 제어",
                ),
                "유형": st.column_config.SelectboxColumn(
                    "유형", width="small", options=list(LONG), required=True,
                    help="AI 1차 분류 결과. 관리자가 바꾸면 바뀐 유형으로 제어",
                ),
                "검토": st.column_config.TextColumn(
                    "검토", width="small", help="⚠ = DBSCAN이 패턴이 특이하다고 본 설비 (유형 확인 필요)"
                ),
            },
        )
        types = {c: LONG.get(t, ai_types[c]) for c, t in zip(edited["설비"], edited["유형"])}
        sel = edited.assign(
            _r=edited["순위"].fillna(len(cand) + 1), _d=range(len(edited))
        ).sort_values(["_r", "_d"])
        sel = sel[sel["자동제어"].fillna(False).astype(bool) & (sel["설비"].map(types) != RIGID)]
        name_to_idx = {c: i for i, c in enumerate(names)}
        order = [name_to_idx[c] for c in sel["설비"]]
        st.caption("적용 순서: " + (" → ".join(sel["설비"]) if order else "없음 (자동제어 꺼짐)"))
        st.caption("경직 부하는 자동제어하지 않습니다. 순위가 같으면 위에 있는 설비가 먼저입니다.")
    greedy = mode == MODE_GREEDY
    flex_set = {i for i, c in enumerate(names) if types[c] == FLEX}
    rank_of = {i: k + 1 for k, i in enumerate(order)}

    # ---- 사이드바: 유연부하 속도 제어 (자동 방식 + 수동 가동률) ----
    st.sidebar.markdown("---")
    st.sidebar.header("🎚️ 유연부하 속도 제어")
    flex_mode = st.sidebar.radio("AI 자동제어 시 유연부하 조작", [FLEX_SPEED, FLEX_ONOFF], key="flex_mode")
    flex_min = None
    if flex_mode == FLEX_SPEED:
        flex_min = st.sidebar.slider("감속 하한 (최소 가동률 %)", 0, 90, 50, 5, key="flex_min",
                                     help="AI가 유연부하를 이 가동률 아래로는 줄이지 않습니다. "
                                          "필요한 감축량만큼만 감속하므로 ON/OFF보다 덜 끕니다.") / 100
    with st.sidebar.expander("🔧 유연부하 수동 가동률 (0~100%)", expanded=False):
        st.caption("인버터(VFD) 속도 지령처럼 출력을 0~100%로 조절합니다. "
                   "팬·펌프는 상사법칙 P ∝ N³ → 출력 50%면 속도 약 79%.")
        flex_names = sorted((c for c in names if types[c] == FLEX), key=lambda c: -feats.loc[c, "Max_Power"])
        for c in flex_names:
            v = st.slider(f"{c}", 0, 100, 100, 5, key=f"spd_{c}", format="%d%%")
            if any(k in c for k in FAN_PUMP) and "Heat_Pump" not in c:  # 히트펌프는 압축기라 제외
                if v < 100:
                    st.caption(f"↳ 팬·펌프 환산 속도 ≈ {100 * (v / 100) ** (1 / 3):.0f}%")

    # ---- 사이드바: 나머지 설비 스위치 ----
    st.sidebar.markdown("---")
    st.sidebar.header(f"🎛️ 경직·탄력 설비 수동 ON/OFF")
    with st.sidebar.expander("🔌 개별 설비 전원 조작 (부하 유형별)", expanded=False):
        for i in sorted(range(len(names)), key=lambda i: (types[names[i]], names[i])):
            if types[names[i]] != FLEX:
                st.toggle(f"{names[i]} ({types[names[i]]})", value=True, key=f"sw_{names[i]}")
    cap = np.array([
        ss.get(f"spd_{c}", 100) / 100 if types[c] == FLEX else (1.0 if ss.get(f"sw_{c}", True) else 0.0)
        for c in names
    ])

    # ---- 시뮬레이션 ----
    step = min(ss.current_step, max_step)
    end = start_idx + step
    res = simulate(
        P[start_idx:end + 1], mib[start_idx:end + 1], order, flex_set, cap, limit, greedy, flex_min
    )
    t_now = ts[end].strftime("%Y-%m-%d %H:%M")
    season_now = bundle["season_of"][end] or "-"
    status = int(res["status"][-1])
    ratio_dec = res["d_dec"][-1] / limit * 100
    raw_now = res["raw"][-1]
    cut_now = res["cut"][-1]

    how = " (우선순위대로 P_warn 아래까지)" if greedy else " (대상 전부)"
    verb = "감속" if flex_min is not None else "차단"
    if status == 0:
        st.success(f"🟢 **[{season_now} {t_now} 정상]** 이번 15분 구간 예상 수요율 {ratio_dec:.1f}% - 개입 없음")
    elif status == 1:
        st.warning(f"🟡 **[{season_now} {t_now} 경고]** 예상 수요율 {ratio_dec:.1f}% "
                   f"(≥ P_warn {pw:,.0f}{unit}) - 유연부하 {verb}{how}")
    else:
        st.error(
            f"🔴 **[{season_now} {t_now} 비상]** 예상 수요율 {ratio_dec:.1f}% (≥100% 한계값) - "
            f"유연부하 {verb} + 탄력부하 15분 지연{how}"
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재 전력 (1분)", f"{raw_now:,.0f} {unit}")
    c2.metric("피크 한계값", f"{limit:,.0f} {unit}")
    c3.metric(
        "구간 예상 수요 (무제어)", f"{res['proj_raw'][-1]:,.0f} {unit}",
        delta=f"{res['proj_raw'][-1] / limit * 100:.0f}% of 한계",
        delta_color="off",
    )
    c4.metric(
        "구간 예상 수요 (제어 후)", f"{res['proj_ctrl'][-1]:,.0f} {unit}",
        delta=f"{res['proj_ctrl'][-1] / limit * 100:.0f}% of 한계",
        delta_color="off",
    )
    c5.metric(
        "현재 감축 전력", f"{cut_now:,.0f} {unit}",
        delta=f"-{cut_now:,.0f} {unit}" if cut_now > 0 else None,
    )

    # ---- 그래프 ----
    col_chart, col_gauge = st.columns([2, 1])
    with col_chart:
        lo = max(0, len(res["raw"]) - WINDOW_MIN)
        x = list(ts[start_idx + lo:end + 1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=res["raw"][lo:], mode="lines", name="측정 전력 (1분)",
                                 line=dict(color="red", width=2)))
        fig.add_trace(go.Scatter(x=x, y=res["ctrl"][lo:], mode="lines", name="제어 후 전력 (1분)",
                                 line=dict(color="green", width=2, dash="dash")))
        fig.add_hline(y=limit, line_dash="solid", line_color="black",
                      annotation_text=f"P_max {limit:,.0f} (15분 평균 기준)")
        fig.add_hline(y=pw, line_dash="dot", line_color="orange",
                      annotation_text=f"P_warn {pw:,.0f}",
                      annotation_position="bottom right")
        # P_peak max/min은 서로 5kW 차이라 글씨가 겹치므로 범례로 표시
        for val, name, color in ((P_PEAKMAX_KW, "P_peak max", "purple"),
                                 (P_PEAKMIN_KW, "P_peak min", "deepskyblue")):
            fig.add_trace(go.Scatter(x=x, y=[val] * len(x), mode="lines", name=f"{name} {val:,.1f}",
                                     line=dict(color=color, width=1.5, dash="dot"), hoverinfo="skip"))
        fig.update_layout(title="📈 1분 전력 추이 (최근 60분)", xaxis_title="시각",
                          yaxis_title=f"전력 ({unit})", height=330)
        st.plotly_chart(fig, use_container_width=True)

    with col_gauge:
        val = float(res["proj_raw"][-1])
        gmax = max(limit * 1.2, val * 1.1)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            gauge={
                "axis": {"range": [0, gmax]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, pw], "color": "lightgreen"},
                    {"range": [pw, limit * EMERG_RATIO], "color": "yellow"},
                    {"range": [limit * EMERG_RATIO, gmax], "color": "salmon"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": limit},
            },
        ))
        fig_g.update_layout(title="구간 예상 수요 (무제어 기준)", height=330)
        st.plotly_chart(fig_g, use_container_width=True)

    labels, raw_m, ctrl_m = block_series(res, mib[start_idx:end + 1], ts[start_idx:end + 1], WINDOW_BLOCKS)
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(x=labels, y=raw_m, name="제어 전 15분 수요", marker_color="red"))
    fig_b.add_trace(go.Bar(x=labels, y=ctrl_m, name="제어 후 15분 수요", marker_color="green"))
    # P_peak range: 폭이 5.5kW라 얇으므로 막대 위에 반투명으로 칠하고, 이름은 P_warn 글씨와 겹치지 않게 범례로 표시
    fig_b.add_hrect(y0=P_PEAKMIN_KW, y1=P_PEAKMAX_KW, fillcolor="rgba(147, 112, 219, 0.55)", line_width=0,
                    layer="above")
    fig_b.add_trace(go.Scatter(x=[None], y=[None], mode="markers", hoverinfo="skip",
                               marker=dict(symbol="square", size=14, color="rgba(147, 112, 219, 0.55)"),
                               name=f"P_peak range {P_PEAKMIN_KW:,.1f}~{P_PEAKMAX_KW:,.1f}"))
    fig_b.add_hline(y=limit, line_dash="solid", line_color="black",
                    annotation_text=f"P_max {limit:,.0f}")
    fig_b.add_hline(y=pw, line_dash="dot", line_color="orange",
                    annotation_text=f"P_warn {pw:,.0f}", annotation_position="bottom left")
    y_top = max([limit] + raw_m + ctrl_m) * 1.01  # P_max 선이 보이도록 1% 여유, 넘는 막대가 있으면 그만큼 확장
    fig_b.update_layout(title="📊 15분 구간 수요전력 (요금 산정 기준, 진행 중 구간은 현재까지 평균)",
                        barmode="group", yaxis_title=f"전력 ({unit})", yaxis_range=[2500, y_top], height=330)
    st.plotly_chart(fig_b, use_container_width=True)

    # ---- 의사결정 로그 ----
    st.subheader("🤖 수동 조작 및 AI 실시간 의사결정 로그 (현재 시점)")
    p_now = P[end]
    rows = []
    for i, c in enumerate(names):
        if cap[i] < 1 and p_now[i] > 0:
            label = "OFF (수동 차단)" if cap[i] == 0 else f"수동 가동률 {cap[i] * 100:.0f}%"
            rows.append(["-", c, label, types[c], round(p_now[i], 1), round(p_now[i] * (1 - cap[i]), 1),
                         "관리자 수동 조작"])
    for i, kind, r in res["actions"]:
        label, why = ACTION_TEXT[kind]
        if kind.startswith("SPD"):
            label = f"SPEED {(p_now[i] * cap[i] - r) / p_now[i] * 100:.0f}% (AI 자동)"
        rows.append([str(rank_of[i]), names[i], label, types[names[i]], round(p_now[i], 1), round(r, 1), why])
    if rows:
        df_act = pd.DataFrame(
            rows, columns=["순위", "대상 설비", "제어 상태", "부하 유형", f"현재 전력({unit})",
                           f"감축량({unit})", "제어 사유"]
        )
        st.dataframe(df_act, use_container_width=True, hide_index=True)
    else:
        st.info("현재 제어 중인 설비가 없습니다.")

    # ---- 부가 정보 ----
    with st.expander("🧩 부하 분류 결과 (K-Means 군집 + DBSCAN 이상 설비 검토 + 관리자 확정)"):
        labels = bundle["labels"]
        ft = pd.DataFrame({
            f"평균({unit})": feats["Mean_Power"],
            "부하 비중(%)": feats["Share"] * 100,
            "가동률": feats["On_Ratio"],
            "주기성": feats["Periodicity"],
            "DBSCAN 검토": feats["Review"].map({True: "⚠ 검토 필요", False: ""}),
            "AI 분류": feats["Load_Type"],
            "최종 유형": [types[c] for c in feats.index],
        })
        if labels:
            ft["파일 라벨"] = [labels.get(c, "-") for c in feats.index]
        st.dataframe(ft.round(3).sort_values(["AI 분류", f"평균({unit})"], ascending=[True, False]),
                     use_container_width=True)
        st.caption(
            f"AI 분류 설비 수: {pd.Series(ai_types).value_counts().sort_index().to_dict()} · "
            f"최종: {pd.Series(types).value_counts().sort_index().to_dict()}"
        )
        if labels:
            ai_ok = sum(ai_types[c] == t for c, t in labels.items())
            fin_ok = sum(types[c] == t for c, t in labels.items())
            st.info(f"파일에 포함된 부하 라벨과 비교: AI 분류 {ai_ok}/{len(labels)}개 일치 · "
                    f"관리자 확정 후 {fin_ok}/{len(labels)}개 일치")

    with st.expander(f"📋 {season} 전체 시뮬레이션 요약 (수동 조작 없음, 현재 우선순위/제어 방식 기준, 계산에 수 초 소요)"
                     .replace("~", "\\~")):
        if st.button("요약 계산 실행"):
            r = season_summary(bundle, order, flex_set, greedy, flex_min)
            s1, s2, s3 = st.columns(3)
            s1.metric("15분 구간 수", f"{r['blocks']:,}")
            s2.metric("최대 15분 수요 (제어 전 → 후)", f"{r['raw_max']:,.0f} → {r['ctrl_max']:,.0f} {unit}")
            s3.metric("한계값 초과 구간 (전 → 후)", f"{r['over_raw']} → {r['over_ctrl']}")
            s4, s5, s6 = st.columns(3)
            s4.metric("WARNING 이상 (제어 발동) 구간", f"{r['warn_blocks']} ({r['warn_blocks'] / r['blocks'] * 100:.2f}%)")
            s5.metric("EMERGENCY (탄력 지연 포함) 구간", f"{r['emerg_blocks']} ({r['emerg_blocks'] / r['blocks'] * 100:.2f}%)")
            s6.metric("감축 에너지", f"{r['energy_cut_kwh']:,.1f} kWh ({r['energy_cut_kwh'] / r['energy_raw_kwh'] * 100:.2f}%)")
            cols_s = st.columns(len(r["by_season"]) + 1)
            for col, (nm, (a, b)) in zip(cols_s, r["by_season"].items()):
                col.metric(f"{nm} 최대 15분 (전 → 후)", f"{a:,.0f} → {b:,.0f} {unit}")
            cols_s[-1].metric("기본요금 절감 (연, 제어 비용 미반영)", f"{r['saving_won'] / 1e4:,.0f} 만원")

    # ---- 부하 구성비 시나리오 ----
    with st.expander("⚖️ 부하 구성비 시나리오: 경직·탄력·유연 비율을 바꾸면 절감액이 어떻게 달라지나 (계산에 수십 초 소요)"):
        cur = type_shares(P, names, types)
        st.caption(
            f"현재 구성비(관리자 확정 유형, 에너지 기준): 경직 {cur[RIGID] * 100:.1f}% · "
            f"탄력 {cur[ELASTIC] * 100:.1f}% · 유연 {cur[FLEX] * 100:.1f}%. "
            "방법: 유형별로 1분 전력에 같은 계수를 곱해 목표 비율로 맞춘 '가상 공장'을 만든다 "
            "(총 에너지·시간 패턴 유지). 전년도 데이터도 같은 계수로 바꿔 P_max를 다시 정하고, "
            "P_warn은 P_max 대비 비율을 유지한다."
        )
        r1, r2 = st.columns(2)
        f_pct = r1.slider("유연부하 비율 (%)", 0, 60, int(round(cur[FLEX] * 100)), 1, key="ratio_flex")
        e_pct = r2.slider("탄력부하 비율 (%)", 0, 40, int(round(cur[ELASTIC] * 100)), 1, key="ratio_elastic")
        g_pct = 100 - f_pct - e_pct
        if g_pct < 0:
            st.error("유연 + 탄력이 100%를 넘습니다.")
        else:
            st.write(f"→ 경직부하 비율 **{g_pct}%**")
            deep = st.checkbox("P_warn을 25kW씩 낮추며 최대 감축 여력까지 계산 (약 1~2분)", key="ratio_deep")
            scen = (("현재 구성", cur),
                    ("시나리오", {RIGID: g_pct / 100, ELASTIC: e_pct / 100, FLEX: f_pct / 100}))
            if st.button("시나리오 계산", key="ratio_run"):
                rows = []
                for nm, sh in scen:
                    r = ratio_scenario(bundle, names, types, order, flex_set, greedy, flex_min, sh)
                    rows.append({
                        "구분": nm,
                        "경직·탄력·유연(%)": f"{sh[RIGID] * 100:.0f} · {sh[ELASTIC] * 100:.0f} · {sh[FLEX] * 100:.0f}",
                        f"P_max({unit})": round(r["limit"], 1),
                        f"P_warn({unit})": round(r["warn"], 1),
                        f"무제어 최대({unit})": round(r["raw_max"], 1),
                        f"제어 후 최대({unit})": round(r["ctrl_max"], 1),
                        f"감축({unit})": round(r["raw_max"] - r["ctrl_max"], 1),
                        "기본요금 절감(만원/년)": round(r["saving_won"] / 1e4),
                        f"유연 최대 감속 시 최대({unit})": round(r["floor_max"], 1),
                        "감축 에너지(kWh)": round(r["energy_cut_kwh"]),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("'유연 최대 감속 시 최대' = 자동제어 대상 유연부하를 항상 최소 가동률로 돌렸을 때의 15분 최대 "
                           "(탄력 지연 제외). P_warn보다 크면 유연부하만으로는 P_warn을 지킬 수 없다는 뜻. "
                           "절감액은 기본요금만 계산했고 생산 손실·보충 작업 비용은 넣지 않았다.")
                if deep:
                    fig_r = go.Figure()
                    best_rows = []
                    for (nm, sh), color in zip(scen, ("gray", "royalblue")):
                        pts, raw_max, lim = depth_sweep(bundle, names, types, order, flex_set, greedy, flex_min, sh)
                        xs = [p[0] for p in pts]
                        save = [(raw_max - p[2]) * FEE_WON_PER_KW_YEAR / 1e4 for p in pts]
                        fig_r.add_trace(go.Scatter(x=xs, y=save, mode="lines+markers", name=nm,
                                                   line=dict(color=color, width=2)))
                        k = int(np.argmax(save))
                        best_rows.append({"구분": nm, f"최적 감축 목표({unit})": xs[k],
                                          f"그때 제어 후 최대({unit})": round(pts[k][2], 1),
                                          "기본요금 절감 상한(만원/년)": round(save[k])})
                    fig_r.update_layout(title="P_warn 깊이별 기본요금 절감액 (제어 비용 미반영)",
                                        xaxis_title=f"감축 목표 = P_max − P_warn ({unit})",
                                        yaxis_title="만원/년", height=360)
                    st.plotly_chart(fig_r, use_container_width=True)
                    st.dataframe(pd.DataFrame(best_rows), use_container_width=True, hide_index=True)

    # ---- 재생 루프 ----
    if ss.is_running:
        time.sleep(speed)
        if ss.current_step < max_step:
            ss.current_step += 1
        else:
            ss.is_running = False
        st.rerun()


if __name__ == "__main__":
    main()
