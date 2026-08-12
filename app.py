
import re
import zipfile
import tempfile
import copy
import sqlite3
import hashlib
import io
import json
import os
import shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 mean_absolute_error, r2_score, confusion_matrix,
                                 brier_score_loss, log_loss, roc_auc_score)
    from sklearn.inspection import permutation_importance
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBRegressor = None
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import psycopg
    PSYCOPG_AVAILABLE = True
except Exception:
    psycopg = None
    PSYCOPG_AVAILABLE = False


st.set_page_config(page_title="PhysioSentinel AI: Predictive Physiological Intelligence Platform", layout="wide")

# Fases ampliadas:
# - Basal + Basal2-Basal5 permiten seleccionar varias ventanas basales.
# - R1-R6 permiten seleccionar más de dos ventanas de recuperación.
PHASES = ["Basal"] + [f"Basal{i}" for i in range(2, 6)] + [f"E{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)]
PHASE_GROUP = {
    "Basal": "Basal",
    **{f"Basal{i}": "Basal" for i in range(2, 6)},
    **{f"E{i}": "Ejercicio" for i in range(1, 7)},
    **{f"R{i}": "Recuperación" for i in range(1, 7)},
}
PHASE_COLORS = {
    "Basal": "rgba(0,150,255,0.24)",
    "Ejercicio": "rgba(255,140,0,0.20)",
    "Recuperación": "rgba(0,200,100,0.20)",
}
PHASE_LINE_COLORS = {
    "Basal": "#0096ff",
    "Ejercicio": "#ff8c00",
    "Recuperación": "#00c864",
}


# ============================================================
# KUBIOS ADVANCED SETTINGS II - equivalencia explícita
# ============================================================
KUBIOS_ENTROPY_M = 2
KUBIOS_ENTROPY_R_FACTOR = 0.2
KUBIOS_DFA_ALPHA1_RANGE = (4, 12)
KUBIOS_DFA_ALPHA2_RANGE = (13, 64)
KUBIOS_RQA_EMB_DIM = 10
KUBIOS_RQA_THRESHOLD_SD = 3.1623
KUBIOS_MSE_MAX_SCALE = 20

MSE_ZERO_MODE_OPTIONS = {
    "Clásico SampEn: A=0 -> no calculado": "nan",
    "Pseudoconteo 0.5: A=0 -> A=0.5": "half_count",
    "Pseudoconteo 1.0: A=0 -> A=1": "one_count",
    "RCMSE / Composite Kubios-like": "rcmse",
}
DEFAULT_MSE_ZERO_MODE_LABEL = "Clásico SampEn: A=0 -> no calculado"

MSE_RADIUS_MODE_OPTIONS = {
    "r fijo: 0.2 x SD de señal λ500": "fixed_entropy_sd",
    "r por escala: 0.2 x SD de cada coarse-grain": "scale_sd",
    "r fijo: 0.2 x SD RR corregido sin λ": "fixed_raw_sd",
}
DEFAULT_MSE_RADIUS_MODE_LABEL = "r fijo: 0.2 x SD de señal λ500"

THEILER_WINDOW_OPTIONS = {
    "Sin exclusión temporal": 0,
    "Theiler 1 beat": 1,
    "Theiler 2 beats": 2,
    "Theiler 3 beats": 3,
    "Theiler 4 beats": 4,
    "Theiler 5 beats": 5,
}
DEFAULT_THEILER_WINDOW_LABEL = "Sin exclusión temporal"

FS_INTERP = 4.0
LAMBDA_DEFAULT = 500

PARAM_GROUPS = {
    "Tiempo": ["MeanHR", "MeanRR", "SDNN", "RMSSD", "pNN50", "SD1", "SD2"],
    "Frecuencia": ["VLF", "LF", "HF", "TOTAL", "LF_HF", "VLF_LS", "LF_LS", "HF_LS", "TOTAL_LS", "LF_HF_LS", "VLF_AR", "LF_AR", "HF_AR", "TOTAL_AR", "LF_HF_AR", "VLF_WAV_MEAN", "LF_WAV_MEAN", "HF_WAV_MEAN", "VLF_WAV_SD", "LF_WAV_SD", "HF_WAV_SD", "VLF_DOM_PCT", "LF_DOM_PCT", "HF_DOM_PCT", "VLF_EPISODES_N", "LF_EPISODES_N", "HF_EPISODES_N", "VLF_EPISODE_MEAN_S", "LF_EPISODE_MEAN_S", "HF_EPISODE_MEAN_S", "VLF_EPISODE_MAX_S", "LF_EPISODE_MAX_S", "HF_EPISODE_MAX_S", "WAV_TRANSITIONS_N", "WAV_TRANSITIONS_PER_MIN", "WAV_ENTROPY_BANDS", "WAV_ENTROPY_GLOBAL", "LF_WAV", "HF_WAV", "LF_HF_WAV"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "D2", "ApEn", "SampEn", "Lyapunov_LLE", "Hurst", "KatzFD", "PetrosianFD", "DispEn"],
    "MSE 1-20": [f"MSE{i}" for i in range(1, 21)],
    "MDE 1-20": [f"MDE{i}" for i in range(1, 21)],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax", "ShanEn"],
    "Control Kubios": ["Entropy_lambda", "Entropy_m", "Entropy_r_factor", "Entropy_SD_ms", "Entropy_r_ms", "Entropy_N", "DFA_alpha1_range", "DFA_alpha2_range", "RQA_threshold_SD", "RQA_emb_dim", "MSE_zero_policy", "SampEn_Theiler", "MSE_radius_mode"],
}
DEFAULT_MULTI = ["RMSSD", "SDNN", "SD1", "SD2", "LF", "HF", "DFA_alpha1", "DFA_alpha2", "D2"]

DOMAIN_GROUPS = {
    "Amplitud": ["SDNN", "SD2", "TOTAL"],
    "Vagal": ["RMSSD", "SD1", "HF", "pNN50"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "D2", "ApEn", "SampEn", "Lyapunov_LLE", "Hurst", "KatzFD", "PetrosianFD", "DispEn"],
    "MSE 1-20": [f"MSE{i}" for i in range(1, 21)],
    "MDE 1-20": [f"MDE{i}" for i in range(1, 21)],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax", "ShanEn"],
    "Control Kubios": ["Entropy_lambda", "Entropy_m", "Entropy_r_factor", "Entropy_SD_ms", "Entropy_r_ms", "Entropy_N", "DFA_alpha1_range", "DFA_alpha2_range", "RQA_threshold_SD", "RQA_emb_dim", "MSE_zero_policy", "SampEn_Theiler", "MSE_radius_mode"],
}

MSE_COLUMNS = [f"MSE{i}" for i in range(1, 21)]


def sanitize_name(name):
    name = Path(str(name)).stem
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return name or "registro"



def extract_datetime_from_name(name):
    """
    Extrae fecha/hora desde nombres de archivo.

    Admite:
    - papa_2026-06-15_17-25-11
    - papa_2026-06-15_17-01-40
    - papa_026-04-14_17-33-22  -> interpreta 026 como 2026
    - 2026-06-12 11-24-39
    - 20220615...
    """
    txt = str(name)

    patterns = [
        # yyyy-mm-dd_hh-mm-ss
        r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})[ _-](\d{1,2})[-_](\d{1,2})[-_](\d{1,2})",
        # yyy-mm-dd_hh-mm-ss cuando por truncado aparece 026-...
        r"(?<!\d)(\d{3})[-_](\d{1,2})[-_](\d{1,2})[ _-](\d{1,2})[-_](\d{1,2})[-_](\d{1,2})",
        # yyyy-mm-dd
        r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})",
        # yyy-mm-dd
        r"(?<!\d)(\d{3})[-_](\d{1,2})[-_](\d{1,2})",
        # yyyymmdd_hhmmss o yyyymmdd
        r"(20\d{2})(\d{2})(\d{2})[ _-]?(\d{2})?(\d{2})?(\d{2})?",
    ]

    for pat in patterns:
        m = re.search(pat, txt)
        if not m:
            continue

        groups = [g for g in m.groups()]
        try:
            y = int(groups[0])
            if 0 <= y < 1000:
                # ejemplo 026 -> 2026
                y = 2000 + y

            mo = int(groups[1])
            d = int(groups[2])

            h = int(groups[3]) if len(groups) > 3 and groups[3] not in [None, ""] else 0
            mi = int(groups[4]) if len(groups) > 4 and groups[4] not in [None, ""] else 0
            s = int(groups[5]) if len(groups) > 5 and groups[5] not in [None, ""] else 0

            if 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
                return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=s)
        except Exception:
            pass

    return pd.Timestamp.max


def sort_records_chronologically(record_data):
    return dict(sorted(
        record_data.items(),
        key=lambda kv: (extract_datetime_from_name(kv[0]), kv[0])
    ))


def read_rri_file(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    vals = []
    for line in text.replace(";", "\n").replace("\t", "\n").splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        for p in line.split():
            try:
                vals.append(float(p))
            except Exception:
                pass

    rr = np.asarray(vals, dtype=float)
    rr = rr[np.isfinite(rr)]

    if len(rr) == 0:
        raise ValueError("No se han detectado RRi numéricos.")

    if np.nanmedian(rr) > 10:
        rr = rr / 1000.0

    rr = rr[(rr >= 0.3) & (rr <= 2.0)]

    if len(rr) == 0:
        raise ValueError("Tras el filtrado fisiológico no quedan RRi válidos.")

    return rr


def _local_median_excluding(rr, i, radius=5):
    """
    Mediana local de RR alrededor de i excluyendo el punto i.
    """
    n = len(rr)
    lo = max(0, i - radius)
    hi = min(n, i + radius + 1)
    vals = np.concatenate([rr[lo:i], rr[i+1:hi]])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nanmedian(rr)
    return np.nanmedian(vals)


def _cubic_interpolate_bad(rr, bad_mask):
    rr = np.asarray(rr, dtype=float)
    out = rr.copy()
    idx = np.arange(len(rr))
    good = (~bad_mask) & np.isfinite(rr)
    bad = bad_mask | (~np.isfinite(rr))

    if np.sum(good) < 4:
        if np.sum(good) >= 2 and np.sum(bad) > 0:
            out[bad] = np.interp(idx[bad], idx[good], rr[good])
        return out

    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(idx[good], rr[good], extrapolate=True)
        out[bad] = cs(idx[bad])
    except Exception:
        out[bad] = np.interp(idx[bad], idx[good], rr[good])

    # Seguridad fisiológica general.
    med = np.nanmedian(rr[good])
    out = np.where(np.isfinite(out), out, med)
    out = np.clip(out, 0.30, 2.00)
    return out


def correct_artifacts_kubios_like(rr, level="none", window=5):
    """
    v12.0: corrección artefactos mejorada tipo Kubios / Lipponen-Tarvainen aproximada.

    Incorpora dos familias:
    1) Threshold-based Kubios-like:
       very low=0.45 s, low=0.35 s, medium=0.25 s, strong=0.15 s, very strong=0.05 s,
       ajustado por frecuencia cardíaca media/local.
    2) Patrón dRR tipo automático:
       detección de saltos NP/PN/NPN/PNP sobre diferencias sucesivas dRR con umbral
       adaptativo basado en dispersión local de 90 latidos.

    Correcciones:
    - intervalos aislados anómalos se sustituyen por spline cúbico;
    - posible latido perdido: RR largo compatible con suma de dos RR normales se divide;
    - posible latido extra: dos RR cortos consecutivos compatibles con un RR normal se fusionan;
    - resto: interpolación cúbica local.
    """
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr)]
    n0 = len(rr)

    if level == "none" or n0 < 10:
        return rr.copy(), np.zeros(n0, dtype=bool), {
            "level": level,
            "n_artifacts": 0,
            "percent_artifacts": 0.0,
            "note": "sin corrección",
        }

    thresholds = {
        "very low": 0.45,
        "low": 0.35,
        "medium": 0.25,
        "strong": 0.15,
        "very strong": 0.05,
        "kubios scientific": 0.05,
        "kubios auto": 0.05,
    }
    base_th = thresholds.get(str(level).lower(), 0.25)

    rr_work = rr.copy()
    original_index_artifacts = np.zeros(len(rr_work), dtype=bool)

    # ============================================================
    # Paso A: reconstrucción de latidos perdidos / extra.
    # ============================================================
    reconstructed = []
    reconstructed_bad = []
    i = 0
    missed_n = 0
    extra_n = 0

    while i < len(rr_work):
        med = _local_median_excluding(rr_work, i, radius=5)
        if not np.isfinite(med) or med <= 0:
            med = np.nanmedian(rr_work)

        # Umbral adaptado a FC: con RR más largo tolera algo más.
        hr_scale = np.clip(med / 1.0, 0.55, 1.45)
        th_local = base_th * hr_scale

        # Latido perdido: un RR aproximadamente doble de la mediana local.
        # Ejemplo: 1.8 s cuando alrededor hay 0.9 s.
        if rr_work[i] > 1.55 * med and abs(rr_work[i] / 2.0 - med) < max(th_local, 0.08):
            reconstructed.extend([rr_work[i] / 2.0, rr_work[i] / 2.0])
            reconstructed_bad.extend([True, True])
            missed_n += 1
            original_index_artifacts[i] = True
            i += 1
            continue

        # Latido extra: dos RR consecutivos cortos cuya suma se parece a la mediana.
        if i < len(rr_work) - 1:
            med2 = _local_median_excluding(rr_work, i, radius=5)
            if np.isfinite(med2) and med2 > 0:
                s = rr_work[i] + rr_work[i+1]
                if rr_work[i] < 0.75 * med2 and rr_work[i+1] < 0.75 * med2 and abs(s - med2) < max(th_local, 0.08):
                    reconstructed.append(s)
                    reconstructed_bad.append(True)
                    original_index_artifacts[i] = True
                    original_index_artifacts[i+1] = True
                    extra_n += 1
                    i += 2
                    continue

        reconstructed.append(rr_work[i])
        reconstructed_bad.append(False)
        i += 1

    rr2 = np.asarray(reconstructed, dtype=float)
    pre_bad = np.asarray(reconstructed_bad, dtype=bool)

    # ============================================================
    # Paso B: detección por mediana local robusta threshold-based.
    # ============================================================
    n = len(rr2)
    ser = pd.Series(rr2)
    # Kubios menciona media/mediana local robusta. Usamos mediana 11 y 21 para estabilidad.
    local_med_11 = ser.rolling(window=11, center=True, min_periods=1).median().to_numpy()
    local_med_21 = ser.rolling(window=21, center=True, min_periods=1).median().to_numpy()
    local_med = np.where(np.isfinite(local_med_21), local_med_21, local_med_11)

    hr_scale_vec = np.clip(local_med / 1.0, 0.55, 1.45)
    th_vec = base_th * hr_scale_vec

    bad_threshold = np.abs(rr2 - local_med) > th_vec
    bad_phys = (rr2 < 0.30) | (rr2 > 2.00)

    # ============================================================
    # Paso C: detección automática basada en dRR con umbral adaptativo.
    # ============================================================
    drr = np.diff(rr2, prepend=rr2[0])
    abs_drr = np.abs(drr)
    th_adapt = np.zeros(n)

    for i in range(n):
        lo = max(0, i - 45)
        hi = min(n, i + 46)
        vals = abs_drr[lo:hi]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 8:
            qd = np.nanmedian(abs_drr) / 0.6745 if np.nanmedian(abs_drr) > 0 else 0.03
        else:
            q75, q25 = np.nanpercentile(vals, [75, 25])
            qd = (q75 - q25) / 2.0
            if not np.isfinite(qd) or qd <= 0:
                qd = np.nanmedian(vals) / 0.6745 if np.nanmedian(vals) > 0 else 0.03
        th_adapt[i] = max(5.2 * qd, base_th * 0.60)

    jump = abs_drr > th_adapt

    bad_pattern = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        s1 = np.sign(drr[i])
        s2 = np.sign(drr[i+1])
        # NP / PN: cambio corto-largo o largo-corto
        if jump[i] and jump[i+1] and s1 != 0 and s2 != 0 and s1 != s2:
            bad_pattern[i] = True
            # Para misdetecciones consecutivas, marcar vecindario pequeño.
            if abs(drr[i]) > 1.5 * th_adapt[i] or abs(drr[i+1]) > 1.5 * th_adapt[i+1]:
                bad_pattern[i+1] = True

    # NPN / PNP en ventana de 3 diferencias.
    for i in range(1, n - 2):
        signs = [np.sign(drr[i]), np.sign(drr[i+1]), np.sign(drr[i+2])]
        if all(jump[i:i+3]) and signs[0] != 0 and signs[1] != 0 and signs[2] != 0:
            if signs == [-1, 1, -1] or signs == [1, -1, 1]:
                bad_pattern[i+1] = True

    bad = pre_bad | bad_threshold | bad_phys | bad_pattern

    # En very strong o kubios scientific: segunda pasada más sensible usando dRR local.
    if str(level).lower() in ["very strong", "kubios scientific", "kubios auto"]:
        # Marcar spikes aislados que se separan de ambos vecinos pero vecinos coherentes entre sí.
        for i in range(1, n - 1):
            neigh_med = np.median([rr2[i-1], rr2[i+1]])
            if abs(rr2[i] - neigh_med) > max(base_th * np.clip(neigh_med, 0.55, 1.45), 0.035):
                if abs(rr2[i-1] - rr2[i+1]) < max(0.12 * neigh_med, 0.08):
                    bad[i] = True

    # Evitar sobrecorrección total. Si >60%, relajar a artefactos de alta confianza.
    if np.mean(bad) > 0.60:
        bad = pre_bad | bad_phys | (bad_threshold & bad_pattern)
        if np.mean(bad) < 0.01:
            bad = pre_bad | bad_phys | bad_threshold

    rr_corr = _cubic_interpolate_bad(rr2, bad)

    # Tercera pasada opcional para very strong: corrige residuos grandes tras interpolación.
    if str(level).lower() in ["very strong", "kubios scientific", "kubios auto"]:
        for _ in range(2):
            serc = pd.Series(rr_corr)
            lm = serc.rolling(window=11, center=True, min_periods=1).median().to_numpy()
            th2 = np.maximum(0.04, base_th * np.clip(lm, 0.55, 1.45))
            residual_bad = np.abs(rr_corr - lm) > th2
            residual_bad = residual_bad | (rr_corr < 0.30) | (rr_corr > 2.00)
            if np.sum(residual_bad & ~bad) == 0:
                break
            bad = bad | residual_bad
            rr_corr = _cubic_interpolate_bad(rr_corr, bad)

    info = {
        "level": level,
        "n_artifacts": int(np.sum(bad)),
        "percent_artifacts": float(100 * np.mean(bad)) if len(bad) else 0.0,
        "threshold_seconds_base": float(base_th),
        "missed_beats_corrected": int(missed_n),
        "extra_beats_corrected": int(extra_n),
        "adaptive_drr": True,
        "pattern_detection": "NP/PN/NPN/PNP aproximado",
        "interpolation": "cubic spline",
        "note": "v12.0 Kubios/Lipponen-Tarvainen aproximado: mediana local + dRR adaptativo + reconstrucción missed/extra",
    }

    return rr_corr, bad, info


def cumulative_time(rr):
    return np.cumsum(rr)


def sec_to_hms(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_sec(s):
    parts = [float(p) for p in str(s).strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def cut_segment(rr, start_s, end_s):
    t = cumulative_time(rr)
    return rr[(t >= start_s) & (t <= end_s)]


def empty_windows():
    return {ph: None for ph in PHASES}


def default_windows(t_max):
    """
    Autodivisión flexible del registro.

    Compatible con fases ampliadas:
    Basal, Basal2-Basal5, E1-E6, R1-R6.

    Por defecto:
    - Basal ocupa los primeros 5 min si el registro lo permite.
    - E1-E6 cubren el bloque intermedio.
    - R1-R6 cubren la parte final.
    - Basal2-Basal5 quedan vacías para que el usuario pueda definirlas manualmente.
    """
    t_max = float(max(t_max, 1.0))
    w = empty_windows()

    if t_max < 120:
        step = max(t_max / max(len(PHASES), 1), 10)
        for i, ph in enumerate(PHASES):
            w[ph] = [min(i * step, t_max), min((i + 1) * step, t_max)]
        return w

    # Basal principal
    basal_end = min(300.0, t_max)
    w["Basal"] = [0.0, basal_end]

    # Mantener basales adicionales vacías para edición manual
    for ph in [p for p in PHASES if p.startswith("Basal") and p != "Basal"]:
        w[ph] = None

    # Distribución del resto entre ejercicio y recuperación
    remaining_start = basal_end
    remaining = max(0.0, t_max - remaining_start)

    if remaining <= 0:
        return w

    # 60% del tiempo restante para ejercicio, 40% para recuperación
    exercise_total = remaining * 0.60
    recovery_total = remaining * 0.40

    e_step = exercise_total / 6.0 if exercise_total > 0 else 0
    for i in range(1, 7):
        w[f"E{i}"] = [
            min(remaining_start + (i - 1) * e_step, t_max),
            min(remaining_start + i * e_step, t_max),
        ]

    r_start = remaining_start + exercise_total
    r_step = recovery_total / 6.0 if recovery_total > 0 else 0
    for i in range(1, 7):
        w[f"R{i}"] = [
            min(r_start + (i - 1) * r_step, t_max),
            min(r_start + i * r_step, t_max),
        ]

    return w


def smoothness_priors_detrend(y, lam=500):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y - np.mean(y) if n else y

    I = sparse.eye(n, format="csc")
    e = np.ones(n)
    D2 = sparse.diags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], shape=(n - 2, n), format="csc")
    trend = spsolve(I + (lam ** 2) * (D2.T @ D2), y)
    return y - trend


def interpolate_rr(rr, fs=FS_INTERP, apply_lambda=False, lam=500):
    t = cumulative_time(rr)
    if len(t) < 5:
        return np.array([]), np.array([])

    t = t - t[0]
    x = rr.copy()
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]

    if len(t) < 5:
        return np.array([]), np.array([])

    ti = np.arange(0, t[-1], 1 / fs)

    if len(ti) < 5:
        return np.array([]), np.array([])

    xi = CubicSpline(t, x, bc_type="natural")(ti)

    if apply_lambda:
        xi = smoothness_priors_detrend(xi, lam)

    return ti, xi


def time_metrics(rr):
    rr_ms = rr * 1000.0
    diff = np.diff(rr_ms)
    mean_rr = np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) else np.nan
    nn50 = int(np.sum(np.abs(diff) > 50)) if len(diff) else 0
    pnn50 = 100 * nn50 / len(diff) if len(diff) else np.nan
    sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
    sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

    return {
        "N_RRi": len(rr),
        "Duration_s": float(np.sum(rr)),
        "MeanRR": mean_rr,
        "MeanHR": 60000 / mean_rr if mean_rr > 0 else np.nan,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "NN50": nn50,
        "pNN50": pnn50,
        "SD1": sd1,
        "SD2": sd2,
    }


def psd_metrics(rr):
    ti, xi = interpolate_rr(rr, fs=FS_INTERP, apply_lambda=True, lam=LAMBDA_DEFAULT)

    if len(xi) < 32:
        return {"VLF": np.nan, "LF": np.nan, "HF": np.nan, "TOTAL": np.nan, "LF_HF": np.nan}

    xi_ms = xi * 1000
    xi_ms = xi_ms - np.mean(xi_ms)
    nperseg = min(int(256 * FS_INTERP), len(xi_ms))
    noverlap = int(0.5 * nperseg)

    f, pxx = signal.welch(
        xi_ms,
        fs=FS_INTERP,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    def bp(lo, hi):
        mask = (f >= lo) & (f < hi)
        return np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else np.nan

    vlf, lf, hf = bp(0.0033, 0.04), bp(0.04, 0.15), bp(0.15, 0.40)
    total = np.nansum([vlf, lf, hf])

    return {"VLF": vlf, "LF": lf, "HF": hf, "TOTAL": total, "LF_HF": lf / hf if pd.notna(hf) and hf > 0 else np.nan}


def _phi_apen(x, m, r):
    n = len(x)

    if n <= m + 1:
        return np.nan

    pats = np.array([x[i:i + m] for i in range(n - m + 1)])
    vals = []

    for p in pats:
        dist = np.max(np.abs(pats - p), axis=1)
        c = np.mean(dist <= r)
        if c > 0:
            vals.append(np.log(c))

    return np.mean(vals) if vals else np.nan


def apen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    r = r_ratio * np.std(x, ddof=1)

    if not np.isfinite(r) or r == 0:
        return np.nan

    return _phi_apen(x, m, r) - _phi_apen(x, m + 1, r)


def sampen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    n = len(x)

    if n <= m + 2:
        return np.nan

    r = r_ratio * np.std(x, ddof=1)

    if not np.isfinite(r) or r == 0:
        return np.nan

    def count(mm):
        pats = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = 0
        for i in range(len(pats) - 1):
            dist = np.max(np.abs(pats[i + 1:] - pats[i]), axis=1)
            c += np.sum(dist <= r)
        return c

    b, a = count(m), count(m + 1)

    if a == 0 or b == 0:
        return np.nan

    return -np.log(a / b)


def dfa_calc(x):
    """
    DFA aproximado con rangos iguales a Kubios Advanced Settings II:

    - alpha1 / N1: 4-12 beats
    - alpha2 / N2: 13-64 beats

    La v10.2 usaba 4-16 y 16-64; por eso alpha2 podía salir desplazada.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 50:
        return np.nan, np.nan

    y = np.cumsum(x - np.mean(x))

    max_scale = min(KUBIOS_DFA_ALPHA2_RANGE[1], max(5, n // 4))
    scales = np.arange(4, max_scale + 1, dtype=int)

    ss, ff = [], []

    for s in scales:
        if s < 4 or n // s < 2:
            continue

        rms = []
        for i in range(n // s):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            co = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(co, t)) ** 2)))

        val = np.sqrt(np.mean(np.asarray(rms) ** 2))
        if val > 0 and np.isfinite(val):
            ss.append(s)
            ff.append(val)

    ss, ff = np.asarray(ss), np.asarray(ff)

    if len(ss) < 4:
        return np.nan, np.nan

    a1_min, a1_max = KUBIOS_DFA_ALPHA1_RANGE
    a2_min, a2_max = KUBIOS_DFA_ALPHA2_RANGE

    m1 = (ss >= a1_min) & (ss <= a1_max)
    m2 = (ss >= a2_min) & (ss <= a2_max)

    alpha1 = np.polyfit(np.log(ss[m1]), np.log(ff[m1]), 1)[0] if np.sum(m1) >= 2 else np.nan
    alpha2 = np.polyfit(np.log(ss[m2]), np.log(ff[m2]), 1)[0] if np.sum(m2) >= 2 else np.nan

    return alpha1, alpha2



def d2_calc(x, emb_dim=10, tau=1, max_n=700):
    """
    Dimensión de correlación D2 aproximada.

    Implementación Grassberger-Procaccia simplificada:
    - embedding emb_dim=10, tau=1
    - distancia Chebyshev
    - ajuste log(C(r)) vs log(r) en zona intermedia
    No pretende ser idéntica al motor propietario de Kubios, pero permite comparar D2.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) > max_n:
        x = x[np.linspace(0, len(x) - 1, max_n).astype(int)]

    n = len(x) - (emb_dim - 1) * tau
    if n < 40:
        return np.nan

    X = np.array([x[i:i + emb_dim * tau:tau] for i in range(n)])
    if X.shape[0] < 40:
        return np.nan

    # Normalización para estabilidad numérica
    sd = np.std(X)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    X = (X - np.mean(X)) / sd

    d = pdist(X, metric="chebyshev")
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 100:
        return np.nan

    # Radios en percentiles intermedios para evitar saturación.
    r_min, r_max = np.percentile(d, [5, 60])
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min <= 0 or r_max <= r_min:
        return np.nan

    radii = np.logspace(np.log10(r_min), np.log10(r_max), 24)
    C = np.array([np.mean(d < r) for r in radii])

    mask = (C > 0.01) & (C < 0.80) & np.isfinite(C)
    if np.sum(mask) < 5:
        mask = (C > 0) & (C < 1) & np.isfinite(C)

    if np.sum(mask) < 3:
        return np.nan

    try:
        slope = np.polyfit(np.log(radii[mask]), np.log(C[mask]), 1)[0]
        return float(slope) if np.isfinite(slope) else np.nan
    except Exception:
        return np.nan


def rqa_calc(x, emb_dim=KUBIOS_RQA_EMB_DIM, tau=1, l_min=2, max_n=500):
    x = np.asarray(x, dtype=float)

    if len(x) > max_n:
        x = x[np.linspace(0, len(x) - 1, max_n).astype(int)]

    n = len(x) - (emb_dim - 1) * tau

    if n < 20:
        return {"REC": np.nan, "DET": np.nan, "Lmean": np.nan, "Lmax": np.nan, "ShanEn": np.nan}

    D = squareform(pdist(np.array([x[i:i + emb_dim * tau:tau] for i in range(n)])))
    radius = KUBIOS_RQA_THRESHOLD_SD * np.std(x, ddof=1)
    R = (D <= radius).astype(int)
    np.fill_diagonal(R, 0)
    rec = 100 * R.sum() / (n * n - n)

    lens = []

    for k in range(-n + 1, n):
        diag = np.diag(R, k=k)
        c = 0

        for val in diag:
            if val:
                c += 1
            else:
                if c >= l_min:
                    lens.append(c)
                c = 0

        if c >= l_min:
            lens.append(c)

    if not lens:
        return {"REC": rec, "DET": 0, "Lmean": 0, "Lmax": 0, "ShanEn": 0}

    lens = np.asarray(lens)
    det = 100 * lens.sum() / R.sum() if R.sum() > 0 else 0
    vals, counts = np.unique(lens, return_counts=True)
    p = counts / counts.sum()

    return {"REC": rec, "DET": det, "Lmean": np.mean(lens), "Lmax": np.max(lens), "ShanEn": -np.sum(p * np.log(p))}




def hvg_graph(x, max_nodes=500):
    if nx is None:
        return None

    x = np.asarray(x, dtype=float)
    if len(x) > max_nodes:
        idx = np.linspace(0, len(x) - 1, max_nodes).astype(int)
        x = x[idx]

    n = len(x)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        G.add_edge(i, i + 1)
        for j in range(i + 2, n):
            if np.max(x[i + 1:j]) < min(x[i], x[j]):
                G.add_edge(i, j)

    return G



def classify_hvg_graph_type(metrics):
    """
    Clasificación orientativa del tipo de grafo HVG.

    Tipos:
    - Libre de escala / jerárquico
    - Small-world funcional
    - Lineal / cadena
    - Regular / homogéneo
    - Complejo mixto
    """
    try:
        nodes = float(metrics.get("HVG_nodes", np.nan))
        edges = float(metrics.get("HVG_edges", np.nan))
        degree_mean = float(metrics.get("HVG_degree_mean", np.nan))
        degree_max = float(metrics.get("HVG_degree_max", np.nan))
        hubs = float(metrics.get("HVG_hubs_p90", np.nan))
        clustering = float(metrics.get("HVG_clustering", np.nan))
        lam = float(metrics.get("HVG_lambda", np.nan))
        path = float(metrics.get("HVG_path_length", np.nan))
        diameter = float(metrics.get("HVG_diameter", np.nan))
    except Exception:
        return {
            "HVG_graph_type": "No clasificable",
            "HVG_graph_interpretation": "No hay métricas suficientes para clasificar el grafo.",
            "HVG_graph_score_scale_free": np.nan,
            "HVG_graph_score_small_world": np.nan,
            "HVG_graph_score_chain": np.nan,
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay métricas suficientes para valorar compactación/dispersión.",
        }

    if not np.isfinite(nodes) or nodes < 20:
        return {
            "HVG_graph_type": "No clasificable",
            "HVG_graph_interpretation": "Ventana demasiado corta o grafo insuficiente.",
            "HVG_graph_score_scale_free": np.nan,
            "HVG_graph_score_small_world": np.nan,
            "HVG_graph_score_chain": np.nan,
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay métricas suficientes para valorar compactación/dispersión.",
        }

    edge_density = edges / max(nodes, 1)
    hub_ratio = hubs / max(nodes, 1)
    degree_contrast = degree_max / max(degree_mean, 1e-9)
    diameter_rel = diameter / max(nodes, 1) if np.isfinite(diameter) else np.nan
    path_rel = path / max(nodes, 1) if np.isfinite(path) else np.nan

    scale_free_score = 0
    if np.isfinite(degree_contrast):
        scale_free_score += min(45, 12 * degree_contrast)
    if np.isfinite(hub_ratio):
        scale_free_score += min(25, 300 * hub_ratio)
    if np.isfinite(lam):
        if lam < 0.45:
            scale_free_score += 20
        elif lam < 0.75:
            scale_free_score += 12
        elif lam < 1.1:
            scale_free_score += 6
    if np.isfinite(clustering) and clustering > 0.08:
        scale_free_score += 10
    scale_free_score = float(min(100, scale_free_score))

    small_world_score = 0
    if np.isfinite(clustering):
        small_world_score += min(45, clustering * 120)
    if np.isfinite(path_rel):
        if path_rel < 0.12:
            small_world_score += 30
        elif path_rel < 0.20:
            small_world_score += 18
        elif path_rel < 0.30:
            small_world_score += 8
    if np.isfinite(diameter_rel):
        if diameter_rel < 0.25:
            small_world_score += 20
        elif diameter_rel < 0.40:
            small_world_score += 10
    if np.isfinite(edge_density) and edge_density > 1.3:
        small_world_score += 5
    small_world_score = float(min(100, small_world_score))

    chain_score = 0
    if np.isfinite(edge_density):
        if edge_density < 1.15:
            chain_score += 40
        elif edge_density < 1.35:
            chain_score += 25
    if np.isfinite(degree_mean):
        if degree_mean < 2.4:
            chain_score += 25
        elif degree_mean < 3.0:
            chain_score += 12
    if np.isfinite(diameter_rel):
        if diameter_rel > 0.45:
            chain_score += 25
        elif diameter_rel > 0.30:
            chain_score += 12
    if np.isfinite(clustering) and clustering < 0.05:
        chain_score += 10
    chain_score = float(min(100, chain_score))

    if chain_score >= 65:
        graph_type = "Lineal / cadena"
        interp = (
            "Grafo con pocas conexiones transversales, bajo grado medio y/o diámetro relativamente alto. "
            "Sugiere una dinámica RRi más secuencial, con menor integración global."
        )
    elif scale_free_score >= 60 and scale_free_score >= small_world_score:
        graph_type = "Libre de escala / jerárquico"
        interp = (
            "Grafo con hubs relativamente marcados y distribución de grados heterogénea. "
            "Sugiere una dinámica con nodos dominantes que conectan distintas partes de la señal."
        )
    elif small_world_score >= 60:
        graph_type = "Small-world funcional"
        interp = (
            "Grafo con agrupamiento local y caminos relativamente cortos. "
            "Sugiere equilibrio entre especialización local e integración global."
        )
    elif scale_free_score >= 45 and small_world_score >= 45:
        graph_type = "Complejo mixto"
        interp = (
            "Combina rasgos de hubs y conectividad local/global. "
            "Puede indicar una organización intermedia de la dinámica RRi."
        )
    else:
        graph_type = "Regular / homogéneo"
        interp = (
            "Grafo sin hubs claramente dominantes y con conectividad relativamente homogénea. "
            "Sugiere una dinámica más uniforme o menos jerárquica."
        )

    return {
        "HVG_graph_type": graph_type,
        "HVG_graph_interpretation": interp,
        "HVG_graph_score_scale_free": round(scale_free_score, 1),
        "HVG_graph_score_small_world": round(small_world_score, 1),
        "HVG_graph_score_chain": round(chain_score, 1),
    }




# ============================================================
# INTERPRETACIÓN AVANZADA HVG / GRAFOS
# ============================================================

def _safe_float(x, default=np.nan):
    try:
        v = pd.to_numeric(x, errors="coerce")
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


def hvg_reference_ranges():
    """
    Rangos orientativos para interpretación clínica/topológica.
    No son rangos diagnósticos cerrados; sirven para contextualizar.
    """
    return pd.DataFrame([
        {
            "Métrica": "HVG_clustering",
            "Qué mide": "Agrupamiento local de la red.",
            "Muy bajo": "< 0.20",
            "Bajo": "0.20 - 0.40",
            "Normal/orientativo": "0.40 - 0.70",
            "Alto": "> 0.70",
            "Lectura clínica/topológica": "Más alto = mayor compactación local y organización por vecindarios."
        },
        {
            "Métrica": "HVG_degree_mean",
            "Qué mide": "Conexiones promedio por nodo.",
            "Muy bajo": "< 2.5",
            "Bajo": "2.5 - 3.5",
            "Normal/orientativo": "3.5 - 5",
            "Alto": "> 5",
            "Lectura clínica/topológica": "Más alto = mayor conectividad global de la señal transformada en red."
        },
        {
            "Métrica": "HVG_degree_max",
            "Qué mide": "Grado del nodo más conectado.",
            "Muy bajo": "< 6",
            "Bajo": "6 - 10",
            "Normal/orientativo": "10 - 20",
            "Alto": "> 20",
            "Lectura clínica/topológica": "Valores altos indican presencia de hubs o nodos dominantes."
        },
        {
            "Métrica": "HVG_hubs_p90",
            "Qué mide": "Nodos con conectividad alta, por encima del percentil 90.",
            "Muy bajo": "< 20",
            "Bajo": "20 - 40",
            "Normal/orientativo": "40 - 80",
            "Alto": "> 80",
            "Lectura clínica/topológica": "Más hubs suelen indicar mayor centralización e integración."
        },
        {
            "Métrica": "HVG_lambda",
            "Qué mide": "Pendiente/exponente aproximado de la distribución de grados.",
            "Muy bajo": "< 0.30",
            "Bajo": "0.30 - 0.80",
            "Normal/orientativo": "0.80 - 1.50",
            "Alto": "> 1.50",
            "Lectura clínica/topológica": "Valores bajos-moderados son compatibles con cola pesada/hubs; valores altos sugieren red más homogénea."
        },
        {
            "Métrica": "HVG_path_length",
            "Qué mide": "Camino medio entre nodos.",
            "Muy bajo": "< 8",
            "Bajo": "8 - 15",
            "Normal/orientativo": "15 - 25",
            "Alto": "> 25",
            "Lectura clínica/topológica": "Menor camino medio = mejor integración global."
        },
        {
            "Métrica": "HVG_diameter",
            "Qué mide": "Distancia máxima entre dos nodos conectados.",
            "Muy bajo": "< 10",
            "Bajo": "10 - 25",
            "Normal/orientativo": "25 - 40",
            "Alto": "> 40",
            "Lectura clínica/topológica": "Diámetro menor = grafo más compacto; diámetro alto = red más dispersa."
        },
    ])


def hvg_metric_reference_label(metric, value):
    """
    Etiqueta cualitativa orientativa por métrica.
    """
    v = _safe_float(value)

    if not np.isfinite(v):
        return "No clasificable"

    if metric == "HVG_clustering":
        if v < 0.20: return "Muy bajo"
        if v < 0.40: return "Bajo"
        if v <= 0.70: return "Normal/orientativo"
        return "Alto"

    if metric == "HVG_degree_mean":
        if v < 2.5: return "Muy bajo"
        if v < 3.5: return "Bajo"
        if v <= 5: return "Normal/orientativo"
        return "Alto"

    if metric == "HVG_degree_max":
        if v < 6: return "Muy bajo"
        if v < 10: return "Bajo"
        if v <= 20: return "Normal/orientativo"
        return "Alto"

    if metric == "HVG_hubs_p90":
        if v < 20: return "Muy bajo"
        if v < 40: return "Bajo"
        if v <= 80: return "Normal/orientativo"
        return "Alto"

    if metric == "HVG_lambda":
        if v < 0.30: return "Muy bajo"
        if v < 0.80: return "Bajo/compatible hubs"
        if v <= 1.50: return "Normal/orientativo"
        return "Alto/homogéneo"

    if metric == "HVG_path_length":
        if v < 8: return "Muy bajo/compacto"
        if v < 15: return "Bajo/compacto"
        if v <= 25: return "Normal/orientativo"
        return "Alto/disperso"

    if metric == "HVG_diameter":
        if v < 10: return "Muy bajo/compacto"
        if v < 25: return "Bajo/compacto"
        if v <= 40: return "Normal/orientativo"
        return "Alto/disperso"

    return ""


def hvg_topology_state(metrics):
    """
    Clasificación compactación local vs dispersión global.

    Se combina información de:
    - clustering
    - hubs
    - grado máximo/medio
    - camino medio
    - diámetro

    Devuelve:
    - estado textual
    - índice aproximado en escala -2 a +2
    - explicación.
    """
    nodes = _safe_float(metrics.get("HVG_nodes"))
    clustering = _safe_float(metrics.get("HVG_clustering"))
    hubs = _safe_float(metrics.get("HVG_hubs_p90"))
    degree_mean = _safe_float(metrics.get("HVG_degree_mean"))
    degree_max = _safe_float(metrics.get("HVG_degree_max"))
    path = _safe_float(metrics.get("HVG_path_length"))
    diameter = _safe_float(metrics.get("HVG_diameter"))

    if not np.isfinite(nodes) or nodes <= 0:
        return {
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay nodos suficientes para valorar compactación/dispersión."
        }

    hub_ratio = hubs / max(nodes, 1) if np.isfinite(hubs) else np.nan
    degree_contrast = degree_max / max(degree_mean, 1e-9) if np.isfinite(degree_max) and np.isfinite(degree_mean) else np.nan
    path_rel = path / max(nodes, 1) if np.isfinite(path) else np.nan
    diameter_rel = diameter / max(nodes, 1) if np.isfinite(diameter) else np.nan

    score = 0.0

    # Compactación local
    if np.isfinite(clustering):
        if clustering >= 0.70: score += 0.9
        elif clustering >= 0.50: score += 0.6
        elif clustering >= 0.35: score += 0.25
        elif clustering < 0.20: score -= 0.5

    if np.isfinite(hub_ratio):
        if hub_ratio >= 0.12: score += 0.45
        elif hub_ratio >= 0.08: score += 0.30
        elif hub_ratio < 0.04: score -= 0.25

    if np.isfinite(degree_contrast):
        if degree_contrast >= 4.0: score += 0.45
        elif degree_contrast >= 3.0: score += 0.25

    # Dispersión global
    if np.isfinite(path_rel):
        if path_rel < 0.08: score += 0.40
        elif path_rel < 0.15: score += 0.25
        elif path_rel > 0.30: score -= 0.55

    if np.isfinite(diameter_rel):
        if diameter_rel < 0.15: score += 0.45
        elif diameter_rel < 0.25: score += 0.25
        elif diameter_rel > 0.40: score -= 0.60

    score = float(np.clip(score, -2.0, 2.0))

    if score >= 1.0:
        state = "Compacto local"
        interp = (
            "Red con alta compactación local: predominan agrupamientos, hubs y distancias relativamente cortas. "
            "Sugiere una organización más integrada y centralizada."
        )
    elif score >= 0.3:
        state = "Tendencia compacta"
        interp = (
            "Red con tendencia a la compactación: conserva conectividad local/global razonable, aunque sin máxima centralización."
        )
    elif score > -0.3:
        state = "Equilibrado"
        interp = (
            "Red con equilibrio entre integración y dispersión. No predomina claramente la compactación ni la fragmentación."
        )
    elif score > -1.0:
        state = "Tendencia dispersa"
        interp = (
            "Red con tendencia a mayor dispersión: menor compactación local o caminos más largos entre nodos."
        )
    else:
        state = "Disperso global"
        interp = (
            "Red más fragmentada o menos integrada globalmente, con caminos/diámetro relativamente largos y menor centralización."
        )

    return {
        "HVG_topology_state": state,
        "HVG_compactness_index": round(score, 2),
        "HVG_topology_interpretation": interp
    }


def hvg_summary_card(metrics):
    """
    Resumen corto para mostrar encima de las tablas.
    """
    graph_type = metrics.get("HVG_graph_type", "No clasificable")
    topology = metrics.get("HVG_topology_state", "No clasificable")
    compactness = metrics.get("HVG_compactness_index", np.nan)

    scale_free = metrics.get("HVG_graph_score_scale_free", np.nan)
    small_world = metrics.get("HVG_graph_score_small_world", np.nan)
    chain = metrics.get("HVG_graph_score_chain", np.nan)

    return pd.DataFrame([
        {"Aspecto": "Tipo de grafo", "Resultado": graph_type},
        {"Aspecto": "Organización topológica", "Resultado": topology},
        {"Aspecto": "Índice compactación (-2 a +2)", "Resultado": compactness},
        {"Aspecto": "Score libre de escala (0-100)", "Resultado": scale_free},
        {"Aspecto": "Score small-world (0-100)", "Resultado": small_world},
        {"Aspecto": "Score cadena/dispersión (0-100)", "Resultado": chain},
        {"Aspecto": "Lectura compactación/dispersión", "Resultado": metrics.get("HVG_topology_interpretation", "")},
        {"Aspecto": "Lectura tipo de grafo", "Resultado": metrics.get("HVG_graph_interpretation", "")},
    ])


def hvg_reference_value_table(metrics_df):
    """
    Tabla larga con valor, rango orientativo y significado de cada métrica HVG.
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    hvg_cols = [
        "HVG_graph_type",
        "HVG_topology_state",
        "HVG_compactness_index",
        "HVG_graph_score_scale_free",
        "HVG_graph_score_small_world",
        "HVG_graph_score_chain",
        "HVG_nodes",
        "HVG_edges",
        "HVG_degree_mean",
        "HVG_degree_max",
        "HVG_hubs_p90",
        "HVG_clustering",
        "HVG_lambda",
        "HVG_path_length",
        "HVG_diameter",
        "HVG_graph_interpretation",
        "HVG_topology_interpretation",
    ]

    explanations = {
        "HVG_graph_type": "Tipo de organización topológica dominante.",
        "HVG_topology_state": "Clasificación compactación local vs dispersión global.",
        "HVG_compactness_index": "Índice aproximado -2 a +2: valores positivos indican compactación local; negativos, dispersión global.",
        "HVG_graph_score_scale_free": "Score 0-100 de rasgos libre de escala / hubs.",
        "HVG_graph_score_small_world": "Score 0-100 de rasgos small-world: clustering + caminos cortos.",
        "HVG_graph_score_chain": "Score 0-100 de rasgos lineales/cadena.",
        "HVG_nodes": "Número de nodos analizados.",
        "HVG_edges": "Número de conexiones visibles entre nodos.",
        "HVG_degree_mean": "Conexiones promedio por nodo.",
        "HVG_degree_max": "Grado del nodo más conectado.",
        "HVG_hubs_p90": "Número de nodos con conectividad alta.",
        "HVG_clustering": "Agrupamiento local de la red.",
        "HVG_lambda": "Pendiente/exponente aproximado de la distribución de grados.",
        "HVG_path_length": "Camino medio entre nodos.",
        "HVG_diameter": "Distancia máxima entre dos nodos conectados.",
        "HVG_graph_interpretation": "Interpretación automática del tipo de grafo.",
        "HVG_topology_interpretation": "Interpretación automática de compactación/dispersión.",
    }

    rows = []
    for fase, row in metrics_df.iterrows():
        for col in hvg_cols:
            if col in metrics_df.columns:
                rows.append({
                    "Fase": fase,
                    "Métrica": col,
                    "Valor": row[col],
                    "Rango orientativo": hvg_metric_reference_label(col, row[col]),
                    "Qué significa": explanations.get(col, ""),
                })

    return pd.DataFrame(rows)


def hvg_metrics(rr, max_nodes=500):
    if nx is None:
        return {
            "HVG_nodes": np.nan,
            "HVG_edges": np.nan,
            "HVG_degree_mean": np.nan,
            "HVG_degree_max": np.nan,
            "HVG_hubs_p90": np.nan,
            "HVG_clustering": np.nan,
            "HVG_lambda": np.nan,
            "HVG_path_length": np.nan,
            "HVG_diameter": np.nan,
            "HVG_graph_type": "No clasificable",
            "HVG_graph_interpretation": "No hay métricas suficientes para clasificar el grafo.",
            "HVG_graph_score_scale_free": np.nan,
            "HVG_graph_score_small_world": np.nan,
            "HVG_graph_score_chain": np.nan,
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay métricas suficientes para valorar compactación/dispersión.",
        }

    G = hvg_graph(rr, max_nodes=max_nodes)
    if G is None or G.number_of_nodes() < 20:
        return {
            "HVG_nodes": G.number_of_nodes() if G is not None else 0,
            "HVG_edges": np.nan,
            "HVG_degree_mean": np.nan,
            "HVG_degree_max": np.nan,
            "HVG_hubs_p90": np.nan,
            "HVG_clustering": np.nan,
            "HVG_lambda": np.nan,
            "HVG_path_length": np.nan,
            "HVG_diameter": np.nan,
            "HVG_graph_type": "No clasificable",
            "HVG_graph_interpretation": "No hay métricas suficientes para clasificar el grafo.",
            "HVG_graph_score_scale_free": np.nan,
            "HVG_graph_score_small_world": np.nan,
            "HVG_graph_score_chain": np.nan,
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay métricas suficientes para valorar compactación/dispersión.",
        }

    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])

    vals, counts = np.unique(deg, return_counts=True)
    p = counts / counts.sum()
    mask = (vals > 1) & (p > 0)
    lam = -np.polyfit(vals[mask], np.log(p[mask]), 1)[0] if np.sum(mask) >= 2 else np.nan

    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        path_length = np.nan
        diameter = np.nan

    base_metrics = {
        "HVG_nodes": n,
        "HVG_edges": m,
        "HVG_degree_mean": 2 * m / n if n else np.nan,
        "HVG_degree_max": np.max(deg) if len(deg) else np.nan,
        "HVG_hubs_p90": int(np.sum(deg >= np.percentile(deg, 90))) if len(deg) else np.nan,
        "HVG_clustering": nx.average_clustering(G) if n else np.nan,
        "HVG_lambda": lam,
        "HVG_path_length": path_length,
        "HVG_diameter": diameter,
    }
    base_metrics.update(classify_hvg_graph_type(base_metrics))
    base_metrics.update(hvg_topology_state(base_metrics))
    return base_metrics


def hvg_network_figure(rr, title="HVG", max_nodes=140):
    fig = go.Figure()
    if nx is None:
        fig.update_layout(title="NetworkX no disponible")
        return fig

    G = hvg_graph(rr, max_nodes=max_nodes)
    if G is None or G.number_of_nodes() == 0:
        fig.update_layout(title="Sin grafo")
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.18, iterations=60)

    edge_x, edge_y = [], []
    for a, b in G.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]

    deg = dict(G.degree())
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [6 + deg[n] * 2.5 for n in G.nodes()]
    node_text = [f"n={n}<br>grado={deg[n]}" for n in G.nodes()]

    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=node_size), text=node_text, hoverinfo="text", showlegend=False))
    fig.update_layout(title=title, height=520, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig





def poincare_panel_figure(record_data, global_windows, record_windows, phase, use_independent):
    """
    Poincaré en paneles separados.

    v15.4.0: si se comparan varios pacientes, cada paciente ocupa una columna
    de la cuadrícula y sus registros se apilan cronológicamente en esa columna.
    """
    records = list(record_data.keys())
    n = len(records)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title="Sin registros")
        return fig

    multi_patient = _has_multiple_patients(records)
    if multi_patient:
        groups = _records_grouped_by_patient(records)
        patient_ids = list(groups.keys())
        cols = len(patient_ids)
        rows = max(len(v) for v in groups.values())
        slots = []
        titles = []
        for row in range(rows):
            for pid in patient_ids:
                recs = groups[pid]
                rec = recs[row] if row < len(recs) else None
                slots.append(rec)
                if rec is None:
                    titles.append("")
                else:
                    titles.append(
                        f"{_patient_display_name(pid)} · {_record_axis_label(rec, multiline=False)}"
                    )
    else:
        cols = min(2, n)
        rows = int(np.ceil(n / cols))
        slots = records + [None] * (rows * cols - n)
        titles = [_short_record_label(r, 30) if r is not None else "" for r in slots]

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        horizontal_spacing=0.08 if cols <= 2 else 0.05,
        vertical_spacing=_safe_vertical_spacing(rows, 0.14)
    )

    global_min = np.inf
    global_max = -np.inf
    cache = {}
    for rec in records:
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)
        if w is None:
            cache[rec] = None
            continue
        seg = cut_segment(record_data[rec]["rr"], w[0], w[1])
        if len(seg) < 3:
            cache[rec] = None
            continue
        rr_ms = seg * 1000
        x = rr_ms[:-1]
        y = rr_ms[1:]
        diff = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan
        cache[rec] = (x, y, sd1, sd2)
        global_min = min(global_min, np.nanmin(x), np.nanmin(y))
        global_max = max(global_max, np.nanmax(x), np.nanmax(y))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        fig = go.Figure()
        fig.update_layout(title=f"Poincaré {phase}: sin datos suficientes")
        return fig

    pad = max(20, 0.05 * (global_max - global_min))
    global_min -= pad
    global_max += pad

    for idx, rec in enumerate(slots):
        r = idx // cols + 1
        c = idx % cols + 1
        if rec is None:
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)
            continue
        item = cache.get(rec)
        if item is None:
            fig.add_annotation(
                text="Sin datos suficientes", x=0.5, y=0.5,
                xref=f"x{idx+1 if idx > 0 else ''} domain",
                yref=f"y{idx+1 if idx > 0 else ''} domain", showarrow=False
            )
            continue
        x, y, sd1, sd2 = item
        pid = infer_patient_id(rec)
        color = _export_color_for(list(_records_grouped_by_patient(records).keys()).index(normalize_patient_id(pid))) if multi_patient else _export_color_for(idx)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=5, opacity=0.62, color=color),
            name=_short_record_label(rec, 24), showlegend=False,
            hovertemplate="RR(n): %{x:.1f} ms<br>RR(n+1): %{y:.1f} ms<extra></extra>"
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=[global_min, global_max], y=[global_min, global_max],
            mode="lines", line=dict(width=1, dash="dash", color=color),
            showlegend=False, hoverinfo="skip"
        ), row=r, col=c)
        fig.add_annotation(
            text=f"SD1={sd1:.1f} ms<br>SD2={sd2:.1f} ms",
            x=0.03, y=0.97,
            xref=f"x{idx+1 if idx > 0 else ''} domain",
            yref=f"y{idx+1 if idx > 0 else ''} domain",
            showarrow=False, align="left",
            bgcolor="rgba(0,0,0,0.25)",
            bordercolor="rgba(255,255,255,0.25)"
        )
        fig.update_xaxes(range=[global_min, global_max], title_text="RR(n) ms", row=r, col=c)
        fig.update_yaxes(range=[global_min, global_max], title_text="RR(n+1) ms", row=r, col=c)

    # Cabecera de columnas cuando hay varios pacientes.
    if multi_patient:
        for ci, pid in enumerate(patient_ids, start=1):
            center = (ci - 0.5) / cols
            fig.add_annotation(
                x=center, y=1.08, xref="paper", yref="paper",
                text=f"<b>{_patient_display_name(pid)}</b>",
                showarrow=False, font=dict(size=15)
            )

    fig.update_layout(
        height=max(560, rows * 470),
        title=f"Poincaré en paneles separados · {phase}",
        margin=dict(l=40, r=40, t=105 if multi_patient else 80, b=40)
    )
    return fig

def hvg_network_compare_figure(record_data, global_windows, record_windows, phase, use_independent, max_nodes=120):
    """
    Muestra grafos HVG en paneles comparables.

    v15.4.0: en comparación multipaciente, cada paciente ocupa una columna y
    sus registros se apilan cronológicamente dentro de esa columna.
    """
    if nx is None:
        fig = go.Figure()
        fig.update_layout(title="NetworkX no disponible")
        return fig

    records = list(record_data.keys())
    n = len(records)
    if n == 0:
        return go.Figure()

    multi_patient = _has_multiple_patients(records)
    if multi_patient:
        groups = _records_grouped_by_patient(records)
        patient_ids = list(groups.keys())
        cols = len(patient_ids)
        rows = max(len(v) for v in groups.values())
        slots, titles = [], []
        for row in range(rows):
            for pid in patient_ids:
                recs = groups[pid]
                rec = recs[row] if row < len(recs) else None
                slots.append(rec)
                titles.append(
                    f"{_patient_display_name(pid)} · {_record_axis_label(rec, multiline=False)}"
                    if rec is not None else ""
                )
    else:
        cols = min(2, n)
        rows = int(np.ceil(n / cols))
        slots = records + [None] * (rows * cols - n)
        titles = [_short_record_label(r, 28) if r is not None else "" for r in slots]

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        horizontal_spacing=0.04 if cols <= 2 else 0.03,
        vertical_spacing=_safe_vertical_spacing(rows, 0.12)
    )

    group_keys = list(_records_grouped_by_patient(records).keys()) if multi_patient else []

    for idx, rec in enumerate(slots):
        r = idx // cols + 1
        c = idx % cols + 1
        if rec is None:
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)
            continue

        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)
        if w is None:
            continue
        seg = cut_segment(record_data[rec]["rr"], w[0], w[1])
        if len(seg) < 20:
            continue
        G = hvg_graph(seg, max_nodes=max_nodes)
        if G is None or G.number_of_nodes() == 0:
            continue

        pos = nx.spring_layout(G, seed=42, k=0.20, iterations=60)
        edge_x, edge_y = [], []
        for a, b in G.edges():
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]

        deg = dict(G.degree())
        node_x = [pos[nn][0] for nn in G.nodes()]
        node_y = [pos[nn][1] for nn in G.nodes()]
        node_size = [5 + deg[nn] * 2.2 for nn in G.nodes()]
        node_text = [f"{rec}<br>n={nn}<br>grado={deg[nn]}" for nn in G.nodes()]
        if multi_patient:
            pid = normalize_patient_id(infer_patient_id(rec))
            color = _export_color_for(group_keys.index(pid))
        else:
            color = _export_color_for(idx)

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.45, color=color),
            hoverinfo="skip", showlegend=False
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(size=node_size, opacity=0.82, color=color),
            text=node_text, hoverinfo="text", showlegend=False
        ), row=r, col=c)
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    if multi_patient:
        for ci, pid in enumerate(patient_ids, start=1):
            center = (ci - 0.5) / cols
            fig.add_annotation(
                x=center, y=1.08, xref="paper", yref="paper",
                text=f"<b>{_patient_display_name(pid)}</b>",
                showarrow=False, font=dict(size=15)
            )

    fig.update_layout(
        height=max(520, rows * 440),
        title=f"HVG comparativo · {phase}",
        margin=dict(l=20, r=20, t=100 if multi_patient else 70, b=20)
    )
    return fig

def poincare_figure(record_data, global_windows, record_windows, phase, use_independent):
    fig = go.Figure()

    for rec, data in record_data.items():
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)
        if w is None:
            continue

        seg = cut_segment(data["rr"], w[0], w[1])
        if len(seg) < 3:
            continue

        rr_ms = seg * 1000
        x = rr_ms[:-1]
        y = rr_ms[1:]
        diff = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"{rec} · SD1={sd1:.1f}, SD2={sd2:.1f}",
            marker=dict(size=6, opacity=0.65)
        ))

    fig.update_layout(
        title=f"Poincaré comparativo · {phase}",
        height=560,
        xaxis_title="RR(n) ms",
        yaxis_title="RR(n+1) ms",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig






# ============================================================
# ENTROPÍAS COHERENTES: SampEn y MSE con la misma entrada y tolerancia
# ============================================================

def _resolve_entropy_radius(series, reference, r_factor=0.2, radius_mode="fixed_entropy_sd"):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    ref = x if reference is None else np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    base = x if radius_mode == "scale_sd" else ref
    if len(base) <= 2:
        return np.nan
    return float(r_factor * np.std(base, ddof=1))


def _sample_entropy_counts(x, m=2, r=None, theiler_window=0):
    """
    Conteos SampEn Richman-Moorman:
    - distancia Chebyshev,
    - sin self-matches,
    - compara sólo i < j,
    - opcionalmente excluye comparaciones temporalmente próximas
      mediante ventana de Theiler.

    theiler_window:
    - 0: no excluye vecinos temporales aparte del self-match.
    - 1: excluye patrones consecutivos, |i-j| <= 1.
    - 2: excluye |i-j| <= 2, etc.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) <= m + 2:
        return np.nan, np.nan

    if r is None:
        sd = np.std(x, ddof=1)
        r = KUBIOS_ENTROPY_R_FACTOR * sd

    if not np.isfinite(r) or r <= 0:
        return np.nan, np.nan

    theiler_window = int(max(0, theiler_window or 0))

    def _count(mm):
        n_templates = len(x) - mm + 1
        if n_templates <= 1:
            return np.nan

        templates = np.array([x[i:i + mm] for i in range(n_templates)])
        c = 0

        for i in range(n_templates - 1):
            start_j = i + 1
            dist = np.max(np.abs(templates[start_j:] - templates[i]), axis=1)
            if theiler_window > 0:
                js = np.arange(start_j, n_templates)
                keep = (js - i) > theiler_window
                dist = dist[keep]
            c += np.sum(dist <= r)

        return float(c)

    b = _count(m)
    a = _count(m + 1)
    return b, a


def _sample_entropy_core(x, m=2, r=None, zero_policy="nan", theiler_window=0):
    """
    SampEn con parámetros Kubios visibles:
    - m = 2
    - r = 0.2 x SD
    - sin self-matches
    - ventana de Theiler opcional.

    v11.3 corrige un error de v11.2 donde se había introducido
    accidentalmente una referencia a variables no definidas (cg/ref).
    """
    b, a = _sample_entropy_counts(x, m=m, r=r, theiler_window=theiler_window)

    if not np.isfinite(b) or b <= 0:
        return np.nan

    if not np.isfinite(a) or a <= 0:
        if zero_policy == "half_count":
            a = 0.5
        elif zero_policy == "one_count":
            a = 1.0
        else:
            return np.nan

    return -np.log(a / b)


def _prepare_entropy_rr_lambda500(rr):
    """
    Entrada única para ApEn/SampEn/MSE:
    RR en ms con smoothness priors λ=500, como estaba definido para la app.

    Importante:
    - La entrada debe ser RR en segundos.
    - Devuelve RR en ms detrendido por smoothness priors.
    """
    x_ms = np.asarray(rr, dtype=float) * 1000.0
    x_ms = x_ms[np.isfinite(x_ms)]

    if len(x_ms) < 5:
        return x_ms

    return smoothness_priors_detrend(x_ms, LAMBDA_DEFAULT)


def _entropy_debug_values(rr_entropy):
    """
    Valores de control para verificar si λ=500 y tolerancia se aplican.
    """
    x = np.asarray(rr_entropy, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) <= 2:
        return {
            "Entropy_lambda": LAMBDA_DEFAULT,
            "Entropy_m": KUBIOS_ENTROPY_M,
            "Entropy_r_factor": KUBIOS_ENTROPY_R_FACTOR,
            "Entropy_SD_ms": np.nan,
            "Entropy_r_ms": np.nan,
            "Entropy_N": len(x),
        }

    sd = np.std(x, ddof=1)
    return {
        "Entropy_lambda": LAMBDA_DEFAULT,
        "Entropy_m": KUBIOS_ENTROPY_M,
        "Entropy_r_factor": KUBIOS_ENTROPY_R_FACTOR,
        "Entropy_SD_ms": sd,
        "Entropy_r_ms": KUBIOS_ENTROPY_R_FACTOR * sd,
        "Entropy_N": len(x),
    }


def sample_entropy_common(rr_entropy, m=KUBIOS_ENTROPY_M, r_factor=KUBIOS_ENTROPY_R_FACTOR, r_reference=None, zero_policy=None, theiler_window=None, radius_mode=None):
    """
    SampEn común para SampEn y MSE1.
    """
    x = np.asarray(rr_entropy, dtype=float)
    x = x[np.isfinite(x)]

    ref = x if r_reference is None else np.asarray(r_reference, dtype=float)
    ref = ref[np.isfinite(ref)]

    if len(x) <= m + 2 or len(ref) <= 2:
        return np.nan

    r = _resolve_entropy_radius(x, ref, r_factor=r_factor, radius_mode=(radius_mode or (st.session_state.get("mse_radius_mode", "fixed_entropy_sd") if "st" in globals() else "fixed_entropy_sd")))

    if zero_policy is None:
        zero_policy = st.session_state.get("mse_zero_policy", "nan") if "st" in globals() else "nan"
    if theiler_window is None:
        theiler_window = st.session_state.get("sampen_theiler_window", 0) if "st" in globals() else 0

    return _sample_entropy_core(x, m=m, r=r, zero_policy=zero_policy, theiler_window=theiler_window)


def coarse_grain_series(x, scale):
    """
    Coarse-graining clásico de MSE.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    scale = int(scale)
    if scale <= 1:
        return x.copy()

    n = len(x) // scale
    if n <= 2:
        return np.array([], dtype=float)

    return x[:n * scale].reshape(n, scale).mean(axis=1)



def _coarse_grain_offset_series(x, scale, offset):
    """
    Coarse-graining con desplazamiento para Composite/RCMSE.
    offset va de 0 a scale-1.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    scale = int(scale)
    offset = int(offset)

    if scale <= 1:
        return x.copy()

    xs = x[offset:]
    n = len(xs) // scale
    if n <= 2:
        return np.array([], dtype=float)

    return xs[:n * scale].reshape(n, scale).mean(axis=1)


def _sample_entropy_counts_from_series(x, m=2, r=None, theiler_window=0):
    """
    Alias explícito para conteos SampEn de una serie ya coarse-grained.
    El radio r ya llega resuelto por el llamador; no debe recalcularse aquí.
    """
    return _sample_entropy_counts(x, m=m, r=r, theiler_window=theiler_window)


def rcmse_common(rr_entropy, scales=KUBIOS_MSE_MAX_SCALE, m=KUBIOS_ENTROPY_M, r_factor=KUBIOS_ENTROPY_R_FACTOR, r_reference=None, theiler_window=0, radius_mode='fixed_entropy_sd'):
    """
    Refined Composite Multiscale Entropy aproximado.

    Para cada escala tau:
    1) construye tau series coarse-grained con offsets 0..tau-1,
    2) suma los conteos B y A de SampEn en todos los offsets,
    3) calcula -ln(sum(A)/sum(B)).

    Esto evita muchos A=0 de MSE clásico y suele comportarse más parecido a
    implementaciones comerciales en escalas altas.
    """
    x = np.asarray(rr_entropy, dtype=float)
    x = x[np.isfinite(x)]

    ref = x if r_reference is None else np.asarray(r_reference, dtype=float)
    ref = ref[np.isfinite(ref)]

    if len(x) <= m + 2 or len(ref) <= 2:
        return {f"MSE{i}": np.nan for i in range(1, int(scales) + 1)}

    out = {}

    for scale in range(1, int(scales) + 1):
        if scale <= 1:
            out[f"MSE{scale}"] = _sample_entropy_core(x, m=m, r=_resolve_entropy_radius(x, ref, r_factor=r_factor, radius_mode=radius_mode), zero_policy="nan", theiler_window=theiler_window)
            continue

        B_total = 0.0
        A_total = 0.0
        valid_offsets = 0

        for offset in range(scale):
            cg = _coarse_grain_offset_series(x, scale, offset)
            if len(cg) <= m + 2:
                continue

            B, A = _sample_entropy_counts_from_series(cg, m=m, r=_resolve_entropy_radius(cg, ref, r_factor=r_factor, radius_mode=radius_mode), theiler_window=theiler_window)
            if np.isfinite(B) and B > 0:
                B_total += B
                if np.isfinite(A) and A > 0:
                    A_total += A
                valid_offsets += 1

        if valid_offsets == 0 or B_total <= 0:
            out[f"MSE{scale}"] = np.nan
        elif A_total <= 0:
            out[f"MSE{scale}"] = np.nan
        else:
            out[f"MSE{scale}"] = -np.log(A_total / B_total)

    return out


def rcmse_diagnostic_rows(rr_entropy, scales=20, m=2, r_factor=0.2, r_reference=None, theiler_window=0, radius_mode=None):
    """
    Diagnóstico RCMSE por escala: suma de B/A en todos los offsets.
    """
    x = np.asarray(rr_entropy, dtype=float)
    x = x[np.isfinite(x)]

    ref = x if r_reference is None else np.asarray(r_reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if radius_mode is None:
        radius_mode = st.session_state.get("mse_radius_mode", "fixed_entropy_sd") if "st" in globals() else "fixed_entropy_sd"

    if len(ref) > 2:
        sd_ref = np.std(ref, ddof=1)
        r = r_factor * sd_ref
    else:
        sd_ref = np.nan
        r = np.nan

    rows = []

    for scale in range(1, int(scales) + 1):
        if not np.isfinite(r) or r <= 0:
            rows.append({
                "Escala": scale, "RCMSE_offsets_validos": 0, "RCMSE_B_total": np.nan,
                "RCMSE_A_total": np.nan, "RCMSE_A/B": np.nan, "RCMSE": np.nan,
                "RCMSE_estado": "r inválido"
            })
            continue

        B_total = 0.0
        A_total = 0.0
        valid_offsets = 0
        n_values = []

        for offset in range(scale):
            cg = _coarse_grain_offset_series(x, scale, offset)
            n_values.append(len(cg))
            if len(cg) <= m + 2:
                continue
            B, A = _sample_entropy_counts_from_series(cg, m=m, r=_resolve_entropy_radius(cg, ref, r_factor=r_factor, radius_mode=radius_mode), theiler_window=theiler_window)
            if np.isfinite(B) and B > 0:
                B_total += B
                if np.isfinite(A) and A > 0:
                    A_total += A
                valid_offsets += 1

        if valid_offsets == 0 or B_total <= 0:
            val = np.nan
            ratio = np.nan
            estado = "No calculado: B total=0"
        elif A_total <= 0:
            val = np.nan
            ratio = 0.0
            estado = "No calculado: A total=0"
        else:
            ratio = A_total / B_total
            val = -np.log(ratio)
            estado = "Calculado"

        rows.append({
            "Escala": scale,
            "RCMSE_offsets_validos": valid_offsets,
            "RCMSE_N_min": min(n_values) if n_values else np.nan,
            "RCMSE_N_max": max(n_values) if n_values else np.nan,
            "RCMSE_B_total": B_total,
            "RCMSE_A_total": A_total,
            "RCMSE_A/B": ratio,
            "RCMSE": val,
            "RCMSE_estado": estado,
        })

    return pd.DataFrame(rows)


def mse_common(rr_entropy, scales=KUBIOS_MSE_MAX_SCALE, m=KUBIOS_ENTROPY_M, r_factor=KUBIOS_ENTROPY_R_FACTOR, r_reference=None, zero_policy="nan", theiler_window=None, radius_mode=None):
    """
    MSE v10.6.

    Modos:
    - zero_policy="nan": SampEn clásica, A=0 -> NaN.
    - zero_policy="half_count": si A=0, usa A=0.5.
    - zero_policy="one_count": si A=0, usa A=1.

    Esto permite comparar explícitamente la app con Kubios cuando Kubios devuelve
    valores MSE en escalas donde SampEn clásica tendría A=0.
    """
    if theiler_window is None:
        theiler_window = st.session_state.get("sampen_theiler_window", 0) if "st" in globals() else 0
    if radius_mode is None:
        radius_mode = st.session_state.get("mse_radius_mode", "fixed_entropy_sd") if "st" in globals() else "fixed_entropy_sd"

    if zero_policy == "rcmse":
        return rcmse_common(
            rr_entropy,
            scales=scales,
            m=m,
            r_factor=r_factor,
            r_reference=r_reference,
            theiler_window=theiler_window,
            radius_mode=radius_mode
        )

    x = np.asarray(rr_entropy, dtype=float)
    x = x[np.isfinite(x)]

    ref = x if r_reference is None else np.asarray(r_reference, dtype=float)
    ref = ref[np.isfinite(ref)]

    if len(x) <= m + 2 or len(ref) <= 2:
        return {f"MSE{i}": np.nan for i in range(1, int(scales) + 1)}

    out = {}

    for scale in range(1, int(scales) + 1):
        cg = coarse_grain_series(x, scale)

        if len(cg) <= m + 2:
            out[f"MSE{scale}"] = np.nan
        else:
            r_scale = _resolve_entropy_radius(cg, ref, r_factor=r_factor, radius_mode=radius_mode)
            out[f"MSE{scale}"] = _sample_entropy_core(cg, m=m, r=r_scale, zero_policy=zero_policy, theiler_window=theiler_window)

    r1 = _resolve_entropy_radius(x, ref, r_factor=r_factor, radius_mode=radius_mode)
    out["MSE1"] = _sample_entropy_core(x, m=m, r=r1, zero_policy=zero_policy, theiler_window=theiler_window)
    return out


def sample_entropy_fast(x, m=2, r_ratio=0.2, max_n=None):
    """
    Compatibilidad con versiones antiguas.
    """
    return sample_entropy_common(x, m=m, r_factor=r_ratio, r_reference=x)


def coarse_grain(x, scale):
    """
    Compatibilidad con versiones antiguas.
    """
    return coarse_grain_series(x, scale)


def mse_metrics(rr, scales=20, max_scale=None, m=2, r=0.2, zero_policy=None, theiler_window=None, radius_mode=None):
    """
    Wrapper compatible con llamadas antiguas.

    v10.8 corregida:
    si no se pasa zero_policy explícitamente, usa el modo seleccionado
    en la barra lateral para que el cambio tenga efecto real.
    """
    if max_scale is not None:
        scales = max_scale

    if zero_policy is None:
        zero_policy = st.session_state.get("mse_zero_policy", "nan") if "st" in globals() else "nan"
    if theiler_window is None:
        theiler_window = st.session_state.get("sampen_theiler_window", 0) if "st" in globals() else 0

    return mse_common(rr, scales=scales, m=m, r_factor=r, r_reference=rr, zero_policy=zero_policy, theiler_window=theiler_window, radius_mode=(radius_mode or (st.session_state.get("mse_radius_mode", "fixed_entropy_sd") if "st" in globals() else "fixed_entropy_sd")))


def enforce_entropy_dataframe_consistency(df):
    """
    Garantía final en tablas:
    si existen SampEn y MSE1, MSE1 se iguala a SampEn.
    """
    try:
        if isinstance(df, pd.DataFrame):
            if "SampEn" in df.columns and "MSE1" in df.columns:
                df["MSE1"] = df["SampEn"]
    except Exception:
        pass

    return df


def enforce_entropy_consistency(metrics, rr_entropy, mse_zero_policy=None):
    """
    Fuerza coherencia interna:
    SampEn y MSE1 se calculan con la misma entrada y misma tolerancia.
    """
    try:
        if mse_zero_policy is None:
            mse_zero_policy = st.session_state.get("mse_zero_policy", "nan") if "st" in globals() else "nan"

        ent = sample_entropy_common(
            rr_entropy,
            m=KUBIOS_ENTROPY_M,
            r_factor=KUBIOS_ENTROPY_R_FACTOR,
            r_reference=rr_entropy,
            zero_policy=mse_zero_policy,
            theiler_window=theiler_window
        )
        mse_vals = mse_common(
            rr_entropy,
            scales=KUBIOS_MSE_MAX_SCALE,
            m=KUBIOS_ENTROPY_M,
            r_factor=KUBIOS_ENTROPY_R_FACTOR,
            r_reference=rr_entropy,
            zero_policy=mse_zero_policy
        )
        metrics["SampEn"] = ent
        metrics.update(mse_vals)
    except Exception:
        pass

    return metrics



# ============================================================
# DIAGNÓSTICO SAMPEN / MSE PARA COMPARAR CON KUBIOS
# ============================================================

def sample_entropy_diagnostic_rows(rr_entropy, scales=20, m=2, r_factor=0.2, r_reference=None, theiler_window=0, radius_mode=None):
    """
    Tabla diagnóstica para SampEn/MSE.

    Incluye los tres modos MSE:
    - clásico: A=0 -> NaN
    - pseudoconteo 0.5
    - pseudoconteo 1.0
    """
    x = np.asarray(rr_entropy, dtype=float)
    x = x[np.isfinite(x)]

    ref = x if r_reference is None else np.asarray(r_reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if radius_mode is None:
        radius_mode = st.session_state.get("mse_radius_mode", "fixed_entropy_sd") if "st" in globals() else "fixed_entropy_sd"

    if len(ref) > 2:
        sd_ref = np.std(ref, ddof=1)
        r = r_factor * sd_ref
    else:
        sd_ref = np.nan
        r = np.nan

    rows = []

    for scale in range(1, int(scales) + 1):
        cg = coarse_grain_series(x, scale)

        if len(cg) <= m + 2 or not np.isfinite(r) or r <= 0:
            rows.append({
                "Escala": scale,
                "N": len(cg),
                "SD_escala_ms": np.std(cg, ddof=1) if len(cg) > 2 else np.nan,
                "SD_referencia_ms": sd_ref,
                "r_ms": r,
                "Theiler": theiler_window,
                "B_matches_m": np.nan,
                "A_matches_m1": np.nan,
                "A/B": np.nan,
                "MSE_clasico": np.nan,
                "MSE_A0_05": np.nan,
                "MSE_A0_1": np.nan,
                "Estado": "No calculado: pocos puntos o r inválido",
            })
            continue

        B, A = _sample_entropy_counts(cg, m=m, r=_resolve_entropy_radius(cg, ref, r_factor=r_factor, radius_mode=radius_mode), theiler_window=theiler_window)

        if not np.isfinite(B) or B <= 0:
            val_classic = np.nan
            val_half = np.nan
            val_one = np.nan
            ratio = np.nan
            estado = "No calculado: B=0"
        elif not np.isfinite(A) or A <= 0:
            val_classic = np.nan
            val_half = -np.log(0.5 / B)
            val_one = -np.log(1.0 / B)
            ratio = 0.0
            estado = "A=0: clásico no calcula; pseudoconteos disponibles"
        else:
            ratio = A / B
            val_classic = -np.log(ratio)
            val_half = val_classic
            val_one = val_classic
            estado = "Calculado"

        rows.append({
            "Escala": scale,
            "N": len(cg),
            "SD_escala_ms": np.std(cg, ddof=1) if len(cg) > 2 else np.nan,
            "SD_referencia_ms": sd_ref,
            "r_ms": r,
            "Theiler": theiler_window,
            "B_matches_m": B,
            "A_matches_m1": A,
            "A/B": ratio,
            "MSE_clasico": val_classic,
            "MSE_A0_05": val_half,
            "MSE_A0_1": val_one,
            "Estado": estado,
        })

    return pd.DataFrame(rows)


def entropy_kubios_diagnostic_table(rr):
    """
    Construye la tabla diagnóstica completa desde RR en segundos.
    Incluye MSE clásico, pseudoconteos y RCMSE/Composite.
    """
    rr_ms = np.asarray(rr, dtype=float) * 1000.0
    rr_entropy = smoothness_priors_detrend(rr_ms, LAMBDA_DEFAULT)
    theiler_window = st.session_state.get("sampen_theiler_window", 0) if "st" in globals() else 0

    diag = sample_entropy_diagnostic_rows(
        rr_entropy,
        scales=KUBIOS_MSE_MAX_SCALE,
        m=KUBIOS_ENTROPY_M,
        r_factor=KUBIOS_ENTROPY_R_FACTOR,
        r_reference=rr_entropy,
        theiler_window=theiler_window,
        radius_mode=st.session_state.get("mse_radius_mode", "fixed_entropy_sd") if "st" in globals() else "fixed_entropy_sd"
    )

    diag_rc = rcmse_diagnostic_rows(
        rr_entropy,
        scales=KUBIOS_MSE_MAX_SCALE,
        m=KUBIOS_ENTROPY_M,
        r_factor=KUBIOS_ENTROPY_R_FACTOR,
        r_reference=rr_entropy,
        theiler_window=theiler_window,
        radius_mode=st.session_state.get("mse_radius_mode", "fixed_entropy_sd") if "st" in globals() else "fixed_entropy_sd"
    )

    try:
        diag = diag.merge(diag_rc, on="Escala", how="left")
    except Exception:
        pass

    diag.insert(0, "Lambda", LAMBDA_DEFAULT)
    diag.insert(1, "m", KUBIOS_ENTROPY_M)
    diag.insert(2, "r_factor", KUBIOS_ENTROPY_R_FACTOR)

    return diag


def entropy_diagnostic_figure(diag_df):
    """
    Figura diagnóstica MSE con tres modos:
    clásico, pseudoconteo 0.5 y pseudoconteo 1.0.
    """
    fig = go.Figure()

    if diag_df is None or diag_df.empty:
        fig.update_layout(title="Diagnóstico MSE: sin datos")
        return fig

    if "MSE_clasico" in diag_df.columns:
        fig.add_trace(go.Bar(
            x=diag_df["Escala"],
            y=diag_df["MSE_clasico"],
            name="Clásico A=0→NaN",
            opacity=0.55,
            hovertemplate="Escala %{x}<br>Clásico=%{y:.4f}<extra></extra>",
        ))

    if "MSE_A0_05" in diag_df.columns:
        fig.add_trace(go.Scatter(
            x=diag_df["Escala"],
            y=diag_df["MSE_A0_05"],
            mode="lines+markers",
            name="A=0→0.5",
            line=dict(width=3),
            hovertemplate="Escala %{x}<br>A0=0.5: %{y:.4f}<extra></extra>",
        ))

    if "MSE_A0_1" in diag_df.columns:
        fig.add_trace(go.Scatter(
            x=diag_df["Escala"],
            y=diag_df["MSE_A0_1"],
            mode="lines+markers",
            name="A=0→1.0",
            line=dict(width=3, dash="dash"),
            hovertemplate="Escala %{x}<br>A0=1.0: %{y:.4f}<extra></extra>",
        ))


    if "RCMSE" in diag_df.columns:
        fig.add_trace(go.Scatter(
            x=diag_df["Escala"],
            y=diag_df["RCMSE"],
            mode="lines+markers",
            name="RCMSE / Composite",
            line=dict(width=4),
            hovertemplate="Escala %{x}<br>RCMSE=%{y:.4f}<extra></extra>",
        ))

    bad = diag_df[diag_df["Estado"] != "Calculado"]
    if not bad.empty:
        fig.add_trace(go.Scatter(
            x=bad["Escala"],
            y=[0] * len(bad),
            mode="markers+text",
            name="A/B insuficiente",
            text=bad["Estado"],
            textposition="top center",
            marker=dict(size=10, symbol="x"),
            hovertemplate="Escala %{x}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Diagnóstico SampEn / MSE: clásico, pseudoconteos y RCMSE",
        xaxis_title="Escala MSE",
        yaxis_title="SampEn / MSE",
        height=560,
        bargap=0.18,
        hovermode="closest",
    )
    fig.update_xaxes(dtick=1)

    return fig


def domain_reference_table():
    """
    Definiciones y valores orientativos de dominios normalizados a Basal = 100%.
    """
    return pd.DataFrame([
        {
            "Dominio": "Amplitud",
            "Incluye": "SDNN, SD2, Total Power",
            "Qué representa": "Magnitud global de las oscilaciones cardiovasculares.",
            "Referencia": "Basal = 100%",
            "Interpretación": "<80% disminución clara; 80-120% cambio moderado/estable; >120% aumento respecto a basal."
        },
        {
            "Dominio": "Vagal",
            "Incluye": "RMSSD, SD1, HF, pNN50",
            "Qué representa": "Regulación rápida parasimpática/vagal.",
            "Referencia": "Basal = 100%",
            "Interpretación": "<80% reducción vagal; 80-120% mantenimiento; >120% aumento de modulación vagal."
        },
        {
            "Dominio": "Complejidad",
            "Incluye": "DFA α1, DFA α2, ApEn, SampEn, D2",
            "Qué representa": "Riqueza, irregularidad y capacidad de adaptación dinámica.",
            "Referencia": "Basal = 100%",
            "Interpretación": "<80% menor complejidad; 80-120% estable; >120% mayor complejidad/adaptabilidad."
        },
        {
            "Dominio": "MSE 1-20",
            "Incluye": "Entropía multiescala MSE1-MSE20",
            "Qué representa": "Complejidad en escalas temporales cortas, medias y largas.",
            "Referencia": "Basal = 100%",
            "Interpretación": "<80% pérdida de complejidad multiescala; >120% aumento de complejidad multiescala."
        },
        {
            "Dominio": "Recurrencia",
            "Incluye": "REC, DET, Lmean, Lmax, ShanEn",
            "Qué representa": "Repetición, persistencia y organización temporal de patrones.",
            "Referencia": "Basal = 100%",
            "Interpretación": "Aumentos pueden indicar mayor repetición/regularidad; descensos pueden indicar menor recurrencia o menor estabilidad de patrones."
        },
    ])


def domain_values(metrics_df, method="median"):
    """
    Dominios normalizados a Basal = 100%.
    Sólo usa variables numéricas.
    """
    if metrics_df is None or metrics_df.empty or "Basal" not in metrics_df.index:
        return pd.DataFrame()

    base = metrics_df.loc["Basal"]
    rows = []

    for ph in [p for p in PHASES if p in metrics_df.index]:
        row = {"Fase": ph}

        for dom, vars_ in DOMAIN_GROUPS.items():
            vals = []

            for v in vars_:
                if v in metrics_df.columns and v in base.index:
                    b = pd.to_numeric(base[v], errors="coerce")
                    x = pd.to_numeric(metrics_df.loc[ph, v], errors="coerce")

                    if pd.notna(b) and pd.notna(x) and float(b) != 0:
                        vals.append(100.0 * float(x) / float(b))

            if vals:
                row[dom] = float(np.nanmedian(vals) if method == "median" else np.nanmean(vals))
            else:
                row[dom] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("Fase") if rows else pd.DataFrame()


def domains_figure(metrics_df, method="median", title="Dominios Amplitud / Vagal / Complejidad / Recurrencia"):
    """
    Dominios normalizados como columnas verticales + líneas de tendencia suavizadas.
    Basal = 100%.
    """
    dom = domain_values(metrics_df, method=method)
    fig = go.Figure()

    if dom.empty:
        fig.update_layout(title="No hay dominios disponibles. Se necesita Basal válido.")
        return fig

    phases = [p for p in PHASES if p in dom.index]
    x_base = np.arange(len(phases), dtype=float)
    cols = list(dom.columns)
    n = max(1, len(cols))
    bar_width = min(0.72 / n, 0.16)

    for i, col in enumerate(cols):
        color = _export_color_for(i)
        y = [dom.loc[ph, col] if ph in dom.index else np.nan for ph in phases]
        y = [float(v) if pd.notna(v) else np.nan for v in y]
        offset = (i - (n - 1) / 2) * bar_width

        fig.add_trace(go.Bar(
            x=x_base + offset,
            y=y,
            width=bar_width,
            name=f"{col} · columnas",
            marker=dict(color=color),
            opacity=0.52,
            customdata=phases,
            hovertemplate=f"{col}<br>Fase: %{{customdata}}<br>Índice: %{{y:.1f}}%<extra></extra>",
        ))

        xs, ys = _smooth_line_xy(y)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"{col} · tendencia",
            line=dict(width=3.5, color=color),
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=x_base,
            y=y,
            mode="markers+text",
            name=f"{col} · puntos",
            marker=dict(size=8, color=color),
            text=[f"{v:.1f}" if pd.notna(v) else "" for v in y],
            textposition="top center",
            showlegend=False,
            customdata=phases,
            hovertemplate=f"{col}<br>Fase: %{{customdata}}<br>Índice: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dash", annotation_text="Basal = 100%")

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=phases,
        title_text="Fase",
    )

    fig.update_layout(
        title=title + " · columnas + tendencia suavizada",
        height=680,
        xaxis_title="Fase",
        yaxis_title="Índice normalizado (%)",
        hovermode="closest",
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        legend_title_text="Dominio",
        margin=dict(l=60, r=40, t=80, b=80),
    )
    return fig


def mse_figure(metrics_df, title="MSE 1-20"):
    """
    MSE 1-20: columnas agrupadas por fase + líneas de tendencia suavizadas por escala.
    """
    fig = go.Figure()
    mse_cols = [c for c in MSE_COLUMNS if c in metrics_df.columns]

    if metrics_df is None or metrics_df.empty or not mse_cols:
        fig.update_layout(title="No hay MSE disponible")
        return fig

    phases = [p for p in PHASES if p in metrics_df.index]
    if not phases:
        fig.update_layout(title="No hay fases válidas para MSE")
        return fig

    x_base = np.arange(len(phases), dtype=float)
    n = max(1, len(mse_cols))
    bar_width = min(0.78 / n, 0.035)

    for i, col in enumerate(mse_cols):
        scale = col.replace("MSE", "")
        color = _export_color_for(i)
        y = [metrics_df.loc[ph, col] if ph in metrics_df.index else np.nan for ph in phases]
        y = [float(v) if pd.notna(v) else np.nan for v in y]
        offset = (i - (n - 1) / 2) * bar_width

        fig.add_trace(go.Bar(
            x=x_base + offset,
            y=y,
            width=bar_width,
            name=f"MSE {scale}",
            marker=dict(color=color),
            opacity=0.38,
            customdata=phases,
            hovertemplate=f"Escala MSE {scale}<br>Fase: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
        ))

        xs, ys = _smooth_line_xy(y)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"MSE {scale} tendencia",
            line=dict(width=2.2, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=x_base,
            y=y,
            mode="markers",
            name=f"MSE {scale} puntos",
            marker=dict(size=5, color=color),
            showlegend=False,
            customdata=phases,
            hovertemplate=f"Escala MSE {scale}<br>Fase: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=phases,
        title_text="Fase",
    )

    fig.update_layout(
        title=title + " · columnas + tendencia suavizada",
        height=740,
        barmode="group",
        bargap=0.20,
        bargroupgap=0.01,
        xaxis_title="Fase",
        yaxis_title="Valor / Sample entropy",
        hovermode="closest",
        legend_title_text="Escala MSE",
        margin=dict(l=60, r=40, t=80, b=80),
    )
    return fig



def mse_compare_figure(long_df, phases, scales=None):
    """
    Comparativa MSE: columnas verticales + líneas suavizadas por registro/fase.
    """
    if scales is None:
        scales = list(range(1, 21))

    cols = [f"MSE{s}" for s in scales if f"MSE{s}" in long_df.columns]
    fig = go.Figure()

    if long_df.empty or not cols:
        fig.update_layout(title="No hay MSE disponible")
        return fig

    records_order = sorted(
        list(long_df["Registro"].dropna().unique()),
        key=lambda r: (extract_datetime_from_name(r), r)
    )

    x_base = np.arange(len(cols), dtype=float)

    trace_i = 0
    for rec_i, rec in enumerate(records_order):
        drec = long_df[long_df["Registro"] == rec]
        for ph_i, ph in enumerate(phases):
            dph = drec[drec["Fase"] == ph]
            if dph.empty:
                continue

            y = [pd.to_numeric(dph.iloc[0][c], errors="coerce") for c in cols]
            y = [float(v) if pd.notna(v) else np.nan for v in y]
            color = _export_color_for(trace_i)
            offset = (trace_i % max(1, len(records_order) * len(phases)) - ((len(records_order) * len(phases)) - 1) / 2) * min(0.70 / max(1, len(records_order) * len(phases)), 0.025)

            fig.add_trace(go.Bar(
                x=x_base + offset,
                y=y,
                width=min(0.70 / max(1, len(records_order) * len(phases)), 0.025),
                name=f"{_short_record_label(rec, 24)} · {ph}",
                marker=dict(color=color),
                opacity=0.38,
                hovertemplate=f"{_short_record_label(rec, 32)}<br>{ph}<br>Escala: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
                customdata=[c.replace("MSE", "") for c in cols],
            ))

            xs, ys = _smooth_line_xy(y)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{_short_record_label(rec, 24)} · {ph} tendencia",
                line=dict(width=3, color=color),
                hoverinfo="skip",
                showlegend=False,
            ))

            fig.add_trace(go.Scatter(
                x=x_base,
                y=y,
                mode="markers",
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"{_short_record_label(rec, 32)}<br>{ph}<br>Escala: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
                customdata=[c.replace("MSE", "") for c in cols],
            ))

            trace_i += 1

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=[c.replace("MSE", "") for c in cols],
        title_text="Escala MSE",
        dtick=1,
    )

    fig.update_layout(
        title="Comparativa MSE 1-20 · columnas + líneas suavizadas",
        height=720,
        xaxis_title="Escala MSE",
        yaxis_title="Valor / Sample entropy",
        hovermode="closest",
        barmode="group",
        bargap=0.18,
        bargroupgap=0.01,
        legend_title_text="Registro · fase",
        margin=dict(l=60, r=40, t=80, b=80),
    )
    return fig




def hvg_wide_table(long_df):
    """
    Tabla ancha HVG comparativa incluyendo tipo de grafo, compactación y scores.
    """
    if long_df is None or long_df.empty:
        return pd.DataFrame()

    hvg_cols = [
        "HVG_graph_type",
        "HVG_topology_state",
        "HVG_compactness_index",
        "HVG_graph_score_scale_free",
        "HVG_graph_score_small_world",
        "HVG_graph_score_chain",
        "HVG_nodes",
        "HVG_edges",
        "HVG_degree_mean",
        "HVG_degree_max",
        "HVG_hubs_p90",
        "HVG_clustering",
        "HVG_lambda",
        "HVG_path_length",
        "HVG_diameter",
    ]
    cols = ["Registro", "Fase"] + [c for c in hvg_cols if c in long_df.columns]
    if len(cols) <= 2:
        return pd.DataFrame()
    return long_df[cols].copy()



# ============================================================
# MÉTODOS AVANZADOS FRECUENCIALES Y NO LINEALES v11.2
# ============================================================

def lomb_psd_metrics(rr):
    """
    Lomb-Scargle sobre RRi no equiespaciados.
    Útil como alternativa a Welch cuando se quiere evitar interpolar primero.
    """
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr)]
    if len(rr) < 32:
        return {k: np.nan for k in ["VLF_LS", "LF_LS", "HF_LS", "TOTAL_LS", "LF_HF_LS"]}

    t = np.cumsum(rr)
    t = t - t[0]
    x = rr * 1000.0
    x = x - np.mean(x)

    f = np.linspace(0.0033, 0.40, 2048)
    try:
        pxx = signal.lombscargle(t, x, 2 * np.pi * f, normalize=True)
        # Reescalado aproximado a ms²/Hz para hacerlo comparable en forma, no idéntico a Welch.
        pxx = pxx * np.var(x, ddof=1) / np.trapezoid(pxx, f) if np.trapezoid(pxx, f) > 0 else pxx
    except Exception:
        return {k: np.nan for k in ["VLF_LS", "LF_LS", "HF_LS", "TOTAL_LS", "LF_HF_LS"]}

    def bp(lo, hi):
        mask = (f >= lo) & (f < hi)
        return np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else np.nan

    vlf, lf, hf = bp(0.0033, 0.04), bp(0.04, 0.15), bp(0.15, 0.40)
    total = np.nansum([vlf, lf, hf])
    return {
        "VLF_LS": vlf, "LF_LS": lf, "HF_LS": hf, "TOTAL_LS": total,
        "LF_HF_LS": lf / hf if pd.notna(hf) and hf > 0 else np.nan
    }


def ar_psd_metrics(rr, order=16):
    """
    PSD autorregresiva por Yule-Walker.
    Da una estimación espectral alternativa con buena resolución en ventanas cortas.
    """
    try:
        ti, xi = interpolate_rr(rr, fs=FS_INTERP, apply_lambda=True, lam=LAMBDA_DEFAULT)
        x = xi * 1000.0
        x = x - np.mean(x)
        n = len(x)
        if n < max(64, order * 4):
            return {k: np.nan for k in ["VLF_AR", "LF_AR", "HF_AR", "TOTAL_AR", "LF_HF_AR"]}

        # autocorrelación sesgada
        r = np.correlate(x, x, mode="full")[n-1:n+order] / n
        R = np.array([[r[abs(i-j)] for j in range(order)] for i in range(order)])
        rhs = r[1:order+1]
        a = np.linalg.solve(R + np.eye(order)*1e-9, rhs)
        noise_var = max(r[0] - np.dot(a, rhs), 1e-12)

        f = np.linspace(0.0033, 0.40, 2048)
        z = np.exp(-2j * np.pi * f[:, None] * np.arange(1, order+1) / FS_INTERP)
        den = np.abs(1 - np.dot(z, a)) ** 2
        pxx = noise_var / den / FS_INTERP

        def bp(lo, hi):
            mask = (f >= lo) & (f < hi)
            return np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else np.nan

        vlf, lf, hf = bp(0.0033, 0.04), bp(0.04, 0.15), bp(0.15, 0.40)
        total = np.nansum([vlf, lf, hf])
        return {
            "VLF_AR": vlf, "LF_AR": lf, "HF_AR": hf, "TOTAL_AR": total,
            "LF_HF_AR": lf / hf if pd.notna(hf) and hf > 0 else np.nan
        }
    except Exception:
        return {k: np.nan for k in ["VLF_AR", "LF_AR", "HF_AR", "TOTAL_AR", "LF_HF_AR"]}



def _entropy_from_probs(probs):
    p = np.asarray(probs, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return np.nan
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))


def _episodes_from_labels(labels, times, target_label):
    labels = np.asarray(labels)
    times = np.asarray(times, dtype=float)
    if len(labels) == 0 or len(times) == 0:
        return 0, np.nan, np.nan

    if len(times) > 1:
        dt = float(np.nanmedian(np.diff(times)))
    else:
        dt = np.nan

    durations = []
    i = 0
    while i < len(labels):
        if labels[i] == target_label:
            j = i
            while j + 1 < len(labels) and labels[j + 1] == target_label:
                j += 1
            if np.isfinite(dt):
                durations.append((j - i + 1) * dt)
            i = j + 1
        else:
            i += 1

    if not durations:
        return 0, np.nan, np.nan
    return len(durations), float(np.mean(durations)), float(np.max(durations))


def wavelet_temporal_metrics(rr):
    """
    Métricas wavelet/STFT por bandas VLF/LF/HF.

    Calcula:
    - potencia media y SD temporal por banda,
    - porcentaje de tiempo en que cada banda domina,
    - episodios de dominancia por banda,
    - transiciones entre bandas dominantes,
    - entropía de dominancia por bandas,
    - entropía global del escalograma.
    """
    keys = [
        "VLF_WAV_MEAN","LF_WAV_MEAN","HF_WAV_MEAN",
        "VLF_WAV_SD","LF_WAV_SD","HF_WAV_SD",
        "VLF_DOM_PCT","LF_DOM_PCT","HF_DOM_PCT",
        "VLF_EPISODES_N","LF_EPISODES_N","HF_EPISODES_N",
        "VLF_EPISODE_MEAN_S","LF_EPISODE_MEAN_S","HF_EPISODE_MEAN_S",
        "VLF_EPISODE_MAX_S","LF_EPISODE_MAX_S","HF_EPISODE_MAX_S",
        "WAV_TRANSITIONS_N","WAV_TRANSITIONS_PER_MIN",
        "WAV_ENTROPY_BANDS","WAV_ENTROPY_GLOBAL",
        "VLF_WAV_MEAN","LF_WAV_MEAN","HF_WAV_MEAN","VLF_WAV_SD","LF_WAV_SD","HF_WAV_SD","VLF_DOM_PCT","LF_DOM_PCT","HF_DOM_PCT","VLF_EPISODES_N","LF_EPISODES_N","HF_EPISODES_N","VLF_EPISODE_MEAN_S","LF_EPISODE_MEAN_S","HF_EPISODE_MEAN_S","VLF_EPISODE_MAX_S","LF_EPISODE_MAX_S","HF_EPISODE_MAX_S","WAV_TRANSITIONS_N","WAV_TRANSITIONS_PER_MIN","WAV_ENTROPY_BANDS","WAV_ENTROPY_GLOBAL","LF_WAV","HF_WAV","LF_HF_WAV",
    ]
    out = {k: np.nan for k in keys}
    try:
        ti, xi = interpolate_rr(rr, fs=FS_INTERP, apply_lambda=True, lam=LAMBDA_DEFAULT)
        x = xi * 1000.0
        x = x - np.nanmean(x)
        if len(x) < 64:
            return out

        nperseg = min(max(64, int(64 * FS_INTERP)), len(x))
        noverlap = int(0.80 * nperseg) if nperseg > 10 else 0
        f, tt, Zxx = signal.stft(
            x,
            fs=FS_INTERP,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
            padded=False,
        )
        p = np.abs(Zxx) ** 2

        def band_ts(lo, hi):
            mask = (f >= lo) & (f < hi)
            if not np.any(mask):
                return np.full(len(tt), np.nan)
            return np.trapezoid(p[mask, :], f[mask], axis=0)

        vlf_ts = band_ts(0.0033, 0.04)
        lf_ts = band_ts(0.04, 0.15)
        hf_ts = band_ts(0.15, 0.40)
        bands = np.vstack([vlf_ts, lf_ts, hf_ts])
        valid = np.all(np.isfinite(bands), axis=0)

        if not np.any(valid):
            return out

        vlf_ts, lf_ts, hf_ts = vlf_ts[valid], lf_ts[valid], hf_ts[valid]
        tt_valid = tt[valid]
        bands_valid = np.vstack([vlf_ts, lf_ts, hf_ts])

        # v11.8:
        # Para calcular dominancia no comparamos potencia absoluta, porque una banda
        # puede dominar sólo por escala/amplitud media. Primero normalizamos cada
        # banda por su propia media temporal:
        # VLF_n(t)=VLF(t)/mean(VLF), LF_n(t)=LF(t)/mean(LF), HF_n(t)=HF(t)/mean(HF)
        band_means = np.nanmean(bands_valid, axis=1)
        band_means = np.where(np.isfinite(band_means) & (band_means > 0), band_means, np.nan)
        bands_norm = bands_valid / band_means[:, None]

        # Si alguna banda no tiene media válida, evitamos que domine artificialmente.
        bands_norm = np.where(np.isfinite(bands_norm), bands_norm, -np.inf)
        labels = np.argmax(bands_norm, axis=0)  # 0 VLF, 1 LF, 2 HF sobre potencia normalizada

        for name, arr in [("VLF", vlf_ts), ("LF", lf_ts), ("HF", hf_ts)]:
            out[f"{name}_WAV_MEAN"] = float(np.nanmean(arr))
            out[f"{name}_WAV_SD"] = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0

        total_points = len(labels)
        for idx, name in enumerate(["VLF", "LF", "HF"]):
            out[f"{name}_DOM_PCT"] = float(100.0 * np.sum(labels == idx) / total_points)
            n_ep, mean_s, max_s = _episodes_from_labels(labels, tt_valid, idx)
            out[f"{name}_EPISODES_N"] = float(n_ep)
            out[f"{name}_EPISODE_MEAN_S"] = mean_s
            out[f"{name}_EPISODE_MAX_S"] = max_s

        transitions = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
        duration_min = (tt_valid[-1] - tt_valid[0]) / 60.0 if len(tt_valid) > 1 else np.nan
        out["WAV_TRANSITIONS_N"] = float(transitions)
        out["WAV_TRANSITIONS_PER_MIN"] = float(transitions / duration_min) if np.isfinite(duration_min) and duration_min > 0 else np.nan

        dom_probs = [np.mean(labels == i) for i in range(3)]
        out["WAV_ENTROPY_BANDS"] = _entropy_from_probs(dom_probs) / np.log(3)

        all_power = p[(f >= 0.0033) & (f <= 0.40), :]
        flat = all_power[np.isfinite(all_power) & (all_power > 0)]
        if len(flat) > 0:
            prob = flat / np.sum(flat)
            out["WAV_ENTROPY_GLOBAL"] = float(-np.sum(prob * np.log(prob)) / np.log(len(prob))) if len(prob) > 1 else 0.0

        out["LF_WAV"] = out["LF_WAV_MEAN"]
        out["HF_WAV"] = out["HF_WAV_MEAN"]
        out["LF_HF_WAV"] = out["LF_WAV_MEAN"] / out["HF_WAV_MEAN"] if out["HF_WAV_MEAN"] > 0 else np.nan

        return out
    except Exception:
        return out


def wavelet_band_metrics(rr):
    """
    v11.7: devuelve métricas wavelet/STFT completas por banda VLF/LF/HF.
    Mantiene también LF_WAV/HF_WAV/LF_HF_WAV por compatibilidad.
    """
    return wavelet_temporal_metrics(rr)


def hurst_rs(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 64:
        return np.nan
    sizes = np.unique(np.floor(np.logspace(np.log10(8), np.log10(max(9, n//4)), 12)).astype(int))
    rs, ss = [], []
    for s in sizes:
        vals = []
        for i in range(n // s):
            seg = x[i*s:(i+1)*s]
            y = np.cumsum(seg - np.mean(seg))
            R = np.max(y) - np.min(y)
            S = np.std(seg, ddof=1)
            if S > 0:
                vals.append(R/S)
        if vals:
            rs.append(np.mean(vals)); ss.append(s)
    if len(rs) < 3:
        return np.nan
    return float(np.polyfit(np.log(ss), np.log(rs), 1)[0])


def katz_fd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0]))
    n = len(x)
    if L <= 0 or d <= 0:
        return np.nan
    return float(np.log10(n) / (np.log10(d / L) + np.log10(n)))


def petrosian_fd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    diff = np.diff(x)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)
    n = len(x)
    if N_delta <= 0:
        return np.nan
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta))))


def dispersion_entropy(x, m=3, c=6, tau=1):
    """
    Dispersion Entropy básica.
    Más rápida y estable que SampEn en ventanas cortas.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < (m - 1) * tau + 5:
        return np.nan
    sd = np.std(x)
    if sd <= 0:
        return np.nan

    # Normal CDF aproximada con erf
    from math import erf, sqrt
    z = (x - np.mean(x)) / (sd + 1e-12)
    y = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in z])
    cls = np.clip(np.floor(c * y).astype(int), 0, c - 1)

    patterns = []
    for i in range(n - (m - 1) * tau):
        pat = tuple(cls[i + k * tau] for k in range(m))
        patterns.append(pat)
    if not patterns:
        return np.nan
    _, counts = np.unique(patterns, return_counts=True, axis=0)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def mde_metrics(x, max_scale=20, m=3, c=6):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {}
    for s in range(1, max_scale + 1):
        cg = coarse_grain_series(x, s)
        out[f"MDE{s}"] = dispersion_entropy(cg, m=m, c=c, tau=1) if len(cg) > (m+2) else np.nan
    return out



def _embed_time_series(x, emb_dim=6, tau=1):
    """
    Reconstrucción del espacio de fases por delay embedding.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x) - (emb_dim - 1) * tau
    if n <= 1:
        return np.empty((0, emb_dim))
    return np.column_stack([x[i * tau:i * tau + n] for i in range(emb_dim)])


def lyapunov_rosenstein(
    x,
    emb_dim=6,
    tau=1,
    theiler=20,
    max_t=30,
    fit_start=1,
    fit_end=10,
):
    """
    Largest Lyapunov Exponent aproximado con algoritmo de Rosenstein.

    Pasos:
    1) reconstruye el atractor con delay embedding,
    2) para cada punto busca el vecino más cercano excluyendo una ventana temporal Theiler,
    3) calcula la divergencia media log(d(k)) a lo largo de k,
    4) estima la pendiente lineal de log(d(k)) entre fit_start y fit_end.

    Devuelve la pendiente en unidades de "por latido" si x es RRi por latidos.
    En HRV de 5 min debe interpretarse de forma relativa/longitudinal.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 80:
        return np.nan

    # Normalizar para que la escala absoluta no domine las distancias.
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    xz = (x - np.mean(x)) / sd

    Y = _embed_time_series(xz, emb_dim=emb_dim, tau=tau)
    n = len(Y)
    if n < max(30, theiler + max_t + 5):
        return np.nan

    max_t = int(min(max_t, n // 3))
    fit_end = int(min(fit_end, max_t - 1))
    fit_start = int(max(1, min(fit_start, fit_end - 1)))

    # Matriz de distancias. Para ventanas de 5 min es manejable.
    try:
        D = squareform(pdist(Y, metric="euclidean"))
    except Exception:
        return np.nan

    nn = np.full(n, -1, dtype=int)

    for i in range(n):
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        drow = D[i].copy()
        drow[lo:hi] = np.inf
        j = int(np.argmin(drow))
        if np.isfinite(drow[j]) and drow[j] > 0:
            nn[i] = j

    div = []
    eps = 1e-12

    for k in range(max_t):
        vals = []
        for i, j in enumerate(nn):
            if j < 0:
                continue
            if i + k < n and j + k < n:
                d = np.linalg.norm(Y[i + k] - Y[j + k])
                if np.isfinite(d) and d > 0:
                    vals.append(np.log(d + eps))
        div.append(np.mean(vals) if vals else np.nan)

    div = np.asarray(div, dtype=float)
    ks = np.arange(len(div))
    mask = np.isfinite(div) & (ks >= fit_start) & (ks <= fit_end)

    if np.sum(mask) < 3:
        return np.nan

    try:
        slope = np.polyfit(ks[mask], div[mask], 1)[0]
        return float(slope)
    except Exception:
        return np.nan


def lyapunov_interpretation(lle):
    """
    Interpretación orientativa para RRi 5 min.
    No son puntos de corte diagnósticos.
    """
    if lle is None or not np.isfinite(lle):
        return "No calculado: ventana corta, señal demasiado regular o embedding insuficiente."
    if lle < 0.0:
        return "LLE negativo: convergencia/regularidad extrema; interpretar con cautela en HRV."
    if lle < 0.03:
        return "LLE muy bajo: dinámica rígida o muy estable; baja sensibilidad a cambios iniciales."
    if lle < 0.15:
        return "LLE bajo-moderado: rango fisiológico orientativo; estabilidad con cierta adaptabilidad."
    if lle < 0.30:
        return "LLE moderado-alto: mayor divergencia dinámica; posible alta adaptabilidad o irregularidad."
    return "LLE alto: divergencia rápida; puede reflejar caos, ruido, arritmia o señal poco estacionaria."


def advanced_nonlinear_metrics(rr):
    rr_ms = np.asarray(rr, dtype=float) * 1000.0
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    rr_ent = smoothness_priors_detrend(rr_ms, LAMBDA_DEFAULT)
    out = {
        "Lyapunov_LLE": lyapunov_rosenstein(rr_ent, emb_dim=6, tau=1, theiler=20, max_t=30, fit_start=1, fit_end=10),
        "Hurst": hurst_rs(rr_ms),
        "KatzFD": katz_fd(rr_ms),
        "PetrosianFD": petrosian_fd(rr_ms),
        "DispEn": dispersion_entropy(rr_ent, m=3, c=6, tau=1),
    }
    out.update(mde_metrics(rr_ent, max_scale=20, m=3, c=6))
    return out



# ============================================================
# WAVELET/STFT SCALOGRAM + AYUDA INTERPRETATIVA v11.4
# ============================================================

def wavelet_scalogram_figure(rr, windows=None, title="Wavelet/STFT scalogram"):
    """
    Mapa tiempo-frecuencia ligero basado en STFT.
    No es una CWT Morlet estricta, pero permite visualizar lo que clínicamente
    interesa: cuándo aparece/desaparece HF y cuándo emerge LF.
    """
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr)]

    if len(rr) < 30:
        fig = go.Figure()
        fig.update_layout(title=f"{title} · señal insuficiente")
        return fig

    try:
        ti, xi = interpolate_rr(rr, fs=FS_INTERP, apply_lambda=True, lam=LAMBDA_DEFAULT)
        x = xi * 1000.0
        x = x - np.mean(x)

        nperseg = min(max(64, int(64 * FS_INTERP)), len(x))
        if len(x) < nperseg:
            nperseg = len(x)
        noverlap = int(0.80 * nperseg) if nperseg > 10 else 0

        f, tt, Zxx = signal.stft(
            x,
            fs=FS_INTERP,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
            padded=False,
        )
        power = np.abs(Zxx) ** 2
        mask = (f >= 0.0033) & (f <= 0.40)
        f2 = f[mask]
        p2 = power[mask, :]

        if len(f2) == 0 or p2.size == 0:
            fig = go.Figure()
            fig.update_layout(title=f"{title} · sin contenido frecuencial")
            return fig

        # Potencias por banda a lo largo del tiempo
        def band_ts(lo, hi):
            m = (f2 >= lo) & (f2 < hi)
            if not np.any(m):
                return np.full(len(tt), np.nan)
            return np.trapezoid(p2[m, :], f2[m], axis=0)

        lf_ts = band_ts(0.04, 0.15)
        hf_ts = band_ts(0.15, 0.40)
        vlf_ts = band_ts(0.0033, 0.04)
        ratio = np.divide(lf_ts, hf_ts, out=np.full_like(lf_ts, np.nan, dtype=float), where=hf_ts > 0)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.68, 0.32],
            subplot_titles=(
                "Scalogram tiempo-frecuencia: VLF/LF/HF",
                "Evolución temporal de potencia LF, HF y LF/HF"
            ),
            specs=[[{"type": "heatmap"}], [{"type": "xy"}]],
        )

        z = np.log10(p2 + np.nanpercentile(p2[p2 > 0], 5) * 0.1 if np.any(p2 > 0) else p2 + 1e-12)

        fig.add_trace(
            go.Heatmap(
                x=tt / 60.0,
                y=f2,
                z=z,
                colorbar=dict(title="log potencia"),
                hovertemplate="Tiempo=%{x:.2f} min<br>Frecuencia=%{y:.3f} Hz<br>logP=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(go.Scatter(x=tt / 60.0, y=vlf_ts, mode="lines", name="VLF tiempo", line=dict(color="green", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=tt / 60.0, y=lf_ts, mode="lines", name="LF tiempo", line=dict(color="blue", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=tt / 60.0, y=hf_ts, mode="lines", name="HF tiempo", line=dict(color="deeppink", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=tt / 60.0, y=ratio, mode="lines", name="LF/HF tiempo", line=dict(color="red", width=2), yaxis="y3"), row=2, col=1)

        # Límites de bandas
        for y in [0.04, 0.15]:
            fig.add_hline(y=y, line_dash="dash", line_width=1, row=1, col=1)

        # Rectángulos de fases si existen
        if windows:
            for ph, w in windows.items():
                if w is None:
                    continue
                try:
                    s, e = float(w[0]) / 60.0, float(w[1]) / 60.0
                    grp = PHASE_GROUP.get(ph, ph)
                    fig.add_vrect(
                        x0=s,
                        x1=e,
                        fillcolor=PHASE_LINE_COLORS.get(grp, "#888"),
                        opacity=0.12,
                        line_width=0,
                        row=1,
                        col=1,
                    )
                    fig.add_vrect(
                        x0=s,
                        x1=e,
                        fillcolor=PHASE_LINE_COLORS.get(grp, "#888"),
                        opacity=0.08,
                        line_width=0,
                        row=2,
                        col=1,
                    )
                    fig.add_annotation(
                        x=(s + e) / 2,
                        y=0.395,
                        text=ph,
                        showarrow=False,
                        font=dict(size=10),
                        row=1,
                        col=1,
                    )
                except Exception:
                    pass

        fig.update_yaxes(title_text="Frecuencia (Hz)", range=[0.0033, 0.40], row=1, col=1)
        fig.update_yaxes(title_text="Potencia", row=2, col=1)
        fig.update_xaxes(title_text="Tiempo (min)", row=2, col=1)

        fig.update_layout(
            title=title,
            height=780,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            margin=dict(l=70, r=80, t=90, b=90),
        )
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"{title} · error: {e}")
        return fig


def advanced_methods_reference_markdown():
    """
    Texto interno de ayuda para la app.
    """
    return """
### Métodos frecuenciales

| Parámetro | Qué mide | Fórmula / idea | Interpretación orientativa |
|---|---|---|---|
| VLF, LF, HF, TOTAL | Potencia espectral por bandas mediante Welch/FFT | Integral de PSD en VLF 0.0033-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz | HF suele reflejar modulación vagal respiratoria; LF oscilaciones barorreflejas/mixtas; VLF procesos lentos. |
| VLF_LS, LF_LS, HF_LS | Lo mismo, pero con Lomb-Scargle | Estima PSD sin interpolar RRi: útil para muestreo irregular | Útil cuando la señal RRi es irregular o cuando se quiere minimizar el efecto de interpolación. |
| VLF_AR, LF_AR, HF_AR | PSD por modelo autorregresivo | x(n)=Σ a_k·x(n-k)+e(n) | Puede definir picos LF/HF con más claridad en ventanas cortas; depende del orden AR. |
| VLF_WAV_MEAN, LF_WAV_MEAN, HF_WAV_MEAN | Potencia tiempo-frecuencia media por banda | Media temporal de la potencia STFT/wavelet en cada banda | Indica cuánto peso medio tiene cada banda durante la ventana. |
| VLF_WAV_SD, LF_WAV_SD, HF_WAV_SD | Variabilidad temporal de cada banda | SD temporal de la potencia por banda | Alto = potencia en ráfagas/cambios; bajo = banda estable. |
| VLF_DOM_PCT, LF_DOM_PCT, HF_DOM_PCT | Dominancia temporal | % de puntos temporales donde esa banda es la mayor tras normalizar cada banda por su media: VLF_n=VLF/mean(VLF), LF_n=LF/mean(LF), HF_n=HF/mean(HF) | HF alto = predominio respiratorio/vagal relativo; LF alto = barorreflejo relativo; VLF alto = regulación lenta relativa. |
| EPISODES y TRANSITIONS | Episodios y cambios de régimen | Conteo/duración de dominios VLF/LF/HF y cambios entre ellos | Transiciones altas = movilidad entre regímenes relativos; transiciones bajas = régimen fijo. |
| WAV_ENTROPY_BANDS / GLOBAL | Entropía de distribución energética | H=-Σp·log(p), normalizada | Alta = energía distribuida; baja = energía concentrada en una banda/tiempo. |

### Scalogram wavelet/STFT

El scalogram muestra frecuencia y tiempo simultáneamente:

- eje X = tiempo;
- eje Y = frecuencia;
- color = potencia;
- línea 0.04 Hz separa VLF/LF;
- línea 0.15 Hz separa LF/HF.

Sirve para ver:

- cuándo aparece HF;
- cuándo desaparece HF;
- cuándo emerge LF;
- si hay cambios transitorios dentro de una misma fase;
- si una ventana de 5 minutos es realmente estacionaria.

### Métodos no lineales avanzados

| Parámetro | Qué mide | Fórmula / idea | Referencia orientativa |
|---|---|---|---|
| Lyapunov_LLE | Estabilidad dinámica / sensibilidad a condiciones iniciales | Algoritmo de Rosenstein: pendiente de la divergencia media log(d(k)) entre trayectorias vecinas | <0.03 rígido; 0.03-0.15 adaptabilidad fisiológica orientativa; 0.15-0.30 alta divergencia; >0.30 posible ruido/arritmia/inestabilidad. |
| Hurst | Memoria/persistencia de largo plazo | R/S: pendiente log(R/S) vs log(n) | H≈0.5 aleatorio; H>0.5 persistente; H<0.5 antipersistente. |
| KatzFD | Dimensión fractal geométrica | FD=log10(n)/(log10(d/L)+log10(n)) | Mayor valor = trayectoria más tortuosa. Comparar sobre todo longitudinalmente. |
| PetrosianFD | Cambios de dirección de la señal | Usa número de cambios de signo de la derivada | Rápido y estable; mayor valor = más cambios locales. |
| DispEn | Entropía de dispersión | Convierte la señal en clases y estima diversidad de patrones | Más robusta que SampEn en ventanas cortas. Mayor valor = mayor diversidad de patrones. |
| MDE1-20 | Dispersion Entropy multiescala | DispEn aplicada a señales coarse-grained 1-20 | Alternativa moderna a MSE cuando MSE clásico falla por A=0. |

### Índice de Lyapunov / Rosenstein

El exponente máximo de Lyapunov estima si dos trayectorias inicialmente muy próximas se separan rápido o lentamente.

Fórmula conceptual:

λ = pendiente de log(d(k)) frente a k

donde d(k) es la distancia media entre trayectorias vecinas tras k pasos.

Interpretación orientativa en RRi de 5 minutos:

- LLE < 0.03: dinámica muy rígida o excesivamente estable.
- 0.03-0.15: estabilidad con adaptabilidad fisiológica.
- 0.15-0.30: divergencia aumentada; puede indicar alta adaptabilidad o irregularidad.
- >0.30: posible ruido, arritmia, no estacionariedad o dinámica muy inestable.

No debe usarse como diagnóstico aislado. Tiene más valor en comparación por fase o seguimiento longitudinal.

### Recomendación para ventanas de 5 minutos

Para lectura principal:

- Frecuencia: Welch + Lomb-Scargle como contraste.
- Dinámica temporal: scalogram para comprobar si HF/LF cambian dentro de la ventana.
- Complejidad clásica: SampEn y MSE sólo hasta escalas válidas.
- Complejidad avanzada: DispEn y MDE1-20.
- Fractalidad: DFA α1/α2 + Hurst/Katz/Petrosian como complemento.

Importante: no hay valores universales cerrados para todos estos parámetros. Lo más fiable es comparar por fase, por paciente y longitudinalmente.
"""



# ============================================================
# ÍNDICES FISIOLÓGICOS MULTIVARIADOS v13.0
# ============================================================

def _clip_score(value, low, high, invert=False):
    """Convierte una métrica a 0-100 mediante límites orientativos robustos."""
    try:
        v = float(value)
        if not np.isfinite(v) or high <= low:
            return np.nan
        s = 100.0 * (v - low) / (high - low)
        s = float(np.clip(s, 0.0, 100.0))
        return 100.0 - s if invert else s
    except Exception:
        return np.nan


def _log_score(value, low, high, invert=False):
    try:
        v = float(value)
        if not np.isfinite(v) or v <= 0:
            return np.nan
        return _clip_score(np.log10(v), np.log10(low), np.log10(high), invert=invert)
    except Exception:
        return np.nan


def _optimal_score(value, center, tolerance):
    """100 cerca del centro; cae suavemente al alejarse."""
    try:
        v = float(value)
        if not np.isfinite(v) or tolerance <= 0:
            return np.nan
        return float(100.0 * np.exp(-0.5 * ((v-center)/tolerance)**2))
    except Exception:
        return np.nan


def _weighted_mean_available(items):
    vals, weights = [], []
    for value, weight in items:
        try:
            v=float(value)
            if np.isfinite(v):
                vals.append(v); weights.append(float(weight))
        except Exception:
            pass
    if not vals or np.sum(weights) <= 0:
        return np.nan
    return float(np.average(vals, weights=weights))


def _multiscale_area_score(row, prefix='MDE', max_scale=20):
    vals=[]
    for i in range(1, max_scale+1):
        try:
            v=float(row.get(f'{prefix}{i}', np.nan))
            if np.isfinite(v): vals.append(v)
        except Exception:
            pass
    if not vals:
        return np.nan
    # Escala orientativa que conserva comparabilidad dentro de la app.
    mean_val=float(np.mean(vals))
    if prefix == 'MDE':
        return _clip_score(mean_val, 1.5, 4.5)
    return _clip_score(mean_val, 0.5, 2.2)


def physiological_indices_from_row(row):
    """
    Construye índices 0-100 a partir de métricas convergentes.
    Son índices fisiológicos transparentes y explicables, no probabilidades clínicas
    ni un modelo entrenado con desenlaces.
    """
    get=lambda k: row.get(k, np.nan)

    vagal = _weighted_mean_available([
        (_clip_score(get('RMSSD'), 10, 60), 1.4),
        (_clip_score(get('SD1'), 7, 42), 1.2),
        (_log_score(get('HF'), 20, 1200), 1.1),
        (_log_score(get('HF_LS'), 20, 1200), 0.5),
        (_clip_score(get('HF_DOM_PCT'), 10, 65), 0.8),
    ])

    amplitude = _weighted_mean_available([
        (_clip_score(get('SDNN'), 15, 70), 1.3),
        (_clip_score(get('SD2'), 20, 95), 1.1),
        (_log_score(get('TOTAL'), 80, 4000), 1.0),
        (_log_score(get('TOTAL_LS'), 80, 4000), 0.4),
    ])

    lle = get('Lyapunov_LLE')
    lle_score = _optimal_score(lle, 0.10, 0.10)
    complexity = _weighted_mean_available([
        (_clip_score(get('SampEn'), 0.6, 2.2), 1.2),
        (_clip_score(get('DispEn'), 2.0, 4.5), 1.1),
        (_clip_score(get('D2'), 0.8, 4.0), 0.8),
        (lle_score, 0.7),
        (_multiscale_area_score(row, 'MDE', 20), 1.2),
        (_multiscale_area_score(row, 'MSE', 20), 0.6),
    ])

    dfa1 = _optimal_score(get('DFA_alpha1'), 1.0, 0.30)
    wave_entropy = _weighted_mean_available([
        (_clip_score(get('WAV_ENTROPY_BANDS'), 0.25, 0.90), 1.0),
        (_clip_score(get('WAV_ENTROPY_GLOBAL'), 0.25, 0.90), 0.8),
    ])
    transitions = _clip_score(get('WAV_TRANSITIONS_PER_MIN'), 0.2, 2.5)

    det_rigid = _clip_score(get('DET'), 85, 99.5)
    lmax_rel = np.nan
    try:
        n=float(get('N_RRi')); lm=float(get('Lmax'))
        if np.isfinite(n) and n>0 and np.isfinite(lm): lmax_rel=_clip_score(lm/n, 0.05, 0.75)
    except Exception: pass
    hurst_rigid = _clip_score(get('Hurst'), 0.55, 1.05)
    entropy_inverse = 100-complexity if np.isfinite(complexity) else np.nan
    transition_inverse = 100-transitions if np.isfinite(transitions) else np.nan
    rigidity = _weighted_mean_available([
        (det_rigid, 1.2), (lmax_rel, 1.0), (hurst_rigid, 0.8),
        (entropy_inverse, 1.2), (transition_inverse, 0.6),
    ])

    slow = _weighted_mean_available([
        (_clip_score(get('VLF_DOM_PCT'), 15, 65), 1.2),
        (_log_score(get('VLF'), 20, 1500), 0.8),
        (_log_score(get('VLF_LS'), 20, 1500), 0.4),
        (_clip_score(get('Hurst'), 0.5, 1.05), 0.6),
        (_clip_score(get('DFA_alpha2'), 0.8, 1.6), 0.5),
    ])

    adaptability = _weighted_mean_available([
        (vagal, 1.0), (amplitude, 0.8), (complexity, 1.2),
        (dfa1, 1.0), (wave_entropy, 0.7), (transitions, 0.6),
        ((100-rigidity) if np.isfinite(rigidity) else np.nan, 1.1),
    ])

    # Calidad/confianza: penaliza ventanas cortas y artefactos si están disponibles.
    n_score=_clip_score(get('N_RRi'), 120, 350)
    confidence=_weighted_mean_available([(n_score,1.0)])

    return {
        'IDX_Vagal': vagal,
        'IDX_Amplitud': amplitude,
        'IDX_Complejidad': complexity,
        'IDX_Rigidez': rigidity,
        'IDX_Adaptabilidad': adaptability,
        'IDX_Regulacion_Lenta': slow,
        'IDX_Confianza': confidence,
    }


def _index_level(v, reverse=False):
    if v is None or not np.isfinite(v): return 'No calculado'
    if v < 25: level='Muy bajo'
    elif v < 45: level='Bajo'
    elif v < 60: level='Intermedio'
    elif v < 80: level='Alto'
    else: level='Muy alto'
    if not reverse: return level
    return {'Muy bajo':'Muy baja','Bajo':'Baja','Intermedio':'Intermedia','Alto':'Alta','Muy alto':'Muy alta'}.get(level, level)


def autonomic_profile_from_indices(indices):
    """Clasificación explicable del perfil autonómico, sin diagnóstico médico."""
    v=indices.get('IDX_Vagal', np.nan)
    a=indices.get('IDX_Adaptabilidad', np.nan)
    r=indices.get('IDX_Rigidez', np.nan)
    s=indices.get('IDX_Regulacion_Lenta', np.nan)
    c=indices.get('IDX_Complejidad', np.nan)

    labels=[]
    if np.isfinite(a):
        labels.append('adaptabilidad conservada' if a>=60 else ('adaptabilidad intermedia' if a>=40 else 'adaptabilidad reducida'))
    if np.isfinite(v):
        labels.append('modulación vagal alta' if v>=65 else ('modulación vagal intermedia' if v>=40 else 'modulación vagal baja'))
    if np.isfinite(r) and r>=65: labels.append('rigidez dinámica elevada')
    if np.isfinite(s) and s>=65: labels.append('regulación lenta dominante')
    if np.isfinite(c) and c<35: labels.append('complejidad reducida')
    return '; '.join(labels).capitalize() if labels else 'Perfil no clasificable'


def build_physiological_indices(long_df):
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    rows=[]
    for _, row in long_df.iterrows():
        idxs=physiological_indices_from_row(row)
        out={'Registro':row.get('Registro',''), 'Fase':row.get('Fase','')}
        out.update(idxs)
        for _hvg_col in [
            'HVG_degree_mean','HVG_degree_max','HVG_hubs_p90','HVG_clustering','HVG_lambda',
            'HVG_path_length','HVG_diameter','HVG_compactness_index',
            'HVG_graph_score_small_world','HVG_graph_score_scale_free'
        ]:
            out[_hvg_col] = pd.to_numeric(row.get(_hvg_col), errors='coerce')
        # v15.4.12 · conservar fisiología primaria para aprendizaje longitudinal.
        for _pcol in V1548_PRIMARY_FEATURES if 'V1548_PRIMARY_FEATURES' in globals() else []:
            if _pcol != 'Artefactos_pct':
                out[_pcol] = pd.to_numeric(row.get(_pcol), errors='coerce')
        out['Perfil_autonomico']=autonomic_profile_from_indices(idxs)
        rows.append(out)
    return pd.DataFrame(rows)


def physiological_indices_reference_table():
    return pd.DataFrame([
        {'Índice':'IDX_Vagal','Integra':'RMSSD, SD1, HF, HF_LS y HF_DOM_PCT','0-24':'Muy bajo','25-44':'Bajo','45-59':'Intermedio','60-79':'Alto','80-100':'Muy alto'},
        {'Índice':'IDX_Amplitud','Integra':'SDNN, SD2, TOTAL y TOTAL_LS','0-24':'Muy bajo','25-44':'Bajo','45-59':'Intermedio','60-79':'Alto','80-100':'Muy alto'},
        {'Índice':'IDX_Complejidad','Integra':'SampEn, DispEn, D2, Lyapunov, MDE y MSE','0-24':'Muy baja','25-44':'Baja','45-59':'Intermedia','60-79':'Alta','80-100':'Muy alta'},
        {'Índice':'IDX_Rigidez','Integra':'DET, Lmax/N, Hurst, entropía inversa y baja movilidad wavelet','0-24':'Muy baja','25-44':'Baja','45-59':'Intermedia','60-79':'Alta','80-100':'Muy alta'},
        {'Índice':'IDX_Adaptabilidad','Integra':'Vagalidad, amplitud, complejidad, DFAα1, wavelet e inversa de rigidez','0-24':'Muy baja','25-44':'Baja','45-59':'Intermedia','60-79':'Alta','80-100':'Muy alta'},
        {'Índice':'IDX_Regulacion_Lenta','Integra':'VLF, VLF_LS, VLF_DOM_PCT, Hurst y DFAα2','0-24':'Muy baja','25-44':'Baja','45-59':'Intermedia','60-79':'Alta','80-100':'Muy alta'},
    ])


def _extract_record_datetime(record_name):
    """Extrae fecha/hora del nombre del registro. Devuelve NaT si no hay patrón reconocible."""
    text = str(record_name or '')
    patterns = [
        (r'(20\d{2})[-_](\d{2})[-_](\d{2})[T_ -](\d{2})[-_:](\d{2})(?:[-_:](\d{2}))?', '%Y-%m-%d %H:%M:%S'),
        (r'(20\d{2})[-_](\d{2})[-_](\d{2})', '%Y-%m-%d'),
        (r'(\d{2})[-_](\d{2})[-_](20\d{2})[T_ -](\d{2})[-_:](\d{2})(?:[-_:](\d{2}))?', '%d-%m-%Y %H:%M:%S'),
        (r'(\d{2})[-_](\d{2})[-_](20\d{2})', '%d-%m-%Y'),
    ]
    for pattern, fmt in patterns:
        m = re.search(pattern, text)
        if not m:
            continue
        parts = list(m.groups())
        if '%H' in fmt:
            if parts[-1] is None:
                parts[-1] = '00'
            value = '-'.join(parts[:3]) + ' ' + ':'.join(parts[3:6])
        else:
            value = '-'.join(parts[:3])
        try:
            return pd.to_datetime(value, format=fmt, errors='raise')
        except Exception:
            pass
    return pd.NaT


def chronological_indices_table(indices_df, phase=None):
    """Prepara una fila por registro/fase y la ordena cronológicamente."""
    if indices_df is None or indices_df.empty:
        return pd.DataFrame()
    df = indices_df.copy()
    if phase and phase != 'Todas':
        df = df[df['Fase'].astype(str) == str(phase)]
    df['Fecha_hora'] = df['Registro'].map(_extract_record_datetime)
    # Los nombres sin fecha reconocible permanecen al final conservando el orden de carga.
    df['_orden_original'] = np.arange(len(df))
    df = df.sort_values(['Fecha_hora', '_orden_original'], na_position='last').reset_index(drop=True)
    df['Etiqueta_cronologica'] = df.apply(
        lambda r: (r['Fecha_hora'].strftime('%d/%m/%Y<br>%H:%M:%S') if pd.notna(r['Fecha_hora']) else _record_axis_label(r['Registro'], multiline=True))
                  + (f" · {r['Fase']}" if phase == 'Todas' else ''), axis=1
    )
    return df.drop(columns=['_orden_original'])


def chronological_indices_figure(indices_df, phase='Basal'):
    """
    Columnas agrupadas de índices.

    v15.4.0: cuando hay varios pacientes, coloca sus registros en bloques
    contiguos separados visualmente, sin mezclar cronologías.
    """
    df = chronological_indices_table(indices_df, phase=phase)
    fig = go.Figure()
    cols = ['IDX_Vagal','IDX_Amplitud','IDX_Complejidad','IDX_Rigidez','IDX_Adaptabilidad','IDX_Regulacion_Lenta']
    labels = {
        'IDX_Vagal':'Vagal', 'IDX_Amplitud':'Amplitud', 'IDX_Complejidad':'Complejidad',
        'IDX_Rigidez':'Rigidez', 'IDX_Adaptabilidad':'Adaptabilidad',
        'IDX_Regulacion_Lenta':'Regulación lenta'
    }
    if df.empty:
        return fig, df

    if 'Paciente_ID' in df.columns:
        df['_Paciente_plot'] = df['Paciente_ID'].astype(str).map(normalize_patient_id)
    else:
        df['_Paciente_plot'] = df['Registro'].astype(str).map(lambda x: normalize_patient_id(infer_patient_id(x)))
    patients = sorted(df['_Paciente_plot'].dropna().unique().tolist(), key=lambda x: str(x).lower())
    multi_patient = len(patients) > 1

    if multi_patient:
        ordered_rows = []
        xvals, xtext = [], []
        spans = []
        cursor = 0.0
        gap = 1.25
        for pi, pid in enumerate(patients):
            g = df[df['_Paciente_plot'] == pid].sort_values('Fecha_hora', na_position='last')
            start_x = cursor
            for ridx, row in g.iterrows():
                ordered_rows.append((cursor, ridx))
                xvals.append(cursor)
                date_text = _record_axis_label(row['Registro'], multiline=True)
                xtext.append(f"{_patient_display_name(pid)}<br>{date_text}")
                cursor += 1.0
            end_x = cursor - 1.0
            spans.append((pid, start_x, end_x))
            if pi < len(patients)-1:
                cursor += gap

        odf = df.loc[[idx for _, idx in ordered_rows]].copy()
        xpos = [x for x, _ in ordered_rows]
        custom = np.column_stack([
            odf['_Paciente_plot'].map(_patient_display_name).astype(str),
            odf['Registro'].astype(str),
            odf['Fase'].astype(str)
        ])
        for col in cols:
            if col in odf.columns:
                fig.add_trace(go.Bar(
                    name=labels.get(col,col), x=xpos, y=odf[col],
                    customdata=custom,
                    hovertemplate='Paciente: %{customdata[0]}<br>Registro: %{customdata[1]}<br>'
                                  'Fase: %{customdata[2]}<br>%{fullData.name}: %{y:.1f}/100<extra></extra>'
                ))
        for i in range(len(spans)-1):
            _, _, end_x = spans[i]
            _, next_start, _ = spans[i+1]
            fig.add_vline(x=(end_x+next_start)/2.0, line_width=1, line_dash='dot', opacity=0.35)
        fig.update_xaxes(
            tickmode='array', tickvals=xvals, ticktext=xtext,
            title_text='Paciente · fecha del registro'
        )
        fig.update_layout(
            barmode='group', yaxis=dict(range=[0,100], title='Índice 0-100'),
            title=f'Evolución de los índices agrupada por paciente · {phase}',
            height=610, legend_title='Índice', hovermode='closest',
            margin=dict(l=55, r=35, t=80, b=145)
        )
        return fig, df.drop(columns=['_Paciente_plot'])

    for col in cols:
        if col in df.columns:
            fig.add_trace(go.Bar(
                name=labels.get(col,col), x=df['Etiqueta_cronologica'], y=df[col],
                customdata=np.column_stack([df['Registro'].astype(str), df['Fase'].astype(str)]),
                hovertemplate='<b>%{customdata[0]}</b><br>Fase: %{customdata[1]}<br>%{fullData.name}: %{y:.1f}/100<extra></extra>'
            ))
    fig.update_layout(
        barmode='group', yaxis=dict(range=[0,100], title='Índice 0-100'),
        xaxis_title='Fecha del registro',
        title=f'Evolución cronológica de los índices · {phase}', height=590,
        legend_title='Índice', hovermode='x unified'
    )
    return fig, df.drop(columns=['_Paciente_plot'])


def _smooth_index_series(values, method='Media móvil', window=3):
    """Suaviza una serie corta sin alterar los valores originales mostrados en barras."""
    ser = pd.Series(pd.to_numeric(values, errors='coerce'), dtype=float)
    valid_n = int(ser.notna().sum())
    if valid_n == 0:
        return ser
    window = max(2, min(int(window), max(valid_n, 2)))
    if method == 'Media exponencial':
        return ser.ewm(span=window, adjust=False, min_periods=1).mean()
    if method == 'Mediana móvil':
        return ser.rolling(window=window, min_periods=1, center=True).median()
    return ser.rolling(window=window, min_periods=1, center=True).mean()


def individual_index_chronological_figure(indices_df, index_col, phase='Basal', smooth_method='Media móvil', smooth_window=3):
    """
    Muestra un índice como columnas cronológicas + tendencia.

    v15.4.0:
    - Si hay varios pacientes, los agrupa en bloques separados.
    - La tendencia se calcula de forma independiente dentro de cada paciente.
    - Nunca se conectan observaciones de pacientes distintos.
    """
    df = chronological_indices_table(indices_df, phase=phase)
    labels = {
        'IDX_Vagal':'Vagal', 'IDX_Amplitud':'Amplitud', 'IDX_Complejidad':'Complejidad',
        'IDX_Rigidez':'Rigidez', 'IDX_Adaptabilidad':'Adaptabilidad',
        'IDX_Regulacion_Lenta':'Regulación lenta'
    }
    label = labels.get(index_col, index_col)
    fig = go.Figure()
    if df.empty or index_col not in df.columns:
        fig.update_layout(title=f'{label} · sin datos', height=390)
        return fig, df

    # Preferir Paciente_ID cuando existe; si no, inferir desde Registro.
    if 'Paciente_ID' in df.columns:
        pids = df['Paciente_ID'].astype(str).map(normalize_patient_id)
    else:
        pids = df['Registro'].astype(str).map(lambda x: normalize_patient_id(infer_patient_id(x)))
    df = df.copy()
    df['_Paciente_plot'] = pids
    patient_order = sorted(df['_Paciente_plot'].dropna().unique().tolist(), key=lambda x: str(x).lower())
    multi_patient = len(patient_order) > 1

    if multi_patient:
        cursor = 0.0
        gap = 1.25
        tick_vals, tick_text = [], []
        spans = []
        for pi, pid in enumerate(patient_order):
            g = df[df['_Paciente_plot'] == pid].copy()
            g = g.sort_values('Fecha_hora', na_position='last')
            xs = []
            ys = []
            customs = []
            start_x = cursor
            for _, row in g.iterrows():
                xs.append(cursor)
                tick_vals.append(cursor)
                # v15.4.0: eje X compacto. El paciente queda identificado por color/leyenda.
                tick_text.append(_record_axis_label_compact(row['Registro']))
                val = pd.to_numeric(row.get(index_col), errors='coerce')
                ys.append(float(val) if pd.notna(val) else np.nan)
                customs.append([_patient_display_name(pid), str(row.get('Registro','')), str(row.get('Fase',''))])
                cursor += 1.0
            end_x = cursor - 1.0
            spans.append((pid, start_x, end_x))
            color = _export_color_for(pi)

            fig.add_trace(go.Bar(
                name=f'{_patient_display_name(pid)} · valor',
                x=xs, y=ys, marker=dict(color=color), opacity=0.74,
                customdata=customs,
                hovertemplate='Paciente: %{customdata[0]}<br>Registro: %{customdata[1]}<br>'
                              'Fase: %{customdata[2]}<br>Valor: %{y:.1f}/100<extra></extra>'
            ))

            valid = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
            if len(valid) >= 2:
                vx = np.asarray([x for x, _ in valid], dtype=float)
                vy = [y for _, y in valid]
                smooth = _smooth_index_series(vy, method=smooth_method, window=smooth_window)
                fig.add_trace(go.Scatter(
                    name=f'{_patient_display_name(pid)} · tendencia',
                    x=vx, y=smooth, mode='lines+markers',
                    line=dict(width=3, color=color), marker=dict(size=7, color=color),
                    hovertemplate='Tendencia: %{y:.1f}/100<extra></extra>'
                ))

            if pi < len(patient_order) - 1:
                cursor += gap

        for i in range(len(spans)-1):
            _, _, end_x = spans[i]
            _, next_start, _ = spans[i+1]
            fig.add_vline(x=(end_x + next_start)/2.0, line_width=1, line_dash='dot', opacity=0.35)

        fig.update_layout(
            title=f'{label} · evolución por paciente · {phase}',
            yaxis=dict(range=[0,100], title=f'Índice {label} (0-100)'),
            height=450, hovermode='closest', bargap=0.28,
            legend_title='Paciente',
            margin=dict(l=55, r=35, t=80, b=115)
        )
        fig.update_xaxes(
            title='Fecha / hora del registro',
            tickmode='array', tickvals=tick_vals, ticktext=tick_text, tickangle=-45,
            automargin=True
        )
        return fig, df.drop(columns=['_Paciente_plot'])

    # Un único paciente: comportamiento previo.
    y = pd.to_numeric(df[index_col], errors='coerce')
    smooth = _smooth_index_series(y, method=smooth_method, window=smooth_window)
    custom = np.column_stack([df['Registro'].astype(str), df['Fase'].astype(str)])
    x_single = np.arange(len(df), dtype=float)
    tick_single = [_record_axis_label_compact(r) for r in df['Registro'].astype(str)]
    fig.add_trace(go.Bar(
        name='Valor observado', x=x_single, y=y, customdata=custom,
        hovertemplate='<b>%{customdata[0]}</b><br>Fase: %{customdata[1]}<br>Valor: %{y:.1f}/100<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        name=f'Tendencia suavizada ({smooth_method.lower()}, n={int(smooth_window)})',
        x=x_single, y=smooth, mode='lines+markers',
        line=dict(width=3), marker=dict(size=7),
        hovertemplate='Tendencia: %{y:.1f}/100<extra></extra>'
    ))
    fig.update_layout(
        title=f'{label} · evolución cronológica · {phase}',
        yaxis=dict(range=[0,100], title=f'Índice {label} (0-100)'),
        xaxis=dict(
            title='Fecha / hora del registro', tickangle=-45, automargin=True,
            tickmode='array', tickvals=x_single.tolist(), ticktext=tick_single
        ),
        height=450, hovermode='x unified', bargap=0.28,
        margin=dict(l=55, r=35, t=80, b=120),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig, df.drop(columns=['_Paciente_plot'])

def render_separate_chronological_indices(indices_df, phase, key_prefix):
    """Renderiza los seis índices por separado para evitar que se oculten entre sí."""
    c1, c2 = st.columns([1,1])
    with c1:
        smooth_method = st.selectbox(
            'Suavizado de la tendencia', ['Media móvil','Media exponencial','Mediana móvil'],
            key=f'{key_prefix}_smooth_method'
        )
    with c2:
        _chrono_n = len(chronological_indices_table(indices_df, phase=phase))
        max_n = min(7, max(1, _chrono_n))
        if max_n >= 3:
            default_n = min(3, max_n)
            smooth_window = st.slider(
                'Ventana de suavizado (registros)', 2, max_n, default_n,
                key=f'{key_prefix}_smooth_window'
            )
        else:
            # Streamlit no admite un slider con min_value == max_value.
            # Con 0-2 registros no hay una ventana de suavizado ajustable útil.
            smooth_window = 2
            st.caption('Ventana de suavizado: 2 registros (fija; hacen falta ≥3 registros para ajustarla).')
    index_specs = [
        ('IDX_Vagal','Vagal'), ('IDX_Amplitud','Amplitud'), ('IDX_Complejidad','Complejidad'),
        ('IDX_Rigidez','Rigidez'), ('IDX_Adaptabilidad','Adaptabilidad'),
        ('IDX_Regulacion_Lenta','Regulación lenta')
    ]
    for i in range(0, len(index_specs), 2):
        cols = st.columns(2)
        for j, (index_col, label) in enumerate(index_specs[i:i+2]):
            with cols[j]:
                fig, _ = individual_index_chronological_figure(
                    indices_df, index_col, phase=phase,
                    smooth_method=smooth_method, smooth_window=smooth_window
                )
                st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_{index_col}_{phase}')
    st.caption(
        'Las columnas muestran cada medición real. La línea resume la dirección general y no sustituye los datos originales. '
        'Con pocas observaciones, el suavizado es únicamente descriptivo.'
    )

def physiological_indices_figure(indices_df, record=None):
    df=indices_df.copy()
    if record is not None and 'Registro' in df.columns:
        df=df[df['Registro']==record]
    fig=go.Figure()
    cols=['IDX_Vagal','IDX_Amplitud','IDX_Complejidad','IDX_Rigidez','IDX_Adaptabilidad','IDX_Regulacion_Lenta']
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(name=str(row.get('Fase','')), x=cols, y=[row.get(c,np.nan) for c in cols]))
    fig.update_layout(barmode='group', yaxis=dict(range=[0,100], title='Índice 0-100'), title='Índices fisiológicos multivariados v14.1', height=520, legend_title='Fase')
    return fig


def recovery_index_from_record(metrics_df):
    """Índice de retorno hacia basal; sólo se calcula si existen basal y recuperación."""
    if metrics_df is None or metrics_df.empty or 'Basal' not in metrics_df.index:
        return np.nan, 'No calculado: falta Basal.'
    rec_ph=[p for p in metrics_df.index if str(p).startswith('R')]
    ex_ph=[p for p in metrics_df.index if str(p).startswith('E')]
    if not rec_ph:
        return np.nan, 'No calculado: faltan fases de recuperación.'
    base=metrics_df.loc['Basal']
    rec=metrics_df.loc[rec_ph].mean(numeric_only=True)
    metrics=['MeanHR','RMSSD','SD1','HF','DFA_alpha1','SampEn','DispEn']
    scores=[]
    for m in metrics:
        try:
            b=float(base.get(m,np.nan)); rv=float(rec.get(m,np.nan))
            if not (np.isfinite(b) and np.isfinite(rv)): continue
            if ex_ph:
                ev=float(metrics_df.loc[ex_ph, m].mean())
                denom=abs(ev-b)
            else:
                denom=max(abs(b)*0.25, 1e-9)
            closeness=100*(1-min(abs(rv-b)/(denom+1e-9),1.0))
            scores.append(closeness)
        except Exception: pass
    if not scores:
        return np.nan, 'No calculado: métricas insuficientes.'
    score=float(np.mean(scores))
    txt='Recuperación alta' if score>=70 else ('Recuperación parcial' if score>=40 else 'Recuperación baja')
    return score, txt

GB_INDEX_FEATURES = [
    'IDX_Vagal', 'IDX_Amplitud', 'IDX_Complejidad',
    'IDX_Rigidez', 'IDX_Adaptabilidad', 'IDX_Regulacion_Lenta'
]

def gradient_boosting_training_template():
    """Plantilla mínima para entrenamiento supervisado longitudinal."""
    return pd.DataFrame(columns=[
        'Paciente_ID', 'Fecha', *GB_INDEX_FEATURES, 'Resultado_7d'
    ])

def train_gradient_boosting_model(df, target_col, task='classification', random_state=42):
    """
    Entrena un Gradient Boosting sobre los índices fisiológicos.
    Devuelve modelo, métricas, importancia y predicciones de validación.
    Nunca crea etiquetas: necesita un desenlace observado aportado por el usuario.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError('scikit-learn no está instalado.')
    if target_col not in df.columns:
        raise ValueError(f'No existe la columna objetivo: {target_col}')
    features=[c for c in GB_INDEX_FEATURES if c in df.columns]
    if len(features) < 3:
        raise ValueError('Se necesitan al menos tres índices fisiológicos como predictores.')
    work=df[features+[target_col]].copy()
    work=work.dropna(subset=[target_col])
    if len(work) < 30:
        raise ValueError('Se recomiendan al menos 30 observaciones etiquetadas para un entrenamiento exploratorio.')
    X=work[features].apply(pd.to_numeric, errors='coerce')
    y=work[target_col]

    if task == 'classification':
        y=y.astype(str)
        counts=y.value_counts()
        if len(counts) < 2:
            raise ValueError('La clasificación necesita al menos dos clases.')
        if counts.min() < 5:
            raise ValueError('Cada clase debe contener al menos cinco observaciones.')
        model=Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('gb', GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.035, max_depth=2,
                min_samples_leaf=5, subsample=0.85, random_state=random_state
            ))
        ])
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=random_state,stratify=y)
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        metrics={
            'n_observaciones':len(work), 'n_entrenamiento':len(Xtr), 'n_validacion':len(Xte),
            'accuracy':accuracy_score(yte,pred),
            'balanced_accuracy':balanced_accuracy_score(yte,pred),
            'f1_macro':f1_score(yte,pred,average='macro',zero_division=0),
        }
        cv=StratifiedKFold(n_splits=min(5, int(counts.min())), shuffle=True, random_state=random_state)
        cv_scores=cross_val_score(model,X,y,cv=cv,scoring='balanced_accuracy')
        metrics['cv_balanced_accuracy_media']=float(np.mean(cv_scores))
        metrics['cv_balanced_accuracy_sd']=float(np.std(cv_scores))
        validation=pd.DataFrame({'Real':yte.to_numpy(),'Predicho':pred}, index=yte.index)
        cm=pd.DataFrame(confusion_matrix(yte,pred,labels=model.classes_), index=model.classes_, columns=model.classes_)
        scoring='balanced_accuracy'
    else:
        y=pd.to_numeric(y, errors='coerce')
        valid=y.notna(); X=X.loc[valid]; y=y.loc[valid]
        if len(y) < 30:
            raise ValueError('Se necesitan al menos 30 desenlaces numéricos válidos.')
        model=Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('gb', GradientBoostingRegressor(
                n_estimators=180, learning_rate=0.035, max_depth=2,
                min_samples_leaf=5, subsample=0.85, loss='huber', random_state=random_state
            ))
        ])
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=random_state)
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        metrics={
            'n_observaciones':len(y), 'n_entrenamiento':len(Xtr), 'n_validacion':len(Xte),
            'MAE':mean_absolute_error(yte,pred), 'R2':r2_score(yte,pred),
        }
        cv=KFold(n_splits=min(5,max(2,len(y)//10)),shuffle=True,random_state=random_state)
        cv_scores=-cross_val_score(model,X,y,cv=cv,scoring='neg_mean_absolute_error')
        metrics['cv_MAE_media']=float(np.mean(cv_scores)); metrics['cv_MAE_sd']=float(np.std(cv_scores))
        validation=pd.DataFrame({'Real':yte.to_numpy(),'Predicho':pred}, index=yte.index)
        cm=None; scoring='neg_mean_absolute_error'

    perm=permutation_importance(model,Xte,yte,n_repeats=20,random_state=random_state,scoring=scoring)
    importance=pd.DataFrame({
        'Índice':features, 'Importancia_media':perm.importances_mean, 'Importancia_sd':perm.importances_std
    }).sort_values('Importancia_media',ascending=False)
    return {
        'model':model, 'features':features, 'task':task, 'target':target_col,
        'metrics':metrics, 'importance':importance, 'validation':validation,
        'confusion_matrix':cm, 'classes':list(model.classes_) if task=='classification' else None
    }

def predict_with_gradient_boosting(bundle, indices_df):
    if indices_df is None or indices_df.empty:
        return pd.DataFrame()
    features=bundle['features']
    usable=indices_df.copy()
    X=usable.reindex(columns=features).apply(pd.to_numeric,errors='coerce')
    out=usable[[c for c in ['Registro','Fase'] if c in usable.columns]].copy()
    model=bundle['model']
    out['Prediccion']=model.predict(X)
    if bundle['task']=='classification' and hasattr(model,'predict_proba'):
        probs=model.predict_proba(X)
        for i,cls in enumerate(model.classes_):
            out[f'P_{cls}']=probs[:,i]
        out['Confianza_max']=np.max(probs,axis=1)
    return out

def serialize_gradient_boosting_bundle(bundle):
    import io
    bio=io.BytesIO()
    joblib.dump(bundle,bio)
    return bio.getvalue()

ACTIVE_GB_MODEL_PATH = Path(__file__).resolve().parent / 'modelo_gradient_boosting_activo.joblib'

# ============================================================
# v14.0 · BASE LONGITUDINAL INTERNA Y APRENDIZAJE AUTO-SUPERVISADO
# ============================================================
APP_DATA_DIR = Path(__file__).resolve().parent / '.vrc_data'
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# v15.3.3 · EDITOR TEMPORAL LOCAL EN EL NAVEGADOR
# El arrastre y el contador mm:ss se ejecutan en JavaScript.
# Python/Streamlit sólo recibe la ventana cuando se suelta el ratón.
# ============================================================
_WINDOW_EDITOR_DIR = APP_DATA_DIR / "window_editor_v1533"
_WINDOW_EDITOR_DIR.mkdir(parents=True, exist_ok=True)
_WINDOW_EDITOR_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"/>
<style>
html,body{margin:0;padding:0;background:transparent;color:#e8edf5;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
#wrap{position:relative;width:100%;height:100%;min-height:420px;background:#0e1117;border-radius:8px;overflow:hidden}
canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair}
#hud{position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:5;padding:6px 12px;border-radius:8px;background:rgba(17,24,39,.88);border:1px solid rgba(255,255,255,.16);font-weight:700;font-size:18px;pointer-events:none;white-space:nowrap}
#meta{position:absolute;left:14px;bottom:10px;z-index:5;padding:5px 9px;border-radius:7px;background:rgba(17,24,39,.78);font-size:12px;pointer-events:none}
#hint{position:absolute;right:14px;bottom:10px;z-index:5;padding:5px 9px;border-radius:7px;background:rgba(17,24,39,.78);font-size:12px;pointer-events:none;text-align:right}
#target{position:absolute;right:14px;top:10px;z-index:5;padding:5px 9px;border-radius:7px;background:rgba(17,24,39,.78);font-size:12px;pointer-events:none}
</style></head>
<body><div id="wrap"><canvas id="c"></canvas><div id="hud">00:00</div><div id="target"></div><div id="meta"></div><div id="hint">Arrastra para crear · dentro para mover · bordes para redimensionar</div></div>
<script>
const canvas=document.getElementById('c'), ctx=canvas.getContext('2d'), hud=document.getElementById('hud'), meta=document.getElementById('meta'), target=document.getElementById('target');
let A={times:[],values:[],art_times:[],art_values:[],duration_s:1,start_s:null,end_s:null,target_s:300,height:620,record_name:'',editor_token:''};
let mode=null, anchor=0, dragOffset=0, originalDur=0, pointerId=null;
const pad={l:58,r:22,t:48,b:48};
function msg(type,extra={}){window.parent.postMessage(Object.assign({isStreamlitMessage:true,type},extra),'*')}
function ready(){msg('streamlit:componentReady',{apiVersion:1})}
function setHeight(h){msg('streamlit:setFrameHeight',{height:h})}
function commit(){ if(A.start_s==null||A.end_s==null||A.end_s<=A.start_s)return; msg('streamlit:setComponentValue',{value:{start_s:A.start_s,end_s:A.end_s,duration_s:A.end_s-A.start_s,record_name:A.record_name,editor_token:A.editor_token,nonce:Date.now()},dataType:'json'}); }
function mmss(sec){sec=Math.max(0,Math.round(sec||0));let m=Math.floor(sec/60),s=sec%60;return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0')}
function xToS(x){let w=canvas.clientWidth-pad.l-pad.r;return Math.max(0,Math.min(A.duration_s,(x-pad.l)/Math.max(1,w)*A.duration_s))}
function sToX(s){let w=canvas.clientWidth-pad.l-pad.r;return pad.l+(s/A.duration_s)*w}
function yRange(){let v=A.values.filter(Number.isFinite); if(!v.length)return [0,1];let lo=Math.min(...v),hi=Math.max(...v);let d=Math.max(10,(hi-lo)*.08);return [lo-d,hi+d]}
function yToPx(v,yr){let h=canvas.clientHeight-pad.t-pad.b;return pad.t+(yr[1]-v)/(yr[1]-yr[0])*h}
function resize(){let dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=Math.max(1,Math.floor(w*dpr));canvas.height=Math.max(1,Math.floor(h*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
function draw(){let w=canvas.clientWidth,h=canvas.clientHeight;if(w<10||h<10)return;ctx.clearRect(0,0,w,h);let yr=yRange();
  // plot background/grid
  ctx.fillStyle='#0e1117';ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(148,163,184,.18)';ctx.lineWidth=1;
  for(let i=0;i<=5;i++){let y=pad.t+i*(h-pad.t-pad.b)/5;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();let val=yr[1]-i*(yr[1]-yr[0])/5;ctx.fillStyle='#aab4c4';ctx.font='11px system-ui';ctx.textAlign='right';ctx.fillText(Math.round(val),pad.l-8,y+4)}
  for(let i=0;i<=6;i++){let x=pad.l+i*(w-pad.l-pad.r)/6;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,h-pad.b);ctx.stroke();ctx.fillStyle='#aab4c4';ctx.textAlign='center';ctx.fillText((A.duration_s*i/360).toFixed(1),x,h-pad.b+18)}
  ctx.fillStyle='#aab4c4';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText('Tiempo acumulado (min)',(pad.l+w-pad.r)/2,h-8);
  ctx.save();ctx.translate(14,(pad.t+h-pad.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText('RRi (ms)',0,0);ctx.restore();
  // signal
  if(A.times.length&&A.values.length){ctx.strokeStyle='#2ea8ff';ctx.lineWidth=1.6;ctx.beginPath();let started=false;for(let i=0;i<Math.min(A.times.length,A.values.length);i++){let vx=A.values[i];if(!Number.isFinite(vx))continue;let x=sToX(A.times[i]),y=yToPx(vx,yr);if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)}ctx.stroke()}
  // artifacts
  ctx.strokeStyle='#ff9aa2';ctx.lineWidth=1.3;for(let i=0;i<Math.min(A.art_times.length,A.art_values.length);i++){let v=A.art_values[i];if(!Number.isFinite(v))continue;let x=sToX(A.art_times[i]),y=yToPx(v,yr);ctx.beginPath();ctx.moveTo(x-3,y-3);ctx.lineTo(x+3,y+3);ctx.moveTo(x+3,y-3);ctx.lineTo(x-3,y+3);ctx.stroke()}
  // selection
  if(A.start_s!=null&&A.end_s!=null){let x0=sToX(A.start_s),x1=sToX(A.end_s),dur=A.end_s-A.start_s;let near=Math.abs(dur-A.target_s)<=3;ctx.fillStyle=near?'rgba(46,204,113,.22)':'rgba(255,215,0,.20)';ctx.fillRect(x0,pad.t,x1-x0,h-pad.t-pad.b);ctx.strokeStyle=near?'#2ecc71':'#ffd700';ctx.lineWidth=2;ctx.setLineDash([8,6]);ctx.strokeRect(x0,pad.t,x1-x0,h-pad.t-pad.b);ctx.setLineDash([]);ctx.fillStyle=near?'#2ecc71':'#ffd700';ctx.fillRect(x0-3,pad.t,6,h-pad.t-pad.b);ctx.fillRect(x1-3,pad.t,6,h-pad.t-pad.b);hud.textContent=mmss(dur);hud.style.color=near?'#63e68b':'#ffe66d';meta.textContent=mmss(A.start_s)+' → '+mmss(A.end_s)+'  ·  '+A.record_name}else{hud.textContent='00:00';hud.style.color='#e8edf5';meta.textContent=A.record_name}
  target.textContent='Objetivo '+mmss(A.target_s)+' · doble clic = '+mmss(A.target_s);
}
function pointerX(e){let r=canvas.getBoundingClientRect();return e.clientX-r.left}
canvas.addEventListener('pointerdown',e=>{e.preventDefault();pointerId=e.pointerId;canvas.setPointerCapture(pointerId);let s=xToS(pointerX(e));let x=pointerX(e);let x0=A.start_s==null?-999:sToX(A.start_s),x1=A.end_s==null?-999:sToX(A.end_s);if(A.start_s!=null&&Math.abs(x-x0)<12){mode='left'}else if(A.end_s!=null&&Math.abs(x-x1)<12){mode='right'}else if(A.start_s!=null&&s>=A.start_s&&s<=A.end_s){mode='move';dragOffset=s-A.start_s;originalDur=A.end_s-A.start_s}else{mode='new';anchor=s;A.start_s=s;A.end_s=s}draw()});
canvas.addEventListener('pointermove',e=>{if(!mode)return;e.preventDefault();let s=xToS(pointerX(e));if(mode==='new'){A.start_s=Math.min(anchor,s);A.end_s=Math.max(anchor,s)}else if(mode==='left'){A.start_s=Math.min(s,A.end_s-.25)}else if(mode==='right'){A.end_s=Math.max(s,A.start_s+.25)}else if(mode==='move'){let ns=s-dragOffset;ns=Math.max(0,Math.min(A.duration_s-originalDur,ns));A.start_s=ns;A.end_s=ns+originalDur}draw()});
function up(e){if(!mode)return;e.preventDefault();mode=null;try{canvas.releasePointerCapture(pointerId)}catch(_e){}pointerId=null;draw();commit()}
canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',up);
canvas.addEventListener('dblclick',e=>{e.preventDefault();let c=xToS(pointerX(e)),d=Math.min(A.target_s,A.duration_s),s=Math.max(0,Math.min(A.duration_s-d,c-d/2));A.start_s=s;A.end_s=s+d;draw();commit()});
window.addEventListener('resize',resize);
window.addEventListener('message',ev=>{let d=ev.data||{};if(d.type!=='streamlit:render')return;let a=d.args||{};A.times=a.times||[];A.values=a.values||[];A.art_times=a.art_times||[];A.art_values=a.art_values||[];A.duration_s=Math.max(.1,Number(a.duration_s)||1);A.start_s=(a.start_s===null||a.start_s===undefined)?null:Number(a.start_s);A.end_s=(a.end_s===null||a.end_s===undefined)?null:Number(a.end_s);A.target_s=Number(a.target_s)||300;A.height=Math.max(420,Number(a.height)||620);A.record_name=String(a.record_name||'');A.editor_token=String(a.editor_token||'');document.getElementById('wrap').style.height=A.height+'px';setHeight(A.height);requestAnimationFrame(resize)});
ready();setHeight(620);
</script></body></html>"""
_WINDOW_EDITOR_INDEX = _WINDOW_EDITOR_DIR / "index.html"
try:
    if (not _WINDOW_EDITOR_INDEX.exists()) or (_WINDOW_EDITOR_INDEX.read_text(encoding="utf-8") != _WINDOW_EDITOR_HTML):
        _WINDOW_EDITOR_INDEX.write_text(_WINDOW_EDITOR_HTML, encoding="utf-8")
except Exception:
    pass
_window_editor_component_v1533 = components.declare_component("vrc_window_editor_v1533", path=str(_WINDOW_EDITOR_DIR))


def window_editor_v1533(record_name, data, pending_selection=None, target_s=300.0, height=620, max_points=1400, editor_token="", key=None):
    """Editor temporal sin rerun durante el arrastre; devuelve la ventana sólo al soltar."""
    rr = np.asarray(data.get("rr", []), dtype=float)
    if rr.size == 0:
        return None
    t = cumulative_time(rr)
    tx, vy = _downsample_xy_v15213(t, rr * 1000.0, max_points)
    art_t, art_v = [], []
    mask = np.asarray(data.get("artifact_mask", []), dtype=bool)
    rr_raw = np.asarray(data.get("rr_raw", []), dtype=float)
    if rr_raw.size and mask.size == rr_raw.size and np.any(mask):
        tr = cumulative_time(rr_raw)
        ax, ay = tr[mask], rr_raw[mask] * 1000.0
        ax, ay = _downsample_xy_v15213(ax, ay, min(500, max_points))
        art_t, art_v = ax.tolist(), ay.tolist()
    s0 = e0 = None
    if pending_selection is not None and len(pending_selection) == 2:
        try:
            s0, e0 = float(pending_selection[0]), float(pending_selection[1])
        except Exception:
            s0 = e0 = None
    return _window_editor_component_v1533(
        times=np.asarray(tx, dtype=float).tolist(),
        values=np.asarray(vy, dtype=float).tolist(),
        art_times=art_t,
        art_values=art_v,
        duration_s=float(data.get("duration", np.nansum(rr))),
        start_s=s0,
        end_s=e0,
        target_s=float(target_s),
        height=int(height),
        record_name=str(record_name),
        editor_token=str(editor_token or ""),
        default=None,
        key=key,
    )

LONGITUDINAL_DB_PATH = APP_DATA_DIR / 'vrc_longitudinal.sqlite3' 
AUTO_GB_MODEL_PATH = APP_DATA_DIR / 'modelo_estados_fisiologicos_v140.joblib'

# ============================================================
# v15.4.1 · PERSISTENT CLOUD DATABASE – LAZY SYNC
# ============================================================
# La SQLite local se conserva como caché de ejecución para no alterar ninguna
# consulta/cálculo existente. La fuente persistente es PostgreSQL externo
# (p. ej. Supabase). Se almacena una imagen íntegra y versionada de la SQLite
# en PostgreSQL y se restaura automáticamente cuando nace una instancia nueva.
CLOUD_STORE_KEY_V1540 = 'physiosentinel_longitudinal.sqlite3'
CLOUD_MODEL_KEY_V1540 = 'physiosentinel_motor_hibrido_v1550_xgboost_horizonte_activo.joblib'
CLOUD_PROBABILITY_KEY_V1552 = 'physiosentinel_probability_v1552_xgboost_calibrated.joblib'
CLOUD_TABLE_V1540 = 'physiosentinel_v2_objects'
CLOUD_BACKUP_TABLE_V1540 = 'physiosentinel_v2_backups'


def _persistent_db_url_v1540():
    """Obtiene la URL PostgreSQL sin exponerla ni escribirla en logs."""
    candidates = []
    try:
        cfg = st.secrets.get('persistent_db', {})
        if cfg:
            candidates.extend([cfg.get('url'), cfg.get('database_url'), cfg.get('dsn')])
        candidates.extend([
            st.secrets.get('PHYSIOSENTINEL_DATABASE_URL'),
            st.secrets.get('DATABASE_URL'),
        ])
    except Exception:
        pass
    candidates.extend([
        os.environ.get('PHYSIOSENTINEL_DATABASE_URL'),
        os.environ.get('DATABASE_URL'),
    ])
    for value in candidates:
        if value and str(value).strip().lower().startswith(('postgres://', 'postgresql://')):
            return str(value).strip()
    return None


def _cloud_state_v1540():
    try:
        return st.session_state.setdefault('_persistent_cloud_state_v1540', {
            'checked': False, 'configured': False, 'connected': False,
            'last_error': '', 'last_sync': None, 'last_reason': '',
            'base_sha': None, 'remote_found': False,
            'last_local_sha': None, 'dirty': False, 'dirty_reasons': [],
            'cloud_reads': 0, 'cloud_writes': 0,
        })
    except Exception:
        if not hasattr(_cloud_state_v1540, '_state'):
            _cloud_state_v1540._state = {'checked': False, 'configured': False, 'connected': False,
                'last_error': '', 'last_sync': None, 'last_reason': '', 'base_sha': None, 'remote_found': False,
                'last_local_sha': None, 'dirty': False, 'dirty_reasons': [], 'cloud_reads': 0, 'cloud_writes': 0}
        return _cloud_state_v1540._state


def _cloud_connect_v1540():
    url = _persistent_db_url_v1540()
    state = _cloud_state_v1540()
    state['configured'] = bool(url)
    if not url:
        raise RuntimeError('No hay PHYSIOSENTINEL_DATABASE_URL configurada.')
    if not PSYCOPG_AVAILABLE:
        raise RuntimeError('Falta psycopg. Instala psycopg[binary].')
    return psycopg.connect(url, connect_timeout=8, autocommit=True)


def _cloud_init_schema_v1540(con):
    con.execute(f'''CREATE TABLE IF NOT EXISTS {CLOUD_TABLE_V1540} (
        object_key TEXT PRIMARY KEY,
        app_version TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        sha256 TEXT NOT NULL,
        size_bytes BIGINT NOT NULL,
        payload BYTEA NOT NULL,
        reason TEXT,
        source_instance TEXT
    )''')
    con.execute(f'''CREATE TABLE IF NOT EXISTS {CLOUD_BACKUP_TABLE_V1540} (
        backup_id TEXT PRIMARY KEY,
        object_key TEXT NOT NULL,
        app_version TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        sha256 TEXT NOT NULL,
        size_bytes BIGINT NOT NULL,
        payload BYTEA NOT NULL,
        reason TEXT,
        source_instance TEXT
    )''')
    con.execute(f'CREATE INDEX IF NOT EXISTS idx_ps_backup_key_date ON {CLOUD_BACKUP_TABLE_V1540}(object_key, created_at DESC)')


def _instance_id_v1540():
    raw = '|'.join([os.environ.get('HOSTNAME',''), str(os.getpid())])
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


def _cloud_get_object_v1540(object_key):
    state = _cloud_state_v1540()
    try:
        with _cloud_connect_v1540() as con:
            _cloud_init_schema_v1540(con)
            row = con.execute(
                f'SELECT sha256, payload, updated_at, size_bytes FROM {CLOUD_TABLE_V1540} WHERE object_key=%s',
                (str(object_key),)
            ).fetchone()
        state['connected'] = True; state['last_error'] = ''
        state['cloud_reads'] = int(state.get('cloud_reads', 0)) + 1
        if not row:
            return None
        return {'sha256': row[0], 'payload': bytes(row[1]), 'updated_at': row[2], 'size_bytes': int(row[3] or 0)}
    except Exception as exc:
        state['connected'] = False; state['last_error'] = str(exc)
        return None


def _cloud_put_bytes_v1540(object_key, payload, reason='', base_sha=None, force=False):
    """v15.5.0 · Persistencia V2 sin crecimiento explosivo.

    Mantiene una sola fila viva por objeto y una retención rotatoria estricta.
    SQLite: como máximo 1 backup automático cada 6 h. Todos los objetos: máximo 5 backups.
    """
    state = _cloud_state_v1540()
    payload = bytes(payload or b'')
    if not payload:
        return False, 'payload vacío'
    sha = hashlib.sha256(payload).hexdigest()
    object_key = str(object_key)
    reason = str(reason or '')
    make_backup = False
    try:
        with _cloud_connect_v1540() as con:
            _cloud_init_schema_v1540(con)
            current = con.execute(
                f'SELECT sha256 FROM {CLOUD_TABLE_V1540} WHERE object_key=%s', (object_key,)
            ).fetchone()
            current_sha = current[0] if current else None
            if (not force and current_sha and base_sha and current_sha != base_sha and current_sha != sha):
                msg = 'Conflicto de sincronización: la nube cambió desde la última descarga. Recarga antes de sobrescribir.'
                state['last_error'] = msg
                return False, msg
            if current_sha == sha:
                state['connected'] = True; state['last_error'] = ''; state['base_sha'] = sha
                state['last_sync'] = datetime.now(timezone.utc).isoformat(); state['last_reason'] = reason
                return True, 'sin cambios'

            con.execute(f"""INSERT INTO {CLOUD_TABLE_V1540}
                (object_key,app_version,updated_at,sha256,size_bytes,payload,reason,source_instance)
                VALUES (%s,%s,NOW(),%s,%s,%s,%s,%s)
                ON CONFLICT(object_key) DO UPDATE SET
                    app_version=EXCLUDED.app_version, updated_at=NOW(), sha256=EXCLUDED.sha256,
                    size_bytes=EXCLUDED.size_bytes, payload=EXCLUDED.payload,
                    reason=EXCLUDED.reason, source_instance=EXCLUDED.source_instance""",
                (object_key, '15.5.5', sha, len(payload), payload, reason, _instance_id_v1540()))

            last_b = con.execute(
                f"""SELECT sha256, created_at FROM {CLOUD_BACKUP_TABLE_V1540}
                    WHERE object_key=%s ORDER BY created_at DESC LIMIT 1""", (object_key,)
            ).fetchone()
            if not last_b:
                make_backup = True
            elif str(last_b[0]) != sha:
                if force or object_key != CLOUD_STORE_KEY_V1540:
                    make_backup = True
                else:
                    try:
                        age_s = (datetime.now(timezone.utc) - last_b[1]).total_seconds()
                    except Exception:
                        age_s = 21600
                    make_backup = age_s >= 21600
            if make_backup:
                stamp = datetime.now(timezone.utc).isoformat()
                backup_id = hashlib.sha256(f'{object_key}|{sha}|{stamp}'.encode('utf-8')).hexdigest()
                con.execute(f"""INSERT INTO {CLOUD_BACKUP_TABLE_V1540}
                    (backup_id,object_key,app_version,sha256,size_bytes,payload,reason,source_instance)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (backup_id, object_key, '15.5.5', sha, len(payload), payload, reason, _instance_id_v1540()))
                con.execute(f"""DELETE FROM {CLOUD_BACKUP_TABLE_V1540}
                    WHERE object_key=%s AND backup_id NOT IN (
                        SELECT backup_id FROM {CLOUD_BACKUP_TABLE_V1540}
                        WHERE object_key=%s ORDER BY created_at DESC LIMIT 5
                    )""", (object_key, object_key))

        state['connected'] = True; state['last_error'] = ''; state['base_sha'] = sha
        state['last_sync'] = datetime.now(timezone.utc).isoformat(); state['last_reason'] = reason
        state['cloud_writes'] = int(state.get('cloud_writes', 0)) + 1
        return True, ('sincronizado + backup rotatorio' if make_backup else 'sincronizado sin nuevo backup')
    except Exception as exc:
        state['connected'] = False; state['last_error'] = str(exc)
        return False, str(exc)


def _cloud_restore_once_v1540():
    """Restaura la SQLite remota antes de que la instancia cree una base local vacía."""
    state = _cloud_state_v1540()
    if state.get('checked'):
        return state.get('remote_found', False)
    state['checked'] = True
    if not _persistent_db_url_v1540():
        state['configured'] = False
        return False
    remote = _cloud_get_object_v1540(CLOUD_STORE_KEY_V1540)
    if remote is None:
        state['remote_found'] = False
        return False
    try:
        raw = remote['payload']
        if not raw.startswith(b'SQLite format 3\x00'):
            raise ValueError('La copia persistente no contiene una SQLite válida.')
        tmp = LONGITUDINAL_DB_PATH.with_suffix('.cloud.tmp')
        tmp.write_bytes(raw)
        with sqlite3.connect(tmp) as con:
            chk = con.execute('PRAGMA quick_check').fetchone()
            if not chk or str(chk[0]).lower() != 'ok':
                raise ValueError(f'quick_check={chk}')
        tmp.replace(LONGITUDINAL_DB_PATH)
        state['remote_found'] = True; state['base_sha'] = remote['sha256']; state['last_local_sha'] = remote['sha256']
        state['dirty'] = False; state['dirty_reasons'] = []
        state['connected'] = True; state['last_error'] = ''
        state['last_sync'] = datetime.now(timezone.utc).isoformat(); state['last_reason'] = 'restore_startup'
        return True
    except Exception as exc:
        state['last_error'] = str(exc); state['connected'] = False
        return False


def _cloud_mark_dirty_v1541(reason='data_change'):
    """Marca cambios locales sin hacer ninguna consulta de red."""
    state = _cloud_state_v1540()
    state['dirty'] = True
    reasons = list(state.get('dirty_reasons') or [])
    reason = str(reason or 'data_change')
    if reason not in reasons:
        reasons.append(reason)
    state['dirty_reasons'] = reasons[-12:]
    return True


def _cloud_checkpoint_v1540(reason='data_change', min_interval_s=300, force=False):
    """Lazy Sync v15.4.1: solo toca PostgreSQL si la SQLite cambió realmente.

    Primero calcula SHA-256 local. Si coincide con la última imagen confirmada en nube,
    no abre conexión. Los reruns de Streamlit por navegación quedan así fuera de la red.
    """
    if not _persistent_db_url_v1540() or not LONGITUDINAL_DB_PATH.exists():
        return False
    state = _cloud_state_v1540()
    try:
        with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
            con.execute('PRAGMA wal_checkpoint(FULL)')
            con.commit()
    except Exception:
        pass
    try:
        raw = LONGITUDINAL_DB_PATH.read_bytes()
    except Exception as exc:
        state['last_error'] = str(exc)
        return False
    local_sha = hashlib.sha256(raw).hexdigest()
    state['last_local_sha'] = local_sha
    if not force and state.get('base_sha') and local_sha == state.get('base_sha'):
        state['dirty'] = False; state['dirty_reasons'] = []
        return True
    effective_interval_s = max(300.0, float(min_interval_s or 0)) if not force else 0.0
    if effective_interval_s and state.get('last_sync') and not force:
        try:
            elapsed = (datetime.now(timezone.utc) - pd.Timestamp(state['last_sync']).to_pydatetime()).total_seconds()
            if elapsed < effective_interval_s:
                _cloud_mark_dirty_v1541(reason)
                return True
        except Exception:
            pass
    ok, _ = _cloud_put_bytes_v1540(
        CLOUD_STORE_KEY_V1540, raw, reason=reason,
        base_sha=state.get('base_sha'), force=force
    )
    if ok:
        state['last_local_sha'] = local_sha
    else:
        _cloud_mark_dirty_v1541(reason)
    return bool(ok)

def _cloud_seed_if_empty_v1540():
    """Si se configura PostgreSQL por primera vez, migra automáticamente la SQLite existente."""
    state = _cloud_state_v1540()
    if not _persistent_db_url_v1540() or state.get('remote_found'):
        return
    remote = _cloud_get_object_v1540(CLOUD_STORE_KEY_V1540)
    if remote:
        state['remote_found'] = True; state['base_sha'] = remote['sha256']
        return
    if LONGITUDINAL_DB_PATH.exists() and LONGITUDINAL_DB_PATH.stat().st_size > 0:
        _cloud_checkpoint_v1540('initial_migration_v1540', force=True)
        state['remote_found'] = True


def _cloud_sync_status_v1540():
    state = _cloud_state_v1540()
    return {
        'configured': bool(_persistent_db_url_v1540()),
        'connected': bool(state.get('connected')),
        'last_sync': state.get('last_sync'),
        'last_reason': state.get('last_reason'),
        'error': state.get('last_error',''),
        'remote_found': bool(state.get('remote_found')),
        'dirty': bool(state.get('dirty')),
        'dirty_reasons': list(state.get('dirty_reasons') or []),
        'cloud_reads': int(state.get('cloud_reads', 0)),
        'cloud_writes': int(state.get('cloud_writes', 0)),
    }


# v15.4.0 · seguridad de persistencia en Streamlit Cloud. El disco local de la
# instancia puede reiniciarse en redeploy/restart; por eso se permite restaurar
# explícitamente una copia SQLite descargada previamente.
def _restore_longitudinal_db_v15326(uploaded_file):
    if uploaded_file is None:
        return False, 'Sin archivo'
    try:
        raw = uploaded_file.getvalue()
        if not raw or not raw.startswith(b'SQLite format 3\x00'):
            return False, 'El archivo no parece ser una base SQLite válida.'
        tmp = LONGITUDINAL_DB_PATH.with_suffix('.restore.tmp')
        tmp.write_bytes(raw)
        with sqlite3.connect(tmp) as con:
            con.execute('PRAGMA quick_check').fetchone()
        tmp.replace(LONGITUDINAL_DB_PATH)
        _cloud_checkpoint_v1540('manual_sqlite_restore', force=True)
        return True, 'Base longitudinal restaurada correctamente y sincronizada con la nube.'
    except Exception as exc:
        return False, f'No se pudo restaurar la base: {exc}'


def normalize_patient_id(value):
    """Normaliza el identificador para que Ali, ali y ALI pertenezcan a la misma serie."""
    txt = sanitize_name(str(value or '')).strip('_-').lower()
    return txt or 'paciente_sin_identificar'


def infer_patient_id(record_name):
    """Infere una serie/paciente estable, ignorando etiquetas pre/post y la fecha-hora."""
    stem = sanitize_name(record_name)
    # Quita la fecha y todo lo posterior.
    stem = re.sub(r'[_-]?(?:20)?\d{2,4}[-_]\d{1,2}[-_]\d{1,2}.*$', '', stem)
    stem = re.sub(r'[_-]?\d{8}(?:[_-]?\d{6})?.*$', '', stem)
    # PRE/POST describen el momento del registro, no pacientes distintos.
    stem = re.sub(r'(?:^|[_-])(pre|post)(?:$|[_-])', '_', stem, flags=re.IGNORECASE)
    stem = re.sub(r'[_-]+', '_', stem).strip('_-')
    patient = normalize_patient_id(stem)
    # Alias tipográfico observado en los registros de Antonio Genovart.
    aliases = {
        'antoni_genovart': 'antonio_genovart',
    }
    return aliases.get(patient, patient)


def init_longitudinal_db():
    """Inicializa la caché SQLite y la enlaza con PostgreSQL persistente v15.4.1 Lazy Sync."""
    _cloud_restore_once_v1540()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                observation_key TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                record_name TEXT NOT NULL,
                phase TEXT NOT NULL,
                record_datetime TEXT,
                saved_at TEXT NOT NULL,
                IDX_Vagal REAL,
                IDX_Amplitud REAL,
                IDX_Complejidad REAL,
                IDX_Rigidez REAL,
                IDX_Adaptabilidad REAL,
                IDX_Regulacion_Lenta REAL,
                IDX_Confianza REAL,
                Perfil_autonomico TEXT,
                HVG_degree_mean REAL,
                HVG_degree_max REAL,
                HVG_hubs_p90 REAL,
                HVG_clustering REAL,
                HVG_lambda REAL,
                HVG_path_length REAL,
                HVG_diameter REAL,
                HVG_compactness_index REAL,
                HVG_graph_score_small_world REAL,
                HVG_graph_score_scale_free REAL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS raw_records (
                record_key TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                record_name TEXT NOT NULL,
                filename TEXT,
                record_datetime TEXT,
                saved_at TEXT NOT NULL,
                duration_s REAL,
                n_rri INTEGER,
                rr_raw_npz BLOB NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS segmentations (
                record_key TEXT NOT NULL,
                record_name TEXT NOT NULL,
                phase TEXT NOT NULL,
                start_s REAL NOT NULL,
                end_s REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (record_key, phase)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS temporal_selections (
                record_key TEXT PRIMARY KEY,
                record_name TEXT NOT NULL,
                filename TEXT,
                start_s REAL NOT NULL,
                end_s REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS analysis_snapshots (
                record_key TEXT PRIMARY KEY,
                record_name TEXT NOT NULL,
                app_version TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                results_json TEXT NOT NULL,
                rr_corrected_npz BLOB,
                artifact_mask_npz BLOB
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS patient_analysis_bundles (
                bundle_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                app_version TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                record_keys_json TEXT NOT NULL,
                record_count INTEGER NOT NULL DEFAULT 0,
                config_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        # Migración compatible desde bases anteriores.
        _obs_cols = {r[1] for r in con.execute('PRAGMA table_info(observations)').fetchall()}
        _hvg_sql_cols = {
            'HVG_degree_mean':'REAL', 'HVG_degree_max':'REAL', 'HVG_hubs_p90':'REAL',
            'HVG_clustering':'REAL', 'HVG_lambda':'REAL', 'HVG_path_length':'REAL',
            'HVG_diameter':'REAL', 'HVG_compactness_index':'REAL',
            'HVG_graph_score_small_world':'REAL', 'HVG_graph_score_scale_free':'REAL'
        }
        for _name, _typ in _hvg_sql_cols.items():
            if _name not in _obs_cols:
                con.execute(f'ALTER TABLE observations ADD COLUMN {_name} {_typ}')
        _primary_sql_cols = {c:'REAL' for c in [
            'SDNN','RMSSD','SD1','SD2','HF','LF','VLF','MeanHR','DFA_alpha1','DFA_alpha2','SampEn','D2','Hurst','Lyapunov_LLE',
            'REC','DET','Lmean','Lmax','VLF_WAV_MEAN','LF_WAV_MEAN','HF_WAV_MEAN','VLF_DOM_PCT','LF_DOM_PCT','HF_DOM_PCT',
            'WAV_ENTROPY_BANDS','WAV_ENTROPY_GLOBAL','N_RRi','Duration_s','Artefactos_pct'
        ]}
        _obs_cols = {r[1] for r in con.execute('PRAGMA table_info(observations)').fetchall()}
        for _name,_typ in _primary_sql_cols.items():
            if _name not in _obs_cols:
                con.execute(f'ALTER TABLE observations ADD COLUMN {_name} {_typ}')
        con.execute('CREATE INDEX IF NOT EXISTS idx_obs_patient_phase_date ON observations(patient_id, phase, record_datetime)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_raw_patient_date ON raw_records(patient_id, record_datetime)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_seg_record ON segmentations(record_key, phase)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_temp_selection_record ON temporal_selections(record_key)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_snapshot_record ON analysis_snapshots(record_key)')
        con.execute('CREATE INDEX IF NOT EXISTS idx_bundle_patient ON patient_analysis_bundles(patient_id, saved_at)')
        con.commit()
    _cloud_seed_if_empty_v1540()


def _raw_record_key(record_name, filename=''):
    return hashlib.sha256(f"{str(record_name).strip()}|{str(filename).strip()}".encode('utf-8')).hexdigest()


def _encode_rr_array(rr_raw):
    bio = io.BytesIO()
    np.savez_compressed(bio, rr=np.asarray(rr_raw, dtype=np.float64))
    return bio.getvalue()


def _decode_rr_array(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return np.asarray(z['rr'], dtype=float)


def save_raw_record_to_db(record_name, filename, rr_raw):
    """Guarda el RRi solo si es nuevo o cambió; la repetición de un rerun no escribe."""
    init_longitudinal_db()
    record_name = sanitize_name(record_name)
    filename = str(filename or record_name)
    patient = infer_patient_id(record_name)
    dt = _extract_record_datetime(record_name)
    dt_text = dt.isoformat() if pd.notna(dt) else None
    rr_arr = np.asarray(rr_raw, dtype=float)
    key = _raw_record_key(record_name, filename)
    rr_blob = _encode_rr_array(rr_arr)
    duration_s = float(np.nansum(rr_arr)); n_rri = int(len(rr_arr))
    changed = False
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        prev = con.execute(
            "SELECT patient_id,record_name,filename,record_datetime,duration_s,n_rri,rr_raw_npz FROM raw_records WHERE record_key=?",
            (key,),
        ).fetchone()
        same = False
        if prev:
            try:
                same = (
                    str(prev[0] or '') == patient and str(prev[1] or '') == record_name and
                    str(prev[2] or '') == filename and str(prev[3] or '') == str(dt_text or '') and
                    abs(float(prev[4] or 0.0)-duration_s) < 1e-9 and int(prev[5] or 0) == n_rri and
                    bytes(prev[6] or b'') == bytes(rr_blob)
                )
            except Exception:
                same = False
        if not same:
            now = datetime.now(timezone.utc).isoformat()
            con.execute("""
                INSERT INTO raw_records (
                    record_key, patient_id, record_name, filename, record_datetime,
                    saved_at, duration_s, n_rri, rr_raw_npz
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(record_key) DO UPDATE SET
                    patient_id=excluded.patient_id, record_name=excluded.record_name,
                    filename=excluded.filename, record_datetime=excluded.record_datetime,
                    saved_at=excluded.saved_at, duration_s=excluded.duration_s,
                    n_rri=excluded.n_rri, rr_raw_npz=excluded.rr_raw_npz
            """, (key, patient, record_name, filename, dt_text, now,
                  duration_s, n_rri, sqlite3.Binary(rr_blob)))
            con.commit(); changed = True
    if changed:
        _cloud_checkpoint_v1540('raw_record_saved')
    return key

def list_saved_raw_records():
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        return pd.read_sql_query("""
            SELECT record_key, patient_id, record_name, filename, record_datetime,
                   saved_at, duration_s, n_rri
            FROM raw_records
            WHERE patient_id NOT LIKE '__trash__:%'
            ORDER BY COALESCE(record_datetime, saved_at), saved_at
        """, con)


def load_saved_raw_records(record_keys=None):
    init_longitudinal_db()
    query = 'SELECT * FROM raw_records'
    params = []
    if record_keys is not None:
        keys = list(record_keys)
        if not keys:
            return {}
        query += ' WHERE record_key IN (' + ','.join(['?'] * len(keys)) + ')'
        params.extend(keys)
    query += ' ORDER BY COALESCE(record_datetime, saved_at), saved_at'
    out = {}
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        cur = con.execute(query, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    for vals in rows:
        row = dict(zip(cols, vals))
        try:
            row['rr_raw'] = _decode_rr_array(row.pop('rr_raw_npz'))
            out[row['record_key']] = row
        except Exception:
            continue
    return out


def save_record_segmentation(record_name, filename, windows):
    init_longitudinal_db()
    key = _raw_record_key(record_name, filename)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        con.execute('DELETE FROM segmentations WHERE record_key=?', (key,))
        for phase in PHASES:
            win = (windows or {}).get(phase)
            if win is None or len(win) != 2:
                continue
            try:
                start_s, end_s = float(win[0]), float(win[1])
            except Exception:
                continue
            if not (np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s):
                continue
            con.execute("""
                INSERT INTO segmentations (
                    record_key, record_name, phase, start_s, end_s, active, updated_at
                ) VALUES (?,?,?,?,?,?,?)
            """, (key, record_name, phase, start_s, end_s, 1, now))
        con.commit()
    _cloud_checkpoint_v1540('segmentation_saved')


def load_record_segmentation(record_name, filename):
    init_longitudinal_db()
    key = _raw_record_key(record_name, filename)
    windows = empty_windows()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        rows = con.execute("""
            SELECT phase, start_s, end_s FROM segmentations
            WHERE record_key=? ORDER BY phase
        """, (key,)).fetchall()
    for phase, start_s, end_s in rows:
        if phase in windows:
            windows[phase] = [float(start_s), float(end_s)]
    return windows


def save_temporal_selection(record_name, filename, start_s, end_s):
    """Guarda inmediatamente el último tramo arrastrado, incluso antes de asignarlo a una fase."""
    init_longitudinal_db()
    start_s, end_s = float(start_s), float(end_s)
    if not (np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s):
        return
    key = _raw_record_key(record_name, filename)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        con.execute("""
            INSERT INTO temporal_selections (record_key, record_name, filename, start_s, end_s, updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(record_key) DO UPDATE SET
                record_name=excluded.record_name,
                filename=excluded.filename,
                start_s=excluded.start_s,
                end_s=excluded.end_s,
                updated_at=excluded.updated_at
        """, (key, sanitize_name(record_name), str(filename or record_name), start_s, end_s, now))
        con.commit()
    _cloud_mark_dirty_v1541('temporal_selection')


def load_temporal_selection(record_name, filename):
    """Recupera el último tramo arrastrado para el registro."""
    init_longitudinal_db()
    key = _raw_record_key(record_name, filename)
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        row = con.execute(
            'SELECT start_s, end_s FROM temporal_selections WHERE record_key=?',
            (key,),
        ).fetchone()
    if not row:
        return None
    start_s, end_s = float(row[0]), float(row[1])
    return [start_s, end_s] if end_s > start_s else None


def clear_temporal_selection(record_name, filename):
    init_longitudinal_db()
    key = _raw_record_key(record_name, filename)
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        con.execute('DELETE FROM temporal_selections WHERE record_key=?', (key,))
        con.commit()
    _cloud_mark_dirty_v1541('temporal_selection_cleared')


def _build_attractor_snapshot_config(rr_corrected):
    """Captura el estado del atractor sin alterar su algoritmo de cálculo."""
    try:
        rr = np.asarray(rr_corrected, dtype=float)
        rr = rr[np.isfinite(rr)]
    except Exception:
        rr = np.asarray([], dtype=float)
    tau_mode = st.session_state.get("attractor_tau_mode_v15320", "Automático por registro")
    try:
        m = int(st.session_state.get("attractor_m_v15320", 3))
    except Exception:
        m = 3
    try:
        tau_common = int(st.session_state.get("attractor_tau_common_v15320", 1))
    except Exception:
        tau_common = 1
    tau = None
    if len(rr) >= 10:
        try:
            tau = int(_attractor_delay(rr)) if tau_mode == "Automático por registro" else int(tau_common)
            tau = max(1, min(tau, max(1, len(rr)//max(3, m))))
        except Exception:
            tau = None
    try:
        rr_hash = hashlib.sha256(np.ascontiguousarray(rr, dtype=np.float64).tobytes()).hexdigest() if len(rr) else None
    except Exception:
        rr_hash = None
    return {
        "schema": "attractor_snapshot_v1",
        "saved_with_app": "15.3.28",
        "tau_mode": tau_mode,
        "tau": tau,
        "m": m,
        "tau_common": tau_common,
        "rr_count": int(len(rr)),
        "rr_sha256": rr_hash,
    }


def save_analysis_snapshot(record_key, record_name, metrics_df, rr_corrected, artifact_mask, config):
    """Guarda el análisis ya calculado. Al recuperarlo no se vuelven a aplicar filtros ni métricas."""
    init_longitudinal_db()
    if metrics_df is None or metrics_df.empty:
        return False
    now = datetime.now(timezone.utc).isoformat()
    results_json = metrics_df.to_json(orient='split', date_format='iso', double_precision=15)
    config_json = json.dumps(config, ensure_ascii=False, default=str)
    rr_blob = _encode_rr_array(np.asarray(rr_corrected, dtype=float))
    mask_blob = _encode_rr_array(np.asarray(artifact_mask, dtype=np.uint8))
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        con.execute("""
            INSERT INTO analysis_snapshots
            (record_key, record_name, app_version, saved_at, config_json, results_json, rr_corrected_npz, artifact_mask_npz)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(record_key) DO UPDATE SET
              record_name=excluded.record_name, app_version=excluded.app_version, saved_at=excluded.saved_at,
              config_json=excluded.config_json, results_json=excluded.results_json,
              rr_corrected_npz=excluded.rr_corrected_npz, artifact_mask_npz=excluded.artifact_mask_npz
        """, (record_key, record_name, '15.3.28', now, config_json, results_json,
              sqlite3.Binary(rr_blob), sqlite3.Binary(mask_blob)))
        con.commit()
    _cloud_checkpoint_v1540('analysis_snapshot_saved')
    return True


def load_analysis_snapshot(record_key):
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        row = con.execute("SELECT app_version,saved_at,config_json,results_json,rr_corrected_npz,artifact_mask_npz FROM analysis_snapshots WHERE record_key=?", (record_key,)).fetchone()
    if not row:
        return None
    try:
        return {
            'app_version': row[0], 'saved_at': row[1], 'config': json.loads(row[2]),
            'results': pd.read_json(io.StringIO(row[3]), orient='split'),
            'rr_corrected': _decode_rr_array(row[4]) if row[4] else None,
            'artifact_mask': _decode_rr_array(row[5]).astype(bool) if row[5] else None,
        }
    except Exception:
        return None



def _patient_bundle_id(patient_id):
    patient = normalize_patient_id(patient_id)
    return hashlib.sha256(f"patient_bundle|{patient}".encode("utf-8")).hexdigest()


def _canonical_patient_rows(catalog_df, patient_id):
    """Devuelve todos los RRi asignados al paciente.

    v15.4.0: respeta primero ``raw_records.patient_id`` para permitir añadir una visita
    a un histórico existente sin tener que volver a cargar los registros antiguos.
    Solo usa el nombre del archivo como compatibilidad con bases anteriores.
    """
    if catalog_df is None or catalog_df.empty:
        return pd.DataFrame(columns=[])
    patient = infer_patient_id(patient_id)
    if "patient_id" in catalog_df.columns:
        stored = catalog_df["patient_id"].fillna("").astype(str).map(infer_patient_id)
        inferred = catalog_df["record_name"].astype(str).map(infer_patient_id)
        mask = (stored == patient) | ((stored == "") & (inferred == patient))
    else:
        mask = catalog_df["record_name"].astype(str).map(lambda x: infer_patient_id(x) == patient)
    out = catalog_df.loc[mask].copy()
    if not out.empty:
        out = out.sort_values(["record_datetime", "saved_at"], na_position="last")
    return out


def assign_record_to_patient(record_key, patient_id):
    """Asigna un RRi ya persistido a un histórico longitudinal existente.

    No altera el nombre ni los datos RRi; únicamente fija la pertenencia longitudinal.
    """
    init_longitudinal_db()
    patient = infer_patient_id(patient_id)
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        con.execute("UPDATE raw_records SET patient_id=? WHERE record_key=?", (patient, str(record_key)))
        con.commit()
    _cloud_checkpoint_v1540('record_assigned_to_patient')
    return patient


def save_patient_analysis_bundle(patient_id, record_keys=None, config=None, include_all_snapshotted=True):
    """Guarda/actualiza un histórico longitudinal acumulativo por paciente.

    Nunca reduce un bundle existente: hace unión entre sus miembros previos, los registros
    indicados y todos los RRi del paciente que ya dispongan de snapshot analítico.
    """
    init_longitudinal_db()
    patient = infer_patient_id(patient_id)
    catalog = list_saved_raw_records()
    patient_rows = _canonical_patient_rows(catalog, patient)
    valid_keys = set(patient_rows["record_key"].astype(str).tolist()) if not patient_rows.empty else set()
    incoming = {str(k) for k in (record_keys or []) if str(k) in valid_keys}
    bundle_id = _patient_bundle_id(patient)
    now = datetime.now(timezone.utc).isoformat()
    previous = set()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        row = con.execute("SELECT record_keys_json FROM patient_analysis_bundles WHERE bundle_id=?", (bundle_id,)).fetchone()
        if row:
            try:
                previous = {str(k) for k in json.loads(row[0] or "[]")}
            except Exception:
                previous = set()
        snap_keys = set()
        if include_all_snapshotted and valid_keys:
            placeholders = ",".join(["?"] * len(valid_keys))
            snap_rows = con.execute(
                f"SELECT record_key FROM analysis_snapshots WHERE record_key IN ({placeholders})",
                tuple(valid_keys),
            ).fetchall()
            snap_keys = {str(r[0]) for r in snap_rows}
        merged = (previous | incoming | snap_keys) & valid_keys
        # Si todavía no existe snapshot de alguno de los registros indicados, se conserva igualmente
        # como miembro del histórico; al abrirlo se mostrará como pendiente, no se perderá silenciosamente.
        merged |= incoming
        ordered = []
        if not patient_rows.empty:
            for k in patient_rows["record_key"].astype(str).tolist():
                if k in merged:
                    ordered.append(k)
        display = patient.replace("_", " ").title()
        cfg = dict(config or {})
        ordered_json = json.dumps(ordered, ensure_ascii=False)
        cfg_json = json.dumps(cfg, ensure_ascii=False, default=str, sort_keys=True)
        existing = con.execute(
            "SELECT patient_id,display_name,record_keys_json,record_count,config_json FROM patient_analysis_bundles WHERE bundle_id=?",
            (bundle_id,),
        ).fetchone()
        bundle_changed = True
        if existing:
            try:
                old_cfg = json.dumps(json.loads(existing[4] or '{}'), ensure_ascii=False, default=str, sort_keys=True)
            except Exception:
                old_cfg = str(existing[4] or '{}')
            bundle_changed = not (
                str(existing[0] or '') == patient and str(existing[1] or '') == display and
                str(existing[2] or '') == ordered_json and int(existing[3] or 0) == len(ordered) and
                old_cfg == cfg_json
            )
        if bundle_changed:
            _bundle_saved_at = datetime.now(timezone.utc).isoformat()
            con.execute("""
                INSERT INTO patient_analysis_bundles
                (bundle_id, patient_id, display_name, app_version, saved_at, record_keys_json, record_count, config_json)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(bundle_id) DO UPDATE SET
                  patient_id=excluded.patient_id, display_name=excluded.display_name,
                  app_version=excluded.app_version, saved_at=excluded.saved_at,
                  record_keys_json=excluded.record_keys_json, record_count=excluded.record_count,
                  config_json=excluded.config_json
            """, (bundle_id, patient, display, "15.4.4", _bundle_saved_at, ordered_json,
                  len(ordered), cfg_json))
            con.commit()
    if bundle_changed:
        _cloud_checkpoint_v1540('patient_bundle_saved')
    return bundle_id, ordered


def list_patient_analysis_bundles():
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        return pd.read_sql_query("""
            SELECT bundle_id, patient_id, display_name, app_version, saved_at,
                   record_keys_json, record_count, config_json
            FROM patient_analysis_bundles
            ORDER BY display_name, saved_at DESC
        """, con)


def ensure_patient_bundles_from_existing_history():
    """Migración compatible, una sola vez por sesión, sin tráfico por navegación."""
    try:
        if st.session_state.get('_bundles_migrated_once_v1541', False):
            return
        st.session_state['_bundles_migrated_once_v1541'] = True
    except Exception:
        pass
    catalog = list_saved_raw_records()
    if catalog is None or catalog.empty:
        return
    patients = sorted({infer_patient_id(x) for x in catalog["record_name"].astype(str)})
    for patient in patients:
        rows = _canonical_patient_rows(catalog, patient)
        if rows.empty:
            continue
        # Solo migra pacientes que todavía NO tienen bundle. Abrir/navegar nunca
        # reescribe un histórico existente ni modifica su config_json.
        bundle_id = _patient_bundle_id(patient)
        with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
            exists = con.execute("SELECT 1 FROM patient_analysis_bundles WHERE bundle_id=?", (bundle_id,)).fetchone()
        if exists:
            continue
        save_patient_analysis_bundle(
            patient, rows["record_key"].astype(str).tolist(),
            config={"migrated": True}, include_all_snapshotted=True
        )

def load_patient_analysis_bundle(bundle_id):
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        row = con.execute("""
            SELECT patient_id,display_name,app_version,saved_at,record_keys_json,record_count,config_json
            FROM patient_analysis_bundles WHERE bundle_id=?
        """, (bundle_id,)).fetchone()
    if not row:
        return None
    try:
        keys = [str(k) for k in json.loads(row[4] or "[]")]
    except Exception:
        keys = []
    try:
        cfg = json.loads(row[6] or "{}")
    except Exception:
        cfg = {}
    return {
        "bundle_id": bundle_id, "patient_id": row[0], "display_name": row[1],
        "app_version": row[2], "saved_at": row[3], "record_keys": keys,
        "record_count": int(row[5] or len(keys)), "config": cfg,
    }



def _trash_patient_id(patient_id):
    patient = infer_patient_id(patient_id)
    return f"__trash__:{patient}"


def list_trashed_raw_records(patient_id=None):
    """Lista registros enviados a la papelera sin exponerlos al histórico activo."""
    init_longitudinal_db()
    query = """
        SELECT record_key, patient_id, record_name, filename, record_datetime,
               saved_at, duration_s, n_rri
        FROM raw_records
        WHERE patient_id LIKE '__trash__:%'
    """
    params = []
    if patient_id:
        query += " AND patient_id=?"
        params.append(_trash_patient_id(patient_id))
    query += " ORDER BY COALESCE(record_datetime, saved_at), saved_at"
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        return pd.read_sql_query(query, con, params=params)


def _refresh_patient_bundle_after_membership_change(patient_id):
    """Reconstruye localmente el bundle del paciente tras mover/restaurar/enviar a papelera."""
    patient = infer_patient_id(patient_id)
    catalog = list_saved_raw_records()
    rows = _canonical_patient_rows(catalog, patient)
    bundle_id = _patient_bundle_id(patient)
    valid_keys = rows['record_key'].astype(str).tolist() if not rows.empty else []
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        existing = con.execute(
            "SELECT config_json FROM patient_analysis_bundles WHERE bundle_id=?", (bundle_id,)
        ).fetchone()
        try:
            cfg = json.loads(existing[0] or '{}') if existing else {}
        except Exception:
            cfg = {}
        if not valid_keys:
            con.execute("DELETE FROM patient_analysis_bundles WHERE bundle_id=?", (bundle_id,))
            con.commit()
            return
        now = datetime.now(timezone.utc).isoformat()
        display = patient.replace('_', ' ').title()
        ordered_json = json.dumps(valid_keys, ensure_ascii=False)
        con.execute("""
            INSERT INTO patient_analysis_bundles
            (bundle_id, patient_id, display_name, app_version, saved_at, record_keys_json, record_count, config_json)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(bundle_id) DO UPDATE SET
              patient_id=excluded.patient_id, display_name=excluded.display_name,
              app_version=excluded.app_version, saved_at=excluded.saved_at,
              record_keys_json=excluded.record_keys_json, record_count=excluded.record_count,
              config_json=excluded.config_json
        """, (bundle_id, patient, display, '15.4.4', now, ordered_json, len(valid_keys),
              json.dumps(cfg, ensure_ascii=False, default=str)))
        con.commit()


def move_saved_records_to_patient(record_keys, target_patient_id, sync_cloud=True):
    """Mueve uno o varios registros entre pacientes conservando RRi, snapshots y métricas."""
    keys = [str(k) for k in dict.fromkeys(record_keys or []) if str(k)]
    if not keys:
        return 0, []
    target = infer_patient_id(target_patient_id)
    sources = set()
    moved = []
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        for key in keys:
            row = con.execute("SELECT patient_id,record_name FROM raw_records WHERE record_key=?", (key,)).fetchone()
            if not row:
                continue
            source = str(row[0] or '')
            if source.startswith('__trash__:'):
                source = source.split(':', 1)[1]
            source = infer_patient_id(source or row[1])
            sources.add(source)
            con.execute("UPDATE raw_records SET patient_id=? WHERE record_key=?", (target, key))
            con.execute("UPDATE observations SET patient_id=? WHERE record_name=?", (target, row[1]))
            moved.append(key)
        con.commit()
    for source in sorted(sources | {target}):
        _refresh_patient_bundle_after_membership_change(source)
    if moved and sync_cloud:
        _cloud_checkpoint_v1540('historical_records_moved')
    return len(moved), sorted(sources)


def trash_saved_records(record_keys, sync_cloud=True):
    """Envía registros a una papelera recuperable sin destruir RRi, snapshots ni resultados."""
    keys = [str(k) for k in dict.fromkeys(record_keys or []) if str(k)]
    if not keys:
        return 0, []
    sources = set()
    trashed = []
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        for key in keys:
            row = con.execute("SELECT patient_id,record_name FROM raw_records WHERE record_key=?", (key,)).fetchone()
            if not row:
                continue
            source = str(row[0] or '')
            if source.startswith('__trash__:'):
                continue
            source = infer_patient_id(source or row[1])
            sources.add(source)
            trash_id = _trash_patient_id(source)
            con.execute("UPDATE raw_records SET patient_id=? WHERE record_key=?", (trash_id, key))
            con.execute("UPDATE observations SET patient_id=? WHERE record_name=?", (trash_id, row[1]))
            trashed.append(key)
        con.commit()
    for source in sorted(sources):
        _refresh_patient_bundle_after_membership_change(source)
    if trashed and sync_cloud:
        _cloud_checkpoint_v1540('historical_records_trashed')
    return len(trashed), sorted(sources)


def restore_trashed_records(record_keys, target_patient_id=None, sync_cloud=True):
    """Restaura desde papelera; por defecto vuelve al paciente de origen codificado en patient_id."""
    keys = [str(k) for k in dict.fromkeys(record_keys or []) if str(k)]
    restored = []
    targets = set()
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        for key in keys:
            row = con.execute("SELECT patient_id,record_name FROM raw_records WHERE record_key=?", (key,)).fetchone()
            if not row or not str(row[0] or '').startswith('__trash__:'):
                continue
            original = str(row[0]).split(':', 1)[1]
            target = infer_patient_id(target_patient_id or original or row[1])
            targets.add(target)
            con.execute("UPDATE raw_records SET patient_id=? WHERE record_key=?", (target, key))
            con.execute("UPDATE observations SET patient_id=? WHERE record_name=?", (target, row[1]))
            restored.append(key)
        con.commit()
    for target in sorted(targets):
        _refresh_patient_bundle_after_membership_change(target)
    if restored and sync_cloud:
        _cloud_checkpoint_v1540('historical_records_restored')
    return len(restored), sorted(targets)

def delete_saved_raw_record(record_key):
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        row = con.execute('SELECT record_name FROM raw_records WHERE record_key=?', (record_key,)).fetchone()
        if row:
            con.execute('DELETE FROM observations WHERE record_name=?', (row[0],))
        con.execute('DELETE FROM segmentations WHERE record_key=?', (record_key,))
        con.execute('DELETE FROM temporal_selections WHERE record_key=?', (record_key,))
        con.execute('DELETE FROM analysis_snapshots WHERE record_key=?', (record_key,))
        con.execute('DELETE FROM raw_records WHERE record_key=?', (record_key,))
        con.commit()
    _cloud_checkpoint_v1540('historical_record_deleted')

def _safe_float_sql(value):
    try:
        v=float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def save_indices_to_longitudinal_db(indices_df):
    """Guarda índices + HVG + fisiología primaria v15.4.12 solo cuando cambian."""
    init_longitudinal_db()
    if indices_df is None or indices_df.empty: return 0
    feature_cols=['IDX_Vagal','IDX_Amplitud','IDX_Complejidad','IDX_Rigidez','IDX_Adaptabilidad','IDX_Regulacion_Lenta','IDX_Confianza'] + V15_HVG_FEATURES + V1548_PRIMARY_FEATURES
    saved=0
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        db_cols={r[1] for r in con.execute('PRAGMA table_info(observations)').fetchall()}
        feature_cols=[c for c in feature_cols if c in db_cols]
        for _,row in indices_df.iterrows():
            record=str(row.get('Registro','')).strip(); phase=str(row.get('Fase','')).strip()
            if not record or not phase: continue
            patient=infer_patient_id(record); dt=_extract_record_datetime(record); dt_text=dt.isoformat() if pd.notna(dt) else None
            key=hashlib.sha256(f'{patient}|{record}|{phase}'.encode('utf-8')).hexdigest(); profile=str(row.get('Perfil_autonomico',''))
            numeric=[_safe_float_sql(row.get(c)) for c in feature_cols]
            prev=con.execute('SELECT '+','.join(feature_cols)+' FROM observations WHERE observation_key=?',(key,)).fetchone()
            same=False
            if prev is not None:
                same=True
                for a,b in zip(prev,numeric):
                    if a is None and b is None: continue
                    try:
                        if a is None or b is None or abs(float(a)-float(b))>1e-10*max(1.0,abs(float(a)),abs(float(b))): same=False; break
                    except Exception:
                        if str(a)!=str(b): same=False; break
            if same: continue
            now=datetime.now(timezone.utc).isoformat()
            cols=['observation_key','patient_id','record_name','phase','record_datetime','saved_at','Perfil_autonomico']+feature_cols
            vals=[key,patient,record,phase,dt_text,now,profile]+numeric
            placeholders=','.join(['?']*len(cols))
            updates=','.join([f'{c}=excluded.{c}' for c in ['patient_id','record_datetime','saved_at','Perfil_autonomico']+feature_cols])
            con.execute(f"INSERT INTO observations ({','.join(cols)}) VALUES ({placeholders}) ON CONFLICT(observation_key) DO UPDATE SET {updates}",vals)
            saved+=1
        if saved: con.commit()
    if saved: _cloud_checkpoint_v1540('physiological_indices_saved')
    return saved

def load_longitudinal_observations(patient_id=None, phase=None):
    init_longitudinal_db()
    query='SELECT * FROM observations'
    clauses=[]; params=[]
    if patient_id and patient_id != 'Todos':
        clauses.append('patient_id=?'); params.append(patient_id)
    if phase and phase != 'Todas':
        clauses.append('phase=?'); params.append(phase)
    if clauses:
        query += ' WHERE ' + ' AND '.join(clauses)
    query += ' ORDER BY COALESCE(record_datetime, saved_at), saved_at'
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        df=pd.read_sql_query(query, con, params=params)
    if not df.empty:
        df['Fecha_hora']=pd.to_datetime(df['record_datetime'], errors='coerce')
        df=df.rename(columns={'record_name':'Registro','phase':'Fase','patient_id':'Paciente_ID'})
        # Unifica series que sólo difieren por mayúsculas, espacios o guiones.
        df['Paciente_ID'] = df['Paciente_ID'].map(normalize_patient_id)
        df['Registro'] = df['Registro'].astype(str).map(sanitize_name)
        # Las versiones antiguas pudieron guardar la misma observación con patient_id distinto.
        # Conservamos la actualización más reciente de cada registro-fase.
        if 'saved_at' in df.columns:
            df['_saved_sort'] = pd.to_datetime(df['saved_at'], errors='coerce')
            df = df.sort_values(['Fecha_hora', '_saved_sort'], na_position='last')
        df = df.drop_duplicates(subset=['Registro', 'Fase'], keep='last')
        df = df.drop(columns=['_saved_sort'], errors='ignore').reset_index(drop=True)
    return df


def autonomic_composite(row):
    """Resumen direccional para crear el desenlace del siguiente registro; no es desenlace clínico."""
    vals=[]; weights=[]
    for col,w,reverse in [
        ('IDX_Vagal',1.0,False), ('IDX_Amplitud',0.8,False),
        ('IDX_Complejidad',1.1,False), ('IDX_Adaptabilidad',1.3,False),
        ('IDX_Rigidez',1.1,True)
    ]:
        v=_safe_float_sql(row.get(col))
        if v is not None:
            vals.append((100-v) if reverse else v); weights.append(w)
    return float(np.average(vals, weights=weights)) if vals else np.nan


def build_self_supervised_transitions(history_df, stable_threshold=3.0):
    """Cada observación aprende del cambio observado en el registro cronológico siguiente."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    rows=[]
    df=history_df.copy()
    df['Fecha_orden']=pd.to_datetime(df.get('Fecha_hora'), errors='coerce')
    for (patient,phase), grp in df.groupby(['Paciente_ID','Fase'], dropna=False):
        grp=grp.sort_values(['Fecha_orden','saved_at'], na_position='last').reset_index(drop=True)
        for i in range(len(grp)-1):
            current=grp.iloc[i]; nxt=grp.iloc[i+1]
            s0=autonomic_composite(current); s1=autonomic_composite(nxt)
            if not (np.isfinite(s0) and np.isfinite(s1)):
                continue
            delta=float(s1-s0)
            label='Favorable' if delta > stable_threshold else ('Desfavorable' if delta < -stable_threshold else 'Estable')
            out={
                'Paciente_ID':patient, 'Fase':phase,
                'Registro_origen':current.get('Registro',''), 'Registro_siguiente':nxt.get('Registro',''),
                'Fecha_origen':current.get('Fecha_orden'), 'Fecha_siguiente':nxt.get('Fecha_orden'),
                'Delta_compuesto_siguiente':delta, 'Evolucion_siguiente':label,
            }
            for f in GB_INDEX_FEATURES:
                out[f]=current.get(f, np.nan)
            rows.append(out)
    return pd.DataFrame(rows)


def train_auto_longitudinal_model(transitions, random_state=42):
    """Reentrena Gradient Boosting automáticamente al crecer la base."""
    if not SKLEARN_AVAILABLE or transitions is None or transitions.empty:
        return None, 'Sin transiciones suficientes.'
    work=transitions.dropna(subset=['Evolucion_siguiente']).copy()
    counts=work['Evolucion_siguiente'].value_counts()
    if len(work) < 8:
        return None, f'Acumulando aprendizaje: {len(work)}/8 transiciones mínimas.'
    if len(counts) < 2:
        return None, 'Aún sólo existe una clase de evolución; se necesitan al menos dos patrones distintos.'
    if counts.min() < 2:
        return None, 'La clase menos frecuente necesita al menos dos transiciones.'
    X=work[GB_INDEX_FEATURES].apply(pd.to_numeric, errors='coerce')
    y=work['Evolucion_siguiente'].astype(str)
    model=Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('gb', GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.04, max_depth=2,
            min_samples_leaf=2, subsample=0.85, random_state=random_state
        ))
    ])
    model.fit(X,y)
    metrics={'n_transiciones':len(work), 'clases':dict(counts), 'modo':'auto-supervisado longitudinal'}
    min_class=int(counts.min())
    if len(work) >= 12 and min_class >= 3:
        n_splits=min(5,min_class)
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores=cross_val_score(model,X,y,cv=cv,scoring='balanced_accuracy')
        metrics['cv_balanced_accuracy_media']=float(np.mean(scores))
        metrics['cv_balanced_accuracy_sd']=float(np.std(scores))
    bundle={
        'model':model, 'features':GB_INDEX_FEATURES, 'task':'classification',
        'target':'Evolucion_siguiente', 'classes':list(model.classes_),
        'metrics':metrics, 'auto_supervised':True,
        'trained_at':datetime.now(timezone.utc).isoformat(),
    }
    try:
        joblib.dump(bundle, AUTO_GB_MODEL_PATH)
    except Exception:
        pass
    return bundle, 'Modelo autonómico auto-supervisado actualizado.'


def load_auto_longitudinal_model():
    if not SKLEARN_AVAILABLE or not AUTO_GB_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(AUTO_GB_MODEL_PATH)
    except Exception:
        return None


def current_autonomic_trend(history_df, patient_id=None, phase='Basal'):
    if history_df is None or history_df.empty:
        return None
    df=history_df.copy()
    if patient_id and patient_id != 'Todos': df=df[df['Paciente_ID']==patient_id]
    if phase and phase != 'Todas': df=df[df['Fase']==phase]
    df=df.sort_values(['Fecha_hora','saved_at'],na_position='last')
    if len(df)<2: return None
    a,b=df.iloc[-2],df.iloc[-1]
    s0,s1=autonomic_composite(a),autonomic_composite(b)
    if not(np.isfinite(s0) and np.isfinite(s1)): return None
    delta=s1-s0
    label='Favorable' if delta>3 else ('Desfavorable' if delta<-3 else 'Estable')
    return {'anterior':a.get('Registro',''),'actual':b.get('Registro',''),'delta':delta,'label':label,'score_actual':s1}



# ============================================================
# v14.0 · MOTOR DE PREDICCIÓN DE ESTADOS FISIOLÓGICOS
# Aprendizaje incremental por lotes con promoción condicionada
# ============================================================
V14_MODEL_PATH = APP_DATA_DIR / 'motor_predictivo_fisiologico_v140.joblib'
V14_INDEX_FEATURES = list(GB_INDEX_FEATURES)
V14_STATE_FEATURES = (
    V14_INDEX_FEATURES
    + [f'DELTA_{c}' for c in V14_INDEX_FEATURES]
    + [f'MEAN3_{c}' for c in V14_INDEX_FEATURES]
    + [f'STD3_{c}' for c in V14_INDEX_FEATURES]
    + ['COMPUESTO_ACTUAL','DELTA_COMPUESTO','TENDENCIA_3','DIAS_DESDE_ANTERIOR']
)


def _v14_sorted_group(grp):
    g=grp.copy()
    g['_dt']=pd.to_datetime(g.get('Fecha_hora'),errors='coerce')
    return g.sort_values(['_dt','saved_at'],na_position='last').reset_index(drop=True)


def _v14_feature_row(grp, i):
    """Features del estado actual i usando sólo información disponible hasta ese momento."""
    cur=grp.iloc[i]
    prev=grp.iloc[i-1] if i>0 else None
    start=max(0,i-2)
    hist=grp.iloc[start:i+1]
    out={}
    for c in V14_INDEX_FEATURES:
        x=_safe_float_sql(cur.get(c))
        xp=_safe_float_sql(prev.get(c)) if prev is not None else None
        out[c]=x
        out[f'DELTA_{c}']=(x-xp) if x is not None and xp is not None else 0.0
        vals=pd.to_numeric(hist[c],errors='coerce') if c in hist else pd.Series(dtype=float)
        out[f'MEAN3_{c}']=float(vals.mean()) if vals.notna().any() else np.nan
        out[f'STD3_{c}']=float(vals.std(ddof=0)) if vals.notna().any() else 0.0
    comp=autonomic_composite(cur)
    comp_prev=autonomic_composite(prev) if prev is not None else np.nan
    comps=[autonomic_composite(grp.iloc[j]) for j in range(start,i+1)]
    comps=[x for x in comps if np.isfinite(x)]
    out['COMPUESTO_ACTUAL']=comp
    out['DELTA_COMPUESTO']=(comp-comp_prev) if np.isfinite(comp) and np.isfinite(comp_prev) else 0.0
    out['TENDENCIA_3']=(comps[-1]-comps[0])/(len(comps)-1) if len(comps)>1 else 0.0
    dt=pd.to_datetime(cur.get('_dt'),errors='coerce')
    dtp=pd.to_datetime(prev.get('_dt'),errors='coerce') if prev is not None else pd.NaT
    out['DIAS_DESDE_ANTERIOR']=float((dt-dtp).total_seconds()/86400.0) if pd.notna(dt) and pd.notna(dtp) else np.nan
    return out


def build_v14_prediction_dataset(history_df, stable_threshold=3.0):
    """Construye X(t) -> estado e índices en t+1, evitando fuga de información futura."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    rows=[]
    for (patient,phase), raw in history_df.groupby(['Paciente_ID','Fase'],dropna=False):
        grp=_v14_sorted_group(raw)
        for i in range(len(grp)-1):
            cur,nxt=grp.iloc[i],grp.iloc[i+1]
            feat=_v14_feature_row(grp,i)
            s0,s1=autonomic_composite(cur),autonomic_composite(nxt)
            if not(np.isfinite(s0) and np.isfinite(s1)):
                continue
            delta=float(s1-s0)
            label='Favorable' if delta>stable_threshold else ('Desfavorable' if delta<-stable_threshold else 'Estable')
            row={
                'Paciente_ID':patient,'Fase':phase,
                'Registro_origen':cur.get('Registro',''),'Registro_objetivo':nxt.get('Registro',''),
                'Fecha_origen':cur.get('_dt'),'Fecha_objetivo':nxt.get('_dt'),
                'Estado_siguiente':label,'Delta_compuesto_siguiente':delta,
            }
            row.update(feat)
            for c in V14_INDEX_FEATURES:
                row[f'TARGET_{c}']=_safe_float_sql(nxt.get(c))
            rows.append(row)
    return pd.DataFrame(rows)


def _v14_signature(dataset):
    if dataset is None or dataset.empty:
        return ''
    cols=['Paciente_ID','Fase','Registro_origen','Registro_objetivo','Estado_siguiente']
    payload=dataset[[c for c in cols if c in dataset]].astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _v14_validation_score(X,y,random_state=42):
    counts=y.value_counts()
    if len(y)<12 or len(counts)<2 or counts.min()<3:
        return None,None
    n_splits=min(5,int(counts.min()))
    pipe=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('gb',GradientBoostingClassifier(n_estimators=180,learning_rate=0.035,max_depth=2,min_samples_leaf=2,subsample=.85,random_state=random_state))
    ])
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    scores=cross_val_score(pipe,X,y,cv=cv,scoring='balanced_accuracy')
    return float(np.mean(scores)),float(np.std(scores))


def train_v14_candidate(dataset, previous_bundle=None, random_state=42):
    if not SKLEARN_AVAILABLE or dataset is None or dataset.empty:
        return None,'Sin ejemplos predictivos.'
    work=dataset.dropna(subset=['Estado_siguiente']).copy()
    counts=work['Estado_siguiente'].value_counts()
    if len(work)<8:
        return None,f'Acumulando datos: {len(work)}/8 transiciones mínimas.'
    if len(counts)<2 or counts.min()<2:
        return None,'Se necesitan al menos dos estados y dos ejemplos de la clase minoritaria.'
    X=work.reindex(columns=V14_STATE_FEATURES).apply(pd.to_numeric,errors='coerce')
    y=work['Estado_siguiente'].astype(str)
    cv_mean,cv_sd=_v14_validation_score(X,y,random_state)
    classifier=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('gb',GradientBoostingClassifier(n_estimators=220,learning_rate=.03,max_depth=2,min_samples_leaf=2,subsample=.88,random_state=random_state))
    ])
    classifier.fit(X,y)
    regressors={}; residual_mae={}
    for c in V14_INDEX_FEATURES:
        target=pd.to_numeric(work[f'TARGET_{c}'],errors='coerce')
        mask=target.notna()
        if mask.sum()<6:
            continue
        reg=Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('gb',GradientBoostingRegressor(n_estimators=180,learning_rate=.035,max_depth=2,min_samples_leaf=2,subsample=.88,loss='huber',random_state=random_state))
        ])
        reg.fit(X.loc[mask],target.loc[mask])
        pred=reg.predict(X.loc[mask])
        regressors[c]=reg
        residual_mae[c]=float(mean_absolute_error(target.loc[mask],pred))
    prev_gen=int(previous_bundle.get('generation',0)) if isinstance(previous_bundle,dict) else 0
    bundle={
        'classifier':classifier,'regressors':regressors,'residual_mae':residual_mae,
        'features':V14_STATE_FEATURES,'index_features':V14_INDEX_FEATURES,
        'classes':list(classifier.classes_),'n_samples':len(work),'class_counts':dict(counts),
        'cv_balanced_accuracy':cv_mean,'cv_sd':cv_sd,'signature':_v14_signature(work),
        'generation':prev_gen+1,'trained_at':datetime.now(timezone.utc).isoformat(),
        'mode':'incremental_batch_gradient_boosting_v14'
    }
    return bundle,'Candidato predictivo entrenado.'


def load_v14_bundle():
    if not SKLEARN_AVAILABLE or not V14_MODEL_PATH.exists(): return None
    try:
        b=joblib.load(V14_MODEL_PATH)
        return b if isinstance(b,dict) and 'classifier' in b else None
    except Exception:
        return None


def incremental_update_v14(dataset, min_new_examples=1, tolerance=0.05):
    """Actualiza por lotes: entrena sólo si hay ejemplos nuevos y promociona si no degrada validación."""
    active=load_v14_bundle()
    sig=_v14_signature(dataset)
    if active and active.get('signature')==sig:
        return active,'Modelo sin cambios: no hay transiciones nuevas.',False
    old_n=int(active.get('n_samples',0)) if active else 0
    new_n=len(dataset) if dataset is not None else 0
    if active and new_n-old_n<min_new_examples:
        return active,f'Esperando lote incremental: {new_n-old_n}/{min_new_examples} ejemplos nuevos.',False
    candidate,status=train_v14_candidate(dataset,active)
    if candidate is None:
        return active,status,False
    old_score=active.get('cv_balanced_accuracy') if active else None
    new_score=candidate.get('cv_balanced_accuracy')
    promote=(active is None or old_score is None or new_score is None or new_score>=old_score-tolerance)
    if promote:
        try: joblib.dump(candidate,V14_MODEL_PATH)
        except Exception: pass
        return candidate,f'Modelo actualizado: generación {candidate["generation"]}, {candidate["n_samples"]} ejemplos.',True
    return active,f'Candidato no promocionado: validación {new_score:.3f} < modelo activo {old_score:.3f}.',False


def latest_v14_feature_frame(history_df,patient_id,phase):
    df=history_df[(history_df['Paciente_ID']==patient_id)&(history_df['Fase']==phase)].copy()
    if df.empty:return pd.DataFrame(),None
    grp=_v14_sorted_group(df)
    feat=_v14_feature_row(grp,len(grp)-1)
    meta={'Registro':grp.iloc[-1].get('Registro',''),'Fecha':grp.iloc[-1].get('_dt'),'n_historial':len(grp)}
    return pd.DataFrame([feat]).reindex(columns=V14_STATE_FEATURES),meta


def predict_v14_next_state(bundle,history_df,patient_id,phase):
    if not bundle:return None
    X,meta=latest_v14_feature_frame(history_df,patient_id,phase)
    if X.empty:return None
    clf=bundle['classifier']; pred=str(clf.predict(X)[0])
    probs=clf.predict_proba(X)[0]
    prob_map={str(c):float(p) for c,p in zip(clf.classes_,probs)}
    idx_rows=[]
    for c,reg in bundle.get('regressors',{}).items():
        value=float(np.clip(reg.predict(X)[0],0,100)); mae=float(bundle.get('residual_mae',{}).get(c,np.nan))
        idx_rows.append({'Índice':c,'Predicción_siguiente':value,'Incertidumbre_aprox_±MAE':mae})
    return {'prediccion':pred,'probabilidades':prob_map,'indices':pd.DataFrame(idx_rows),'meta':meta}

def load_gradient_boosting_bundle_bytes(raw_bytes):
    import io
    bundle = joblib.load(io.BytesIO(raw_bytes))
    required = {'model','features','task','target'}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError('El archivo no contiene un modelo Gradient Boosting compatible con la aplicación.')
    return bundle


def save_active_gradient_boosting_bundle(bundle):
    """Guarda el modelo activo junto a app.py para reutilizarlo en próximos arranques locales."""
    joblib.dump(bundle, ACTIVE_GB_MODEL_PATH)
    return ACTIVE_GB_MODEL_PATH


def auto_load_active_gradient_boosting_bundle():
    if not SKLEARN_AVAILABLE or not ACTIVE_GB_MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(ACTIVE_GB_MODEL_PATH)
        required = {'model','features','task','target'}
        return bundle if isinstance(bundle, dict) and required.issubset(bundle) else None
    except Exception:
        return None


def calculate_all(rr, include_rqa=True, include_hvg=False, mse_zero_policy=None, theiler_window=None, radius_mode=None):
    """
    Calcula métricas HRV por ventana.

    v10.3:
    - Entropías ApEn, SampEn y MSE: RR en ms con smoothness priors λ=500.
    - SampEn/MSE: m=2, r=0.2 x SD.
    - DFA: alpha1 4-12, alpha2 13-64.
    - RQA: emb_dim=10, threshold=sqrt(10)≈3.1623 x SD.
    """
    rr_ms = rr * 1000.0
    out = {}

    if mse_zero_policy is None:
        mse_zero_policy = st.session_state.get("mse_zero_policy", "nan") if "st" in globals() else "nan"
    if theiler_window is None:
        theiler_window = st.session_state.get("sampen_theiler_window", 0) if "st" in globals() else 0
    if radius_mode is None:
        radius_mode = st.session_state.get("mse_radius_mode", "fixed_entropy_sd") if "st" in globals() else "fixed_entropy_sd"

    # Lineales y frecuencia
    out.update(time_metrics(rr))
    out.update(psd_metrics(rr))
    out.update(lomb_psd_metrics(rr))
    out.update(ar_psd_metrics(rr))
    out.update(wavelet_band_metrics(rr))

    # No lineales sin suavizado
    a1, a2 = dfa_calc(rr_ms)
    out["DFA_alpha1"], out["DFA_alpha2"] = a1, a2
    out["D2"] = d2_calc(rr_ms)
    out.update(advanced_nonlinear_metrics(rr))

    if include_rqa:
        out.update(rqa_calc(rr_ms))

    if include_hvg:
        out.update(hvg_metrics(rr))

    # Entropías con lambda 500
    rr_entropy = smoothness_priors_detrend(rr_ms, LAMBDA_DEFAULT)
    rr_radius_reference = rr_ms if radius_mode == "fixed_raw_sd" else rr_entropy

    # ApEn se calcula sobre la misma señal λ=500
    out["ApEn"] = apen_calc(rr_ms)

    out["SampEn"] = sample_entropy_common(
        rr_entropy,
        m=KUBIOS_ENTROPY_M,
        r_factor=KUBIOS_ENTROPY_R_FACTOR,
        r_reference=rr_radius_reference,
        zero_policy=mse_zero_policy,
        theiler_window=theiler_window,
        radius_mode=radius_mode
    )

    out.update(
        mse_common(
            rr_entropy,
            scales=KUBIOS_MSE_MAX_SCALE,
            m=KUBIOS_ENTROPY_M,
            r_factor=KUBIOS_ENTROPY_R_FACTOR,
            r_reference=rr_radius_reference,
            zero_policy=mse_zero_policy
        )
    )

    # Garantía final: MSE1 = SampEn
    out["MSE1"] = out["SampEn"]

    # Variables de auditoría para verificar configuración Kubios/λ
    out.update(_entropy_debug_values(rr_entropy))
    out["DFA_alpha1_range"] = f"{KUBIOS_DFA_ALPHA1_RANGE[0]}-{KUBIOS_DFA_ALPHA1_RANGE[1]}"
    out["DFA_alpha2_range"] = f"{KUBIOS_DFA_ALPHA2_RANGE[0]}-{KUBIOS_DFA_ALPHA2_RANGE[1]}"
    out["RQA_threshold_SD"] = KUBIOS_RQA_THRESHOLD_SD
    out["RQA_emb_dim"] = KUBIOS_RQA_EMB_DIM
    out["MSE_zero_policy"] = mse_zero_policy
    out["SampEn_Theiler"] = theiler_window
    out["MSE_radius_mode"] = radius_mode

    # Índices fisiológicos v13.0 calculados sobre las métricas disponibles de la ventana.
    out.update(physiological_indices_from_row(out))

    return out


def get_record_windows(global_windows, record_windows, rec, use_independent):
    if use_independent:
        return record_windows.get(rec, global_windows)
    return global_windows


def calculate_record(rr, windows, active_phases, min_rr, include_rqa, include_hvg=False, mse_zero_policy=None, theiler_window=None, radius_mode=None):
    rows, segments, valid = [], {}, {}

    for ph in PHASES:
        w = windows.get(ph)
        if w is None:
            segments[ph] = np.array([])
            valid[ph] = False
            continue

        s, e = w
        seg = cut_segment(rr, s, e)
        segments[ph] = seg
        valid[ph] = len(seg) >= min_rr and ph in active_phases

        if valid[ph]:
            res = calculate_all(seg, include_rqa=include_rqa, include_hvg=include_hvg, mse_zero_policy=mse_zero_policy, theiler_window=theiler_window, radius_mode=radius_mode)
            res["Fase"] = ph
            rows.append(res)

    return (pd.DataFrame(rows).set_index("Fase") if rows else pd.DataFrame()), segments, valid


def build_long(records_results):
    rows = []

    for rec, df in records_results.items():
        if df is None or df.empty:
            continue

        tmp = df.copy()
        tmp.insert(0, "Registro", rec)
        tmp.insert(1, "Fase", tmp.index)
        rows.append(tmp.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_windows_to_fig(fig, windows):
    for ph, w in windows.items():
        if w is None:
            continue

        s, e = w
        group = PHASE_GROUP.get(ph, ph)
        fig.add_vrect(
            x0=s / 60,
            x1=e / 60,
            fillcolor=PHASE_COLORS.get(group, "rgba(180,180,180,.15)"),
            line_width=0,
            annotation_text=ph,
            annotation_position="top left",
        )


def _downsample_xy_v15213(x, y, max_points=1000):
    """Reduce únicamente los puntos dibujados; nunca modifica la señal usada para calcular métricas."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = min(len(x), len(y))
    if n <= max_points or max_points is None or max_points <= 0:
        return x[:n], y[:n]
    idx = np.linspace(0, n - 1, int(max_points)).astype(int)
    return x[idx], y[idx]


def rr_plot(record_data, global_windows, record_windows, view_mode, selected_record, use_independent, pending_selection=None, max_display_points=1000, chart_height=520):
    fig = go.Figure()
    names = [selected_record] if view_mode == "Registro principal" else list(record_data.keys())

    for name in names:
        rr = record_data[name]["rr"]
        t = cumulative_time(rr) / 60

        if np.any(record_data[name].get("artifact_mask", np.array([]))):
            rr_raw = record_data[name]["rr_raw"]
            t_raw = cumulative_time(rr_raw) / 60
            mask = record_data[name]["artifact_mask"]

            _txr, _yr = _downsample_xy_v15213(t_raw, rr_raw * 1000, max_display_points)
            _txc, _yc = _downsample_xy_v15213(t, rr * 1000, max_display_points)
            fig.add_trace(go.Scatter(x=_txr, y=_yr, mode="lines", name=f"{name} original", opacity=0.25))
            fig.add_trace(go.Scatter(x=_txc, y=_yc, mode="lines", name=f"{name} corregido"))

            if len(mask) == len(rr_raw):
                _ax, _ay = t_raw[mask], rr_raw[mask] * 1000
                _ax, _ay = _downsample_xy_v15213(_ax, _ay, min(max_display_points, 800))
                fig.add_trace(go.Scatter(x=_ax, y=_ay, mode="markers", name=f"{name} artefactos", marker=dict(symbol="x", size=8)))
        else:
            _tx, _ty = _downsample_xy_v15213(t, rr * 1000, max_display_points)
            fig.add_trace(go.Scatter(x=_tx, y=_ty, mode="lines", name=name))

    if view_mode == "Registro principal":
        windows = get_record_windows(global_windows, record_windows, selected_record, use_independent)
        add_windows_to_fig(fig, windows)

        # v15.2.4: el último tramo arrastrado permanece visible aunque Streamlit haga rerun.
        if pending_selection is not None and len(pending_selection) == 2:
            try:
                ps, pe = float(pending_selection[0]), float(pending_selection[1])
                if np.isfinite(ps) and np.isfinite(pe) and pe > ps:
                    fig.add_vrect(
                        x0=ps / 60.0,
                        x1=pe / 60.0,
                        fillcolor="rgba(255,215,0,0.22)",
                        line=dict(color="#ffd700", width=2, dash="dash"),
                        annotation_text=f"Selección guardada {sec_to_hms(ps)}–{sec_to_hms(pe)}",
                        annotation_position="top right",
                    )
            except Exception:
                pass

    # Trazas invisibles de ayuda para que la selección con recuadro capture el rango X completo.
    # Plotly/Streamlit devuelve puntos seleccionados, no las coordenadas exactas del recuadro.
    # Estas líneas invisibles hacen que el rango X sea más estable aunque el recuadro no toque muchos puntos RRi.
    # En panel individual, el eje X debe ajustarse al registro visible. Antes se usaba
    # la duración máxima de todos los archivos, comprimiendo los registros cortos.
    if view_mode == "Registro principal" and selected_record in record_data:
        all_durations = [record_data[selected_record]["duration"]]
    else:
        all_durations = [data["duration"] for data in record_data.values()]
    max_x_min = max(all_durations) / 60 if all_durations else 1
    helper_x = np.linspace(0, max_x_min, 350 if max_display_points <= 1500 else 800)

    y_values = []
    _y_sources = ({selected_record: record_data[selected_record]}
                  if view_mode == "Registro principal" and selected_record in record_data
                  else record_data)
    for data in _y_sources.values():
        if len(data["rr"]) > 0:
            y_values.extend(list(data["rr"] * 1000))

    if y_values:
        y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
        if y_max > y_min:
            for y0 in np.linspace(y_min, y_max, 6 if max_display_points <= 1500 else 10):
                fig.add_trace(go.Scatter(
                    x=helper_x,
                    y=np.full_like(helper_x, y0),
                    mode="markers",
                    marker=dict(size=3, opacity=0.01),
                    name="_selector_helper",
                    hoverinfo="skip",
                    showlegend=False,
                ))

    fig.update_layout(
        height=int(chart_height),
        xaxis_title="Tiempo acumulado (min)",
        yaxis_title="RRi (ms)",
        hovermode="x unified",
        dragmode="select",
        margin=dict(l=55, r=25, t=65, b=70),
    )
    fig.update_xaxes(rangeslider_visible=True)

    return fig


def comparison_bar_line(pivot, variable):
    """
    Comparación por columnas verticales + tendencias.

    v15.4.0:
    - Si hay varios pacientes, los registros se agrupan físicamente por paciente.
    - Las tendencias nunca conectan pacientes distintos.
    - Con varias fases, cada fase conserva barras agrupadas por registro y una
      tendencia longitudinal independiente dentro de cada paciente.
    """
    if pivot is None or pivot.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{variable}: sin datos para graficar", height=520)
        return fig

    cols_sorted = sorted(list(pivot.columns), key=lambda r: (extract_datetime_from_name(r), r))
    pivot = pivot.reindex(columns=cols_sorted)
    phases = list(pivot.index)
    multi_patient = _has_multiple_patients(cols_sorted)
    fig = go.Figure()

    if multi_patient:
        patient_groups = _records_grouped_by_patient(cols_sorted)
        patient_order = list(patient_groups.keys())

        # Posiciones: todos los registros del paciente A juntos, después un hueco,
        # luego todos los registros del paciente B, etc.
        rec_x = {}
        tick_vals, tick_text = [], []
        group_spans = []
        cursor = 0.0
        gap = 1.35
        for gi, pid in enumerate(patient_order):
            recs = patient_groups[pid]
            start_x = cursor
            for rec in recs:
                rec_x[rec] = cursor
                tick_vals.append(cursor)
                tick_text.append(_comparison_axis_label(rec, multiline=True, include_patient=True))
                cursor += 1.0
            end_x = cursor - 1.0
            group_spans.append((pid, start_x, end_x, recs))
            if gi < len(patient_order) - 1:
                cursor += gap

        if len(phases) == 1:
            ph = phases[0]
            for pi, (pid, start_x, end_x, recs) in enumerate(group_spans):
                color = _export_color_for(pi)
                px = [rec_x[r] for r in recs]
                py = []
                labels = []
                for rec in recs:
                    v = pd.to_numeric(pivot.loc[ph, rec], errors="coerce")
                    py.append(float(v) if pd.notna(v) else np.nan)
                    labels.append(_comparison_axis_label(rec, multiline=True, include_patient=True))
                pname = _patient_display_name(pid)

                fig.add_trace(go.Bar(
                    x=px, y=py, name=pname, marker=dict(color=color), opacity=0.74,
                    customdata=labels,
                    hovertemplate="%{customdata}<br>Fase: " + str(ph) +
                                  f"<br>{variable}: " + "%{y:.3f}<extra></extra>"
                ))
                valid = [(x, y) for x, y in zip(px, py) if np.isfinite(y)]
                if len(valid) >= 2:
                    vx = np.asarray([a for a, _ in valid], dtype=float)
                    vy = [b for _, b in valid]
                    sx, sy = _smooth_line_xy(vy)
                    sx = np.interp(sx, np.arange(len(vx), dtype=float), vx)
                    fig.add_trace(go.Scatter(
                        x=sx, y=sy, mode="lines", name=f"{pname} · tendencia",
                        line=dict(width=4, color=color), hoverinfo="skip"
                    ))
                fig.add_trace(go.Scatter(
                    x=px, y=py, mode="markers", marker=dict(size=8, color=color),
                    showlegend=False, customdata=labels,
                    hovertemplate="%{customdata}<br>Fase: " + str(ph) +
                                  f"<br>{variable}: " + "%{y:.3f}<extra></extra>"
                ))
        else:
            # Varias fases: el eje X sigue siendo longitudinal por paciente.
            # Cada fase es una barra dentro del registro y su tendencia sólo se
            # dibuja dentro del paciente correspondiente.
            nph = max(1, len(phases))
            bar_width = min(0.72 / nph, 0.20)
            for pi, (pid, start_x, end_x, recs) in enumerate(group_spans):
                pname = _patient_display_name(pid)
                for fi, ph in enumerate(phases):
                    color = _export_color_for(pi * max(1, len(phases)) + fi)
                    offset = (fi - (nph - 1) / 2.0) * bar_width
                    px = [rec_x[r] + offset for r in recs]
                    center_x = [rec_x[r] for r in recs]
                    py = []
                    custom = []
                    for rec in recs:
                        v = pd.to_numeric(pivot.loc[ph, rec], errors="coerce")
                        py.append(float(v) if pd.notna(v) else np.nan)
                        custom.append([pname, ph, _record_axis_label(rec, multiline=False)])
                    fig.add_trace(go.Bar(
                        x=px, y=py, width=bar_width,
                        name=f"{pname} · {ph}", marker=dict(color=color), opacity=0.72,
                        customdata=custom,
                        hovertemplate="Paciente: %{customdata[0]}<br>Fase: %{customdata[1]}<br>"
                                      "Fecha: %{customdata[2]}<br>" + f"{variable}: " +
                                      "%{y:.3f}<extra></extra>"
                    ))
                    valid = [(x, y) for x, y in zip(center_x, py) if np.isfinite(y)]
                    if len(valid) >= 2:
                        vx = np.asarray([a for a, _ in valid], dtype=float)
                        vy = [b for _, b in valid]
                        sx, sy = _smooth_line_xy(vy)
                        sx = np.interp(sx, np.arange(len(vx), dtype=float), vx)
                        fig.add_trace(go.Scatter(
                            x=sx, y=sy, mode="lines",
                            name=f"{pname} · {ph} · tendencia",
                            line=dict(width=3.2, color=color), hoverinfo="skip"
                        ))

        # Separadores y rótulos de paciente.
        for gi, (pid, start_x, end_x, recs) in enumerate(group_spans):
            center = (start_x + end_x) / 2.0
            fig.add_annotation(
                x=center, y=-0.22, xref="x", yref="paper",
                text=f"<b>{_patient_display_name(pid)}</b>",
                showarrow=False, font=dict(size=13)
            )
            if gi < len(group_spans) - 1:
                next_start = group_spans[gi + 1][1]
                fig.add_vline(
                    x=(end_x + next_start) / 2.0,
                    line_width=1, line_dash="dot", opacity=0.38
                )

        fig.update_xaxes(
            tickmode="array", tickvals=tick_vals, ticktext=tick_text,
            title_text="Paciente · fecha del registro"
        )
        fig.update_layout(
            height=630, yaxis_title=variable, barmode="group",
            hovermode="closest", bargap=0.24, bargroupgap=0.06,
            title=f"{variable}: comparación agrupada por paciente",
            legend_title_text="Paciente · fase",
            margin=dict(l=60, r=40, t=80, b=160)
        )
        return fig

    # Un solo paciente: conservar el comportamiento previo.
    if len(phases) == 1:
        ph = phases[0]
        x_labels = [_comparison_axis_label(rec, multiline=True, include_patient=False) for rec in cols_sorted]
        y_vals = [pd.to_numeric(pivot.loc[ph, rec], errors="coerce") for rec in cols_sorted]
        y_vals = [float(v) if pd.notna(v) else np.nan for v in y_vals]
        x_num = np.arange(len(x_labels), dtype=float)
        color = _export_color_for(0)

        fig.add_trace(go.Bar(
            x=x_num, y=y_vals, name=f"{variable} · columnas",
            marker=dict(color=color), opacity=0.72,
            hovertemplate="Registro: %{customdata}<br>Fase: " + str(ph) +
                          f"<br>{variable}: " + "%{y:.3f}<extra></extra>",
            customdata=x_labels,
        ))
        xs, ys = _smooth_line_xy(y_vals)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{variable} · tendencia suavizada",
            line=dict(width=4, color=color), hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x_num, y=y_vals, mode="markers", name=f"{variable} · puntos",
            marker=dict(size=8, color=color),
            hovertemplate="Registro: %{customdata}<br>Fase: " + str(ph) +
                          f"<br>{variable}: " + "%{y:.3f}<extra></extra>",
            customdata=x_labels, showlegend=False,
        ))
        fig.update_xaxes(
            tickmode="array", tickvals=list(x_num), ticktext=x_labels,
            title_text="Fecha del registro"
        )
        fig.update_layout(
            height=560, title=f"{variable}: columnas + tendencia suavizada en {ph}",
            yaxis_title=variable, hovermode="closest", bargap=0.28
        )
        return fig

    x_base = np.arange(len(phases), dtype=float)
    nrec = max(1, len(cols_sorted))
    bar_width = min(0.72 / nrec, 0.18)
    for i, rec in enumerate(cols_sorted):
        color = _export_color_for(i)
        y = [pd.to_numeric(pivot.loc[ph, rec], errors="coerce") for ph in phases]
        y = [float(v) if pd.notna(v) else np.nan for v in y]
        offset = (i - (nrec - 1) / 2) * bar_width
        x_bar = x_base + offset
        fig.add_trace(go.Bar(
            x=x_bar, y=y, width=bar_width,
            name=f"{_short_record_label(rec, 24)} · columnas",
            marker=dict(color=color), opacity=0.70, customdata=phases,
            hovertemplate="Registro: " + _short_record_label(rec, 32) +
                          "<br>Fase: %{customdata}<br>" + f"{variable}: " +
                          "%{y:.3f}<extra></extra>"
        ))
        xs, ys = _smooth_line_xy(y)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{_short_record_label(rec, 24)} · tendencia",
            line=dict(width=3.5, color=color), hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=x_base, y=y, mode="markers",
            marker=dict(size=7, color=color), showlegend=False,
            customdata=phases,
            hovertemplate="Registro: " + _short_record_label(rec, 32) +
                          "<br>Fase: %{customdata}<br>" + f"{variable}: " +
                          "%{y:.3f}<extra></extra>"
        ))
    fig.update_xaxes(
        tickmode="array", tickvals=list(x_base), ticktext=phases, title_text="Fase"
    )
    fig.update_layout(
        height=580, title=f"{variable}: columnas verticales + líneas de tendencia suavizadas",
        yaxis_title=variable, barmode="group", hovermode="closest",
        bargap=0.24, bargroupgap=0.08, legend_title_text="Registro"
    )
    return fig

def dashboard_compare(long_df, phases, params):
    params = [p for p in params if p in long_df.columns]

    if len(params) == 0:
        return go.Figure()

    cols = 2
    rows = int(np.ceil(len(params) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=params)

    for idx, p in enumerate(params):
        r = idx // cols + 1
        c = idx % cols + 1
        pivot = long_df[long_df["Fase"].isin(phases)].pivot_table(index="Fase", columns="Registro", values=p, aggfunc="first").reindex(phases)

        for rec in pivot.columns:
            fig.add_trace(go.Bar(x=list(pivot.index), y=pivot[rec], name=f"{_record_axis_label(rec)} · {p}", opacity=0.60, showlegend=(idx == 0)), row=r, col=c)
            fig.add_trace(go.Scatter(x=list(pivot.index), y=pivot[rec], mode="lines+markers", name=f"{_record_axis_label(rec)} tendencia", showlegend=False), row=r, col=c)

    fig.update_layout(height=max(440, rows * 340), barmode="group", title="Dashboard comparativo: barras + tendencia por parámetro")

    return fig



def _short_record_label(name, max_len=22):
    txt = str(name)
    if len(txt) <= max_len:
        return txt
    return txt[:max_len - 1] + "…"


def _record_axis_label(name, include_seconds=True, multiline=False):
    """Etiqueta cronológica sin nombre del paciente.

    - En ejes, ``multiline=True`` devuelve fecha y hora en dos líneas:
      ``04/08/2026<br>10:33:44``.
    - En leyendas y textos conserva una sola línea.
    - Si no existe una fecha reconocible, usa el identificador original.
    """
    dt = extract_datetime_from_name(name)
    if pd.notna(dt) and dt is not pd.Timestamp.max:
        try:
            date_txt = pd.Timestamp(dt).strftime("%d/%m/%Y")
            time_fmt = "%H:%M:%S" if include_seconds else "%H:%M"
            time_txt = pd.Timestamp(dt).strftime(time_fmt)
            return f"{date_txt}<br>{time_txt}" if multiline else f"{date_txt} {time_txt}"
        except Exception:
            pass
    dt2 = _extract_record_datetime(name) if '_extract_record_datetime' in globals() else pd.NaT
    if pd.notna(dt2):
        date_txt = pd.Timestamp(dt2).strftime("%d/%m/%Y")
        time_fmt = "%H:%M:%S" if include_seconds else "%H:%M"
        time_txt = pd.Timestamp(dt2).strftime(time_fmt)
        return f"{date_txt}<br>{time_txt}" if multiline else f"{date_txt} {time_txt}"
    return str(name)


def _record_axis_label_compact(name):
    """Etiqueta corta para ejes densos: DD/MM HH:MM, sin nombre del paciente."""
    dt = extract_datetime_from_name(name)
    if pd.notna(dt) and dt is not pd.Timestamp.max:
        try:
            return pd.Timestamp(dt).strftime("%d/%m %H:%M")
        except Exception:
            pass
    dt2 = _extract_record_datetime(name) if '_extract_record_datetime' in globals() else pd.NaT
    if pd.notna(dt2):
        try:
            return pd.Timestamp(dt2).strftime("%d/%m %H:%M")
        except Exception:
            pass
    return _short_record_label(name, max_len=14)


def _patient_axis_label(name):
    """Nombre legible del paciente inferido desde el identificador del registro."""
    try:
        pid = infer_patient_id(str(name))
        return str(pid).replace("_", " ").strip().title() or str(name)
    except Exception:
        return str(name)


def _comparison_axis_label(name, multiline=True, include_patient=False):
    date_label = _record_axis_label(name, multiline=multiline)
    if include_patient:
        sep = "<br>" if multiline else " · "
        return f"{_patient_axis_label(name)}{sep}{date_label}"
    return date_label


def _has_multiple_patients(record_names):
    try:
        return len({infer_patient_id(str(r)) for r in record_names}) > 1
    except Exception:
        return False


def _records_grouped_by_patient(record_names):
    """Agrupa registros por paciente y ordena cada grupo cronológicamente."""
    groups = {}
    for rec in list(record_names):
        try:
            pid = normalize_patient_id(infer_patient_id(str(rec)))
        except Exception:
            pid = str(rec)
        groups.setdefault(pid, []).append(rec)
    ordered = {}
    for pid in sorted(groups.keys(), key=lambda x: str(x).lower()):
        ordered[pid] = sorted(groups[pid], key=lambda r: (extract_datetime_from_name(r), str(r)))
    return ordered


def _patient_display_name(pid):
    return str(pid).replace('_', ' ').strip().title()


def _patient_page_controls(patient_ids, key_prefix, page_size=3):
    """Paginación puramente visual/local para comparaciones multipaciente.

    No consulta PostgreSQL. Devuelve como máximo `page_size` pacientes para
    mantener cada panel legible cuando hay muchos históricos simultáneos.
    """
    ids = list(patient_ids or [])
    if not ids:
        return [], 1, 1
    page_size = max(1, int(page_size))
    n_pages = max(1, int(np.ceil(len(ids) / page_size)))
    state_key = f"{key_prefix}_patient_page"
    cur = int(st.session_state.get(state_key, 1) or 1)
    cur = min(max(cur, 1), n_pages)
    st.session_state[state_key] = cur

    if n_pages > 1:
        cprev, cmid, cnext = st.columns([1, 2, 1])
        with cprev:
            if st.button("◀ Pacientes anteriores", key=f"{key_prefix}_prev", use_container_width=True, disabled=cur <= 1):
                st.session_state[state_key] = max(1, cur - 1)
                st.rerun()
        with cmid:
            st.markdown(f"<div style='text-align:center;padding-top:0.45rem'><b>Página {cur}/{n_pages}</b> · máximo {page_size} pacientes por página</div>", unsafe_allow_html=True)
        with cnext:
            if st.button("Pacientes siguientes ▶", key=f"{key_prefix}_next", use_container_width=True, disabled=cur >= n_pages):
                st.session_state[state_key] = min(n_pages, cur + 1)
                st.rerun()

    start = (cur - 1) * page_size
    visible = ids[start:start + page_size]
    return visible, cur, n_pages


def _safe_vertical_spacing(rows, preferred=0.12):
    """Plotly exige spacing <= 1/(rows-1). Evita ValueError en históricos grandes."""
    try:
        rows = int(rows)
    except Exception:
        rows = 1
    if rows <= 1:
        return 0.0
    return float(min(float(preferred), 0.82 / max(1, rows - 1)))


def _record_page_controls(total_rows, key_prefix, page_size=6, label="registros por paciente"):
    """Paginación local de visitas/registros; nunca consulta PostgreSQL."""
    total_rows = max(0, int(total_rows or 0))
    page_size = max(1, int(page_size))
    n_pages = max(1, int(np.ceil(total_rows / page_size))) if total_rows else 1
    state_key = f"{key_prefix}_record_page"
    cur = int(st.session_state.get(state_key, 1) or 1)
    cur = min(max(cur, 1), n_pages)
    st.session_state[state_key] = cur
    if n_pages > 1:
        cprev, cmid, cnext = st.columns([1, 2, 1])
        with cprev:
            if st.button("◀ Registros anteriores", key=f"{key_prefix}_rprev", use_container_width=True, disabled=cur <= 1):
                st.session_state[state_key] = max(1, cur - 1); st.rerun()
        with cmid:
            st.markdown(f"<div style='text-align:center;padding-top:0.45rem'><b>Bloque {cur}/{n_pages}</b> · {page_size} {label}</div>", unsafe_allow_html=True)
        with cnext:
            if st.button("Registros siguientes ▶", key=f"{key_prefix}_rnext", use_container_width=True, disabled=cur >= n_pages):
                st.session_state[state_key] = min(n_pages, cur + 1); st.rerun()
    return cur, n_pages, (cur - 1) * page_size, min(total_rows, cur * page_size)


def _dataframe_record_slice(df, start=0, end=20, record_col="Registro"):
    """Recorta SOLO la visualización a un bloque de registros cronológicos.

    Los cálculos y el histórico completo permanecen intactos. Pensado para 100-200+
    visitas sin construir figuras gigantes en cada rerun de Streamlit.
    """
    if df is None or df.empty or record_col not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    tmp = df.copy()
    if "Fecha_hora" in tmp.columns:
        tmp["__dt_v1543"] = pd.to_datetime(tmp["Fecha_hora"], errors="coerce")
        order = (tmp[[record_col, "__dt_v1543"]].drop_duplicates(record_col)
                 .sort_values("__dt_v1543", na_position="last")[record_col].astype(str).tolist())
        tmp = tmp.drop(columns=["__dt_v1543"], errors="ignore")
    else:
        order = sorted(tmp[record_col].dropna().astype(str).unique().tolist(),
                       key=lambda r: (extract_datetime_from_name(str(r)), str(r)))
    wanted = set(order[int(start):int(end)])
    return tmp[tmp[record_col].astype(str).isin(wanted)].copy()


def _record_count(df, record_col="Registro"):
    if df is None or df.empty or record_col not in df.columns:
        return 0
    return int(df[record_col].dropna().astype(str).nunique())


def _record_data_visit_slice(record_data, patient_ids=None, start=0, end=6):
    """Conserva solo el bloque cronológico visible por paciente."""
    groups = _records_grouped_by_patient(record_data.keys())
    wanted = set(normalize_patient_id(p) for p in patient_ids) if patient_ids else set(groups.keys())
    selected = []
    for pid, recs in groups.items():
        if normalize_patient_id(pid) in wanted:
            selected.extend(recs[int(start):int(end)])
    return {r: record_data[r] for r in selected if r in record_data}


def _pairs_visit_slice(pairs, patient_ids=None, start=0, end=6):
    groups = {}
    wanted = set(normalize_patient_id(p) for p in patient_ids) if patient_ids else None
    for p in pairs or []:
        rec = str(p.get("Registro", ""))
        pid = normalize_patient_id(infer_patient_id(rec))
        if wanted is not None and pid not in wanted:
            continue
        groups.setdefault(pid, []).append(p)
    out = []
    for pid, vals in groups.items():
        vals = sorted(vals, key=lambda q: (extract_datetime_from_name(str(q.get("Registro",""))), str(q.get("Registro","")), str(q.get("Fase",""))))
        out.extend(vals[int(start):int(end)])
    return out


def _record_data_for_patients(record_data, patient_ids):
    wanted = {normalize_patient_id(p) for p in (patient_ids or [])}
    if not wanted:
        return {}
    out = {}
    for rec, data in record_data.items():
        try:
            pid = normalize_patient_id(infer_patient_id(str(rec)))
        except Exception:
            pid = normalize_patient_id(str(rec))
        if pid in wanted:
            out[rec] = data
    return out


def _interp_line_from_phase_values(phases, values, points=160):
    """
    Línea suavizada segura. Usa interpolación lineal si hay pocos puntos.
    Evita dependencias gráficas raras en Streamlit.
    """
    x = np.arange(len(phases), dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)

    if np.sum(mask) == 0:
        return [], []

    if np.sum(mask) == 1:
        return x[mask], y[mask]

    xs = np.linspace(x[mask].min(), x[mask].max(), points)
    ys = np.interp(xs, x[mask], y[mask])
    return xs, ys



def _add_subplot_side_legend(fig, row, col, items, title=None, x_pad=0.018, y_pad=0.018):
    """
    Leyenda manual dentro del subplot, en su esquina superior derecha.

    Motivo:
    La versión anterior colocaba algunas leyendas en coordenadas paper fuera del
    panel correspondiente. Esta versión usa los dominios reales del subplot y
    ancla la leyenda dentro del área del gráfico para que pertenezca visualmente
    a su panel y no se desplace al margen inferior.
    """
    try:
        # Plotly >=5: get_subplot devuelve un objeto con xaxis/yaxis y dominios.
        subplot = fig.get_subplot(row, col)
        xdom = subplot.xaxis.domain
        ydom = subplot.yaxis.domain
    except Exception:
        try:
            # Fallback por numeración de subplots
            ncols = 2
            idx = (row - 1) * ncols + col
            xaxis_name = "xaxis" if idx == 1 else f"xaxis{idx}"
            yaxis_name = "yaxis" if idx == 1 else f"yaxis{idx}"
            xdom = getattr(fig.layout, xaxis_name).domain
            ydom = getattr(fig.layout, yaxis_name).domain
        except Exception:
            return

    # Posición dentro del área del subplot, no fuera.
    x0 = xdom[1] - x_pad
    y0 = ydom[1] - y_pad

    # Caja semitransparente para legibilidad
    legend_text = ""
    if title:
        legend_text += f"<b>{title}</b><br>"

    for label, color, symbol in items:
        legend_text += f"<span style='color:{color}; font-size:14px'>{symbol}</span> {label}<br>"

    fig.add_annotation(
        x=x0,
        y=y0,
        xref="paper",
        yref="paper",
        text=legend_text,
        showarrow=False,
        align="left",
        xanchor="right",
        yanchor="top",
        font=dict(size=10, color="#FAFAFA"),
        bgcolor="rgba(14,17,23,0.78)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=4,
    )


def dashboard_bar_smooth(long_df, phases, params):
    """
    Dashboard evolutivo: columnas + tendencia.

    v15.4.0:
    - En comparaciones multipaciente, agrupa físicamente los registros por paciente.
    - Cada paciente tiene tendencia independiente; nunca se conectan pacientes.
    - Mantiene paneles por parámetro y el comportamiento previo para un solo paciente.
    """
    params = [p for p in params if p in long_df.columns]
    phases = [p for p in phases if p in PHASES]

    if len(params) == 0 or len(phases) == 0 or long_df.empty or "Registro" not in long_df.columns or "Fase" not in long_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Sin datos para graficar", height=450)
        return fig

    records_order = sorted(
        list(long_df["Registro"].dropna().unique()),
        key=lambda r: (extract_datetime_from_name(r), r)
    )
    multi_patient = _has_multiple_patients(records_order)
    patient_groups = _records_grouped_by_patient(records_order) if multi_patient else None

    cols = 1 if len(params) <= 3 else 2
    rows = int(np.ceil(len(params) / cols))
    h_spacing = 0.26 if cols == 2 else 0.16
    v_spacing = min(0.12, 0.9 / max(rows - 1, 1)) if rows > 1 else 0.0

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=params,
        horizontal_spacing=h_spacing, vertical_spacing=v_spacing,
    )

    one_phase = len(phases) == 1

    for idx, param in enumerate(params):
        rr = idx // cols + 1
        cc = idx % cols + 1
        dfp = long_df[long_df["Fase"].isin(phases)].copy()

        if multi_patient:
            # Posiciones comunes para este panel: bloques separados por paciente.
            rec_x = {}
            tick_vals, tick_text = [], []
            spans = []
            cursor = 0.0
            gap = 1.25
            for gi, (pid, recs) in enumerate(patient_groups.items()):
                start_x = cursor
                for rec in recs:
                    rec_x[rec] = cursor
                    tick_vals.append(cursor)
                    tick_text.append(_comparison_axis_label(rec, multiline=True, include_patient=True))
                    cursor += 1.0
                end_x = cursor - 1.0
                spans.append((pid, start_x, end_x, recs))
                if gi < len(patient_groups) - 1:
                    cursor += gap

            if one_phase:
                ph = phases[0]
                d = dfp[dfp["Fase"] == ph].set_index("Registro")
                for pi, (pid, start_x, end_x, recs) in enumerate(spans):
                    color = _export_color_for(pi)
                    px, py, custom = [], [], []
                    for rec in recs:
                        if rec not in d.index:
                            continue
                        px.append(rec_x[rec])
                        val = pd.to_numeric(d.loc[rec, param], errors="coerce") if param in d.columns else np.nan
                        py.append(float(val) if pd.notna(val) else np.nan)
                        custom.append([_patient_display_name(pid), _record_axis_label(rec, multiline=False), ph])
                    pname = _patient_display_name(pid)
                    fig.add_trace(go.Bar(
                        x=px, y=py, name=f"{pname} · columnas",
                        marker=dict(color=color), opacity=0.74,
                        showlegend=(idx == 0),
                        customdata=custom,
                        hovertemplate="Paciente: %{customdata[0]}<br>Fecha: %{customdata[1]}<br>"
                                      "Fase: %{customdata[2]}<br>" + f"{param}: " +
                                      "%{y:.3f}<extra></extra>"
                    ), row=rr, col=cc)
                    valid = [(x, y) for x, y in zip(px, py) if np.isfinite(y)]
                    if len(valid) >= 2:
                        vx = np.asarray([x for x, _ in valid], dtype=float)
                        vy = [y for _, y in valid]
                        sx, sy = _smooth_line_xy(vy)
                        sx = np.interp(sx, np.arange(len(vx), dtype=float), vx)
                        fig.add_trace(go.Scatter(
                            x=sx, y=sy, mode="lines",
                            name=f"{pname} · tendencia",
                            line=dict(width=4, color=color),
                            showlegend=(idx == 0), hoverinfo="skip"
                        ), row=rr, col=cc)
                    fig.add_trace(go.Scatter(
                        x=px, y=py, mode="markers",
                        marker=dict(size=7, color=color),
                        showlegend=False, customdata=custom,
                        hovertemplate="Paciente: %{customdata[0]}<br>Fecha: %{customdata[1]}<br>"
                                      "Fase: %{customdata[2]}<br>" + f"{param}: " +
                                      "%{y:.3f}<extra></extra>"
                    ), row=rr, col=cc)
            else:
                nph = max(1, len(phases))
                bw = min(0.72 / nph, 0.20)
                for pi, (pid, start_x, end_x, recs) in enumerate(spans):
                    pname = _patient_display_name(pid)
                    for fi, ph in enumerate(phases):
                        d = dfp[dfp["Fase"] == ph].set_index("Registro")
                        color = _export_color_for(pi * nph + fi)
                        offset = (fi - (nph - 1) / 2.0) * bw
                        px, cx, py, custom = [], [], [], []
                        for rec in recs:
                            if rec not in d.index:
                                continue
                            px.append(rec_x[rec] + offset)
                            cx.append(rec_x[rec])
                            val = pd.to_numeric(d.loc[rec, param], errors="coerce") if param in d.columns else np.nan
                            py.append(float(val) if pd.notna(val) else np.nan)
                            custom.append([pname, _record_axis_label(rec, multiline=False), ph])
                        fig.add_trace(go.Bar(
                            x=px, y=py, width=bw, name=f"{pname} · {ph}",
                            marker=dict(color=color), opacity=0.72,
                            showlegend=(idx == 0), customdata=custom,
                            hovertemplate="Paciente: %{customdata[0]}<br>Fecha: %{customdata[1]}<br>"
                                          "Fase: %{customdata[2]}<br>" + f"{param}: " +
                                          "%{y:.3f}<extra></extra>"
                        ), row=rr, col=cc)
                        valid = [(x, y) for x, y in zip(cx, py) if np.isfinite(y)]
                        if len(valid) >= 2:
                            vx = np.asarray([x for x, _ in valid], dtype=float)
                            vy = [y for _, y in valid]
                            sx, sy = _smooth_line_xy(vy)
                            sx = np.interp(sx, np.arange(len(vx), dtype=float), vx)
                            fig.add_trace(go.Scatter(
                                x=sx, y=sy, mode="lines",
                                name=f"{pname} · {ph} · tendencia",
                                line=dict(width=3.2, color=color),
                                showlegend=False, hoverinfo="skip"
                            ), row=rr, col=cc)

            for gi, (pid, start_x, end_x, recs) in enumerate(spans):
                if gi < len(spans) - 1:
                    next_start = spans[gi + 1][1]
                    # add_vline cannot target a subplot reliably in all Plotly versions;
                    # use a shape tied to the subplot axis instead through add_vline row/col.
                    try:
                        fig.add_vline(
                            x=(end_x + next_start) / 2.0,
                            line_width=1, line_dash="dot", opacity=0.35,
                            row=rr, col=cc
                        )
                    except Exception:
                        pass

            fig.update_xaxes(
                title_text="Paciente · fecha del registro",
                tickmode="array", tickvals=tick_vals, ticktext=tick_text,
                tickangle=0, row=rr, col=cc
            )
        else:
            color = _export_color_for(idx)
            if one_phase:
                ph = phases[0]
                d = dfp[dfp["Fase"] == ph].set_index("Registro")
                labels, y_vals = [], []
                for rec in records_order:
                    if rec in d.index:
                        labels.append(_comparison_axis_label(rec, multiline=True, include_patient=False))
                        val = pd.to_numeric(d.loc[rec, param], errors="coerce") if param in d.columns else np.nan
                        y_vals.append(float(val) if pd.notna(val) else np.nan)
                x_num = np.arange(len(labels), dtype=float)
                fig.add_trace(go.Bar(
                    x=x_num, y=y_vals, name=f"{param} columnas",
                    marker=dict(color=color), opacity=0.72, showlegend=False,
                    customdata=labels,
                    hovertemplate="Registro: %{customdata}<br>Fase: " + ph +
                                  f"<br>{param}: " + "%{y:.3f}<extra></extra>"
                ), row=rr, col=cc)
                sx, sy = _smooth_line_xy(y_vals)
                fig.add_trace(go.Scatter(
                    x=sx, y=sy, mode="lines", name=f"{param} tendencia suavizada",
                    line=dict(width=4, color=color), showlegend=False, hoverinfo="skip"
                ), row=rr, col=cc)
                fig.add_trace(go.Scatter(
                    x=x_num, y=y_vals, mode="markers",
                    marker=dict(size=7, color=color), showlegend=False,
                    customdata=labels,
                    hovertemplate="Registro: %{customdata}<br>Fase: " + ph +
                                  f"<br>{param}: " + "%{y:.3f}<extra></extra>"
                ), row=rr, col=cc)
                fig.update_xaxes(
                    title_text="Fecha del registro", tickmode="array",
                    tickvals=list(x_num), ticktext=labels, tickangle=0,
                    row=rr, col=cc
                )
            else:
                labels, y_vals, custom = [], [], []
                for ph in phases:
                    d = dfp[dfp["Fase"] == ph].set_index("Registro")
                    for rec in records_order:
                        if rec in d.index:
                            labels.append(f"{ph}<br>{_comparison_axis_label(rec, multiline=True, include_patient=False)}")
                            val = pd.to_numeric(d.loc[rec, param], errors="coerce") if param in d.columns else np.nan
                            y_vals.append(float(val) if pd.notna(val) else np.nan)
                            custom.append([ph, rec])
                x_num = np.arange(len(labels), dtype=float)
                fig.add_trace(go.Bar(
                    x=x_num, y=y_vals, name=f"{param} columnas",
                    marker=dict(color=color), opacity=0.72, showlegend=False,
                    customdata=custom,
                    hovertemplate="Fase: %{customdata[0]}<br>Registro: %{customdata[1]}<br>" +
                                  f"{param}: " + "%{y:.3f}<extra></extra>"
                ), row=rr, col=cc)
                sx, sy = _smooth_line_xy(y_vals)
                fig.add_trace(go.Scatter(
                    x=sx, y=sy, mode="lines", name=f"{param} tendencia suavizada",
                    line=dict(width=4, color=color), showlegend=False, hoverinfo="skip"
                ), row=rr, col=cc)
                fig.update_xaxes(
                    title_text="Fase · fecha", tickmode="array",
                    tickvals=list(x_num), ticktext=labels, tickangle=0,
                    row=rr, col=cc
                )

        if not multi_patient:
            _add_subplot_side_legend(
                fig, rr, cc,
                [("Columnas", _export_color_for(idx), "■"),
                 ("Tendencia", _export_color_for(idx), "━")],
                title=param
            )

    fig.update_layout(
        height=max(480, rows * 410),
        title="Dashboard evolutivo: columnas verticales + línea suavizada",
        barmode="group", hovermode="closest",
        margin=dict(l=55, r=40, t=80, b=120),
        legend_title_text="Paciente · fase" if multi_patient else "Parámetro"
    )
    return fig

def phase_rr_overlay(record_data, global_windows, record_windows, phase, use_independent):
    fig = go.Figure()

    for rec, data in record_data.items():
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)

        if w is None:
            continue

        s, e = w
        seg = cut_segment(data["rr"], s, e)

        if len(seg) < 3:
            continue

        t = cumulative_time(seg)
        t = t - t[0]
        fig.add_trace(go.Scatter(x=t / 60, y=seg * 1000, mode="lines", name=rec))

    fig.update_layout(height=440, title=f"RRi superpuesto dentro de {phase}", xaxis_title="Tiempo dentro de fase (min)", yaxis_title="RRi (ms)")

    return fig


def windows_table(global_windows, record_windows, records, record_data, records_segments, records_valid, use_independent):
    """Tabla de ventanas tolerante a históricos congelados.

    En un snapshot histórico se conservan resultados ya calculados, pero no es necesario
    reconstruir ``records_segments``. Esta función obtiene N/validez desde el segmento
    cuando existe y, si no, desde el DataFrame congelado (por ejemplo N_RRi).
    """
    rows = []

    for ph in PHASES:
        row = {"Fase": ph}

        if not use_independent:
            w = global_windows.get(ph)
            if w is None:
                row.update({"Inicio": "", "Fin": "", "Duración_min": np.nan})
            else:
                row.update({"Inicio": sec_to_hms(w[0]), "Fin": sec_to_hms(w[1]), "Duración_min": round((w[1] - w[0]) / 60, 2)})

        for rec in records:
            w = get_record_windows(global_windows, record_windows, rec, use_independent).get(ph)
            if use_independent:
                row[f"{rec}_inicio"] = sec_to_hms(w[0]) if w else ""
                row[f"{rec}_fin"] = sec_to_hms(w[1]) if w else ""

            seg_map = records_segments.get(rec, {}) or {}
            valid_map = records_valid.get(rec, {})
            phase_seg = seg_map.get(ph) if isinstance(seg_map, dict) else None

            n_val = len(phase_seg) if phase_seg is not None else None
            ok_val = False
            if isinstance(valid_map, dict):
                ok_val = bool(valid_map.get(ph, False))
            elif isinstance(valid_map, (list, tuple, set, pd.Index, np.ndarray)):
                ok_val = ph in list(valid_map)

            # Historial congelado: recuperar N/validez directamente de los resultados guardados.
            if n_val is None:
                snap = (record_data.get(rec, {}) or {}).get("analysis_snapshot")
                snap_df = snap.get("results") if isinstance(snap, dict) else None
                if isinstance(snap_df, pd.DataFrame) and ph in snap_df.index:
                    ok_val = True
                    try:
                        raw_n = snap_df.loc[ph, "N_RRi"] if "N_RRi" in snap_df.columns else np.nan
                        if isinstance(raw_n, pd.Series):
                            raw_n = raw_n.iloc[0]
                        n_val = int(round(float(raw_n))) if pd.notna(raw_n) else 0
                    except Exception:
                        n_val = 0
                else:
                    n_val = 0

            row[f"{rec}_N"] = int(n_val)
            row[f"{rec}_OK"] = bool(ok_val)

        rows.append(row)

    return enforce_entropy_dataframe_consistency(pd.DataFrame(rows))



def _fmt_num(x, digits=2):
    try:
        if pd.isna(x):
            return "no calculado"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _arrow_change(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or a == 0:
            return "no calculable"
        pct = 100 * (b - a) / abs(a)
        arrow = "↑" if pct > 5 else ("↓" if pct < -5 else "≈")
        return f"{arrow} {pct:.1f}%"
    except Exception:
        return "no calculable"


def _interpret_metric(metric, value):
    """
    Interpreta sólo métricas fisiológicas numéricas.
    v11.9.1: evita errores cuando el valor es None, texto o parámetro de configuración.
    """
    if metric in [
        "MSE_zero_policy", "MSE_radius_mode", "DFA_alpha1_range", "DFA_alpha2_range",
        "RQA_threshold_SD", "RQA_emb_dim"
    ]:
        return "Parámetro de configuración del análisis, no métrica fisiológica."

    if value is None:
        return "No calculado o ventana insuficiente."

    try:
        if pd.isna(value):
            return "No calculado o ventana insuficiente."
    except Exception:
        pass

    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in ["", "none", "nan", "no calculado", "no calculado o ventana insuficiente"]:
            return "No calculado o ventana insuficiente."
        return "Valor textual/configuración; no requiere interpretación fisiológica numérica."

    try:
        v = float(value)
    except Exception:
        return "Valor no numérico; no requiere interpretación fisiológica."


    if metric == "SDNN":
        if v < 30:
            return "SDNN bajo: menor variabilidad global y menor reserva adaptativa cardiovascular."
        if v < 50:
            return "SDNN moderadamente reducido: posible disminución de variabilidad global."
        return "SDNN conservado/alto: mayor variabilidad global."
    if metric == "RMSSD":
        if v < 15:
            return "RMSSD bajo: menor modulación vagal rápida."
        if v < 30:
            return "RMSSD moderado-bajo: posible reducción parasimpática."
        return "RMSSD conservado/alto: modulación vagal relativamente preservada."
    if metric == "pNN50":
        return "pNN50 refleja variabilidad rápida latido a latido."
    if metric == "HF":
        return "HF se interpreta principalmente como modulación vagal respiratoria, junto con RMSSD y SD1."
    if metric == "LF":
        return "LF se relaciona con oscilaciones barorreflejas y modulación autonómica mixta."
    if metric == "VLF":
        return "VLF refleja oscilaciones lentas, relacionadas con regulación sistémica lenta."
    if metric == "TOTAL":
        return "TOTAL resume la potencia espectral global y la reserva autonómica frecuencial."
    if metric == "SD1":
        return "SD1 representa variabilidad rápida, muy relacionada con RMSSD."
    if metric == "SD2":
        return "SD2 representa variabilidad de más largo plazo en Poincaré."
    if metric == "DFA_alpha1":
        if v < 0.6:
            return "DFA α1 bajo: patrón más aleatorio/menos correlacionado a corto plazo."
        if v > 1.4:
            return "DFA α1 alto: tendencia a mayor rigidez/correlación."
        return "DFA α1 intermedio: organización fractal a corto plazo relativamente conservada."
    if metric == "DFA_alpha2":
        return "DFA α2 describe correlaciones fractales de más largo plazo."
    if metric in ["ApEn", "SampEn"]:
        if v < 0.5:
            return f"{metric} bajo: señal más regular y menos impredecible."
        return f"{metric} relativamente mayor: más irregularidad/complejidad."
    if metric == "REC":
        return "REC alto indica mayor recurrencia: más repetición de estados."
    if metric == "DET":
        return "DET alto indica trayectorias más deterministas y predecibles."
    if metric == "Lmax":
        return "Lmax alto se asocia a secuencias repetitivas más largas."
    if metric == "ShanEn":
        return "ShanEn resume diversidad de longitudes diagonales; mayor valor sugiere mayor variedad dinámica."

    if metric == "Lyapunov_LLE":
        return lyapunov_interpretation(v)
    if metric == "Hurst":
        if v < 0.45:
            return "Hurst <0.45: antipersistencia; la señal tiende a invertir cambios previos."
        if v <= 0.60:
            return "Hurst 0.45-0.60: comportamiento próximo a aleatorio/equilibrado."
        if v <= 0.80:
            return "Hurst 0.60-0.80: persistencia moderada; memoria temporal de largo plazo."
        return "Hurst >0.80: persistencia alta; posible predominio de regulación lenta o rigidez dinámica."
    if metric == "KatzFD":
        if v < 1.2:
            return "KatzFD bajo: trayectoria geométrica simple."
        if v < 2.0:
            return "KatzFD intermedio: complejidad geométrica moderada."
        return "KatzFD alto: trayectoria más tortuosa; comparar longitudinalmente y con artefactos."
    if metric == "PetrosianFD":
        if v < 1.02:
            return "PetrosianFD bajo: pocos cambios de dirección."
        if v < 1.08:
            return "PetrosianFD moderado: variación local presente."
        return "PetrosianFD alto: más cambios de dirección; posible mayor irregularidad local."
    if metric == "DispEn":
        if v < 2.5:
            return "DispEn bajo: menor diversidad de patrones."
        if v < 4.0:
            return "DispEn intermedio: diversidad de patrones moderada."
        return "DispEn alto: elevada diversidad de patrones; suele ser estable en ventanas cortas."
    if metric.startswith("MDE"):
        return "MDE es Dispersion Entropy multiescala; valores más altos indican mayor diversidad de patrones en esa escala."
    if metric.endswith("_LS"):
        return "Lomb-Scargle: potencia espectral calculada sin interpolar RRi; útil como contraste en señales irregulares."
    if metric.endswith("_AR"):
        return "AR/Yule-Walker: estimación espectral autorregresiva; puede definir picos LF/HF en ventanas cortas."
    if metric.endswith("_WAV"):
        return "Wavelet/STFT: potencia tiempo-frecuencia media; el scalogram muestra cuándo aparece/desaparece LF/HF."


    if metric == "HF_DOM_PCT":
        if v >= 50:
            return "HF_DOM_PCT alto: predominio respiratorio/vagal relativo tras normalización por la media de HF."
        if v < 20:
            return "HF_DOM_PCT bajo: menor presencia sostenida de la banda respiratoria/vagal."
        return "HF_DOM_PCT intermedio: presencia vagal respiratoria parcial."
    if metric == "LF_DOM_PCT":
        if v >= 50:
            return "LF_DOM_PCT alto: predominio relativo de oscilaciones LF/barorreflejas tras normalización por la media de LF."
        if v < 20:
            return "LF_DOM_PCT bajo: menor presencia barorrefleja organizada."
        return "LF_DOM_PCT intermedio: presencia LF parcial."
    if metric == "VLF_DOM_PCT":
        if v >= 50:
            return "VLF_DOM_PCT alto: predominio relativo de regulación lenta tras normalización; valorar termorregulación, fatiga, RAS, inflamación o no estacionariedad."
        if v < 20:
            return "VLF_DOM_PCT bajo: menor dominio de oscilaciones lentas."
        return "VLF_DOM_PCT intermedio: presencia moderada de regulación lenta."
    if metric == "WAV_ENTROPY_BANDS":
        if v >= 0.75:
            return "WAV_ENTROPY_BANDS alta: energía/dominancia distribuida entre varias bandas."
        if v < 0.40:
            return "WAV_ENTROPY_BANDS baja: energía concentrada en una banda dominante."
        return "WAV_ENTROPY_BANDS intermedia: distribución parcial entre bandas."
    if metric == "WAV_ENTROPY_GLOBAL":
        if v >= 0.75:
            return "WAV_ENTROPY_GLOBAL alta: scalogram con energía distribuida en tiempo y frecuencia."
        if v < 0.40:
            return "WAV_ENTROPY_GLOBAL baja: energía concentrada, patrón más fijo."
        return "WAV_ENTROPY_GLOBAL intermedia."
    if metric == "WAV_TRANSITIONS_N":
        if v >= 6:
            return "TRANSITIONS_N alto: mayor movilidad entre regímenes VLF/LF/HF."
        if v <= 1:
            return "TRANSITIONS_N bajo: dominio más fijo o rígido."
        return "TRANSITIONS_N moderado: movilidad intermedia entre regímenes."
    if metric == "WAV_TRANSITIONS_PER_MIN":
        if v >= 2:
            return "TRANSITIONS_PER_MIN alto: alta movilidad entre bandas dominantes."
        if v < 0.5:
            return "TRANSITIONS_PER_MIN bajo: régimen dominante estable."
        return "TRANSITIONS_PER_MIN moderado."
    if "_EPISODES_N" in metric:
        return "Número de episodios en los que esa banda fue dominante."
    if "_EPISODE_MEAN_S" in metric:
        return "Duración media en segundos de los episodios de dominancia de esa banda."
    if "_EPISODE_MAX_S" in metric:
        return "Duración máxima en segundos de un episodio dominante de esa banda."
    if metric.endswith("_WAV_MEAN"):
        return "Potencia wavelet/STFT media de la banda durante la ventana."
    if metric.endswith("_WAV_SD"):
        return "Variabilidad temporal de la potencia wavelet/STFT de la banda; alto = ráfagas o fluctuaciones."

    if metric.startswith("HVG_"):
        return "Métrica HVG: describe la topología de la señal RRi transformada en red."
    return ""



# ============================================================
# TABLAS DE REFERENCIA, VALOR OBTENIDO E INTERPRETACIÓN v11.9
# ============================================================

def _metric_reference(metric):
    """
    Valores normales/orientativos.

    Importante:
    - En HRV no todos los parámetros tienen normalidad universal.
    - Los clásicos de 5 min se basan en rangos orientativos habituales.
    - Los parámetros modernos se interpretan mejor longitudinalmente y por fase.
    """
    refs = {
        "MeanHR": "Reposo adulto orientativo: 60-100 lpm; deportistas/ancianos/medicación pueden salir fuera.",
        "MeanRR": "Depende de FC; ≈600-1000 ms en reposo 60-100 lpm.",
        "SDNN": "5 min: >50 ms conservado; 30-50 ms moderado; <30 ms bajo.",
        "RMSSD": "5 min: >30 ms conservado; 15-30 ms moderado-bajo; <15 ms bajo.",
        "pNN50": "No hay corte universal; valores más altos suelen reflejar mayor variabilidad rápida.",
        "SD1": "Relacionado con RMSSD; SD1≈RMSSD/√2. Mayor = variabilidad rápida preservada.",
        "SD2": "Variabilidad de más largo plazo; comparar por fase y longitudinalmente.",
        "VLF": "5 min: interpretación limitada; alto = mayor peso de oscilaciones lentas.",
        "LF": "0.04-0.15 Hz; interpretar como oscilación barorrefleja/mixta, no simpático aislado.",
        "HF": "0.15-0.40 Hz; suele reflejar modulación vagal-respiratoria.",
        "TOTAL": "Potencia total 0.0033-0.40 Hz; mayor = mayor reserva frecuencial.",
        "LF_HF": "No usar como balance simpático-vagal aislado; interpretar con LF, HF y respiración.",
        "DFA_alpha1": "5 min: ≈0.75-1.25 fisiológico orientativo; <0.6 aleatorización; >1.4 rigidez/correlación alta.",
        "DFA_alpha2": "Correlaciones largas; referencia dependiente de ventana. Comparar longitudinalmente.",
        "D2": "Dimensionalidad del atractor; mayor = dinámica más compleja. Sensible a longitud de señal.",
        "ApEn": "Mayor = más irregularidad; menor = más regularidad. Muy dependiente de N y parámetros.",
        "SampEn": "Mayor = más irregularidad/complejidad; menor = regularidad. En 5 min interpretar con MSE/MDE.",
        "Lyapunov_LLE": "<0.03 rígido; 0.03-0.15 adaptabilidad fisiológica; 0.15-0.30 divergencia alta; >0.30 posible ruido/arritmia.",
        "Hurst": "≈0.5 aleatorio; >0.5 persistente; <0.5 antipersistente.",
        "KatzFD": "Sin corte universal; mayor = trayectoria más tortuosa. Uso comparativo.",
        "PetrosianFD": "Sin corte universal; mayor = más cambios de dirección. Uso comparativo.",
        "DispEn": "<2.5 baja; 2.5-4 moderada; >4 alta diversidad orientativa.",
        "REC": "Mayor REC = más repetición de estados; interpretar junto a DET/Lmax.",
        "DET": "DET alto = trayectorias más deterministas/predecibles.",
        "Lmean": "Longitud media de diagonales; mayor = secuencias repetitivas más largas.",
        "Lmax": "Secuencia determinista máxima; mayor = dinámica más repetitiva/prolongada.",
        "ShanEn": "Diversidad de diagonales RQA; mayor = mayor variedad dinámica.",
        "VLF_LS": "Lomb-Scargle VLF; contraste sin interpolación.",
        "LF_LS": "Lomb-Scargle LF; contraste sin interpolación.",
        "HF_LS": "Lomb-Scargle HF; contraste sin interpolación.",
        "TOTAL_LS": "Potencia total Lomb-Scargle; comparar con Welch.",
        "LF_HF_LS": "Relación LF/HF Lomb-Scargle; interpretación prudente.",
        "VLF_AR": "AR VLF; depende del orden del modelo.",
        "LF_AR": "AR LF; puede definir picos en ventanas cortas.",
        "HF_AR": "AR HF; contraste del componente respiratorio.",
        "TOTAL_AR": "Potencia total AR; comparar con Welch/LS.",
        "LF_HF_AR": "Relación LF/HF AR; sensible al modelo.",
        "VLF_WAV_MEAN": "Potencia media VLF tiempo-frecuencia; alto = peso medio lento.",
        "LF_WAV_MEAN": "Potencia media LF tiempo-frecuencia; alto = peso medio LF.",
        "HF_WAV_MEAN": "Potencia media HF tiempo-frecuencia; alto = peso medio vagal-respiratorio.",
        "VLF_WAV_SD": "SD temporal VLF; alto = VLF fluctuante/en ráfagas.",
        "LF_WAV_SD": "SD temporal LF; alto = LF fluctuante/en ráfagas.",
        "HF_WAV_SD": "SD temporal HF; alto = HF fluctuante/en ráfagas.",
        "VLF_DOM_PCT": "Dominancia normalizada: >50% alto; 20-50% intermedio; <20% bajo.",
        "LF_DOM_PCT": "Dominancia normalizada: >50% alto; 20-50% intermedio; <20% bajo.",
        "HF_DOM_PCT": "Dominancia normalizada: >50% alto; 20-50% intermedio; <20% bajo.",
        "VLF_EPISODES_N": "Número de episodios VLF dominantes; alto = más alternancia.",
        "LF_EPISODES_N": "Número de episodios LF dominantes; alto = más alternancia.",
        "HF_EPISODES_N": "Número de episodios HF dominantes; alto = más alternancia.",
        "VLF_EPISODE_MEAN_S": "Duración media de episodios VLF; mayor = dominio lento sostenido.",
        "LF_EPISODE_MEAN_S": "Duración media de episodios LF; mayor = dominio LF sostenido.",
        "HF_EPISODE_MEAN_S": "Duración media de episodios HF; mayor = dominio HF sostenido.",
        "VLF_EPISODE_MAX_S": "Duración máxima VLF; mayor = periodo lento prolongado.",
        "LF_EPISODE_MAX_S": "Duración máxima LF; mayor = periodo LF prolongado.",
        "HF_EPISODE_MAX_S": "Duración máxima HF; mayor = periodo vagal-respiratorio prolongado.",
        "WAV_TRANSITIONS_N": "0-1 bajo; 2-5 moderado; ≥6 alto orientativo.",
        "WAV_TRANSITIONS_PER_MIN": "<0.5 bajo; 0.5-2 moderado; ≥2 alto orientativo.",
        "WAV_ENTROPY_BANDS": "0-0.4 baja; 0.4-0.75 intermedia; >0.75 alta.",
        "WAV_ENTROPY_GLOBAL": "0-0.4 baja; 0.4-0.75 intermedia; >0.75 alta.",
    }

    if metric.startswith("MSE"):
        return "MSE multiescala: mayor = mayor complejidad en esa escala; None si SampEn clásico no tiene coincidencias suficientes."
    if metric.startswith("MDE"):
        return "MDE multiescala: mayor = mayor diversidad de patrones; más estable que MSE en ventanas cortas."
    if metric.startswith("HVG_"):
        return "Métrica de grafo HVG: sin normalidad universal; interpretar por fase y longitudinalmente."

    return refs.get(metric, "Sin rango universal; interpretar por fase, longitudinalmente y junto al contexto clínico.")



def _is_interpretable_metric(metric, value=None):
    """
    Excluye campos de configuración y valores textuales de las tablas clínicas.
    """
    skip_exact = {
        "MSE_zero_policy", "MSE_radius_mode", "DFA_alpha1_range", "DFA_alpha2_range",
        "RQA_threshold_SD", "RQA_emb_dim"
    }
    if metric in skip_exact:
        return False
    if str(metric).startswith("_"):
        return False

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in ["", "none", "nan", "no calculado", "no calculado o ventana insuficiente"]:
            return True
        # Textual settings should not be included as clinical metrics.
        return False

    try:
        float(value)
        return True
    except Exception:
        return False


def reference_interpretation_table(metrics_df, phase=None, metrics=None):
    """
    Tabla larga: Métrica | Referencia | Valor obtenido | Interpretación.
    v11.9.1 filtra campos de configuración y valores textuales.
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=["Fase", "Métrica", "Referencia", "Valor obtenido", "Interpretación"])

    phases = [phase] if phase in metrics_df.index else list(metrics_df.index)
    if metrics is None:
        metrics = [c for c in metrics_df.columns if not str(c).startswith("_")]

    rows = []
    for ph in phases:
        for m in metrics:
            if m not in metrics_df.columns:
                continue
            val = metrics_df.loc[ph, m]

            if not _is_interpretable_metric(m, val):
                continue

            rows.append({
                "Fase": ph,
                "Métrica": m,
                "Referencia": _metric_reference(m),
                "Valor obtenido": val,
                "Interpretación": _interpret_metric(m, val),
            })

    return pd.DataFrame(rows)



def _single_record_report(record_name, metrics_df, windows, rr=None):
    lines = []
    lines.append(f"## Registro: {record_name}")
    lines.append("")
    if metrics_df is None or metrics_df.empty:
        lines.append("No hay ventanas válidas suficientes para generar interpretación.")
        return "\n".join(lines)

    phases = [p for p in PHASES if p in metrics_df.index]
    ref = "Basal" if "Basal" in metrics_df.index else phases[0]
    base = metrics_df.loc[ref]

    lines.append("### Ventanas analizadas")
    lines.append("")
    lines.append("| Fase | Inicio | Fin | Duración min |")
    lines.append("|---|---:|---:|---:|")
    for ph in phases:
        w = windows.get(ph)
        if w is not None:
            lines.append(f"| {ph} | {sec_to_hms(w[0])} | {sec_to_hms(w[1])} | {(w[1]-w[0])/60:.2f} |")
    lines.append("")

    lines.append("### Resumen ejecutivo")
    lines.append("")
    lines.append(
        f"Se analiza **{record_name}** usando como referencia la fase **{ref}**. "
        "La lectura integra HRV temporal, frecuencia clásica y avanzada, complejidad moderna, recurrencia, MDE, Lyapunov y grafos HVG si están disponibles."
    )
    lines.append("")

    metrics = [
        "MeanHR","SDNN","RMSSD","pNN50","SD1","SD2",
        "VLF","LF","HF","TOTAL","LF_HF",
        "VLF_LS","LF_LS","HF_LS","TOTAL_LS","LF_HF_LS",
        "VLF_AR","LF_AR","HF_AR","TOTAL_AR","LF_HF_AR",
        "LF_WAV","HF_WAV","LF_HF_WAV",
        "DFA_alpha1","DFA_alpha2","D2","ApEn","SampEn",
        "Lyapunov_LLE","Hurst","KatzFD","PetrosianFD","DispEn",
        "REC","DET","Lmean","Lmax","ShanEn",
        "HVG_edges","HVG_degree_mean","HVG_degree_max","HVG_hubs_p90","HVG_clustering","HVG_lambda","HVG_path_length","HVG_diameter"
    ] + [f"MSE{i}" for i in range(1,21)] + [f"MDE{i}" for i in range(1,21)]
    metrics = [m for m in metrics if m in metrics_df.columns]

    lines.append("### Valores principales por fase")
    lines.append("")
    lines.append("| Parámetro | Referencia/normalidad | " + " | ".join(phases) + " | Interpretación referencia |")
    lines.append("|---|---|" + "|".join(["---:"]*len(phases)) + "|---|")
    for m in metrics:
        if m in base.index and not _is_interpretable_metric(m, base[m]):
            continue
        vals = [_fmt_num(metrics_df.loc[ph, m]) if ph in metrics_df.index else "" for ph in phases]
        interp = _interpret_metric(m, base[m]) if m in base.index else ""
        ref_txt = _metric_reference(m)
        lines.append("| " + m + " | " + ref_txt + " | " + " | ".join(vals) + " | " + interp + " |")
    lines.append("")

    # Tabla completa de referencia / valor / interpretación para fase de referencia
    lines.append("### Tabla clínica: referencia, valor obtenido e interpretación")
    lines.append("")
    lines.append(f"Se muestran los valores de la fase de referencia **{ref}**. Los rangos son orientativos y no sustituyen la interpretación clínica ni la comparación longitudinal.")
    lines.append("")
    lines.append("| Métrica | Referencia/normalidad | Valor obtenido | Interpretación |")
    lines.append("|---|---|---:|---|")
    for m in metrics:
        if m in base.index and _is_interpretable_metric(m, base.get(m)):
            lines.append(f"| {m} | {_metric_reference(m)} | {_fmt_num(base.get(m))} | {_interpret_metric(m, base.get(m))} |")
    lines.append("")

    # Métodos modernos añadidos
    modern_cols = [
        "Lyapunov_LLE", "Hurst", "KatzFD", "PetrosianFD", "DispEn",
        "VLF_LS", "LF_LS", "HF_LS", "TOTAL_LS", "LF_HF_LS",
        "VLF_AR", "LF_AR", "HF_AR", "TOTAL_AR", "LF_HF_AR",
        "LF_WAV", "HF_WAV", "LF_HF_WAV",
    ]
    modern_present = [c for c in modern_cols if c in metrics_df.columns]
    if modern_present:
        lines.append("### Métricas modernas: definición, referencia e interpretación")
        lines.append("")
        lines.append("Estas métricas no sustituyen a los parámetros clásicos de Kubios, sino que añaden información sobre estabilidad dinámica, fractalidad, diversidad de patrones y análisis frecuencial alternativo.")
        lines.append("")
        lines.append("| Métrica | Qué mide | Referencia orientativa | Valor referencia | Interpretación |")
        lines.append("|---|---|---|---:|---|")
        definitions = {
            "Lyapunov_LLE": ("Estabilidad dinámica; velocidad de separación de trayectorias vecinas mediante Rosenstein.", "<0.03 rígido; 0.03-0.15 adaptabilidad fisiológica; 0.15-0.30 divergencia alta; >0.30 posible ruido/arritmia/no estacionariedad."),
            "Hurst": ("Memoria/persistencia de largo plazo.", "≈0.5 aleatorio; >0.5 persistente; <0.5 antipersistente."),
            "KatzFD": ("Dimensión fractal geométrica/tortuosidad de la serie.", "Más alto = trayectoria más tortuosa; usar sobre todo comparativamente."),
            "PetrosianFD": ("Complejidad por cambios de signo en la derivada.", "Más alto = más cambios locales de dirección."),
            "DispEn": ("Diversidad de patrones simbólicos; alternativa robusta a SampEn en ventanas cortas.", "<2.5 baja; 2.5-4 moderada; >4 alta diversidad orientativa."),
            "VLF_LS": ("Potencia VLF por Lomb-Scargle sin interpolación previa.", "Contraste frente a Welch; útil si RRi es irregular."),
            "LF_LS": ("Potencia LF por Lomb-Scargle.", "Contraste de oscilaciones barorreflejas/mixtas sin interpolación."),
            "HF_LS": ("Potencia HF por Lomb-Scargle.", "Contraste de modulación vagal respiratoria sin interpolación."),
            "TOTAL_LS": ("Potencia total por Lomb-Scargle.", "Comparar con TOTAL Welch; diferencias grandes sugieren efecto de interpolación/irregularidad."),
            "LF_HF_LS": ("Relación LF/HF por Lomb-Scargle.", "Índice orientativo; no usar como balance simpático-vagal aislado."),
            "VLF_AR": ("Potencia VLF por modelo autorregresivo.", "Puede resaltar picos en ventanas cortas; depende del orden del modelo."),
            "LF_AR": ("Potencia LF por modelo autorregresivo.", "Estimación alternativa de LF."),
            "HF_AR": ("Potencia HF por modelo autorregresivo.", "Estimación alternativa de HF."),
            "TOTAL_AR": ("Potencia total por modelo autorregresivo.", "Comparar con Welch y Lomb-Scargle."),
            "LF_HF_AR": ("Relación LF/HF por AR.", "Interpretación prudente; sensible al modelo."),
            "VLF_WAV_MEAN": ("Potencia media VLF en STFT/wavelet.", "Mayor valor = mayor peso medio de regulación lenta."),
            "LF_WAV_MEAN": ("Potencia media LF en STFT/wavelet.", "Mayor valor = mayor peso medio barorreflejo/LF."),
            "HF_WAV_MEAN": ("Potencia media HF en STFT/wavelet.", "Mayor valor = mayor peso respiratorio-vagal medio."),
            "VLF_WAV_SD": ("Variabilidad temporal VLF.", "Alto = VLF en ráfagas o cambiante."),
            "LF_WAV_SD": ("Variabilidad temporal LF.", "Alto = LF fluctuante/transitorio."),
            "HF_WAV_SD": ("Variabilidad temporal HF.", "Alto = HF variable por respiración o cambios vagales."),
            "VLF_DOM_PCT": ("% de tiempo con VLF dominante.", "Alto = mayor peso de regulación lenta."),
            "LF_DOM_PCT": ("% de tiempo con LF dominante.", "Alto = mayor presencia LF/barorrefleja."),
            "HF_DOM_PCT": ("% de tiempo con HF dominante.", "Alto = predominio respiratorio/vagal."),
            "WAV_TRANSITIONS_N": ("Número de cambios entre VLF/LF/HF dominantes.", "Alto = movilidad entre regímenes; bajo = dominio fijo."),
            "WAV_TRANSITIONS_PER_MIN": ("Transiciones por minuto.", "Normaliza la movilidad por duración."),
            "WAV_ENTROPY_BANDS": ("Entropía de dominancia entre VLF/LF/HF.", "Alta = energía distribuida; baja = energía concentrada."),
            "WAV_ENTROPY_GLOBAL": ("Entropía global del scalogram.", "Alta = riqueza tiempo-frecuencia; baja = concentración energética."),
            "LF_WAV": ("Potencia LF media en análisis tiempo-frecuencia STFT/wavelet-like.", "El valor medio resume la ventana; el scalogram muestra cuándo emerge LF."),
            "HF_WAV": ("Potencia HF media en análisis tiempo-frecuencia STFT/wavelet-like.", "El valor medio resume la ventana; el scalogram muestra cuándo aparece/desaparece HF."),
            "LF_HF_WAV": ("Relación LF/HF media tiempo-frecuencia.", "Útil para cambios transitorios; mirar junto al scalogram."),
        }
        for m in modern_present:
            val = base.get(m, np.nan)
            what, ref_txt = definitions.get(m, ("Métrica moderna avanzada.", "Interpretación comparativa/longitudinal."))
            lines.append(f"| {m} | {what} | {ref_txt} | {_fmt_num(val)} | {_interpret_metric(m, val)} |")
        lines.append("")

    # MDE multiescala
    mde_cols = [f"MDE{i}" for i in range(1,21) if f"MDE{i}" in metrics_df.columns]
    if mde_cols:
        lines.append("### MDE 1-20: Dispersion Entropy multiescala")
        lines.append("")
        lines.append("MDE aplica Dispersion Entropy a escalas temporales progresivamente más gruesas. Es más estable que MSE clásico en ventanas de 5 minutos porque no depende tanto de encontrar coincidencias exactas A/B.")
        lines.append("")
        lines.append("| Escala | Valor referencia | Interpretación |")
        lines.append("|---:|---:|---|")
        for c in mde_cols:
            val = base.get(c, np.nan)
            lines.append(f"| {c.replace('MDE','')} | {_fmt_num(val)} | {_interpret_metric(c, val)} |")
        lines.append("")

    # Wavelet / scalogram
    if any(c in metrics_df.columns for c in ["LF_WAV","HF_WAV","LF_HF_WAV"]):
        lines.append("### Wavelet/STFT: lectura de cambios transitorios")
        lines.append("")
        lines.append("Los valores LF_WAV, HF_WAV y LF_HF_WAV son resúmenes medios de una matriz tiempo-frecuencia. En v11.8 la dominancia VLF/LF/HF se calcula tras normalizar cada banda por su propia media temporal: VLF_n=VLF/mean(VLF), LF_n=LF/mean(LF), HF_n=HF/mean(HF). Así se detecta qué banda destaca relativamente en cada momento, no sólo cuál tiene más potencia absoluta.")
        lines.append("")
        lines.append("- **HF_WAV**: potencia respiratoria-vagal media en el tiempo.")
        lines.append("- **LF_WAV**: potencia media de oscilaciones LF/barorreflejas.")
        lines.append("- **LF_HF_WAV**: relación media tiempo-frecuencia; no equivale a diagnóstico de balance simpático-vagal.")
        lines.append("")
        wave_cols = [
            "VLF_WAV_MEAN","LF_WAV_MEAN","HF_WAV_MEAN",
            "VLF_WAV_SD","LF_WAV_SD","HF_WAV_SD",
            "VLF_DOM_PCT","LF_DOM_PCT","HF_DOM_PCT",
            "VLF_EPISODES_N","LF_EPISODES_N","HF_EPISODES_N",
            "VLF_EPISODE_MEAN_S","LF_EPISODE_MEAN_S","HF_EPISODE_MEAN_S",
            "VLF_EPISODE_MAX_S","LF_EPISODE_MAX_S","HF_EPISODE_MAX_S",
            "WAV_TRANSITIONS_N","WAV_TRANSITIONS_PER_MIN",
            "WAV_ENTROPY_BANDS","WAV_ENTROPY_GLOBAL"
        ]
        wave_cols = [c for c in wave_cols if c in metrics_df.columns]
        if wave_cols:
            lines.append("| Métrica wavelet | Valor referencia | Interpretación |")
            lines.append("|---|---:|---|")
            for c in wave_cols:
                val = base.get(c, np.nan)
                lines.append(f"| {c} | {_fmt_num(val)} | {_interpret_metric(c, val)} |")
            lines.append("")

    # Dominios normalizados
    dom = domain_values(metrics_df, method="median")
    if not dom.empty:
        lines.append("### Dominios normalizados")
        lines.append("")
        lines.append("Basal = 100%. Valores inferiores a 100% indican reducción relativa frente a Basal; superiores indican incremento relativo.")
        lines.append("")
        lines.append("| Fase | Amplitud | Vagal | Complejidad | Recurrencia |")
        lines.append("|---|---:|---:|---:|---:|")
        for ph in dom.index:
            lines.append(f"| {ph} | {_fmt_num(dom.loc[ph].get('Amplitud'))} | {_fmt_num(dom.loc[ph].get('Vagal'))} | {_fmt_num(dom.loc[ph].get('Complejidad'))} | {_fmt_num(dom.loc[ph].get('Recurrencia'))} |")
        lines.append("")

    if any(c in metrics_df.columns for c in MSE_COLUMNS):
        lines.append("### MSE 1-20")
        lines.append("")
        lines.append("La entropía multiescala evalúa complejidad a diferentes escalas temporales. Descensos amplios en varias escalas sugieren pérdida de complejidad multiescala.")
        lines.append("")

        # Diagnóstico Kubios SampEn/MSE incluido en el informe
        if rr is not None:
            lines.append("#### Diagnóstico Kubios SampEn / MSE")
            lines.append("")
            lines.append("Este bloque permite comprobar por qué algunas escalas MSE no coinciden con Kubios. Muestra el número de puntos por escala, la tolerancia usada, los conteos B/A y las tres alternativas de cálculo cuando A=0.")
            lines.append("")
            lines.append("| Fase | Escala | N | r ms | Theiler | B | A | A/B | Clásico | A0=0.5 | A0=1.0 | RCMSE | Estado |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

            for ph in phases:
                w = windows.get(ph)
                if w is None:
                    continue
                try:
                    seg_diag = cut_segment(rr, w[0], w[1])
                    if len(seg_diag) < 5:
                        continue
                    diag_df = entropy_kubios_diagnostic_table(seg_diag)
                    for _, drow in diag_df.iterrows():
                        lines.append(
                            f"| {ph} | {int(drow.get('Escala'))} | "
                            f"{_fmt_num(drow.get('N'),0)} | "
                            f"{_fmt_num(drow.get('r_ms'))} | "
                            f"{_fmt_num(drow.get('Theiler'),0)} | "
                            f"{_fmt_num(drow.get('B_matches_m'),0)} | "
                            f"{_fmt_num(drow.get('A_matches_m1'),0)} | "
                            f"{_fmt_num(drow.get('A/B'))} | "
                            f"{_fmt_num(drow.get('MSE_clasico'))} | "
                            f"{_fmt_num(drow.get('MSE_A0_05'))} | "
                            f"{_fmt_num(drow.get('MSE_A0_1'))} | "
                            f"{_fmt_num(drow.get('RCMSE'))} | "
                            f"{drow.get('Estado','')} |"
                        )
                except Exception as e:
                    lines.append(f"| {ph} |  |  |  |  |  |  |  | Error diagnóstico: {e} |")
            lines.append("")

    lines.append("### Integración HRV + grafos HVG")
    lines.append("")
    has_hvg = any(c in metrics_df.columns for c in ["HVG_edges","HVG_hubs_p90","HVG_clustering","HVG_lambda"])
    if not has_hvg:
        lines.append("Las métricas HVG/grafos no están disponibles. Activa **Calcular HVG/grafos** para incluir esta parte del informe.")
    else:
        lines.append("El HVG transforma la señal RRi en una red. Una señal con mayor riqueza temporal suele generar más diversidad de conexiones, hubs y organización topológica.")
        if "SDNN" in base.index and "HVG_edges" in base.index:
            lines.append(f"- SDNN referencia = {_fmt_num(base['SDNN'])}; aristas HVG = {_fmt_num(base['HVG_edges'],0)}. Menor variabilidad global suele asociarse a menor riqueza topológica.")
        if "RMSSD" in base.index and "HVG_hubs_p90" in base.index:
            lines.append(f"- RMSSD referencia = {_fmt_num(base['RMSSD'])}; hubs p90 = {_fmt_num(base['HVG_hubs_p90'],0)}. La variabilidad rápida puede relacionarse con nodos altamente conectados.")
        if "SampEn" in base.index and "HVG_lambda" in base.index:
            lines.append(f"- SampEn referencia = {_fmt_num(base['SampEn'])}; lambda HVG = {_fmt_num(base['HVG_lambda'])}. Menor entropía y lambda elevada pueden sugerir dinámica más regular.")
        if "HVG_clustering" in base.index:
            lines.append(f"- Clustering HVG = {_fmt_num(base['HVG_clustering'])}. Refleja agrupamiento local en la red.")
    lines.append("")

    lines.append("### Conclusión orientativa")
    flags = []
    if "SDNN" in base.index and pd.notna(base["SDNN"]) and base["SDNN"] < 50:
        flags.append("menor variabilidad global")
    if "RMSSD" in base.index and pd.notna(base["RMSSD"]) and base["RMSSD"] < 30:
        flags.append("menor modulación vagal rápida")
    if "SampEn" in base.index and pd.notna(base["SampEn"]) and base["SampEn"] < 0.5:
        flags.append("menor complejidad/irregularidad")
    if has_hvg:
        flags.append("topología HVG disponible para contrastar dinámica temporal y estructura de red")

    if flags:
        lines.append("El patrón conjunto sugiere: " + ", ".join(flags) + ".")
    else:
        lines.append("El patrón debe interpretarse con la clínica, calidad de registro y contexto de medición.")
    lines.append("")
    lines.append("> Informe automático orientativo. No sustituye juicio clínico ni diagnóstico médico.")
    lines.append("")
    return "\n".join(lines)


def generate_auto_report(record_data, records_results, global_windows, record_windows, active_phases, use_independent, long_df):
    lines = []
    lines.append("# Informe automático VRC / HRV + grafos HVG")
    lines.append("")
    lines.append("Integra parámetros temporales, frecuenciales, no lineales, recurrencia, Poincaré y grafos HVG cuando están disponibles.")
    lines.append("")
    lines.append("## Registros incluidos")
    lines.append("")
    records_order = sorted(list(record_data.keys()), key=lambda r: (extract_datetime_from_name(r), r))
    for rec in records_order:
        data = record_data[rec]
        dt = extract_datetime_from_name(rec)
        dt_txt = "" if dt is pd.Timestamp.max else f" · fecha detectada: {dt}"
        lines.append(f"- **{rec}** · duración: {data['duration']/60:.2f} min{dt_txt} · archivo: `{data.get('filename','')}`")
    lines.append("")

    for rec in records_order:
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        lines.append(_single_record_report(rec, records_results.get(rec, pd.DataFrame()), windows, record_data[rec].get('rr')))

    if long_df is not None and not long_df.empty and len(records_order) >= 2:
        lines.append("## Comparación cronológica entre todos los registros")
        lines.append("")
        lines.append("Los registros se ordenan de más antiguo a más reciente según la fecha detectada en el nombre del archivo.")
        comp_metrics = ["SDNN","RMSSD","SD1","SD2","VLF","LF","HF","TOTAL","DFA_alpha1","DFA_alpha2","ApEn","SampEn","REC","DET","ShanEn",
                        "HVG_edges","HVG_hubs_p90","HVG_clustering","HVG_lambda"]
        comp_metrics = [m for m in comp_metrics if m in long_df.columns]
        phases = [p for p in PHASES if p in long_df["Fase"].unique()]

        for ph in phases:
            dph = long_df[long_df["Fase"] == ph].set_index("Registro")
            present_records = [r for r in records_order if r in dph.index]
            if len(present_records) < 2:
                continue

            lines.append(f"### Fase {ph}")
            lines.append("")
            header = "| Parámetro | " + " | ".join(present_records) + " | Cambio primero→último |"
            lines.append(header)
            lines.append("|---|" + "|".join(["---:"] * len(present_records)) + "|---:|")

            for m in comp_metrics:
                vals = []
                for r in present_records:
                    vals.append(_fmt_num(dph.loc[r, m]) if m in dph.columns else "")
                first_val = dph.loc[present_records[0], m] if m in dph.columns else np.nan
                last_val = dph.loc[present_records[-1], m] if m in dph.columns else np.nan
                lines.append("| " + m + " | " + " | ".join(vals) + " | " + _arrow_change(first_val, last_val) + " |")
            lines.append("")

            lines.append("#### Cambios consecutivos")
            lines.append("")
            lines.append("| Parámetro | " + " | ".join([f"{present_records[i]}→{present_records[i+1]}" for i in range(len(present_records)-1)]) + " |")
            lines.append("|---|" + "|".join(["---:"] * (len(present_records)-1)) + "|")
            for m in comp_metrics:
                changes = []
                for i in range(len(present_records)-1):
                    a = dph.loc[present_records[i], m] if m in dph.columns else np.nan
                    b = dph.loc[present_records[i+1], m] if m in dph.columns else np.nan
                    changes.append(_arrow_change(a, b))
                lines.append("| " + m + " | " + " | ".join(changes) + " |")
            lines.append("")

        lines.append("### Lectura integrada de evolución")
        lines.append("")
        lines.append(
            "Una reducción cronológica conjunta de SDNN, RMSSD, SD1/SD2 y potencia total junto con menor número de aristas, hubs o clustering HVG sugiere pérdida de riqueza dinámica y simplificación topológica. "
            "Un aumento de entropía, potencia y conectividad HVG sugiere mayor flexibilidad autonómica. La interpretación debe contrastarse con clínica, medicación, calidad de señal, hora del día y condiciones del registro."
        )
    lines.append("## Índices fisiológicos multivariados v14.1")
    lines.append("")
    idx_report = build_physiological_indices(long_df)
    if idx_report.empty:
        lines.append("No hay datos suficientes para calcular los índices.")
    else:
        lines.append("Los índices 0-100 sintetizan métricas convergentes. No son diagnósticos ni probabilidades clínicas entrenadas.")
        lines.append("")
        cols_idx = ["Registro","Fase","IDX_Vagal","IDX_Amplitud","IDX_Complejidad","IDX_Rigidez","IDX_Adaptabilidad","IDX_Regulacion_Lenta","Perfil_autonomico"]
        lines.append("| " + " | ".join(cols_idx) + " |")
        lines.append("|" + "|".join(["---"]*len(cols_idx)) + "|")
        for _, rr_idx in idx_report.iterrows():
            vals=[]
            for c in cols_idx:
                v=rr_idx.get(c, "")
                vals.append(f"{v:.1f}" if isinstance(v,(int,float,np.integer,np.floating)) and np.isfinite(v) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
        lines.append("**Lectura:** vagalidad, amplitud, complejidad y adaptabilidad altas suelen reflejar mayor reserva; rigidez alta debe interpretarse junto a entropía, recurrencia y calidad de señal; regulación lenta alta indica predominio de mecanismos lentos, no necesariamente patología.")
        lines.append("")

    return "\n".join(lines)

def markdown_to_simple_html(md_text):
    escaped = md_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    out = ["<html><head><meta charset='utf-8'><title>Informe HRV</title></head><body>"]
    for line in escaped.splitlines():
        if line.startswith("# "):
            out.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            out.append(f"<p>• {line[2:]}</p>")
        elif line.startswith("|"):
            out.append(f"<pre>{line}</pre>")
        elif line.startswith("&gt;"):
            out.append(f"<blockquote>{line[4:]}</blockquote>")
        elif line.strip() == "":
            out.append("<br>")
        else:
            out.append(f"<p>{line}</p>")
    out.append("</body></html>")
    return "\n".join(out)




def poincare_all_phases_panel_figure(record_data, global_windows, record_windows, record_name, use_independent):
    """
    Un archivo / registro con varias fases:
    muestra Poincaré de TODAS las fases válidas en paneles.
    """
    windows = get_record_windows(global_windows, record_windows, record_name, use_independent)
    valid_phases = [ph for ph in PHASES if windows.get(ph) is not None]

    if not valid_phases:
        fig = go.Figure()
        fig.update_layout(title="No hay fases definidas para este registro")
        return fig

    cols = 2
    rows = int(np.ceil(len(valid_phases) / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=valid_phases,
        horizontal_spacing=0.08,
        vertical_spacing=_safe_vertical_spacing(rows, 0.14),
    )

    cache = {}
    global_min, global_max = np.inf, -np.inf
    rr = record_data[record_name]["rr"]

    for ph in valid_phases:
        w = windows.get(ph)
        seg = cut_segment(rr, w[0], w[1]) if w is not None else np.array([])
        if len(seg) < 3:
            cache[ph] = None
            continue

        rr_ms = seg * 1000
        x, y = rr_ms[:-1], rr_ms[1:]
        diff = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan
        cache[ph] = (x, y, sd1, sd2, len(seg))

        global_min = min(global_min, np.nanmin(x), np.nanmin(y))
        global_max = max(global_max, np.nanmax(x), np.nanmax(y))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        fig = go.Figure()
        fig.update_layout(title="No hay suficientes RRi para Poincaré por fases")
        return fig

    pad = max(20, 0.05 * (global_max - global_min))
    global_min -= pad
    global_max += pad

    for idx, ph in enumerate(valid_phases):
        r = idx // cols + 1
        c = idx % cols + 1
        item = cache.get(ph)

        if item is None:
            fig.add_annotation(text="Sin datos suficientes", x=0.5, y=0.5, xref=f"x{idx+1 if idx>0 else ''} domain",
                               yref=f"y{idx+1 if idx>0 else ''} domain", showarrow=False)
            continue

        x, y, sd1, sd2, nseg = item
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(size=5, opacity=0.62),
                showlegend=False,
                hovertemplate="RR(n): %{x:.1f} ms<br>RR(n+1): %{y:.1f} ms<extra></extra>",
            ),
            row=r, col=c
        )
        fig.add_trace(
            go.Scatter(x=[global_min, global_max], y=[global_min, global_max],
                       mode="lines", line=dict(width=1, dash="dash"),
                       showlegend=False, hoverinfo="skip"),
            row=r, col=c
        )
        fig.add_annotation(
            text=f"N={nseg}<br>SD1={sd1:.1f} ms<br>SD2={sd2:.1f} ms",
            x=0.03, y=0.97,
            xref=f"x{idx+1 if idx>0 else ''} domain",
            yref=f"y{idx+1 if idx>0 else ''} domain",
            showarrow=False, align="left",
            bgcolor="rgba(0,0,0,0.25)",
            bordercolor="rgba(255,255,255,0.25)",
        )
        fig.update_xaxes(range=[global_min, global_max], title_text="RR(n) ms", row=r, col=c)
        fig.update_yaxes(range=[global_min, global_max], title_text="RR(n+1) ms", row=r, col=c,
                         scaleanchor=f"x{idx+1 if idx>0 else ''}", scaleratio=1)

    fig.update_layout(
        height=max(650, rows * 470),
        title=f"Poincaré por fases · {record_name}",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def hvg_all_phases_panel_figure(record_data, global_windows, record_windows, record_name, use_independent, max_nodes=120):
    """
    Un archivo / registro con varias fases:
    muestra HVG de TODAS las fases válidas en paneles.
    """
    if nx is None:
        fig = go.Figure()
        fig.update_layout(title="NetworkX no disponible")
        return fig

    windows = get_record_windows(global_windows, record_windows, record_name, use_independent)
    valid_phases = [ph for ph in PHASES if windows.get(ph) is not None]

    if not valid_phases:
        fig = go.Figure()
        fig.update_layout(title="No hay fases definidas para este registro")
        return fig

    cols = 2
    rows = int(np.ceil(len(valid_phases) / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=valid_phases,
        horizontal_spacing=0.04,
        vertical_spacing=_safe_vertical_spacing(rows, 0.12),
    )

    rr = record_data[record_name]["rr"]

    for idx, ph in enumerate(valid_phases):
        r = idx // cols + 1
        c = idx % cols + 1
        w = windows.get(ph)
        seg = cut_segment(rr, w[0], w[1]) if w is not None else np.array([])

        if len(seg) < 20:
            fig.add_annotation(text="Sin datos suficientes", x=0.5, y=0.5, xref=f"x{idx+1 if idx>0 else ''} domain",
                               yref=f"y{idx+1 if idx>0 else ''} domain", showarrow=False)
            continue

        G = hvg_graph(seg, max_nodes=max_nodes)
        if G is None or G.number_of_nodes() == 0:
            continue

        pos = nx.spring_layout(G, seed=42, k=0.20, iterations=60)
        edge_x, edge_y = [], []
        for a, b in G.edges():
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]

        deg = dict(G.degree())
        node_x = [pos[nn][0] for nn in G.nodes()]
        node_y = [pos[nn][1] for nn in G.nodes()]
        node_size = [5 + deg[nn] * 2.0 for nn in G.nodes()]
        node_text = [f"{ph}<br>n={nn}<br>grado={deg[nn]}" for nn in G.nodes()]

        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(width=0.45), hoverinfo="skip",
                                 showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                                 marker=dict(size=node_size, opacity=0.82),
                                 text=node_text, hoverinfo="text",
                                 showlegend=False), row=r, col=c)
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    fig.update_layout(
        height=max(650, rows * 440),
        title=f"HVG por fases · {record_name}",
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def hvg_metrics_all_phases_figure(metrics_df):
    """
    Secuencia de métricas HVG por fases para un solo registro.
    """
    if metrics_df is None or metrics_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No hay métricas HVG")
        return fig

    hvg_cols = ["HVG_edges", "HVG_degree_mean", "HVG_degree_max", "HVG_hubs_p90",
                "HVG_clustering", "HVG_lambda", "HVG_path_length", "HVG_diameter"]
    hvg_cols = [c for c in hvg_cols if c in metrics_df.columns]
    phases = [p for p in PHASES if p in metrics_df.index]

    fig = go.Figure()
    for col in hvg_cols:
        y = [metrics_df.loc[ph, col] if ph in metrics_df.index else np.nan for ph in phases]
        fig.add_trace(go.Scatter(x=phases, y=y, mode="lines+markers", name=col, line=dict(width=3)))

    fig.update_layout(
        height=560,
        title="Secuencia de métricas HVG por fases",
        xaxis_title="Fase",
        yaxis_title="Valor",
        hovermode="x unified",
    )
    return fig




def _smooth_line_xy(y_values, smooth_points=220):
    """
    Suavizado visual seguro para líneas de tendencia.

    - 1 punto: punto único.
    - 2 puntos: línea recta inevitable.
    - 3 puntos: curva cuadrática suavizada.
    - >=4 puntos: CubicSpline natural.
    """
    y = np.asarray(y_values, dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)

    n_valid = int(np.sum(mask))
    if n_valid == 0:
        return [], []
    if n_valid == 1:
        return x[mask], y[mask]

    xs = np.linspace(x[mask].min(), x[mask].max(), smooth_points)

    try:
        if n_valid >= 4:
            cs = CubicSpline(x[mask], y[mask], bc_type="natural")
            ys = cs(xs)
        elif n_valid == 3:
            # Con tres fases/registros se puede generar una curva suave cuadrática.
            coef = np.polyfit(x[mask], y[mask], deg=2)
            ys = np.polyval(coef, xs)
        else:
            # Con dos puntos no existe suavizado real sin inventar información.
            ys = np.interp(xs, x[mask], y[mask])
    except Exception:
        ys = np.interp(xs, x[mask], y[mask])

    return xs, ys


def _add_bars_and_smooth_lines(fig, metrics_df, row, col, metrics, title, yaxis_title="Valor", secondary_y_metric=None, secondary_y_title=None):
    """
    Añade columnas verticales + líneas suavizadas superpuestas en un panel.
    """
    phases = [p for p in PHASES if p in metrics_df.index]
    if not phases:
        return

    x_base = np.arange(len(phases), dtype=float)
    present = [m for m in metrics if m in metrics_df.columns]
    n = max(1, len(present))
    bar_width = min(0.72 / n, 0.18)

    for i, m in enumerate(present):
        y = [metrics_df.loc[ph, m] if ph in metrics_df.index else np.nan for ph in phases]
        offset = (i - (n - 1) / 2) * bar_width
        x_bar = x_base + offset
        use_secondary = (secondary_y_metric is not None and m == secondary_y_metric)

        fig.add_trace(
            go.Bar(
                x=x_bar,
                y=y,
                width=bar_width,
                name=m,
                opacity=0.72,
                hovertemplate=f"{m}<br>Fase: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
                customdata=phases,
                showlegend=True,
            ),
            row=row,
            col=col,
            secondary_y=use_secondary,
        )

        xs, ys = _smooth_line_xy(y)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{m} tendencia",
                line=dict(width=3),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
            secondary_y=use_secondary,
        )

        fig.add_trace(
            go.Scatter(
                x=x_base,
                y=y,
                mode="markers",
                name=f"{m} puntos",
                marker=dict(size=6),
                hovertemplate=f"{m}<br>Fase: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
                customdata=phases,
                showlegend=False,
            ),
            row=row,
            col=col,
            secondary_y=use_secondary,
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=phases,
        title_text="Fase",
        row=row,
        col=col,
    )
    fig.update_yaxes(title_text=yaxis_title, row=row, col=col, secondary_y=False)
    if secondary_y_metric is not None and secondary_y_title:
        fig.update_yaxes(title_text=secondary_y_title, row=row, col=col, secondary_y=True)


def hrv_phase_summary_figure(metrics_df):
    """
    Figura tipo referencia:
    6 paneles con columnas verticales por fase y líneas de tendencia suavizadas.
    """
    if metrics_df is None or metrics_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No hay datos HRV para graficar")
        return fig

    specs = [
        [{"secondary_y": True}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}],
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=specs,
        subplot_titles=[
            "1) RMSSD, SDNN, pNN50",
            "2) VLF, LF, HF, TOTAL",
            "3) SD1, SD2",
            "4) DFA α1, α2, ApEn, SampEn",
            "5) Recurrence Plot",
            "6) Multiscale Entropy (MSE 1-20)",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.11,
    )

    _add_bars_and_smooth_lines(
        fig, metrics_df, 1, 1,
        ["RMSSD", "SDNN", "pNN50"],
        "1) RMSSD, SDNN, pNN50",
        yaxis_title="ms",
        secondary_y_metric="pNN50",
        secondary_y_title="pNN50 (%)",
    )

    _add_bars_and_smooth_lines(
        fig, metrics_df, 1, 2,
        ["VLF", "LF", "HF", "TOTAL"],
        "2) VLF, LF, HF, TOTAL",
        yaxis_title="ms²",
    )

    _add_bars_and_smooth_lines(
        fig, metrics_df, 2, 1,
        ["SD1", "SD2"],
        "3) SD1, SD2",
        yaxis_title="ms",
    )

    complexity_vars = [m for m in ["DFA_alpha1", "DFA_alpha2", "D2", "ApEn", "SampEn"] if m in metrics_df.columns]
    _add_bars_and_smooth_lines(
        fig, metrics_df, 2, 2,
        complexity_vars,
        "4) DFA α1, α2, D2, ApEn, SampEn",
        yaxis_title="Valor",
    )

    recurrence_vars = [m for m in ["Lmean", "Lmax", "REC", "DET", "ShanEn"] if m in metrics_df.columns]
    _add_bars_and_smooth_lines(
        fig, metrics_df, 3, 1,
        recurrence_vars,
        "5) Recurrence Plot",
        yaxis_title="Valor",
    )

    mse_vars = [f"MSE{i}" for i in range(1, 21) if f"MSE{i}" in metrics_df.columns]
    _add_bars_and_smooth_lines(
        fig, metrics_df, 3, 2,
        mse_vars,
        "6) Multiscale Entropy (MSE 1-20)",
        yaxis_title="Valor",
    )

    fig.update_layout(
        height=1350,
        title="Resumen HRV por fases: columnas verticales + líneas suavizadas",
        barmode="group",
        hovermode="closest",
        legend_title_text="Parámetro",
        bargap=0.22,
        bargroupgap=0.02,
        margin=dict(l=60, r=40, t=100, b=70),
    )

    return fig


def hrv_phase_summary_record_panels(record_data, records_results):
    """
    Si hay varios registros: un panel visual por registro usando la misma lógica.
    Devuelve dict nombre_figura -> figura.
    """
    figs = {}
    for rec, df in records_results.items():
        if df is not None and not df.empty:
            figs[rec] = hrv_phase_summary_figure(df)
    return figs


# ============================================================
# v15.2.13 · CACHÉ DE CÁLCULO POR REGISTRO/VENTANA
# ============================================================
@st.cache_data(show_spinner=False, max_entries=256)
def calculate_record_cached_v15213(rr, windows, active_phases, min_rr, include_rqa, include_hvg, mse_zero_policy, theiler_window, radius_mode):
    # Copias defensivas para impedir que el resultado cacheado sea modificado por la interfaz.
    return calculate_record(
        np.asarray(rr, dtype=float), copy.deepcopy(windows), list(active_phases), int(min_rr),
        bool(include_rqa), include_hvg=bool(include_hvg), mse_zero_policy=str(mse_zero_policy),
        theiler_window=int(theiler_window), radius_mode=str(radius_mode)
    )

# ============================================================
# APP
# ============================================================

st.title("PhysioSentinel AI: Predictive Physiological Intelligence Platform")
st.caption("v15.5.5 · Persistent Cloud Architecture II · Control Autonómico longitudinal progresivo V/S/L/C/R · Decoupled Calibrated Probability Layer · Nivel 2 XGBoost")
st.caption("Longitudinal Scale: diseñado para históricos de 100–200+ registros por paciente; los cálculos conservan el histórico completo y las visualizaciones pesadas se paginan localmente.")
st.caption("Modo ligero: navegación optimizada, paneles paginados, caché de cálculos y gráficas RRi aligeradas.")

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Sube uno o varios CSV/TXT con RRi",
        type=["csv", "txt"],
        accept_multiple_files=True,
        help="Los nuevos RRi se guardan automáticamente en la base interna v15.2."
    )

    st.markdown("### Rendimiento")
    light_mode_v15213 = st.checkbox(
        "Modo Ligero",
        value=True,
        help=("Reduce los puntos dibujados, pagina los paneles independientes y reutiliza "
              "los resultados ya calculados. Los RRi completos siguen utilizándose en los cálculos."),
    )
    max_plot_points_v15213 = st.select_slider(
        "Puntos máximos por curva",
        options=[500, 750, 1000, 1500, 2500, 5000],
        value=1000 if light_mode_v15213 else 5000,
        disabled=not light_mode_v15213,
        help="Solo afecta a la visualización, no a las métricas.",
    )

    # v15.3.8 · separación estricta entre la sesión actual y el historial.
    # Un cambio real en los archivos subidos devuelve automáticamente el foco a la sesión,
    # pero después el usuario puede abrir explícitamente el historial sin que se mezcle con ellos.
    _upload_signature_v1538 = tuple(sorted((getattr(f, "name", ""), int(getattr(f, "size", 0) or 0)) for f in (uploaded_files or [])))
    _prev_upload_signature_v1538 = st.session_state.get("upload_signature_v1538")
    if _upload_signature_v1538 and _upload_signature_v1538 != _prev_upload_signature_v1538:
        st.session_state["history_source_v1538"] = "Sesión actual"
        # v15.3.15: una carga nueva abre SIEMPRE un espacio de trabajo de segmentación limpio y con un editor nuevo.
        # No se heredan ventanas ni selecciones persistentes hasta que el usuario lo solicite.
        st.session_state["session_seg_mode_v15314"] = "Nueva segmentación (editable)"
        # v15.3.15: cada carga real crea una nueva generación de editores.
        # Esto impide que un custom component reutilice el último valor devuelto por
        # una ejecución anterior del mismo record_key.
        st.session_state["seg_editor_epoch_v15315"] = int(st.session_state.get("seg_editor_epoch_v15315", 0)) + 1
        for _k in (
            "history_patient_v1537", "history_record_v1537", "history_compare_records_v1537",
            "history_compare_patients_v15310", "global_windows_v50", "record_windows_v50",
            "restored_records_v152", "pending_selections_v1522", "pending_selection_v50",
            "active_phases_v50", "segmentation_signatures_v15213"
        ):
            st.session_state.pop(_k, None)
    st.session_state["upload_signature_v1538"] = _upload_signature_v1538
    st.session_state.setdefault("seg_editor_epoch_v15315", 1)

    _saved_catalog_v152 = list_saved_raw_records()
    try:
        ensure_patient_bundles_from_existing_history()
    except Exception:
        pass
    _patient_bundles_v1539 = list_patient_analysis_bundles()
    with st.expander("Historial longitudinal v15.5.5 · Persistent Cloud Architecture II", expanded=True):
        st.caption("Arquitectura V2: una fila actual por objeto + backups deduplicados SHA-256, máximo 5 por objeto; SQLite automático como máximo cada 6 h.")
        _cloud_status_1540 = _cloud_sync_status_v1540()
        if _cloud_status_1540['configured']:
            if _cloud_status_1540['connected']:
                st.success("☁️ Base persistente externa conectada · PostgreSQL")
                if _cloud_status_1540.get('last_sync'):
                    try:
                        _cloud_ts = pd.to_datetime(_cloud_status_1540['last_sync'], utc=True)
                        _cloud_dt = _cloud_ts.tz_convert(ZoneInfo('Europe/Madrid')).strftime('%d/%m/%Y %H:%M:%S')
                        _cloud_dt += ' · hora local Europe/Madrid'
                    except Exception:
                        _cloud_dt = str(_cloud_status_1540['last_sync'])
                    st.caption(f"Última escritura real en nube: {_cloud_dt} · {_cloud_status_1540.get('last_reason') or '—'}")
                if _cloud_status_1540.get('dirty'):
                    st.caption("🟡 Cambios locales pendientes: se consolidarán en el próximo guardado persistente o sincronización manual.")
                else:
                    st.caption("⚡ Lazy Sync activo: cambiar pestañas, selectores o gráficos no consulta PostgreSQL.")
            else:
                st.warning("☁️ PostgreSQL configurado, pero no está conectado. La sesión continúa con caché local.")
            _cs1, _cs2 = st.columns(2)
            if _cs1.button("Sincronizar histórico con nube", key="cloud_sync_now_v1540"):
                if _cloud_checkpoint_v1540('manual_sync', force=True):
                    st.success("Histórico sincronizado en PostgreSQL V2. Los backups se deduplican y rotan automáticamente (máx. 5; SQLite automático como máximo cada 6 h).")
                    st.rerun()
                else:
                    st.error("No se pudo sincronizar. Revisa la conexión persistente.")
            if _cs2.button("Recuperar última copia de nube", key="cloud_restore_now_v1540"):
                _remote = _cloud_get_object_v1540(CLOUD_STORE_KEY_V1540)
                if _remote and _remote.get('payload', b'').startswith(b'SQLite format 3\x00'):
                    _tmp = LONGITUDINAL_DB_PATH.with_suffix('.manual-cloud.tmp')
                    _tmp.write_bytes(_remote['payload'])
                    with sqlite3.connect(_tmp) as _chk:
                        _ok = _chk.execute('PRAGMA quick_check').fetchone()
                    if _ok and str(_ok[0]).lower() == 'ok':
                        _tmp.replace(LONGITUDINAL_DB_PATH)
                        _cloud_state_v1540()['base_sha'] = _remote['sha256']
                        st.success("Última copia persistente recuperada.")
                        st.rerun()
                    else:
                        st.error("La copia remota no superó la comprobación de integridad.")
                else:
                    st.warning("Todavía no existe una copia persistente recuperable.")
        else:
            st.info("☁️ Persistencia externa preparada pero aún no configurada. Añade `PHYSIOSENTINEL_DATABASE_URL` en Streamlit Secrets. Hasta entonces la app usa SQLite local como caché.")
        st.caption("Persistent Cloud Architecture II: PostgreSQL conserva una única imagen actual por objeto. Las escrituras automáticas del SQLite se consolidan como máximo cada 5 min; los backups se deduplican por SHA-256, se limitan a 5 por objeto y el SQLite completo se versiona automáticamente como máximo cada 6 h. La navegación sigue en SQLite/RAM local. El botón Sincronizar fuerza escritura inmediata.")
        _restore_db_file_v15326 = st.file_uploader("Restaurar copia SQLite longitudinal", type=["sqlite3","sqlite","db"], key="restore_longitudinal_db_v15326")
        if _restore_db_file_v15326 is not None and st.button("Restaurar base ahora", key="restore_longitudinal_db_btn_v15326"):
            _ok_restore, _msg_restore = _restore_longitudinal_db_v15326(_restore_db_file_v15326)
            if _ok_restore:
                st.success(_msg_restore)
                st.rerun()
            else:
                st.error(_msg_restore)
        _history_source_v1538 = st.radio(
            "Fuente activa",
            ["Sesión actual", "Histórico guardado"],
            horizontal=False,
            key="history_source_v1538",
            help=("Sesión actual utiliza exclusivamente los archivos cargados ahora. Histórico guardado "
                  "se activa solo cuando lo eliges y abre los snapshots persistentes sin mezclarlos con la sesión."),
        )
        _history_mode_v1537 = "Abrir histórico completo del paciente"
        if _history_source_v1538 == "Histórico guardado":
            _history_mode_v1537 = st.radio(
                "Modo del historial",
                ["Abrir histórico completo del paciente", "Comparar pacientes históricos"],
                horizontal=False,
                key="history_mode_v1538",
                help=("Abrir un análisis guardado carga el histórico longitudinal completo del paciente. "
                      "Comparar varios históricos permite elegir pacientes por nombre y carga automáticamente "
                      "todos los registros conservados de cada uno."),
            )
        elif uploaded_files:
            st.success(f"Sesión actual activa: {len(uploaded_files)} archivo(s) cargado(s). El historial no interviene.")
        else:
            st.caption("Sesión actual activa. Sube archivos o cambia explícitamente a Histórico guardado.")

        # v15.4.0 · histórico incremental REAL: permite subir solo la nueva visita y
        # anexarla explícitamente a un paciente ya existente.
        _incremental_target_patient_v15325 = None
        if not _patient_bundles_v1539.empty:
            st.markdown("#### Añadir nuevos registros a un histórico")
            _inc_rows_v15325 = _patient_bundles_v1539.copy()
            _inc_rows_v15325 = _inc_rows_v15325.sort_values(["display_name", "saved_at"]).drop_duplicates("patient_id", keep="last")
            _inc_labels_v15325 = {
                str(r.patient_id): f"{r.display_name} · {int(r.record_count or 0)} registro(s)"
                for r in _inc_rows_v15325.itertuples()
            }
            _inc_ids_v15325 = list(_inc_labels_v15325.keys())
            _incremental_enabled_v15325 = st.checkbox(
                "Añadir esta sesión a un paciente histórico existente",
                value=False,
                key="incremental_enabled_v15325",
                help=("Actívalo cuando hayas subido únicamente la visita nueva. Al guardar, los registros "
                      "de esta sesión se añadirán al histórico elegido sin recalcular ni volver a cargar los anteriores."),
            )
            _inc_patient_id_v15325 = st.selectbox(
                "Paciente histórico destino",
                _inc_ids_v15325,
                format_func=lambda x: _inc_labels_v15325.get(str(x), str(x)),
                key="incremental_patient_v15325",
                disabled=not _incremental_enabled_v15325,
            ) if _inc_ids_v15325 else None
            if _incremental_enabled_v15325 and _inc_patient_id_v15325:
                _incremental_target_patient_v15325 = infer_patient_id(_inc_patient_id_v15325)
                st.session_state["incremental_target_patient_v15325"] = _incremental_target_patient_v15325
                _old_count = int(_inc_rows_v15325.loc[_inc_rows_v15325["patient_id"].astype(str)==str(_inc_patient_id_v15325), "record_count"].max() or 0)
                if uploaded_files:
                    st.success(
                        f"Modo incremental activo: {len(uploaded_files)} archivo(s) nuevo(s) se añadirán a "
                        f"{_inc_labels_v15325.get(str(_inc_patient_id_v15325), _inc_patient_id_v15325)}. "
                        f"Los {_old_count} registro(s) previos permanecen intactos."
                    )
                else:
                    st.info("Sube arriba únicamente el/los registro(s) nuevos del paciente; no necesitas cargar los anteriores.")
            else:
                st.session_state.pop("incremental_target_patient_v15325", None)
            st.caption(
                "El histórico incremental conserva los snapshots anteriores y calcula/guarda únicamente los registros nuevos de esta sesión."
            )

        # v15.4.4 · gestor seguro de registros históricos: mover entre pacientes,
        # papelera recuperable y detección de posibles asignaciones erróneas.
        if not _saved_catalog_v152.empty:
            with st.expander("Gestionar registros del histórico", expanded=False):
                st.caption(
                    "Corrige asignaciones sin perder cálculos. Mover conserva RRi, segmentaciones, snapshots, "
                    "Poincaré, HVG, atractores, índices y resultados asociados. Eliminar envía primero a una papelera recuperable."
                )
                _mgr_catalog = _saved_catalog_v152.copy()
                _mgr_catalog['patient_norm'] = _mgr_catalog['patient_id'].fillna('').astype(str).map(infer_patient_id)
                _mgr_patients = sorted([p for p in _mgr_catalog['patient_norm'].dropna().astype(str).unique().tolist() if p])
                _mgr_patient = st.selectbox(
                    "Paciente a gestionar",
                    _mgr_patients,
                    format_func=lambda x: str(x).replace('_',' ').title(),
                    key='history_manager_patient_v1544',
                ) if _mgr_patients else None
                if _mgr_patient:
                    _mgr_rows = _mgr_catalog[_mgr_catalog['patient_norm']==_mgr_patient].copy()
                    _mgr_rows = _mgr_rows.sort_values(['record_datetime','saved_at'], na_position='last')
                    _mgr_labels = {}
                    _mismatch = []
                    for _r in _mgr_rows.itertuples():
                        _inferred = infer_patient_id(str(_r.record_name))
                        _warn = (_inferred != _mgr_patient)
                        _label = f"{'⚠️ ' if _warn else ''}{_r.record_name}"
                        if getattr(_r, 'record_datetime', None):
                            _label += f" · {_r.record_datetime}"
                        _label += f" · {int(getattr(_r,'n_rri',0) or 0)} RRi"
                        _mgr_labels[str(_r.record_key)] = _label
                        if _warn:
                            _mismatch.append(str(_r.record_key))
                    if _mismatch:
                        st.warning(
                            f"Se detectan {len(_mismatch)} registro(s) cuyo nombre parece corresponder a otro paciente. "
                            "Revísalos antes de mover o eliminar."
                        )
                    _mgr_selected = st.multiselect(
                        "Registros a gestionar",
                        list(_mgr_labels.keys()),
                        format_func=lambda x: _mgr_labels.get(str(x), str(x)),
                        key='history_manager_records_v1544',
                    )
                    _mgr_c1, _mgr_c2 = st.columns(2)
                    with _mgr_c1:
                        _target_candidates = [p for p in _mgr_patients if p != _mgr_patient]
                        _mgr_target = st.selectbox(
                            "Mover a otro paciente",
                            ['— Selecciona destino —'] + _target_candidates,
                            format_func=lambda x: x if x.startswith('—') else str(x).replace('_',' ').title(),
                            key='history_manager_target_v1544',
                        )
                        if st.button("Mover registros seleccionados", key='history_manager_move_v1544',
                                     disabled=(not _mgr_selected or _mgr_target.startswith('—')),
                                     use_container_width=True):
                            _n_moved, _sources = move_saved_records_to_patient(_mgr_selected, _mgr_target, sync_cloud=True)
                            st.success(f"{_n_moved} registro(s) movido(s) a {str(_mgr_target).replace('_',' ').title()} sin perder resultados.")
                            st.session_state.pop('history_manager_records_v1544', None)
                            st.rerun()
                    with _mgr_c2:
                        _confirm_delete = st.checkbox(
                            f"Confirmo enviar {len(_mgr_selected)} registro(s) a la papelera",
                            key='history_manager_confirm_trash_v1544',
                            disabled=not _mgr_selected,
                        )
                        if st.button("Enviar a papelera", key='history_manager_trash_v1544',
                                     disabled=(not _mgr_selected or not _confirm_delete),
                                     use_container_width=True):
                            _n_trash, _sources = trash_saved_records(_mgr_selected, sync_cloud=True)
                            st.success(f"{_n_trash} registro(s) enviados a la papelera recuperable.")
                            st.session_state.pop('history_manager_records_v1544', None)
                            st.rerun()

                _trash_rows = list_trashed_raw_records()
                if _trash_rows is not None and not _trash_rows.empty:
                    st.divider()
                    st.markdown("##### Papelera recuperable")
                    _trash_labels = {}
                    for _r in _trash_rows.itertuples():
                        _orig = str(_r.patient_id).split(':',1)[1] if ':' in str(_r.patient_id) else str(_r.patient_id)
                        _trash_labels[str(_r.record_key)] = (
                            f"{_r.record_name} · origen: {_orig.replace('_',' ').title()} · {int(getattr(_r,'n_rri',0) or 0)} RRi"
                        )
                    _restore_sel = st.multiselect(
                        "Registros en papelera",
                        list(_trash_labels.keys()),
                        format_func=lambda x: _trash_labels.get(str(x), str(x)),
                        key='history_manager_restore_sel_v1544',
                    )
                    if st.button("Restaurar seleccionados", key='history_manager_restore_v1544',
                                 disabled=not _restore_sel, use_container_width=True):
                        _n_restored, _targets = restore_trashed_records(_restore_sel, sync_cloud=True)
                        st.success(f"{_n_restored} registro(s) restaurado(s) a su histórico de origen.")
                        st.session_state.pop('history_manager_restore_sel_v1544', None)
                        st.rerun()

        # v15.4.0 · al añadir una visita, el histórico previo del paciente se incorpora
        # al contexto de trabajo SIN obligar a volver a subir sus archivos. Los registros
        # previos se leen desde SQLite/snapshot y el/los registros subidos permanecen
        # como la única parte nueva editable/calculable de la visita.
        _incremental_history_keys_v15327 = []
        if (_history_source_v1538 == "Sesión actual" and
                st.session_state.get("incremental_target_patient_v15325")):
            try:
                _inc_target_v15327 = st.session_state.get("incremental_target_patient_v15325")
                _inc_existing_rows_v15327 = _canonical_patient_rows(_saved_catalog_v152, _inc_target_v15327)
                if not _inc_existing_rows_v15327.empty:
                    _incremental_history_keys_v15327 = list(dict.fromkeys(
                        _inc_existing_rows_v15327["record_key"].astype(str).tolist()
                    ))
                    st.info(
                        f"Contexto longitudinal activo: {len(_incremental_history_keys_v15327)} registro(s) previos "
                        "se cargarán automáticamente desde el histórico y se mostrarán junto a la nueva visita. "
                        "No necesitas volver a subirlos."
                    )
            except Exception as _inc_ctx_err_v15327:
                _incremental_history_keys_v15327 = []
                st.warning(f"No se pudo preparar el contexto histórico incremental: {_inc_ctx_err_v15327}")

        # v15.3.14 · las segmentaciones de la sesión actual son un espacio de trabajo separado.
        # Por defecto no se restauran ni sobrescriben las ventanas históricas de esos mismos archivos.
        _session_seg_mode_v15314 = "Nueva segmentación (editable)"
        if _history_source_v1538 == "Sesión actual" and uploaded_files:
            _session_seg_mode_v15314 = st.radio(
                "Segmentación de la sesión",
                ["Nueva segmentación (editable)", "Recuperar segmentación anterior"],
                key="session_seg_mode_v15314",
                help=("Nueva segmentación empieza con ventanas vacías y no modifica el histórico hasta Guardar análisis. "
                      "Recuperar segmentación anterior copia las ventanas guardadas como punto de partida, pero las ediciones "
                      "siguen siendo temporales hasta Guardar análisis."),
            )
            _prev_seg_mode_v15314 = st.session_state.get("session_seg_mode_applied_v15314")
            if _prev_seg_mode_v15314 != _session_seg_mode_v15314:
                for _k in ("global_windows_v50", "record_windows_v50", "restored_records_v152",
                           "pending_selections_v1522", "pending_selection_v50", "active_phases_v50",
                           "segmentation_signatures_v15213"):
                    st.session_state.pop(_k, None)
                st.session_state["session_seg_mode_applied_v15314"] = _session_seg_mode_v15314
                st.session_state["seg_editor_epoch_v15315"] = int(st.session_state.get("seg_editor_epoch_v15315", 0)) + 1
                st.rerun()
        _catalog_options_v152 = _saved_catalog_v152['record_key'].tolist() if not _saved_catalog_v152.empty else []
        _catalog_labels_v152 = {
            str(r.record_key): f"{r.record_name} · {int(r.n_rri or 0)} RRi"
            for r in _saved_catalog_v152.itertuples()
        } if not _saved_catalog_v152.empty else {}

        selected_saved_keys_v152 = []
        _selected_snapshot_v1537 = None
        _selected_snapshot_key_v1537 = None
        _selected_snapshot_record_v1537 = None

        if _history_source_v1538 == "Histórico guardado" and not _saved_catalog_v152.empty and _history_mode_v1537 == "Abrir histórico completo del paciente":
            # v15.3.13: el histórico principal es longitudinal por paciente, no un snapshot aislado por archivo.
            _patients_v1537 = sorted({infer_patient_id(x) for x in _saved_catalog_v152['record_name'].astype(str)})
            _patient_v1537 = st.selectbox(
                "Paciente", _patients_v1537,
                format_func=lambda p: str(p).replace('_',' ').title(),
                key="history_patient_v1537"
            ) if _patients_v1537 else None
            if _patient_v1537:
                _patient_rows_v1537 = _canonical_patient_rows(_saved_catalog_v152, _patient_v1537)
                _bundle_rows_v1539 = _patient_bundles_v1539[
                    _patient_bundles_v1539['patient_id'].astype(str) == str(infer_patient_id(_patient_v1537))
                ].copy() if not _patient_bundles_v1539.empty else pd.DataFrame()
                if _bundle_rows_v1539.empty:
                    _bundle_id_v1539, _bundle_keys_v1539 = save_patient_analysis_bundle(
                        _patient_v1537, _patient_rows_v1537['record_key'].astype(str).tolist(),
                        config={"auto_created": True}
                    )
                    _patient_bundles_v1539 = list_patient_analysis_bundles()
                    _bundle_rows_v1539 = _patient_bundles_v1539[_patient_bundles_v1539['bundle_id'].astype(str)==str(_bundle_id_v1539)]
                _bundle_opts_v1539 = _bundle_rows_v1539['bundle_id'].astype(str).tolist()
                _bundle_labels_v1539 = {}
                for _br in _bundle_rows_v1539.itertuples():
                    _bundle_labels_v1539[str(_br.bundle_id)] = f"Histórico completo · {int(_br.record_count or 0)} registros"
                _bundle_id_v1539 = st.selectbox(
                    "Histórico del paciente", _bundle_opts_v1539,
                    format_func=lambda b: _bundle_labels_v1539.get(str(b), str(b)),
                    key="history_bundle_v1539"
                ) if _bundle_opts_v1539 else None
                if _bundle_id_v1539:
                    _bundle_v1539 = load_patient_analysis_bundle(_bundle_id_v1539)
                    selected_saved_keys_v152 = list((_bundle_v1539 or {}).get('record_keys', []))
                    # Mantener todos los RRi del paciente aunque alguno todavía no tenga snapshot analítico.
                    _all_patient_keys_v1539 = _patient_rows_v1537['record_key'].astype(str).tolist() if not _patient_rows_v1537.empty else []
                    selected_saved_keys_v152 = list(dict.fromkeys(selected_saved_keys_v152 + _all_patient_keys_v1539))
                    _snapshots_available_v1539 = {k: load_analysis_snapshot(k) for k in selected_saved_keys_v152}
                    _n_snap_v1539 = sum(v is not None for v in _snapshots_available_v1539.values())
                    st.success(
                        f"Histórico longitudinal: {len(selected_saved_keys_v152)} registros del paciente. "
                        f"{_n_snap_v1539} con resultados congelados; "
                        f"{len(selected_saved_keys_v152)-_n_snap_v1539} pendientes de snapshot."
                    )
                    # Para bloquear controles usamos la configuración del primer snapshot disponible;
                    # cada registro recupera internamente su propio snapshot/configuración.
                    _first_snap_v1539 = next((v for v in _snapshots_available_v1539.values() if v is not None), None)
                    _selected_snapshot_v1537 = _first_snap_v1539
                    _selected_snapshot_key_v1537 = _bundle_id_v1539
                    _selected_snapshot_record_v1537 = f"{str(_patient_v1537).replace('_',' ').title()} · {len(selected_saved_keys_v152)} registros"
        elif _history_source_v1538 == "Histórico guardado" and not _saved_catalog_v152.empty:
            # v15.3.13: la comparación histórica se elige por PACIENTE, no archivo por archivo.
            # Cada paciente expande automáticamente todos sus registros longitudinales conservados.
            _compare_patients_v15310 = sorted({infer_patient_id(x) for x in _saved_catalog_v152['record_name'].astype(str)})
            _selected_compare_patients_v15310 = st.multiselect(
                "Pacientes históricos a comparar",
                _compare_patients_v15310,
                default=[],
                format_func=lambda p: str(p).replace('_', ' ').title(),
                key="history_compare_patients_v15310",
                placeholder="Selecciona uno o varios pacientes",
            )
            selected_saved_keys_v152 = []
            _compare_summary_v15310 = []
            for _pid_v15310 in _selected_compare_patients_v15310:
                _rows_v15310 = _canonical_patient_rows(_saved_catalog_v152, _pid_v15310)
                _keys_v15310 = _rows_v15310['record_key'].astype(str).tolist() if not _rows_v15310.empty else []
                selected_saved_keys_v152.extend(_keys_v15310)
                _compare_summary_v15310.append({
                    'Paciente': str(_pid_v15310).replace('_', ' ').title(),
                    'Registros': len(_keys_v15310),
                })
            selected_saved_keys_v152 = list(dict.fromkeys(selected_saved_keys_v152))
            if _compare_summary_v15310:
                st.dataframe(pd.DataFrame(_compare_summary_v15310), hide_index=True, use_container_width=True)
                st.success(
                    f"Comparación longitudinal: {len(_selected_compare_patients_v15310)} paciente(s), "
                    f"{len(selected_saved_keys_v152)} registro(s) históricos cargados automáticamente."
                )
            else:
                st.caption("Selecciona pacientes. La app cargará automáticamente todos los registros históricos de cada uno.")

        # v15.4.0 · en modo incremental, además de la nueva subida se cargan
        # automáticamente todos los registros previos del paciente destino.
        if (_history_source_v1538 == "Sesión actual" and
                st.session_state.get("incremental_target_patient_v15325") and
                globals().get("_incremental_history_keys_v15327")):
            selected_saved_keys_v152 = list(dict.fromkeys(
                list(selected_saved_keys_v152) + list(_incremental_history_keys_v15327)
            ))
        auto_load_history_v152 = bool(
            (_history_source_v1538 == "Histórico guardado" and selected_saved_keys_v152)
            or (_history_source_v1538 == "Sesión actual" and
                st.session_state.get("incremental_target_patient_v15325") and selected_saved_keys_v152)
        )
        st.caption(f"{len(_catalog_options_v152)} registro(s) RRi conservado(s) en la base interna.")
        if LONGITUDINAL_DB_PATH.exists():
            st.download_button(
                "Descargar copia completa de la base",
                LONGITUDINAL_DB_PATH.read_bytes(),
                file_name="modelo_predictivo_salud_v1539.sqlite3",
                mime="application/octet-stream",
                key="download_full_db_v1537_sidebar",
            )

    _frozen_hist_v1537 = bool(_history_source_v1538 == "Histórico guardado" and _history_mode_v1537 == "Abrir histórico completo del paciente" and _selected_snapshot_v1537 is not None)
    _hist_cfg_v1537 = (_selected_snapshot_v1537 or {}).get('config', {}) if _frozen_hist_v1537 else {}
    min_rr = st.number_input("Mínimo RRi por ventana", min_value=10, max_value=300,
                             value=int(_hist_cfg_v1537.get('min_rr', 30)), step=5, disabled=_frozen_hist_v1537)
    include_rqa = st.checkbox("Calcular RQA", value=bool(_hist_cfg_v1537.get('include_rqa', False)),
                              help="Puede tardar en ventanas largas.", disabled=_frozen_hist_v1537)
    include_hvg = st.checkbox("Calcular HVG/grafos", value=bool(_hist_cfg_v1537.get('include_hvg', False)),
                              help="Más lento. Actívalo cuando ya tengas las ventanas definidas.", disabled=_frozen_hist_v1537)
    mse_zero_policy_label = st.selectbox(
        "Modo MSE si A=0",
        list(MSE_ZERO_MODE_OPTIONS.keys()),
        index=list(MSE_ZERO_MODE_OPTIONS.keys()).index(
            next((k for k, v in MSE_ZERO_MODE_OPTIONS.items() if v == (_hist_cfg_v1537.get("mse_zero_policy") if _frozen_hist_v1537 else st.session_state.get("mse_zero_policy", "nan"))), DEFAULT_MSE_ZERO_MODE_LABEL)
        ),
        disabled=_frozen_hist_v1537,
        help=(
            "Clásico deja no calculado cuando A=0. "
            "Los modos 0.5 y 1.0 aplican pseudoconteo para comparar con valores MSE de Kubios en escalas altas."
        ),
    )
    mse_zero_policy = MSE_ZERO_MODE_OPTIONS[mse_zero_policy_label]
    st.session_state["mse_zero_policy"] = mse_zero_policy
    st.caption(f"Modo MSE activo: {mse_zero_policy_label}")
    st.sidebar.info("Al cambiar este modo, la app recalcula SampEn/MSE en la siguiente ejecución.")

    sampen_theiler_label = st.selectbox(
        "Exclusión temporal SampEn/MSE",
        list(THEILER_WINDOW_OPTIONS.keys()),
        index=list(THEILER_WINDOW_OPTIONS.values()).index(
            _hist_cfg_v1537.get("sampen_theiler_window", st.session_state.get("sampen_theiler_window", 0))
        ) if _hist_cfg_v1537.get("sampen_theiler_window", st.session_state.get("sampen_theiler_window", 0)) in THEILER_WINDOW_OPTIONS.values() else 0,
        disabled=_frozen_hist_v1537,
        help="Prueba tipo ventana de Theiler. Excluye comparaciones entre patrones próximos en el tiempo."
    )
    st.session_state["sampen_theiler_window"] = THEILER_WINDOW_OPTIONS[sampen_theiler_label]
    st.caption(f"Theiler activo: {st.session_state['sampen_theiler_window']} beat(s)")

    mse_radius_label = st.selectbox(
        "Radio r para SampEn/MSE",
        list(MSE_RADIUS_MODE_OPTIONS.keys()),
        index=list(MSE_RADIUS_MODE_OPTIONS.values()).index(_hist_cfg_v1537.get("mse_radius_mode", st.session_state.get("mse_radius_mode", "fixed_entropy_sd")))
        if _hist_cfg_v1537.get("mse_radius_mode", st.session_state.get("mse_radius_mode", "fixed_entropy_sd")) in MSE_RADIUS_MODE_OPTIONS.values() else 0,
        disabled=_frozen_hist_v1537,
        help="Compara r fijo con λ500, r por escala y r fijo del RR corregido sin λ."
    )
    st.session_state["mse_radius_mode"] = MSE_RADIUS_MODE_OPTIONS[mse_radius_label]
    st.caption(f"Radio activo: {mse_radius_label}")
    artifact_level = st.selectbox(
        "Corrección de artefactos",
        ["none", "very low", "low", "medium", "strong", "very strong", "kubios scientific"],
        index=["none", "very low", "low", "medium", "strong", "very strong", "kubios scientific"].index(_hist_cfg_v1537.get('artifact_level','none')) if _hist_cfg_v1537.get('artifact_level','none') in ["none", "very low", "low", "medium", "strong", "very strong", "kubios scientific"] else 0,
        help="v12.0: mediana local + dRR adaptativo + patrones NP/PN/NPN/PNP + interpolación cúbica. Use kubios scientific para máxima aproximación.",
        disabled=_frozen_hist_v1537,
    )
    domain_method = st.selectbox("Cálculo dominios", ["median", "mean"],
                                 index=["median","mean"].index(_hist_cfg_v1537.get('domain_method','median')) if _hist_cfg_v1537.get('domain_method','median') in ["median","mean"] else 0,
                                 disabled=_frozen_hist_v1537)
    st.caption("Consejo: para ventanas de ~30 s usa mínimo RRi 20-30; para 5 min usa 30-110 según el caso.")

record_data = {}
errors = []
loaded_from_history_v152 = []
newly_saved_v152 = []

if auto_load_history_v152 and selected_saved_keys_v152:
    for _key, _stored in load_saved_raw_records(selected_saved_keys_v152).items():
        try:
            rr_raw = np.asarray(_stored['rr_raw'], dtype=float)
            _snapshot = load_analysis_snapshot(_key)
            if _snapshot is not None and _snapshot.get('rr_corrected') is not None:
                rr = np.asarray(_snapshot['rr_corrected'], dtype=float)
                artifact_mask = np.asarray(_snapshot.get('artifact_mask', np.zeros(len(rr), dtype=bool)), dtype=bool)
                artifact_info = {'source': 'snapshot_v15.3.14', 'restored': True}
            else:
                rr, artifact_mask, artifact_info = correct_artifacts_kubios_like(rr_raw, level=artifact_level)
            name = sanitize_name(_stored.get('record_name') or _stored.get('filename') or _key)
            base, k = name, 2
            while name in record_data:
                name = f"{base}_{k}"
                k += 1
            record_data[name] = {
                "rr": rr,
                "rr_raw": rr_raw,
                "artifact_mask": artifact_mask,
                "artifact_info": artifact_info,
                "duration": float(np.sum(rr)),
                "filename": _stored.get('filename') or name,
                "record_key": _key,
                "source": "historial",
                "analysis_snapshot": _snapshot,
            }
            loaded_from_history_v152.append(name)
        except Exception as e:
            errors.append(f"Historial {_stored.get('record_name', _key)}: {e}")

for uf in (uploaded_files or []):
    try:
        rr_raw = read_rri_file(uf)
        rr, artifact_mask, artifact_info = correct_artifacts_kubios_like(rr_raw, level=artifact_level)
        name = sanitize_name(uf.name)
        base, k = name, 2

        existing_same = [r for r, d in record_data.items() if sanitize_name(d.get('filename', '')) == sanitize_name(uf.name)]
        for r in existing_same:
            record_data.pop(r, None)
            if r in loaded_from_history_v152:
                loaded_from_history_v152.remove(r)

        while name in record_data:
            name = f"{base}_{k}"
            k += 1

        record_key = save_raw_record_to_db(name, uf.name, rr_raw)
        record_data[name] = {
            "rr": rr,
            "rr_raw": rr_raw,
            "artifact_mask": artifact_mask,
            "artifact_info": artifact_info,
            "duration": float(np.sum(rr)),
            "filename": uf.name,
            "record_key": record_key,
            "source": "subido",
        }
        newly_saved_v152.append(name)
    except Exception as e:
        errors.append(f"{uf.name}: {e}")

if errors:
    st.error("\n".join(errors))

if '_frozen_hist_v1537' in globals() and _frozen_hist_v1537:
    st.info(
        f"Modo histórico congelado: {_selected_snapshot_record_v1537 or 'registro seleccionado'}. "
        "Solo este histórico longitudinal está activo. Cada registro recupera su propia configuración y métricas guardadas; "
        "las gráficas se reconstruyen visualmente a partir de esos resultados, sin recalcular el análisis fisiológico."
    )

if not record_data:
    st.info("Sube un registro RRi o activa algún registro del historial persistente v15.2.")
    st.stop()

# Orden cronológico de más antiguo a más reciente usando la fecha del nombre del archivo.
record_data = sort_records_chronologically(record_data)

records = list(record_data.keys())
# v15.3.8 · contexto visible y aislado.
if _history_source_v1538 == "Sesión actual":
    _session_patients_v1538 = sorted({normalize_patient_id(infer_patient_id(r)) for r in records if r})
    st.sidebar.caption("Fuente: Sesión actual")
    if _session_patients_v1538:
        st.sidebar.caption("Paciente(s) activo(s): " + ", ".join(_session_patients_v1538))
else:
    st.sidebar.caption("Fuente: Histórico guardado")
selected_record = st.sidebar.selectbox("Registro principal", records)
t_max = record_data[selected_record]["duration"]

# ============================================================
# Estado robusto de segmentación
# ============================================================
if "selected_record_v50" not in st.session_state or st.session_state.selected_record_v50 != selected_record:
    st.session_state.selected_record_v50 = selected_record

st.session_state.setdefault("global_windows_v50", empty_windows())
st.session_state.setdefault("record_windows_v50", {})
st.session_state.setdefault("restored_records_v152", [])
# v15.3.14: solo el histórico o la opción explícita "Recuperar segmentación anterior"
# restauran ventanas persistentes. Una sesión nueva parte de cero y permanece editable.
_restore_segmentation_v15314 = (
    _history_source_v1538 == "Histórico guardado"
    or globals().get("_session_seg_mode_v15314", "Nueva segmentación (editable)") == "Recuperar segmentación anterior"
)
for rec in records:
    if rec not in st.session_state.record_windows_v50:
        st.session_state.record_windows_v50[rec] = empty_windows()
    if rec not in st.session_state.restored_records_v152:
        # v15.4.0 · en una visita incremental los registros históricos conservan
        # sus ventanas previas, mientras que el registro recién subido empieza vacío
        # y puede segmentarse libremente.
        _is_incremental_history_v15327 = bool(
            _history_source_v1538 == "Sesión actual"
            and st.session_state.get("incremental_target_patient_v15325")
            and record_data.get(rec, {}).get("source") == "historial"
        )
        if _restore_segmentation_v15314 or _is_incremental_history_v15327:
            _restored_w = load_record_segmentation(rec, record_data[rec].get("filename", rec))
            if any(_restored_w.get(ph) is not None for ph in PHASES):
                st.session_state.record_windows_v50[rec] = _restored_w
        else:
            st.session_state.record_windows_v50[rec] = empty_windows()
        st.session_state.restored_records_v152.append(rec)

_restored_active_v152 = []
for _rec in records:
    for _ph, _win in st.session_state.record_windows_v50.get(_rec, {}).items():
        if _win is not None and _ph not in _restored_active_v152:
            _restored_active_v152.append(_ph)

st.session_state.setdefault("pending_selections_v1522", {})
# Compatibilidad con sesiones de versiones anteriores.
if "pending_selection_v50" in st.session_state and st.session_state.pending_selection_v50 is not None:
    st.session_state.pending_selections_v1522[selected_record] = list(st.session_state.pending_selection_v50)
st.session_state.pending_selection_v50 = st.session_state.pending_selections_v1522.get(selected_record)

# v15.3.14: el último tramo persistente solo se restaura si el usuario abre histórico
# o solicita explícitamente recuperar la segmentación anterior.
_restore_pending_selected_v15327 = bool(
    _restore_segmentation_v15314
    or (_history_source_v1538 == "Sesión actual"
        and st.session_state.get("incremental_target_patient_v15325")
        and record_data.get(selected_record, {}).get("source") == "historial")
)
if selected_record not in st.session_state.pending_selections_v1522 and _restore_pending_selected_v15327:
    _saved_pending_v1522 = load_temporal_selection(
        selected_record,
        record_data[selected_record].get("filename", selected_record),
    )
    if _saved_pending_v1522 is not None:
        st.session_state.pending_selections_v1522[selected_record] = _saved_pending_v1522
        st.session_state.pending_selection_v50 = _saved_pending_v1522

st.session_state.setdefault("active_phases_v50", _restored_active_v152 or ["Basal"])
st.session_state.setdefault("use_independent_v70", True)

with st.sidebar.expander("Segmentación", expanded=True):
    use_independent = st.checkbox("Ventanas independientes por registro", value=st.session_state.get("use_independent_v70", False), key="use_independent_checkbox_v70")
    st.session_state.use_independent_v70 = use_independent
    active_phases = st.multiselect("Fases activas para calcular", PHASES, default=st.session_state.active_phases_v50)
    st.session_state.active_phases_v50 = active_phases

    c_basal, c_rec = st.columns(2)
    with c_basal:
        if st.button("Activar basales", help="Activa Basal, Basal2, Basal3, Basal4 y Basal5"):
            st.session_state.active_phases_v50 = [p for p in PHASES if PHASE_GROUP.get(p) == "Basal"]
            st.rerun()
    with c_rec:
        if st.button("Activar recuperaciones", help="Activa R1-R6"):
            st.session_state.active_phases_v50 = [p for p in PHASES if PHASE_GROUP.get(p) == "Recuperación"]
            st.rerun()

    if st.button("Limpiar todas las ventanas"):
        st.session_state.global_windows_v50 = empty_windows()
        st.session_state.record_windows_v50 = {rec: empty_windows() for rec in records}
        st.session_state.pending_selection_v50 = None
        st.session_state.pending_selections_v1522 = {}
        if _history_source_v1538 != "Sesión actual":
            for _rec in records:
                clear_temporal_selection(_rec, record_data[_rec].get("filename", _rec))
        st.rerun()

    if st.button("Autodividir todo el registro"):
        if use_independent:
            st.session_state.record_windows_v50[selected_record] = default_windows(t_max)
        else:
            st.session_state.global_windows_v50 = default_windows(t_max)
        st.session_state.active_phases_v50 = PHASES.copy()
        st.rerun()

    if use_independent and st.button("Copiar ventanas del registro principal a todos"):
        base_w = st.session_state.record_windows_v50.get(selected_record, empty_windows())
        st.session_state.record_windows_v50 = {rec: {ph: (list(base_w[ph]) if base_w[ph] is not None else None) for ph in PHASES} for rec in records}
        st.rerun()

# v15.3.14: durante la Sesión actual las ventanas son de trabajo (scratch).
# No se escriben en SQLite hasta el guardado explícito del análisis.
_persist_segmentation_edits_v15314 = bool(_history_source_v1538 != "Sesión actual")

if _history_source_v1538 == "Sesión actual" and uploaded_files:
    st.sidebar.caption(
        "Segmentación actual: editable y temporal. El histórico no se modifica hasta «Guardar análisis actual en el historial»."
    )

if artifact_level != "none":
    with st.sidebar.expander("Resumen artefactos", expanded=True):
        for rec, data in record_data.items():
            info = data.get("artifact_info", {})
            st.write(f"**{rec}**: {info.get('n_artifacts', 0)} ({info.get('percent_artifacts', 0):.2f}%)")



# ============================================================
# v15.1 · MOTOR DE PREDICCIÓN FISIOLÓGICA CONTINUA
# Funciona desde 1 observación (pronóstico basal de persistencia) y mejora con historia longitudinal.
# ============================================================
V15_INDEX_FEATURES = list(GB_INDEX_FEATURES)
V15_HVG_CORE_FEATURES = ['HVG_degree_mean','HVG_degree_max','HVG_hubs_p90','HVG_clustering','HVG_lambda','HVG_path_length','HVG_diameter','HVG_compactness_index']
V15_HVG_SCORE_FEATURES = ['HVG_graph_score_small_world','HVG_graph_score_scale_free']
V15_HVG_FEATURES = V15_HVG_CORE_FEATURES + V15_HVG_SCORE_FEATURES
V15_PREDICTIVE_TARGETS = V15_INDEX_FEATURES + V15_HVG_FEATURES
HVG_LABELS = {
    'HVG_degree_mean':'conectividad media','HVG_degree_max':'grado máximo','HVG_hubs_p90':'hubs',
    'HVG_clustering':'clustering','HVG_lambda':'lambda','HVG_path_length':'longitud de camino',
    'HVG_diameter':'diámetro','HVG_compactness_index':'compactación',
    'HVG_graph_score_small_world':'score small-world','HVG_graph_score_scale_free':'score scale-free'}


def _v15_patient_phase_history(history_df, patient_id, phase):
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    df = history_df[(history_df['Paciente_ID'].astype(str) == str(patient_id)) &
                    (history_df['Fase'].astype(str) == str(phase))].copy()
    if df.empty:
        return df
    df['_dt'] = pd.to_datetime(df.get('Fecha_hora'), errors='coerce')
    df = df.sort_values(['_dt', 'saved_at'], na_position='last').reset_index(drop=True)
    return df


def _v15_time_axis(df):
    """Eje temporal en días; usa orden secuencial si faltan fechas válidas."""
    dt = pd.to_datetime(df.get('_dt'), errors='coerce')
    if len(df) >= 2 and dt.notna().sum() >= 2:
        first = dt.dropna().iloc[0]
        t = (dt - first).dt.total_seconds() / 86400.0
        t = t.interpolate(limit_direction='both')
        vals = t.to_numpy(dtype=float)
        if np.all(np.isfinite(vals)) and np.ptp(vals) > 0:
            return vals, 'días'
    return np.arange(len(df), dtype=float), 'registros'


def _v15_robust_slope(t, y):
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = np.asarray(t)[mask], np.asarray(y)[mask]
    if len(y) < 2:
        return 0.0, float(y[-1]) if len(y) else np.nan
    slopes=[]
    for i in range(len(y)-1):
        for j in range(i+1,len(y)):
            dt=t[j]-t[i]
            if abs(dt)>1e-12:
                slopes.append((y[j]-y[i])/dt)
    slope=float(np.median(slopes)) if slopes else 0.0
    intercept=float(np.median(y-slope*t))
    return slope,intercept


def _v15_backtest_mae(t, y):
    """Error walk-forward del mismo predictor, sin usar información futura."""
    y=np.asarray(y,dtype=float); t=np.asarray(t,dtype=float)
    errors=[]
    for k in range(2,len(y)):
        tk,yk=t[:k],y[:k]
        if not np.isfinite(yk).all():
            continue
        slope,intercept=_v15_robust_slope(tk,yk)
        step=np.median(np.diff(tk)) if len(tk)>1 else 1.0
        step=step if np.isfinite(step) and step>0 else 1.0
        pred=intercept+slope*(tk[-1]+step)
        errors.append(abs(y[k]-pred))
    return float(np.median(errors)) if errors else np.nan


def _v15_clip_target(metric, value):
    try: v=float(value)
    except Exception: return np.nan
    if not np.isfinite(v): return np.nan
    if metric in V15_INDEX_FEATURES or metric in V15_HVG_SCORE_FEATURES: return float(np.clip(v,0,100))
    if metric=='HVG_clustering': return float(np.clip(v,0,1))
    if metric=='HVG_compactness_index': return float(np.clip(v,-2.5,2.5))
    if metric in {'HVG_degree_mean','HVG_degree_max','HVG_hubs_p90','HVG_path_length','HVG_diameter'}: return float(max(0,v))
    return v


def _hvg_robust_change_scale(y):
    y=np.asarray(y,dtype=float); y=y[np.isfinite(y)]
    if len(y)>=3:
        d=np.diff(y); med=np.nanmedian(d); mad=1.4826*np.nanmedian(np.abs(d-med))
        if np.isfinite(mad) and mad>1e-9: return float(mad)
        q75,q25=np.nanpercentile(d,[75,25]); s=(q75-q25)/1.349
        if np.isfinite(s) and s>1e-9: return float(s)
    if len(y)>=2 and abs(y[-1]-y[-2])>1e-9: return abs(float(y[-1]-y[-2]))
    if len(y): return max(abs(float(np.nanmedian(y)))*0.05,1e-6)
    return np.nan


def _hvg_reorg_label(score):
    if not np.isfinite(score): return 'No estimable'
    if score<0.50: return 'Estable'
    if score<1.00: return 'Reorganización leve'
    if score<2.00: return 'Reorganización moderada'
    return 'Reorganización marcada'


def _hvg_score_weight(metric,current,hist_values):
    if metric not in V15_HVG_SCORE_FEATURES: return 1.0
    w=0.25
    if np.isfinite(current) and (current>=95 or current<=5): w*=0.15
    vals=np.asarray(hist_values,dtype=float); vals=vals[np.isfinite(vals)]
    if len(vals)>=3 and np.nanstd(vals)<1.0: w*=0.25
    return float(w)


def _hvg_direction_text(table,change_col='Cambio_observado',z_col='Z_observado'):
    if table is None or table.empty: return 'Sin cambios topológicos estimables.'
    ranked=[]
    for _,r in table.iterrows():
        z=pd.to_numeric(r.get(z_col),errors='coerce'); d=pd.to_numeric(r.get(change_col),errors='coerce')
        if not (np.isfinite(z) and np.isfinite(d)) or abs(z)<0.35 or abs(d)<1e-12: continue
        metric=str(r.get('Métrica','')); w=pd.to_numeric(r.get('Peso_topológico'),errors='coerce')
        if metric in V15_HVG_SCORE_FEATURES and np.isfinite(w) and w<0.10: continue
        ranked.append((abs(float(z)),'↑' if d>0 else '↓',HVG_LABELS.get(metric,metric)))
    ranked=sorted(ranked,reverse=True)[:4]
    return ', '.join(f'{a} {n}' for _,a,n in ranked)+'.' if ranked else 'Sin dirección dominante respecto al propio histórico.'


def predict_v15_hvg_topology(history_df,patient_id,phase,horizon_days=None):
    df=_v15_patient_phase_history(history_df,patient_id,phase)
    if df.empty: return None
    t,unit=_v15_time_axis(df); difft=np.diff(t); default_step=float(np.nanmedian(difft[difft>0])) if np.any(difft>0) else 1.0
    step=float(horizon_days) if (horizon_days is not None and unit=='días' and np.isfinite(horizon_days) and float(horizon_days)>0) else default_step
    tnext=float(t[-1]+step)
    rows=[]; os=[]; ow=[]; ps=[]; pw=[]
    for c in V15_HVG_FEATURES:
        if c not in df.columns: continue
        y=pd.to_numeric(df[c],errors='coerce').to_numpy(dtype=float); mask=np.isfinite(y)
        if mask.sum()<1: continue
        tv=t[mask]; yv=y[mask]; last=float(yv[-1]); n=len(yv)
        if n==1: last_delta=0.0; slope=0.0; raw=last
        elif n==2: last_delta=float(yv[-1]-yv[-2]); _ratio=float(step/default_step) if default_step>1e-12 else 1.0; raw=last+0.60*last_delta*_ratio; slope=0.60*last_delta/(default_step if default_step>0 else 1)
        else:
            last_delta=float(yv[-1]-yv[-2]); slope,intercept=_v15_robust_slope(tv,yv); trend=intercept+slope*tnext; _ratio=float(step/default_step) if default_step>1e-12 else 1.0; momentum=last+0.55*last_delta*_ratio; wt=min(0.85,0.55+0.05*n); raw=wt*trend+(1-wt)*momentum
        pred=_v15_clip_target(c,raw); scale=_hvg_robust_change_scale(yv); obs=float(yv[-1]-yv[-2]) if n>=2 else np.nan; pdlt=float(pred-last)
        zo=abs(obs)/scale if np.isfinite(obs) and np.isfinite(scale) and scale>0 else np.nan; zp=abs(pdlt)/scale if np.isfinite(pdlt) and np.isfinite(scale) and scale>0 else np.nan
        w=_hvg_score_weight(c,last,yv)
        if np.isfinite(zo): os.append(min(float(zo),4)); ow.append(w)
        if np.isfinite(zp): ps.append(min(float(zp),4)); pw.append(w)
        bt=_v15_backtest_mae(tv,yv) if n>=3 else np.nan; unc=float(bt) if np.isfinite(bt) else (scale if np.isfinite(scale) else np.nan)
        rows.append({'Métrica':c,'Actual':last,'Predicción_t+1':pred,'Cambio_previsto':pdlt,'Cambio_observado':obs,'Escala_personal_cambio':scale,'Z_observado':zo,'Z_previsto':zp,'Peso_topológico':w,'Incertidumbre_1SD':unc,f'Pendiente_por_{unit}':slope})
    if not rows: return None
    tab=pd.DataFrame(rows)
    def agg(v,w):
        v=np.asarray(v,dtype=float); w=np.asarray(w,dtype=float); m=np.isfinite(v)&np.isfinite(w)&(w>0)
        return float(np.sqrt(np.average(v[m]**2,weights=w[m]))) if m.any() else np.nan
    so,sp=agg(os,ow),agg(ps,pw)
    return {'tabla':tab,'score_actual':so,'descriptor_actual':_hvg_reorg_label(so),'direccion_actual':_hvg_direction_text(tab,'Cambio_observado','Z_observado'),'score_previsto':sp,'descriptor_previsto':_hvg_reorg_label(sp),'direccion_prevista':_hvg_direction_text(tab,'Cambio_previsto','Z_previsto'),'n_historial':len(df),'registro_actual':df.iloc[-1].get('Registro','')}


def predict_v15_continuous(history_df, patient_id, phase, horizon_days=None):
    """
    Predice directamente los seis índices del siguiente registro.

    - 1 observación: pronóstico basal de persistencia (el siguiente estado se centra en el actual)
      con incertidumbre amplia. No inventa una tendencia inexistente.
    - 2 observaciones: extrapolación amortiguada del último cambio.
    - >=3: tendencia robusta Theil-Sen simplificada + corrección de impulso reciente.
    - Incertidumbre: error walk-forward y dispersión de residuos/diferencias.
    """
    df=_v15_patient_phase_history(history_df,patient_id,phase)
    if len(df)<1:
        return None
    t,unit=_v15_time_axis(df)
    difft=np.diff(t)
    default_step=float(np.nanmedian(difft[difft>0])) if np.any(difft>0) else 1.0
    step=float(horizon_days) if (horizon_days is not None and unit=='días' and np.isfinite(horizon_days) and float(horizon_days)>0) else default_step
    tnext=float(t[-1]+step)
    rows=[]
    current=[]; predicted=[]; uncertainties=[]
    for c in V15_INDEX_FEATURES:
        y=pd.to_numeric(df.get(c),errors='coerce').to_numpy(dtype=float)
        mask=np.isfinite(y)
        if mask.sum()<1:
            continue
        tv=t[mask]; yv=y[mask]
        last=float(yv[-1])
        if len(yv)==1:
            # Pronóstico basal: persistencia del estado actual, sin dirección temporal inventada.
            last_delta=0.0
            slope=0.0
            raw_pred=last
        elif len(yv)==2:
            last_delta=float(yv[-1]-yv[-2])
            # Evita que dos puntos generen extrapolaciones extremas.
            shrink=0.60
            _ratio=float(step/default_step) if default_step>1e-12 else 1.0
            raw_pred=last+shrink*last_delta*_ratio
            slope=(shrink*last_delta)/(default_step if default_step>0 else 1.0)
        else:
            last_delta=float(yv[-1]-yv[-2])
            slope,intercept=_v15_robust_slope(tv,yv)
            trend_pred=intercept+slope*tnext
            _ratio=float(step/default_step) if default_step>1e-12 else 1.0
            momentum_pred=last+0.55*last_delta*_ratio
            weight_trend=min(0.85,0.55+0.05*len(yv))
            raw_pred=weight_trend*trend_pred+(1-weight_trend)*momentum_pred
        pred=float(np.clip(raw_pred,0,100))
        residuals=yv-(np.polyval(np.polyfit(tv,yv,1),tv) if len(yv)>=3 and np.ptp(tv)>0 else np.full_like(yv,np.mean(yv)))
        if len(yv)==1:
            residual_sd=np.nan
            diff_sd=np.nan
            bt=np.nan
            # Intervalo deliberadamente amplio: hay estado actual, pero no trayectoria observada.
            unc=12.5
        else:
            residual_sd=float(np.nanstd(residuals,ddof=1)) if len(yv)>2 else abs(last_delta)*0.5
            diff_sd=float(np.nanstd(np.diff(yv),ddof=1)) if len(yv)>2 else abs(last_delta)*0.5
            bt=_v15_backtest_mae(tv,yv)
            candidates=[x for x in [bt,residual_sd,diff_sd] if np.isfinite(x)]
            unc=max(2.0,float(np.median(candidates)) if candidates else 5.0)
        unc=float(np.clip(unc,2.0,25.0))
        lo=max(0.0,pred-1.96*unc); hi=min(100.0,pred+1.96*unc)
        direction='Ascenso' if pred-last>1.0 else ('Descenso' if pred-last<-1.0 else 'Estable')
        rows.append({'Índice':c,'Actual':last,'Predicción_t+1':pred,'Cambio_previsto':pred-last,
                     'Dirección':direction,'Incertidumbre_1SD':unc,'IC95_inferior':lo,'IC95_superior':hi,
                     f'Pendiente_por_{unit}':slope})
        current.append(last); predicted.append(pred); uncertainties.append(unc)
    if not rows:
        return None
    table=pd.DataFrame(rows)
    # Compuesto favorable: rigidez resta; los demás índices positivos salvo regulación lenta, neutral.
    def comp(vals):
        m=dict(zip([r['Índice'] for r in rows],vals))
        parts=[]
        for key,w,sign in [('IDX_Vagal',1,1),('IDX_Amplitud',1,1),('IDX_Complejidad',1,1),
                           ('IDX_Rigidez',1,-1),('IDX_Adaptabilidad',1.2,1)]:
            if key in m: parts.append((sign*m[key],w))
        return sum(v*w for v,w in parts)/sum(w for _,w in parts) if parts else np.nan
    c0,c1=comp(current),comp(predicted)
    delta=float(c1-c0) if np.isfinite(c0) and np.isfinite(c1) else np.nan
    state='Favorable' if delta>3 else ('Desfavorable' if delta<-3 else 'Estable')
    mean_unc=float(np.mean(uncertainties))
    n=len(df)
    confidence=float(np.clip(100*(1-mean_unc/35.0)*(1-np.exp(-n/4.0)),3,95))
    # Novedad del último punto respecto a su propia historia previa.
    novelty=[]
    if len(df)>=3:
        for c in V15_INDEX_FEATURES:
            y=pd.to_numeric(df.get(c),errors='coerce').to_numpy(dtype=float)
            if len(y)>=3 and np.isfinite(y[-1]) and np.isfinite(y[:-1]).sum()>=2:
                med=np.nanmedian(y[:-1]); mad=np.nanmedian(np.abs(y[:-1]-med))*1.4826
                if mad>1e-6: novelty.append(abs(y[-1]-med)/mad)
    anomaly=float(np.nanmedian(novelty)) if novelty else np.nan
    anomaly_label=('Alta' if anomaly>=3 else ('Moderada' if anomaly>=2 else 'Baja')) if np.isfinite(anomaly) else 'No estimable'
    topology=predict_v15_hvg_topology(history_df,patient_id,phase,horizon_days=horizon_days)
    return {'tabla':table,'estado_previsto':state,'compuesto_actual':c0,'compuesto_predicho':c1,
            'delta_compuesto':delta,'confianza':confidence,'n_historial':n,'unidad_tiempo':unit,
            'paso_previsto':step,'horizonte_dias':(float(step) if unit=='días' else np.nan),'registro_actual':df.iloc[-1].get('Registro',''),
            'anomalia_score':anomaly,'anomalia_label':anomaly_label,'history':df,'topologia_hvg':topology,
            'modo_prediccion':('Persistencia basal (1 registro)' if n==1 else ('Extrapolación amortiguada (2 registros)' if n==2 else 'Tendencia robusta longitudinal'))}


# ============================================================
# v15.5.2 · MOTOR HÍBRIDO XGBOOST + CAPA PROBABILÍSTICA CALIBRADA INDEPENDIENTE
# Nivel 1: tendencia fisiológica robusta e interpretable.
# Nivel 2: XGBoost aprende el residuo del Nivel 1 y solo se activa
#          si mejora la validación temporal sin fuga de información.
# ============================================================
# Persistencia controlada: por defecto local; puede apuntar a un volumen externo
# mediante la variable de entorno VRC_MODEL_DIR (servidor propio, volumen montado, etc.).
V153_STORAGE_DIR = Path(os.environ.get('VRC_MODEL_DIR', str(APP_DATA_DIR))).expanduser().resolve()
V153_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
V153_MODEL_PATH = V153_STORAGE_DIR / 'motor_hibrido_v1550_xgboost_horizonte_activo.joblib'
V1552_PROBABILITY_PATH = V153_STORAGE_DIR / 'probability_layer_v1552_xgboost_calibrated.joblib'
V153_CANDIDATE_PATH = V153_STORAGE_DIR / 'motor_hibrido_v15411_xgboost_horizonte_candidato.joblib'
V153_ARCHIVE_DIR = V153_STORAGE_DIR / 'model_archive'


def _v1551_imported_model_state():
    """Estado de sesión del modelo importado manualmente.

    Un bundle importado y validado tiene prioridad durante toda la sesión. Esto evita
    que una restauración cloud posterior al rerun sustituya silenciosamente el modelo
    recién importado por una copia anterior de Supabase.
    """
    try:
        return st.session_state.setdefault('_v1551_imported_xgb_model', {})
    except Exception:
        if not hasattr(_v1551_imported_model_state, '_state'):
            _v1551_imported_model_state._state = {}
        return _v1551_imported_model_state._state


def _cloud_restore_model_once_v1540():
    state = _cloud_state_v1540()
    # v15.5.1: un modelo importado manualmente en esta sesión es autoritativo.
    pinned = _v1551_imported_model_state()
    if pinned.get('bundle') is not None:
        state['_model_restore_checked'] = True
        return
    if state.get('_model_restore_checked'):
        return
    state['_model_restore_checked'] = True
    if V153_MODEL_PATH.exists() or not _persistent_db_url_v1540():
        return
    remote = _cloud_get_object_v1540(CLOUD_MODEL_KEY_V1540)
    if remote and remote.get('payload'):
        try:
            V153_MODEL_PATH.write_bytes(remote['payload'])
            state['_model_cloud_sha'] = remote.get('sha256')
        except Exception as exc:
            state['last_error'] = str(exc)


def _cloud_push_active_model_v1540(reason='model_updated', payload=None):
    raw = bytes(payload) if payload is not None else (V153_MODEL_PATH.read_bytes() if V153_MODEL_PATH.exists() else b'')
    if not raw or not _persistent_db_url_v1540():
        return False
    expected_sha = hashlib.sha256(raw).hexdigest()
    ok, _ = _cloud_put_bytes_v1540(CLOUD_MODEL_KEY_V1540, raw, reason=reason, force=True)
    if not ok:
        return False
    # v15.5.1: verificación inmediata de que la nube contiene EXACTAMENTE el modelo subido.
    remote = _cloud_get_object_v1540(CLOUD_MODEL_KEY_V1540)
    verified = bool(remote and remote.get('sha256') == expected_sha)
    state = _cloud_state_v1540()
    state['_model_cloud_sha'] = remote.get('sha256') if remote else None
    state['_model_cloud_verified'] = verified
    if not verified:
        state['last_error'] = 'El modelo se escribió, pero no pudo verificarse el SHA-256 remoto.'
    return verified
V153_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
V153_MIN_EXAMPLES_DEFAULT = 100
V153_NEW_TRANSITIONS_DEFAULT = 25

# v15.4.12 · Capa 2: parámetros fisiológicos primarios seleccionados.
# Se guardan longitudinalmente y se entregan a XGBoost sin sustituir los índices compuestos.
V1548_PRIMARY_FEATURES = [
    'SDNN','RMSSD','SD1','SD2','HF','LF','VLF','MeanHR',
    'DFA_alpha1','DFA_alpha2','SampEn','D2','Hurst','Lyapunov_LLE',
    'REC','DET','Lmean','Lmax',
    'VLF_WAV_MEAN','LF_WAV_MEAN','HF_WAV_MEAN',
    'VLF_DOM_PCT','LF_DOM_PCT','HF_DOM_PCT',
    'WAV_ENTROPY_BANDS','WAV_ENTROPY_GLOBAL',
    'N_RRi','Duration_s','Artefactos_pct',
]
V1548_ALL_DYNAMIC_BASES = V15_PREDICTIVE_TARGETS + V1548_PRIMARY_FEATURES
V153_FEATURES = []
# Capa 1 mantiene además la predicción e incertidumbre del Nivel 1.
for _c153 in V15_PREDICTIVE_TARGETS:
    V153_FEATURES += [f'{_c153}__actual',f'{_c153}__pred_n1',f'{_c153}__delta_n1',f'{_c153}__unc_n1']
# Capa 3: dinámica longitudinal homogénea para índices, HVG y fisiología primaria.
for _c153 in V1548_ALL_DYNAMIC_BASES:
    V153_FEATURES += [f'{_c153}__actual_dyn',f'{_c153}__delta_ultimo',f'{_c153}__media3',f'{_c153}__pendiente',f'{_c153}__variabilidad_reciente']
V153_FEATURES += ['HVG_reorg_score_actual','HVG_reorg_score_previsto','n_historial','paso_previsto','horizonte_dias','fase_rango','grupo_basal','grupo_ejercicio','grupo_recuperacion']

def _v153_phase_features(phase):
    phase = str(phase)
    rank = PHASES.index(phase) if phase in PHASES else 999
    grp = PHASE_GROUP.get(phase, '')
    return {
        'fase_rango': float(rank),
        'grupo_basal': float(grp == 'Basal'),
        'grupo_ejercicio': float(grp == 'Ejercicio'),
        'grupo_recuperacion': float(grp == 'Recuperación'),
    }


def _v1548_recent_dynamics(hist, c):
    y=pd.to_numeric(hist.get(c),errors='coerce').dropna() if c in hist else pd.Series(dtype=float)
    if len(y)==0:
        return {'actual_dyn':np.nan,'delta_ultimo':np.nan,'media3':np.nan,'pendiente':np.nan,'variabilidad_reciente':np.nan}
    actual=float(y.iloc[-1])
    delta=float(y.iloc[-1]-y.iloc[-2]) if len(y)>=2 else 0.0
    tail=y.tail(3)
    mean3=float(tail.mean())
    var_recent=float(tail.std(ddof=1)) if len(tail)>=2 else 0.0
    # Pendiente robusta reciente: Theil-Sen simplificada sobre hasta 5 observaciones.
    yt=y.tail(5).to_numpy(dtype=float)
    tt=np.arange(len(yt),dtype=float)
    slope,_=_v15_robust_slope(tt,yt) if len(yt)>=2 else (0.0,actual)
    return {'actual_dyn':actual,'delta_ultimo':delta,'media3':mean3,'pendiente':float(slope),'variabilidad_reciente':var_recent}


def _v153_feature_row(level1_pred,phase,horizon_days=None):
    if not level1_pred: return None
    phys=level1_pred.get('tabla',pd.DataFrame()); phys=phys.set_index('Índice') if not phys.empty and 'Índice' in phys.columns else pd.DataFrame()
    topo_obj=level1_pred.get('topologia_hvg') or {}; topo=topo_obj.get('tabla',pd.DataFrame()) if isinstance(topo_obj,dict) else pd.DataFrame(); topo=topo.set_index('Métrica') if not topo.empty and 'Métrica' in topo.columns else pd.DataFrame()
    hist=level1_pred.get('history',pd.DataFrame()); feat={}
    # Capa 1: estado actual + predicción, cambio e incertidumbre del Nivel 1.
    for c in V15_PREDICTIVE_TARGETS:
        source=phys if c in V15_INDEX_FEATURES else topo
        if not source.empty and c in source.index:
            r=source.loc[c]
            feat[f'{c}__actual']=pd.to_numeric(r.get('Actual'),errors='coerce')
            feat[f'{c}__pred_n1']=pd.to_numeric(r.get('Predicción_t+1'),errors='coerce')
            feat[f'{c}__delta_n1']=pd.to_numeric(r.get('Cambio_previsto'),errors='coerce')
            feat[f'{c}__unc_n1']=pd.to_numeric(r.get('Incertidumbre_1SD'),errors='coerce')
        else:
            for suf in ('actual','pred_n1','delta_n1','unc_n1'): feat[f'{c}__{suf}']=np.nan
    # Capa 3: dinámica longitudinal de TODOS los predictores de capa 1 y 2.
    for c in V1548_ALL_DYNAMIC_BASES:
        dyn=_v1548_recent_dynamics(hist,c)
        for suf,val in dyn.items(): feat[f'{c}__{suf}']=val
    feat['HVG_reorg_score_actual']=pd.to_numeric(topo_obj.get('score_actual'),errors='coerce') if topo_obj else np.nan
    feat['HVG_reorg_score_previsto']=pd.to_numeric(topo_obj.get('score_previsto'),errors='coerce') if topo_obj else np.nan
    feat['n_historial']=float(level1_pred.get('n_historial',0)); feat['paso_previsto']=float(level1_pred.get('paso_previsto',1)); feat['horizonte_dias']=float(horizon_days if horizon_days is not None else level1_pred.get('horizonte_dias',level1_pred.get('paso_previsto',1))); feat.update(_v153_phase_features(phase)); return feat


def _v153_training_examples(history_df):
    if history_df is None or history_df.empty: return pd.DataFrame(),{c:np.array([]) for c in V15_PREDICTIVE_TARGETS},pd.DataFrame()
    work=history_df.copy(); work['Fecha_hora']=pd.to_datetime(work.get('Fecha_hora'),errors='coerce'); examples=[]; targets={c:[] for c in V15_PREDICTIVE_TARGETS}; meta=[]
    for (patient,phase),grp in work.groupby(['Paciente_ID','Fase'],dropna=False):
        grp=grp.sort_values(['Fecha_hora','saved_at'],na_position='last').reset_index(drop=True)
        for i in range(1,len(grp)):
            prev_dt=pd.to_datetime(grp.iloc[i-1].get('Fecha_hora'),errors='coerce'); next_dt=pd.to_datetime(grp.iloc[i].get('Fecha_hora'),errors='coerce')
            gap_days=float((next_dt-prev_dt).total_seconds()/86400.0) if pd.notna(prev_dt) and pd.notna(next_dt) and next_dt>prev_dt else np.nan
            p1=predict_v15_continuous(grp.iloc[:i].copy(),patient,phase,horizon_days=(gap_days if np.isfinite(gap_days) else None)); feat=_v153_feature_row(p1,phase,horizon_days=(gap_days if np.isfinite(gap_days) else None))
            if feat is None: continue
            nxt=grp.iloc[i]; examples.append(feat)
            for c in V15_PREDICTIVE_TARGETS: targets[c].append(pd.to_numeric(nxt.get(c),errors='coerce'))
            meta.append({'Paciente_ID':patient,'Fase':phase,'Fecha_objetivo':nxt.get('Fecha_hora'),'Registro_objetivo':nxt.get('Registro',''),'Horizonte_dias':gap_days})
    return pd.DataFrame(examples).reindex(columns=V153_FEATURES),{c:np.asarray(v,dtype=float) for c,v in targets.items()},pd.DataFrame(meta)

def _v153_history_signature(history_df):
    if history_df is None or history_df.empty:
        return 'empty'
    cols=[c for c in ['Paciente_ID','Registro','Fase','Fecha_hora','saved_at']+V15_PREDICTIVE_TARGETS+V1548_PRIMARY_FEATURES if c in history_df.columns]
    sort_cols=[c for c in ['Paciente_ID','Registro','Fase','Fecha_hora','saved_at'] if c in cols]
    payload=history_df[cols].astype(str).sort_values(sort_cols).to_csv(index=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _v153_bundle_score(bundle):
    """MAE final agregado, ponderado por el N del TEST intocable."""
    if not bundle:
        return np.nan
    vals=[]; weights=[]
    for m in bundle.get('metrics',{}).values():
        mae=pd.to_numeric(m.get('mae_test_hibrido',m.get('mae_hibrido')),errors='coerce')
        n=pd.to_numeric(m.get('n_test',m.get('n_val')),errors='coerce')
        if np.isfinite(mae) and np.isfinite(n) and n>0:
            vals.append(float(mae)); weights.append(float(n))
    return float(np.average(vals,weights=weights)) if vals else np.nan


def _v1548_select_features(X_train, corr_threshold=0.985, min_valid=8):
    """Selección entrenada EXCLUSIVAMENTE en Train para evitar fuga temporal.

    1) elimina columnas sin suficientes valores; 2) constantes/casi constantes;
    3) elimina una de cada pareja casi idéntica/correlacionada. No usa y ni Test.
    """
    if X_train is None or X_train.empty:
        return [], {'insufficient':[], 'constant':[], 'correlated':[]}
    kept=[]; dropped={'insufficient':[], 'constant':[], 'correlated':[]}
    for c in X_train.columns:
        v=pd.to_numeric(X_train[c],errors='coerce')
        vv=v[np.isfinite(v)]
        if len(vv)<int(min_valid):
            dropped['insufficient'].append(c); continue
        scale=max(1.0,float(np.nanmedian(np.abs(vv))))
        if float(np.nanstd(vv)) <= 1e-10*scale:
            dropped['constant'].append(c); continue
        kept.append(c)
    if len(kept)<=1:
        return kept,dropped
    Z=X_train[kept].apply(pd.to_numeric,errors='coerce')
    Z=Z.fillna(Z.median(numeric_only=True)).fillna(0.0)
    corr=Z.corr(method='spearman').abs()
    selected=[]
    for c in kept:
        redundant=False
        for k in selected:
            try:
                if float(corr.loc[c,k]) >= float(corr_threshold):
                    redundant=True; break
            except Exception:
                pass
        if redundant: dropped['correlated'].append(c)
        else: selected.append(c)
    return selected,dropped


def _v1548_feature_importance(pipe, selected_features):
    try:
        imp=np.asarray(pipe.named_steps['xgb'].feature_importances_,dtype=float)
        rows=[{'Variable':f,'Importancia':float(v)} for f,v in zip(selected_features,imp) if np.isfinite(v)]
        rows=sorted(rows,key=lambda r:r['Importancia'],reverse=True)
        return rows
    except Exception:
        return []


def _v153_walk_forward_diagnostic(X, y, base_all, ordered_pretest_idx, target_name, weight, n_splits=4):
    """Walk-forward expanding-window, con selección de variables repetida dentro de cada fold."""
    idx=np.asarray(ordered_pretest_idx,dtype=int)
    idx=idx[np.isfinite(y[idx]) & np.isfinite(base_all[idx])]
    if len(idx)<48 or weight<=0:
        return {'wf_folds':0,'wf_mae_nivel1':np.nan,'wf_mae_hibrido':np.nan,'wf_mejora_pct':np.nan}
    initial=max(40,int(np.floor(len(idx)*0.50))); remaining=len(idx)-initial
    if remaining<8:
        return {'wf_folds':0,'wf_mae_nivel1':np.nan,'wf_mae_hibrido':np.nan,'wf_mejora_pct':np.nan}
    fold_size=max(4,int(np.floor(remaining/max(1,n_splits))))
    base_err=[]; hybrid_err=[]; folds=0; cursor=initial
    while cursor < len(idx) and folds<n_splits:
        stop=len(idx) if folds==n_splits-1 else min(len(idx),cursor+fold_size)
        tr=idx[:cursor]; te=idx[cursor:stop]
        if len(tr)<40 or len(te)<4: break
        selected,_=_v1548_select_features(X.iloc[tr])
        if not selected: break
        residual=y-base_all
        pipe=Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('xgb',XGBRegressor(
                n_estimators=420, learning_rate=0.025, max_depth=3,
                min_child_weight=5.0, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.05, reg_lambda=1.5,
                objective='reg:pseudohubererror', eval_metric='mae',
                tree_method='hist', n_jobs=1, random_state=1548+folds, verbosity=0,
            ))
        ])
        pipe.fit(X.iloc[tr][selected],residual[tr])
        corr=pipe.predict(X.iloc[te][selected])
        pred=np.asarray([_v15_clip_target(target_name,v) for v in (base_all[te]+float(weight)*corr)],dtype=float)
        base_err.extend(np.abs(y[te]-base_all[te]).tolist()); hybrid_err.extend(np.abs(y[te]-pred).tolist())
        folds+=1; cursor=stop
    if not base_err:
        return {'wf_folds':0,'wf_mae_nivel1':np.nan,'wf_mae_hibrido':np.nan,'wf_mejora_pct':np.nan}
    b=float(np.mean(base_err)); h=float(np.mean(hybrid_err)); imp=(b-h)/b if b>1e-9 else 0.0
    return {'wf_folds':folds,'wf_mae_nivel1':b,'wf_mae_hibrido':h,'wf_mejora_pct':100*imp}



def _v15411_compound_from_row(values):
    """Compuesto favorable usado por el motor: Rigidez invierte signo; Regulación lenta no entra."""
    parts=[]
    for key,w,sign in [('IDX_Vagal',1.0,1.0),('IDX_Amplitud',1.0,1.0),('IDX_Complejidad',1.0,1.0),('IDX_Rigidez',1.0,-1.0),('IDX_Adaptabilidad',1.2,1.0)]:
        v=pd.to_numeric(values.get(key),errors='coerce') if hasattr(values,'get') else np.nan
        if np.isfinite(v): parts.append((sign*float(v),w))
    return float(sum(v*w for v,w in parts)/sum(w for _,w in parts)) if parts else np.nan


def _v15411_favorable_labels(X, targets):
    labels=[]; deltas=[]
    for i in range(len(X)):
        cur={c:pd.to_numeric(X.iloc[i].get(f'{c}__actual'),errors='coerce') for c in V15_INDEX_FEATURES}
        nxt={c:(targets[c][i] if c in targets and i<len(targets[c]) else np.nan) for c in V15_INDEX_FEATURES}
        c0=_v15411_compound_from_row(cur); c1=_v15411_compound_from_row(nxt)
        d=float(c1-c0) if np.isfinite(c0) and np.isfinite(c1) else np.nan
        deltas.append(d); labels.append(1.0 if np.isfinite(d) and d>3.0 else (0.0 if np.isfinite(d) else np.nan))
    return np.asarray(labels,dtype=float),np.asarray(deltas,dtype=float)


def _v15411_logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))


def _v15411_train_probability_model(X, targets, train_idx, val_idx, test_idx):
    """Probabilidad calibrada de evolución favorable (>+3 puntos de compuesto) a un horizonte explícito.
    XGBoost aprende en Train, Platt scaling se ajusta SOLO en Validation y Test se abre una sola vez.
    """
    y,delta=_v15411_favorable_labels(X,targets)
    valid=np.isfinite(y) & np.isfinite(pd.to_numeric(X.get('horizonte_dias'),errors='coerce').to_numpy(dtype=float))
    tr=np.intersect1d(train_idx,np.where(valid)[0]); va=np.intersect1d(val_idx,np.where(valid)[0]); te=np.intersect1d(test_idx,np.where(valid)[0])
    if len(tr)<40 or len(va)<8 or len(te)<8 or len(np.unique(y[tr]))<2 or len(np.unique(y[va]))<2:
        return None, {'estado':'No calibrable','motivo':'Se requieren >=40/8/8 casos válidos y ambas clases en Train y Validation.'}
    selected,dropped=_v1548_select_features(X.iloc[tr])
    if 'horizonte_dias' not in selected:
        selected=['horizonte_dias']+[c for c in selected if c!='horizonte_dias']
    pos=max(1,int(np.sum(y[tr]==1))); neg=max(1,int(np.sum(y[tr]==0))); spw=float(neg/pos)
    pipe=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('xgb',XGBClassifier(n_estimators=420,learning_rate=0.025,max_depth=3,min_child_weight=5.0,
            subsample=0.85,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=1.5,
            objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=1,
            random_state=15411,verbosity=0,scale_pos_weight=spw))
    ])
    pipe.fit(X.iloc[tr][selected],y[tr].astype(int))
    raw_val=pipe.predict_proba(X.iloc[va][selected])[:,1]
    calibrator=LogisticRegression(solver='lbfgs',random_state=15411)
    calibrator.fit(_v15411_logit(raw_val).reshape(-1,1),y[va].astype(int))
    raw_test=pipe.predict_proba(X.iloc[te][selected])[:,1]
    p_test=calibrator.predict_proba(_v15411_logit(raw_test).reshape(-1,1))[:,1]
    p_train=float(np.mean(y[tr]))
    baseline=np.full(len(te),p_train,dtype=float)
    brier=float(brier_score_loss(y[te],p_test)); brier_base=float(brier_score_loss(y[te],baseline))
    ll=float(log_loss(y[te],p_test,labels=[0,1])); ll_base=float(log_loss(y[te],baseline,labels=[0,1]))
    auc=float(roc_auc_score(y[te],p_test)) if len(np.unique(y[te]))==2 else np.nan
    horizons=pd.to_numeric(X.loc[tr,'horizonte_dias'],errors='coerce').dropna().to_numpy(dtype=float)
    support={'min':float(np.min(horizons)),'p10':float(np.percentile(horizons,10)),'median':float(np.median(horizons)),'p90':float(np.percentile(horizons,90)),'max':float(np.max(horizons))} if len(horizons) else {}
    horizon_variation_ok=bool(len(np.unique(np.round(horizons,3)))>=3 and support and (support['p90']-support['p10'])>=1.0)
    validated=bool(np.isfinite(brier) and brier < brier_base and horizon_variation_ok)
    spec={'pipeline':pipe,'calibrator':calibrator,'features':selected,'dropped_features':dropped,'validated':validated,
          'definition':'Favorable = compuesto fisiológico siguiente - compuesto actual > +3 puntos',
          'horizon_support_days':support,'horizon_variation_ok':horizon_variation_ok,
          'metrics':{'n_train':len(tr),'n_validation':len(va),'n_test':len(te),'prevalencia_train_pct':100*p_train,
                     'brier_test':brier,'brier_baseline_test':brier_base,'logloss_test':ll,'logloss_baseline_test':ll_base,'auc_test':auc,
                     'mejora_brier_vs_baseline_pct':100*(brier_base-brier)/brier_base if brier_base>1e-12 else np.nan,'horizon_variation_ok':horizon_variation_ok}}
    return spec,spec['metrics']


def predict_v15411_favorable_probability(history_df,patient_id,phase,horizon_days,bundle=None):
    bundle=bundle or load_v153_level2()
    spec=(bundle or {}).get('probability_model') if bundle else None
    if not spec or not spec.get('validated'):
        return {'available':False,'reason':'La capa probabilística aún no dispone de calibración fuera de muestra superior a la prevalencia basal o no existe variación temporal suficiente para aprender el efecto del horizonte.'}
    p1=predict_v15_continuous(history_df,patient_id,phase,horizon_days=float(horizon_days))
    if p1 is None: return {'available':False,'reason':'No hay historia suficiente para construir las variables del horizonte solicitado.'}
    row=_v153_feature_row(p1,phase,horizon_days=float(horizon_days)); Xp=pd.DataFrame([row])
    features=spec.get('features',[])
    raw=float(spec['pipeline'].predict_proba(Xp.reindex(columns=features))[:,1][0])
    prob=float(spec['calibrator'].predict_proba(_v15411_logit([raw]).reshape(-1,1))[:,1][0])
    sup=spec.get('horizon_support_days',{}); h=float(horizon_days)
    if sup and sup.get('p10') is not None:
        if h < sup['min'] or h > sup['max']: support='Extrapolación fuera del rango observado'
        elif h < sup['p10'] or h > sup['p90']: support='Soporte temporal limitado'
        else: support='Dentro del soporte temporal principal'
    else: support='Soporte temporal no estimable'
    return {'available':True,'probability':prob,'probability_pct':100*prob,'horizon_days':h,'support':support,
            'support_days':sup,'metrics':spec.get('metrics',{}),'definition':spec.get('definition','')}


def train_v153_level2(history_df, min_examples=V153_MIN_EXAMPLES_DEFAULT, save_path=None):
    """XGBoost Nivel 2 con Train 70% / Validation 15% / Test final 15%.

    El peso se selecciona exclusivamente en Validation. El Test final permanece
    intocable hasta una única evaluación final y es el único que decide si la
    salida supera al Nivel 1 en >=2%. Walk-forward se calcula solo en el 85%
    pre-test como diagnóstico adicional de estabilidad temporal.
    """
    if not SKLEARN_AVAILABLE:
        return None, 'scikit-learn no está disponible.'
    if not XGBOOST_AVAILABLE:
        return None, 'XGBoost no está disponible. Instala la dependencia xgboost.'
    X,targets,meta=_v153_training_examples(history_df)
    n=len(X)
    if n < int(min_examples):
        return None, f'Acumulando datos: {n}/{int(min_examples)} transiciones aprendibles.'
    order=np.argsort(pd.to_datetime(meta['Fecha_objetivo'],errors='coerce').fillna(pd.Timestamp.min).to_numpy())
    cut_train=max(1,min(n-2,int(np.floor(n*0.70))))
    cut_val=max(cut_train+1,min(n-1,int(np.floor(n*0.85))))
    train_idx=order[:cut_train]; val_idx=order[cut_train:cut_val]; test_idx=order[cut_val:]
    pretest_idx=order[:cut_val]
    models={}; metrics={}; active_targets=[]
    for c in V15_PREDICTIVE_TARGETS:
        y=targets[c]
        valid=np.isfinite(y)
        tr=np.intersect1d(train_idx,np.where(valid)[0]); va=np.intersect1d(val_idx,np.where(valid)[0]); te=np.intersect1d(test_idx,np.where(valid)[0])
        if len(tr)<40 or len(va)<8 or len(te)<8:
            continue
        base_col=f'{c}__pred_n1'
        base_all=pd.to_numeric(X[base_col],errors='coerce').to_numpy(dtype=float)
        tr=tr[np.isfinite(base_all[tr])]; va=va[np.isfinite(base_all[va])]; te=te[np.isfinite(base_all[te])]
        if len(tr)<40 or len(va)<8 or len(te)<8:
            continue
        residual=y-base_all
        # v15.4.12: selección de variables se aprende EXCLUSIVAMENTE con Train.
        selected_features,dropped_features=_v1548_select_features(X.iloc[tr])
        if not selected_features:
            continue
        Xsel=X[selected_features]
        pipe=Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('xgb',XGBRegressor(
                n_estimators=420,
                learning_rate=0.025,
                max_depth=3,
                min_child_weight=5.0,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.5,
                objective='reg:pseudohubererror',
                eval_metric='mae',
                tree_method='hist',
                n_jobs=1,
                random_state=153,
                verbosity=0,
            ))
        ])
        pipe.fit(Xsel.iloc[tr],residual[tr])

        # 15% VALIDATION: aquí y solo aquí se elige w.
        corr_val=pipe.predict(Xsel.iloc[va])
        val_base_mae=float(mean_absolute_error(y[va],base_all[va]))
        best_w=0.0; best_val_mae=val_base_mae
        for w in (0.20,0.35,0.50,0.65,0.80,1.0):
            cand=np.asarray([_v15_clip_target(c,v) for v in (base_all[va]+w*corr_val)],dtype=float)
            mae=float(mean_absolute_error(y[va],cand))
            if mae < best_val_mae:
                best_val_mae,best_w=mae,float(w)
        val_improvement=(val_base_mae-best_val_mae)/val_base_mae if val_base_mae>1e-9 else 0.0

        # Walk-forward adicional: SOLO 85% pre-test; nunca mira el test final.
        wf=_v153_walk_forward_diagnostic(X,y,base_all,pretest_idx,c,best_w,n_splits=4)

        # 15% TEST FINAL INTOCABLE: una sola evaluación con el peso ya congelado.
        corr_test=pipe.predict(Xsel.iloc[te])
        test_base_mae=float(mean_absolute_error(y[te],base_all[te]))
        test_pred=np.asarray([_v15_clip_target(c,v) for v in (base_all[te]+best_w*corr_test)],dtype=float)
        test_hybrid_mae=float(mean_absolute_error(y[te],test_pred))
        test_improvement=(test_base_mae-test_hybrid_mae)/test_base_mae if test_base_mae>1e-9 else 0.0
        active=bool(best_w>0 and test_improvement>=0.02)

        _imp_rows=_v1548_feature_importance(pipe,selected_features)
        models[c]={'pipeline':pipe,'features':selected_features,'weight':best_w if active else 0.0,'selected_weight_validation':best_w,'active':active,'feature_importance':_imp_rows,'dropped_features':dropped_features}
        metrics[c]={
            'n_train':len(tr),'n_validation':len(va),'n_test':len(te),
            'mae_validation_nivel1':val_base_mae,'mae_validation_hibrido':best_val_mae,
            'mejora_validation_pct':100*val_improvement,'peso_seleccionado_validation':best_w,
            'mae_test_nivel1':test_base_mae,'mae_test_hibrido':test_hybrid_mae,
            'mejora_test_pct':100*test_improvement,
            'peso_correccion':best_w if active else 0.0,
            'promocion_test_ge_2pct':active,
            'n_features_iniciales':len(V153_FEATURES),'n_features_usadas':len(selected_features),
            'n_descartadas_constantes':len(dropped_features.get('constant',[])),
            'n_descartadas_correlacion':len(dropped_features.get('correlated',[])),
            **wf,
            # Alias de compatibilidad: ahora representan el TEST final, no validation.
            'n_val':len(te),'mae_nivel1':test_base_mae,'mae_hibrido':test_hybrid_mae,'mejora_pct':100*test_improvement,
        }
        if active: active_targets.append(c)
    if not models:
        return None,'No hay suficientes valores válidos por índice para Train 70% / Validation 15% / Test 15% (mínimos 40/8/8).'
    probability_model,probability_metrics=_v15411_train_probability_model(X,targets,train_idx,val_idx,test_idx)

    bundle={'version':'15.4.11-xgb-horizon-prob','engine':'XGBoost','engine_class':'XGBRegressor',
            'xgboost_version':getattr(xgb,'__version__','unknown') if XGBOOST_AVAILABLE else None,
            'created_at':datetime.now(timezone.utc).isoformat(),
            'history_signature':_v153_history_signature(history_df),'n_examples':n,
            'features':V153_FEATURES,'primary_features':V1548_PRIMARY_FEATURES,'models':models,'metrics':metrics,'probability_model':probability_model,'probability_metrics':probability_metrics,
            'active_targets':active_targets,'min_examples':int(min_examples),
            'split_counts':{'train':len(train_idx),'validation':len(val_idx),'test':len(test_idx)},
            'validation_rule':'horizon_days explicit in every transition; smart feature selection on TRAIN only -> XGBoost regression 70/15/15 + untouched TEST; calibrated favorable-probability layer: XGBClassifier Train 70%, Platt calibration Validation 15%, untouched Test 15%; probability considered calibrated only if Test Brier beats prevalence baseline; walk-forward diagnostic restricted to pre-test 85%'}
    bundle['validation_score']=_v153_bundle_score(bundle)
    if save_path is not None:
        joblib.dump(bundle,Path(save_path))
    msg=(f'Candidato XGBoost evaluado con {n} transiciones: '
         f'Train {len(train_idx)} · Validation {len(val_idx)} · Test final {len(test_idx)}. '
         f'Corrige {len(active_targets)}/{len(V15_PREDICTIVE_TARGETS)} salidas que superan al Nivel 1 >=2% en TEST intocable. ' + (f'Capa probabilística calibrada disponible (Brier test {probability_metrics.get("brier_test",np.nan):.3f}).' if probability_model and probability_model.get('validated') else 'Capa probabilística aún no validada frente a prevalencia basal.'))
    return bundle,msg

def _v1552_probability_artifact_from_bundle(bundle):
    """Extrae únicamente una capa probabilística que YA haya superado su validación propia."""
    if not isinstance(bundle, dict):
        return None
    pm = bundle.get('probability_model') or {}
    if not isinstance(pm, dict) or not pm.get('validated'):
        return None
    return {
        'artifact_version': '15.5.2-probability-layer',
        'engine': 'XGBoost',
        'probability_model': pm,
        'probability_metrics': bundle.get('probability_metrics') or pm.get('metrics', {}) or {},
        'source_bundle_version': bundle.get('version'),
        'source_n_examples': int(bundle.get('n_examples', 0) or 0),
        'source_created_at': bundle.get('created_at'),
        'source_history_signature': bundle.get('history_signature'),
        'saved_at': datetime.now(timezone.utc).isoformat(),
    }


def _v1552_save_validated_probability_layer(bundle, reason='probability_validated'):
    """Persiste la última capa probabilística validada independientemente del regresor activo."""
    artifact = _v1552_probability_artifact_from_bundle(bundle)
    if not artifact:
        return False
    try:
        raw_bio = io.BytesIO()
        joblib.dump(artifact, raw_bio)
        raw = raw_bio.getvalue()
        V1552_PROBABILITY_PATH.write_bytes(raw)
        if _persistent_db_url_v1540():
            ok, _ = _cloud_put_bytes_v1540(
                CLOUD_PROBABILITY_KEY_V1552, raw,
                reason=str(reason or 'probability_validated'), force=True
            )
            if not ok:
                return False
        return True
    except Exception:
        return False


def _v1552_load_validated_probability_layer():
    """Carga la última probabilidad validada; la nube sólo se consulta si no existe copia local."""
    artifact = None
    if V1552_PROBABILITY_PATH.exists():
        try:
            artifact = joblib.load(V1552_PROBABILITY_PATH)
        except Exception:
            artifact = None
    if artifact is None and _persistent_db_url_v1540():
        remote = _cloud_get_object_v1540(CLOUD_PROBABILITY_KEY_V1552)
        if remote and remote.get('payload'):
            try:
                raw = bytes(remote['payload'])
                V1552_PROBABILITY_PATH.write_bytes(raw)
                artifact = joblib.load(io.BytesIO(raw))
            except Exception:
                artifact = None
    if not isinstance(artifact, dict):
        return None
    pm = artifact.get('probability_model') or {}
    if not isinstance(pm, dict) or not pm.get('validated'):
        return None
    return artifact


def _v1552_attach_last_validated_probability(bundle):
    """Mantiene independiente la promoción del regresor y la de la probabilidad.

    Un candidato regresivo puede sustituir al regresor activo, pero NO puede borrar una
    capa probabilística anterior que sí superó Brier TEST + diversidad temporal.
    """
    if not isinstance(bundle, dict):
        return bundle
    current_pm = bundle.get('probability_model') or {}
    if isinstance(current_pm, dict) and current_pm.get('validated'):
        # La capa del propio bundle es válida y pasa a ser la última capa validada.
        _v1552_save_validated_probability_layer(bundle, 'probability_validated_in_active_bundle')
        out = bundle
        out['probability_layer_source'] = {
            'mode': 'same_bundle',
            'source_n_examples': int(bundle.get('n_examples', 0) or 0),
            'source_created_at': bundle.get('created_at'),
        }
        return out
    artifact = _v1552_load_validated_probability_layer()
    if not artifact:
        return bundle
    out = copy.deepcopy(bundle)
    out['probability_model'] = artifact.get('probability_model')
    out['probability_metrics'] = artifact.get('probability_metrics') or {}
    out['probability_layer_source'] = {
        'mode': 'preserved_last_validated',
        'source_n_examples': int(artifact.get('source_n_examples', 0) or 0),
        'source_created_at': artifact.get('source_created_at'),
        'source_bundle_version': artifact.get('source_bundle_version'),
    }
    return out


def _v153_promote_candidate(candidate, active=None, tolerance=0.01):
    """Promueve solo si existe corrección validada y no empeora el modelo activo."""
    if not candidate or not candidate.get('active_targets'):
        return False,'Candidato rechazado: ningún índice mejoró al Nivel 1 en validación temporal.'
    cscore=_v153_bundle_score(candidate)
    if active is None:
        promote=True; reason='Primer modelo validado.'
    else:
        ascore=_v153_bundle_score(active)
        if not np.isfinite(ascore):
            promote=True; reason='El modelo activo no dispone de validación comparable.'
        else:
            promote=bool(np.isfinite(cscore) and cscore <= ascore*(1.0+tolerance))
            reason=(f'MAE agregado candidato {cscore:.3f} frente a activo {ascore:.3f}.')
    if not promote:
        return False,'Candidato no promocionado: '+reason
    # v15.5.2: la promoción regresiva y la probabilística son independientes.
    # Si el candidato no calibra la probabilidad, conserva la última capa que sí validó.
    if isinstance(active, dict) and bool((active.get('probability_model') or {}).get('validated')):
        _v1552_save_validated_probability_layer(active, 'preserved_before_regression_promotion')
    if bool((candidate.get('probability_model') or {}).get('validated')):
        _v1552_save_validated_probability_layer(candidate, 'new_probability_layer_promoted')
    candidate = _v1552_attach_last_validated_probability(candidate)
    if V153_MODEL_PATH.exists():
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        try: shutil.copy2(V153_MODEL_PATH,V153_ARCHIVE_DIR/f'motor_hibrido_{stamp}.joblib')
        except Exception: pass
    joblib.dump(candidate,V153_MODEL_PATH)
    _cloud_push_active_model_v1540('xgboost_promoted')
    _cloud_checkpoint_v1540('model_promotion_state')
    return True,'Candidato promocionado y activado. '+reason

def load_v153_level2():
    # v15.5.1: prioridad absoluta al bundle importado manualmente durante la sesión.
    pinned = _v1551_imported_model_state()
    pb = pinned.get('bundle')
    if isinstance(pb, dict) and pb.get('version') == '15.4.11-xgb-horizon-prob' and str(pb.get('engine','')).lower() == 'xgboost':
        return _v1552_attach_last_validated_probability(pb)
    _cloud_restore_model_once_v1540()
    if not SKLEARN_AVAILABLE or not XGBOOST_AVAILABLE or not V153_MODEL_PATH.exists():
        return None
    try:
        b=joblib.load(V153_MODEL_PATH)
        if not isinstance(b,dict):
            return None
        if b.get('version') != '15.4.11-xgb-horizon-prob' or str(b.get('engine','')).lower() != 'xgboost':
            return None
        return _v1552_attach_last_validated_probability(b)
    except Exception:
        return None


def init_v153_training_state():
    init_longitudinal_db()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS model_training_state (
            model_name TEXT PRIMARY KEY,
            active_version TEXT,
            trained_at TEXT,
            n_transitions_trained INTEGER NOT NULL DEFAULT 0,
            last_history_signature TEXT,
            last_attempt_signature TEXT,
            last_attempt_at TEXT,
            last_status TEXT,
            last_message TEXT,
            active_model_path TEXT,
            validation_score REAL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS model_training_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempted_at TEXT NOT NULL,
            history_signature TEXT,
            n_transitions INTEGER,
            n_new_transitions INTEGER,
            status TEXT,
            promoted INTEGER,
            validation_score REAL,
            message TEXT
        )""")
        con.execute("INSERT OR IGNORE INTO model_training_state(model_name) VALUES ('v153_hybrid')")
        con.commit()
    _cloud_checkpoint_v1540('training_schema_state', min_interval_s=30)


def get_v153_training_state():
    init_v153_training_state()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        row=con.execute("SELECT * FROM model_training_state WHERE model_name='v153_hybrid'").fetchone()
        cols=[d[0] for d in con.execute("SELECT * FROM model_training_state LIMIT 0").description]
    return dict(zip(cols,row)) if row else {}


def _save_v153_training_state(**updates):
    init_v153_training_state()
    if not updates: return
    allowed={'active_version','trained_at','n_transitions_trained','last_history_signature','last_attempt_signature',
             'last_attempt_at','last_status','last_message','active_model_path','validation_score'}
    updates={k:v for k,v in updates.items() if k in allowed}
    if not updates: return
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        sets=', '.join(f'{k}=?' for k in updates)
        con.execute(f"UPDATE model_training_state SET {sets} WHERE model_name='v153_hybrid'",list(updates.values()))
        con.commit()
    _cloud_checkpoint_v1540('training_state_updated', min_interval_s=5)


def _log_v153_run(signature,n_total,n_new,status,promoted,score,message):
    init_v153_training_state()
    with sqlite3.connect(LONGITUDINAL_DB_PATH) as con:
        con.execute("""INSERT INTO model_training_runs(
            attempted_at,history_signature,n_transitions,n_new_transitions,status,promoted,validation_score,message
        ) VALUES (?,?,?,?,?,?,?,?)""",(
            datetime.now(timezone.utc).isoformat(),signature,int(n_total),int(n_new),str(status),int(bool(promoted)),
            float(score) if np.isfinite(score) else None,str(message)
        ))
        con.commit()
    _cloud_checkpoint_v1540('training_run_logged')


def maybe_auto_train_v153(history_df,min_examples,new_threshold,force=False):
    """Entrenamiento condicionado durante una ejecución de Streamlit; no crea un daemon."""
    X,_,_=_v153_training_examples(history_df)
    n_total=len(X); sig=_v153_history_signature(history_df)
    state=get_v153_training_state()
    active_xgb=load_v153_level2()
    # Migración v15.4.12: un modelo previo sin smart feature selection no bloquea el primer XGBoost v15.4.12.
    n_prev=int(state.get('n_transitions_trained') or 0) if active_xgb is not None else 0
    n_new=max(0,n_total-n_prev)
    enough_total=n_total>=int(min_examples)
    enough_new=n_new>=int(new_threshold)
    same_attempt=(state.get('last_attempt_signature')==sig) if active_xgb is not None else False
    should=bool(force or (enough_total and ((active_xgb is None) or enough_new) and not same_attempt))
    result={'attempted':False,'promoted':False,'n_total':n_total,'n_new':n_new,'signature':sig,
            'message':'','candidate':None,'active':active_xgb}
    if not should:
        if not enough_total: result['message']=f'Acumulando: {n_total}/{int(min_examples)} transiciones totales.'
        elif not enough_new: result['message']=f'Nuevas transiciones: {n_new}/{int(new_threshold)} para el próximo entrenamiento.'
        else: result['message']='Este conjunto ya fue evaluado; se espera una transición nueva.'
        return result
    result['attempted']=True
    _save_v153_training_state(last_attempt_signature=sig,last_attempt_at=datetime.now(timezone.utc).isoformat(),
                              last_status='entrenando',last_message='Entrenando candidato')
    try:
        candidate,msg=train_v153_level2(history_df,int(min_examples),save_path=V153_CANDIDATE_PATH)
        if not candidate:
            _save_v153_training_state(last_status='rechazado',last_message=msg)
            _log_v153_run(sig,n_total,n_new,'rechazado',False,np.nan,msg)
            result['message']=msg; return result
        active=result['active']
        promoted,pmsg=_v153_promote_candidate(candidate,active)
        score=_v153_bundle_score(candidate)
        result.update(candidate=candidate,promoted=promoted,active=(candidate if promoted else active),message=msg+' '+pmsg)
        if promoted:
            _save_v153_training_state(active_version=candidate.get('version'),trained_at=candidate.get('created_at'),
                n_transitions_trained=n_total,last_history_signature=sig,last_status='promocionado',last_message=result['message'],
                active_model_path=str(V153_MODEL_PATH),validation_score=score)
        else:
            _save_v153_training_state(last_status='no_promocionado',last_message=result['message'])
        _log_v153_run(sig,n_total,n_new,'promocionado' if promoted else 'no_promocionado',promoted,score,result['message'])
    except Exception as exc:
        msg=f'Entrenamiento fallido: {type(exc).__name__}: {exc}'
        _save_v153_training_state(last_status='error',last_message=msg)
        _log_v153_run(sig,n_total,n_new,'error',False,np.nan,msg)
        result['message']=msg
    return result


def import_v153_model_bytes(raw_bytes):
    # v15.5.1: importación autoritativa y verificable del bundle XGBoost calibrado.
    raw_bytes = bytes(raw_bytes or b'')
    bundle=joblib.load(io.BytesIO(raw_bytes))
    if (not isinstance(bundle,dict) or bundle.get('version') != '15.4.11-xgb-horizon-prob'
            or str(bundle.get('engine','')).lower() != 'xgboost' or 'models' not in bundle):
        raise ValueError('Modelo incompatible: el Nivel 2 acepta únicamente bundles XGBoost con horizonte temporal explícito, probabilidad calibrada y Test final intocable.')

    # Conservar exactamente los bytes validados recibidos; evita una reserialización innecesaria.
    V153_MODEL_PATH.write_bytes(raw_bytes)
    imported_sha = hashlib.sha256(raw_bytes).hexdigest()
    pinned = _v1551_imported_model_state()
    pinned.clear()
    pinned.update({
        'bundle': bundle,
        'sha256': imported_sha,
        'imported_at': datetime.now(timezone.utc).isoformat(),
        'n_examples': int(bundle.get('n_examples',0)),
        'probability_validated': bool((bundle.get('probability_model') or {}).get('validated')),
    })
    # Bloquear cualquier restore cloud posterior en esta sesión.
    cloud_state = _cloud_state_v1540()
    cloud_state['_model_restore_checked'] = True
    cloud_state['_model_import_sha'] = imported_sha

    # v15.5.2: si la importación contiene una probabilidad realmente validada,
    # conservarla además como capa independiente para que futuros regresores no la borren.
    if bool((bundle.get('probability_model') or {}).get('validated')):
        if not _v1552_save_validated_probability_layer(bundle, 'validated_probability_imported_v1552'):
            raise RuntimeError('El modelo se importó, pero no pudo persistirse su capa probabilística calibrada independiente.')

    # El modelo importado sustituye de forma inmediata el objeto activo en Supabase y se verifica por SHA.
    if _persistent_db_url_v1540():
        if not _cloud_push_active_model_v1540('validated_model_imported_v1551', payload=raw_bytes):
            raise RuntimeError('El modelo se importó localmente, pero no pudo confirmarse que Supabase conservara exactamente el mismo SHA-256. Revisa la conexión y vuelve a importar.')

    _save_v153_training_state(active_version=bundle.get('version'),trained_at=bundle.get('created_at'),
        n_transitions_trained=int(bundle.get('n_examples',0)),last_history_signature=bundle.get('history_signature'),
        last_status='importado_calibrado_v1551',
        last_message=('Modelo XGBoost importado manualmente y fijado como autoritativo. '
                      f'Probabilidad calibrada={bool((bundle.get("probability_model") or {}).get("validated"))}.'),
        active_model_path=str(V153_MODEL_PATH), validation_score=_v153_bundle_score(bundle))
    return bundle


def _refresh_hvg_descriptor_from_table(topo_obj):
    if not topo_obj or topo_obj.get('tabla') is None or topo_obj['tabla'].empty: return topo_obj
    tab=topo_obj['tabla'].copy(); tab['Z_previsto']=pd.to_numeric(tab['Cambio_previsto'],errors='coerce').abs()/pd.to_numeric(tab['Escala_personal_cambio'],errors='coerce').replace(0,np.nan); vals=[]; weights=[]
    for _,r in tab.iterrows():
        z=pd.to_numeric(r.get('Z_previsto'),errors='coerce'); w=pd.to_numeric(r.get('Peso_topológico'),errors='coerce')
        if np.isfinite(z) and np.isfinite(w) and w>0: vals.append(min(float(z),4)); weights.append(float(w))
    score=float(np.sqrt(np.average(np.asarray(vals)**2,weights=np.asarray(weights)))) if vals else np.nan; out=copy.deepcopy(topo_obj); out['tabla']=tab; out['score_previsto']=score; out['descriptor_previsto']=_hvg_reorg_label(score); out['direccion_prevista']=_hvg_direction_text(tab,'Cambio_previsto','Z_previsto'); return out


def predict_v153_hybrid(history_df,patient_id,phase,bundle=None,horizon_days=None):
    p1=predict_v15_continuous(history_df,patient_id,phase,horizon_days=horizon_days)
    if p1 is None: return None
    bundle=bundle or load_v153_level2(); p=copy.deepcopy(p1); p['nivel2_activo']=False; p['nivel2_indices']=[]; p['nivel2_topologia']=[]; p['nivel2_version']=None; p['modo_prediccion_nivel1']=p1.get('modo_prediccion','')
    if not bundle: p['modo_prediccion']='Nivel 1 · '+p1.get('modo_prediccion',''); return p
    X=pd.DataFrame([_v153_feature_row(p1,phase,horizon_days=horizon_days)]).reindex(columns=bundle.get('features',V153_FEATURES)); tab=p['tabla'].copy(); cp=[]; ct=[]
    for i,r in tab.iterrows():
        c=r['Índice']; spec=bundle.get('models',{}).get(c)
        if not spec or not spec.get('active') or spec.get('weight',0)<=0: continue
        try:
            _xf=spec.get('features') or bundle.get('features',V153_FEATURES); residual=float(spec['pipeline'].predict(X.reindex(columns=_xf))[0]); w=float(spec['weight']); old=float(r['Predicción_t+1']); new=_v15_clip_target(c,old+w*residual); tab.at[i,'Predicción_Nivel1']=old; tab.at[i,'Corrección_ML']=new-old; tab.at[i,'Peso_ML']=w; tab.at[i,'Predicción_t+1']=new; tab.at[i,'Cambio_previsto']=new-float(r['Actual']); tab.at[i,'Dirección']='Ascenso' if new-float(r['Actual'])>1 else ('Descenso' if new-float(r['Actual'])<-1 else 'Estable'); cp.append(c)
        except Exception: pass
    topo=copy.deepcopy(p.get('topologia_hvg'))
    if topo and topo.get('tabla') is not None and not topo['tabla'].empty:
        tt=topo['tabla'].copy()
        for i,r in tt.iterrows():
            c=r['Métrica']; spec=bundle.get('models',{}).get(c)
            if not spec or not spec.get('active') or spec.get('weight',0)<=0: continue
            try:
                _xf=spec.get('features') or bundle.get('features',V153_FEATURES); residual=float(spec['pipeline'].predict(X.reindex(columns=_xf))[0]); w=float(spec['weight']); old=float(r['Predicción_t+1']); new=_v15_clip_target(c,old+w*residual); tt.at[i,'Predicción_Nivel1']=old; tt.at[i,'Corrección_ML']=new-old; tt.at[i,'Peso_ML']=w; tt.at[i,'Predicción_t+1']=new; tt.at[i,'Cambio_previsto']=new-float(r['Actual']); ct.append(c)
            except Exception: pass
        topo['tabla']=tt; p['topologia_hvg']=_refresh_hvg_descriptor_from_table(topo)
    if cp:
        m={str(r['Índice']):float(r['Predicción_t+1']) for _,r in tab.iterrows()}; parts=[]
        for key,w,sign in [('IDX_Vagal',1,1),('IDX_Amplitud',1,1),('IDX_Complejidad',1,1),('IDX_Rigidez',1,-1),('IDX_Adaptabilidad',1.2,1)]:
            if key in m: parts.append((sign*m[key],w))
        c1=sum(v*w for v,w in parts)/sum(w for _,w in parts) if parts else p['compuesto_predicho']; p['compuesto_predicho_nivel1']=p['compuesto_predicho']; p['compuesto_predicho']=float(c1); p['delta_compuesto']=float(c1-p['compuesto_actual']); p['estado_previsto']='Favorable' if p['delta_compuesto']>3 else ('Desfavorable' if p['delta_compuesto']<-3 else 'Estable')
    if cp or ct: p['nivel2_activo']=True; p['nivel2_indices']=cp; p['nivel2_topologia']=ct; p['nivel2_version']=bundle.get('version'); p['modo_prediccion']=f'Híbrido: Nivel 1 + XGBoost ({len(cp)} índices; {len(ct)} métricas HVG corregidas)'
    else: p['modo_prediccion']='Nivel 1 · '+p1.get('modo_prediccion','')+'; Nivel 2 sin mejora validada para esta salida'
    p['probabilidad_favorable']=predict_v15411_favorable_probability(history_df,patient_id,phase,(horizon_days if horizon_days is not None else p1.get('paso_previsto',7)),bundle=bundle)
    p['horizonte_dias']=float(horizon_days if horizon_days is not None else p1.get('paso_previsto',np.nan))
    p['tabla']=tab; return p

def v15_current_evolution_figure(history_df, patient_id):
    """Muestra siempre la evolución disponible del paciente, incluso con una sola fase/registro."""
    if history_df is None or history_df.empty:
        return None, pd.DataFrame()
    df=history_df[history_df['Paciente_ID'].astype(str)==str(patient_id)].copy()
    if df.empty:
        return None, df
    df['_dt']=pd.to_datetime(df.get('Fecha_hora'),errors='coerce')
    phase_rank={p:i for i,p in enumerate(PHASES)}
    df['_phase_rank']=df['Fase'].map(phase_rank).fillna(999)
    df=df.sort_values(['_dt','Registro','_phase_rank','saved_at'],na_position='last').reset_index(drop=True)
    df['Punto']=df.apply(lambda r: _record_axis_label(r['Registro'], multiline=True)+'<br>'+str(r['Fase']), axis=1)
    fig=go.Figure()
    for c in V15_INDEX_FEATURES:
        if c in df.columns and pd.to_numeric(df[c],errors='coerce').notna().any():
            fig.add_trace(go.Scatter(x=df['Punto'],y=pd.to_numeric(df[c],errors='coerce'),mode='lines+markers',name=c.replace('IDX_','')))
    fig.update_layout(height=520,title='Evolución fisiológica actualmente disponible',yaxis=dict(range=[0,100],title='Índice 0-100'),xaxis_title='Fecha · fase',legend_title='Índice')
    return fig,df


def v15_prediction_figure(prediction):
    tab=prediction['tabla']
    fig=go.Figure()
    fig.add_trace(go.Bar(x=tab['Índice'],y=tab['Actual'],name='Actual'))
    fig.add_trace(go.Bar(x=tab['Índice'],y=tab['Predicción_t+1'],name='Predicción t+1',
                         error_y=dict(type='data',array=1.96*tab['Incertidumbre_1SD'],visible=True)))
    fig.update_layout(barmode='group',height=480,yaxis=dict(range=[0,110],title='Índice 0-100'),
                      title='Estado fisiológico actual y siguiente estado previsto')
    return fig


def predict_v15_all_phases(history_df, patient_id, horizon_days=None):
    """Calcula simultáneamente la predicción de todas las fases disponibles del paciente."""
    if history_df is None or history_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    phases = sorted(
        history_df.loc[history_df['Paciente_ID'].astype(str) == str(patient_id), 'Fase']
        .dropna().astype(str).unique(),
        key=lambda x: PHASES.index(x) if x in PHASES else 999
    )
    predictions = {}
    summary_rows = []
    detail_rows = []
    for phase in phases:
        pred = predict_v153_hybrid(history_df, patient_id, phase, horizon_days=horizon_days)
        if pred is None:
            continue
        predictions[phase] = pred
        summary_rows.append({
            'Fase': phase,
            'Estado_previsto': pred['estado_previsto'],
            'Registros': pred['n_historial'],
            'Modo': pred['modo_prediccion'],
            'Compuesto_actual': pred['compuesto_actual'],
            'Compuesto_predicho': pred['compuesto_predicho'],
            'Delta_compuesto': pred['delta_compuesto'],
            'Confianza_pct': pred['confianza'],
            'Novedad': pred['anomalia_label'],
            'Novedad_score': pred['anomalia_score'],
            'Registro_actual': pred['registro_actual'],
            'Horizonte_dias': pred.get('horizonte_dias',np.nan),
            'Prob_favorable_pct': (pred.get('probabilidad_favorable') or {}).get('probability_pct',np.nan),
            'Soporte_probabilidad': (pred.get('probabilidad_favorable') or {}).get('support','No disponible'),
            'Reorganizacion_HVG': (pred.get('topologia_hvg') or {}).get('descriptor_actual','No estimable'),
            'Score_reorganizacion_HVG': (pred.get('topologia_hvg') or {}).get('score_actual',np.nan),
            'Direccion_HVG': (pred.get('topologia_hvg') or {}).get('direccion_actual',''),
        })
        t = pred['tabla'].copy()
        t.insert(0, 'Fase', phase)
        detail_rows.append(t)
    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return predictions, summary, details


def v15_multiphase_summary_figure(summary_df):
    fig = go.Figure()
    if summary_df is None or summary_df.empty:
        fig.update_layout(title='Sin predicciones multifase disponibles')
        return fig
    fig.add_trace(go.Bar(
        x=summary_df['Fase'], y=summary_df['Compuesto_actual'], name='Compuesto actual',
        hovertemplate='Fase: %{x}<br>Actual: %{y:.1f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=summary_df['Fase'], y=summary_df['Compuesto_predicho'], name='Compuesto previsto',
        hovertemplate='Fase: %{x}<br>Previsto: %{y:.1f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=summary_df['Fase'], y=summary_df['Confianza_pct'], mode='lines+markers',
        name='Confianza %', yaxis='y2',
        hovertemplate='Fase: %{x}<br>Confianza: %{y:.0f}%<extra></extra>'
    ))
    fig.update_layout(
        barmode='group', height=520,
        title='Predicción simultánea del siguiente estado por fase',
        yaxis=dict(title='Compuesto fisiológico 0-100', range=[0, 105]),
        yaxis2=dict(title='Confianza metodológica (%)', overlaying='y', side='right', range=[0, 100]),
        xaxis_title='Fase', legend_title='Resultado'
    )
    return fig


def v15_multiphase_indices_figure(details_df):
    fig = go.Figure()
    if details_df is None or details_df.empty:
        fig.update_layout(title='Sin predicciones por índice')
        return fig
    for phase in details_df['Fase'].drop_duplicates():
        d = details_df[details_df['Fase'] == phase]
        fig.add_trace(go.Scatter(
            x=d['Índice'], y=d['Predicción_t+1'], mode='lines+markers', name=str(phase),
            error_y=dict(type='data', array=1.96*d['Incertidumbre_1SD'], visible=True),
            hovertemplate=f'Fase: {phase}<br>Índice: %{{x}}<br>Predicción: %{{y:.1f}}<extra></extra>'
        ))
    fig.update_layout(
        height=560, title='Predicción t+1 de todos los índices y fases',
        yaxis=dict(title='Índice previsto 0-100', range=[0, 110]),
        xaxis_title='Índice fisiológico', legend_title='Fase'
    )
    return fig



def _attractor_delay(rr, max_lag=120):
    """Retardo práctico por primera caída de autocorrelación bajo 1/e."""
    x=np.asarray(rr,dtype=float); x=x[np.isfinite(x)]
    if len(x)<20: return 1
    x=x-np.mean(x); den=float(np.dot(x,x))
    if den<=0: return 1
    lim=min(max_lag,max(2,len(x)//5))
    for lag in range(1,lim+1):
        ac=float(np.dot(x[:-lag],x[lag:])/den)
        if ac <= 1/np.e: return lag
    return max(1,min(10,lim))

def _delay_embedding(rr,tau=1,m=3):
    x=np.asarray(rr,dtype=float); x=x[np.isfinite(x)]
    n=len(x)-(m-1)*tau
    if n<=3: return np.empty((0,m))
    return np.column_stack([x[(m-1-j)*tau:(m-1-j)*tau+n] for j in range(m)])

def _attractor_descriptor(rr, emb):
    """Descriptor prudente: orientación exploratoria, no diagnóstico de caos."""
    if len(emb)<30: return "Indeterminado", "Evidencia insuficiente: registro demasiado corto."
    spread=np.std(emb,axis=0); cv=float(np.mean(spread)/(abs(np.mean(rr))+1e-9))
    # periodicidad aproximada mediante autocorrelación secundaria
    x=np.asarray(rr,dtype=float); x=x[np.isfinite(x)]; x=x-np.mean(x)
    ac=np.correlate(x,x,mode='full')[len(x)-1:]; ac=ac/(ac[0]+1e-12)
    sec=float(np.nanmax(ac[2:min(len(ac),max(5,len(x)//3))])) if len(ac)>5 else 0
    if cv < 0.008: return "Compatible con punto fijo / dinámica muy concentrada", "Variabilidad geométrica muy baja."
    if sec > 0.72: return "Compatible con ciclo límite / dinámica periódica", f"Recurrencia temporal marcada (AC secundaria≈{sec:.2f})."
    if sec > 0.45: return "Compatible con dinámica cuasiperiódica o recurrente", f"Recurrencia temporal intermedia (AC secundaria≈{sec:.2f})."
    return "Dinámica compleja / estocástica; atractor extraño no demostrado", "La geometría es dispersa sin periodicidad dominante. Para afirmar caos determinista se requieren pruebas adicionales y datos suficientes."



# ============================================================
# v15.5.5 · CONTROL AUTONÓMICO MULTIVARIADO · REFERENCIA LONGITUDINAL PROGRESIVA V/S/L/C/R/Q
# Clasificación heurística/explicable de formas operativas del SNA.
# - No interpreta LF o LF/HF como medidas puras de actividad simpática.
# - Prioriza normalización longitudinal intra-paciente.
# - Separa: Vagal (V), Alerta/barorreflejo (S), Regulación lenta (L),
#   Complejidad/movilidad (C), Reserva/amplitud (R) y Calidad/contexto (Q).
# - Evita contar como votos independientes todas las variantes Welch/LS/AR/Wavelet:
#   se agregan como consensos por banda.
# - MSE/MDE se resumen por escalas para evitar 40 votos redundantes.
# ============================================================
AUTONOMIC_FORMS = {
    'Desacoplamiento autonómico': {
        'coste': 'Alto / potencialmente ineficiente',
        'interpretacion': 'Patrón compatible con pérdida de coherencia entre escalas rápidas, lentas y topológicas. Puede existir mucha movilidad o complejidad sin una reserva/amplitud suficiente o sin organización de red concordante.',
        'clinica': 'Priorizar dosificación conservadora, control de síntomas y estabilidad hemodinámica. Valorar pausas, respiración no forzada, transiciones posturales progresivas y reevaluación tras la carga. No interpretar este patrón como diagnóstico aislado.'
    },
    'Coactivación con predominio simpático': {
        'coste': 'Moderado-alto',
        'interpretacion': 'Coexisten modulación vagal conservada/elevada y mecanismos rápidos de alerta/barorreflejo, con predominio de la movilización rápida. Puede reflejar alta capacidad de respuesta con mayor coste alostático.',
        'clinica': 'Puede tolerar trabajo activo si la respuesta clínica es estable, pero conviene vigilar recuperación, frecuencia cardiaca, percepción de esfuerzo y retorno del componente vagal después de la carga.'
    },
    'Coactivación con predominio parasimpático': {
        'coste': 'Moderado / eficiente si es transitorio',
        'interpretacion': 'Coexistencia de mecanismos de demanda con un freno vagal dominante. Es compatible con recuperación activa o control vagal eficaz durante una situación de demanda.',
        'clinica': 'Puede ser compatible con progresión terapéutica si la respuesta es estable. Comparar con la evolución post-esfuerzo y confirmar que no exista bradicardia sintomática, fatiga desproporcionada o discordancia clínica.'
    },
    'Reciprocidad · predominio vagal puro': {
        'coste': 'Bajo',
        'interpretacion': 'Predominio de restauración: aumento/conservación vagal acompañado de reducción de mecanismos rápidos de alerta y baja rigidez dinámica. Representa la forma operativa de menor coste alostático del algoritmo.',
        'clinica': 'Contexto generalmente favorable para recuperación, aprendizaje motor y trabajo de precisión. La decisión de aumentar carga debe seguir dependiendo de síntomas, objetivos y tolerancia funcional.'
    },
    'Retirada vagal · predominio simpático': {
        'coste': 'Alto si persiste',
        'interpretacion': 'Descenso del freno vagal con predominio de mecanismos de alerta/rigidez. Puede ser una respuesta adaptativa aguda; si se mantiene longitudinalmente sugiere mayor coste de sostener la demanda.',
        'clinica': 'Ajustar intensidad y densidad del trabajo, comprobar recuperación entre bloques y evitar inferir estrés psicológico a partir de HRV. Repetir medición en condiciones comparables.'
    },
    'Retirada simpática · desconexión del drive / atonía': {
        'coste': 'Variable; posible baja disponibilidad de drive',
        'interpretacion': 'Reducción marcada de componentes LF/VLF y del vector de alerta sin hiperactivación vagal compensatoria. Se interpreta como menor drive vasomotor/alerta, no como una medida directa del tono simpático neural.',
        'clinica': 'Ser prudente con cambios posturales y cargas que requieran respuesta cardiovascular rápida; correlacionar con presión arterial, síntomas, medicación y estado general antes de atribuir significado clínico.'
    },
}

# Matriz visible y auditable. Los pesos son relativos dentro de cada vector, no coeficientes clínicos universales.
AUTONOMIC_MATRIX_V1554 = [
    # Vagal V
    {'Vector':'V · Vagal','Variable':'RMSSD','Peso':1.00,'Rol':'Primaria','Dirección/uso':'↑ respecto al basal apoya modulación vagal rápida'},
    {'Vector':'V · Vagal','Variable':'SD1','Peso':1.00,'Rol':'Primaria','Dirección/uso':'↑ respecto al basal apoya modulación vagal rápida'},
    {'Vector':'V · Vagal','Variable':'HF_DOM_PCT','Peso':0.90,'Rol':'Primaria','Dirección/uso':'Mayor dominancia HF apoya componente respiratorio/vagal'},
    {'Vector':'V · Vagal','Variable':'pNN50','Peso':0.55,'Rol':'Secundaria','Dirección/uso':'Apoyo vagal; menor peso por redundancia con RMSSD/SD1'},
    {'Vector':'V · Vagal','Variable':'HF consenso (HF/HF_LS/HF_AR/HF_WAV_MEAN)','Peso':0.70,'Rol':'Consenso','Dirección/uso':'Mediana de Z entre métodos; evita cuatro votos separados'},
    {'Vector':'V · Vagal','Variable':'IDX_Vagal','Peso':0.40,'Rol':'Índice integrado','Dirección/uso':'Confirmación; peso reducido para evitar doble conteo'},
    # Alerta S
    {'Vector':'S · Alerta/barorreflejo','Variable':'DFA_alpha1','Peso':0.75,'Rol':'Primaria','Dirección/uso':'>1.0–1.15 apoya organización de alerta; <0.5 se deriva a desacoplamiento; no “más=mejor”'},
    {'Vector':'S · Alerta/barorreflejo','Variable':'LF_DOM_PCT','Peso':1.00,'Rol':'Primaria','Dirección/uso':'Dominancia LF como organización rápida/barorrefleja, no simpático puro'},
    {'Vector':'S · Alerta/barorreflejo','Variable':'MeanHR','Peso':0.55,'Rol':'Secundaria','Dirección/uso':'↑ intra-paciente apoya movilización rápida si converge'},
    {'Vector':'S · Alerta/barorreflejo','Variable':'LF_HF','Peso':0.45,'Rol':'Contextual','Dirección/uso':'Sólo contextual; nunca se interpreta aisladamente como balance simpático-vagal'},
    {'Vector':'S · Alerta/barorreflejo','Variable':'LF consenso (LF/LF_LS/LF_AR/LF_WAV_MEAN)','Peso':0.55,'Rol':'Consenso','Dirección/uso':'Convergencia entre estimadores LF'},
    {'Vector':'S · Alerta/barorreflejo','Variable':'IDX_Rigidez','Peso':0.50,'Rol':'Índice integrado','Dirección/uso':'↑ apoya mantenimiento/rigidez; peso reducido por doble conteo'},
    # Regulación lenta L
    {'Vector':'L · Regulación lenta','Variable':'VLF_DOM_PCT','Peso':1.00,'Rol':'Primaria','Dirección/uso':'Dominancia lenta relativa'},
    {'Vector':'L · Regulación lenta','Variable':'VLF consenso (VLF/VLF_LS/VLF_AR/VLF_WAV_MEAN)','Peso':0.75,'Rol':'Consenso','Dirección/uso':'Convergencia de potencia VLF entre métodos'},
    {'Vector':'L · Regulación lenta','Variable':'DFA_alpha2','Peso':0.50,'Rol':'Secundaria','Dirección/uso':'Organización de correlaciones largas; se interpreta longitudinalmente'},
    {'Vector':'L · Regulación lenta','Variable':'Hurst','Peso':0.35,'Rol':'Secundaria','Dirección/uso':'Persistencia de largo alcance; no se trata como “más saludable”'},
    {'Vector':'L · Regulación lenta','Variable':'IDX_Regulacion_Lenta','Peso':0.45,'Rol':'Índice integrado','Dirección/uso':'Confirmación; peso reducido para evitar doble conteo'},
    # Complejidad C
    {'Vector':'C · Complejidad/movilidad','Variable':'SampEn','Peso':0.85,'Rol':'Primaria','Dirección/uso':'Complejidad irregular; se interpreta respecto al basal, no como “más=mejor”'},
    {'Vector':'C · Complejidad/movilidad','Variable':'DispEn','Peso':0.65,'Rol':'Primaria','Dirección/uso':'Diversidad de patrones'},
    {'Vector':'C · Complejidad/movilidad','Variable':'D2','Peso':0.60,'Rol':'Primaria','Dirección/uso':'Dimensionalidad dinámica'},
    {'Vector':'C · Complejidad/movilidad','Variable':'Lyapunov_LLE','Peso':0.45,'Rol':'Secundaria','Dirección/uso':'Sensibilidad local; extremos se interpretan con cautela'},
    {'Vector':'C · Complejidad/movilidad','Variable':'WAV_ENTROPY_GLOBAL','Peso':0.75,'Rol':'Primaria','Dirección/uso':'Distribución tiempo-frecuencia; alta + reserva baja puede indicar desacoplamiento'},
    {'Vector':'C · Complejidad/movilidad','Variable':'WAV_TRANSITIONS_PER_MIN','Peso':0.60,'Rol':'Primaria','Dirección/uso':'Movilidad entre regímenes; excesiva sin coherencia puede ser ineficiente'},
    {'Vector':'C · Complejidad/movilidad','Variable':'HVG_graph_score_small_world','Peso':0.75,'Rol':'Primaria','Dirección/uso':'Organización local-global de red'},
    {'Vector':'C · Complejidad/movilidad','Variable':'HVG_clustering','Peso':0.45,'Rol':'Secundaria','Dirección/uso':'Organización local de red'},
    {'Vector':'C · Complejidad/movilidad','Variable':'MSE resumen 1–5/6–10/11–20','Peso':0.45,'Rol':'Resumen multiescala','Dirección/uso':'Tres bloques agregados; evita 20 votos correlacionados'},
    {'Vector':'C · Complejidad/movilidad','Variable':'MDE resumen 1–5/6–10/11–20','Peso':0.45,'Rol':'Resumen multiescala','Dirección/uso':'Tres bloques agregados; evita 20 votos correlacionados'},
    {'Vector':'C · Complejidad/movilidad','Variable':'RQA (DET y Lmax/N)','Peso':0.40,'Rol':'Secundaria','Dirección/uso':'Recurrencia/rigidez; se usa de forma inversa dentro de C y directa como evidencia de rigidez'},
    {'Vector':'C · Complejidad/movilidad','Variable':'Tipo de atractor','Peso':'contexto','Rol':'Estructural','Dirección/uso':'Ciclo límite/punto fijo/complejo sólo refuerzan una regla si convergen con el resto'},
    {'Vector':'C · Complejidad/movilidad','Variable':'IDX_Complejidad','Peso':0.35,'Rol':'Índice integrado','Dirección/uso':'Confirmación con peso reducido'},
    # Reserva R
    {'Vector':'R · Reserva/amplitud','Variable':'SDNN','Peso':1.00,'Rol':'Primaria','Dirección/uso':'Reserva global de variabilidad'},
    {'Vector':'R · Reserva/amplitud','Variable':'SD2','Peso':0.90,'Rol':'Primaria','Dirección/uso':'Variabilidad de largo eje Poincaré'},
    {'Vector':'R · Reserva/amplitud','Variable':'TOTAL consenso (TOTAL/TOTAL_LS/TOTAL_AR)','Peso':0.80,'Rol':'Consenso','Dirección/uso':'Potencia total agregada entre métodos'},
    {'Vector':'R · Reserva/amplitud','Variable':'IDX_Amplitud','Peso':0.45,'Rol':'Índice integrado','Dirección/uso':'Confirmación con peso reducido'},
    # Quality Q
    {'Vector':'Q · Calidad/contexto','Variable':'N_RRi','Peso':'control','Rol':'Calidad','Dirección/uso':'Reduce confianza si el registro es corto'},
    {'Vector':'Q · Calidad/contexto','Variable':'Duration_s','Peso':'control','Rol':'Calidad','Dirección/uso':'Reduce confianza si la ventana temporal es insuficiente'},
    {'Vector':'Q · Calidad/contexto','Variable':'Artefactos_pct','Peso':'control','Rol':'Calidad','Dirección/uso':'Reduce confianza al aumentar artefactos'},
    {'Vector':'Q · Calidad/contexto','Variable':'IDX_Confianza','Peso':'control','Rol':'Calidad','Dirección/uso':'Modula confianza si está disponible; no decide el estado'},
]


def _autonomic_num(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _autonomic_transform_series(s, mode='linear'):
    x=pd.to_numeric(pd.Series(s),errors='coerce').astype(float)
    if mode=='log':
        x=x.where(x>=0)
        return np.log1p(x)
    return x


def _autonomic_single_reference_scale(metric, reference_value, mode='linear'):
    """Escala conservadora para expresar el cambio frente a 1 sola referencia.
    No pretende ser una desviación estándar poblacional: únicamente hace comparables
    variables con unidades distintas hasta disponer de suficiente histórico.
    """
    floors={
        'RMSSD':10.0,'SD1':7.0,'pNN50':10.0,'HF_DOM_PCT':10.0,'LF_DOM_PCT':10.0,'VLF_DOM_PCT':10.0,
        'MeanHR':5.0,'DFA_alpha1':0.15,'DFA_alpha2':0.15,'LF_HF':0.50,'Hurst':0.10,
        'SampEn':0.20,'DispEn':0.10,'D2':0.50,'Lyapunov_LLE':0.05,'WAV_ENTROPY_GLOBAL':0.10,
        'WAV_TRANSITIONS_PER_MIN':0.50,'HVG_graph_score_small_world':0.15,'HVG_clustering':0.05,
        'DET':5.0,'AUTO_Lmax_rel':0.05,'SDNN':10.0,'SD2':10.0,
        'IDX_Vagal':10.0,'IDX_Amplitud':10.0,'IDX_Complejidad':10.0,'IDX_Rigidez':10.0,
        'IDX_Regulacion_Lenta':10.0,'TOTAL':100.0,
        'AUTO_MSE_short':0.20,'AUTO_MSE_mid':0.20,'AUTO_MSE_long':0.20,
        'AUTO_MDE_short':0.20,'AUTO_MDE_mid':0.20,'AUTO_MDE_long':0.20,
    }
    rv=_autonomic_num(reference_value)
    if mode=='log':
        # En log1p una diferencia de ~0.20 equivale aproximadamente a un cambio relativo moderado.
        return 0.20
    floor=float(floors.get(str(metric),0.0) or 0.0)
    rel=0.20*abs(rv) if np.isfinite(rv) else 0.0
    scale=max(floor,rel,1e-9)
    return scale


def _autonomic_progressive_z(series, value, metric='', mode='linear'):
    """Normalización longitudinal progresiva usando EXCLUSIVAMENTE referencias previas.

    n=0  -> sin referencia.
    n=1  -> cambio individual normalizado respecto al registro previo (no se denomina Z estadístico).
    n=2-4 -> estandarización respecto a media/SD previas; si la SD colapsa, usa cambio normalizado.
    n>=5 -> Z robusto mediana/MAD; SD como fallback.
    """
    x_raw=pd.to_numeric(pd.Series(series),errors='coerce').astype(float).dropna()
    v_raw=_autonomic_num(value)
    if not np.isfinite(v_raw) or len(x_raw)==0:
        return np.nan
    x=_autonomic_transform_series(x_raw,mode=mode).dropna()
    if mode=='log':
        v=np.log1p(v_raw) if v_raw>=0 else np.nan
    else:
        v=v_raw
    if not np.isfinite(v) or len(x)==0:
        return np.nan
    n=len(x)
    if n>=5:
        med=float(x.median())
        mad=float(np.median(np.abs(x-med)))
        if np.isfinite(mad) and mad>1e-12:
            return float(np.clip((v-med)/(1.4826*mad),-4.0,4.0))
        sd=float(x.std(ddof=1))
        if np.isfinite(sd) and sd>1e-12:
            return float(np.clip((v-float(x.mean()))/sd,-4.0,4.0))
    elif n>=2:
        sd=float(x.std(ddof=1))
        if np.isfinite(sd) and sd>1e-12:
            return float(np.clip((v-float(x.mean()))/sd,-4.0,4.0))
    # 1 referencia, o histórico casi constante: diferencia normalizada respecto a la referencia central.
    center=float(x.iloc[-1] if n==1 else x.median())
    if mode=='log':
        scale=0.20
    else:
        raw_center=float(x_raw.iloc[-1] if n==1 else x_raw.median())
        scale=_autonomic_single_reference_scale(metric,raw_center,mode=mode)
    return float(np.clip((v-center)/scale,-4.0,4.0))


def _autonomic_row_mean(row, cols):
    vals=[_autonomic_num(row.get(c,np.nan)) for c in cols]
    vals=[v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _autonomic_add_derived(df):
    """Añade resúmenes no redundantes usados sólo por Control Autonómico."""
    if df is None or df.empty: return df
    out=df.copy()
    for prefix in ('MSE','MDE'):
        for label,a,b in [('short',1,5),('mid',6,10),('long',11,20)]:
            cols=[f'{prefix}{i}' for i in range(a,b+1) if f'{prefix}{i}' in out.columns]
            if cols:
                out[f'AUTO_{prefix}_{label}']=out[cols].apply(lambda r:pd.to_numeric(r,errors='coerce').mean(),axis=1)
        cols=[f'{prefix}{i}' for i in range(1,21) if f'{prefix}{i}' in out.columns]
        if len(cols)>=3:
            def _slope(r):
                y=pd.to_numeric(r[cols],errors='coerce').to_numpy(dtype=float)
                ok=np.isfinite(y)
                if ok.sum()<3: return np.nan
                xx=np.arange(1,len(cols)+1,dtype=float)[ok]
                return float(np.polyfit(xx,y[ok],1)[0])
            out[f'AUTO_{prefix}_slope']=out.apply(_slope,axis=1)
    if 'Lmax' in out.columns and 'N_RRi' in out.columns:
        lm=pd.to_numeric(out['Lmax'],errors='coerce'); nn=pd.to_numeric(out['N_RRi'],errors='coerce')
        out['AUTO_Lmax_rel']=lm/nn.where(nn>0)
    return out


def _autonomic_merge_data(long_df, indices_df):
    """Une métricas primarias + índices sin recalcular la fisiología."""
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    df=long_df.copy()
    if 'Registro' not in df.columns or 'Fase' not in df.columns:
        return pd.DataFrame()
    df['Registro']=df['Registro'].astype(str)
    df['Fase']=df['Fase'].astype(str)
    idx_cols=['Registro','Fase','IDX_Vagal','IDX_Amplitud','IDX_Rigidez','IDX_Regulacion_Lenta','IDX_Complejidad','IDX_Adaptabilidad','IDX_Confianza']
    if indices_df is not None and not indices_df.empty:
        ii=indices_df[[c for c in idx_cols if c in indices_df.columns]].copy()
        if {'Registro','Fase'}.issubset(ii.columns):
            ii['Registro']=ii['Registro'].astype(str); ii['Fase']=ii['Fase'].astype(str)
            for c in [x for x in idx_cols if x not in ('Registro','Fase') and x in df.columns]:
                df=df.drop(columns=[c])
            df=df.merge(ii.drop_duplicates(['Registro','Fase'],keep='last'),on=['Registro','Fase'],how='left')
    df['Paciente_ID']=df.get('Paciente_ID',df['Registro'].map(infer_patient_id))
    df['Paciente_ID']=df['Paciente_ID'].astype(str).map(normalize_patient_id)
    if 'Fecha_hora' not in df.columns:
        df['Fecha_hora']=df['Registro'].map(_extract_record_datetime)
    else:
        df['Fecha_hora']=pd.to_datetime(df['Fecha_hora'],errors='coerce')
    df=_autonomic_add_derived(df)
    return df.sort_values(['Paciente_ID','Fecha_hora','Registro','Fase'],na_position='last').reset_index(drop=True)


def _autonomic_attractor_label(record, record_data):
    """Descriptor exploratorio del registro activo; no fuerza recálculo de históricos incompletos."""
    try:
        rd=(record_data or {}).get(str(record),{}) or {}
        rr=np.asarray(rd.get('rr',[]),dtype=float)
        rr=rr[np.isfinite(rr)]
        if len(rr)<20:
            return 'No disponible'
        emb=_delay_embedding(rr,1,3)
        desc,_=_attractor_descriptor(rr,emb)
        return str(desc)
    except Exception:
        return 'No disponible'


def _autonomic_reference_group(df, row):
    """Referencia longitudinal estrictamente anterior al registro interpretado.

    Prioridad: misma fase del mismo paciente. Para fases no basales sin antecedentes de esa
    fase, puede usar basales PREVIOS sólo como referencia secundaria. Nunca usa registros
    posteriores ni el propio registro para construir su referencia.
    """
    pid=str(row.get('Paciente_ID','')); phase=str(row.get('Fase','')); rec=str(row.get('Registro',''))
    pat=df[df['Paciente_ID'].astype(str)==pid].copy()
    if pat.empty:
        return pat
    pat['Fecha_hora']=pd.to_datetime(pat.get('Fecha_hora'),errors='coerce')
    row_dt=pd.to_datetime(row.get('Fecha_hora'),errors='coerce')
    # Excluir la observación actual incluso si está duplicada entre caché actual e histórico.
    pat=pat[~((pat['Registro'].astype(str)==rec)&(pat['Fase'].astype(str)==phase))].copy()
    if pd.notna(row_dt):
        dated=pat['Fecha_hora'].notna()
        pat=pat[(~dated)|(pat['Fecha_hora']<row_dt)].copy()
    same=pat[pat['Fase'].astype(str)==phase].copy()
    same=same.sort_values(['Fecha_hora','Registro'],na_position='last')
    if not same.empty:
        return same
    if not phase.lower().startswith('basal'):
        basal=pat[pat['Fase'].astype(str).str.lower().str.startswith('basal')].copy()
        return basal.sort_values(['Fecha_hora','Registro'],na_position='last')
    return same


def _autonomic_reference_method(n):
    n=int(n or 0)
    if n<=0: return 'Sin referencia longitudinal previa'
    if n==1: return 'Δ individual respecto al registro previo'
    if n<5: return f'Estandarización individual con {n} registros previos'
    return f'Z robusto individual con {n} registros previos (mediana/MAD)'


def _autonomic_delta_previous(df, row, col):
    pid=str(row.get('Paciente_ID','')); phase=str(row.get('Fase','')); rec=str(row.get('Registro',''))
    g=df[(df['Paciente_ID'].astype(str)==pid)&(df['Fase'].astype(str)==phase)].copy()
    if g.empty or col not in g.columns: return np.nan
    g=g.sort_values(['Fecha_hora','Registro'],na_position='last').reset_index(drop=True)
    hits=g.index[g['Registro'].astype(str)==rec].tolist()
    if not hits: return np.nan
    i=hits[-1]
    if i<1: return np.nan
    a=_autonomic_num(g.loc[i,col]); b=_autonomic_num(g.loc[i-1,col])
    return float(a-b) if np.isfinite(a) and np.isfinite(b) else np.nan


def _autonomic_consensus_z(ref,row,cols,mode='log'):
    vals=[]
    for c in cols:
        if c in ref.columns:
            zz=_autonomic_progressive_z(ref[c],row.get(c,np.nan),metric=c,mode=mode)
            if np.isfinite(zz): vals.append(zz)
    return float(np.median(vals)) if vals else np.nan


def _autonomic_weighted_score(items):
    vals=[]; weights=[]
    for value,weight in items:
        if np.isfinite(value) and np.isfinite(weight) and weight>0:
            vals.append(float(value)); weights.append(float(weight))
    return float(np.average(vals,weights=weights)) if vals else np.nan


def _autonomic_quality_score(row):
    """0..1. Modula confianza; nunca decide por sí sola una forma autonómica."""
    parts=[]
    n=_autonomic_num(row.get('N_RRi',np.nan))
    if np.isfinite(n): parts.append(np.clip((n-30.0)/120.0,0.25,1.0))
    dur=_autonomic_num(row.get('Duration_s',np.nan))
    if np.isfinite(dur): parts.append(np.clip((dur-60.0)/240.0,0.35,1.0))
    art=_autonomic_num(row.get('Artefactos_pct',np.nan))
    if np.isfinite(art): parts.append(np.clip(1.0-art/20.0,0.20,1.0))
    ic=_autonomic_num(row.get('IDX_Confianza',np.nan))
    if np.isfinite(ic): parts.append(np.clip(ic/100.0,0.20,1.0))
    return float(np.mean(parts)) if parts else 0.75


def autonomic_control_assessment(df, row, record_data=None):
    """Matriz definitiva V/S/L/C/R/Q con reglas A–F y trazabilidad numérica."""
    ref=_autonomic_reference_group(df,row)
    get=lambda k:_autonomic_num(row.get(k,np.nan))
    z=lambda k,mode='linear':_autonomic_progressive_z(ref[k] if k in ref.columns else [],get(k),metric=k,mode=mode)

    # Z individuales principales
    zmap={}
    linear_cols=['RMSSD','SD1','HF_DOM_PCT','pNN50','DFA_alpha1','LF_DOM_PCT','MeanHR','LF_HF','IDX_Rigidez',
                 'VLF_DOM_PCT','DFA_alpha2','Hurst','IDX_Regulacion_Lenta','SampEn','DispEn','D2','Lyapunov_LLE',
                 'WAV_ENTROPY_GLOBAL','WAV_TRANSITIONS_PER_MIN','HVG_graph_score_small_world','HVG_clustering','DET',
                 'AUTO_Lmax_rel','SDNN','SD2','IDX_Vagal','IDX_Amplitud','IDX_Complejidad','TOTAL']
    for c in linear_cols:
        zmap[c]=z(c)

    hf_cons=_autonomic_consensus_z(ref,row,['HF','HF_LS','HF_AR','HF_WAV_MEAN'],mode='log')
    lf_cons=_autonomic_consensus_z(ref,row,['LF','LF_LS','LF_AR','LF_WAV_MEAN'],mode='log')
    vlf_cons=_autonomic_consensus_z(ref,row,['VLF','VLF_LS','VLF_AR','VLF_WAV_MEAN'],mode='log')
    total_cons=_autonomic_consensus_z(ref,row,['TOTAL','TOTAL_LS','TOTAL_AR'],mode='log')
    mse_z=[]; mde_z=[]
    for c in ['AUTO_MSE_short','AUTO_MSE_mid','AUTO_MSE_long']:
        if c in ref.columns:
            zz=z(c)
            if np.isfinite(zz): mse_z.append(zz)
    for c in ['AUTO_MDE_short','AUTO_MDE_mid','AUTO_MDE_long']:
        if c in ref.columns:
            zz=z(c)
            if np.isfinite(zz): mde_z.append(zz)
    mse_cons=float(np.mean(mse_z)) if mse_z else np.nan
    mde_cons=float(np.mean(mde_z)) if mde_z else np.nan

    # DFAα1: se incorpora longitudinalmente, pero su regla fisiológica conserva umbrales no monotónicos.
    V=_autonomic_weighted_score([
        (zmap['RMSSD'],1.00),(zmap['SD1'],1.00),(zmap['HF_DOM_PCT'],0.90),(zmap['pNN50'],0.55),
        (hf_cons,0.70),(zmap['IDX_Vagal'],0.40)
    ])
    S=_autonomic_weighted_score([
        (zmap['DFA_alpha1'],0.75),(zmap['LF_DOM_PCT'],1.00),(zmap['MeanHR'],0.55),(zmap['LF_HF'],0.45),
        (lf_cons,0.55),(zmap['IDX_Rigidez'],0.50)
    ])
    L=_autonomic_weighted_score([
        (zmap['VLF_DOM_PCT'],1.00),(vlf_cons,0.75),(zmap['DFA_alpha2'],0.50),(zmap['Hurst'],0.35),
        (zmap['IDX_Regulacion_Lenta'],0.45)
    ])
    # DET y Lmax/N entran invertidos: más determinismo/diagonales largas = más rigidez, no más complejidad.
    C=_autonomic_weighted_score([
        (zmap['SampEn'],0.85),(zmap['DispEn'],0.65),(zmap['D2'],0.60),(zmap['Lyapunov_LLE'],0.45),
        (zmap['WAV_ENTROPY_GLOBAL'],0.75),(zmap['WAV_TRANSITIONS_PER_MIN'],0.60),
        (zmap['HVG_graph_score_small_world'],0.75),(zmap['HVG_clustering'],0.45),
        (mse_cons,0.45),(mde_cons,0.45),(-zmap['DET'] if np.isfinite(zmap['DET']) else np.nan,0.25),
        (-zmap['AUTO_Lmax_rel'] if np.isfinite(zmap['AUTO_Lmax_rel']) else np.nan,0.25),(zmap['IDX_Complejidad'],0.35)
    ])
    R=_autonomic_weighted_score([
        (zmap['SDNN'],1.00),(zmap['SD2'],0.90),(total_cons,0.80),(zmap['IDX_Amplitud'],0.45)
    ])
    Q=_autonomic_quality_score(row)

    dfa1=get('DFA_alpha1'); lfdom=get('LF_DOM_PCT'); hfdom=get('HF_DOM_PCT'); vlfdom=get('VLF_DOM_PCT')
    lf_hf=get('LF_HF'); rigid=get('IDX_Rigidez'); went=get('WAV_ENTROPY_GLOBAL'); trans=get('WAV_TRANSITIONS_PER_MIN')
    sw=get('HVG_graph_score_small_world'); total=get('TOTAL')
    attr=_autonomic_attractor_label(row.get('Registro',''),record_data)
    d_lf=_autonomic_delta_previous(df,row,'LF_DOM_PCT'); d_vlf=_autonomic_delta_previous(df,row,'VLF_DOM_PCT')

    zv=zmap.get('IDX_Vagal',np.nan); zhf=zmap.get('HF_DOM_PCT',np.nan)
    discord=bool(np.isfinite(zv) and np.isfinite(zhf) and zv*zhf<0 and abs(zv-zhf)>=1.0)
    total_very_low=bool((np.isfinite(R) and R<-0.8) or (np.isfinite(zmap.get('TOTAL',np.nan)) and zmap['TOTAL']<-0.8) or (np.isfinite(total) and total<80))
    went_high=bool((np.isfinite(zmap.get('WAV_ENTROPY_GLOBAL',np.nan)) and zmap['WAV_ENTROPY_GLOBAL']>0.8) or (np.isfinite(went) and went>0.75))
    trans_high=bool((np.isfinite(zmap.get('WAV_TRANSITIONS_PER_MIN',np.nan)) and zmap['WAV_TRANSITIONS_PER_MIN']>0.8) or (np.isfinite(trans) and trans>=2.0))
    rigid_high=bool((np.isfinite(rigid) and rigid>=65) or (np.isfinite(zmap['IDX_Rigidez']) and zmap['IDX_Rigidez']>0.8))
    rigid_low=bool((np.isfinite(rigid) and rigid<45) or (np.isfinite(zmap['IDX_Rigidez']) and zmap['IDX_Rigidez']<-0.5))
    cycle_limit='ciclo límite' in attr.lower()
    fixed_like='punto fijo' in attr.lower()

    rules=[]
    A=(went_high and total_very_low) or (np.isfinite(sw) and sw<0.2 and trans_high) or (np.isfinite(dfa1) and dfa1<0.5 and discord)
    rules.append(('Desacoplamiento autonómico',A))
    B=(np.isfinite(V) and V>0.5) and (np.isfinite(S) and S>0.35) and ((np.isfinite(dfa1) and dfa1>1.15) or (np.isfinite(lfdom) and lfdom>40)) and (np.isfinite(lfdom) and np.isfinite(hfdom) and lfdom>hfdom)
    rules.append(('Coactivación con predominio simpático',B))
    Cc=(np.isfinite(V) and V>1.0) and ((np.isfinite(S) and S>0.1) or (np.isfinite(lfdom) and lfdom>30)) and (np.isfinite(lf_hf) and lf_hf<1.5) and (np.isfinite(hfdom) and np.isfinite(lfdom) and hfdom>lfdom)
    rules.append(('Coactivación con predominio parasimpático',Cc))
    D=(np.isfinite(V) and V>0.5) and ((np.isfinite(S) and S<0.0) or (np.isfinite(dfa1) and dfa1<=1.0)) and (np.isfinite(d_lf) and d_lf<0) and (np.isfinite(d_vlf) and d_vlf<0) and rigid_low and (not np.isfinite(R) or R>-1.0)
    rules.append(('Reciprocidad · predominio vagal puro',D))
    E=(np.isfinite(V) and V<-0.8) and ((np.isfinite(S) and S>0.5) or rigid_high) and (cycle_limit or (rigid_high and np.isfinite(S) and S>0.8))
    rules.append(('Retirada vagal · predominio simpático',E))
    F=(np.isfinite(zmap.get('VLF_DOM_PCT',np.nan)) and zmap['VLF_DOM_PCT']<-0.8) and (np.isfinite(zmap.get('LF_DOM_PCT',np.nan)) and zmap['LF_DOM_PCT']<-0.8) and (np.isfinite(V) and abs(V)<=0.5) and (not np.isfinite(S) or S<0.2)
    rules.append(('Retirada simpática · desconexión del drive / atonía',F))

    matched=[name for name,ok in rules if ok]
    exact=bool(matched)
    if matched:
        state=matched[0]
    else:
        scores={
            'Desacoplamiento autonómico': float(went_high)+float(total_very_low)+float(np.isfinite(sw) and sw<0.2)+float(trans_high)+float(np.isfinite(dfa1) and dfa1<0.5)+float(discord)+0.35*max(0,-R if np.isfinite(R) else 0),
            'Coactivación con predominio simpático': max(0,V if np.isfinite(V) else 0)+max(0,S if np.isfinite(S) else 0)+float(np.isfinite(lfdom) and np.isfinite(hfdom) and lfdom>hfdom),
            'Coactivación con predominio parasimpático': max(0,V if np.isfinite(V) else 0)+0.5*max(0,S if np.isfinite(S) else 0)+float(np.isfinite(hfdom) and np.isfinite(lfdom) and hfdom>lfdom)+float(np.isfinite(lf_hf) and lf_hf<1.5),
            'Reciprocidad · predominio vagal puro': max(0,V if np.isfinite(V) else 0)+max(0,-S if np.isfinite(S) else 0)+float(np.isfinite(dfa1) and dfa1<=1.0)+float(np.isfinite(d_lf) and d_lf<0)+float(np.isfinite(d_vlf) and d_vlf<0)+float(rigid_low),
            'Retirada vagal · predominio simpático': max(0,-V if np.isfinite(V) else 0)+max(0,S if np.isfinite(S) else 0)+float(rigid_high)+float(cycle_limit),
            'Retirada simpática · desconexión del drive / atonía': max(0,-zmap.get('VLF_DOM_PCT',0) if np.isfinite(zmap.get('VLF_DOM_PCT',np.nan)) else 0)+max(0,-zmap.get('LF_DOM_PCT',0) if np.isfinite(zmap.get('LF_DOM_PCT',np.nan)) else 0)+float(np.isfinite(V) and abs(V)<=0.5)+float(fixed_like),
        }
        state=max(scores,key=scores.get)

    # Disponibilidad de núcleo prioritario, sin penalizar por parámetros técnicos ausentes.
    core=['RMSSD','SD1','HF_DOM_PCT','DFA_alpha1','LF_DOM_PCT','VLF_DOM_PCT','SDNN','SD2','WAV_ENTROPY_GLOBAL','WAV_TRANSITIONS_PER_MIN','SampEn']
    avail=sum(np.isfinite(get(k)) for k in core)/len(core)
    base_conf=(0.50+0.50*avail) if exact else (0.28+0.42*avail)
    conf=float(np.clip(100*base_conf*(0.65+0.35*Q),0,100))

    reason=[]
    def add(k,label=None):
        v=get(k); zz=zmap.get(k,np.nan)
        if np.isfinite(v): reason.append(f"{label or k}={v:.3g}"+(f" (Z={zz:+.2f})" if np.isfinite(zz) else ""))
    for k in ['RMSSD','SD1','pNN50','HF_DOM_PCT','DFA_alpha1','LF_DOM_PCT','LF_HF','MeanHR','IDX_Rigidez','VLF_DOM_PCT','DFA_alpha2','Hurst','SampEn','DispEn','D2','Lyapunov_LLE','WAV_ENTROPY_GLOBAL','WAV_TRANSITIONS_PER_MIN','HVG_graph_score_small_world','HVG_clustering','SDNN','SD2','TOTAL']:
        add(k)
    if np.isfinite(hf_cons): reason.append(f"HF_consenso_Z={hf_cons:+.2f}")
    if np.isfinite(lf_cons): reason.append(f"LF_consenso_Z={lf_cons:+.2f}")
    if np.isfinite(vlf_cons): reason.append(f"VLF_consenso_Z={vlf_cons:+.2f}")
    if np.isfinite(total_cons): reason.append(f"TOTAL_consenso_Z={total_cons:+.2f}")
    if np.isfinite(mse_cons): reason.append(f"MSE_multiescala_Z={mse_cons:+.2f}")
    if np.isfinite(mde_cons): reason.append(f"MDE_multiescala_Z={mde_cons:+.2f}")
    reason.append(f"Atractor={attr}")
    if np.isfinite(d_lf): reason.append(f"ΔLF_DOM={d_lf:+.2f} pp")
    if np.isfinite(d_vlf): reason.append(f"ΔVLF_DOM={d_vlf:+.2f} pp")

    spec=AUTONOMIC_FORMS[state]
    return {
        'Estado_control_autonomico':state,
        'Regla_estricta_cumplida':exact,
        'Confianza_clasificacion_pct':conf,
        'Vector_V_Z':V,'Vector_S_Z':S,'Vector_L_Z':L,'Vector_C_Z':C,'Vector_R_Z':R,'Calidad_Q':Q,
        'HF_consenso_Z':hf_cons,'LF_consenso_Z':lf_cons,'VLF_consenso_Z':vlf_cons,'TOTAL_consenso_Z':total_cons,
        'MSE_multiescala_Z':mse_cons,'MDE_multiescala_Z':mde_cons,
        'Atractor':attr,'Coste_alostatico':spec['coste'],
        'Interpretacion_fisiologica':spec['interpretacion'],'Implicacion_clinica':spec['clinica'],
        'Justificacion':' · '.join(reason),
        'Referencia_n':int(len(ref)),
        'Referencia_metodo':_autonomic_reference_method(len(ref)),
        'Referencia_fase':str(row.get('Fase','')),
    }


def build_autonomic_control_table(long_df,indices_df,record_data=None,reference_history=None):
    current=_autonomic_merge_data(long_df,indices_df)
    if current.empty: return pd.DataFrame(),current
    ref=current.copy()
    if reference_history is not None and isinstance(reference_history,pd.DataFrame) and not reference_history.empty:
        h=reference_history.copy()
        if 'Registro' in h.columns and 'Fase' in h.columns:
            h['Registro']=h['Registro'].astype(str); h['Fase']=h['Fase'].astype(str)
            if 'Paciente_ID' not in h.columns:
                h['Paciente_ID']=h['Registro'].map(infer_patient_id)
            h['Paciente_ID']=h['Paciente_ID'].astype(str).map(normalize_patient_id)
            if 'Fecha_hora' in h.columns:
                h['Fecha_hora']=pd.to_datetime(h['Fecha_hora'],errors='coerce')
            else:
                h['Fecha_hora']=h['Registro'].map(_extract_record_datetime)
            h=_autonomic_add_derived(h)
            ref=pd.concat([h,current],ignore_index=True,sort=False)
            ref=ref.drop_duplicates(['Paciente_ID','Registro','Fase'],keep='last')
            ref=ref.sort_values(['Paciente_ID','Fecha_hora','Registro','Fase'],na_position='last').reset_index(drop=True)
    rows=[]
    for _,r in current.iterrows():
        a=autonomic_control_assessment(ref,r,record_data=record_data)
        rows.append({'Registro':r.get('Registro',''),'Fase':r.get('Fase',''),'Fecha_hora':r.get('Fecha_hora',pd.NaT),'Paciente_ID':r.get('Paciente_ID',''),**a})
    return pd.DataFrame(rows),ref

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(["1) Segmentar tipo Kubios", "2) HRV", "3) Comparar", "4) No lineales / MSE / Atractor", "5) Poincaré / Grafos", "6) Dashboard", "7) Informe", "8) Exportar", "9) Índices fisiológicos", "10) Control autonómico", "11) Predicción fisiológica"])



# ============================================================
# SCRIPT LOCAL HTML -> PNG
# ============================================================



ARRANCAR_CONVERTIDOR_BAT = '@echo off\ntitle Convertidor HRV HTML a PNG\n\necho ============================================================\necho  VRC / HRV RRi Analyzer Pro - Convertidor HTML/Localhost a PNG\necho ============================================================\necho.\necho Este arrancador funciona desde cualquier carpeta porque usa %%~dp0.\necho.\n\necho Abriendo Streamlit local en el navegador...\nstart "" "http://localhost:8501/"\n\necho.\necho Esperando a que cargue la app...\ntimeout /t 5 >nul\n\necho.\necho Generando captura PNG de http://localhost:8501/ ...\npython "%~dp0capture_streamlit_localhost_png.py" "http://localhost:8501/" "%~dp0captura_streamlit.png"\n\necho.\necho Si quieres convertir los HTML exportados a PNG, ejecutando ahora:\npython "%~dp0convert_html_to_png.py"\n\necho.\necho ============================================================\necho  Proceso terminado.\necho  Captura principal:\necho  %~dp0captura_streamlit.png\necho.\necho  PNG desde HTML, si existen:\necho  %~dp0graficos\\png_from_html\necho ============================================================\necho.\npause\n'

CAPTURE_STREAMLIT_LOCALHOST_PNG_SCRIPT = r"""
# capture_streamlit_localhost_png.py
# Captura la app Streamlit o cualquier URL local como PNG.
#
# Uso:
#   1) Ejecuta tu app local:
#        streamlit run app.py
#
#   2) Instala Playwright:
#        pip install playwright
#        python -m playwright install chromium
#
#   3) Ejecuta:
#        python capture_streamlit_localhost_png.py
#
# Por defecto captura:
#        http://localhost:8501/
#
# También puedes cambiar URL y salida:
#        python capture_streamlit_localhost_png.py http://localhost:8501/ captura.png

from pathlib import Path
import sys
import asyncio
from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("captura_streamlit.png")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1920, "height": 1400},
            device_scale_factor=2
        )

        print(f"Abriendo: {URL}")
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # Captura página completa
        await page.screenshot(path=str(OUT), full_page=True)

        await browser.close()

    print(f"PNG guardado en: {OUT.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
"""

CONVERT_HTML_TO_PNG_SCRIPT = r"""
# convert_html_to_png.py
# Convierte todos los gráficos HTML exportados por la app a PNG.
#
# Uso local:
#   1) Instala dependencias:
#        pip install playwright
#        python -m playwright install chromium
#
#   2) Descomprime el ZIP exportado por la app.
#
#   3) Ejecuta:
#        python convert_html_to_png.py
#
# El script buscará la carpeta graficos/html y creará graficos/png_from_html.

from pathlib import Path
import asyncio
from playwright.async_api import async_playwright

BASE = Path(__file__).resolve().parent
HTML_DIR = BASE / "graficos" / "html"
OUT_DIR = BASE / "graficos" / "png_from_html"
OUT_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    html_files = sorted(HTML_DIR.glob("*.html"))
    if not html_files:
        print(f"No se han encontrado HTML en: {HTML_DIR}")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1800, "height": 1100},
            device_scale_factor=2
        )

        for html in html_files:
            url = html.resolve().as_uri()
            out = OUT_DIR / (html.stem + ".png")
            print(f"Convirtiendo: {html.name} -> {out.name}")
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(1200)
            await page.screenshot(path=str(out), full_page=True)

        await browser.close()

    print(f"Listo. PNG guardados en: {OUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
"""

# ============================================================
# EXPORTACIÓN DE GRÁFICOS
# ============================================================

def _safe_filename(text, max_len=90):
    """
    Nombre de archivo seguro.
    """
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "grafico"
    return text[:max_len]



# Paleta fija para exportación en color.
EXPORT_COLORWAY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#00bcd4", "#ff4b4b", "#4caf50", "#ffc107", "#9c27b0",
    "#03a9f4", "#ff9800", "#8bc34a", "#f44336", "#673ab7",
]

def _export_color_for(i):
    return EXPORT_COLORWAY[int(i) % len(EXPORT_COLORWAY)]


def _trace_has_color(trace, attr_path):
    """
    Comprueba si un trace tiene color explícito.
    attr_path ejemplos:
    - ("marker", "color")
    - ("line", "color")
    """
    try:
        obj = trace
        for attr in attr_path:
            obj = getattr(obj, attr)
        if obj is None:
            return False
        if isinstance(obj, (list, tuple, np.ndarray)):
            return len(obj) > 0
        return str(obj) != ""
    except Exception:
        return False


def _prepare_plotly_fig_for_color_export(fig):
    """
    Prepara una copia de la figura para exportar con colores fijos.

    Motivo:
    En Streamlit los gráficos pueden verse coloreados por el tema del navegador,
    pero al exportar con Kaleido/HTML fuera de Streamlit algunos traces sin color explícito
    pueden salir grises. Aquí se fija:
    - plantilla oscura,
    - colorway,
    - fondo oscuro,
    - color explícito en barras, líneas, marcadores y redes.
    """
    try:
        f = copy.deepcopy(fig)
    except Exception:
        f = fig

    try:
        f.update_layout(
            template="plotly_dark",
            colorway=EXPORT_COLORWAY,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            legend=dict(
                bgcolor="rgba(14,17,23,0.75)",
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
                font=dict(color="#FAFAFA"),
            ),
        )

        for ax in list(f.layout):
            if str(ax).startswith("xaxis") or str(ax).startswith("yaxis"):
                try:
                    f.layout[ax].update(
                        gridcolor="rgba(255,255,255,0.12)",
                        zerolinecolor="rgba(255,255,255,0.20)",
                        linecolor="rgba(255,255,255,0.25)",
                        tickfont=dict(color="#FAFAFA"),
                        titlefont=dict(color="#FAFAFA"),
                    )
                except Exception:
                    pass

        for i, tr in enumerate(f.data):
            col = _export_color_for(i)

            # Barras
            if getattr(tr, "type", "") == "bar":
                if not _trace_has_color(tr, ("marker", "color")):
                    tr.marker.color = col
                try:
                    tr.marker.line.color = "rgba(255,255,255,0.25)"
                    tr.marker.line.width = 0.5
                except Exception:
                    pass

            # Líneas y puntos
            if getattr(tr, "type", "") == "scatter":
                mode = str(getattr(tr, "mode", "") or "")

                if "lines" in mode:
                    if not _trace_has_color(tr, ("line", "color")):
                        tr.line.color = col
                    if not getattr(tr.line, "width", None):
                        tr.line.width = 2.5

                if "markers" in mode:
                    if not _trace_has_color(tr, ("marker", "color")):
                        tr.marker.color = col
                    if not getattr(tr.marker, "size", None):
                        tr.marker.size = 7

                # Si son aristas de grafos sin color, dar gris azulado visible, no negro.
                if mode == "lines" and (getattr(tr, "showlegend", None) is False):
                    name = str(getattr(tr, "name", "") or "").lower()
                    if ("edge" in name) or ("arista" in name) or len(getattr(tr, "x", []) or []) > 100:
                        if not _trace_has_color(tr, ("line", "color")):
                            tr.line.color = "rgba(120,180,255,0.35)"
                            tr.line.width = 0.8

    except Exception:
        pass

    return f


def _write_plotly_html(fig, out_path, title=None):
    """
    Guarda una figura Plotly como HTML interactivo manteniendo colores fijos.
    """
    try:
        fig_export = _prepare_plotly_fig_for_color_export(fig)
        fig_export.write_html(
            str(out_path),
            include_plotlyjs="cdn",
            full_html=True,
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": _safe_filename(title or out_path.stem),
                    "height": 1000,
                    "width": 1600,
                    "scale": 2,
                },
                "displaylogo": False,
            },
        )
        return True
    except Exception:
        return False


def _try_write_plotly_png(fig, out_path):
    """
    Intenta guardar PNG en color si Kaleido está disponible.
    """
    try:
        fig_export = _prepare_plotly_fig_for_color_export(fig)
        fig_export.write_image(
            str(out_path),
            width=1800,
            height=1100,
            scale=2,
            format="png",
        )
        return True
    except Exception:
        return False




def build_attractor_export_bundle(record_data, records, max_points=1200):
    """Reconstruye para exportación los atractores con el mismo estado/snapshot usado por la app."""
    rows = []
    figures = []
    tau_mode = st.session_state.get("attractor_tau_mode_v15320", "Automático por registro")
    tau_common = int(st.session_state.get("attractor_tau_common_v15320", 1) or 1)
    m_default = int(st.session_state.get("attractor_m_v15320", 3) or 3)
    for rec in list(records or []):
        rd = record_data.get(rec, {}) or {}
        rr = np.asarray(rd.get("rr", []), dtype=float)
        rr = rr[np.isfinite(rr)]
        snap = rd.get("analysis_snapshot") if rd.get("source") == "historial" else None
        snap_attr = (((snap or {}).get("config") or {}).get("attractor_snapshot") if snap else None)
        if rd.get("source") == "historial" and not snap_attr:
            rows.append({"Registro": rec, "Tau": np.nan, "Embedding_m": np.nan,
                         "Descriptor": "Histórico no reproducible",
                         "Evidencia": "Snapshot heredado incompleto: no se recalcula el atractor.",
                         "Verificado_snapshot": False, "N_puntos_embedding": 0})
            continue
        if len(rr) < 10:
            rows.append({"Registro": rec, "Tau": np.nan, "Embedding_m": m_default,
                         "Descriptor": "Insuficiente", "Evidencia": "Muy pocos RRi",
                         "Verificado_snapshot": False, "N_puntos_embedding": 0})
            continue
        verified = False
        if snap_attr:
            try:
                tau = int(snap_attr.get("tau"))
                m_rec = int(snap_attr.get("m", 3))
            except Exception:
                rows.append({"Registro": rec, "Tau": np.nan, "Embedding_m": np.nan,
                             "Descriptor": "Histórico no reproducible",
                             "Evidencia": "Parámetros de atractor incompletos en snapshot.",
                             "Verificado_snapshot": False, "N_puntos_embedding": 0})
                continue
            expected_hash = snap_attr.get("rr_sha256")
            actual_hash = hashlib.sha256(np.ascontiguousarray(rr, dtype=np.float64).tobytes()).hexdigest()
            if expected_hash and expected_hash != actual_hash:
                rows.append({"Registro": rec, "Tau": tau, "Embedding_m": m_rec,
                             "Descriptor": "Histórico no reproducible",
                             "Evidencia": "La serie RRi recuperada no coincide con la serie congelada.",
                             "Verificado_snapshot": False, "N_puntos_embedding": 0})
                continue
            verified = True
        else:
            m_rec = m_default
            tau = int(_attractor_delay(rr)) if tau_mode == "Automático por registro" else tau_common
            tau = max(1, min(tau, max(1, len(rr)//max(3, m_rec))))
        emb = _delay_embedding(rr, tau, m_rec)
        desc, evidence = _attractor_descriptor(rr, emb)
        if verified:
            evidence = "Atractor histórico verificado ✓ · " + str(evidence)
        rows.append({"Registro": rec, "Tau": tau, "Embedding_m": m_rec,
                     "Descriptor": desc, "Evidencia": evidence,
                     "Verificado_snapshot": verified, "N_puntos_embedding": int(len(emb))})
        if len(emb) == 0:
            continue
        show = emb if len(emb) <= max_points else emb[np.linspace(0, len(emb)-1, int(max_points)).astype(int)]
        if m_rec >= 3:
            fig3 = go.Figure(go.Scatter3d(
                x=show[:,0], y=show[:,1], z=show[:,2], mode="lines", name=str(rec),
                hovertemplate=("<b>"+str(rec)+"</b><br>RR(t)=%{x:.1f}<br>RR(t-τ)=%{y:.1f}<br>RR(t-2τ)=%{z:.1f}<extra></extra>")
            ))
            fig3.update_layout(title=f"Atractor reconstruido 3D · {rec}", height=650,
                               scene=dict(xaxis_title="RR(t)", yaxis_title="RR(t-τ)", zaxis_title="RR(t-2τ)"),
                               showlegend=False)
            figures.append((f"30_Atractor_3D_{rec}", fig3))
        fig2 = go.Figure(go.Scatter(
            x=show[:,0], y=show[:,1], mode="lines", name=str(rec),
            hovertemplate=("<b>"+str(rec)+"</b><br>RR(t)=%{x:.1f}<br>RR(t-τ)=%{y:.1f}<extra></extra>")
        ))
        fig2.update_layout(title=f"Espacio de estados 2D · {rec}", height=520,
                           xaxis_title="RR(t)", yaxis_title="RR(t-τ)", showlegend=False)
        figures.append((f"31_Atractor_2D_{rec}", fig2))
    return pd.DataFrame(rows), figures


def build_prediction_export_bundle(history_df, records=None):
    """Exporta trazabilidad completa del motor v15.3.x para los pacientes visibles."""
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame())
    if history_df is None or history_df.empty:
        return empty
    df = history_df.copy()
    visible_pids = []
    for rec in list(records or []):
        try:
            pid = normalize_patient_id(infer_patient_id(rec))
            if pid and pid not in visible_pids:
                visible_pids.append(pid)
        except Exception:
            pass
    if visible_pids:
        mask = df['Paciente_ID'].astype(str).map(normalize_patient_id).isin(visible_pids)
        scoped = df[mask].copy()
        if not scoped.empty:
            df = scoped
    patients = sorted(df['Paciente_ID'].dropna().astype(str).unique().tolist(), key=lambda x: str(x).lower())
    summary_all, details_all, topo_all, figures = [], [], [], []
    for pid in patients:
        preds, summary, details = predict_v15_all_phases(df, pid)
        if not summary.empty:
            summary = summary.copy()
            summary.insert(0, 'Paciente_ID', pid)
            # Trazabilidad explícita Nivel 1 / Nivel 2.
            extra = []
            for _, r in summary.iterrows():
                pred = preds.get(str(r['Fase']))
                extra.append({
                    'Nivel1_modo': (pred or {}).get('modo_prediccion_nivel1', (pred or {}).get('modo_prediccion','')),
                    'Nivel2_activo': bool((pred or {}).get('nivel2_activo', False)),
                    'Nivel2_indices': ', '.join(map(str, (pred or {}).get('nivel2_indices', []) or [])),
                    'Nivel2_topologia_HVG': ', '.join(map(str, (pred or {}).get('nivel2_topologia', []) or [])),
                    'Nivel2_version': (pred or {}).get('nivel2_version'),
                    'Compuesto_predicho_Nivel1': (pred or {}).get('compuesto_predicho_nivel1', (pred or {}).get('compuesto_predicho', np.nan)),
                })
            summary = pd.concat([summary.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
            summary_all.append(summary)
            figures.append((f"40_Prediccion_resumen_{pid}", v15_multiphase_summary_figure(summary)))
        if not details.empty:
            details = details.copy(); details.insert(0, 'Paciente_ID', pid); details_all.append(details)
            figures.append((f"41_Prediccion_indices_{pid}", v15_multiphase_indices_figure(details)))
        for phase, pred in preds.items():
            figures.append((f"42_Prediccion_detalle_{pid}_{phase}", v15_prediction_figure(pred)))
            topo = pred.get('topologia_hvg') or {}
            tdf = topo.get('tabla')
            if isinstance(tdf, pd.DataFrame) and not tdf.empty:
                tx = tdf.copy()
                tx.insert(0, 'Paciente_ID', pid); tx.insert(1, 'Fase', phase)
                tx.insert(2, 'Descriptor_actual', topo.get('descriptor_actual',''))
                tx.insert(3, 'Descriptor_previsto', topo.get('descriptor_previsto',''))
                tx.insert(4, 'Score_actual', topo.get('score_actual',np.nan))
                tx.insert(5, 'Score_previsto', topo.get('score_previsto',np.nan))
                tx.insert(6, 'Direccion_actual', topo.get('direccion_actual',''))
                tx.insert(7, 'Direccion_prevista', topo.get('direccion_prevista',''))
                topo_all.append(tx)
    summary_df = pd.concat(summary_all, ignore_index=True) if summary_all else pd.DataFrame()
    details_df = pd.concat(details_all, ignore_index=True) if details_all else pd.DataFrame()
    topo_df = pd.concat(topo_all, ignore_index=True) if topo_all else pd.DataFrame()
    try:
        state = pd.DataFrame([get_v153_training_state()])
    except Exception:
        state = pd.DataFrame()
    try:
        bundle = load_v153_level2()
        metrics = pd.DataFrame(bundle.get('metrics', {})).T.reset_index().rename(columns={'index':'Salida'}) if bundle else pd.DataFrame()
        targets = pd.DataFrame({'Salida_ML_validada': bundle.get('active_targets', [])}) if bundle else pd.DataFrame()
    except Exception:
        metrics, targets = pd.DataFrame(), pd.DataFrame()
    return summary_df, details_df, topo_df, state, figures, metrics, targets


def append_complete_export_report(base_report, indices_df, attractor_df, pred_summary, pred_details, pred_topology, model_state):
    """Añade al informe exportado las capas nuevas sin alterar el informe clínico base."""
    parts = [base_report.rstrip(), "", "## Índices fisiológicos multivariados", ""]
    if indices_df is None or indices_df.empty:
        parts.append("No hay índices fisiológicos disponibles en este contexto.")
    else:
        cols = [c for c in ['Registro','Fase','IDX_Vagal','IDX_Amplitud','IDX_Complejidad','IDX_Rigidez','IDX_Adaptabilidad','IDX_Regulacion_Lenta','Perfil_autonomico'] if c in indices_df.columns]
        parts.append(indices_df[cols].round(3).to_markdown(index=False))
    parts += ["", "## Atractores / dinámica reconstruida", ""]
    if attractor_df is None or attractor_df.empty:
        parts.append("No hay atractores exportables.")
    else:
        parts.append(attractor_df.round(4).to_markdown(index=False))
    parts += ["", "## Predicción fisiológica híbrida v15.3.x", ""]
    if pred_summary is None or pred_summary.empty:
        parts.append("No hay predicción fisiológica exportable para el contexto actual.")
    else:
        parts.append("### Resumen por paciente y fase")
        parts.append(pred_summary.round(4).to_markdown(index=False))
        parts.append("")
        parts.append("La columna `Nivel2_activo` indica si Gradient Boosting intervino realmente en esa salida. Si es falso, la predicción permanece íntegramente en Nivel 1.")
    if pred_topology is not None and not pred_topology.empty:
        parts += ["", "### Evolución topológica HVG usada por el motor", pred_topology.round(4).to_markdown(index=False)]
    if model_state is not None and not model_state.empty:
        parts += ["", "### Estado del autoaprendizaje Nivel 2", model_state.astype(str).to_markdown(index=False)]
    parts += ["", "### Nota metodológica", "Las salidas predictivas describen evolución fisiológica experimental y no equivalen a diagnóstico ni a probabilidad clínica validada."]
    return "\n".join(parts) + "\n"

def build_all_export_figures(
    record_data,
    records_results,
    long_df,
    records,
    selected_record,
    global_windows,
    record_windows,
    active_phases,
    use_independent,
    domain_method,
    include_hvg,
    dashboard_params=None,
    dashboard_phases=None,
):
    """
    Construye todos los gráficos principales de la app para exportarlos.

    Exporta HTML interactivo:
    - resumen HRV por registro,
    - dominios por registro,
    - MSE por registro,
    - comparativa MSE entre registros,
    - dashboard comparativo,
    - Poincaré por fases y por fase,
    - HVG por fases, comparativo e individual si está activado.
    """
    figures = []

    def add(name, fig):
        try:
            if fig is not None:
                figures.append((name, fig))
        except Exception:
            pass

    available_phases = []
    try:
        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
    except Exception:
        available_phases = active_phases or ["Basal"]

    if not available_phases:
        available_phases = active_phases or ["Basal"]

    # 1) HRV, dominios y MSE por registro
    for rec in records:
        dfrec = records_results.get(rec, pd.DataFrame())
        if dfrec is None or dfrec.empty:
            continue

        add(f"01_HRV_resumen_{rec}", hrv_phase_summary_figure(dfrec))
        add(f"02_Dominios_{rec}", domains_figure(dfrec, method=domain_method, title=f"Dominios · {rec}"))
        add(f"03_MSE_1_20_{rec}", mse_figure(dfrec, title=f"MSE 1-20 · {rec}"))

        # Poincaré: todas las fases del registro
        try:
            add(
                f"04_Poincare_panel_fases_{rec}",
                poincare_all_phases_panel_figure(
                    record_data,
                    global_windows,
                    record_windows,
                    rec,
                    use_independent,
                ),
            )
        except Exception:
            pass

        # HVG: todas las fases del registro
        if include_hvg:
            try:
                add(
                    f"05_HVG_panel_fases_{rec}",
                    hvg_all_phases_panel_figure(
                        record_data,
                        global_windows,
                        record_windows,
                        rec,
                        use_independent,
                        max_nodes=120,
                    ),
                )
            except Exception:
                pass

            try:
                add(
                    f"06_HVG_metricas_fases_{rec}",
                    hvg_metrics_all_phases_figure(dfrec),
                )
            except Exception:
                pass

    # 2) Comparativas entre registros
    if long_df is not None and not long_df.empty:
        # Dashboard general
        try:
            numeric_vars = [
                c for c in long_df.columns
                if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])
            ]
            default_params = [p for p in (dashboard_params or DEFAULT_MULTI) if p in numeric_vars]
            if not default_params:
                default_params = numeric_vars[:8]
            phases_for_dash = dashboard_phases or available_phases
            if default_params:
                add(
                    "10_Dashboard_comparativo_barras_linea_suavizada",
                    dashboard_bar_smooth(long_df, phases_for_dash, default_params),
                )
        except Exception:
            pass

        # Comparativas individuales de TODOS los parámetros numéricos calculados.
        # v15.4.5 Complete Scientific Export: no se limita a un subconjunto de métricas.
        try:
            all_numeric_params = [
                c for c in long_df.columns
                if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])
            ]
            for param in all_numeric_params:
                pivot = long_df.pivot_table(index="Fase", columns="Registro", values=param, aggfunc="first")
                if pivot is not None and not pivot.empty:
                    add(f"11_Comparativa_{param}", comparison_bar_line(pivot, param))
        except Exception:
            pass

        # Comparativa MSE
        try:
            add(
                "12_Comparativa_MSE_1_20",
                mse_compare_figure(long_df, available_phases, scales=list(range(1, 21))),
            )
        except Exception:
            pass

        # RRi superpuesto por fase
        for ph in available_phases:
            try:
                add(
                    f"13_RRi_superpuesto_{ph}",
                    phase_rr_overlay(record_data, global_windows, record_windows, ph, use_independent),
                )
            except Exception:
                pass

        # Poincaré/HVG por fase. Históricos extensos se exportan por lotes para
        # evitar figuras gigantes (100-200+ visitas siguen incluidas, en varias páginas).
        _export_records = sorted(record_data.keys(), key=lambda r: (extract_datetime_from_name(str(r)), str(r)))
        _export_batch_size = 18
        _export_batches = [_export_records[i:i+_export_batch_size] for i in range(0, len(_export_records), _export_batch_size)] or [[]]
        for ph in available_phases:
            for _bi, _batch in enumerate(_export_batches, start=1):
                if not _batch:
                    continue
                _batch_data = {r: record_data[r] for r in _batch}
                try:
                    add(
                        f"14_Poincare_panel_{ph}_parte_{_bi:02d}",
                        poincare_panel_figure(_batch_data, global_windows, record_windows, ph, use_independent),
                    )
                except Exception:
                    pass
            try:
                add(
                    f"15_Poincare_superpuesto_{ph}",
                    poincare_figure(record_data, global_windows, record_windows, ph, use_independent),
                )
            except Exception:
                pass

        # HVG por fase, también por lotes en históricos grandes.
        if include_hvg:
            for ph in available_phases:
                for _bi, _batch in enumerate(_export_batches, start=1):
                    if not _batch:
                        continue
                    _batch_data = {r: record_data[r] for r in _batch}
                    try:
                        add(
                            f"16_HVG_comparativo_{ph}_parte_{_bi:02d}",
                            hvg_network_compare_figure(
                                _batch_data,
                                global_windows,
                                record_windows,
                                ph,
                                use_independent,
                                max_nodes=120,
                            ),
                        )
                    except Exception:
                        pass

                try:
                    hvg_cols_export = [
                        "HVG_graph_score_scale_free",
                        "HVG_graph_score_small_world",
                        "HVG_graph_score_chain",
                        "HVG_compactness_index",
                        "HVG_nodes",
                        "HVG_edges",
                        "HVG_degree_mean",
                        "HVG_degree_max",
                        "HVG_hubs_p90",
                        "HVG_clustering",
                        "HVG_lambda",
                        "HVG_path_length",
                        "HVG_diameter",
                    ]
                    hvg_df = long_df[long_df["Fase"] == ph]
                    for hvg_param in hvg_cols_export:
                        if hvg_param in hvg_df.columns and pd.api.types.is_numeric_dtype(hvg_df[hvg_param]):
                            pivot_hvg = hvg_df.pivot_table(index="Fase", columns="Registro", values=hvg_param, aggfunc="first")
                            if pivot_hvg is not None and not pivot_hvg.empty:
                                add(f"17_HVG_metrica_{hvg_param}_{ph}", comparison_bar_line(pivot_hvg, hvg_param))
                except Exception:
                    pass

    return figures


def write_all_graph_exports(figures, outdir, formats=("html",)):
    """
    Guarda todos los gráficos en los formatos indicados.

    Formatos:
    - html: siempre recomendado; no depende de motores externos.
    - png: usa Plotly + Kaleido. Si Kaleido no está disponible, no rompe la app.
    - svg: usa Plotly + Kaleido. Útil para publicaciones.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    formats = set(formats or ["html"])

    html_dir = outdir / "html"
    png_dir = outdir / "png"
    svg_dir = outdir / "svg"

    if "html" in formats:
        html_dir.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        png_dir.mkdir(parents=True, exist_ok=True)
    if "svg" in formats:
        svg_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    used_names = set()
    png_errors = []

    for i, (name, fig) in enumerate(figures, start=1):
        base = _safe_filename(f"{i:03d}_{name}")
        while base in used_names:
            base = _safe_filename(base + "_copy")
        used_names.add(base)

        html_name = ""
        png_name = ""
        svg_name = ""
        ok_html = False
        ok_png = False
        ok_svg = False

        if "html" in formats:
            html_path = html_dir / f"{base}.html"
            ok_html = _write_plotly_html(fig, html_path, title=name)
            if ok_html:
                html_name = f"html/{html_path.name}"

        if "png" in formats:
            png_path = png_dir / f"{base}.png"
            ok_png = _try_write_plotly_png(fig, png_path)
            if ok_png:
                png_name = f"png/{png_path.name}"
            else:
                png_errors.append(base)

        if "svg" in formats:
            try:
                svg_path = svg_dir / f"{base}.svg"
                _prepare_plotly_fig_for_color_export(fig).write_image(str(svg_path), width=1800, height=1100, scale=1, format="svg")
                ok_svg = True
                svg_name = f"svg/{svg_path.name}"
            except Exception:
                ok_svg = False

        index_rows.append({
            "N": i,
            "Grafico": name,
            "HTML": html_name,
            "PNG": png_name,
            "SVG": svg_name,
            "Exportado_HTML": ok_html,
            "Exportado_PNG": ok_png,
            "Exportado_SVG": ok_svg,
        })

    index_df = pd.DataFrame(index_rows)
    index_path = outdir / "indice_graficos_exportados.csv"
    index_df.to_csv(index_path, index=False)

    if png_errors:
        (outdir / "AVISO_PNG.txt").write_text(
            "Algunos PNG/SVG no se han podido generar automáticamente.\n\n"
            "Motivo habitual: falta Kaleido o Chrome/Chromium en Streamlit Cloud.\n\n"
            "Soluciones:\n"
            "1) Añadir kaleido a requirements.txt.\n"
            "2) Mantener exportación HTML, que siempre funciona.\n"
            "3) Usar el script convert_html_to_png.py incluido en este ZIP en tu ordenador local.\n\n"
            f"Gráficos no convertidos: {len(png_errors)}\n",
            encoding="utf-8"
        )

    return index_df


# v15.3.14 · la sesión actual NO persiste segmentaciones durante los reruns.
# Solo se guardan automáticamente fuera del espacio de trabajo actual; en sesión se consolidan
# mediante el botón explícito «Guardar / actualizar snapshot de este análisis».
if _persist_segmentation_edits_v15314:
    try:
        _effective_independent_v152 = st.session_state.get("use_independent_v70", True)
        st.session_state.setdefault("segmentation_signatures_v15213", {})
        for _rec, _data in record_data.items():
            _effective_w = get_record_windows(
                st.session_state.get("global_windows_v50", empty_windows()),
                st.session_state.get("record_windows_v50", {}),
                _rec,
                _effective_independent_v152,
            )
            _sig = hashlib.sha256(repr(_effective_w).encode("utf-8")).hexdigest()
            if st.session_state.segmentation_signatures_v15213.get(_rec) != _sig:
                save_record_segmentation(_rec, _data.get("filename", _rec), _effective_w)
                st.session_state.segmentation_signatures_v15213[_rec] = _sig
    except Exception as _seg_save_error_v152:
        st.sidebar.warning(f"No se pudo guardar alguna segmentación: {_seg_save_error_v152}")

# central calculation
records_results, records_segments, records_valid = {}, {}, {}

global_windows_safe = st.session_state.get("global_windows_v50", empty_windows())
record_windows_safe = st.session_state.get("record_windows_v50", {})
for rec in records:
    record_windows_safe.setdefault(rec, empty_windows())
active_phases = (_hist_cfg_v1537.get('active_phases') if globals().get('_frozen_hist_v1537', False) else None) or st.session_state.get("active_phases_v50", ["Basal"])
use_independent = bool(_hist_cfg_v1537.get('use_independent', st.session_state.get("use_independent_v70", False))) if globals().get('_frozen_hist_v1537', False) else st.session_state.get("use_independent_v70", False)

for rec, data in record_data.items():
    w = get_record_windows(global_windows_safe, record_windows_safe, rec, use_independent)
    _snap = data.get('analysis_snapshot') if data.get('source') == 'historial' else None
    if _snap is not None and _snap.get('results') is not None and not _snap['results'].empty:
        # Historial congelado: se reutiliza exactamente el resultado guardado; no se recalculan filtros/métricas.
        df = _snap['results'].copy()
        segs = {}
        # v15.3.13: records_valid conserva SIEMPRE la misma estructura que en cálculos normales:
        # {fase: bool}. En versiones anteriores los snapshots devolvían una lista de fases,
        # lo que rompía pd.DataFrame(records_valid) y varios .get(...) al recuperar históricos.
        _snapshot_phases = {str(ph) for ph in df.index}
        valid = {ph: (ph in _snapshot_phases) for ph in PHASES}
    elif globals().get('_frozen_hist_v1537', False) and data.get('source') == 'historial':
        # Un histórico sin snapshot no se recalcula silenciosamente con la configuración actual.
        # Se conserva el RRi en el bundle, pero se marca como pendiente de snapshot.
        df = pd.DataFrame()
        segs = {}
        valid = {ph: False for ph in PHASES}
    else:
        df, segs, valid = calculate_record_cached_v15213(
            data["rr"], w, tuple(active_phases), min_rr, include_rqa, include_hvg,
            st.session_state.get("mse_zero_policy", "nan"),
            st.session_state.get("sampen_theiler_window", 0),
            st.session_state.get("mse_radius_mode", "fixed_entropy_sd"),
        )
    # Defensa adicional frente a bases creadas por versiones previas.
    if not isinstance(valid, dict):
        try:
            _valid_set_v15313 = {str(x) for x in valid}
        except Exception:
            _valid_set_v15313 = set()
        valid = {ph: (ph in _valid_set_v15313) for ph in PHASES}
    else:
        valid = {ph: bool(valid.get(ph, False)) for ph in PHASES}
    records_results[rec], records_segments[rec], records_valid[rec] = df, segs, valid

# Persistencia del análisis completo para registros nuevos/activos. Los históricos con snapshot no se sobrescriben.
_snapshot_config_v1536 = {
    'artifact_level': artifact_level, 'min_rr': int(min_rr), 'include_rqa': bool(include_rqa),
    'include_hvg': bool(include_hvg), 'mse_zero_policy': st.session_state.get('mse_zero_policy','nan'),
    'sampen_theiler_window': st.session_state.get('sampen_theiler_window',0),
    'mse_radius_mode': st.session_state.get('mse_radius_mode','fixed_entropy_sd'),
    'domain_method': domain_method, 'active_phases': list(active_phases), 'use_independent': bool(use_independent)
}
# v15.4.0: un histórico NUNCA crea ni actualiza snapshots al abrirse.
# Si un registro heredado carece de snapshot completo, se mantiene como histórico incompleto
# y no se recalculan silenciosamente sus resultados ni su atractor con parámetros actuales.

# v15.3.13 · histórico longitudinal acumulativo por paciente.
# Después de congelar los snapshots de la sesión, actualiza automáticamente el bundle del paciente
# haciendo UNIÓN con lo ya guardado: cargar menos registros nunca reduce su histórico completo.
if _history_source_v1538 == "Sesión actual" and record_data:
    try:
        _bundle_groups_v1539 = {}
        for _rec, _data in record_data.items():
            _pat = infer_patient_id(_rec)
            _rk = _data.get('record_key') or _raw_record_key(_rec, _data.get('filename', _rec))
            _bundle_groups_v1539.setdefault(_pat, []).append(_rk)
        for _pat, _keys in _bundle_groups_v1539.items():
            save_patient_analysis_bundle(_pat, _keys, config={
                'last_session_version': '15.3.28',
                'active_phases': list(active_phases),
                'updated_from_session': True,
            }, include_all_snapshotted=True)
    except Exception as _bundle_error_v1539:
        st.sidebar.warning(f"No se pudo actualizar el histórico longitudinal: {_bundle_error_v1539}")

metrics_df = records_results[selected_record]
long_df = build_long(records_results)
indices_df = build_physiological_indices(long_df)
if indices_df is not None and not indices_df.empty:
    indices_df['Artefactos_pct'] = indices_df['Registro'].map(lambda _r: float((record_data.get(_r,{}).get('artifact_info') or {}).get('percent_artifacts',0.0)) if _r in record_data else np.nan)

# v15.4.0 · Persistencia explícita y MULTIRREGISTRO del análisis final.
# El problema de versiones previas era que el histórico longitudinal podía conservar los RRi
# de todos los registros, pero el usuario debía congelar los resultados uno por uno.
# Ahora un único botón congela TODOS los registros calculados de la sesión y verifica cuántos
# snapshots han quedado realmente persistidos antes de permitir cerrar la aplicación.
def _persist_one_complete_analysis_v15324(_rec, patient_override=None):
    _dsv = record_data[_rec]
    _dfsv = records_results.get(_rec)
    if _dfsv is None or _dfsv.empty:
        return False, "sin resultados válidos"
    _cfgsv = dict(_snapshot_config_v1536)
    _cfgsv['windows'] = get_record_windows(global_windows_safe, record_windows_safe, _rec, use_independent)
    _cfgsv['record_name'] = _rec
    _cfgsv['attractor_snapshot'] = _build_attractor_snapshot_config(_dsv.get('rr'))
    _cfgsv['snapshot_complete'] = True
    _cfgsv['snapshot_scope'] = 'full_record_analysis'
    _rksv = _dsv.get('record_key') or _raw_record_key(_rec, _dsv.get('filename', _rec))
    _ok = save_analysis_snapshot(
        _rksv, _rec, _dfsv, _dsv.get('rr'), _dsv.get('artifact_mask'), _cfgsv
    )
    if not _ok:
        return False, "fallo al escribir snapshot"
    # Consolidar la segmentación EXACTA que produjo esos resultados.
    _wins = get_record_windows(global_windows_safe, record_windows_safe, _rec, use_independent)
    save_record_segmentation(_rec, _dsv.get('filename', _rec), _wins)
    _pending = st.session_state.get('pending_selections_v1522', {}).get(_rec)
    if _pending is not None:
        save_temporal_selection(
            _rec, _dsv.get('filename', _rec), float(_pending[0]), float(_pending[1])
        )
    # v15.4.0: si la sesión se está añadiendo a un histórico existente,
    # asignar este registro a ese paciente ANTES de reconstruir el bundle.
    _bundle_patient = infer_patient_id(patient_override) if patient_override else infer_patient_id(_rec)
    try:
        if patient_override:
            assign_record_to_patient(_rksv, _bundle_patient)
        save_patient_analysis_bundle(
            _bundle_patient, [_rksv],
            config={
                'last_session_version':'15.3.28', 'explicit_save':True, 'complete_snapshot':True,
                'incremental_append': bool(patient_override), 'incremental_target_patient': _bundle_patient
            },
            include_all_snapshotted=True
        )
    except Exception:
        pass
    # Verificación de lectura inmediata: no damos por guardado algo que no se puede recuperar.
    _verify = load_analysis_snapshot(_rksv)
    if _verify is None or _verify.get('results') is None or _verify['results'].empty:
        return False, "snapshot escrito pero no recuperable"
    return True, f"{len(_verify['results'])} fase(s) congelada(s)"

if record_data and not globals().get('_frozen_hist_v1537', False):
    with st.expander("Guardar análisis actual en el historial", expanded=True):
        _append_target_v15325 = st.session_state.get("incremental_target_patient_v15325")
        if _append_target_v15325:
            _prev_bundle_v15325 = list_patient_analysis_bundles()
            _prev_match_v15325 = _prev_bundle_v15325[_prev_bundle_v15325["patient_id"].astype(str).map(infer_patient_id) == infer_patient_id(_append_target_v15325)] if not _prev_bundle_v15325.empty else pd.DataFrame()
            _prev_count_v15325 = int(_prev_match_v15325["record_count"].max()) if not _prev_match_v15325.empty else 0
            st.success(
                f"Añadir a histórico existente: {_append_target_v15325.replace('_',' ').title()} · "
                f"{_prev_count_v15325} registro(s) previos. Solo se guardarán/calcularán los de esta sesión y se anexarán al conjunto."
            )
        st.caption(
            "Congela resultados, configuración, ventanas, RRi corregido y estado del atractor. "
            "Usa «Guardar TODOS» antes de cerrar la app. En modo incremental NO necesitas volver a cargar los registros anteriores."
        )
        _save_targets_v15327 = (
            [_r for _r, _d in record_data.items() if _d.get("source") == "subido"]
            if _append_target_v15325 else list(record_data.keys())
        )
        _valid_to_freeze_v15324 = [
            _r for _r in _save_targets_v15327
            if records_results.get(_r) is not None and not records_results.get(_r).empty
        ]
        _already_frozen_v15324 = 0
        for _r in record_data.keys():
            _d = record_data[_r]
            _rk = _d.get('record_key') or _raw_record_key(_r, _d.get('filename', _r))
            if load_analysis_snapshot(_rk) is not None:
                _already_frozen_v15324 += 1
        st.caption(
            f"Contexto visible: {len(record_data)} registro(s) · nuevos a guardar: {len(_save_targets_v15327)} · "
            f"nuevos con resultados calculados: {len(_valid_to_freeze_v15324)} · "
            f"snapshots recuperables en el contexto: {_already_frozen_v15324}"
        )
        _c_all, _c_one = st.columns([1.25, 1])
        with _c_all:
            if st.button("Guardar TODOS los análisis calculados", key="save_all_snapshots_btn_v15324", type="primary", use_container_width=True):
                _status_rows = []
                _saved = 0
                for _rec in list(_save_targets_v15327):
                    try:
                        _ok, _msg = _persist_one_complete_analysis_v15324(_rec, patient_override=_append_target_v15325)
                    except Exception as _e:
                        _ok, _msg = False, str(_e)
                    _saved += int(_ok)
                    _status_rows.append({'Registro': _rec, 'Guardado': 'Sí' if _ok else 'No', 'Detalle': _msg})
                st.dataframe(pd.DataFrame(_status_rows), use_container_width=True, hide_index=True)
                if _saved == len(_save_targets_v15327):
                    if _append_target_v15325:
                        _after_bundles = list_patient_analysis_bundles()
                        _after_match = _after_bundles[_after_bundles["patient_id"].astype(str).map(infer_patient_id) == infer_patient_id(_append_target_v15325)] if not _after_bundles.empty else pd.DataFrame()
                        _after_count = int(_after_match["record_count"].max()) if not _after_match.empty else _saved
                        st.success(
                            f"Visita añadida correctamente. Histórico de {_append_target_v15325.replace('_',' ').title()}: "
                            f"{_after_count} registro(s) totales. Se han añadido {_saved} registro(s) nuevos de esta sesión sin recalcular los anteriores."
                        )
                    else:
                        st.success(f"Histórico completo guardado: {_saved}/{len(_save_targets_v15327)} registros nuevos con resultados congelados y verificados.")
                elif _saved:
                    st.warning(f"Guardado parcial: {_saved}/{len(_save_targets_v15327)} registros nuevos. Los no guardados aparecen en la tabla.")
                else:
                    st.error("No se ha podido congelar ningún registro. Revisa que existan fases válidas calculadas.")
        with _c_one:
            _save_snapshot_rec_v1537 = st.selectbox("Guardar solo un registro", list(record_data.keys()), key="save_snapshot_rec_v1537")
            if st.button("Guardar registro seleccionado", key="save_snapshot_btn_v1537", use_container_width=True):
                try:
                    _ok, _msg = _persist_one_complete_analysis_v15324(_save_snapshot_rec_v1537, patient_override=_append_target_v15325)
                except Exception as _e:
                    _ok, _msg = False, str(_e)
                if _ok:
                    st.success(f"{_save_snapshot_rec_v1537}: {_msg}.")
                else:
                    st.warning(f"No se pudo guardar {_save_snapshot_rec_v1537}: {_msg}.")

# v14.0: base longitudinal + dataset predictivo + actualización incremental por lotes.
try:
    _saved_obs_v140 = save_indices_to_longitudinal_db(indices_df)
    history_df_v140 = load_longitudinal_observations()
    transitions_df_v140 = build_self_supervised_transitions(history_df_v140)
    prediction_df_v140 = build_v14_prediction_dataset(history_df_v140)
    auto_bundle_v140, auto_status_v140, auto_promoted_v140 = incremental_update_v14(prediction_df_v140)
except Exception as _db_error_v140:
    _saved_obs_v140 = 0
    history_df_v140 = pd.DataFrame()
    transitions_df_v140 = pd.DataFrame()
    prediction_df_v140 = pd.DataFrame()
    auto_bundle_v140 = None
    auto_promoted_v140 = False
    auto_status_v140 = f'Base/motor predictivo no disponible: {_db_error_v140}'

# v13.2: carga automática del modelo persistente y aplicación directa sobre los índices actuales.
if SKLEARN_AVAILABLE and 'gb_bundle_v132' not in st.session_state:
    _auto_bundle = auto_load_active_gradient_boosting_bundle()
    if _auto_bundle is not None:
        st.session_state['gb_bundle_v132'] = _auto_bundle

with tab1:
    st.subheader("Segmentación tipo Kubios · ventanas móviles v15.4.5")
    _hc1, _hc2, _hc3 = st.columns(3)
    _hc1.metric("Registros cargados", len(record_data))
    _hc2.metric("Recuperados del historial", len(loaded_from_history_v152))
    _hc3.metric("Nuevos guardados", len(newly_saved_v152))
    if loaded_from_history_v152:
        st.success("Análisis histórico recuperado: se restauran los resultados guardados sin volver a aplicar filtros ni recalcular métricas.")
    elif not _saved_catalog_v152.empty:
        st.info("Hay registros guardados. Puedes abrirlos desde «Historial longitudinal v15.4.5» sin que interfieran con la segmentación de la sesión actual.")
    if light_mode_v15213:
        st.success("Modo Ligero activo: gráficas RRi reducidas para pantalla, paneles paginados y métricas reutilizadas desde caché.")
    st.write(
        "1) Encuadra una región con el ratón: el tramo se guarda automáticamente. "
        "2) La ventana permanece visible aunque cambies de pestaña o la app se recalcule. "
        "3) Pulsa **Asignar a Basal/E1/E2...** cuando quieras convertirla en fase. "
        "Sólo se calcularán las fases activas."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        view_mode = st.radio("Vista", ["Registro principal", "Todos superpuestos", "Paneles independientes"], index=0)
    with c2:
        if view_mode == "Paneles independientes":
            st.info(
                "Cada registro se muestra en su propio panel. Puedes encuadrar un intervalo distinto "
                "y asignarle una fase independiente antes de comparar los resultados."
            )
        else:
            st.info(
                "En 'Todos superpuestos', el mismo intervalo temporal puede guardarse y asignarse "
                "simultáneamente a todos los registros visibles."
            )

    apply_selection_to_all_v1524 = False
    if view_mode == "Todos superpuestos" and len(records) > 1:
        apply_selection_to_all_v1524 = st.checkbox(
            "Aplicar el tramo seleccionado a todos los registros superpuestos",
            value=True,
            key="apply_selection_to_all_v1524",
            help=(
                "Al asignar una fase, la misma ventana temporal se copia a cada registro cargado. "
                "Si un registro es más corto, el final se ajusta automáticamente a su duración."
            ),
        )
        if apply_selection_to_all_v1524:
            st.caption(f"Destino simultáneo: {len(records)} registros cargados.")

    if view_mode == "Paneles independientes":
        st.markdown("### Selección independiente de fases por registro")
        st.caption(
            "Cada panel mantiene su propia selección temporal y su propia fase. "
            "Las ventanas se guardan en SQLite y se recuperan al volver a abrir la aplicación."
        )

        # Este modo siempre trabaja con ventanas independientes, sin alterar la opción global del resto de pestañas.
        _all_panel_records = list(records)
        if light_mode_v15213:
            _pgc1, _pgc2 = st.columns([1, 2])
            with _pgc1:
                _panel_page_size_choice = st.selectbox(
                    "Registros por página",
                    [1, 2, 4, "Todos"],
                    index=2,
                    key="panel_page_size_v15214",
                    help="La paginación solo limita los paneles visibles; todos los registros siguen cargados.",
                )
            if _panel_page_size_choice == "Todos":
                _panel_page_size = max(1, len(_all_panel_records))
                _panel_pages = 1
                _panel_page = 1
                with _pgc2:
                    st.info(f"Mostrando simultáneamente los {len(_all_panel_records)} registros cargados.")
            else:
                _panel_page_size = int(_panel_page_size_choice)
                _panel_pages = max(1, int(np.ceil(len(_all_panel_records) / _panel_page_size)))
                with _pgc2:
                    _panel_page = st.number_input(
                        "Página", min_value=1, max_value=_panel_pages, value=1, step=1,
                        key="panel_page_v15214"
                    )
            _start = (int(_panel_page) - 1) * int(_panel_page_size)
            _panel_records = _all_panel_records[_start:_start + int(_panel_page_size)]
            st.caption(
                f"Mostrando {_start + 1}–{min(_start + len(_panel_records), len(_all_panel_records))} "
                f"de {len(_all_panel_records)} registros cargados. "
                + (f"Página {int(_panel_page)} de {_panel_pages}." if _panel_pages > 1 else "")
            )
        else:
            _panel_records = _all_panel_records

        _szc1, _szc2 = st.columns(2)
        with _szc1:
            _panel_width_mode = st.radio(
                "Disposición de los registros",
                ["Grande · uno por fila", "Compacta · dos por fila"],
                index=0,
                horizontal=True,
                key="panel_width_mode_v15215",
                help="La disposición grande facilita encuadrar con precisión una fase temporal.",
            )
        with _szc2:
            _panel_height = st.select_slider(
                "Altura de cada gráfica",
                options=[520, 620, 720, 820],
                value=720,
                key="panel_height_v15215",
                format_func=lambda x: f"{x} px",
            )
        st.caption("El eje temporal de cada panel se ajusta ahora a la duración real de ese registro, evitando que la señal quede comprimida.")

        _wc1, _wc2 = st.columns(2)
        with _wc1:
            _target_window_min_v1532 = st.select_slider(
                "Duración objetivo de la ventana",
                options=[1, 2, 3, 4, 5, 6, 8, 10],
                value=5,
                key="target_window_min_v1532",
                format_func=lambda x: f"{x} min",
                help="5 min es el valor recomendado por defecto. Puedes cambiarlo para registros especiales.",
            )
        with _wc2:
            st.info(
                "El contador se actualiza en tiempo real sin recalcular Streamlit. "
                "Doble clic sobre el registro crea directamente la ventana objetivo."
            )
        st.caption("El editor local permite crear, redimensionar y desplazar la ventana completa sin ejecutar los cálculos hasta soltar el ratón.")
        if (_history_source_v1538 == "Sesión actual" and
                globals().get("_session_seg_mode_v15314", "Nueva segmentación (editable)") == "Nueva segmentación (editable)"):
            st.caption("Nueva segmentación: ninguna ventana se crea automáticamente. La duración de 5 min es solo una referencia; crea la ventana con el ratón o doble clic.")
            if st.button("Reiniciar ventanas de esta sesión", key="reset_current_segmentation_v15315"):
                st.session_state["global_windows_v50"] = empty_windows()
                st.session_state["record_windows_v50"] = {r: empty_windows() for r in records}
                st.session_state["pending_selections_v1522"] = {}
                st.session_state["pending_selection_v50"] = None
                st.session_state["active_phases_v50"] = []
                st.session_state["restored_records_v152"] = list(records)
                st.session_state["seg_editor_epoch_v15315"] = int(st.session_state.get("seg_editor_epoch_v15315", 1)) + 1
                st.rerun()

        _panel_cols = ([st.container()] if _panel_width_mode.startswith("Grande")
                       else (st.columns(2) if len(_panel_records) > 1 else [st.container()]))

        # v15.3.15: token/clave de editor aislados por generación de sesión y modo.
        # En "Nueva segmentación" jamás se acepta un evento emitido por una instancia anterior.
        _seg_epoch_v15315 = int(st.session_state.get("seg_editor_epoch_v15315", 1))
        _seg_mode_token_v15315 = ("new" if globals().get("_session_seg_mode_v15314", "Nueva segmentación (editable)").startswith("Nueva") else "restore")

        for _idx, _rec in enumerate(_panel_records):
            with _panel_cols[_idx % len(_panel_cols)]:
                st.markdown(f"#### {_rec}")
                _pending_rec = st.session_state.pending_selections_v1522.get(_rec)
                _rec_hash = hashlib.md5(_rec.encode()).hexdigest()[:10]
                _editor_event = window_editor_v1533(
                    _rec,
                    record_data[_rec],
                    pending_selection=_pending_rec,
                    target_s=float(_target_window_min_v1532) * 60.0,
                    height=_panel_height,
                    max_points=max_plot_points_v15213 if light_mode_v15213 else 3000,
                    editor_token=f"{_seg_epoch_v15315}:{_seg_mode_token_v15315}:{_rec_hash}",
                    key=f"window_editor_v15315_{_seg_epoch_v15315}_{_seg_mode_token_v15315}_{_rec_hash}",
                )
                _expected_editor_token_v15315 = f"{_seg_epoch_v15315}:{_seg_mode_token_v15315}:{_rec_hash}"
                if isinstance(_editor_event, dict) and str(_editor_event.get("editor_token", "")) == _expected_editor_token_v15315:
                    try:
                        _s_new = float(_editor_event.get("start_s"))
                        _e_new = float(_editor_event.get("end_s"))
                        _dur = float(record_data[_rec].get("duration", 0.0))
                        _s_new = max(0.0, min(_s_new, _dur))
                        _e_new = max(0.0, min(_e_new, _dur))
                        if _e_new > _s_new:
                            _new_sel = [_s_new, _e_new]
                            _old_sel = st.session_state.pending_selections_v1522.get(_rec)
                            if _old_sel is None or not np.allclose(_old_sel, _new_sel, atol=0.20):
                                st.session_state.pending_selections_v1522[_rec] = _new_sel
                                st.session_state[f"raw_mouse_duration_v1532_{_rec_hash}"] = float(_e_new - _s_new)
                                if _persist_segmentation_edits_v15314:
                                    save_temporal_selection(_rec, record_data[_rec].get("filename", _rec), _s_new, _e_new)
                                st.rerun()
                    except Exception:
                        pass

                _pending_rec = st.session_state.pending_selections_v1522.get(_rec)
                if _pending_rec is not None:
                    _s, _e = _pending_rec
                    _sel_duration = float(_e - _s)
                    st.success(
                        f"Selección activa: {sec_to_hms(_s)} – {sec_to_hms(_e)} "
                        f"(duración {sec_to_hms(_sel_duration)})"
                    )
                    _raw_key = f"raw_mouse_duration_v1532_{hashlib.md5(_rec.encode()).hexdigest()[:10]}"
                    if _raw_key in st.session_state:
                        _raw_mouse_d = float(st.session_state[_raw_key])
                        st.caption(
                            f"Última edición confirmada: duración {sec_to_hms(_raw_mouse_d)}. "
                            "El contador del gráfico se actualiza continuamente mientras mueves el ratón."
                        )

                    st.caption(
                        "Editor v15.3.3: el contador mm:ss se actualiza dentro del gráfico mientras arrastras. "
                        "Arrastra dentro de la ventana para moverla en bloque, o sus bordes para redimensionarla. "
                        f"Doble clic crea una ventana exacta de {int(_target_window_min_v1532)} min centrada en ese punto."
                    )

                    _pc1, _pc2 = st.columns([2, 1])
                    with _pc1:
                        _phase_rec = st.selectbox(
                            "Fase para este registro",
                            PHASES,
                            key=f"phase_independent_{hashlib.md5(_rec.encode()).hexdigest()[:10]}",
                        )
                    with _pc2:
                        st.write("")
                        st.write("")
                        if st.button(
                            "Asignar",
                            key=f"assign_independent_{hashlib.md5(_rec.encode()).hexdigest()[:10]}",
                            use_container_width=True,
                        ):
                            st.session_state.record_windows_v50.setdefault(_rec, empty_windows())
                            st.session_state.record_windows_v50[_rec][_phase_rec] = [float(_s), float(_e)]
                            if _persist_segmentation_edits_v15314:
                                if _persist_segmentation_edits_v15314:
                                    save_record_segmentation(
                                        _rec,
                                        record_data[_rec].get("filename", _rec),
                                        get_record_windows(
                                            st.session_state.global_windows_v50,
                                            st.session_state.record_windows_v50,
                                            _rec,
                                            True,
                                        ),
                                    )
                            if _phase_rec not in st.session_state.active_phases_v50:
                                st.session_state.active_phases_v50.append(_phase_rec)
                            st.session_state[f"assigned_notice_{_rec}"] = _phase_rec
                            st.rerun()

                    _notice = st.session_state.pop(f"assigned_notice_{_rec}", None)
                    if _notice:
                        st.success(f"{_notice} guardada para este registro.")

                    if st.button(
                        "Eliminar selección",
                        key=f"clear_independent_{hashlib.md5(_rec.encode()).hexdigest()[:10]}",
                    ):
                        st.session_state.pending_selections_v1522.pop(_rec, None)
                        if _persist_segmentation_edits_v15314:
                            clear_temporal_selection(
                                _rec, record_data[_rec].get("filename", _rec)
                            )
                        st.rerun()
                else:
                    st.info("Arrastra un recuadro sobre este registro para definir su intervalo.")

        st.info(
            "Después de asignar las fases independientes, utiliza la pestaña Comparar. "
            "Cada registro conservará la ventana que hayas definido aquí."
        )

    else:
        fig = rr_plot(
            record_data,
            st.session_state.global_windows_v50,
            st.session_state.record_windows_v50,
            view_mode,
            selected_record,
            use_independent,
            pending_selection=st.session_state.pending_selections_v1522.get(selected_record),
            max_display_points=max_plot_points_v15213 if light_mode_v15213 else 5000,
        )

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=("box", "lasso"),
            key="rr_select_v50",
        )

        if event and getattr(event, "selection", None):
            pts = event.selection.get("points", [])
            xs = [p.get("x") for p in pts if p.get("x") is not None]

            if xs:
                s_sel, e_sel = min(xs) * 60, max(xs) * 60
                if e_sel > s_sel:
                    # v15.2.4: guardar inmediatamente. En vista superpuesta puede
                    # propagarse el mismo intervalo a todos los registros visibles.
                    _selection_targets = (
                        list(records)
                        if view_mode == "Todos superpuestos" and apply_selection_to_all_v1524
                        else [selected_record]
                    )
                    _selection_changed = False
                    for _rec in _selection_targets:
                        _rec_duration = float(record_data[_rec].get("duration", 0.0))
                        _rec_start = max(0.0, min(float(s_sel), _rec_duration))
                        _rec_end = max(0.0, min(float(e_sel), _rec_duration))
                        if _rec_end <= _rec_start:
                            continue
                        _new_sel = [_rec_start, _rec_end]
                        _old_sel = st.session_state.pending_selections_v1522.get(_rec)
                        if _old_sel is None or not np.allclose(_old_sel, _new_sel, atol=0.25):
                            st.session_state.pending_selections_v1522[_rec] = _new_sel
                            if _persist_segmentation_edits_v15314:
                                save_temporal_selection(
                                    _rec,
                                    record_data[_rec].get("filename", _rec),
                                    _rec_start,
                                    _rec_end,
                                )
                            _selection_changed = True

                    st.session_state.pending_selection_v50 = (
                        st.session_state.pending_selections_v1522.get(selected_record)
                    )
                    if _selection_changed:
                        st.rerun()

        _pending_current_v1522 = st.session_state.pending_selections_v1522.get(selected_record)
        st.session_state.pending_selection_v50 = _pending_current_v1522

        if _pending_current_v1522 is not None:
            s_sel, e_sel = _pending_current_v1522
            _duration_sel = e_sel - s_sel
            st.success(
                f"Tramo temporal guardado automáticamente: "
                f"{sec_to_hms(s_sel)} - {sec_to_hms(e_sel)} "
                f"(duración {sec_to_hms(_duration_sel)})"
            )
            st.caption(
                "La zona amarilla discontinua es la selección temporal activa. "
                "Se conserva al recalcular, cambiar de pestaña y volver a abrir la aplicación."
            )

            st.markdown("### Asignar tramo guardado a fase")
            phase_cols = st.columns(10)

            for idx, ph in enumerate(PHASES):
                with phase_cols[idx % 10]:
                    if st.button(ph, key=f"assign_{ph}_v1524"):
                        _assignment_targets = (
                            list(records)
                            if view_mode == "Todos superpuestos" and apply_selection_to_all_v1524
                            else [selected_record]
                        )
                        _assigned_records = []
                        _skipped_records = []

                        if use_independent:
                            for _rec in _assignment_targets:
                                _rec_duration = float(record_data[_rec].get("duration", 0.0))
                                # Preferir la selección persistente individual ya propagada.
                                _rec_sel = st.session_state.pending_selections_v1522.get(_rec, [s_sel, e_sel])
                                _rs = max(0.0, min(float(_rec_sel[0]), _rec_duration))
                                _re = max(0.0, min(float(_rec_sel[1]), _rec_duration))
                                if _re <= _rs:
                                    _skipped_records.append(_rec)
                                    continue

                                st.session_state.record_windows_v50.setdefault(_rec, empty_windows())
                                st.session_state.record_windows_v50[_rec][ph] = [_rs, _re]
                                if _persist_segmentation_edits_v15314:
                                    save_record_segmentation(
                                        _rec,
                                        record_data[_rec].get("filename", _rec),
                                        get_record_windows(
                                            st.session_state.global_windows_v50,
                                            st.session_state.record_windows_v50,
                                            _rec,
                                            True,
                                        ),
                                    )
                                _assigned_records.append(_rec)
                        else:
                            # Con ventanas globales, una única asignación ya se aplica a todos.
                            st.session_state.global_windows_v50[ph] = [s_sel, e_sel]
                            _assigned_records = list(_assignment_targets)
                            for _rec in _assignment_targets:
                                if _persist_segmentation_edits_v15314:
                                    save_record_segmentation(
                                        _rec,
                                        record_data[_rec].get("filename", _rec),
                                        get_record_windows(
                                            st.session_state.global_windows_v50,
                                            st.session_state.record_windows_v50,
                                            _rec,
                                            False,
                                        ),
                                    )

                        if ph not in st.session_state.active_phases_v50:
                            st.session_state.active_phases_v50.append(ph)

                        st.session_state["last_multirecord_assignment_v1524"] = {
                            "phase": ph,
                            "assigned": _assigned_records,
                            "skipped": _skipped_records,
                        }
                        st.rerun()

            _last_assign_v1524 = st.session_state.pop("last_multirecord_assignment_v1524", None)
            if _last_assign_v1524:
                _n_assigned = len(_last_assign_v1524.get("assigned", []))
                _phase_assigned = _last_assign_v1524.get("phase", "")
                if _n_assigned:
                    st.success(
                        f"Fase {_phase_assigned} asignada simultáneamente a {_n_assigned} registro(s)."
                    )
                if _last_assign_v1524.get("skipped"):
                    st.warning(
                        "No se pudo aplicar la ventana a: "
                        + ", ".join(_last_assign_v1524["skipped"])
                        + ". El intervalo queda fuera de la duración de esos registros."
                    )

            _sc1, _sc2 = st.columns(2)
            with _sc1:
                if st.button("Eliminar tramo temporal guardado", key="clear_pending_v1522"):
                    st.session_state.pending_selections_v1522.pop(selected_record, None)
                    st.session_state.pending_selection_v50 = None
                    if _persist_segmentation_edits_v15314:
                        clear_temporal_selection(
                            selected_record,
                            record_data[selected_record].get("filename", selected_record),
                        )
                    st.rerun()
            with _sc2:
                st.info("Para sustituirlo, arrastra directamente un nuevo cuadro sobre el registro.")

    st.markdown("### Ventanas definidas")
    win_df = windows_table(
        st.session_state.global_windows_v50,
        st.session_state.record_windows_v50,
        records,
        record_data,
        records_segments,
        records_valid,
        use_independent,
    )
    st.dataframe(win_df, use_container_width=True)

    st.markdown("### Edición manual opcional")
    manual_phase = st.selectbox("Fase a editar manualmente", PHASES)
    current_w = get_record_windows(st.session_state.global_windows_v50, st.session_state.record_windows_v50, selected_record, use_independent).get(manual_phase)

    if current_w is None:
        ini_default, fin_default = "00:00:00", "00:05:00"
    else:
        ini_default, fin_default = sec_to_hms(current_w[0]), sec_to_hms(current_w[1])

    c_ini, c_fin, c_apply, c_clear = st.columns([1, 1, 1, 1])
    with c_ini:
        ini_txt = st.text_input("Inicio", ini_default)
    with c_fin:
        fin_txt = st.text_input("Fin", fin_default)
    with c_apply:
        st.write("")
        st.write("")
        if st.button("Aplicar manual"):
            try:
                s, e = hms_to_sec(ini_txt), hms_to_sec(fin_txt)
                if e <= s:
                    st.warning("El final debe ser mayor que el inicio.")
                else:
                    if use_independent:
                        st.session_state.record_windows_v50[selected_record][manual_phase] = [s, e]
                    else:
                        st.session_state.global_windows_v50[manual_phase] = [s, e]
                    if manual_phase not in st.session_state.active_phases_v50:
                        st.session_state.active_phases_v50.append(manual_phase)
                    st.rerun()
            except Exception:
                st.warning("Formato no válido. Usa HH:MM:SS.")
    with c_clear:
        st.write("")
        st.write("")
        if st.button("Borrar fase"):
            if use_independent:
                st.session_state.record_windows_v50[selected_record][manual_phase] = None
            else:
                st.session_state.global_windows_v50[manual_phase] = None
            if manual_phase in st.session_state.active_phases_v50:
                st.session_state.active_phases_v50.remove(manual_phase)
            st.rerun()

with tab2:
    st.subheader(f"HRV: {selected_record}")

    if metrics_df.empty:
        st.info("No hay ventanas válidas para el registro principal. Define ventanas, activa fases o baja el mínimo RRi.")
    else:
        st.markdown("### Resumen visual por fases")
        st.caption("Columnas verticales = valores por fase. Líneas = tendencia suavizada superpuesta.")
        st.plotly_chart(
            hrv_phase_summary_figure(metrics_df),
            use_container_width=True,
            key=f"hrv_summary_{selected_record}"
        )

        with st.expander("Definiciones, fórmulas y referencias interpretativas de los nuevos métodos", expanded=False):
            st.markdown(advanced_methods_reference_markdown())

        st.markdown("### Wavelet/STFT scalogram: cambios transitorios LF/HF")
        st.caption(
            "Este gráfico muestra cómo cambia la potencia por frecuencia a lo largo del tiempo. "
            "Permite ver cuándo aparece o desaparece HF, cuándo emerge LF y si hay transiciones dentro de la ventana."
        )
        scalogram_windows = get_record_windows(
            st.session_state.global_windows_v50,
            st.session_state.record_windows_v50,
            selected_record,
            use_independent,
        )
        st.plotly_chart(
            wavelet_scalogram_figure(
                record_data[selected_record]["rr"],
                windows=scalogram_windows,
                title=f"Scalogram LF/HF · {selected_record}"
            ),
            use_container_width=True,
            key=f"wavelet_scalogram_{selected_record}"
        )

        wave_cols_ref = [
            "VLF_DOM_PCT","LF_DOM_PCT","HF_DOM_PCT",
            "WAV_TRANSITIONS_N","WAV_TRANSITIONS_PER_MIN",
            "WAV_ENTROPY_BANDS","WAV_ENTROPY_GLOBAL"
        ]
        wave_present_ref = [c for c in wave_cols_ref if c in metrics_df.columns]
        if wave_present_ref:
            st.markdown("### Interpretación wavelet automática")
            rows_ref = []
            for ph, row in metrics_df.iterrows():
                for c in wave_present_ref:
                    rows_ref.append({
                        "Fase": ph,
                        "Métrica": c,
                        "Valor": row.get(c, np.nan),
                        "Referencia": _interpret_metric(c, row.get(c, np.nan)),
                    })
            st.dataframe(pd.DataFrame(rows_ref), use_container_width=True)

        for group, cols in PARAM_GROUPS.items():
            present = [c for c in cols if c in metrics_df.columns]
            if present:
                st.markdown(f"### {group}")
                st.dataframe(metrics_df[present], use_container_width=True)

                with st.expander(f"Referencia · valor obtenido · interpretación · {group}", expanded=False):
                    ref_df = reference_interpretation_table(metrics_df, phase=("Basal" if "Basal" in metrics_df.index else metrics_df.index[0]), metrics=present)
                    st.dataframe(ref_df, use_container_width=True)

        st.markdown("### Tabla global: referencia, valor obtenido e interpretación")
        phase_ref_global = st.selectbox(
            "Fase para tabla global de interpretación",
            list(metrics_df.index),
            index=list(metrics_df.index).index("Basal") if "Basal" in metrics_df.index else 0,
            key="global_reference_phase_v119"
        )
        st.dataframe(reference_interpretation_table(metrics_df, phase=phase_ref_global), use_container_width=True)

        if "Lyapunov_LLE" in metrics_df.columns:
            st.markdown("### Interpretación orientativa Lyapunov_LLE")
            lyap_rows = []
            for ph, row in metrics_df.iterrows():
                val = row.get("Lyapunov_LLE", np.nan)
                lyap_rows.append({
                    "Fase": ph,
                    "Lyapunov_LLE": val,
                    "Interpretación": lyapunov_interpretation(val),
                })
            st.dataframe(pd.DataFrame(lyap_rows), use_container_width=True)

with tab3:
    st.subheader("Comparar registros")

    if len(records) < 2:
        st.info("Sube dos o más registros.")
    elif long_df.empty:
        st.info("No hay datos comparables. Define ventanas, activa fases o baja el mínimo RRi.")
    else:
        valid_summary = pd.DataFrame.from_dict(records_valid, orient="index").reindex(columns=PHASES).fillna(False).astype(bool)
        st.markdown("### Ventanas válidas")
        st.dataframe(valid_summary, use_container_width=True)

        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
        selected_phases = st.multiselect("Fases a comparar", PHASES, default=available_phases)
        numeric_vars = [c for c in long_df.columns if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])]

        default_var = "RMSSD" if "RMSSD" in numeric_vars else numeric_vars[0]
        df_sel = long_df[long_df["Fase"].isin(selected_phases)] if selected_phases else long_df

        st.markdown("### Variables principales: comparación múltiple")
        st.caption(
            "Selecciona una o varias variables. Cada variable se representa en su propio panel, "
            "igual que en el panel inferior. También puedes cargar grupos completos o seleccionar todos."
        )

        main_state_key = "compare_main_vars_v1528"
        if main_state_key not in st.session_state:
            st.session_state[main_state_key] = [default_var]
        else:
            st.session_state[main_state_key] = [
                p for p in st.session_state[main_state_key] if p in numeric_vars
            ]
            if not st.session_state[main_state_key] and default_var in numeric_vars:
                st.session_state[main_state_key] = [default_var]

        group_options = {
            group: [p for p in cols if p in numeric_vars]
            for group, cols in PARAM_GROUPS.items()
        }
        group_options = {k: v for k, v in group_options.items() if v}

        c_all_main, c_rec_main, c_clear_main, c_group_main = st.columns([1.15, 1.3, 0.9, 2.65])
        with c_all_main:
            if st.button("Seleccionar todos", key="compare_main_select_all_v1528", use_container_width=True):
                st.session_state[main_state_key] = list(numeric_vars)
                st.rerun()
        with c_rec_main:
            recommended_main = [p for p in DEFAULT_MULTI if p in numeric_vars]
            if st.button("Selección recomendada", key="compare_main_recommended_v1528", use_container_width=True):
                st.session_state[main_state_key] = recommended_main or [default_var]
                st.rerun()
        with c_clear_main:
            if st.button("Limpiar", key="compare_main_clear_v1528", use_container_width=True):
                st.session_state[main_state_key] = []
                st.rerun()
        with c_group_main:
            group_col_select, group_col_button = st.columns([2.2, 0.8])
            with group_col_select:
                selected_group_main = st.selectbox(
                    "Añadir grupo de variables",
                    ["— Selecciona un grupo —"] + list(group_options.keys()),
                    key="compare_main_group_selector_v1528",
                )
            with group_col_button:
                st.write("")
                st.write("")
                add_group_main = st.button(
                    "Añadir",
                    key="compare_main_add_group_v1528",
                    use_container_width=True,
                    disabled=(selected_group_main == "— Selecciona un grupo —"),
                )

        if add_group_main and selected_group_main != "— Selecciona un grupo —":
            group_cols = group_options.get(selected_group_main, [])
            merged = list(dict.fromkeys(st.session_state[main_state_key] + group_cols))
            st.session_state[main_state_key] = merged
            st.rerun()

        st.caption(
            f"{len(st.session_state[main_state_key])} de {len(numeric_vars)} variables principales seleccionadas"
        )
        main_variables = st.multiselect(
            "Variables principales",
            numeric_vars,
            key=main_state_key,
            placeholder="Elige una o varias variables",
        )

        if main_variables:
            phases_for_compare = selected_phases or available_phases
            st.caption(
                "Cada variable conserva el formato comparativo original: las fases aparecen en el eje horizontal, "
                "los registros se muestran como barras agrupadas y cada registro mantiene su línea de tendencia."
            )

            # Renderizar cada variable como una comparación real entre fases y registros,
            # no como una serie cronológica aplanada.
            for idx_var, variable in enumerate(main_variables):
                pivot = (
                    df_sel.pivot_table(
                        index="Fase",
                        columns="Registro",
                        values=variable,
                        aggfunc="first",
                    )
                    .reindex(phases_for_compare)
                )
                # Eliminar únicamente filas/columnas completamente vacías.
                pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
                st.markdown(f"#### {variable}: comparación entre fases")
                if pivot.empty:
                    st.info(f"No hay datos válidos para {variable} en las fases seleccionadas.")
                else:
                    st.plotly_chart(
                        comparison_bar_line(pivot, variable),
                        use_container_width=True,
                        key=f"compare_main_phase_variable_v1528_{idx_var}_{variable}",
                    )

            with st.expander("Ver tablas de valores por variable", expanded=False):
                for variable in main_variables:
                    pivot = (
                        df_sel.pivot_table(
                            index="Fase",
                            columns="Registro",
                            values=variable,
                            aggfunc="first",
                        )
                        .reindex(phases_for_compare)
                    )
                    st.markdown(f"#### {variable}")
                    st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Selecciona al menos una variable principal para generar la comparación.")

        st.markdown("### Panel de varios parámetros: barras + línea suavizada")
        st.caption("Selecciona varios parámetros simultáneamente o usa «Seleccionar todos». Cada parámetro se muestra en su propio panel.")
        param_defaults = [p for p in DEFAULT_MULTI if p in numeric_vars]
        state_key = "compare_params_v1526"
        if state_key not in st.session_state:
            st.session_state[state_key] = param_defaults
        else:
            # Elimina valores que ya no estén disponibles tras cambiar filtros o registros.
            st.session_state[state_key] = [p for p in st.session_state[state_key] if p in numeric_vars]

        c_all, c_default, c_clear, c_info = st.columns([1.25, 1.35, 1.0, 3.4])
        with c_all:
            if st.button("Seleccionar todos", key="compare_select_all_v1526", use_container_width=True):
                st.session_state[state_key] = list(numeric_vars)
                st.rerun()
        with c_default:
            if st.button("Selección recomendada", key="compare_select_default_v1526", use_container_width=True):
                st.session_state[state_key] = list(param_defaults)
                st.rerun()
        with c_clear:
            if st.button("Limpiar", key="compare_clear_all_v1526", use_container_width=True):
                st.session_state[state_key] = []
                st.rerun()
        with c_info:
            st.caption(f"{len(st.session_state[state_key])} de {len(numeric_vars)} parámetros seleccionados")

        params = st.multiselect(
            "Parámetros del panel",
            numeric_vars,
            key=state_key,
            placeholder="Elige uno o varios parámetros",
        )
        if params:
            st.plotly_chart(
                dashboard_bar_smooth(long_df, selected_phases or available_phases, params),
                use_container_width=True,
                key="compare_dashboard_params_smooth_v1526",
            )
        else:
            st.info("Selecciona al menos un parámetro para generar el panel comparativo.")

        ph_overlay = st.selectbox("RRi superpuesto por fase", selected_phases or available_phases)
        st.plotly_chart(
            phase_rr_overlay(record_data, st.session_state.global_windows_v50, st.session_state.record_windows_v50, ph_overlay, use_independent),
            use_container_width=True,
            key=f"phase_overlay_{ph_overlay}",
        )

        st.markdown("### Tabla completa filtrada")
        st.dataframe(df_sel, use_container_width=True)



with tab4:
    st.subheader("Parámetros no lineales: dominios, MSE y MDE")
    with st.expander("Guía rápida: SampEn, MSE, DispEn, MDE, fractales y recurrencia", expanded=False):
        st.markdown(advanced_methods_reference_markdown())

    if metrics_df.empty:
        st.info("No hay ventanas válidas para mostrar dominios o MSE.")
    else:
        st.markdown("### Dominios Amplitud / Vagal / Complejidad / Recurrencia")
        st.caption("Normalizado a Basal = 100%. Amplitud: SDNN, SD2, Total Power. Vagal: RMSSD, SD1, HF, pNN50. Complejidad: DFA α1, DFA α2, ApEn, SampEn. Recurrencia: REC, DET, Lmean, Lmax, ShanEn.")
        st.plotly_chart(
            domains_figure(metrics_df, method=domain_method, title=f"Dominios · {selected_record}"),
            use_container_width=True,
            key="domains_principal"
        )
        st.dataframe(domain_values(metrics_df, method=domain_method), use_container_width=True)

        st.markdown("### MSE 1-20 del registro principal")
        st.plotly_chart(
            mse_figure(metrics_df, title=f"MSE 1-20 · {selected_record}"),
            use_container_width=True,
            key="mse_principal"
        )

    if not long_df.empty and len(records) >= 2:
        st.markdown("### Comparativa MSE 1-20 entre registros")
        available_phases_mse = [p for p in PHASES if p in long_df["Fase"].unique()]
        phases_mse = st.multiselect("Fases para comparar MSE", PHASES, default=available_phases_mse, key="mse_compare_phases")
        scale_range = st.slider("Escalas MSE", 1, 20, (1, 20), key="mse_scale_range")
        scales = list(range(scale_range[0], scale_range[1] + 1))
        st.plotly_chart(
            mse_compare_figure(long_df, phases_mse or available_phases_mse, scales=scales),
            use_container_width=True,
            key="mse_compare"
        )



    st.markdown("---")
    st.markdown("### Atractores / Dinámica reconstruida · comparación multirregistro")
    st.caption(
        "Reconstrucción del espacio de estados directamente desde RRi. Puedes comparar varios registros a la vez, "
        "incluidos todos los registros cargados. La clasificación es exploratoria y no identifica un atractor extraño "
        "a partir de una sola métrica o de la apariencia visual."
    )

    _attr_key = "attractor_records_v15320"
    if _attr_key not in st.session_state:
        st.session_state[_attr_key] = [selected_record] if selected_record in records else (records[:1] if records else [])

    _ba1, _ba2, _ba3 = st.columns([1, 1, 2])
    with _ba1:
        if st.button("Seleccionar todos", key="attr_select_all_v15320", use_container_width=True):
            st.session_state[_attr_key] = list(records)
            st.rerun()
    with _ba2:
        if st.button("Limpiar", key="attr_clear_v15320", use_container_width=True):
            st.session_state[_attr_key] = []
            st.rerun()
    with _ba3:
        st.caption(f"{len(st.session_state.get(_attr_key, []))} de {len(records)} registros seleccionados")

    _attr_records = st.multiselect(
        "Registros para reconstruir/comparar",
        options=list(records),
        key=_attr_key,
        help="Selecciona varios registros o pulsa «Seleccionar todos». Cada registro conserva su propio retardo automático si eliges ese modo.",
    )

    if _attr_records:
        _ac1, _ac2, _ac3, _ac4 = st.columns(4)
        with _ac1:
            _tau_mode = st.selectbox(
                "Retardo τ",
                ["Automático por registro", "Común manual"],
                key="attractor_tau_mode_v15320",
            )
        with _ac2:
            _tau_common = st.number_input(
                "τ común (latidos)", min_value=1, max_value=200, value=1, step=1,
                disabled=_tau_mode != "Común manual", key="attractor_tau_common_v15320"
            )
        with _ac3:
            _m = st.selectbox("Dimensión de embedding", [2,3,4,5,6], index=1, key="attractor_m_v15320")
        with _ac4:
            _maxpts = st.select_slider(
                "Puntos por registro", options=[300,500,1000,2000,5000], value=1000, key="attractor_pts_v15320"
            )

        _view_mode = st.radio(
            "Visualización de atractores",
            ["Paneles independientes", "Comparación superpuesta"],
            index=0,
            horizontal=True,
            key="attractor_view_mode_v15322",
            help="Paneles independientes muestra cada atractor en su propia gráfica. Comparación superpuesta conserva la vista conjunta anterior."
        )
        _panel_layout = st.selectbox(
            "Disposición de paneles independientes",
            ["1 por fila", "2 por fila"],
            index=1 if len(_attr_records) > 1 else 0,
            key="attractor_panel_layout_v15322",
            disabled=_view_mode != "Paneles independientes",
        )

        if any(record_data.get(r, {}).get("source") == "historial" for r in _attr_records):
            st.caption("Históricos reproducibles: si el snapshot conserva el estado del atractor, PhysioSentinel usa exactamente la serie RRi corregida, τ y m guardados. Los snapshots heredados incompletos no se recalculan.")

        _attr_rows=[]
        _fig3 = go.Figure()
        _fig2 = go.Figure()
        _valid_attr = 0
        _attr_cache_v15323 = {}
        for _rec in _attr_records:
            _rd = record_data.get(_rec, {})
            _rr = np.asarray(_rd.get("rr", []), dtype=float)
            _rr = _rr[np.isfinite(_rr)]
            _snap = _rd.get("analysis_snapshot") if _rd.get("source") == "historial" else None
            _snap_attr = ((_snap or {}).get("config") or {}).get("attractor_snapshot") if _snap else None
            _hist_incomplete = bool(_rd.get("source") == "historial" and not _snap_attr)
            if _hist_incomplete:
                _attr_rows.append({"Registro": _rec, "τ": np.nan, "m": np.nan, "Descriptor": "Histórico no reproducible", "Evidencia": "Snapshot heredado incompleto: no se recalcula el atractor."})
                _attr_cache_v15323[_rec] = {"ok": False, "reason": "snapshot_incomplete"}
                continue
            if len(_rr) < 10:
                _attr_rows.append({"Registro": _rec, "τ": np.nan, "m": int(_m), "Descriptor": "Insuficiente", "Evidencia": "Muy pocos RRi"})
                _attr_cache_v15323[_rec] = {"ok": False, "reason": "short"}
                continue
            if _snap_attr:
                try:
                    _tau = int(_snap_attr.get("tau"))
                    _m_rec = int(_snap_attr.get("m", 3))
                except Exception:
                    _attr_rows.append({"Registro": _rec, "τ": np.nan, "m": np.nan, "Descriptor": "Histórico no reproducible", "Evidencia": "Parámetros de atractor incompletos en snapshot."})
                    _attr_cache_v15323[_rec] = {"ok": False, "reason": "params_incomplete"}
                    continue
                _expected_hash = _snap_attr.get("rr_sha256")
                _actual_hash = hashlib.sha256(np.ascontiguousarray(_rr, dtype=np.float64).tobytes()).hexdigest()
                if _expected_hash and _expected_hash != _actual_hash:
                    _attr_rows.append({"Registro": _rec, "τ": _tau, "m": _m_rec, "Descriptor": "Histórico no reproducible", "Evidencia": "La serie RRi recuperada no coincide con la serie congelada."})
                    _attr_cache_v15323[_rec] = {"ok": False, "reason": "rr_hash_mismatch"}
                    continue
                _verified = True
            else:
                _m_rec = int(_m)
                _tau = int(_attractor_delay(_rr)) if _tau_mode == "Automático por registro" else int(_tau_common)
                _tau = max(1, min(_tau, max(1, len(_rr)//max(3,_m_rec))))
                _verified = False
            _emb = _delay_embedding(_rr, _tau, _m_rec)
            _desc, _evidence = _attractor_descriptor(_rr, _emb)
            if _verified:
                _evidence = "Atractor histórico verificado ✓ · " + str(_evidence)
            _attr_rows.append({"Registro": _rec, "τ": _tau, "m": _m_rec, "Descriptor": _desc, "Evidencia": _evidence})
            _attr_cache_v15323[_rec] = {"ok": len(_emb) > 0, "rr": _rr, "tau": _tau, "m": _m_rec, "emb": _emb, "verified": _verified}
            if len(_emb) == 0:
                continue
            _valid_attr += 1
            _show = _emb if len(_emb) <= _maxpts else _emb[np.linspace(0, len(_emb)-1, int(_maxpts)).astype(int)]
            if _m_rec >= 3:
                _fig3.add_trace(go.Scatter3d(
                    x=_show[:,0], y=_show[:,1], z=_show[:,2], mode="lines", name=_short_record_label(_rec, 28),
                    hovertemplate=("<b>"+str(_rec)+"</b><br>RR(t)=%{x:.1f}<br>RR(t-τ)=%{y:.1f}<br>RR(t-2τ)=%{z:.1f}<extra></extra>")
                ))
            _fig2.add_trace(go.Scatter(
                x=_show[:,0], y=_show[:,1], mode="lines", name=_short_record_label(_rec, 28),
                hovertemplate=("<b>"+str(_rec)+"</b><br>RR(t)=%{x:.1f}<br>RR(t-τ)=%{y:.1f}<extra></extra>")
            ))

        if _attr_rows:
            st.dataframe(pd.DataFrame(_attr_rows), use_container_width=True, hide_index=True)

        if _valid_attr:
            if _view_mode == "Comparación superpuesta":
                if any((v.get("ok") and int(v.get("m", 0)) >= 3) for v in _attr_cache_v15323.values()):
                    _fig3.update_layout(
                        title="Comparación superpuesta de atractores reconstruidos · 3D",
                        height=680,
                        scene=dict(xaxis_title="RR(t)", yaxis_title="RR(t-τ)", zaxis_title="RR(t-2τ)"),
                        legend_title="Registro"
                    )
                    st.plotly_chart(_fig3, use_container_width=True, key="attractor_compare_3d_v15322")
                _fig2.update_layout(
                    title="Comparación superpuesta del espacio de estados · 2D",
                    height=560,
                    xaxis_title="RR(t)", yaxis_title="RR(t-τ)", legend_title="Registro"
                )
                st.plotly_chart(_fig2, use_container_width=True, key="attractor_compare_2d_v15322")
            else:
                st.markdown("#### Atractores reconstruidos · paneles independientes")
                _row_lookup = {str(r.get("Registro")): r for r in _attr_rows}
                _attr_multi_patient = _has_multiple_patients(_attr_records)

                if _attr_multi_patient:
                    st.caption("Comparación multipaciente paginada: cada paciente ocupa una columna, con un máximo de 3 pacientes visibles por página; sus registros se apilan cronológicamente dentro de esa columna. La paginación es local y no consulta PostgreSQL.")
                    _attr_groups = _records_grouped_by_patient(_attr_records)
                    _attr_all_pids = list(_attr_groups.keys())
                    _attr_pids, _attr_page, _attr_pages = _patient_page_controls(
                        _attr_all_pids, "attractor_v1543", page_size=3
                    )
                    _attr_ncols = max(1, len(_attr_pids))
                    _attr_total_rows = max((len(_attr_groups.get(pid, [])) for pid in _attr_pids), default=0)
                    _arp, _arnp, _attr_start, _attr_end = _record_page_controls(
                        _attr_total_rows, "attractor_records_v1543", page_size=4, label="registros por paciente"
                    )
                    _attr_nrows = min(4, max(0, _attr_total_rows - _attr_start))
                    st.caption(f"Bloque visible: registros {_attr_start + 1 if _attr_total_rows else 0}–{min(_attr_end, _attr_total_rows)} de hasta {_attr_total_rows} por paciente. Todos permanecen cargados en el histórico.")

                    _head_cols = st.columns(_attr_ncols)
                    for _ci, _pid in enumerate(_attr_pids):
                        with _head_cols[_ci]:
                            st.markdown(f"### {_patient_display_name(_pid)}")

                    for _ri in range(_attr_nrows):
                        _cols = st.columns(_attr_ncols)
                        for _ci, _pid in enumerate(_attr_pids):
                            _recs = _attr_groups.get(_pid, [])[_attr_start:_attr_end]
                            with _cols[_ci]:
                                if _ri >= len(_recs):
                                    st.empty()
                                    continue
                                _rec = _recs[_ri]
                                _cached = _attr_cache_v15323.get(_rec, {})
                                _meta = _row_lookup.get(str(_rec), {})
                                st.markdown(f"**{_record_axis_label(_rec, multiline=False)}**")
                                if not _cached.get("ok"):
                                    st.warning(f"{_rec}: {_meta.get('Evidencia', 'Atractor no disponible')}")
                                    continue
                                _tau = int(_cached.get("tau"))
                                _m_rec = int(_cached.get("m"))
                                _emb = np.asarray(_cached.get("emb"))
                                _show = _emb if len(_emb)<=_maxpts else _emb[np.linspace(0,len(_emb)-1,int(_maxpts)).astype(int)]
                                if _meta:
                                    st.caption(f"τ={_meta.get('τ', _tau)} · m={_meta.get('m', _m_rec)} · {_meta.get('Descriptor', '')}")
                                _patient_color = _export_color_for(_attr_all_pids.index(_pid))
                                if _m_rec>=3:
                                    _fi=go.Figure(go.Scatter3d(x=_show[:,0],y=_show[:,1],z=_show[:,2],mode="lines",name=_rec,line=dict(color=_patient_color)))
                                    _fi.update_layout(height=500,scene=dict(xaxis_title="RR(t)",yaxis_title="RR(t-τ)",zaxis_title="RR(t-2τ)"),showlegend=False,margin=dict(l=0,r=0,t=20,b=0))
                                else:
                                    _fi=go.Figure(go.Scatter(x=_show[:,0],y=_show[:,1],mode="lines",name=_rec,line=dict(color=_patient_color)))
                                    _fi.update_layout(height=500,xaxis_title="RR(t)",yaxis_title="RR(t-τ)",showlegend=False,margin=dict(l=20,r=10,t=20,b=35))
                                st.plotly_chart(_fi,use_container_width=True,key=f"attr_pat_{hashlib.md5((str(_pid)+'|'+str(_rec)).encode()).hexdigest()[:12]}_v1543")
                else:
                    st.caption("Cada registro se representa en su propia gráfica. En históricos extensos se pagina para no renderizar decenas o cientos de atractores simultáneamente.")
                    _attr_records_sorted = sorted(_attr_records, key=lambda r: (extract_datetime_from_name(str(r)), str(r)))
                    _asp, _asnp, _as0, _as1 = _record_page_controls(len(_attr_records_sorted), "attractor_single_v1543", page_size=12, label="registros")
                    _attr_records_visible = _attr_records_sorted[_as0:_as1]
                    _ncols = 1 if _panel_layout == "1 por fila" else 2
                    for _i in range(0, len(_attr_records_visible), _ncols):
                        _cols = st.columns(_ncols)
                        for _j, _rec in enumerate(_attr_records_visible[_i:_i+_ncols]):
                            with _cols[_j]:
                                _cached = _attr_cache_v15323.get(_rec, {})
                                _meta = _row_lookup.get(str(_rec), {})
                                st.markdown(f"**{_short_record_label(_rec, 42)}**")
                                if not _cached.get("ok"):
                                    st.warning(f"{_rec}: {_meta.get('Evidencia', 'Atractor no disponible')}")
                                    continue
                                _tau = int(_cached.get("tau"))
                                _m_rec = int(_cached.get("m"))
                                _emb = np.asarray(_cached.get("emb"))
                                _show = _emb if len(_emb)<=_maxpts else _emb[np.linspace(0,len(_emb)-1,int(_maxpts)).astype(int)]
                                if _meta:
                                    st.caption(f"τ={_meta.get('τ', _tau)} · m={_meta.get('m', _m_rec)} · {_meta.get('Descriptor', '')}")
                                if _m_rec>=3:
                                    _fi=go.Figure(go.Scatter3d(x=_show[:,0],y=_show[:,1],z=_show[:,2],mode="lines",name=_rec))
                                    _fi.update_layout(height=500 if _ncols == 1 else 430,scene=dict(xaxis_title="RR(t)",yaxis_title="RR(t-τ)",zaxis_title="RR(t-2τ)"),showlegend=False)
                                else:
                                    _fi=go.Figure(go.Scatter(x=_show[:,0],y=_show[:,1],mode="lines",name=_rec))
                                    _fi.update_layout(height=500 if _ncols == 1 else 430,xaxis_title="RR(t)",yaxis_title="RR(t-τ)",showlegend=False)
                                st.plotly_chart(_fi,use_container_width=True,key=f"attr_ind_{hashlib.md5(str(_rec).encode()).hexdigest()[:10]}_v15331")

            st.info(
                "Interpretación conjunta recomendada: geometría reconstruida + Lyapunov LLE + D2 + SampEn/MSE + DFA/Hurst + RQA. "
                "Una nube compleja, periodicidad aparente o LLE positivo no demuestran por sí solos un tipo de atractor determinado en HRV real."
            )
        else:
            st.warning("Ninguno de los registros seleccionados tiene suficientes RRi para reconstruir el espacio de estados con estos parámetros.")
    else:
        st.info("Selecciona uno o varios registros para reconstruir sus atractores.")



    st.markdown("### Diagnóstico Kubios SampEn / MSE")
    st.caption(
        "Muestra por escala MSE: N, SD, r=0.2×SD, B/A y los modos: clásico, A0=0.5, A0=1.0 y RCMSE/Composite. "
        "El selector del modo activo está en la barra lateral."
    )

    if len(records) > 0:
        diag_rec = st.selectbox("Registro para diagnóstico MSE", records, index=records.index(selected_record) if selected_record in records else 0, key="diag_mse_rec_v104")

        diag_phases = [p for p in PHASES if p in active_phases]
        if not diag_phases:
            diag_phases = [p for p in PHASES if records_valid.get(diag_rec, {}).get(p, False)]

        if diag_phases:
            diag_phase = st.selectbox("Fase para diagnóstico MSE", diag_phases, key="diag_mse_phase_v104")
            diag_windows = get_record_windows(
                st.session_state.global_windows_v50,
                st.session_state.record_windows_v50,
                diag_rec,
                use_independent
            )
            diag_w = diag_windows.get(diag_phase)

            if diag_w is not None:
                diag_seg = cut_segment(record_data[diag_rec]["rr"], diag_w[0], diag_w[1])
                if len(diag_seg) >= min_rr:
                    try:
                        diag_df = entropy_kubios_diagnostic_table(diag_seg)
                        st.plotly_chart(entropy_diagnostic_figure(diag_df), use_container_width=True, key="entropy_diag_fig_v104")
                        st.dataframe(diag_df, use_container_width=True)

                        csv_diag = diag_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Descargar diagnóstico SampEn/MSE CSV",
                            csv_diag,
                            file_name=f"diagnostico_sampen_mse_{diag_rec}_{diag_phase}.csv",
                            mime="text/csv",
                            key="download_entropy_diag_v104"
                        )
                    except Exception as _diag_exc:
                        st.warning(f"El diagnóstico SampEn/MSE no pudo completarse ({type(_diag_exc).__name__}), pero el resto de la pestaña permanece disponible.")
                else:
                    st.info("La fase seleccionada no tiene suficientes RRi para diagnóstico.")
            else:
                st.info("La fase seleccionada no tiene ventana definida.")


# ============================================================
# COMPARACIÓN LIBRE DE FASES ENTRE REGISTROS
# ============================================================

def _valid_phases_for_record(record_data, global_windows, record_windows, rec, use_independent):
    try:
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        return [ph for ph in PHASES if windows.get(ph) is not None]
    except Exception:
        return []


def _free_pairs_default(record_data, global_windows, record_windows, use_independent):
    pairs = []
    for rec in record_data.keys():
        phases = _valid_phases_for_record(record_data, global_windows, record_windows, rec, use_independent)
        if phases:
            pairs.append({"Usar": True, "Registro": rec, "Fase": phases[0]})
    return pairs


def poincare_free_pairs_panel_figure(record_data, global_windows, record_windows, pairs, use_independent):
    """
    Poincaré comparativo libre con agrupación por paciente.

    v15.4.0: si intervienen 2, 3 o más pacientes, cada paciente ocupa una
    columna fija; sus registros/fases se apilan cronológicamente hacia abajo.
    """
    pairs = [p for p in pairs if p.get("Usar", True) and p.get("Registro") in record_data and p.get("Fase")]
    if not pairs:
        fig = go.Figure()
        fig.update_layout(title="No hay pares Registro/Fase seleccionados")
        return fig

    recs = [p["Registro"] for p in pairs]
    multi_patient = _has_multiple_patients(recs)
    if multi_patient:
        grouped = {}
        for p in pairs:
            pid = normalize_patient_id(infer_patient_id(str(p["Registro"])))
            grouped.setdefault(pid, []).append(p)
        grouped = {pid: sorted(vals, key=lambda q: (extract_datetime_from_name(q["Registro"]), str(q["Registro"]), str(q["Fase"])))
                   for pid, vals in sorted(grouped.items(), key=lambda kv: str(kv[0]).lower())}
        patient_ids = list(grouped.keys())
        cols = len(patient_ids)
        rows = max(len(v) for v in grouped.values())
        slots, titles = [], []
        for ri in range(rows):
            for pid in patient_ids:
                vals = grouped[pid]
                item = vals[ri] if ri < len(vals) else None
                slots.append(item)
                titles.append((f"{_record_axis_label(item['Registro'], multiline=False)} · {item['Fase']}" if item else ""))
    else:
        n = len(pairs)
        cols = min(2, n)
        rows = int(np.ceil(n / cols))
        slots = pairs + [None] * (rows * cols - n)
        titles = [f"{_short_record_label(p['Registro'], 26)} · {p['Fase']}" if p else "" for p in slots]
        patient_ids = []

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.06 if cols <= 2 else 0.035, vertical_spacing=_safe_vertical_spacing(rows, 0.14))

    global_min, global_max = np.inf, -np.inf
    cache = {}
    for p in pairs:
        rec, ph = p["Registro"], p["Fase"]
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(ph)
        key = (rec, ph)
        if w is None:
            cache[key] = None
            continue
        seg = cut_segment(record_data[rec]["rr"], w[0], w[1])
        if len(seg) < 3:
            cache[key] = None
            continue
        rr_ms = seg * 1000
        x, y = rr_ms[:-1], rr_ms[1:]
        diff = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan
        cache[key] = (x, y, sd1, sd2, len(seg))
        global_min = min(global_min, np.nanmin(x), np.nanmin(y))
        global_max = max(global_max, np.nanmax(x), np.nanmax(y))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        fig = go.Figure(); fig.update_layout(title="Poincaré libre: sin datos suficientes"); return fig
    pad = max(20, 0.05 * (global_max-global_min)); global_min -= pad; global_max += pad

    for idx, p in enumerate(slots):
        r, c = idx // cols + 1, idx % cols + 1
        if p is None:
            fig.update_xaxes(visible=False,row=r,col=c); fig.update_yaxes(visible=False,row=r,col=c); continue
        rec, ph = p["Registro"], p["Fase"]
        item = cache.get((rec, ph))
        if item is None:
            fig.add_annotation(text="Sin datos suficientes",x=.5,y=.5,xref=f"x{idx+1 if idx>0 else ''} domain",yref=f"y{idx+1 if idx>0 else ''} domain",showarrow=False); continue
        x,y,sd1,sd2,nseg = item
        if multi_patient:
            pid = normalize_patient_id(infer_patient_id(str(rec))); color = _export_color_for(patient_ids.index(pid))
        else:
            color = _export_color_for(idx)
        fig.add_trace(go.Scatter(x=x,y=y,mode="markers",marker=dict(size=5,opacity=.72,color=color),showlegend=False,
                                 text=[f"{_patient_axis_label(rec)}<br>{_record_axis_label(rec,multiline=False)}<br>Fase: {ph}<br>RRn={xx:.1f}<br>RRn+1={yy:.1f}" for xx,yy in zip(x,y)],hoverinfo="text"),row=r,col=c)
        fig.add_trace(go.Scatter(x=[global_min,global_max],y=[global_min,global_max],mode="lines",line=dict(dash="dash",width=1.2,color=color),showlegend=False,hoverinfo="skip"),row=r,col=c)
        fig.add_annotation(text=f"N={nseg}<br>SD1={sd1:.1f} ms<br>SD2={sd2:.1f} ms",x=global_min,y=global_max,xanchor="left",yanchor="top",showarrow=False,bgcolor="rgba(0,0,0,0.45)",bordercolor=color,font=dict(size=10),row=r,col=c)
        fig.update_xaxes(range=[global_min,global_max],title_text="RR(n) ms",row=r,col=c); fig.update_yaxes(range=[global_min,global_max],title_text="RR(n+1) ms",row=r,col=c)

    if multi_patient:
        for ci,pid in enumerate(patient_ids,start=1):
            fig.add_annotation(x=(ci-.5)/cols,y=1.08,xref="paper",yref="paper",text=f"<b>{_patient_display_name(pid)}</b>",showarrow=False,font=dict(size=15))
    fig.update_layout(title="Poincaré comparativo libre · columnas por paciente",height=max(520,rows*480),showlegend=False,hovermode="closest",margin=dict(t=110 if multi_patient else 80))
    return fig


def hvg_network_free_pairs_panel_figure(record_data, global_windows, record_windows, pairs, use_independent, max_nodes=120):
    """HVG comparativo libre agrupado por paciente (v15.4.0)."""
    if nx is None:
        fig=go.Figure(); fig.update_layout(title="NetworkX no disponible"); return fig
    pairs=[p for p in pairs if p.get("Usar",True) and p.get("Registro") in record_data and p.get("Fase")]
    if not pairs:
        fig=go.Figure(); fig.update_layout(title="No hay pares Registro/Fase seleccionados"); return fig

    recs=[p["Registro"] for p in pairs]; multi_patient=_has_multiple_patients(recs)
    if multi_patient:
        grouped={}
        for p in pairs:
            pid=normalize_patient_id(infer_patient_id(str(p["Registro"]))); grouped.setdefault(pid,[]).append(p)
        grouped={pid:sorted(vals,key=lambda q:(extract_datetime_from_name(q["Registro"]),str(q["Registro"]),str(q["Fase"]))) for pid,vals in sorted(grouped.items(),key=lambda kv:str(kv[0]).lower())}
        patient_ids=list(grouped.keys()); cols=len(patient_ids); rows=max(len(v) for v in grouped.values())
        slots=[]; titles=[]
        for ri in range(rows):
            for pid in patient_ids:
                vals=grouped[pid]; item=vals[ri] if ri<len(vals) else None; slots.append(item)
                titles.append(f"{_record_axis_label(item['Registro'],multiline=False)} · {item['Fase']}" if item else "")
    else:
        n=len(pairs); cols=min(2,n); rows=int(np.ceil(n/cols)); slots=pairs+[None]*(rows*cols-n); patient_ids=[]
        titles=[f"{_short_record_label(p['Registro'],26)} · {p['Fase']}" if p else "" for p in slots]

    fig=make_subplots(rows=rows,cols=cols,subplot_titles=titles,horizontal_spacing=.04 if cols<=2 else .025,vertical_spacing=_safe_vertical_spacing(rows, .12))
    for idx,p in enumerate(slots):
        r,c=idx//cols+1,idx%cols+1
        if p is None:
            fig.update_xaxes(visible=False,row=r,col=c); fig.update_yaxes(visible=False,row=r,col=c); continue
        rec,ph=p["Registro"],p["Fase"]
        windows=get_record_windows(global_windows,record_windows,rec,use_independent); w=windows.get(ph)
        if w is None:
            fig.add_annotation(text="Ventana no definida",x=.5,y=.5,xref=f"x{idx+1 if idx>0 else ''} domain",yref=f"y{idx+1 if idx>0 else ''} domain",showarrow=False); continue
        seg=cut_segment(record_data[rec]["rr"],w[0],w[1])
        if len(seg)<20:
            fig.add_annotation(text="Sin datos suficientes",x=.5,y=.5,xref=f"x{idx+1 if idx>0 else ''} domain",yref=f"y{idx+1 if idx>0 else ''} domain",showarrow=False); continue
        G=hvg_graph(seg,max_nodes=max_nodes)
        if G is None or G.number_of_nodes()==0: continue
        pos=nx.spring_layout(G,seed=42,k=.20,iterations=60); edge_x=[]; edge_y=[]
        for a,b in G.edges(): edge_x += [pos[a][0],pos[b][0],None]; edge_y += [pos[a][1],pos[b][1],None]
        deg=dict(G.degree()); node_x=[pos[nn][0] for nn in G.nodes()]; node_y=[pos[nn][1] for nn in G.nodes()]; node_size=[5+deg[nn]*2.2 for nn in G.nodes()]
        if multi_patient:
            pid=normalize_patient_id(infer_patient_id(str(rec))); color=_export_color_for(patient_ids.index(pid))
        else: color=_export_color_for(idx)
        node_text=[f"{_patient_axis_label(rec)}<br>{_record_axis_label(rec,multiline=False)}<br>Fase: {ph}<br>n={nn}<br>grado={deg[nn]}" for nn in G.nodes()]
        fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode="lines",line=dict(width=.55,color=color),opacity=.35,hoverinfo="skip",showlegend=False),row=r,col=c)
        fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers",marker=dict(size=node_size,color=color,opacity=.82),text=node_text,hoverinfo="text",showlegend=False),row=r,col=c)
        fig.update_xaxes(visible=False,row=r,col=c); fig.update_yaxes(visible=False,row=r,col=c)
    if multi_patient:
        for ci,pid in enumerate(patient_ids,start=1):
            fig.add_annotation(x=(ci-.5)/cols,y=1.08,xref="paper",yref="paper",text=f"<b>{_patient_display_name(pid)}</b>",showarrow=False,font=dict(size=15))
    fig.update_layout(title="HVG comparativo libre · columnas por paciente",height=max(520,rows*480),showlegend=False,margin=dict(t=110 if multi_patient else 80))
    return fig


def free_pairs_hvg_metrics_table(long_df, pairs):
    """
    Tabla de métricas HVG para pares libres Registro/Fase.
    """
    if long_df is None or long_df.empty or "Registro" not in long_df.columns or "Fase" not in long_df.columns:
        return pd.DataFrame()

    hvg_cols = [
        "HVG_graph_type", "HVG_topology_state", "HVG_compactness_index",
        "HVG_graph_score_scale_free", "HVG_graph_score_small_world", "HVG_graph_score_chain",
        "HVG_nodes", "HVG_edges", "HVG_degree_mean", "HVG_degree_max", "HVG_hubs_p90",
        "HVG_clustering", "HVG_lambda", "HVG_path_length", "HVG_diameter", "HVG_graph_interpretation"
    ]

    rows = []
    for p in pairs:
        if not p.get("Usar", True):
            continue
        rec, ph = p.get("Registro"), p.get("Fase")
        d = long_df[(long_df["Registro"] == rec) & (long_df["Fase"] == ph)]
        if d.empty:
            continue
        row = {"Registro": rec, "Fase": ph}
        for c in hvg_cols:
            if c in d.columns:
                row[c] = d.iloc[0][c]
        rows.append(row)

    return pd.DataFrame(rows)




with tab5:
    st.subheader("Poincaré y grafos comparativos")

    if len(records) < 1:
        st.info("Sube al menos un registro.")
    else:
        available_phases_pg = [p for p in PHASES if p in active_phases]
        if not available_phases_pg:
            available_phases_pg = [p for p in PHASES if any(records_valid[rec].get(p, False) for rec in records)]

        if not available_phases_pg:
            st.info("No hay fases válidas. Define ventanas y activa fases.")
        else:
            phase_pg = st.selectbox("Fase para Poincaré / grafo", available_phases_pg, key="phase_pg_v101")
            modo_fases_pg = st.radio(
                "Qué quieres mostrar",
                ["Una fase seleccionada", "Todas las fases del registro principal", "Comparación libre de fases"],
                horizontal=True,
                key="modo_fases_pg_v101"
            )

            free_pairs = []
            if modo_fases_pg == "Comparación libre de fases":
                st.info("Selecciona una fase diferente para cada registro. Ejemplo: Basal del archivo 1 vs R1 del archivo 2.")

                if "free_pairs_pg_v101" not in st.session_state:
                    st.session_state.free_pairs_pg_v101 = _free_pairs_default(
                        record_data,
                        st.session_state.global_windows_v50,
                        st.session_state.record_windows_v50,
                        use_independent
                    )

                # Asegurar que todos los registros aparecen
                existing = {p.get("Registro") for p in st.session_state.free_pairs_pg_v101}
                for rec in records:
                    if rec not in existing:
                        phases = _valid_phases_for_record(
                            record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            rec,
                            use_independent
                        )
                        st.session_state.free_pairs_pg_v101.append({
                            "Usar": True,
                            "Registro": rec,
                            "Fase": phases[0] if phases else (active_phases[0] if active_phases else "Basal")
                        })

                # Editor paginado: con 100-200 visitas no se crean cientos de widgets en un solo rerun.
                _records_editor = sorted(records, key=lambda r: (extract_datetime_from_name(str(r)), str(r)))
                _ep, _enp, _e0, _e1 = _record_page_controls(len(_records_editor), "free_pairs_editor_v1543", page_size=20, label="registros para editar")
                _visible_editor_records = _records_editor[_e0:_e1]
                _edited_map = {p.get("Registro"): dict(p) for p in st.session_state.free_pairs_pg_v101}
                for i, rec in enumerate(_visible_editor_records, start=_e0):
                    phases_rec = _valid_phases_for_record(
                        record_data,
                        st.session_state.global_windows_v50,
                        st.session_state.record_windows_v50,
                        rec,
                        use_independent
                    )
                    if not phases_rec:
                        phases_rec = available_phases_pg

                    prev = _edited_map.get(rec)
                    prev_phase = prev.get("Fase") if prev else phases_rec[0]
                    if prev_phase not in phases_rec:
                        prev_phase = phases_rec[0]

                    col_use, col_rec, col_phase = st.columns([0.7, 3.2, 2.0])
                    with col_use:
                        use_pair = st.checkbox("Usar", value=bool(prev.get("Usar", True)) if prev else True, key=f"free_use_{rec}_{i}")
                    with col_rec:
                        st.text_input("Registro", value=rec, disabled=True, key=f"free_rec_{rec}_{i}")
                    with col_phase:
                        ph_sel = st.selectbox("Fase", phases_rec, index=phases_rec.index(prev_phase), key=f"free_phase_{rec}_{i}")
                    _edited_map[rec] = {"Usar": use_pair, "Registro": rec, "Fase": ph_sel}

                edited_pairs = [_edited_map[r] for r in _records_editor if r in _edited_map]
                st.session_state.free_pairs_pg_v101 = edited_pairs
                free_pairs = edited_pairs

                st.caption("Pares activos:")
                st.dataframe(pd.DataFrame([p for p in free_pairs if p.get("Usar", True)]), use_container_width=True)

            # v15.4.3 · paginación longitudinal escalable (máximo 3 columnas).
            # Se calcula una sola vez y se reutiliza en Poincaré y HVG para que ambos
            # módulos muestren exactamente el mismo grupo de pacientes en la página.
            _pg_display_record_data = record_data
            _pg_visible_pids = []
            _pg_free_pairs_display = free_pairs
            if modo_fases_pg != "Todas las fases del registro principal" and _has_multiple_patients(records):
                _pg_groups_all = _records_grouped_by_patient(records)
                _pg_all_pids = list(_pg_groups_all.keys())
                st.markdown("#### Navegación multipaciente")
                st.caption("Máximo 3 pacientes por página para conservar el tamaño de Poincaré y HVG. Cambiar de página es una operación local: no consulta PostgreSQL.")
                _pg_visible_pids, _pg_patient_page, _pg_patient_pages = _patient_page_controls(
                    _pg_all_pids, "poincare_hvg_v1543", page_size=3
                )
                _pg_patient_data = _record_data_for_patients(record_data, _pg_visible_pids)
                _pg_visible_groups = _records_grouped_by_patient(_pg_patient_data.keys())
                _pg_max_visits = max((len(v) for v in _pg_visible_groups.values()), default=0)
                _vrp, _vrnp, _vr0, _vr1 = _record_page_controls(_pg_max_visits, "poincare_hvg_records_v1543", page_size=6, label="registros por paciente")
                _pg_display_record_data = _record_data_visit_slice(record_data, _pg_visible_pids, _vr0, _vr1)
                _pg_free_pairs_display = _pairs_visit_slice(free_pairs, _pg_visible_pids, _vr0, _vr1)
                st.caption(f"Visualización limitada al bloque {_vrp}/{_vrnp}: máximo 6 registros por paciente y 3 pacientes simultáneos (≤18 paneles). El histórico completo permanece cargado.")
            elif modo_fases_pg != "Todas las fases del registro principal":
                # Un solo paciente con histórico largo: 12 registros por página.
                _single_records = sorted(records, key=lambda r: (extract_datetime_from_name(str(r)), str(r)))
                _srp, _srnp, _sr0, _sr1 = _record_page_controls(len(_single_records), "poincare_hvg_single_v1543", page_size=12, label="registros")
                _wanted_records = set(_single_records[_sr0:_sr1])
                _pg_display_record_data = {r: record_data[r] for r in _single_records[_sr0:_sr1] if r in record_data}
                _pg_free_pairs_display = [p for p in free_pairs if p.get("Registro") in _wanted_records]
                if len(_single_records) > 12:
                    st.caption(f"Histórico extenso: mostrando {_sr0+1}–{_sr1} de {len(_single_records)} registros. Todos siguen disponibles para cálculos y predicción.")

            st.markdown("### Poincaré")
            if modo_fases_pg == "Todas las fases del registro principal":
                st.plotly_chart(
                    poincare_all_phases_panel_figure(
                        record_data,
                        st.session_state.global_windows_v50,
                        st.session_state.record_windows_v50,
                        selected_record,
                        use_independent,
                    ),
                    use_container_width=True,
                    key=f"poincare_all_phases_{selected_record}"
                )
            elif modo_fases_pg == "Comparación libre de fases":
                st.plotly_chart(
                    poincare_free_pairs_panel_figure(
                        _pg_display_record_data,
                        st.session_state.global_windows_v50,
                        st.session_state.record_windows_v50,
                        _pg_free_pairs_display,
                        use_independent,
                    ),
                    use_container_width=True,
                    key="poincare_free_pairs_v101"
                )
            else:
                modo_poincare = st.radio(
                    "Modo de visualización Poincaré",
                    ["Paneles separados", "Superpuestos"],
                    horizontal=True,
                    key="modo_poincare_v101"
                )

                if modo_poincare == "Paneles separados":
                    st.plotly_chart(
                        poincare_panel_figure(
                            _pg_display_record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            phase_pg,
                            use_independent,
                        ),
                        use_container_width=True,
                        key=f"poincare_panel_{phase_pg}"
                    )
                else:
                    st.plotly_chart(
                        poincare_figure(
                            _pg_display_record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            phase_pg,
                            use_independent,
                        ),
                        use_container_width=True,
                        key=f"poincare_overlay_{phase_pg}"
                    )

            st.markdown("### Métricas HVG / grafos")
            try:
                selected_metrics_df = records_results.get(selected_record, pd.DataFrame())
                if selected_metrics_df is not None and not selected_metrics_df.empty:
                    st.markdown("#### Resumen topológico HVG")
                    first_valid_hvg = None
                    for _fase, _row in selected_metrics_df.iterrows():
                        if "HVG_nodes" in selected_metrics_df.columns and pd.notna(_row.get("HVG_nodes", np.nan)):
                            first_valid_hvg = _row.to_dict()
                            break
                    if first_valid_hvg is not None:
                        st.dataframe(hvg_summary_card(first_valid_hvg), use_container_width=True)
                        st.info(str(first_valid_hvg.get("HVG_topology_interpretation", "")))
                st.markdown("#### Definiciones y rangos orientativos")
                st.dataframe(hvg_reference_ranges(), use_container_width=True)
            except Exception:
                pass

            if not include_hvg:
                st.warning("Activa 'Calcular HVG/grafos' en la barra lateral para calcular las métricas de grafos.")
            else:
                hvg_cols = [
                    "HVG_graph_type", "HVG_topology_state", "HVG_compactness_index",
                    "HVG_graph_score_scale_free", "HVG_graph_score_small_world", "HVG_graph_score_chain",
                    "HVG_nodes", "HVG_edges", "HVG_degree_mean", "HVG_degree_max", "HVG_hubs_p90",
                    "HVG_clustering", "HVG_lambda", "HVG_path_length", "HVG_diameter", "HVG_graph_interpretation"
                ]

                if modo_fases_pg == "Comparación libre de fases":
                    hvg_df = free_pairs_hvg_metrics_table(long_df, free_pairs)
                else:
                    if "Fase" in long_df.columns:
                        hvg_df = long_df[long_df["Fase"] == phase_pg][["Registro", "Fase"] + [c for c in hvg_cols if c in long_df.columns]]
                    else:
                        hvg_df = pd.DataFrame(columns=["Registro", "Fase"] + [c for c in hvg_cols if c in long_df.columns])

                if hvg_df.empty:
                    st.info("No hay métricas HVG disponibles para la selección actual. Revisa que las fases estén activas y tengan suficientes RRi.")
                else:
                    st.dataframe(hvg_df, use_container_width=True)

                hvg_numeric = [c for c in hvg_cols if c in hvg_df.columns and pd.api.types.is_numeric_dtype(hvg_df[c])]
                if hvg_numeric and not hvg_df.empty:
                    st.markdown("### Panel HVG multivariable")
                    st.caption("Selecciona varias métricas topológicas. Cada métrica se representa en su propia gráfica con columnas y línea de tendencia.")

                    recommended_hvg = [c for c in [
                        "HVG_degree_mean", "HVG_degree_max", "HVG_hubs_p90", "HVG_clustering",
                        "HVG_lambda", "HVG_path_length", "HVG_diameter", "HVG_compactness_index"
                    ] if c in hvg_numeric]
                    hvg_multi_key = "hvg_vars_multiselect_v1535"
                    if hvg_multi_key not in st.session_state:
                        st.session_state[hvg_multi_key] = recommended_hvg[:] if recommended_hvg else hvg_numeric[:4]

                    # Depura selecciones antiguas si cambia el conjunto de métricas disponible.
                    st.session_state[hvg_multi_key] = [c for c in st.session_state[hvg_multi_key] if c in hvg_numeric]

                    b1, b2, b3 = st.columns([1, 1.25, 1])
                    with b1:
                        if st.button("Seleccionar todas", key="hvg_select_all_v1535", use_container_width=True):
                            st.session_state[hvg_multi_key] = hvg_numeric[:]
                    with b2:
                        if st.button("Selección topológica recomendada", key="hvg_select_recommended_v1535", use_container_width=True):
                            st.session_state[hvg_multi_key] = recommended_hvg[:]
                    with b3:
                        if st.button("Limpiar", key="hvg_clear_v1535", use_container_width=True):
                            st.session_state[hvg_multi_key] = []

                    hvg_vars = st.multiselect(
                        "Métricas de grafo a comparar",
                        options=hvg_numeric,
                        key=hvg_multi_key,
                        help="Puedes seleccionar cualquier combinación de métricas HVG o usar 'Seleccionar todas'."
                    )
                    st.caption(f"{len(hvg_vars)} de {len(hvg_numeric)} métricas seleccionadas")

                    if hvg_vars:
                        paginate_hvg = st.checkbox(
                            "Paginar las gráficas para mantener la pestaña ligera",
                            value=True,
                            key="hvg_paginate_v1535"
                        )
                        if paginate_hvg:
                            per_page = st.selectbox(
                                "Gráficas HVG por página", [2, 4, 6, 8], index=1,
                                key="hvg_plots_per_page_v1535"
                            )
                            n_pages = max(1, int(np.ceil(len(hvg_vars) / per_page)))
                            page = st.number_input(
                                "Página HVG", min_value=1, max_value=n_pages, value=1, step=1,
                                key="hvg_plot_page_v1535"
                            )
                            start_i = (int(page) - 1) * int(per_page)
                            shown_hvg_vars = hvg_vars[start_i:start_i + int(per_page)]
                            st.caption(f"Mostrando {start_i + 1}–{start_i + len(shown_hvg_vars)} de {len(hvg_vars)} métricas · página {int(page)} de {n_pages}")
                        else:
                            shown_hvg_vars = hvg_vars

                        # Dos paneles por fila cuando hay más de una métrica; cada eje mantiene su escala propia.
                        for i in range(0, len(shown_hvg_vars), 2):
                            row_cols = st.columns(2) if len(shown_hvg_vars) > 1 else [st.container()]
                            for j, hvg_var in enumerate(shown_hvg_vars[i:i+2]):
                                with row_cols[j]:
                                    if modo_fases_pg == "Comparación libre de fases":
                                        plot_df = hvg_df.copy()
                                        plot_df["Registro_Fase"] = plot_df["Registro"].astype(str) + " · " + plot_df["Fase"].astype(str)
                                        pivot_hvg = plot_df.pivot_table(index="Fase", columns="Registro_Fase", values=hvg_var, aggfunc="first")
                                    else:
                                        pivot_hvg = hvg_df.pivot_table(index="Fase", columns="Registro", values=hvg_var, aggfunc="first")

                                    fig_hvg = comparison_bar_line(pivot_hvg, hvg_var)
                                    fig_hvg.update_layout(height=430, title=hvg_var)
                                    st.plotly_chart(
                                        fig_hvg,
                                        use_container_width=True,
                                        key=f"hvg_compare_multi_{hvg_var}_{modo_fases_pg}_{phase_pg}_v1535"
                                    )
                    else:
                        st.info("Selecciona una o más métricas HVG para generar las gráficas comparativas.")

                if modo_fases_pg == "Todas las fases del registro principal":
                    st.markdown("### Grafos HVG por fases")
                    st.caption("Se muestran todas las fases del registro principal en paneles.")
                    st.plotly_chart(
                        hvg_all_phases_panel_figure(
                            record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            selected_record,
                            use_independent,
                            max_nodes=120
                        ),
                        use_container_width=True,
                        key=f"hvg_all_phases_{selected_record}"
                    )
                    st.plotly_chart(
                        hvg_metrics_all_phases_figure(records_results.get(selected_record, pd.DataFrame())),
                        use_container_width=True,
                        key=f"hvg_metrics_all_phases_{selected_record}"
                    )
                elif modo_fases_pg == "Comparación libre de fases":
                    st.markdown("### Grafos HVG comparativos libres")
                    st.caption("Se muestran fases distintas de registros distintos en paneles comparables.")
                    st.plotly_chart(
                        hvg_network_free_pairs_panel_figure(
                            _pg_display_record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            _pg_free_pairs_display,
                            use_independent,
                            max_nodes=120
                        ),
                        use_container_width=True,
                        key="hvg_network_free_pairs_v101"
                    )
                else:
                    st.markdown("### Grafos HVG comparativos")
                    st.caption("Se muestran los grafos de los registros lado a lado para la misma fase.")
                    st.plotly_chart(
                        hvg_network_compare_figure(
                            _pg_display_record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            phase_pg,
                            use_independent,
                            max_nodes=120
                        ),
                        use_container_width=True,
                        key=f"hvg_network_compare_{phase_pg}"
                    )

                st.markdown("### Grafo HVG individual")
                rec_graph = st.selectbox("Registro para visualizar individual", records, key="rec_graph_v101")
                phases_graph = _valid_phases_for_record(
                    record_data,
                    st.session_state.global_windows_v50,
                    st.session_state.record_windows_v50,
                    rec_graph,
                    use_independent
                ) or available_phases_pg
                ph_graph = st.selectbox("Fase individual", phases_graph, index=0, key=f"ph_graph_v101_{rec_graph}")

                windows_graph = get_record_windows(
                    st.session_state.global_windows_v50,
                    st.session_state.record_windows_v50,
                    rec_graph,
                    use_independent
                )
                w_graph = windows_graph.get(ph_graph)
                if w_graph is not None:
                    seg_graph = cut_segment(record_data[rec_graph]["rr"], w_graph[0], w_graph[1])
                    if len(seg_graph) >= min_rr:
                        st.plotly_chart(
                            hvg_network_figure(seg_graph, title=f"HVG {rec_graph} · {ph_graph}", max_nodes=140),
                            use_container_width=True,
                            key=f"hvg_network_individual_{rec_graph}_{ph_graph}"
                        )
                    else:
                        st.info("La fase seleccionada tiene pocos RRi para visualizar el grafo.")


with tab6:
    st.subheader("Dashboard visual: barras + línea suavizada")

    if long_df.empty:
        st.info("No hay datos.")
    else:
        st.markdown("### Resumen HRV del registro principal")
        st.plotly_chart(
            hrv_phase_summary_figure(metrics_df),
            use_container_width=True,
            key=f"dashboard_hrv_summary_{selected_record}"
        )

        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
        numeric_vars = [c for c in long_df.columns if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])]
        phases_dash = st.multiselect("Fases", PHASES, default=available_phases, key="dash_phases")
        params_dash = st.multiselect("Parámetros", numeric_vars, default=[p for p in DEFAULT_MULTI if p in numeric_vars], key="dash_params")
        if params_dash:
            st.plotly_chart(dashboard_bar_smooth(long_df, phases_dash or available_phases, params_dash), use_container_width=True, key="dashboard_tab_smooth")


with tab7:
    st.subheader("Informe automático HRV + grafos")
    report_md = generate_auto_report(
        record_data,
        records_results,
        st.session_state.global_windows_v50,
        st.session_state.record_windows_v50,
        active_phases,
        use_independent,
        long_df,
    )
    st.markdown(report_md)
    report_html = markdown_to_simple_html(report_md)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Descargar informe Markdown", report_md.encode("utf-8"), file_name="informe_hrv_grafos.md", mime="text/markdown")
    with c2:
        st.download_button("Descargar informe HTML", report_html.encode("utf-8"), file_name="informe_hrv_grafos.html", mime="text/html")



with tab9:
    st.subheader("Índices fisiológicos multivariados v14.1")
    st.info(
        "Estos índices sintetizan métricas convergentes en escalas 0-100. "
        "Son modelos fisiológicos transparentes y explicables; no equivalen a diagnóstico ni a probabilidades clínicas entrenadas."
    )
    st.caption("Eje X compacto: fecha/hora en formato DD/MM HH:MM, inclinada a −45°. El paciente se identifica por el color y la leyenda.")
    if indices_df.empty:
        st.info("No hay datos suficientes para calcular índices.")
    else:
        view_mode = st.radio(
            "Visualización", ["Registro seleccionado", "Comparación cronológica actual", "Histórico interno acumulado"],
            horizontal=True, key="indices_view_mode_v132"
        )
        show_cols=['Registro','Fase','IDX_Vagal','IDX_Amplitud','IDX_Complejidad','IDX_Rigidez','IDX_Adaptabilidad','IDX_Regulacion_Lenta','Perfil_autonomico']
        if view_mode == "Registro seleccionado":
            st.plotly_chart(physiological_indices_figure(indices_df, selected_record), use_container_width=True, key=f"physio_indices_{selected_record}")
            st.dataframe(indices_df[[c for c in show_cols if c in indices_df.columns]], use_container_width=True)
        elif view_mode == "Comparación cronológica actual":
            available_phases = [str(x) for x in indices_df['Fase'].dropna().unique().tolist()]
            phase_options = ['Todas'] + sorted(available_phases, key=lambda x: PHASES.index(x) if x in PHASES else 999)
            default_phase = 'Basal' if 'Basal' in phase_options else phase_options[0]
            chrono_phase = st.selectbox(
                "Fase para comparar entre registros", phase_options,
                index=phase_options.index(default_phase), key='chrono_indices_phase_v132'
            )
            chart_style = st.radio(
                "Formato del gráfico", ["Índices separados con tendencia", "Vista conjunta"],
                horizontal=True, key='chrono_chart_style_v141'
            )
            _idx_current_n = _record_count(indices_df)
            _icp, _icnp, _ic0, _ic1 = _record_page_controls(_idx_current_n, "indices_current_v1543", page_size=24, label="registros")
            _indices_visible_v1543 = _dataframe_record_slice(indices_df, _ic0, _ic1)
            if _idx_current_n > 24:
                st.caption(f"Longitudinal Scale: mostrando registros {_ic0+1}–{_ic1} de {_idx_current_n}. Los índices se calculan sobre el conjunto completo; la paginación solo limita el renderizado.")
            chrono_fig, chrono_df = chronological_indices_figure(_indices_visible_v1543, chrono_phase)
            if chart_style == "Índices separados con tendencia":
                render_separate_chronological_indices(_indices_visible_v1543, chrono_phase, 'current_chrono_v1543')
            else:
                st.plotly_chart(chrono_fig, use_container_width=True, key=f"chrono_indices_{chrono_phase}_v1543")
            chrono_show = ['Fecha_hora','Registro','Fase','IDX_Vagal','IDX_Amplitud','IDX_Complejidad','IDX_Rigidez','IDX_Adaptabilidad','IDX_Regulacion_Lenta','Perfil_autonomico']
            st.dataframe(chrono_df[[c for c in chrono_show if c in chrono_df.columns]], use_container_width=True)
            st.caption("La fecha se obtiene automáticamente del nombre del archivo (por ejemplo, 2026-06-21_17-13-02). Los registros sin fecha reconocible se muestran al final.")
        else:
            if history_df_v140.empty:
                st.info("La base interna todavía no contiene registros.")
            else:
                patients=sorted(history_df_v140['Paciente_ID'].dropna().astype(str).unique().tolist())
                hp=st.selectbox("Paciente/serie", ['Todos']+patients, key='hist_patient_v140')
                phases=sorted(history_df_v140['Fase'].dropna().astype(str).unique().tolist(), key=lambda x: PHASES.index(x) if x in PHASES else 999)
                phase_opts=['Todas']+phases
                hphase=st.selectbox("Fase histórica", phase_opts, index=phase_opts.index('Basal') if 'Basal' in phase_opts else 0, key='hist_phase_v140')
                hdf=history_df_v140.copy()
                if hp!='Todos': hdf=hdf[hdf['Paciente_ID']==hp]
                hist_chart_style = st.radio(
                    "Formato histórico", ["Índices separados con tendencia", "Vista conjunta"],
                    horizontal=True, key='hist_chart_style_v141'
                )
                _idx_hist_n = _record_count(hdf)
                _ihp, _ihnp, _ih0, _ih1 = _record_page_controls(_idx_hist_n, f"indices_history_v1543_{sanitize_name(hp)}", page_size=24, label="registros")
                _hdf_visible_v1543 = _dataframe_record_slice(hdf, _ih0, _ih1)
                if _idx_hist_n > 24:
                    st.caption(f"Histórico extenso: mostrando registros {_ih0+1}–{_ih1} de {_idx_hist_n}; el histórico completo sigue cargado y disponible para predicción.")
                hfig,hshow=chronological_indices_figure(_hdf_visible_v1543, hphase)
                if hist_chart_style == "Índices separados con tendencia":
                    render_separate_chronological_indices(_hdf_visible_v1543, hphase, f'history_chrono_v1543_{hp}')
                else:
                    st.plotly_chart(hfig,use_container_width=True,key=f'history_indices_{hp}_{hphase}_v1543')
                cols_hist=['Paciente_ID','Fecha_hora','Registro','Fase',*GB_INDEX_FEATURES,'Perfil_autonomico']
                st.dataframe(hshow[[c for c in cols_hist if c in hshow.columns]],use_container_width=True)
                st.caption(f"Base interna: {len(history_df_v140)} observaciones registro-fase. Se actualiza automáticamente al calcular y guardar una fase.")

        st.markdown("### Clasificación automática del estado autonómico")
        selected_idx=indices_df[indices_df['Registro']==selected_record]
        for _, r in selected_idx.iterrows():
            st.markdown(f"**{r['Fase']}** — {r['Perfil_autonomico']}")

        rec_score, rec_text = recovery_index_from_record(metrics_df)
        st.markdown("### Índice de recuperación")
        if np.isfinite(rec_score):
            st.metric("Recuperación hacia basal", f"{rec_score:.1f}/100", rec_text)
        else:
            st.info(rec_text)

        with st.expander("Definición y composición de los índices", expanded=True):
            st.dataframe(physiological_indices_reference_table(), use_container_width=True)
            st.markdown("""
**Criterio de construcción.** Cada variable se transforma primero a una escala 0-100 mediante rangos fisiológicos orientativos. Después se calcula una media ponderada usando sólo componentes válidos. La app no inventa valores cuando una métrica falta.

- **IDX_Vagal:** modulación parasimpática rápida y respiratoria.
- **IDX_Amplitud:** reserva global de variabilidad.
- **IDX_Complejidad:** riqueza de patrones y organización multiescala.
- **IDX_Rigidez:** repetición, persistencia y baja movilidad dinámica. En este índice, un valor alto es desfavorable si aparece junto a baja complejidad.
- **IDX_Adaptabilidad:** capacidad integrada para responder y reorganizarse.
- **IDX_Regulacion_Lenta:** peso relativo de mecanismos lentos; no es por sí mismo patológico.

La v15.1 guarda automáticamente cada registro-fase y predice de forma continua los índices del siguiente registro sin exigir clases de entrenamiento. Esto no equivale a mejoría clínica.
""")

        csv_idx=indices_df.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar índices fisiológicos CSV", csv_idx, file_name="indices_fisiologicos_v141.csv", mime="text/csv")



with tab10:
    st.subheader("Control autonómico · referencia longitudinal progresiva V/S/L/C/R")
    st.info(
        "El módulo interpreta el control autonómico como un sistema dinámico multivariado. "
        "Integra cinco dimensiones fisiológicas: **Vagal (V)**, **Alerta/barorreflejo (S)**, **Regulación lenta (L)**, "
        "**Complejidad/movilidad (C)** y **Reserva/amplitud (R)**. La **Calidad/contexto (Q)** modula la confianza, pero no decide el estado."
    )
    st.warning(
        "Módulo fisiológico experimental y explicable: no es un diagnóstico. LF y LF/HF no se consideran medidas puras de actividad simpática. "
        "DFA, Hurst, Lyapunov, entropías y métricas de red no se interpretan como 'cuanto más, mejor'; se valoran longitudinalmente y por convergencia."
    )

    with st.expander("Matriz definitiva de variables, pesos y función fisiológica",expanded=False):
        _matrix_auto_v1554=pd.DataFrame(AUTONOMIC_MATRIX_V1554)
        st.dataframe(_matrix_auto_v1554,use_container_width=True,hide_index=True)
        st.caption(
            "Los pesos son relativos dentro de cada vector. HF/LF/VLF/TOTAL combinan Welch, Lomb-Scargle, AR y wavelet como un único consenso por banda. "
            "MSE y MDE se resumen en escalas 1–5, 6–10 y 11–20 para evitar contar 40 variables correlacionadas como 40 votos independientes."
        )

    _auto_table_v1554,_auto_source_v1554=build_autonomic_control_table(
        long_df,indices_df,record_data=record_data,
        reference_history=(history_df_v140 if 'history_df_v140' in globals() else None)
    )
    if _auto_table_v1554.empty:
        st.info("No hay métricas suficientes para evaluar el Control Autonómico en los registros/fases actuales.")
    else:
        _auto_labels_v1554=[]
        for _i,_r in _auto_table_v1554.iterrows():
            _dt=pd.to_datetime(_r.get('Fecha_hora'),errors='coerce')
            _ds=_dt.strftime('%d/%m/%Y %H:%M:%S') if pd.notna(_dt) else ''
            _auto_labels_v1554.append(f"{_r.get('Registro','')} · {_r.get('Fase','')}" + (f" · {_ds}" if _ds else ''))
        _auto_sel_idx_v1554=st.selectbox(
            "Registro/fase para interpretar",
            range(len(_auto_table_v1554)),
            format_func=lambda i:_auto_labels_v1554[int(i)],
            key="autonomic_control_record_v1554"
        )
        _ar=_auto_table_v1554.iloc[int(_auto_sel_idx_v1554)]
        _c1,_c2,_c3,_c4=st.columns(4)
        _c1.metric("Estado del Control Autonómico",str(_ar['Estado_control_autonomico']))
        _c2.metric("Coste alostático",str(_ar['Coste_alostatico']))
        _c3.metric("Confianza algorítmica",f"{float(_ar['Confianza_clasificacion_pct']):.0f}%")
        _c4.metric("Calidad/contexto Q",f"{100*float(_ar['Calidad_Q']):.0f}%")
        if bool(_ar['Regla_estricta_cumplida']):
            st.success("La forma operativa cumple una de las reglas estrictas A–F del módulo.")
        else:
            st.warning("Ninguna regla A–F se cumple de forma completa. Se muestra la forma operativa de mayor compatibilidad; interpretar con confianza reducida.")

        st.markdown("### Vectores normalizados respecto al histórico previo del paciente")
        _ref_n_v1555=int(_ar.get('Referencia_n',0) or 0)
        _ref_method_v1555=str(_ar.get('Referencia_metodo','Sin referencia longitudinal previa'))
        if _ref_n_v1555>0:
            st.info(f"**Referencia individual:** {_ref_method_v1555}. La interpretación usa sólo registros anteriores del mismo paciente y prioriza la misma fase; nunca utiliza datos futuros.")
        else:
            st.warning("Este es el primer registro/fase disponible del paciente: no existe referencia longitudinal previa. Se conservan únicamente los criterios absolutos que puedan evaluarse y la confianza se reduce.")
        _vec_df=pd.DataFrame({
            'Vector':['Vagal (V)','Alerta/barorreflejo (S)','Regulación lenta (L)','Complejidad/movilidad (C)','Reserva/amplitud (R)'],
            'Z':[_ar['Vector_V_Z'],_ar['Vector_S_Z'],_ar['Vector_L_Z'],_ar['Vector_C_Z'],_ar['Vector_R_Z']]
        })
        _vcols=st.columns(5)
        for _j,_vr in _vec_df.iterrows():
            _vv=pd.to_numeric(_vr['Z'],errors='coerce')
            _ref_suffix_v1555 = "Δ norm." if _ref_n_v1555==1 else ("Z ind." if _ref_n_v1555>=2 else "")
            _vcols[_j].metric(str(_vr['Vector']),f"{_vv:+.2f} {_ref_suffix_v1555}" if np.isfinite(_vv) else "Sin referencia")
        _fig_auto=go.Figure(go.Bar(x=_vec_df['Vector'],y=_vec_df['Z']))
        _fig_auto.add_hline(y=0,line_dash='dash')
        _fig_auto.update_layout(height=380,yaxis_title='Desviación individual normalizada',xaxis_title='',title='Perfil autonómico multivectorial V/S/L/C/R')
        st.plotly_chart(_fig_auto,use_container_width=True,key='autonomic_vectors_v1554')
        st.caption(
            "Una desviación positiva significa que esa dimensión está por encima de la referencia individual previa del paciente; no significa automáticamente mejoría. "
            "Con una sola referencia se muestra Δ normalizado; con 2–4 referencias se usa estandarización individual y con ≥5 referencias Z robusto mediana/MAD. "
            "En C y L, un aumento puede ser adaptativo o ineficiente según reserva, red, entropía y concordancia entre variables."
        )

        st.markdown("### Consenso entre métodos y complejidad multiescala")
        _cc1,_cc2,_cc3,_cc4,_cc5,_cc6=st.columns(6)
        for _col,_label,_box in [
            ('HF_consenso_Z','HF',_cc1),('LF_consenso_Z','LF',_cc2),('VLF_consenso_Z','VLF',_cc3),('TOTAL_consenso_Z','TOTAL',_cc4),
            ('MSE_multiescala_Z','MSE',_cc5),('MDE_multiescala_Z','MDE',_cc6)]:
            _vv=pd.to_numeric(_ar.get(_col,np.nan),errors='coerce')
            _cons_suffix_v1555 = "Δ norm." if _ref_n_v1555==1 else ("Z ind." if _ref_n_v1555>=2 else "")
            _box.metric(f"{_label} consenso",f"{_vv:+.2f} {_cons_suffix_v1555}" if np.isfinite(_vv) else "N/D")

        st.markdown("### Justificación algorítmica")
        st.write(str(_ar['Justificacion']))
        st.caption(
            "La referencia se construye exclusivamente con registros ANTERIORES del mismo paciente, priorizando la misma fase. "
            "Con 1 antecedente se usa el cambio individual normalizado; con 2–4, una estandarización individual; con ≥5, Z robusto mediana/MAD. "
            "Los índices compuestos se usan con menor peso para no contar dos veces RMSSD/SD1/HF, SDNN/SD2/TOTAL u otras métricas ya incluidas directamente."
        )

        st.markdown("### Interpretación fisiológica y coste alostático")
        st.write(str(_ar['Interpretacion_fisiologica']))
        st.markdown("### Implicación clínica / rehabilitadora")
        st.write(str(_ar['Implicacion_clinica']))

        st.markdown("### Evolución cronológica del Control Autonómico")
        _show_cols=['Fecha_hora','Registro','Fase','Estado_control_autonomico','Regla_estricta_cumplida','Confianza_clasificacion_pct',
                    'Referencia_n','Referencia_metodo','Vector_V_Z','Vector_S_Z','Vector_L_Z','Vector_C_Z','Vector_R_Z','Calidad_Q','Coste_alostatico']
        _show_auto=_auto_table_v1554[[c for c in _show_cols if c in _auto_table_v1554.columns]].copy()
        _show_auto=_show_auto.sort_values('Fecha_hora',na_position='last')
        st.dataframe(_show_auto,use_container_width=True,hide_index=True)
        st.download_button(
            "Descargar Control Autonómico CSV",
            _auto_table_v1554.to_csv(index=False).encode('utf-8'),
            file_name='control_autonomico_v1555.csv',mime='text/csv'
        )


with tab11:
    st.subheader("Motor predictivo fisiológico híbrido v15.3.5 · topología HVG longitudinal")
    st.info(
        "**Nivel 1** calcula siempre una tendencia robusta, trazable e interpretable. "
        "**Nivel 2** aprende con transiciones longitudinales y corrige únicamente los índices "
        "en los que XGBoost demuestra una mejora fuera de muestra. Si no hay suficientes "
        "datos o no mejora la validación temporal, la predicción permanece íntegramente en Nivel 1."
    )
    st.info(
        "La predicción puede limitarse a los archivos activos de la sesión, usar todo el historial del paciente "
        "o una selección manual de registros. Por defecto se utilizan exclusivamente los archivos cargados/activos, "
        "evitando que observaciones antiguas entren en la predicción sin indicarlo."
    )

    source_mode = st.radio(
        "Fuente de datos predictivos",
        [
            "Solo archivos cargados actualmente",
            "Historial completo del paciente",
            "Selección manual de registros históricos",
        ],
        index=0,
        horizontal=True,
        key="v1529_prediction_source",
    )

    # v15.2.11: la fuente de sesión se construye DIRECTAMENTE desde los índices
    # calculados en esta ejecución, sin depender de que SQLite los haya recuperado.
    # Esto evita que sólo aparezca el último/primer registro guardado.
    _history_all_v1529 = history_df_v140.copy()
    if not _history_all_v1529.empty:
        _history_all_v1529['Paciente_ID'] = _history_all_v1529['Paciente_ID'].map(normalize_patient_id)
        _history_all_v1529['Registro'] = _history_all_v1529['Registro'].astype(str).map(sanitize_name)
        _history_all_v1529 = _history_all_v1529.drop_duplicates(['Registro', 'Fase'], keep='last')

    def _current_session_history_v15211():
        rows = []
        # 1) Usa todas las filas de índices calculadas normalmente.
        if indices_df is not None and not indices_df.empty:
            for _, _row in indices_df.iterrows():
                _rec = sanitize_name(_row.get('Registro', ''))
                _phase = str(_row.get('Fase', '') or '')
                if not _rec or not _phase:
                    continue
                _out = {
                    'Paciente_ID': normalize_patient_id(infer_patient_id(_rec)),
                    'Registro': _rec,
                    'Fase': _phase,
                    'Fecha_hora': _extract_record_datetime(_rec),
                    'saved_at': datetime.now(timezone.utc).isoformat(),
                }
                for _c in V15_PREDICTIVE_TARGETS:
                    _out[_c] = pd.to_numeric(_row.get(_c), errors='coerce')
                for _c in V1548_PRIMARY_FEATURES:
                    _out[_c] = pd.to_numeric(_row.get(_c), errors='coerce')
                rows.append(_out)

        present = {r['Registro'] for r in rows}
        fallback_phase = (active_phases[0] if active_phases else 'Basal')

        # 2) Si un archivo cargado no produjo ninguna fila por falta de ventana activa,
        # calcula automáticamente la ventana disponible; si no existe, usa el registro
        # completo como análisis basal provisional para no excluir silenciosamente el archivo.
        for _rec, _data in record_data.items():
            _rec_s = sanitize_name(_rec)
            if _rec_s in present:
                continue
            _wins = get_record_windows(global_windows_safe, record_windows_safe, _rec, use_independent)
            _w = _wins.get(fallback_phase) if isinstance(_wins, dict) else None
            if _w is not None:
                _seg = cut_segment(_data['rr'], float(_w[0]), float(_w[1]))
                _fallback_kind = 'ventana activa'
            else:
                _seg = np.asarray(_data['rr'], dtype=float)
                _fallback_kind = 'registro completo provisional'
            if len(_seg) < int(min_rr):
                continue
            try:
                _metrics = calculate_all(
                    _seg, include_rqa=include_rqa, include_hvg=include_hvg,
                    mse_zero_policy=st.session_state.get('mse_zero_policy', 'nan'),
                    theiler_window=st.session_state.get('sampen_theiler_window', 0),
                    radius_mode=st.session_state.get('mse_radius_mode', 'fixed_entropy_sd')
                )
                _idx = physiological_indices_from_row(pd.Series(_metrics))
                _out = {
                    'Paciente_ID': normalize_patient_id(infer_patient_id(_rec_s)),
                    'Registro': _rec_s,
                    'Fase': fallback_phase,
                    'Fecha_hora': _extract_record_datetime(_rec_s),
                    'saved_at': datetime.now(timezone.utc).isoformat(),
                    '_Fuente_calculo': _fallback_kind,
                }
                for _c in V15_INDEX_FEATURES:
                    _out[_c] = pd.to_numeric(_idx.get(_c), errors='coerce')
                for _c in V15_HVG_FEATURES:
                    _out[_c] = pd.to_numeric(_metrics.get(_c), errors='coerce')
                for _c in V1548_PRIMARY_FEATURES:
                    if _c == 'Artefactos_pct':
                        _out[_c] = float((_data.get('artifact_info') or {}).get('percent_artifacts',0.0))
                    else:
                        _out[_c] = pd.to_numeric(_metrics.get(_c), errors='coerce')
                rows.append(_out)
            except Exception:
                pass

        _df = pd.DataFrame(rows)
        if not _df.empty:
            _df = _df.drop_duplicates(['Registro', 'Fase'], keep='last')
            _df['Paciente_ID'] = _df['Paciente_ID'].map(normalize_patient_id)
            _df['Registro'] = _df['Registro'].astype(str).map(sanitize_name)
            _df['Fecha_hora'] = pd.to_datetime(_df['Fecha_hora'], errors='coerce')
        return _df

    _history_active_v1529 = _current_session_history_v15211()
    _active_record_names_v1529 = {sanitize_name(r) for r in record_data.keys()}

    if source_mode == "Solo archivos cargados actualmente":
        _source_history_v1529 = _history_active_v1529
    else:
        _source_history_v1529 = _history_all_v1529

    _patient_pool_v1529 = (
        sorted(_source_history_v1529['Paciente_ID'].dropna().astype(str).map(normalize_patient_id).unique())
        if not _source_history_v1529.empty else []
    )

    st.markdown('### Nivel 2 · XGBoost · autoaprendizaje condicionado')
    _cfg1,_cfg2,_cfg3=st.columns(3)
    with _cfg1:
        _min_ml_v153 = st.number_input('Transiciones totales mínimas', min_value=50, max_value=5000,
                                       value=V153_MIN_EXAMPLES_DEFAULT, step=25, key='v153_min_examples')
    with _cfg2:
        _new_batch_v153 = st.number_input('Transiciones nuevas para reentrenar', min_value=1, max_value=1000,
                                          value=V153_NEW_TRANSITIONS_DEFAULT, step=5, key='v153_new_batch')
    with _cfg3:
        _auto_train_v153 = st.toggle('Entrenamiento automático condicionado', value=True, key='v153_auto_train')

    _force_train_v153 = st.button('Forzar entrenamiento y validación ahora', key='v153_force_train')
    with st.spinner('Evaluando transiciones y, si corresponde, entrenando un candidato...'):
        _auto_result_v153 = maybe_auto_train_v153(
            _history_all_v1529, int(_min_ml_v153), int(_new_batch_v153),
            force=bool(_force_train_v153)
        ) if (_auto_train_v153 or _force_train_v153) else {
            'attempted':False,'promoted':False,'n_total':len(_v153_training_examples(_history_all_v1529)[0]),
            'n_new':max(0,len(_v153_training_examples(_history_all_v1529)[0])-int(get_v153_training_state().get('n_transitions_trained') or 0)),
            'message':'Entrenamiento automático desactivado.','active':load_v153_level2()
        }
    _bundle_v153=_auto_result_v153.get('active') or load_v153_level2()
    _state_v153=get_v153_training_state()
    st.caption('Motor Nivel 2: XGBRegressor + XGBClassifier calibrado · horizonte temporal explícito (días) · selección inteligente en Train · 70% Train / 15% Validation / 15% Test final intocable · v15.5.2 separa la promoción del regresor y de la capa probabilística: el regresor solo si TEST mejora ≥2%; la probabilidad solo si Brier TEST supera baseline y, si un candidato posterior no calibra, se conserva la última capa probabilística validada. Walk-Forward temporal adicional.')
    if not XGBOOST_AVAILABLE:
        st.error('XGBoost no está disponible en este despliegue. Añade `xgboost` a requirements.txt; el Nivel 1 sigue funcionando sin cambios.')
    _cml1,_cml2,_cml3,_cml4,_cml5=st.columns(5)
    _cml1.metric('Transiciones disponibles',int(_auto_result_v153.get('n_total',0)))
    _cml2.metric('Transiciones nuevas',int(_auto_result_v153.get('n_new',0)),delta=f"umbral {int(_new_batch_v153)}")
    _cml3.metric('Usadas por modelo activo',int(_state_v153.get('n_transitions_trained') or 0))
    _cml4.metric('Salidas ML validadas',len(_bundle_v153.get('active_targets',[])) if _bundle_v153 else 0)
    _cml5.metric('Estado',str(_state_v153.get('last_status') or ('Activo' if _bundle_v153 else 'Acumulando')).replace('_',' ').title())
    if _auto_result_v153.get('attempted'):
        (st.success if _auto_result_v153.get('promoted') else st.warning)(_auto_result_v153.get('message',''))
    else:
        st.caption(_auto_result_v153.get('message',''))

    with st.expander('Persistencia y control del modelo',expanded=False):
        st.write(f'**Directorio de persistencia:** `{V153_STORAGE_DIR}`')
        st.caption('v15.4.4 Historical Record Manager: PostgreSQL se consulta al iniciar o al recuperar manualmente; durante la navegación se usa la caché SQLite/RAM y solo se suben cambios persistentes reales.')
        _active_prob_v1551 = bool((_bundle_v153.get('probability_model') or {}).get('validated')) if _bundle_v153 else False
        _pinned_v1551 = _v1551_imported_model_state()
        _prob_source_v1552 = (_bundle_v153.get('probability_layer_source') or {}) if _bundle_v153 else {}
        if _bundle_v153:
            _reg_n_v1552 = int(_bundle_v153.get('n_examples',0) or 0)
            _prob_n_v1552 = int(_prob_source_v1552.get('source_n_examples', _reg_n_v1552) or 0) if _active_prob_v1551 else 0
            _prob_origin_v1552 = (
                'última capa probabilística validada conservada'
                if _prob_source_v1552.get('mode') == 'preserved_last_validated'
                else ('mismo bundle activo' if _active_prob_v1551 else 'sin capa validada')
            )
            st.caption(
                f"Modelo regresivo activo: {_reg_n_v1552} transiciones · "
                f"Capa probabilística calibrada: {'Sí' if _active_prob_v1551 else 'No'}"
                + (f" · origen probabilístico: {_prob_n_v1552} transiciones ({_prob_origin_v1552})" if _active_prob_v1551 else '')
                + f" · origen regresor: {'importación manual fijada' if _pinned_v1551.get('bundle') is not None else 'persistencia activa'}"
            )
            if _active_prob_v1551 and _prob_source_v1552.get('mode') == 'preserved_last_validated':
                st.info('v15.5.2: el regresor XGBoost y la probabilidad calibrada se promocionan de forma independiente. Un regresor nuevo no puede borrar la última capa probabilística que superó su propio TEST.')
        _pc1,_pc2,_pc3=st.columns(3)
        if _bundle_v153:
            _model_bio=io.BytesIO(); joblib.dump(_bundle_v153,_model_bio)
            _pc1.download_button('Descargar modelo XGBoost activo',_model_bio.getvalue(),file_name='motor_hibrido_v1550_xgboost_horizonte_activo.joblib',mime='application/octet-stream')
        if LONGITUDINAL_DB_PATH.exists():
            _pc2.download_button('Descargar SQLite longitudinal',LONGITUDINAL_DB_PATH.read_bytes(),file_name='vrc_longitudinal.sqlite3',mime='application/octet-stream')
        _uploaded_model_v153=_pc3.file_uploader('Importar modelo XGBoost validado',type=['joblib','pkl'],key='v153_import_model')
        if _uploaded_model_v153 is not None and st.button('Validar e importar modelo',key='v153_confirm_import'):
            try:
                _bundle_v153=import_v153_model_bytes(_uploaded_model_v153.getvalue())
                _prob_ok=bool((_bundle_v153.get('probability_model') or {}).get('validated'))
                _pm_imp=_bundle_v153.get('probability_metrics') or {}
                st.success(
                    f"Modelo importado, fijado como activo y verificado en Supabase. "
                    f"Transiciones: {int(_bundle_v153.get('n_examples',0))} · "
                    f"Probabilidad calibrada: {'Sí' if _prob_ok else 'No'}"
                )
                if _prob_ok:
                    st.caption(
                        f"Brier Test: {pd.to_numeric(_pm_imp.get('brier_test'),errors='coerce'):.4f} · "
                        f"Baseline: {pd.to_numeric(_pm_imp.get('brier_baseline_test'),errors='coerce'):.4f} · "
                        f"Mejora: {pd.to_numeric(_pm_imp.get('mejora_brier_vs_baseline_pct'),errors='coerce'):.2f}%"
                    )
                st.rerun()
            except Exception as exc:
                st.error(f'No se pudo importar: {exc}')
        with sqlite3.connect(LONGITUDINAL_DB_PATH) as _conruns:
            _runs_v153=pd.read_sql_query('SELECT * FROM model_training_runs ORDER BY run_id DESC LIMIT 20',_conruns)
        if not _runs_v153.empty:
            st.dataframe(_runs_v153,use_container_width=True,hide_index=True)

    if _bundle_v153:
        _metrics_v153=pd.DataFrame(_bundle_v153.get('metrics',{})).T.reset_index().rename(columns={'index':'Índice'})
        if not _metrics_v153.empty:
            with st.expander('Validación temporal rigurosa · 70/15/15 + Walk-Forward',expanded=False):
                st.dataframe(_metrics_v153.round(3),use_container_width=True,hide_index=True)
        with st.expander('Variables utilizadas por XGBoost',expanded=False):
            st.caption('La selección se aprende únicamente con el 70% Train. Se eliminan variables sin información, constantes/casi constantes y redundancias con |ρ Spearman| ≥ 0.985. El Test final no participa en la selección.')
            _target_fs = st.selectbox('Salida XGBoost', list(_bundle_v153.get('models',{}).keys()), key='v1548_feature_target')
            _spec_fs = _bundle_v153.get('models',{}).get(_target_fs,{})
            _used_fs = _spec_fs.get('features',[])

            def _v15410_feature_family(_name):
                _n=str(_name)
                _u=_n.upper()
                if _u.startswith('IDX_'):
                    return 'Índices fisiológicos'
                if _u.startswith('HVG_'):
                    return 'HVG / topología'
                if any(_u.startswith(p) for p in ('REC','DET','LMEAN','LMAX')):
                    return 'RQA / recurrencia'
                if any(p in _u for p in ('DFA_ALPHA1','DFA_ALPHA2','HURST','LYAPUNOV','D2')):
                    return 'Fractal / dinámica no lineal'
                if any(p in _u for p in ('SAMPEN','APEN','MSE','MDE','DISPEN','SHANEN','ENTROP')):
                    return 'Entropía / complejidad'
                if any(_u.startswith(p) for p in ('SDNN','RMSSD','SD1','SD2','PNN50','NN50','MEANHR','MEANRR')):
                    return 'HRV temporal'
                if any(_u.startswith(p) for p in ('VLF','LF_','HF_','LFHF','LF_HF','TOTAL_POWER','TOTAL')):
                    return 'HRV frecuencial'
                if any(p in _u for p in ('WAV','WAVELET','DOM_PCT','EPISODES_','TRANSITIONS_')):
                    return 'Wavelet'
                if any(p in _u for p in ('N_RRI','DURATION','ARTEFACT','ARTIFACT','CORRECTION','CORREGID')):
                    return 'Calidad / duración'
                if any(p in _u for p in ('HORIZON','PHASE','FASE','TIME','INTERVAL')):
                    return 'Contexto temporal'
                return 'Otros'

            def _v15410_feature_dynamic(_name):
                _n=str(_name)
                _map=(
                    ('__actual_dyn','Actual'),('__delta_ultimo','Δ último'),('__media3','Media 3'),
                    ('__pendiente','Pendiente'),('__variabilidad_reciente','Variabilidad reciente'),
                    ('__pred_n1','Predicción Nivel 1'),('__delta_n1','Δ Nivel 1'),('__unc_n1','Incertidumbre Nivel 1'),
                    ('__actual','Actual')
                )
                for _s,_lab in _map:
                    if _n.endswith(_s):
                        return _lab
                return 'Directa / contextual'

            _pretty_target_fs=str(_target_fs).replace('IDX_','').replace('_',' ')
            _info1,_info2=st.columns([1,2])
            _info1.markdown(f'**Variable objetivo:** `{_target_fs}`')
            _info2.markdown('**Predictores permitidos:** modelo fisiológico multivariado cruzado completo (Índices + HRV + RQA + fractales + entropías + Wavelet + HVG + calidad/duración + dinámica longitudinal).')
            st.info('La variable objetivo indica QUÉ intenta predecir XGBoost. Los predictores pueden proceder de cualquier familia fisiológica: que aparezcan variables de Complejidad, RQA, DFA o HVG al predecir Adaptabilidad es deliberado y no significa que el desplegable mezcle modelos.')
            st.write(f'**Variables iniciales:** {len(_bundle_v153.get("features",[]))} · **utilizadas tras selección:** {len(_used_fs)}')

            _imp_fs = pd.DataFrame(_spec_fs.get('feature_importance',[]))
            if not _imp_fs.empty:
                _imp_fs=_imp_fs.copy()
                _imp_fs['Importancia']=pd.to_numeric(_imp_fs['Importancia'],errors='coerce').fillna(0.0)
                _imp_fs['Familia']=_imp_fs['Variable'].map(_v15410_feature_family)
                _imp_fs['Dinámica']=_imp_fs['Variable'].map(_v15410_feature_dynamic)
                _positive_fs=_imp_fs[_imp_fs['Importancia']>0].sort_values('Importancia',ascending=False).copy()
                _n_pos=len(_positive_fs)
                st.caption(f'Con importancia XGBoost > 0: {_n_pos} · seleccionadas pero sin uso efectivo en los árboles: {max(0,len(_used_fs)-_n_pos)}')

                if not _positive_fs.empty:
                    _fam_fs=(_positive_fs.groupby('Familia',as_index=False)['Importancia'].sum().sort_values('Importancia',ascending=False))
                    _tot_imp=float(_fam_fs['Importancia'].sum())
                    _fam_fs['Contribución_%']=np.where(_tot_imp>0,100.0*_fam_fs['Importancia']/_tot_imp,0.0)
                    st.markdown(f'#### Importancia total por familias para predecir **{_pretty_target_fs}**')
                    _fc1,_fc2=st.columns([1.15,1.85])
                    with _fc1:
                        st.dataframe(_fam_fs[['Familia','Contribución_%']].assign(**{'Contribución_%':lambda d:d['Contribución_%'].round(1)}),use_container_width=True,hide_index=True)
                    with _fc2:
                        try:
                            _fig_fam=px.bar(_fam_fs.iloc[::-1],x='Contribución_%',y='Familia',orientation='h',title=f'Contribución por familia · {_target_fs}')
                            _fig_fam.update_layout(height=max(330,42*len(_fam_fs)),xaxis_title='% de la importancia total',yaxis_title='Familia')
                            st.plotly_chart(_fig_fam,use_container_width=True,key=f'v15410_family_importance_{_target_fs}')
                        except Exception as _fam_exc:
                            st.warning(f'No se pudo dibujar el resumen por familias: {_fam_exc}')

                    st.markdown('#### Predictores individuales con mayor importancia')
                    _top_show=_positive_fs.head(40).copy()
                    st.dataframe(_top_show[['Variable','Familia','Dinámica','Importancia']].assign(Importancia=lambda d:d['Importancia'].round(5)),use_container_width=True,hide_index=True)
                    try:
                        _plot_fs=_positive_fs.head(20).iloc[::-1]
                        _fig_imp=px.bar(_plot_fs,x='Importancia',y='Variable',orientation='h',title=f'Importancia XGBoost · {_target_fs}',hover_data=['Familia','Dinámica'])
                        _fig_imp.update_layout(height=max(420,28*min(20,len(_plot_fs))))
                        st.plotly_chart(_fig_imp,use_container_width=True,key=f'v15410_feature_importance_plot_{_target_fs}')
                    except Exception as _imp_exc:
                        st.warning(f'No se pudo dibujar el gráfico de importancia XGBoost, pero la tabla y el modelo permanecen disponibles: {_imp_exc}')
                else:
                    st.warning('Para esta salida, las variables seleccionadas presentan importancia XGBoost nula. Esto no cambia el modelo; indica que no hubo divisiones útiles atribuibles a esas variables en el modelo almacenado.')
                    st.dataframe(_imp_fs.head(40)[['Variable','Familia','Dinámica','Importancia']],use_container_width=True,hide_index=True)
            _drop_fs=_spec_fs.get('dropped_features',{})
            st.caption(f'Descartadas por datos insuficientes: {len(_drop_fs.get("insufficient",[]))} · constantes: {len(_drop_fs.get("constant",[]))} · correlación/redundancia: {len(_drop_fs.get("correlated",[]))}')

    if not _patient_pool_v1529:
        st.warning("La fuente seleccionada todavía no contiene observaciones fisiológicas calculadas.")
    else:
        pa = st.selectbox('Paciente/serie para predecir', _patient_pool_v1529, key='v15_patient_v1529')
        pa = normalize_patient_id(pa)
        st.caption(f"Serie canónica usada por el motor: {pa}")
        _patient_history_v1529 = _source_history_v1529[
            _source_history_v1529['Paciente_ID'].map(normalize_patient_id) == pa
        ].copy()

        if source_mode == "Selección manual de registros históricos":
            _available_records_v1529 = (
                _patient_history_v1529[['Registro', 'Fecha_hora']]
                .drop_duplicates('Registro')
                .sort_values('Fecha_hora', na_position='last')['Registro']
                .astype(str).tolist()
            )
            _manual_default_v1529 = [r for r in _available_records_v1529 if r in _active_record_names_v1529] or _available_records_v1529
            _manual_key_v1529 = 'v1529_manual_prediction_records'
            # Elimina del estado selecciones antiguas que ya no existen entre las opciones actuales.
            if _manual_key_v1529 in st.session_state:
                _valid_state_records_v1529 = [
                    r for r in st.session_state.get(_manual_key_v1529, [])
                    if r in _available_records_v1529
                ]
                st.session_state[_manual_key_v1529] = _valid_state_records_v1529
            _selected_records_v1529 = st.multiselect(
                "Registros históricos incluidos",
                _available_records_v1529,
                default=_manual_default_v1529,
                format_func=lambda r: _record_axis_label(r, include_seconds=True, multiline=False),
                key=_manual_key_v1529,
            )
            _patient_history_v1529 = _patient_history_v1529[
                _patient_history_v1529['Registro'].isin(_selected_records_v1529)
            ].copy()

        _used_records_v1529 = (
            _patient_history_v1529[['Registro', 'Fecha_hora']]
            .drop_duplicates('Registro')
            .sort_values('Fecha_hora', na_position='last')
        )

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Observaciones usadas", len(_patient_history_v1529))
        series_n=(_patient_history_v1529.groupby(['Paciente_ID','Fase']).size()>=2).sum() if not _patient_history_v1529.empty else 0
        c2.metric("Series con trayectoria", int(series_n))
        c3.metric("Registros usados", int(len(_used_records_v1529)), f"de {len(record_data)} cargados" if source_mode.startswith('Solo') else None)
        c4.metric("Fuente", "Sesión" if source_mode.startswith('Solo') else ("Historial" if source_mode.startswith('Historial') else "Manual"))

        if source_mode.startswith('Solo'):
            _included = set(_used_records_v1529['Registro'].astype(str)) if not _used_records_v1529.empty else set()
            _missing = sorted(_active_record_names_v1529 - _included)
            if _missing:
                st.warning(
                    f"{len(_missing)} archivo(s) cargado(s) no han entrado en la serie seleccionada. "
                    "Comprueba su identificador de paciente, fase asignada e índices calculados: " + ", ".join(_missing)
                )
            else:
                st.success(f"Se están utilizando los {len(_included)} archivos cargados actualmente.")

        with st.expander("Registros utilizados en esta predicción", expanded=True):
            if _used_records_v1529.empty:
                st.warning("No hay registros válidos en la selección.")
            else:
                _used_show_v1529 = _used_records_v1529.copy()
                _used_n_v1543 = len(_used_show_v1529)
                if _used_n_v1543 > 40:
                    _upp, _upnp, _up0, _up1 = _record_page_controls(_used_n_v1543, "prediction_used_records_v1543", page_size=40, label="registros")
                    _used_show_page_v1543 = _used_show_v1529.iloc[_up0:_up1].copy()
                    st.caption(f"Predicción: se usan los {_used_n_v1543} registros. La tabla muestra {_up0+1}–{_up1} para mantener la interfaz ligera.")
                else:
                    _used_show_page_v1543 = _used_show_v1529
                _used_show_page_v1543['Fecha'] = pd.to_datetime(_used_show_page_v1543['Fecha_hora'], errors='coerce').dt.strftime('%d/%m/%Y %H:%M:%S')
                st.dataframe(_used_show_page_v1543[['Fecha', 'Registro']], use_container_width=True, hide_index=True)
                if source_mode == "Solo archivos cargados actualmente":
                    st.caption("La predicción se ha aislado de cualquier observación histórica que no esté activa en la sesión.")

        if st.button("Recalcular predicción con esta fuente", key='v1529_recalculate_prediction'):
            st.rerun()

        # v15.4.12 · Selector temporal explícito y lectura probabilística en el mismo bloque
        with st.expander('Probabilidad calibrada de evolución favorable a X días', expanded=True):
            _pspec=(_bundle_v153.get('probability_model') or {}) if _bundle_v153 else {}
            _pm=(_bundle_v153.get('probability_metrics') or _pspec.get('metrics',{})) if _bundle_v153 else {}

            _horizon_days_v15412 = st.select_slider(
                'Horizonte temporal',
                options=[3,7,14,30],
                value=7,
                key='v15412_horizon_days',
                help='Selecciona el momento futuro al que se refiere la probabilidad. No significa que la tendencia vaya a mantenerse continuamente durante todo el intervalo.'
            )

            _hs=_pspec.get('horizon_support_days',{}) if _pspec else {}
            _h=float(_horizon_days_v15412)
            if _hs and _hs.get('p10') is not None:
                if _h < float(_hs.get('min',_h)) or _h > float(_hs.get('max',_h)):
                    _support_label='Extrapolación fuera del rango observado'
                    _support_level='Muy limitado'
                elif _h < float(_hs.get('p10',_h)) or _h > float(_hs.get('p90',_h)):
                    _support_label='Soporte temporal limitado'
                    _support_level='Limitado'
                else:
                    _support_label='Dentro del soporte temporal principal'
                    _support_level='Adecuado'
            else:
                _support_label='Soporte temporal no estimable'
                _support_level='No estimable'

            _phase_options_v15412=[]
            if not _patient_history_v1529.empty and 'Fase' in _patient_history_v1529.columns:
                _phase_options_v15412=[str(x) for x in _patient_history_v1529['Fase'].dropna().astype(str).unique().tolist()]
            _phase_v15412=st.selectbox('Fase/serie para estimar la probabilidad', _phase_options_v15412 or ['Basal'], key='v15412_probability_phase')

            _prob_v15412=predict_v15411_favorable_probability(
                _patient_history_v1529, pa, _phase_v15412, float(_horizon_days_v15412), bundle=_bundle_v153
            ) if (_bundle_v153 and not _patient_history_v1529.empty) else {'available':False,'reason':'No hay histórico válido o modelo probabilístico activo.'}

            _validated=bool(_pspec and _pspec.get('validated'))
            _brier_imp=float(_pm.get('mejora_brier_vs_baseline_pct',np.nan)) if _pm else np.nan
            _auc=float(_pm.get('auc_test',np.nan)) if _pm else np.nan
            if not _validated:
                _cal_label='No validada'
            elif _support_level=='Muy limitado':
                _cal_label='Validación global, pero fuera de soporte temporal'
            elif _support_level=='Limitado':
                _cal_label='Validada · soporte temporal limitado'
            elif np.isfinite(_brier_imp) and _brier_imp>=10 and np.isfinite(_auc) and _auc>=0.65:
                _cal_label='Validada · soporte fuerte'
            else:
                _cal_label='Validada · soporte moderado'

            _h1,_h2,_h3,_h4=st.columns(4)
            _h1.metric('Horizonte solicitado',f'{int(_horizon_days_v15412)} días')
            if _prob_v15412.get('available'):
                _h2.metric('Probabilidad favorable',f"{_prob_v15412.get('probability_pct',np.nan):.1f}%")
            else:
                _h2.metric('Probabilidad favorable','No calibrada')
            _h3.metric('Soporte temporal',_support_level)
            _h4.metric('Calibración',_cal_label)

            if _hs:
                st.caption(
                    f"Soporte aprendido: p10–p90 ≈ {_hs.get('p10',np.nan):.1f}–{_hs.get('p90',np.nan):.1f} días · "
                    f"mediana {_hs.get('median',np.nan):.1f} días · rango observado {_hs.get('min',np.nan):.1f}–{_hs.get('max',np.nan):.1f} días. "
                    f"Para {int(_horizon_days_v15412)} días: **{_support_label}**."
                )

            if _validated:
                st.success('Capa probabilística validada en Test final: el Brier score calibrado supera a la predicción basal por prevalencia.')
                _pp1,_pp2,_pp3,_pp4=st.columns(4)
                _pp1.metric('Brier Test',f"{_pm.get('brier_test',np.nan):.3f}",f"baseline {_pm.get('brier_baseline_test',np.nan):.3f}")
                _pp2.metric('Mejora Brier',f"{_pm.get('mejora_brier_vs_baseline_pct',np.nan):.1f}%")
                _pp3.metric('AUC Test',f"{_pm.get('auc_test',np.nan):.3f}" if np.isfinite(_pm.get('auc_test',np.nan)) else '—')
                _pp4.metric('Prevalencia Train',f"{_pm.get('prevalencia_train_pct',np.nan):.1f}%")
            else:
                st.warning('La capa probabilística todavía no cumple los requisitos de calibración fuera de muestra y diversidad temporal suficiente.')

            if _prob_v15412.get('available'):
                st.info(
                    f"Interpretación: para **{pa}**, fase **{_phase_v15412}**, el modelo estima una probabilidad calibrada de "
                    f"**{_prob_v15412.get('probability_pct',np.nan):.1f}%** de evolución fisiológica favorable a **{int(_horizon_days_v15412)} días**. "
                    f"Favorable = Δ del compuesto fisiológico > +3 puntos. Esto describe el estado esperado en ese horizonte, no que la tendencia se mantenga de forma continua durante todos esos días."
                )
            else:
                st.caption(_prob_v15412.get('reason','La probabilidad calibrada no está disponible para esta selección.'))

            st.caption('Entrenamiento: XGBClassifier en 70% Train · calibración Platt solo en 15% Validation · evaluación en 15% Test final intocable · horizonte_dias entra explícitamente como predictor.')

        if _patient_history_v1529.empty:
            st.warning("No hay observaciones registro-fase válidas para la fuente y el paciente seleccionados.")
        else:
            evo_fig,evo_df=v15_current_evolution_figure(_patient_history_v1529,pa)
            st.markdown('### Evolución fisiológica actual')
            if evo_fig is not None:
                st.plotly_chart(evo_fig,use_container_width=True,key='v15_current_evolution_v1529')
                show_cols=['Registro','Fase','Fecha_hora']+[c for c in V15_INDEX_FEATURES if c in evo_df.columns]
                st.dataframe(evo_df[show_cols].round(3),use_container_width=True,hide_index=True)

            st.markdown('### Horizonte temporal de predicción')
            st.caption(f"Horizonte activo: **{int(_horizon_days_v15412)} días**. Se selecciona en el bloque «Probabilidad calibrada de evolución favorable a X días» situado arriba.")

            predictions15, summary15, details15 = predict_v15_all_phases(_patient_history_v1529, pa, horizon_days=float(_horizon_days_v15412))
            st.markdown('### Predicciones simultáneas de todas las fases')
            if summary15.empty:
                st.warning('No hay fases con índices válidos para esta selección.')
            else:
                st.plotly_chart(v15_multiphase_summary_figure(summary15), use_container_width=True, key='v15_multiphase_summary_v1529')

                phases_list=list(summary15['Fase'])
                for row_start in range(0,len(phases_list),4):
                    cols=st.columns(4)
                    for col,phase in zip(cols,phases_list[row_start:row_start+4]):
                        r=summary15[summary15['Fase']==phase].iloc[0]
                        with col:
                            st.markdown(f"#### {phase}")
                            st.metric('Estado',str(r['Estado_previsto']),f"Δ {r['Delta_compuesto']:+.1f}")
                            st.metric('Compuesto previsto',f"{r['Compuesto_predicho']:.1f}",f"Actual {r['Compuesto_actual']:.1f}")
                            st.metric('Confianza',f"{r['Confianza_pct']:.0f}%",f"{int(r['Registros'])} registro(s)")
                            st.caption(f"{r['Modo']} · Novedad: {r['Novedad']}")

                summary_show=summary15.copy()
                summary_show['Novedad_score']=summary_show['Novedad_score'].apply(lambda x: round(x,3) if np.isfinite(x) else np.nan)
                st.dataframe(summary_show.round(3),use_container_width=True,hide_index=True)
                st.plotly_chart(v15_multiphase_indices_figure(details15),use_container_width=True,key='v15_multiphase_indices_v1529')

                st.markdown('### Detalle por fase')
                for phase,pred15 in predictions15.items():
                    with st.expander(f"{phase} · {pred15['estado_previsto']} · confianza {pred15['confianza']:.0f}%",expanded=False):
                        a,b,c,d,e=st.columns(5)
                        a.metric('Estado previsto',pred15['estado_previsto'],f"Δ compuesto {pred15['delta_compuesto']:+.1f}")
                        b.metric('Confianza metodológica',f"{pred15['confianza']:.0f}%",f"{pred15['n_historial']} registros")
                        c.metric('Compuesto previsto',f"{pred15['compuesto_predicho']:.1f}",f"Actual {pred15['compuesto_actual']:.1f}")
                        anomaly_text=(f"{pred15['anomalia_score']:.2f}" if np.isfinite(pred15['anomalia_score']) else '—')
                        d.metric('Novedad del último estado',pred15['anomalia_label'],anomaly_text)
                        _pf=pred15.get('probabilidad_favorable') or {}
                        if _pf.get('available'):
                            e.metric(f"P(favorable · {int(round(_pf.get('horizon_days',0)))} d)",f"{_pf.get('probability_pct',np.nan):.1f}%")
                            st.caption(f"Probabilidad calibrada: {_pf.get('support','')}. Favorable = Δ compuesto > +3.")
                        else:
                            e.metric('P(favorable)','No calibrada')
                            st.caption(_pf.get('reason','La capa probabilística no está disponible.'))
                        st.caption(f"Modo: {pred15['modo_prediccion']} · horizonte explícito {pred15.get('horizonte_dias',np.nan):.1f} días")
                        st.plotly_chart(v15_prediction_figure(pred15),use_container_width=True,key=f"v15_prediction_v1529_{sanitize_name(phase)}")
                        st.dataframe(pred15['tabla'].round(3),use_container_width=True,hide_index=True)
                        _topo=pred15.get('topologia_hvg')
                        if _topo:
                            st.markdown('#### Evolución topológica HVG longitudinal')
                            _t1,_t2=st.columns(2)
                            _sa=_topo.get('score_actual',np.nan); _sp=_topo.get('score_previsto',np.nan)
                            _t1.metric('Cambio topológico observado',_topo.get('descriptor_actual','No estimable'),f'score {_sa:.2f}' if np.isfinite(_sa) else '—')
                            _t2.metric('Cambio topológico previsto',_topo.get('descriptor_previsto','No estimable'),f'score {_sp:.2f}' if np.isfinite(_sp) else '—')
                            st.write(f"**Dirección observada:** {_topo.get('direccion_actual','')}")
                            st.write(f"**Dirección prevista:** {_topo.get('direccion_prevista','')}")
                            st.caption('Small-world y scale-free se conservan como contexto, pero se atenúan si están saturados o casi constantes; no dominan el descriptor.')
                            st.dataframe(_topo['tabla'].round(4),use_container_width=True,hide_index=True)

                st.download_button(
                    'Descargar resumen multifase CSV',summary15.to_csv(index=False).encode('utf-8'),
                    file_name=f'prediccion_multifase_v1534_{sanitize_name(pa)}.csv',mime='text/csv'
                )
                st.download_button(
                    'Descargar detalle de todos los índices y fases CSV',details15.to_csv(index=False).encode('utf-8'),
                    file_name=f'prediccion_indices_multifase_v1534_{sanitize_name(pa)}.csv',mime='text/csv'
                )

                if (summary15['Registros']==1).any():
                    st.warning('Las fases con un solo registro muestran un pronóstico basal de persistencia con incertidumbre amplia.')
                if ((summary15['Registros']>=2)&(summary15['Registros']<5)).any():
                    st.caption('Las fases con 2-4 registros ofrecen predicciones tempranas. La confianza aumentará con más observaciones longitudinales.')

    with st.expander('Cómo funciona el motor v15.3.5',expanded=True):
        st.markdown("""
1. **Persistencia longitudinal:** conserva cada registro-fase sin duplicados.  
2. **Espacio fisiológico:** cada observación es un vector de seis índices 0-100.  
3. **Predicción continua:** estima directamente cada índice en `t+1`; no requiere etiquetas ni dos clases.  
4. **Una observación:** pronóstico basal de persistencia con incertidumbre amplia; muestra siempre el estado actual.  
5. **Dos observaciones:** extrapolación amortiguada para evitar sobrepredicción.  
6. **Tres o más:** tendencia robusta tipo Theil–Sen simplificada combinada con el impulso más reciente.  
7. **Incertidumbre:** usa errores walk-forward, residuos y dispersión de los cambios.  
8. **Novedad:** compara el último estado con la distribución histórica personal mediante una distancia robusta basada en MAD.  
9. **Panel HVG multivariable:** usa degree mean/max, hubs, clustering, lambda, path length, diámetro y compactación respecto al propio histórico.  
10. **Reorganización topológica:** Estable / leve / moderada / marcada y dirección ↑/↓, sin asignar automáticamente bueno/malo.  
11. **Anti-techo:** Small-world y Scale-free tienen peso reducido cuando están saturados o casi constantes.  
12. **Nivel 2:** XGBoost usa/corrige estas trayectorias solo si mejora fuera de muestra.  
13. **Horizonte explícito:** cada transición conserva los días reales entre registro origen y objetivo; al predecir puedes solicitar X días.  
14. **Probabilidad favorable:** XGBClassifier aprende Favorable = Δ compuesto > +3, se calibra con Platt exclusivamente en Validation y solo se presenta como probabilidad calibrada si mejora el Brier frente a la prevalencia basal en Test final.  
15. **Estado resumen:** Favorable/Estable/Desfavorable sigue derivándose de índices fisiológicos, no de una lectura monotónica del HVG.

La probabilidad mostrada, cuando está disponible, es una probabilidad fisiológica calibrada de evolución favorable (Δ compuesto > +3) para el horizonte solicitado; no es una probabilidad de diagnóstico, evento clínico ni desenlace médico.
""")
    if LONGITUDINAL_DB_PATH.exists():
        st.download_button('Descargar base longitudinal v15',LONGITUDINAL_DB_PATH.read_bytes(),file_name='base_completa_v1529.sqlite3',mime='application/octet-stream')
    st.caption('En Streamlit Community Cloud el disco puede reiniciarse. Conserva una copia de la base o conecta almacenamiento externo.')

    st.divider()
    st.subheader("Modelo clínico supervisado opcional")
    st.warning(
        "El modelo clínico sólo produce una predicción clínica cuando se entrena con desenlaces reales y etiquetados. "
        "Los índices actuales son predictores; no sustituyen la variable clínica a siete días."
    )
    if not SKLEARN_AVAILABLE:
        st.error("Falta scikit-learn. Instala las dependencias actualizadas del paquete.")
    else:
        st.markdown("### Modelo activo")
        active_bundle = st.session_state.get('gb_bundle_v132')
        if active_bundle is not None:
            st.success(f"Modelo activo: {active_bundle.get('task','')} · objetivo: {active_bundle.get('target','')}")
            automatic_pred = predict_with_gradient_boosting(active_bundle, indices_df)
            if automatic_pred.empty:
                st.info("Carga y analiza registros RRi para aplicar automáticamente el modelo activo.")
            else:
                st.markdown("#### Inferencia automática sobre los índices calculados en esta ejecución")
                st.dataframe(automatic_pred, use_container_width=True)
                st.download_button(
                    "Descargar inferencia automática CSV",
                    automatic_pred.to_csv(index=False).encode('utf-8'),
                    file_name='inferencia_automatica_v140.csv', mime='text/csv',
                    key='download_auto_inference_v132'
                )
        else:
            st.info("No hay un modelo activo. Entrena uno o carga un archivo .joblib ya validado.")

        uploaded_model = st.file_uploader(
            "Cargar un modelo Gradient Boosting previamente entrenado (.joblib)",
            type=['joblib'], key='gb_model_upload_v132'
        )
        if uploaded_model is not None:
            try:
                loaded_bundle = load_gradient_boosting_bundle_bytes(uploaded_model.getvalue())
                st.session_state['gb_bundle_v132'] = loaded_bundle
                st.success("Modelo cargado y activado. La inferencia se aplicará directamente a los índices actuales.")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo cargar el modelo: {e}")

        template=gradient_boosting_training_template()
        st.download_button(
            "Descargar plantilla de entrenamiento CSV",
            template.to_csv(index=False).encode('utf-8'),
            file_name='plantilla_gradient_boosting_v140.csv', mime='text/csv'
        )
        uploaded_ml=st.file_uploader(
            "Sube una tabla CSV o Excel con los índices y un resultado observado",
            type=['csv','xlsx','xls'], key='gb_training_file'
        )
        if uploaded_ml is not None:
            try:
                train_df = pd.read_csv(uploaded_ml) if uploaded_ml.name.lower().endswith('.csv') else pd.read_excel(uploaded_ml)
                st.dataframe(train_df.head(20), use_container_width=True)
                candidate_targets=[c for c in train_df.columns if c not in GB_INDEX_FEATURES and c not in ['Paciente_ID','Fecha','Registro','Fase']]
                if not candidate_targets:
                    st.error("No se detecta una columna de resultado/etiqueta.")
                else:
                    c1,c2=st.columns(2)
                    with c1:
                        target_col=st.selectbox("Variable objetivo",candidate_targets,index=0)
                    with c2:
                        task_label=st.selectbox("Tipo de modelo",["Clasificación", "Regresión"])
                    task='classification' if task_label=='Clasificación' else 'regression'
                    if st.button("Entrenar y validar Gradient Boosting",type='primary'):
                        try:
                            bundle=train_gradient_boosting_model(train_df,target_col,task=task)
                            st.session_state['gb_bundle_v132']=bundle
                            st.success("Modelo entrenado y validado. Revisa el rendimiento antes de interpretar predicciones.")
                        except Exception as e:
                            st.error(f"No se pudo entrenar: {e}")
            except Exception as e:
                st.error(f"No se pudo leer la tabla: {e}")

        bundle=st.session_state.get('gb_bundle_v132')
        if bundle is not None:
            st.markdown("### Rendimiento de validación")
            st.dataframe(pd.DataFrame([bundle['metrics']]),use_container_width=True)
            if bundle.get('confusion_matrix') is not None:
                st.markdown("#### Matriz de confusión")
                st.dataframe(bundle['confusion_matrix'],use_container_width=True)
            st.markdown("### Importancia por permutación")
            st.dataframe(bundle['importance'],use_container_width=True)
            fig_imp=go.Figure(go.Bar(
                x=bundle['importance']['Importancia_media'],
                y=bundle['importance']['Índice'], orientation='h',
                error_x=dict(type='data',array=bundle['importance']['Importancia_sd'])
            ))
            fig_imp.update_layout(title='Contribución predictiva de los índices',height=420,yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp,use_container_width=True,key='gb_importance_plot')

            st.markdown("### Predicción sobre los índices cargados en la app")
            current_pred=predict_with_gradient_boosting(bundle,indices_df)
            if current_pred.empty:
                st.info("Primero calcula índices fisiológicos en la aplicación.")
            else:
                st.dataframe(current_pred,use_container_width=True)
                st.download_button(
                    "Descargar predicciones CSV", current_pred.to_csv(index=False).encode('utf-8'),
                    file_name='predicciones_gradient_boosting_v132.csv',mime='text/csv'
                )
            if st.button("Guardar como modelo activo de la aplicación", type='primary', key='save_active_model_v132'):
                try:
                    saved_path = save_active_gradient_boosting_bundle(bundle)
                    st.success(f"Modelo guardado como activo: {saved_path.name}. Se cargará automáticamente en próximos arranques locales.")
                except Exception as e:
                    st.error(f"No se pudo guardar el modelo activo: {e}")
            st.download_button(
                "Descargar modelo entrenado (.joblib)",
                serialize_gradient_boosting_bundle(bundle),
                file_name='modelo_gradient_boosting_v132.joblib',
                mime='application/octet-stream'
            )
            st.caption(
                "Uso exploratorio/investigación. Para uso clínico se requiere validación externa, calibración, "
                "control de fuga de datos por paciente y una definición previa de mejoría clínicamente relevante."
            )


with tab8:
    st.subheader("Exportar · informe científico completo v15.4.12")
    st.info(
        "La exportación completa incluye métricas HRV/no lineales, MSE/MDE, RQA, Poincaré, HVG, "
        "Atractores reconstruidos, índices fisiológicos (Vagal, Amplitud, Complejidad, Rigidez, "
        "Adaptabilidad y Regulación lenta) y la trazabilidad del motor predictivo híbrido v15.4.12: "
        "compuesto actual/predicho, dirección, incertidumbre, Nivel 1, activación real de Nivel 2, "
        "XGBoost y topología HVG longitudinal. También exporta gráficos de todos los parámetros numéricos disponibles."
    )

    if long_df.empty:
        st.info("No hay datos para exportar.")
    else:
        valid_summary = pd.DataFrame.from_dict(records_valid, orient="index").reindex(columns=PHASES).fillna(False).astype(bool)

        # v15.4.5 · Complete Scientific Export
        # Se preparan también atractores, índices fisiológicos y toda la trazabilidad del motor v15.3.x.
        attractor_export_df, attractor_export_figures = build_attractor_export_bundle(record_data, records, max_points=1200)
        (prediction_export_summary, prediction_export_details, prediction_export_topology,
         prediction_model_state, prediction_export_figures, prediction_model_metrics,
         prediction_model_targets) = build_prediction_export_bundle(history_df_v140, records=records)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            xlsx = tmpdir / "resultados_hrv_comparativa.xlsx"
            csv = tmpdir / "resultados_hrv_comparativa.csv"
            zipf = tmpdir / "resultados_hrv_comparativa.zip"

            long_df.to_csv(csv, index=False)

            with pd.ExcelWriter(xlsx) as writer:
                long_df.to_excel(writer, sheet_name="metricas", index=False)
                valid_summary.to_excel(writer, sheet_name="ventanas_validas")
                if not indices_df.empty:
                    indices_df.to_excel(writer, sheet_name="indices_fisiologicos", index=False)
                    physiological_indices_reference_table().to_excel(writer, sheet_name="indices_referencia", index=False)
                if attractor_export_df is not None and not attractor_export_df.empty:
                    attractor_export_df.to_excel(writer, sheet_name="atractores", index=False)
                if prediction_export_summary is not None and not prediction_export_summary.empty:
                    prediction_export_summary.to_excel(writer, sheet_name="prediccion_resumen", index=False)
                if prediction_export_details is not None and not prediction_export_details.empty:
                    prediction_export_details.to_excel(writer, sheet_name="prediccion_indices", index=False)
                if prediction_export_topology is not None and not prediction_export_topology.empty:
                    prediction_export_topology.to_excel(writer, sheet_name="prediccion_HVG", index=False)
                if prediction_model_state is not None and not prediction_model_state.empty:
                    _pstate_xlsx = prediction_model_state.copy()
                    for _c in _pstate_xlsx.columns:
                        _pstate_xlsx[_c] = _pstate_xlsx[_c].map(lambda v: json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict,list,tuple,set)) else v)
                    _pstate_xlsx.to_excel(writer, sheet_name="modelo_nivel2_estado", index=False)
                if prediction_model_metrics is not None and not prediction_model_metrics.empty:
                    prediction_model_metrics.to_excel(writer, sheet_name="modelo_nivel2_metricas", index=False)
                if prediction_model_targets is not None and not prediction_model_targets.empty:
                    prediction_model_targets.to_excel(writer, sheet_name="modelo_nivel2_salidas", index=False)

                rows_w = []
                for rec in records:
                    w = get_record_windows(st.session_state.global_windows_v50, st.session_state.record_windows_v50, rec, use_independent)
                    for ph in PHASES:
                        ww = w.get(ph)
                        rows_w.append({
                            "Registro": rec,
                            "Fase": ph,
                            "Inicio": sec_to_hms(ww[0]) if ww else "",
                            "Fin": sec_to_hms(ww[1]) if ww else "",
                            "Duracion_min": (ww[1] - ww[0]) / 60 if ww else np.nan,
                            "Activa": ph in active_phases,
                        })
                pd.DataFrame(rows_w).to_excel(writer, sheet_name="ventanas", index=False)

                artifact_rows = []
                for rec, data in record_data.items():
                    info = data.get("artifact_info", {})
                    artifact_rows.append({
                        "Registro": rec,
                        "Nivel_correccion": info.get("level", "none"),
                        "Artefactos_n": info.get("n_artifacts", 0),
                        "Artefactos_pct": info.get("percent_artifacts", 0.0),
                    })
                # Dominios por registro
                dom_rows = []
                for rec, dfrec in records_results.items():
                    dom = domain_values(dfrec, method=domain_method)
                    if not dom.empty:
                        tmp_dom = dom.copy()
                        tmp_dom.insert(0, "Registro", rec)
                        tmp_dom.insert(1, "Fase", tmp_dom.index)
                        dom_rows.append(tmp_dom.reset_index(drop=True))
                if dom_rows:
                    pd.concat(dom_rows, ignore_index=True).to_excel(writer, sheet_name="dominios", index=False)

                # MSE formato largo
                mse_cols_export = [c for c in MSE_COLUMNS if c in long_df.columns]
                if mse_cols_export:
                    long_df[["Registro", "Fase"] + mse_cols_export].to_excel(writer, sheet_name="MSE_1_20", index=False)

                pd.DataFrame(artifact_rows).to_excel(writer, sheet_name="artefactos", index=False)
                report_preview_base = generate_auto_report(
                    record_data,
                    records_results,
                    st.session_state.global_windows_v50,
                    st.session_state.record_windows_v50,
                    active_phases,
                    use_independent,
                    long_df,
                )
                report_preview = append_complete_export_report(
                    report_preview_base, indices_df, attractor_export_df, prediction_export_summary,
                    prediction_export_details, prediction_export_topology, prediction_model_state
                )
                pd.DataFrame({"Informe": report_preview.splitlines()}).to_excel(writer, sheet_name="informe", index=False)

            report_md_base = generate_auto_report(
                record_data,
                records_results,
                st.session_state.global_windows_v50,
                st.session_state.record_windows_v50,
                active_phases,
                use_independent,
                long_df,
            )
            report_md = append_complete_export_report(
                report_md_base, indices_df, attractor_export_df, prediction_export_summary,
                prediction_export_details, prediction_export_topology, prediction_model_state
            )
            report_html = markdown_to_simple_html(report_md)
            p_report_md = tmpdir / "informe_hrv_grafos.md"
            p_report_html = tmpdir / "informe_hrv_grafos.html"
            p_report_md.write_text(report_md, encoding="utf-8")
            p_report_html.write_text(report_html, encoding="utf-8")

            # Tablas independientes CSV para análisis científico y trazabilidad.
            tables_dir = tmpdir / "tablas"
            tables_dir.mkdir(parents=True, exist_ok=True)
            _table_exports = {
                "metricas_completas.csv": long_df,
                "indices_fisiologicos.csv": indices_df,
                "atractores.csv": attractor_export_df,
                "prediccion_resumen_v153.csv": prediction_export_summary,
                "prediccion_indices_v153.csv": prediction_export_details,
                "prediccion_topologia_hvg_v153.csv": prediction_export_topology,
                "modelo_nivel2_estado.csv": prediction_model_state,
                "modelo_nivel2_metricas.csv": prediction_model_metrics,
                "modelo_nivel2_salidas_validadas.csv": prediction_model_targets,
            }
            for _fn, _dfx in _table_exports.items():
                try:
                    if isinstance(_dfx, pd.DataFrame) and not _dfx.empty:
                        _safe_df = _dfx.copy()
                        for _cc in _safe_df.columns:
                            _safe_df[_cc] = _safe_df[_cc].map(lambda v: json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict,list,tuple,set)) else v)
                        _safe_df.to_csv(tables_dir / _fn, index=False, encoding="utf-8-sig")
                except Exception:
                    pass

            manifest = {
                "version": "15.4.8",
                "export_schema": "complete_scientific_export_v2_xgboost",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "records": int(len(records)),
                "metric_rows": int(len(long_df)),
                "indices_rows": int(len(indices_df)) if isinstance(indices_df, pd.DataFrame) else 0,
                "attractor_rows": int(len(attractor_export_df)) if isinstance(attractor_export_df, pd.DataFrame) else 0,
                "prediction_summary_rows": int(len(prediction_export_summary)) if isinstance(prediction_export_summary, pd.DataFrame) else 0,
                "prediction_detail_rows": int(len(prediction_export_details)) if isinstance(prediction_export_details, pd.DataFrame) else 0,
                "includes": ["HRV", "MSE/MDE", "RQA", "HVG", "Poincare", "Atractores", "Indices fisiologicos", "Prediccion v15.3.x", "Nivel 1", "Nivel 2 XGBoost"]
            }
            (tmpdir / "manifest_exportacion_v1548.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            # Si existe Nivel 2 promovido, incluir una copia exacta del modelo activo para auditoría/reproducibilidad.
            _active_v153_export = None
            try:
                _active_v153_export = load_v153_level2()
                if _active_v153_export and SKLEARN_AVAILABLE:
                    joblib.dump(_active_v153_export, tmpdir / "modelo_nivel2_xgboost_activo.joblib")
            except Exception:
                _active_v153_export = None

            st.markdown("### Gráficos")
            export_formats = st.multiselect(
                "Formatos de gráficos a incluir en el ZIP (color fijo)",
                ["PNG", "SVG", "HTML interactivo"],
                default=["PNG", "HTML interactivo"],
                help=(
                    "PNG/SVG requieren Kaleido en el servidor. "
                    "HTML interactivo siempre funciona y además puede convertirse localmente a PNG con el script incluido."
                ),
                key="export_graph_formats_v88",
            )

            formats_internal = []
            if "HTML interactivo" in export_formats:
                formats_internal.append("html")
            if "PNG" in export_formats:
                formats_internal.append("png")
            if "SVG" in export_formats:
                formats_internal.append("svg")
            if not formats_internal:
                formats_internal = ["html"]

            graphs_dir = tmpdir / "graficos"
            figures_to_export = build_all_export_figures(
                record_data=record_data,
                records_results=records_results,
                long_df=long_df,
                records=records,
                selected_record=selected_record,
                global_windows=st.session_state.global_windows_v50,
                record_windows=st.session_state.record_windows_v50,
                active_phases=active_phases,
                use_independent=use_independent,
                domain_method=domain_method,
                include_hvg=include_hvg,
                dashboard_params=st.session_state.get("dash_params", None),
                dashboard_phases=st.session_state.get("dash_phases", None),
            )

            # v15.4.5: añadir explícitamente Atractores, Índices fisiológicos y Predicción v15.3.x.
            figures_to_export.extend(attractor_export_figures)
            figures_to_export.extend(prediction_export_figures)

            if indices_df is not None and not indices_df.empty:
                _idx_cols_export = ['IDX_Vagal','IDX_Amplitud','IDX_Complejidad','IDX_Rigidez','IDX_Adaptabilidad','IDX_Regulacion_Lenta']
                # Panel de índices por cada registro.
                for _rec in records:
                    try:
                        _idf_rec = indices_df[indices_df['Registro'].astype(str) == str(_rec)]
                        if not _idf_rec.empty:
                            figures_to_export.append((f"20_Indices_fisiologicos_{_rec}", physiological_indices_figure(indices_df, _rec)))
                    except Exception:
                        pass
                # Evolución cronológica completa, por fase, tanto conjunta como índice a índice.
                _idx_phases = sorted(indices_df['Fase'].dropna().astype(str).unique().tolist(), key=lambda x: PHASES.index(x) if x in PHASES else 999)
                for _ph in _idx_phases:
                    try:
                        _fig_idx_all, _ = chronological_indices_figure(indices_df, _ph)
                        figures_to_export.append((f"21_Indices_conjuntos_{_ph}", _fig_idx_all))
                    except Exception:
                        pass
                    for _idx_col in _idx_cols_export:
                        if _idx_col in indices_df.columns:
                            try:
                                _fig_idx_one, _ = individual_index_chronological_figure(indices_df, _idx_col, _ph, smooth_method='Media móvil', smooth_window=3)
                                figures_to_export.append((f"22_Indice_{_idx_col}_{_ph}", _fig_idx_one))
                            except Exception:
                                pass

            index_graphs = write_all_graph_exports(figures_to_export, graphs_dir, formats=formats_internal)

            # Script local para convertir HTML exportados a PNG si Kaleido no funciona en Streamlit Cloud.
            converter_script = tmpdir / "convert_html_to_png.py"
            try:
                converter_script.write_text(CONVERT_HTML_TO_PNG_SCRIPT, encoding="utf-8")
            except Exception:
                converter_script.write_text("# Script de conversión no disponible en esta versión.\n", encoding="utf-8")

            localhost_capture_script = tmpdir / "capture_streamlit_localhost_png.py"
            try:
                localhost_capture_script.write_text(CAPTURE_STREAMLIT_LOCALHOST_PNG_SCRIPT, encoding="utf-8")
            except Exception:
                localhost_capture_script.write_text("# Script de captura localhost no disponible en esta versión.\n", encoding="utf-8")

            arrancador_bat = tmpdir / "Arrancar_Convertidor.bat"
            arrancador_bat.write_text(globals().get("ARRANCAR_CONVERTIDOR_BAT", "@echo off\nstart \"\" \"http://localhost:8501/\"\npython \"%~dp0capture_streamlit_localhost_png.py\" \"http://localhost:8501/\" \"%~dp0captura_streamlit.png\"\npause\n"), encoding="utf-8")

            st.caption(
                f"Se han preparado {len(figures_to_export)} gráficos. "
                "Si PNG falla en Streamlit Cloud, descarga el ZIP completo, descomprímelo y ejecuta Arrancar_Convertidor.bat. Ese BAT arranca Streamlit localmente en http://localhost:8501/ y captura PNG."
            )
            if not index_graphs.empty:
                st.dataframe(index_graphs, use_container_width=True)

            with zipfile.ZipFile(zipf, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(xlsx, arcname=xlsx.name)
                z.write(csv, arcname=csv.name)
                z.write(p_report_md, arcname=p_report_md.name)
                z.write(p_report_html, arcname=p_report_html.name)
                _manifest_path = tmpdir / "manifest_exportacion_v1545.json"
                if _manifest_path.exists():
                    z.write(_manifest_path, arcname=_manifest_path.name)
                if tables_dir.exists():
                    for _tp in tables_dir.rglob("*"):
                        if _tp.is_file():
                            z.write(_tp, arcname=f"tablas/{_tp.relative_to(tables_dir)}")
                _model_export_path = tmpdir / "modelo_nivel2_xgboost_activo.joblib"
                if _model_export_path.exists():
                    z.write(_model_export_path, arcname=f"modelo/{_model_export_path.name}")

                # Añadir gráficos exportados, con subcarpetas html/png/svg
                if graphs_dir.exists():
                    for p in graphs_dir.rglob("*"):
                        if p.is_file():
                            z.write(p, arcname=f"graficos/{p.relative_to(graphs_dir)}")

                # Añadir script local de conversión HTML -> PNG
                if converter_script.exists():
                    z.write(converter_script, arcname=converter_script.name)

                # Añadir script local para capturar http://localhost:8501/ como PNG
                if localhost_capture_script.exists():
                    z.write(localhost_capture_script, arcname=localhost_capture_script.name)

                # Añadir arrancador universal Windows
                if arrancador_bat.exists():
                    z.write(arrancador_bat, arcname=arrancador_bat.name)

            st.download_button("Descargar ZIP completo con gráficos", zipf.read_bytes(), file_name="PhysioSentinel_exportacion_completa_v15_4_5.zip", mime="application/zip")

            # ZIP independiente sólo con PNG
            png_zipf = tmpdir / "graficos_png.zip"
            with zipfile.ZipFile(png_zipf, "w", zipfile.ZIP_DEFLATED) as zpng:
                png_root = graphs_dir / "png"
                if png_root.exists():
                    for p in png_root.rglob("*.png"):
                        zpng.write(p, arcname=p.name)
                # También incluye PNG generados desde HTML localmente si existen
                png_from_html = graphs_dir / "png_from_html"
                if png_from_html.exists():
                    for p in png_from_html.rglob("*.png"):
                        zpng.write(p, arcname=p.name)

            if png_zipf.exists() and png_zipf.stat().st_size > 100:
                st.download_button(
                    "Descargar sólo gráficos PNG",
                    png_zipf.read_bytes(),
                    file_name="graficos_hrv_png.zip",
                    mime="application/zip"
                )
            else:
                st.warning("No se han generado PNG directamente en Streamlit. Descarga el ZIP completo y usa convert_html_to_png.py para convertir los HTML a PNG en tu ordenador.")
            st.download_button("Descargar Excel", xlsx.read_bytes(), file_name="PhysioSentinel_resultados_completos_v15_4_5.xlsx")
