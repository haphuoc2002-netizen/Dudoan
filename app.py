"""
NBA AI Analyst - ứng dụng Streamlit
-----------------------------------
Luồng: balldontlie.io (lịch thi đấu + kết quả mới) -> model_nba.pkl (xác suất thắng, tổng điểm)
       -> Claude API (bài phân tích tiếng Việt cho từng trận) -> hiển thị.
Mới:   du_lieu_nba_lich_su.csv + 10 trận gần nhất (balldontlie) -> Power Rating từng đội
       -> so với kèo chấp (spreads/h2h) từ The Odds API -> gắn cờ NGHỊCH LÝ THỊ TRƯỜNG
       -> Claude phân tích sâu nguyên nhân các trận nghịch lý.

Cần có cạnh file này:  model_nba.pkl  (tạo bằng train_model.py)  và  du_lieu_nba_lich_su.csv
Cần khai báo trong st.secrets: APP_PASSWORD, BALLDONTLIE_API_KEY, ANTHROPIC_API_KEY, ODDS_API_KEY
(tùy chọn: CLAUDE_MODEL; ANOMALY_WEB_SEARCH = true để Claude tra tin chấn thương bằng web search)
"""
import copy
import hmac
import html
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from anthropic import Anthropic

# ============================================================================
# LÕI TÍNH ĐẶC TRƯNG (dùng chung cho huấn luyện và ứng dụng web)
# Mọi đặc trưng của một trận chỉ dùng thông tin TRƯỚC trận đó -> không rò rỉ.
# ============================================================================
from datetime import date as _date

HOME_ELO = 65.0      # lợi thế sân nhà quy đổi sang điểm Elo
K_ELO = 20.0         # hệ số cập nhật Elo
ELO_CARRY = 0.75     # giữ lại 75% chênh lệch Elo khi sang mùa mới
HIST_LEN = 15        # số trận gần nhất lưu cho mỗi đội
LG_PTS = 114.0       # điểm trung bình mặc định / đội / trận

FEATURES = [
    "h_elo", "a_elo", "elo_diff", "elo_prob",
    "h_win10", "a_win10", "h_win5", "a_win5", "win10_diff",
    "h_pf10", "a_pf10", "h_pa10", "a_pa10",
    "h_net10", "a_net10", "net10_diff",
    "h_tot10", "a_tot10", "exp_total", "pace_avg", "lg_avg_total",
    "h_season_wpct", "a_season_wpct",
    "h_rest", "a_rest", "rest_diff", "h_b2b", "a_b2b", "h_g7", "a_g7",
    "is_playoff",
]


def new_state():
    return {"teams": {}, "lg_totals": [], "last_date": None, "seen_ids": set()}


def _team(state, abbr):
    return state["teams"].setdefault(
        abbr, {"elo": 1500.0, "season": None, "sw": 0, "sg": 0, "hist": []}
    )


def _view(t, season):
    """Góc nhìn của đội ở mùa `season` mà không làm thay đổi trạng thái."""
    if t["season"] is not None and t["season"] != season:
        return {"elo": 1500.0 + ELO_CARRY * (t["elo"] - 1500.0), "sw": 0, "sg": 0, "hist": t["hist"]}
    return {"elo": t["elo"], "sw": t["sw"], "sg": t["sg"], "hist": t["hist"]}


def _mean(xs, default):
    return sum(xs) / len(xs) if xs else default


def _side(v, d):
    h = v["hist"]
    l10, l5 = h[-10:], h[-5:]
    pf10 = _mean([g[1] for g in l10], LG_PTS)
    pa10 = _mean([g[2] for g in l10], LG_PTS)
    if h:
        rest = min(max((d - _date.fromisoformat(h[-1][0])).days, 0), 10)
    else:
        rest = 7
    g7 = sum(1 for g in h if 0 < (d - _date.fromisoformat(g[0])).days <= 7)
    return {
        "elo": v["elo"],
        "win10": _mean([g[3] for g in l10], 0.5),
        "win5": _mean([g[3] for g in l5], 0.5),
        "pf10": pf10, "pa10": pa10, "net10": pf10 - pa10, "tot10": pf10 + pa10,
        "season_wpct": (v["sw"] / v["sg"]) if v["sg"] > 0 else 0.5,
        "rest": rest, "b2b": int(rest == 1), "g7": g7,
        "sg": v["sg"], "n_hist": len(h),
    }


def build_features(state, home, away, date_iso, season, is_playoff=None):
    """Trả về (dict đặc trưng, dict thông tin phụ) cho trận home vs away."""
    d = _date.fromisoformat(date_iso)
    h = _side(_view(_team(state, home), season), d)
    a = _side(_view(_team(state, away), season), d)
    if is_playoff is None:
        is_playoff = int(h["sg"] >= 82 and a["sg"] >= 82)
    lg = state["lg_totals"][-300:]
    elo_diff = h["elo"] - a["elo"]
    f = {
        "h_elo": h["elo"], "a_elo": a["elo"], "elo_diff": elo_diff,
        "elo_prob": 1.0 / (1.0 + 10 ** (-(elo_diff + HOME_ELO) / 400.0)),
        "h_win10": h["win10"], "a_win10": a["win10"],
        "h_win5": h["win5"], "a_win5": a["win5"],
        "win10_diff": h["win10"] - a["win10"],
        "h_pf10": h["pf10"], "a_pf10": a["pf10"], "h_pa10": h["pa10"], "a_pa10": a["pa10"],
        "h_net10": h["net10"], "a_net10": a["net10"], "net10_diff": h["net10"] - a["net10"],
        "h_tot10": h["tot10"], "a_tot10": a["tot10"],
        "exp_total": (h["pf10"] + a["pa10"]) / 2 + (a["pf10"] + h["pa10"]) / 2,
        "pace_avg": (h["tot10"] + a["tot10"]) / 2,
        "lg_avg_total": _mean(lg, 2 * LG_PTS),
        "h_season_wpct": h["season_wpct"], "a_season_wpct": a["season_wpct"],
        "h_rest": h["rest"], "a_rest": a["rest"], "rest_diff": h["rest"] - a["rest"],
        "h_b2b": h["b2b"], "a_b2b": a["b2b"], "h_g7": h["g7"], "a_g7": a["g7"],
        "is_playoff": int(is_playoff),
    }
    info = {"n_hist_min": min(h["n_hist"], a["n_hist"]), "home": h, "away": a}
    return f, info


def update_state(state, game_id, season, date_iso, home, away, home_pts, away_pts):
    """Cập nhật trạng thái sau khi một trận kết thúc. Trả về False nếu đã xử lý trận này."""
    if game_id is not None:
        if game_id in state["seen_ids"]:
            return False
        state["seen_ids"].add(game_id)
    th, ta = _team(state, home), _team(state, away)
    for t in (th, ta):
        if t["season"] != season:
            if t["season"] is not None:
                t["elo"] = 1500.0 + ELO_CARRY * (t["elo"] - 1500.0)
            t["sw"], t["sg"], t["season"] = 0, 0, season
    # --- Elo (có hệ số theo biên độ thắng) ---
    exp_h = 1.0 / (1.0 + 10 ** (-(th["elo"] - ta["elo"] + HOME_ELO) / 400.0))
    s_h = 1.0 if home_pts > away_pts else 0.0
    mov = abs(home_pts - away_pts)
    elo_gap_w = (th["elo"] - ta["elo"] + HOME_ELO) if s_h == 1.0 else -(th["elo"] - ta["elo"] + HOME_ELO)
    mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_gap_w)
    delta = K_ELO * mult * (s_h - exp_h)
    th["elo"] += delta
    ta["elo"] -= delta
    # --- lịch sử gần đây & thành tích mùa ---
    th["hist"] = (th["hist"] + [(date_iso, home_pts, away_pts, int(s_h), 1)])[-HIST_LEN:]
    ta["hist"] = (ta["hist"] + [(date_iso, away_pts, home_pts, int(1 - s_h), 0)])[-HIST_LEN:]
    th["sg"] += 1; ta["sg"] += 1
    th["sw"] += int(s_h); ta["sw"] += int(1 - s_h)
    state["lg_totals"] = (state["lg_totals"] + [home_pts + away_pts])[-400:]
    if state["last_date"] is None or date_iso > state["last_date"]:
        state["last_date"] = date_iso
    return True


# ============================================================================
# CẤU HÌNH
# ============================================================================
BDL_GAMES_URL = "https://api.balldontlie.io/v1/games"
MODEL_FILE = Path(__file__).with_name("model_nba.pkl")
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"   # có thể ghi đè bằng st.secrets["CLAUDE_MODEL"]
TZ_ET = ZoneInfo("America/New_York")          # lịch NBA/API tính theo giờ Mỹ (ET)
TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")

# --- The Odds API & phát hiện nghịch lý ---
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
ODDS_REGIONS = "us"
HISTORY_CSV = Path(__file__).with_name("du_lieu_nba_lich_su.csv")

# Power Rating
HALF_LIFE_DAYS = 70.0   # trận càng cũ trọng số càng giảm (bán rã 70 ngày)
RIDGE = 4.0             # co rating về 0 để bớt nhiễu khi mẫu nhỏ
MARGIN_CAP = 25.0       # cắt hiệu số các trận thắng đậm để không bị lệch
FORM_N = 10             # số trận gần nhất tính phong độ
W_FORM = 0.25           # trọng số phong độ trong Power Rating

# Ngưỡng nghịch lý (đơn vị: điểm hiệu số)
GAP_YELLOW = 4.0        # lệch kèo > 4 điểm  -> cờ vàng
GAP_RED = 7.0           # lệch kèo >= 7 điểm -> cờ đỏ
PARADOX_RATING_DIFF = 2.0    # đội chấp yếu hơn đội kia >= 2 điểm Power (sân trung lập)
PARADOX_WPCT_DIFF = 0.15     # hoặc tỉ lệ thắng mùa này thấp hơn >= 15 điểm %
PARADOX_MIN_OVERRATE = 2.0   # và thị trường đặt đội chấp cao hơn mô hình >= 2 điểm
MIN_GAMES_WPCT = 10          # cần >= 10 trận mùa này mới dùng tỉ lệ thắng
CONTRA_MIN_SPREAD = 1.5

_ODDS_TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}
ODDS_NAME_TO_ABBR = {v.lower(): k for k, v in _ODDS_TEAMS.items()}
ODDS_NAME_TO_ABBR["la clippers"] = "LAC"

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích bóng rổ NBA, viết bằng tiếng Việt tự nhiên, mạch lạc, dễ hiểu.

Bạn nhận được một khối JSON về MỘT trận đấu: xác suất thắng và tổng điểm dự đoán do mô hình Machine Learning \
(Random Forest / Gradient Boosting huấn luyện trên dữ liệu các trận NBA 2024–2026) tính ra, cùng các chỉ số đầu vào \
(Elo, phong độ 10 trận gần nhất, hiệu số điểm, ngày nghỉ, thành tích mùa giải).

QUY TẮC BẮT BUỘC:
- Chỉ dùng số liệu có trong JSON. TUYỆT ĐỐI không bịa thông tin về chấn thương, đội hình, cầu thủ cụ thể, huấn luyện viên, \
tin tức, lịch sử đối đầu hay tỉ lệ cược.
- Diễn giải đúng mức độ chắc chắn: xác suất 50–58% là rất mong manh (gần như cân bằng), 58–68% là nghiêng rõ, \
trên 68% là nghiêng mạnh. Không nói "chắc chắn thắng". Mô hình chỉ đúng khoảng 65% số trận, hãy thể hiện sự thận trọng đó.
- Giải thích VÌ SAO mô hình nghiêng về một phía bằng cách trích dẫn con số cụ thể từ JSON (Elo, phong độ, hiệu số, nghỉ ngơi...).
- Với tổng điểm: so sánh với khả năng ghi/thủng lưới của hai đội và nêu khoảng sai số 80% được cung cấp.
- Nếu so_tran_da_dau_mua_nay của một đội nhỏ hơn 10, phong độ và Elo còn dựa nhiều vào mùa trước: hãy nói rõ độ bất định cao hơn bình thường. ngay_nghi tối đa được ghi nhận là 10 (nghĩa là từ 10 ngày trở lên).
- Nếu is_playoff = true, nhắc đây là vòng playoff (nhịp độ và cường độ thường khác mùa giải thường).
- Không khuyến khích cá cược. Đây là phân tích thống kê mang tính tham khảo.

ĐỊNH DẠNG (Markdown, khoảng 220–320 từ, không thêm tiêu đề cấp 1/2):
**🎯 Nhận định nhanh:** 1–2 câu.
**📊 Vì sao mô hình nghiêng về phía này:** 3–4 gạch đầu dòng, mỗi dòng có số liệu.
**🔢 Tổng điểm:** 2–3 câu.
**⚠️ Điểm cần lưu ý:** 1–2 gạch đầu dòng về độ bất định hoặc yếu tố mô hình chưa nhìn thấy.
**✅ Kết luận:** 1 câu."""


# ============================================================================
# DỮ LIỆU TỪ balldontlie.io
# ============================================================================
def bdl_get(params, api_key, max_retries=6):
    """Gọi API balldontlie, tự chờ và thử lại khi gặp giới hạn tốc độ (HTTP 429)."""
    for _ in range(max_retries):
        r = requests.get(BDL_GAMES_URL, params=params, headers={"Authorization": api_key}, timeout=30)
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After", 0))
            except ValueError:
                wait = 0
            time.sleep(min(wait or 13, 30))
            continue
        if r.status_code in (401, 403):
            raise PermissionError("balldontlie từ chối khóa API (HTTP %d). Kiểm tra BALLDONTLIE_API_KEY "
                                  "và gói tài khoản." % r.status_code)
        r.raise_for_status()
        return r.json()
    raise RuntimeError("balldontlie vẫn báo vượt giới hạn tốc độ sau nhiều lần thử. Hãy chờ 1 phút rồi bấm lại.")


def fetch_games(start, end, api_key, on_page=None):
    """Lấy toàn bộ trận từ ngày `start` đến `end` (gồm cả hai đầu), tự phân trang."""
    params = {"start_date": start.isoformat(), "end_date": end.isoformat(), "per_page": 100}
    out = []
    while True:
        data = bdl_get(params, api_key)
        out.extend(data.get("data", []))
        if on_page:
            on_page(len(out))
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            return out
        params["cursor"] = cursor


def norm_game(g):
    ht, vt = g.get("home_team") or {}, g.get("visitor_team") or {}
    return {
        "id": g.get("id"), "date": str(g.get("date", ""))[:10], "season": g.get("season"),
        "status": str(g.get("status") or ""), "postseason": bool(g.get("postseason")),
        "home": ht.get("abbreviation"), "away": vt.get("abbreviation"),
        "home_name": ht.get("full_name") or ht.get("name") or ht.get("abbreviation"),
        "away_name": vt.get("full_name") or vt.get("name") or vt.get("abbreviation"),
        "hs": g.get("home_team_score"), "vs": g.get("visitor_team_score"),
        "tip": parse_tip(g),
    }


def parse_tip(g):
    """Giờ bắt đầu trận (UTC) nếu API cung cấp; trả về datetime có múi giờ hoặc None."""
    for key in ("datetime", "status"):
        s = g.get(key)
        if isinstance(s, str) and "T" in s:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                pass
    return None


def is_final(ng):
    return ng["status"].strip().lower().startswith("final") and ng["hs"] is not None and ng["vs"] is not None


def sync_state(state, api_key, end_date, on_page=None):
    """Đưa trạng thái đội (Elo, phong độ...) đến hết `end_date` bằng các trận đã kết thúc từ API.
    Trả về (số trận mới, danh sách trận mới).
    Cần thiết vì file mô hình chỉ biết dữ liệu đến ngày huấn luyện (state['last_date'])."""
    if not state["last_date"]:
        return 0, []
    start = date.fromisoformat(state["last_date"]) + timedelta(days=1)
    if start > end_date:
        return 0, []
    games = [norm_game(g) for g in fetch_games(start, end_date, api_key, on_page)]
    games = [g for g in games if is_final(g) and g["home"] in state["teams"] and g["away"] in state["teams"]]
    games.sort(key=lambda g: (g["date"], g["id"] or 0))
    applied = []
    for g in games:
        if update_state(state, g["id"], int(g["season"]), g["date"], g["home"], g["away"], int(g["hs"]), int(g["vs"])):
            applied.append(g)
    return len(applied), applied   # applied: các trận mới, dùng lại để tính phong độ cho Power Rating


# ============================================================================
# DỰ ĐOÁN
# ============================================================================
def predict_games(bundle, state, games):
    """Chạy mô hình cho danh sách trận (đã chuẩn hóa). Trả về danh sách 'card' dữ liệu."""
    feats, infos, kept = [], [], []
    for g in games:
        if g["home"] not in state["teams"] or g["away"] not in state["teams"]:
            continue  # đội lạ (trận biểu diễn/All-Star...) -> bỏ qua
        f, info = build_features(state, g["home"], g["away"], g["date"], int(g["season"]), is_playoff=int(g["postseason"]))
        feats.append(f); infos.append(info); kept.append(g)
    if not kept:
        return []
    X = pd.DataFrame(feats)[bundle["features"]]
    lo, hi = bundle["clip"]
    p_home = np.clip(bundle["clf"].predict_proba(X)[:, 1], lo, hi)
    totals = bundle["reg"].predict(X)
    q10, q90 = bundle["total_interval"]
    cards = []
    for g, f, info, p, t in zip(kept, feats, infos, p_home, totals):
        cards.append({
            "game": g, "p_home": float(p), "p_away": float(1 - p), "total": float(t),
            "total_lo": float(t + q10), "total_hi": float(t + q90), "feat": f, "info": info,
        })
    cards.sort(key=lambda c: (c["game"]["tip"] or datetime.max.replace(tzinfo=timezone.utc), c["game"]["id"] or 0))
    return cards


def confidence_label(p):
    m = max(p, 1 - p)
    if m < 0.58:
        return "Rất cân bằng"
    if m < 0.68:
        return "Nghiêng rõ"
    return "Nghiêng mạnh"


# ============================================================================
# CLAUDE: VIẾT BÀI PHÂN TÍCH
# ============================================================================
def analysis_payload(card):
    g, f, info = card["game"], card["feat"], card["info"]

    def side(s):
        return {
            "elo": round(s["elo"]), "ty_le_thang_10_tran": round(s["win10"], 2),
            "ty_le_thang_5_tran": round(s["win5"], 2),
            "diem_ghi_TB_10_tran": round(s["pf10"], 1), "diem_thung_luoi_TB_10_tran": round(s["pa10"], 1),
            "hieu_so_TB_10_tran": round(s["net10"], 1), "ty_le_thang_mua_nay": round(s["season_wpct"], 2),
            "so_tran_da_dau_mua_nay": s["sg"], "ngay_nghi": s["rest"],
            "dau_lien_tiep_2_ngay": bool(s["b2b"]), "so_tran_trong_7_ngay_qua": s["g7"],
        }

    return {
        "tran_dau": {"doi_nha": g["home_name"], "doi_khach": g["away_name"], "ngay_ET": g["date"],
                     "is_playoff": bool(f["is_playoff"])},
        "mo_hinh": {
            "xac_suat_doi_nha_thang_%": round(card["p_home"] * 100, 1),
            "xac_suat_doi_khach_thang_%": round(card["p_away"] * 100, 1),
            "tong_diem_du_doan": round(card["total"], 1),
            "khoang_80%_tong_diem": [round(card["total_lo"]), round(card["total_hi"])],
            "xac_suat_theo_Elo_thuan_%": round(f["elo_prob"] * 100, 1),
        },
        "chi_so_doi_nha": side(info["home"]),
        "chi_so_doi_khach": side(info["away"]),
        "tong_hop": {
            "tong_diem_ky_vong_theo_phong_do": round(f["exp_total"], 1),
            "tong_diem_TB_giai_gan_day": round(f["lg_avg_total"], 1),
        },
    }


def generate_analysis(client, model, card):
    resp = client.messages.create(
        model=model, max_tokens=1200, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "Dữ liệu trận đấu (JSON):\n"
                   + json.dumps(analysis_payload(card), ensure_ascii=False, indent=1)}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    if not text:
        raise RuntimeError("Claude trả về nội dung rỗng.")
    return text


def run_analyses(api_key, model, cards, cache, on_done=None, workers=3):
    """Gọi Claude song song cho từng trận. Kết quả: {game_id: (text|None, lỗi|None)}."""
    client = Anthropic(api_key=api_key, max_retries=3, timeout=90.0)
    results, todo = {}, []
    for c in cards:
        key = (c["game"]["id"], round(c["p_home"], 3), round(c["total"]), model)
        if key in cache:
            results[c["game"]["id"]] = (cache[key], None)
            if on_done:
                on_done()
        else:
            todo.append((c, key))

    def job(item):
        c, key = item
        return c, key, generate_analysis(client, model, c)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, it): it for it in todo}
        for fut in as_completed(futs):
            c, key = futs[fut]
            try:
                _, _, text = fut.result()
                cache[key] = text
                results[c["game"]["id"]] = (text, None)
            except Exception as e:  # lỗi một trận không làm hỏng các trận khác
                results[c["game"]["id"]] = (None, f"{type(e).__name__}: {e}")
            if on_done:
                on_done()
    return results


# ============================================================================
# POWER RATING (CSV lịch sử + phong độ 10 trận gần nhất từ balldontlie.io)
# ----------------------------------------------------------------------------
# 1) Rating nền: hồi quy ridge có trọng số theo thời gian trên hiệu số điểm của MỌI trận
#    (margin = sân_nhà + rating_nhà - rating_khách) -> đã hiệu chỉnh theo độ mạnh đối thủ.
# 2) Phong độ: "rating biểu diễn" trung bình của 10 trận gần nhất (hiệu số đã trừ sân nhà,
#    cộng rating đối thủ).
# 3) Power Rating = trộn rating nền với phong độ. Đơn vị: điểm hiệu số trên sân trung lập.
# ============================================================================
def load_history_csv():
    if not HISTORY_CSV.exists():
        raise FileNotFoundError(f"Không thấy {HISTORY_CSV.name}. Hãy đặt file này cùng thư mục với app.py.")
    df = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    return pd.DataFrame({
        "id": df["Ma_Tran"], "date": pd.to_datetime(df["Ngay_Dau"]), "season": df["Mua_Giai"].astype(int),
        "home": df["Viet_Tat_Nha"].astype(str), "away": df["Viet_Tat_Khach"].astype(str),
        "hs": df["Diem_Nha"].astype(int), "vs": df["Diem_Khach"].astype(int),
    })


def combine_games(hist, recent):
    """Gộp CSV với các trận mới đồng bộ từ balldontlie (trùng id thì ưu tiên bản từ API)."""
    parts = [hist]
    if recent:
        parts.append(pd.DataFrame([{
            "id": g["id"], "date": pd.Timestamp(g["date"]), "season": int(g["season"]),
            "home": g["home"], "away": g["away"], "hs": int(g["hs"]), "vs": int(g["vs"]),
        } for g in recent]))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="id", keep="last")
    return df.sort_values(["date", "id"]).reset_index(drop=True)


def compute_power_ratings(games, as_of):
    """Tính Power Rating của từng đội dựa trên các trận diễn ra TRƯỚC ngày `as_of`."""
    as_of = pd.Timestamp(as_of)
    g = games[games["date"] < as_of]
    if len(g) < 200:
        raise ValueError(f"Chỉ có {len(g)} trận lịch sử trước {as_of.date()} - quá ít để tính Power Rating.")
    teams = sorted(set(g["home"]) | set(g["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    X = np.zeros((len(g), n + 1))
    X[:, 0] = 1.0                                  # cột 0: lợi thế sân nhà
    rows = np.arange(len(g))
    X[rows, 1 + g["home"].map(idx).to_numpy()] = 1.0
    X[rows, 1 + g["away"].map(idx).to_numpy()] = -1.0
    y = np.clip((g["hs"] - g["vs"]).to_numpy(float), -MARGIN_CAP, MARGIN_CAP)
    age = (as_of - g["date"]).dt.days.to_numpy(float)
    w = 0.5 ** (age / HALF_LIFE_DAYS)
    w = w / w.max()

    reg = np.diag([0.0] + [RIDGE] * n)             # không phạt hệ số sân nhà
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + reg, X.T @ (y * w))
    hca = float(np.clip(beta[0], 1.0, 4.5))
    base = beta[1:]

    perf = {t: [] for t in teams}                  # rating biểu diễn từng trận
    wins = {t: [] for t in teams}
    for r in g.itertuples(index=False):
        m = float(np.clip(r.hs - r.vs, -MARGIN_CAP, MARGIN_CAP))
        perf[r.home].append((m - hca) + base[idx[r.away]])
        perf[r.away].append((-m + hca) + base[idx[r.home]])
        wins[r.home].append(int(r.hs > r.vs))
        wins[r.away].append(int(r.vs > r.hs))

    season = int(g["season"].max())
    cur = g[g["season"] == season]
    sw = {t: 0 for t in teams}
    sg = {t: 0 for t in teams}
    for r in cur.itertuples(index=False):
        sg[r.home] += 1; sg[r.away] += 1
        sw[r.home] += int(r.hs > r.vs); sw[r.away] += int(r.vs > r.hs)

    out = {}
    for t in teams:
        last = perf[t][-FORM_N:]
        form = float(np.mean(last)) if last else float(base[idx[t]])
        wf = W_FORM * len(last) / FORM_N
        power = (1 - wf) * float(base[idx[t]]) + wf * form
        l10 = wins[t][-FORM_N:]
        out[t] = {"power": power, "base": float(base[idx[t]]), "form": form, "n_form": len(last),
                  "rec10": f"{sum(l10)}-{len(l10) - sum(l10)}",
                  "sg": sg[t], "sw": sw[t], "wpct": (sw[t] / sg[t]) if sg[t] else 0.5}
    ranked = sorted(out, key=lambda t: -out[t]["power"])
    for i, t in enumerate(ranked, 1):
        out[t]["rank"] = i
        out[t]["tier"] = "Mạnh" if i <= 10 else ("Yếu" if i > len(ranked) - 10 else "Trung bình")
    return {"teams": out, "hca": hca, "n_games": len(g), "season": season, "as_of": as_of.date(), "n_teams": n}


def ratings_table(ratings):
    rows = []
    for t, r in sorted(ratings["teams"].items(), key=lambda kv: kv[1]["rank"]):
        rows.append({"Hạng": r["rank"], "Đội": t, "Nhóm": r["tier"], "Power": round(r["power"], 1),
                     "Rating nền": round(r["base"], 1), "Phong độ 10 trận": round(r["form"], 1),
                     "Thành tích 10 trận": r["rec10"],
                     "Thắng mùa này": f"{r['wpct']:.0%} ({r['sw']}/{r['sg']})"})
    return pd.DataFrame(rows)


# ============================================================================
# THE ODDS API: kèo chấp (spreads) + kèo thắng thua (h2h)
# ============================================================================
def odds_abbr(name):
    return ODDS_NAME_TO_ABBR.get(str(name).strip().lower())


def fetch_odds(api_key):
    """Trả về (danh sách sự kiện, số lượt gọi API còn lại)."""
    r = requests.get(ODDS_URL, params={
        "apiKey": api_key, "regions": ODDS_REGIONS, "markets": "spreads,h2h",
        "oddsFormat": "decimal", "dateFormat": "iso",
    }, timeout=30)
    if r.status_code == 401:
        raise PermissionError("The Odds API từ chối khóa (HTTP 401). Kiểm tra ODDS_API_KEY.")
    if r.status_code == 429:
        raise RuntimeError("The Odds API báo hết hạn mức hoặc gọi quá nhanh (HTTP 429).")
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def summarize_event(ev):
    """Gom kèo của nhiều nhà cái thành một kèo đồng thuận (trung vị). None nếu thiếu dữ liệu."""
    home_name, away_name = ev.get("home_team"), ev.get("away_team")
    home, away = odds_abbr(home_name), odds_abbr(away_name)
    if not home or not away:
        return None
    spreads, ml_h, ml_a, p_h = [], [], [], []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            outs = mk.get("outcomes", [])
            if mk.get("key") == "spreads":
                for o in outs:
                    if o.get("name") == home_name and o.get("point") is not None:
                        spreads.append(float(o["point"]))
            elif mk.get("key") == "h2h":
                ph = next((o.get("price") for o in outs if o.get("name") == home_name), None)
                pa = next((o.get("price") for o in outs if o.get("name") == away_name), None)
                if ph and pa and ph > 1 and pa > 1:
                    ml_h.append(float(ph)); ml_a.append(float(pa))
                    qh, qa = 1 / ph, 1 / pa
                    p_h.append(qh / (qh + qa))              # bỏ biên lợi nhuận nhà cái
    if not spreads:
        return None
    tip = None
    try:
        tip = datetime.fromisoformat(str(ev["commence_time"]).replace("Z", "+00:00"))
    except Exception:
        pass
    return {
        "home": home, "away": away, "tip": tip,
        "spread_home": float(np.median(spreads)), "spread_min": min(spreads), "spread_max": max(spreads),
        "n_books": len(spreads),
        "ml_home": float(np.median(ml_h)) if ml_h else None, "ml_away": float(np.median(ml_a)) if ml_a else None,
        "p_home_market": float(np.mean(p_h)) if p_h else None,
    }


def match_event(game, summaries):
    """Tìm kèo của trận `game` (theo cặp đội + ngày ET gần nhất, lệch tối đa 1 ngày)."""
    cands = [s for s in summaries if s["home"] == game["home"] and s["away"] == game["away"]]
    if not cands:
        return None
    gd = date.fromisoformat(game["date"])

    def dist(s):
        return abs((s["tip"].astimezone(TZ_ET).date() - gd).days) if s["tip"] else 0

    best = min(cands, key=dist)
    return best if dist(best) <= 1 else None


# ============================================================================
# PHÁT HIỆN NGHỊCH LÝ THỊ TRƯỜNG
# ============================================================================
def evaluate_market(card, ev, ratings):
    """So mô hình Power Rating với kèo thị trường. Trả về dict (kèm cờ cảnh báo) hoặc None."""
    g = card["game"]
    rt_h, rt_a = ratings["teams"].get(g["home"]), ratings["teams"].get(g["away"])
    if ev is None or rt_h is None or rt_a is None:
        return None
    hca = ratings["hca"]
    neutral_diff = rt_h["power"] - rt_a["power"]            # >0: đội nhà mạnh hơn (sân trung lập)
    model_margin = neutral_diff + hca                       # hiệu số kỳ vọng (nhà - khách)
    spread_home = ev["spread_home"]
    market_margin = -spread_home                            # kèo -5.5 cho đội nhà = thị trường kỳ vọng nhà hơn 5.5
    gap = model_margin - market_margin                      # >0: mô hình ủng hộ đội nhà hơn thị trường

    flags = []
    fav = "home" if spread_home < 0 else ("away" if spread_home > 0 else None)
    if fav:
        fav_ab, dog_ab = (g["home"], g["away"]) if fav == "home" else (g["away"], g["home"])
        rt_f, rt_d = (rt_h, rt_a) if fav == "home" else (rt_a, rt_h)
        sign = 1 if fav == "home" else -1
        fav_edge = sign * neutral_diff                      # >0: đội chấp mạnh hơn thật
        overrate = abs(spread_home) - sign * model_margin   # thị trường đặt đội chấp cao hơn mô hình bao nhiêu điểm
        wpct_gap = rt_f["wpct"] - rt_d["wpct"]
        enough = rt_f["sg"] >= MIN_GAMES_WPCT and rt_d["sg"] >= MIN_GAMES_WPCT
        weaker_power = fav_edge <= -PARADOX_RATING_DIFF
        weaker_wpct = enough and wpct_gap <= -PARADOX_WPCT_DIFF and fav_edge < 0
        if (weaker_power or weaker_wpct) and overrate >= PARADOX_MIN_OVERRATE:
            flags.append({"code": "PARADOX", "level": "red", "text": (
                f"{fav_ab} yếu hơn {dog_ab} (hạng sức mạnh #{rt_f['rank']} vs #{rt_d['rank']}, "
                f"Power {rt_f['power']:+.1f} vs {rt_d['power']:+.1f}, tỉ lệ thắng mùa này {rt_f['wpct']:.0%} vs {rt_d['wpct']:.0%}) "
                f"nhưng lại là đội chấp {abs(spread_home):.1f} điểm.")})
    if abs(gap) > GAP_YELLOW:
        who = g["home"] if gap > 0 else g["away"]
        other = g["away"] if gap > 0 else g["home"]
        flags.append({"code": "GAP", "level": "red" if abs(gap) >= GAP_RED else "yellow", "text": (
            f"Kèo chấp lệch {abs(gap):.1f} điểm so với mô hình: mô hình ước tính {g['home']} "
            f"{'thắng' if model_margin >= 0 else 'thua'} {abs(model_margin):.1f} điểm, thị trường cho {g['home']} "
            f"{'thắng' if market_margin >= 0 else 'thua'} {abs(market_margin):.1f} điểm "
            f"(mô hình ủng hộ {who} hơn, thị trường đánh giá {other} cao hơn mô hình).")})
    if ev["ml_home"] and ev["ml_away"] and abs(spread_home) >= CONTRA_MIN_SPREAD:
        ml_fav = "home" if ev["ml_home"] < ev["ml_away"] else "away"
        if ml_fav != fav:
            flags.append({"code": "CONTRA", "level": "yellow", "text": (
                f"Kèo chấp và kèo thắng/thua mâu thuẫn nhau (chấp {spread_home:+.1f} cho {g['home']} "
                f"nhưng tỉ lệ cược {g['home']} {ev['ml_home']:.2f} / {g['away']} {ev['ml_away']:.2f}).")})

    level = "red" if any(f["level"] == "red" for f in flags) else ("yellow" if flags else None)
    return {"ev": ev, "rt_home": rt_h, "rt_away": rt_a, "hca": hca, "neutral_diff": neutral_diff,
            "model_margin": model_margin, "market_margin": market_margin, "spread_home": spread_home,
            "gap": gap, "flags": flags, "level": level}


def anomaly_sort_key(item):
    card, m = item
    return (0 if m["level"] == "red" else 1, -abs(m["gap"]))


# ============================================================================
# CLAUDE: PHÂN TÍCH SÂU TRẬN NGHỊCH LÝ
# ============================================================================
ANOMALY_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích thị trường kèo NBA, viết bằng tiếng Việt tự nhiên, rõ ràng.

Bạn nhận một khối JSON về MỘT trận mà hệ thống đã gắn cờ "NGHỊCH LÝ THỊ TRƯỜNG": kèo chấp thực tế (đồng thuận nhiều nhà cái) \
lệch đáng kể so với Power Rating của mô hình thống kê, hoặc đội yếu hơn lại là đội chấp điểm.

Nhiệm vụ: đưa ra các GIẢ THUYẾT về nguyên nhân khiến kèo lệch khỏi mô hình, xếp theo mức độ khả dĩ.

QUY TẮC BẮT BUỘC:
- Nhớ rằng nhà cái thường biết nhiều hơn mô hình (chấn thương, tin đội hình phút chót, dòng tiền lớn). Một kèo lệch KHÔNG đồng nghĩa \
với việc thị trường sai; cũng có thể mô hình thiếu thông tin hoặc rating còn dựa nhiều vào dữ liệu cũ.
- Số liệu trong JSON là sự thật duy nhất bạn có về trận này. Mọi nguyên nhân bạn nêu phải gắn với một con số trong JSON hoặc được \
nói rõ là GIẢ THUYẾT chưa kiểm chứng. KHÔNG khẳng định cầu thủ cụ thể nào vắng mặt, bị thương hay nghỉ ngơi nếu không có nguồn.
- Các nhóm nguyên nhân cần xét (chỉ nêu nhóm có cơ sở): (1) sân nhà (hệ số sân nhà, hướng lệch có khớp lợi thế sân nhà không); \
(2) lịch thi đấu dày / back-to-back / số ngày nghỉ / số trận trong 7 ngày; (3) nghi vấn ngôi sao hoặc cầu thủ chủ chốt vắng mặt \
(chỉ là nghi vấn, nêu dấu hiệu gợi ý: kèo dịch chuyển về phía đội yếu hơn mô hình, đội mạnh nhưng được định giá thấp...); \
(4) mô hình bị lệch (mẫu nhỏ, rating nền dựa mùa cũ, phong độ 10 trận đang chênh nhiều so với rating nền, đội đổi đội hình); \
(5) mâu thuẫn giữa kèo chấp và kèo thắng/thua nếu có.
- Nếu `phong_do_10_tran` và `rating_nen` của một đội chênh nhau nhiều, hãy nói rõ mô hình đang tin phong độ hay nền.
- Không khuyến khích cá cược, không đưa lời khuyên đặt cược. Đây là phân tích thống kê mang tính tham khảo.

ĐỊNH DẠNG (Markdown, khoảng 250–350 từ, không dùng tiêu đề cấp 1/2):
**🚩 Nghịch lý là gì:** 2–3 câu, trích số liệu (kèo chấp, dự đoán mô hình, chênh lệch, hạng sức mạnh).
**🔍 Giả thuyết nguyên nhân (xếp theo độ khả dĩ):** 3–4 gạch đầu dòng. Mỗi dòng: tên nguyên nhân – dữ liệu ủng hộ – mức khả dĩ (cao/vừa/thấp).
**🧪 Cần kiểm tra trước khi tin vào kèo:** 2–3 gạch đầu dòng (ví dụ báo cáo chấn thương chính thức, đội hình xuất phát, biến động kèo).
**⚖️ Kết luận:** 1–2 câu: nên tin mô hình hay tin thị trường nhiều hơn trong trường hợp này và vì sao, nêu rõ mức bất định."""

ANOMALY_SEARCH_CLAUSE = """
- Bạn có công cụ tìm kiếm web. Hãy tìm báo cáo chấn thương / đội hình mới nhất của hai đội trong ngày thi đấu (ngay_ET). Chỉ nêu cầu thủ \
vắng mặt khi có nguồn cụ thể và nói rõ nguồn; nếu không tìm thấy, nói rõ là chưa xác minh được."""

ANOMALY_NOSEARCH_CLAUSE = """
- Bạn KHÔNG có dữ liệu chấn thương hay đội hình. Phần nghi vấn ngôi sao vắng mặt chỉ là suy đoán từ hình dạng của kèo."""


def anomaly_payload(card, m):
    g, info, ev = card["game"], card["info"], m["ev"]

    def team(rt, s):
        return {
            "power_rating": round(rt["power"], 1), "hang_suc_manh": f"{rt['rank']}/30", "nhom": rt["tier"],
            "rating_nen": round(rt["base"], 1), "phong_do_10_tran": round(rt["form"], 1),
            "thanh_tich_10_tran": rt["rec10"], "ty_le_thang_mua_nay": round(rt["wpct"], 2),
            "so_tran_mua_nay": rt["sg"], "ngay_nghi": s["rest"], "dau_lien_tiep_2_ngay": bool(s["b2b"]),
            "so_tran_trong_7_ngay_qua": s["g7"],
        }

    return {
        "tran_dau": {"doi_nha": g["home_name"], "doi_khach": g["away_name"], "ngay_ET": g["date"],
                     "is_playoff": bool(g["postseason"])},
        "ke_thi_truong": {
            "ke_chap_doi_nha(am=doi_nha_chap)": round(m["spread_home"], 1),
            "khoang_ke_chap_giua_cac_nha_cai": [ev["spread_min"], ev["spread_max"]], "so_nha_cai": ev["n_books"],
            "ty_le_cuoc_thang_doi_nha": ev["ml_home"], "ty_le_cuoc_thang_doi_khach": ev["ml_away"],
            "xac_suat_doi_nha_thang_theo_ke_%": round(ev["p_home_market"] * 100, 1) if ev["p_home_market"] else None,
        },
        "mo_hinh_power_rating": {
            "he_so_san_nha_diem": round(m["hca"], 1),
            "hieu_so_ky_vong_doi_nha_thang_diem": round(m["model_margin"], 1),
            "ke_chap_doi_nha_theo_mo_hinh": round(-m["model_margin"], 1),
            "chenh_lech_mo_hinh_tru_thi_truong_diem": round(m["gap"], 1),
            "xac_suat_doi_nha_thang_theo_mo_hinh_ML_%": round(card["p_home"] * 100, 1),
        },
        "chi_so_doi_nha": team(m["rt_home"], info["home"]),
        "chi_so_doi_khach": team(m["rt_away"], info["away"]),
        "co_canh_bao": [f["text"] for f in m["flags"]],
    }


def generate_anomaly_analysis(client, model, card, m, use_search):
    user_msg = "Dữ liệu trận đấu nghịch lý (JSON):\n" + json.dumps(anomaly_payload(card, m), ensure_ascii=False, indent=1)
    kw = dict(model=model, max_tokens=1800, messages=[{"role": "user", "content": user_msg}])
    resp = None
    if use_search:
        try:
            resp = client.messages.create(
                system=ANOMALY_SYSTEM_PROMPT + ANOMALY_SEARCH_CLAUSE,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}], **kw)
        except Exception:
            resp = None  # tài khoản chưa bật web search -> quay về chế độ không tìm kiếm
    if resp is None:
        resp = client.messages.create(system=ANOMALY_SYSTEM_PROMPT + ANOMALY_NOSEARCH_CLAUSE, **kw)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    if not text:
        raise RuntimeError("Claude trả về nội dung rỗng.")
    return text


def run_anomaly_analyses(api_key, model, items, cache, use_search=False, on_done=None, workers=3):
    """items: [(card, market)]. Kết quả: {game_id: (text|None, lỗi|None)}."""
    client = Anthropic(api_key=api_key, max_retries=3, timeout=120.0)
    results, todo = {}, []
    for c, m in items:
        gid = c["game"]["id"]
        key = ("anomaly", gid, round(m["spread_home"], 1), round(m["model_margin"], 1), model, use_search)
        if key in cache:
            results[gid] = (cache[key], None)
            if on_done:
                on_done()
        else:
            todo.append((c, m, key))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(generate_anomaly_analysis, client, model, c, m, use_search): (c, key) for c, m, key in todo}
        for fut in as_completed(futs):
            c, key = futs[fut]
            try:
                text = fut.result()
                cache[key] = text
                results[c["game"]["id"]] = (text, None)
            except Exception as e:
                results[c["game"]["id"]] = (None, f"{type(e).__name__}: {e}")
            if on_done:
                on_done()
    return results


# ============================================================================
# GIAO DIỆN STREAMLIT
# ============================================================================
CSS = """
<style>
.pb-top{display:flex;justify-content:space-between;font-weight:700;font-size:1.05rem;margin:.2rem 0 .3rem}
.pb{display:flex;height:14px;border-radius:7px;overflow:hidden;background:rgba(128,128,128,.25)}
.pb .a{background:#3b82f6}.pb .h{background:#f97316}
.gh{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap}
.gh .t{font-size:1.25rem;font-weight:700}.gh .s{opacity:.7;font-size:.9rem}
.badge{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.78rem;font-weight:600;
       background:rgba(128,128,128,.2);margin-left:.4rem}
.flag{display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;
      padding:.55rem .9rem;border-radius:8px;font-weight:800;font-size:1.05rem;margin-bottom:.6rem}
.flag .s{font-weight:500;font-size:.85rem;opacity:.85}
.flag-red{background:rgba(220,38,38,.16);border-left:8px solid #dc2626;color:#ef4444}
.flag-yellow{background:rgba(234,179,8,.18);border-left:8px solid #eab308;color:#ca8a04}
.mini-flag{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.78rem;font-weight:700;margin-left:.4rem}
.mini-red{background:rgba(220,38,38,.2);color:#ef4444}.mini-yellow{background:rgba(234,179,8,.25);color:#ca8a04}
</style>
"""


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


def require_password():
    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.error("Chưa cấu hình `APP_PASSWORD` trong st.secrets (xem hướng dẫn triển khai).")
        st.stop()
    if st.session_state.get("auth_ok"):
        return
    st.title("🏀 NBA AI Analyst")
    with st.form("login"):
        pw = st.text_input("🔒 Mật khẩu cá nhân", type="password")
        ok = st.form_submit_button("Đăng nhập")
    if ok:
        if hmac.compare_digest(pw.encode("utf-8"), str(expected).encode("utf-8")):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            time.sleep(1)  # làm chậm việc dò mật khẩu
            st.error("Sai mật khẩu.")
    st.stop()


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_bundle():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Không thấy {MODEL_FILE.name}. Hãy đặt file này cùng thư mục với app.py.")
    return joblib.load(MODEL_FILE)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_odds_cached(api_key):
    """Cache 5 phút để bấm cập nhật nhiều lần không tốn thêm hạn mức The Odds API."""
    return fetch_odds(api_key)


def fmt_tip(dt):
    return dt.astimezone(TZ_VN).strftime("%H:%M %d/%m") + " (giờ VN)" if dt else "chưa rõ giờ"


def render_card(card, analysis, error, market=None):
    g = card["game"]
    ph, pa = card["p_home"] * 100, card["p_away"] * 100
    with st.container(border=True):
        status = ""
        if is_final(g):
            status = f'<span class="badge">Đã kết thúc: {html.escape(g["away"])} {g["vs"]} – {g["hs"]} {html.escape(g["home"])}</span>'
        elif g["postseason"]:
            status = '<span class="badge">Playoffs</span>'
        if market and market["level"]:
            status += (f'<span class="mini-flag mini-{market["level"]}">'
                       f'{"🔴" if market["level"] == "red" else "🟡"} NGHỊCH LÝ - xem mục cảnh báo phía trên</span>')
        st.markdown(
            f'<div class="gh"><span class="t">{html.escape(g["away_name"])} @ {html.escape(g["home_name"])}{status}</span>'
            f'<span class="s">🕒 {html.escape(fmt_tip(g["tip"]))}</span></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pb-top"><span style="color:#3b82f6">{html.escape(g["away"])} {pa:.0f}%</span>'
            f'<span style="color:#f97316">{ph:.0f}% {html.escape(g["home"])}</span></div>'
            f'<div class="pb"><div class="a" style="width:{pa:.1f}%"></div><div class="h" style="width:{ph:.1f}%"></div></div>',
            unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        fav = g["home_name"] if ph >= 50 else g["away_name"]
        c1.metric("Mô hình nghiêng về", fav)
        c1.caption(f"{max(ph, pa):.0f}% – {confidence_label(card['p_home'])}")
        c2.metric("Tổng điểm dự đoán", f"{card['total']:.0f}")
        c2.caption(f"Khoảng 80%: {card['total_lo']:.0f} – {card['total_hi']:.0f}")
        c3.metric("Xác suất đội nhà thắng theo Elo thuần", f"{card['feat']['elo_prob'] * 100:.0f}%",
                  help="Để đối chiếu với mô hình đầy đủ (đã tính thêm phong độ, nghỉ ngơi...)")

        if market:
            st.caption(
                f"📈 Kèo chấp: **{g['home']} {market['spread_home']:+.1f}** ({market['ev']['n_books']} nhà cái) · "
                f"Power Rating dự đoán: **{g['home']} {-market['model_margin']:+.1f}** · "
                f"Chênh lệch: **{market['gap']:+.1f}** điểm · "
                f"Hạng sức mạnh: {g['away']} #{market['rt_away']['rank']} vs {g['home']} #{market['rt_home']['rank']}")

        if analysis:
            st.markdown(analysis)
        else:
            st.warning(f"Không tạo được bài phân tích cho trận này. {error or ''}")

        with st.expander("Chỉ số đầu vào của mô hình"):
            h, a = card["info"]["home"], card["info"]["away"]
            rows = [
                ("Elo", f"{a['elo']:.0f}", f"{h['elo']:.0f}"),
                ("Thắng 10 trận gần nhất", f"{a['win10']:.0%}", f"{h['win10']:.0%}"),
                ("Điểm ghi TB (10 trận)", f"{a['pf10']:.1f}", f"{h['pf10']:.1f}"),
                ("Điểm thủng lưới TB (10 trận)", f"{a['pa10']:.1f}", f"{h['pa10']:.1f}"),
                ("Hiệu số TB (10 trận)", f"{a['net10']:+.1f}", f"{h['net10']:+.1f}"),
                ("Tỉ lệ thắng mùa này", f"{a['season_wpct']:.0%} ({a['sg']} trận)", f"{h['season_wpct']:.0%} ({h['sg']} trận)"),
                ("Ngày nghỉ", f"{a['rest']}", f"{h['rest']}"),
                ("Đấu 2 ngày liên tiếp", "Có" if a["b2b"] else "Không", "Có" if h["b2b"] else "Không"),
            ]
            st.table(pd.DataFrame(rows, columns=["Chỉ số", f"{g['away']} (khách)", f"{g['home']} (nhà)"]).set_index("Chỉ số"))


def render_anomaly_section(items, analyses, use_search):
    """items: [(card, market)] đã sắp xếp (đỏ trước, lệch nhiều trước)."""
    st.header("🚨 Cảnh báo Kèo Nghịch Lý")
    if not items:
        st.success("Không phát hiện trận nào có kèo bất thường so với Power Rating hôm nay "
                   f"(ngưỡng: lệch > {GAP_YELLOW:.0f} điểm, hoặc đội yếu hơn hẳn lại là đội chấp).")
        return
    n_red = sum(1 for _, m in items if m["level"] == "red")
    st.caption(f"{len(items)} trận bất thường ({n_red} cờ đỏ, {len(items) - n_red} cờ vàng). "
               "Đây là chênh lệch giữa mô hình và thị trường, không phải khuyến nghị đặt cược - "
               "nhà cái thường nắm thêm tin chấn thương/đội hình mà mô hình không có.")
    for card, m in items:
        g = card["game"]
        red = m["level"] == "red"
        with st.container(border=True):
            st.markdown(
                f'<div class="flag flag-{m["level"]}"><span>{"🔴" if red else "🟡"} NGHỊCH LÝ THỊ TRƯỜNG · '
                f'{html.escape(g["away_name"])} @ {html.escape(g["home_name"])}</span>'
                f'<span class="s">🕒 {html.escape(fmt_tip(g["tip"]))}</span></div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Kèo chấp thị trường ({g['home']})", f"{m['spread_home']:+.1f}")
            c1.caption(f"{m['ev']['n_books']} nhà cái: {m['ev']['spread_min']:+.1f} → {m['ev']['spread_max']:+.1f}")
            c2.metric(f"Power Rating dự đoán ({g['home']})", f"{-m['model_margin']:+.1f}")
            c2.caption(f"Sân nhà ≈ {m['hca']:.1f} điểm")
            c3.metric("Chênh lệch kèo", f"{abs(m['gap']):.1f} điểm",
                      help="Độ lệch giữa hiệu số mô hình dự đoán và hiệu số thị trường ngụ ý.")
            c4.metric("Hạng sức mạnh", f"{g['away']} #{m['rt_away']['rank']} · {g['home']} #{m['rt_home']['rank']}")
            c4.caption(f"{m['rt_away']['tier']} vs {m['rt_home']['tier']}")
            for f in m["flags"]:
                st.markdown(f"{'🔴' if f['level'] == 'red' else '🟡'} {f['text']}")
            text, err = analyses.get(g["id"], (None, None))
            st.markdown("---")
            st.markdown("**🧠 Claude phân tích nguyên nhân**" + (" (có tra cứu web)" if use_search else ""))
            if text:
                st.markdown(text)
            else:
                st.warning(f"Không tạo được phân tích nguyên nhân cho trận này. {err or ''}")
            with st.expander("Chi tiết Power Rating & lịch thi đấu"):
                h, a = card["info"]["home"], card["info"]["away"]
                rh, ra = m["rt_home"], m["rt_away"]
                rows = [
                    ("Hạng / nhóm sức mạnh", f"#{ra['rank']} {ra['tier']}", f"#{rh['rank']} {rh['tier']}"),
                    ("Power Rating", f"{ra['power']:+.1f}", f"{rh['power']:+.1f}"),
                    ("Rating nền (cả mùa, hiệu chỉnh đối thủ)", f"{ra['base']:+.1f}", f"{rh['base']:+.1f}"),
                    (f"Phong độ {FORM_N} trận gần nhất", f"{ra['form']:+.1f}", f"{rh['form']:+.1f}"),
                    (f"Thành tích {FORM_N} trận", ra["rec10"], rh["rec10"]),
                    ("Tỉ lệ thắng mùa này", f"{ra['wpct']:.0%} ({ra['sw']}/{ra['sg']})", f"{rh['wpct']:.0%} ({rh['sw']}/{rh['sg']})"),
                    ("Ngày nghỉ", f"{a['rest']}", f"{h['rest']}"),
                    ("Đấu 2 ngày liên tiếp", "Có" if a["b2b"] else "Không", "Có" if h["b2b"] else "Không"),
                    ("Số trận trong 7 ngày qua", f"{a['g7']}", f"{h['g7']}"),
                ]
                st.table(pd.DataFrame(rows, columns=["Chỉ số", f"{g['away']} (khách)", f"{g['home']} (nhà)"]).set_index("Chỉ số"))
                p_mkt = m["ev"]["p_home_market"]
                st.caption(f"Xác suất {g['home']} thắng: mô hình ML {card['p_home']:.0%}"
                           + (f" · thị trường (kèo h2h đã bỏ biên nhà cái) {p_mkt:.0%}" if p_mkt else ""))


def sidebar_info(bundle):
    m, meta = bundle["metrics"], bundle["meta"]
    with st.sidebar:
        st.header("🤖 Về mô hình")
        st.caption(f"Huấn luyện trên {meta['n_games']:,} trận ({meta['data_from']} → {meta['data_to']}).")
        st.write(f"**Xác suất thắng:** {bundle['chosen']['clf']}  \n**Tổng điểm:** {bundle['chosen']['reg']}")
        acc = m.get("cv_acc", {}).get(bundle["chosen"]["clf"])
        if acc:
            st.write(f"Độ chính xác dự đoán đội thắng (kiểm chứng theo thời gian): **~{acc:.0%}**")
        st.write(f"Sai số tổng điểm trung bình: **~{m['reg'][bundle['chosen']['reg']]['mae']:.0f} điểm**")
        st.caption("Mô hình không biết chấn thương, đội hình thi đấu hay tin tức. Chỉ mang tính tham khảo thống kê.")
        if st.button("Đăng xuất"):
            st.session_state.clear()
            st.rerun()


def main():
    st.set_page_config(page_title="NBA AI Analyst", page_icon="🏀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    require_password()

    st.title("🏀 NBA AI Analyst")
    st.caption("Mô hình Machine Learning tính xác suất & tổng điểm → Claude viết phân tích tiếng Việt cho từng trận.")
    try:
        bundle = load_bundle()
    except Exception as e:
        st.error(f"Không tải được mô hình: {e}")
        st.stop()
    sidebar_info(bundle)

    today_et = datetime.now(TZ_ET).date()
    c1, c2 = st.columns([1, 3])
    target = c1.date_input("Ngày thi đấu (giờ Mỹ – ET)", value=today_et,
                           min_value=today_et, max_value=today_et + timedelta(days=7),
                           help="Lịch NBA/API tính theo giờ miền Đông Mỹ.")
    go = c2.button("🔄 Cập nhật trận đấu hôm nay", type="primary")
    c2.caption("Dự đoán dùng dữ liệu các trận đã kết thúc đến hết ngày hôm qua (giờ ET).")

    if go:
        bdl_key, claude_key = get_secret("BALLDONTLIE_API_KEY"), get_secret("ANTHROPIC_API_KEY")
        odds_key = get_secret("ODDS_API_KEY")
        if not bdl_key or not claude_key:
            st.error("Thiếu `BALLDONTLIE_API_KEY` hoặc `ANTHROPIC_API_KEY` trong st.secrets.")
            st.stop()
        model_name = get_secret("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        use_search = str(get_secret("ANOMALY_WEB_SEARCH", False)).strip().lower() in ("1", "true", "yes")
        try:
            with st.status("Đang cập nhật...", expanded=True) as status:
                # 1) đồng bộ trạng thái đội đến hết hôm qua
                live = st.session_state.get("live")
                through = today_et - timedelta(days=1)
                if live is None or live["through"] < through:
                    state = live["state"] if live else copy.deepcopy(bundle["state"])
                    recent = list(live.get("recent", [])) if live else []
                    st.write("📥 Đồng bộ kết quả các trận gần đây từ balldontlie.io ...")
                    holder = st.empty()
                    n_new, new_games = sync_state(state, bdl_key, through, on_page=lambda n: holder.write(f"   đã tải {n} trận"))
                    recent.extend(new_games)
                    live = {"state": state, "through": through, "n_new": n_new, "recent": recent}
                    st.session_state["live"] = live
                    st.write(f"✅ Đã cập nhật {n_new} trận mới vào trạng thái các đội.")
                # 2) lịch thi đấu ngày được chọn
                st.write(f"📅 Lấy lịch thi đấu ngày {target.strftime('%d/%m/%Y')} ...")
                games = [norm_game(g) for g in fetch_games(target, target, bdl_key)]
                cards = predict_games(bundle, live["state"], games)
                if not cards:
                    status.update(label="Không có trận nào trong ngày này", state="complete")
                    st.session_state["results"] = {"date": target, "cards": [], "analysis": {}}
                else:
                    # 3) kèo thị trường + Power Rating + phát hiện nghịch lý
                    market, ratings, odds_note, odds_left = {}, None, None, None
                    if not odds_key:
                        odds_note = "Chưa cấu hình `ODDS_API_KEY` trong st.secrets nên bỏ qua phần kèo & cảnh báo nghịch lý."
                    else:
                        try:
                            st.write("📈 Lấy tỷ lệ kèo (spreads, h2h) từ The Odds API ...")
                            events, odds_left = fetch_odds_cached(odds_key)
                            summaries = [s for s in (summarize_event(e) for e in events) if s]
                            st.write("⚖️ Tính Power Rating (CSV lịch sử + 10 trận gần nhất) ...")
                            hist = load_history_csv()
                            ratings = compute_power_ratings(combine_games(hist, live.get("recent", [])), target)
                            for c in cards:
                                m = evaluate_market(c, match_event(c["game"], summaries), ratings)
                                if m:
                                    market[c["game"]["id"]] = m
                            if not market:
                                odds_note = "The Odds API chưa có kèo cho các trận của ngày này (nhà cái thường mở kèo sát ngày thi đấu)."
                        except Exception as e:
                            ratings, market = None, {}
                            odds_note = f"Không lấy được kèo/Power Rating: {type(e).__name__}: {e}"

                    anomalies = sorted([(c, market[c["game"]["id"]]) for c in cards
                                        if market.get(c["game"]["id"], {}).get("level")], key=anomaly_sort_key)

                    st.write(f"🤖 Mô hình đã tính xác suất cho {len(cards)} trận. Đang nhờ Claude viết phân tích ...")
                    bar = st.progress(0.0)
                    total_jobs = len(cards) + len(anomalies)
                    done = {"n": 0}

                    def tick():
                        done["n"] += 1
                        bar.progress(min(done["n"] / total_jobs, 1.0))

                    cache = st.session_state.setdefault("analysis_cache", {})
                    analyses = run_analyses(claude_key, model_name, cards, cache, on_done=tick)
                    anomaly_analyses = {}
                    if anomalies:
                        st.write(f"🚨 Phát hiện {len(anomalies)} trận nghịch lý. Đang nhờ Claude phân tích sâu nguyên nhân ...")
                        anomaly_analyses = run_anomaly_analyses(claude_key, model_name, anomalies, cache,
                                                                use_search=use_search, on_done=tick)
                    st.session_state["results"] = {
                        "date": target, "cards": cards, "analysis": analyses, "made_at": datetime.now(TZ_VN),
                        "market": market, "anomaly_analysis": anomaly_analyses, "ratings": ratings,
                        "odds_note": odds_note, "odds_left": odds_left, "use_search": use_search,
                    }
                    status.update(label=f"Hoàn tất: {len(cards)} trận" + (f", {len(anomalies)} nghịch lý" if anomalies else ""),
                                  state="complete", expanded=False)
        except PermissionError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Có lỗi khi cập nhật: {type(e).__name__}: {e}")

    res = st.session_state.get("results")
    if res is None:
        st.info("Bấm **Cập nhật trận đấu hôm nay** để bắt đầu.")
        return
    if not res["cards"]:
        st.info(f"Không có trận NBA nào ngày {res['date'].strftime('%d/%m/%Y')} (giờ ET).")
        return

    market = res.get("market") or {}
    st.subheader(f"Ngày {res['date'].strftime('%d/%m/%Y')} (ET) – {len(res['cards'])} trận")
    if res.get("made_at"):
        left = f" · The Odds API còn {res['odds_left']} lượt gọi" if res.get("odds_left") else ""
        st.caption(f"Cập nhật lúc {res['made_at'].strftime('%H:%M %d/%m/%Y')} (giờ VN){left}")
    if res.get("odds_note"):
        st.warning(res["odds_note"])

    # --- mục cảnh báo nghịch lý luôn nằm trên cùng ---
    if res.get("ratings") and market:
        anomalies = sorted([(c, market[c["game"]["id"]]) for c in res["cards"]
                            if market.get(c["game"]["id"], {}).get("level")], key=anomaly_sort_key)
        render_anomaly_section(anomalies, res.get("anomaly_analysis") or {}, res.get("use_search", False))
        st.divider()

    st.subheader("Tổng quan các trận")
    rows = []
    for c in res["cards"]:
        m = market.get(c["game"]["id"])
        row = {
            "Trận": f"{c['game']['away']} @ {c['game']['home']}",
            "Giờ": fmt_tip(c["game"]["tip"]),
            "Xác suất khách thắng": f"{c['p_away']:.0%}",
            "Xác suất nhà thắng": f"{c['p_home']:.0%}",
            "Tổng điểm dự đoán": f"{c['total']:.0f}",
        }
        if market:
            row["Kèo chấp (nhà)"] = f"{m['spread_home']:+.1f}" if m else "-"
            row["Power Rating (nhà)"] = f"{-m['model_margin']:+.1f}" if m else "-"
            row["Chênh lệch"] = f"{m['gap']:+.1f}" if m else "-"
            row["Cảnh báo"] = ("🔴 Nghịch lý" if m["level"] == "red" else "🟡 Nghịch lý") if m and m["level"] else ""
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    for c in res["cards"]:
        text, err = res["analysis"].get(c["game"]["id"], (None, None))
        render_card(c, text, err, market.get(c["game"]["id"]))

    if res.get("ratings"):
        r = res["ratings"]
        with st.expander(f"📊 Bảng xếp hạng Power Rating ({r['n_games']:,} trận, tính đến trước {r['as_of']})"):
            st.caption(f"Đơn vị: điểm hiệu số trên sân trung lập. Lợi thế sân nhà ước lượng ≈ {r['hca']:.1f} điểm. "
                       f"Power = {1 - W_FORM:.0%} rating nền (hồi quy ridge, trọng số giảm theo thời gian, hiệu chỉnh đối thủ) "
                       f"+ {W_FORM:.0%} phong độ {FORM_N} trận gần nhất. Nhóm: Mạnh = top 10, Yếu = 10 đội cuối.")
            st.dataframe(ratings_table(r), hide_index=True)

if __name__ == "__main__":
    main()
