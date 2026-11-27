"""
NBA AI Analyst - ứng dụng Streamlit
-----------------------------------
Luồng: balldontlie.io (lịch thi đấu + kết quả mới) -> model_nba.pkl (xác suất thắng, tổng điểm)
       -> Claude API (bài phân tích tiếng Việt cho từng trận) -> hiển thị.
Mới:   du_lieu_nba_lich_su.csv + 10 trận gần nhất (balldontlie) -> Power Rating từng đội
       -> so với kèo chấp (spreads/h2h) từ The Odds API -> gắn cờ NGHỊCH LÝ THỊ TRƯỜNG
       -> Claude phân tích sâu nguyên nhân các trận nghịch lý.
Mới:   Tab LIVE NFL & NCAAF: ESPN scoreboard (trận status 'in') + kèo live The Odds API -> bắt nghịch lý;
       Claude chỉ chạy khi bấm nút; tự làm mới bằng streamlit-autorefresh (tùy chọn).

Cần có cạnh file này:  model_nba.pkl  (tạo bằng train_model.py)  và  du_lieu_nba_lich_su.csv
Cần khai báo trong st.secrets: APP_PASSWORD, BALLDONTLIE_API_KEY, ANTHROPIC_API_KEY, ODDS_API_KEY
(tùy chọn: CLAUDE_MODEL; ANOMALY_WEB_SEARCH = true để Claude tra tin chấn thương bằng web search)
Thư viện: xem requirements.txt (có streamlit-autorefresh).
"""
import copy
import difflib
import hmac
import html
import json
import re
import time
import unicodedata
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

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:          # thiếu thư viện -> tab Live vẫn chạy, chỉ không tự làm mới được
    st_autorefresh = None

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
# LIVE NFL & NCAAF (ESPN scoreboard + The Odds API live spread/total)
# ----------------------------------------------------------------------------
# - ESPN: miễn phí, không cần khóa. Chỉ lấy trận có status.type.state == "in".
# - The Odds API: chỉ gọi khi CÓ trận đang diễn ra, mỗi giải một lần/chu kỳ, và dùng chung bộ nhớ
#   đệm giữa các lần tự làm mới -> tiết kiệm hạn mức (mỗi lần gọi = 2 credit: spreads + totals).
# - Claude: CHỈ chạy khi bấm nút trên từng trận (chạy nền, không chặn giao diện).
# ============================================================================
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/{slug}/scoreboard"
LIVE_LEAGUES = {
    "NFL": {"slug": "nfl", "odds_key": "americanfootball_nfl", "params": {"limit": 100}, "sd_full": 13.5},
    "NCAAF": {"slug": "college-football", "odds_key": "americanfootball_ncaaf",
              "params": {"groups": 80, "limit": 400}, "sd_full": 15.0},   # groups=80: FBS
}
QUARTER_S = 900                 # mỗi hiệp 15 phút (NFL và NCAAF)
LIVE_ODDS_MAX_AGE_S = 600       # kèo cập nhật quá 10 phút -> coi là cũ (có thể là kèo trước trận)

# Ngưỡng nghịch lý kèo live
LIVE_LEAD_MIN = 7               # dẫn >= 7 điểm mà vẫn là cửa dưới -> nghịch lý
LIVE_MIN_FRAC = 0.03            # bỏ qua phép thử khi còn < 3% thời gian (~1,8 phút)
LIVE_MIN_DRIFT = 4.0            # kèo ngụ ý hiệu số còn lại tối thiểu 4 điểm mới xét
LIVE_Z_YELLOW, LIVE_Z_RED = 1.0, 1.6
PACE_YELLOW_LO, PACE_RED_LO = 0.50, 0.35     # nhịp ghi điểm kèo ngụ ý / nhịp thực tế
PACE_YELLOW_HI, PACE_RED_HI = 1.40, 1.80
PACE_MIN_ELAPSED_S = 900        # cần >= 1 hiệp đã đá mới so nhịp

LIVE_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích bóng bầu dục Mỹ (NFL / NCAA Football), viết tiếng Việt tự nhiên, rõ ràng, súc tích.

Bạn nhận một khối JSON là ẢNH CHỤP một trận đang diễn ra: tỷ số, hiệp, đồng hồ, đội cầm bóng, down & distance, vị trí trên sân, \
tình huống vừa xảy ra, cùng kèo chấp (spread) và Tài/Xỉu (total) trực tiếp đồng thuận từ nhiều nhà cái, và các cờ cảnh báo do hệ thống tính.

QUY TẮC BẮT BUỘC:
- Chỉ dùng số liệu trong JSON. TUYỆT ĐỐI không bịa chấn thương, tên cầu thủ, thời tiết hay thống kê không có trong JSON.
- Dữ liệu ESPN và kèo đến từ hai nguồn khác nhau, có thể lệch nhau từ vài chục giây đến vài phút. Khi tỷ số và kèo có vẻ mâu thuẫn, \
hãy xét khả năng do ĐỘ TRỄ trước khi gọi là bất thường thật.
- Bạn được dùng kiến thức bóng bầu dục phổ thông (ví dụ 4th & dài thường là punt hoặc field goal, trong red zone khả năng ghi điểm cao, \
đội cầm bóng thường được cộng khoảng 2–3 điểm trong kèo live) nhưng phải nói rõ đó là nhận định chung, không phải số liệu.
- Không khuyến khích cá cược, không đưa lời khuyên đặt cược. Đây là phân tích tình huống mang tính tham khảo.

ĐỊNH DẠNG (Markdown, khoảng 200–300 từ, không dùng tiêu đề cấp 1/2):
**🏈 Cục diện hiện tại:** 2–3 câu về tỷ số, thời gian, đội cầm bóng, down & distance, vị trí sân.
**📈 Kèo đang nói gì:** 2–3 câu: spread/total live ngụ ý gì về phần còn lại của trận, so với tỷ số và thời gian.
**🚩 Điểm bất thường & giả thuyết:** 2–3 gạch đầu dòng, mỗi dòng gắn với một con số trong JSON; nếu không có cờ nào thì nói rõ kèo khớp với diễn biến.
**⚠️ Lưu ý:** 1–2 gạch đầu dòng (độ trễ dữ liệu, giới hạn của phân tích).
**✅ Kết luận:** 1 câu."""


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(n, f"{n}th")


def parse_iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def period_label(p):
    if 1 <= p <= 4:
        return f"Q{p}"
    return "OT" if p == 5 else (f"OT{p - 4}" if p > 5 else "—")


# ---------------------------- ESPN ------------------------------------------
def fetch_espn_events(league):
    """Lấy sự kiện từ ESPN cho ngày ET hôm qua + hôm nay (trận khuya vắt qua nửa đêm vẫn được thấy)."""
    cfg = LIVE_LEAGUES[league]
    url = ESPN_SCOREBOARD.format(slug=cfg["slug"])
    today = datetime.now(TZ_ET).date()
    seen, out, ok, last_err = set(), [], 0, None
    for d in (today - timedelta(days=1), today):
        try:
            r = requests.get(url, params={**cfg["params"], "dates": d.strftime("%Y%m%d")},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            data = r.json()
            ok += 1
        except Exception as e:
            last_err = e
            continue
        for ev in data.get("events", []) or []:
            if ev.get("id") not in seen:
                seen.add(ev.get("id"))
                out.append(ev)
    if ok == 0:
        raise RuntimeError(f"Không gọi được ESPN ({league}): {type(last_err).__name__}: {last_err}")
    return out, datetime.now(timezone.utc)


def norm_live_game(ev, league):
    """Chuẩn hóa một sự kiện ESPN. Mọi trường đều phòng thủ vì ESPN không có tài liệu chính thức."""
    comp = (ev.get("competitions") or [{}])[0]
    stt = ev.get("status") or comp.get("status") or {}
    typ = stt.get("type") or {}
    teams = {}
    for c in comp.get("competitors", []) or []:
        t = c.get("team") or {}
        teams[c.get("homeAway")] = {
            "id": str(t.get("id")), "abbr": t.get("abbreviation") or "?",
            "name": t.get("displayName") or t.get("name") or "?",
            "score": int(_num(c.get("score"), 0) or 0),
        }
    if "home" not in teams or "away" not in teams:
        return None
    sit = comp.get("situation") or {}
    poss_id = sit.get("possession")
    poss = next((t["abbr"] for t in teams.values() if poss_id is not None and t["id"] == str(poss_id)), None)

    clock_s = _num(stt.get("clock"))
    disp = stt.get("displayClock") or ""
    if clock_s is None and ":" in disp:
        try:
            mm, ss = disp.split(":")[:2]
            clock_s = int(mm) * 60 + float(ss)
        except ValueError:
            pass

    dd = sit.get("shortDownDistanceText") or sit.get("downDistanceText")
    if not dd and sit.get("down") and sit.get("distance") is not None:
        dd = f"{_ordinal(int(sit['down']))} & {sit['distance']}"
    odds0 = (comp.get("odds") or [{}])[0] or {}
    return {
        "key": f"{league}-{ev.get('id')}", "id": ev.get("id"), "league": league,
        "state": typ.get("state"), "detail": typ.get("shortDetail") or typ.get("detail") or "",
        "period": int(_num(stt.get("period"), 0) or 0), "clock_s": clock_s, "clock_txt": disp or "—",
        "home": teams["home"], "away": teams["away"], "poss": poss,
        "down_dist": dd, "spot": sit.get("possessionText"), "to_endzone": sit.get("yardsToEndzone"),
        "red_zone": bool(sit.get("isRedZone")), "last_play": (sit.get("lastPlay") or {}).get("text"),
        "home_to": sit.get("homeTimeouts"), "away_to": sit.get("awayTimeouts"),
        "espn_odds": {"details": odds0.get("details"), "ou": odds0.get("overUnder"),
                      "provider": (odds0.get("provider") or {}).get("name")} if odds0 else None,
        "start": parse_iso(ev.get("date")),
    }


def live_time_info(g):
    """(giây còn lại, tỉ lệ thời gian còn lại) hoặc None khi hiệp phụ / thiếu đồng hồ."""
    p = g["period"]
    if p < 1 or p > 4 or g["clock_s"] is None:
        return None
    rem = (4 - p) * QUARTER_S + min(max(g["clock_s"], 0.0), QUARTER_S)
    return rem, rem / (4 * QUARTER_S)


# ---------------------------- The Odds API (live) ---------------------------
def fetch_live_odds(api_key, sport_key):
    r = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/", params={
        "apiKey": api_key, "regions": ODDS_REGIONS, "markets": "spreads,totals",
        "oddsFormat": "decimal", "dateFormat": "iso"}, timeout=30)
    if r.status_code == 401:
        raise PermissionError("The Odds API từ chối khóa (HTTP 401). Kiểm tra ODDS_API_KEY.")
    if r.status_code == 429:
        raise RuntimeError("The Odds API báo hết hạn mức hoặc gọi quá nhanh (HTTP 429).")
    r.raise_for_status()
    return r.json(), {"remaining": r.headers.get("x-requests-remaining"), "last": r.headers.get("x-requests-last")}


@st.cache_resource
def odds_store():
    return {}


def get_live_odds(api_key, league, min_interval_s, force=False):
    """Gọi The Odds API tối đa 1 lần / `min_interval_s` giây cho mỗi giải (dùng chung mọi lần tự làm mới)."""
    store, key, now = odds_store(), LIVE_LEAGUES[league]["odds_key"], time.time()
    ent = store.get(key)
    wait = max(min_interval_s, 60) if (ent and ent.get("err")) else min_interval_s   # lỗi -> chờ ít nhất 60s mới thử lại
    if ent and not force and now - ent["t"] < wait:
        return ent
    try:
        events, hdr = fetch_live_odds(api_key, key)
        ent = {"t": now, "events": events, "remaining": hdr["remaining"], "last": hdr["last"], "err": None}
    except Exception as e:
        ent = {"t": now, "events": ent["events"] if ent else [], "remaining": ent.get("remaining") if ent else None,
               "last": None, "err": f"{type(e).__name__}: {e}"}
    store[key] = ent
    return ent


def summarize_live_event(ev, now):
    """Kèo đồng thuận (trung vị) chỉ từ các nhà cái cập nhật gần đây. stale=True nếu không nhà cái nào còn mới."""
    home_name, away_name = ev.get("home_team"), ev.get("away_team")
    sp, tt = [], []                       # (giá trị, tuổi giây)
    for bk in ev.get("bookmakers", []) or []:
        bk_up = parse_iso(bk.get("last_update"))
        for mk in bk.get("markets", []) or []:
            up = parse_iso(mk.get("last_update")) or bk_up
            age = (now - up).total_seconds() if up else None
            outs = mk.get("outcomes", []) or []
            if mk.get("key") == "spreads":
                pt = next((o.get("point") for o in outs if o.get("name") == home_name), None)
                if pt is not None:
                    sp.append((float(pt), age))
            elif mk.get("key") == "totals":
                pt = next((o.get("point") for o in outs if str(o.get("name")).lower() == "over"), None)
                if pt is not None:
                    tt.append((float(pt), age))
    if not sp and not tt:
        return None

    def pick(vals):
        fresh = [v for v, a in vals if a is None or a <= LIVE_ODDS_MAX_AGE_S]
        use = fresh if fresh else [v for v, _ in vals]
        ages = [a for v, a in vals if a is not None]
        return use, len(fresh), (min(ages) if ages else None)

    su, sf, s_age = pick(sp)
    tu, tf, t_age = pick(tt)
    ages = [a for a in (s_age, t_age) if a is not None]
    return {
        "home_name": home_name, "away_name": away_name, "commence": parse_iso(ev.get("commence_time")),
        "spread_home": float(np.median(su)) if su else None,
        "spread_min": min(su) if su else None, "spread_max": max(su) if su else None, "n_spread": len(su),
        "total": float(np.median(tu)) if tu else None,
        "total_min": min(tu) if tu else None, "total_max": max(tu) if tu else None, "n_total": len(tu),
        "age_s": min(ages) if ages else None,
        "stale": (bool(sp) and sf == 0) or (bool(tt) and tf == 0) or (not sp and not tt),
    }


def _norm_team(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\bst\b", "state", s)
    return " ".join(s.split())


def _sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_live_odds(g, summaries):
    """Ghép trận ESPN với sự kiện The Odds API theo tên đội (mờ) + giờ bắt đầu. Tự đảo dấu spread nếu Odds API đảo nhà/khách."""
    gh, ga = _norm_team(g["home"]["name"]), _norm_team(g["away"]["name"])
    best, best_score, swapped = None, 0.0, False
    for s in summaries:
        eh, ea = _norm_team(s["home_name"]), _norm_team(s["away_name"])
        direct, swap = min(_sim(gh, eh), _sim(ga, ea)), min(_sim(gh, ea), _sim(ga, eh))
        score, sw = (direct, False) if direct >= swap else (swap, True)
        if score < 0.80:
            continue
        if g["start"] and s["commence"] and abs((g["start"] - s["commence"]).total_seconds()) > 8 * 3600:
            continue
        if score > best_score:
            best, best_score, swapped = s, score, sw
    if best is None:
        return None
    out = dict(best)
    if swapped and out["spread_home"] is not None:
        out["spread_home"] = -out["spread_home"]
        out["spread_min"], out["spread_max"] = -best["spread_max"], -best["spread_min"]
    return out


# ---------------------------- Bắt nghịch lý kèo live ------------------------
def evaluate_live_market(g, mk):
    """So kèo live với diễn biến thực tế. Trả về dict số liệu + cờ. Không gắn cờ khi kèo cũ hoặc đang hiệp phụ."""
    hs, as_ = g["home"]["score"], g["away"]["score"]
    cur_margin, cur_total = hs - as_, hs + as_
    ti = live_time_info(g)
    out = {"mk": mk, "cur_margin": cur_margin, "cur_total": cur_total, "ti": ti, "flags": [], "level": None,
           "mkt_margin": None, "rem_margin": None, "rem_total": None, "z": None, "ratio": None}
    sh = mk.get("spread_home")
    if sh is not None:
        out["mkt_margin"] = -sh                     # hiệu số cuối trận (nhà - khách) mà kèo ngụ ý
        out["rem_margin"] = out["mkt_margin"] - cur_margin
    if mk.get("total") is not None:
        out["rem_total"] = mk["total"] - cur_total
    if mk["stale"] or ti is None:
        return out
    rem_s, frac = ti
    h, a, flags = g["home"]["abbr"], g["away"]["abbr"], out["flags"]

    if sh is not None:
        leader, trailing = (h, a) if cur_margin > 0 else (a, h)
        fav = h if sh < 0 else (a if sh > 0 else None)
        lead = abs(cur_margin)
        if lead >= LIVE_LEAD_MIN and fav == trailing and abs(sh) >= 1.0:
            flags.append({"code": "LEAD_UNDERDOG", "level": "red" if (lead >= 14 or frac <= 0.25) else "yellow", "text": (
                f"{leader} đang dẫn {lead} điểm ({max(hs, as_)}–{min(hs, as_)}) nhưng thị trường vẫn cho {trailing} là đội chấp "
                f"{abs(sh):.1f} điểm - kèo live ngược với tỷ số.")})
        if frac >= LIVE_MIN_FRAC and out["rem_margin"] is not None:
            sd = LIVE_LEAGUES[g["league"]]["sd_full"] * np.sqrt(frac)
            z = abs(out["rem_margin"]) / sd
            out["z"] = float(z)
            if abs(out["rem_margin"]) >= LIVE_MIN_DRIFT and z >= LIVE_Z_YELLOW:
                who = h if out["rem_margin"] > 0 else a
                flags.append({"code": "DRIFT", "level": "red" if z >= LIVE_Z_RED else "yellow", "text": (
                    f"Kèo live ngụ ý {who} còn thắng thêm {abs(out['rem_margin']):.1f} điểm trong {frac:.0%} thời gian còn lại "
                    f"(dao động tự nhiên chỉ khoảng ±{sd:.1f} điểm) - lớn hơn diễn biến thông thường. "
                    f"Quyền cầm bóng thường chỉ giải thích được 2–3 điểm.")})

    if out["rem_total"] is not None:
        if out["rem_total"] <= 0:
            flags.append({"code": "TOTAL_PASSED", "level": "red", "text": (
                f"Tài/Xỉu live ({mk['total']:.1f}) không cao hơn tổng điểm hiện tại ({cur_total}) - "
                f"kèo có thể chưa cập nhật hoặc đã bị vượt.")})
        elif frac >= LIVE_MIN_FRAC and (3600 - rem_s) >= PACE_MIN_ELAPSED_S and cur_total >= 10:
            elapsed_min = (3600 - rem_s) / 60
            ratio = (out["rem_total"] / (rem_s / 60)) / (cur_total / elapsed_min)
            out["ratio"] = float(ratio)
            lvl = None
            if ratio <= PACE_RED_LO or ratio >= PACE_RED_HI:
                lvl = "red"
            elif ratio <= PACE_YELLOW_LO or ratio >= PACE_YELLOW_HI:
                lvl = "yellow"
            if lvl:
                flags.append({"code": "PACE", "level": lvl, "text": (
                    f"Tài/Xỉu lệch nhịp: trận đang ghi {cur_total / elapsed_min:.2f} điểm/phút, nhưng kèo live ngụ ý phần còn lại chỉ "
                    f"{out['rem_total'] / (rem_s / 60):.2f} điểm/phút (bằng {ratio:.0%} nhịp hiện tại).")})
    out["level"] = "red" if any(f["level"] == "red" for f in flags) else ("yellow" if flags else None)
    return out


def live_sort_key(item):
    g, res = item
    return ({"red": 0, "yellow": 1}.get(res["level"], 2) if res else 3, g["key"])


# ---------------------------- Claude (chạy nền, chỉ khi bấm nút) ------------
def live_payload(g, res):
    mk, ti = (res or {}).get("mk"), (res or {}).get("ti")
    return {
        "giai": g["league"], "trang_thai": g["detail"],
        "tran_dau": {"doi_nha": g["home"]["name"], "doi_khach": g["away"]["name"]},
        "dien_bien": {
            "ty_so_nha": g["home"]["score"], "ty_so_khach": g["away"]["score"],
            "hiep": period_label(g["period"]), "dong_ho_con_lai": g["clock_txt"],
            "ti_le_thoi_gian_con_lai_%": round(ti[1] * 100) if ti else None,
            "doi_cam_bong": g["poss"], "down_va_distance": g["down_dist"], "vi_tri_tren_san": g["spot"],
            "yards_toi_end_zone": g["to_endzone"], "trong_red_zone": g["red_zone"],
            "tinh_huong_vua_xay_ra": g["last_play"],
            "timeout_con_lai": {"nha": g["home_to"], "khach": g["away_to"]},
        },
        "ke_live": None if not mk else {
            "spread_doi_nha(am=doi_nha_chap)": mk.get("spread_home"),
            "khoang_spread_giua_cac_nha_cai": [mk.get("spread_min"), mk.get("spread_max")], "so_nha_cai_spread": mk.get("n_spread"),
            "tai_xiu_live": mk.get("total"), "khoang_tai_xiu": [mk.get("total_min"), mk.get("total_max")],
            "tuoi_du_lieu_ke_giay": None if mk.get("age_s") is None else round(mk["age_s"]),
            "ke_co_the_da_cu": bool(mk.get("stale")),
            "hieu_so_hien_tai_nha_tru_khach": res["cur_margin"],
            "hieu_so_cuoi_tran_ma_ke_ngu_y": res["mkt_margin"],
            "hieu_so_con_lai_ma_ke_ngu_y_cho_doi_nha": None if res["rem_margin"] is None else round(res["rem_margin"], 1),
            "tong_diem_hien_tai": res["cur_total"],
            "tong_diem_con_lai_ma_ke_ngu_y": None if res["rem_total"] is None else round(res["rem_total"], 1),
        },
        "ke_ESPN_tham_khao(co_the_la_ke_truoc_tran)": g["espn_odds"],
        "co_canh_bao": [f["text"] for f in (res or {}).get("flags", [])],
    }


def generate_live_analysis(client, model, payload):
    resp = client.messages.create(
        model=model, max_tokens=1200, system=LIVE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "Ảnh chụp trận đang diễn ra (JSON):\n"
                   + json.dumps(payload, ensure_ascii=False, indent=1)}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    if not text:
        raise RuntimeError("Claude trả về nội dung rỗng.")
    return text


@st.cache_resource
def ai_executor():
    return ThreadPoolExecutor(max_workers=2)


def _live_ai_worker(job, api_key, model, payload):
    try:
        client = Anthropic(api_key=api_key, max_retries=2, timeout=90.0)
        job["text"] = generate_live_analysis(client, model, payload)
        job["status"] = "done"
    except Exception as e:
        job["err"] = f"{type(e).__name__}: {e}"
        job["status"] = "error"


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
.flag-none{background:rgba(128,128,128,.12);border-left:8px solid #6b7280}
.sb{display:flex;align-items:center;justify-content:center;gap:1.6rem;margin:.3rem 0 .7rem;flex-wrap:wrap}
.sb .tm{text-align:center;min-width:110px}.sb .ab{font-weight:700;opacity:.85;font-size:.95rem}
.sb .sc{font-size:2.7rem;font-weight:800;line-height:1.1}
.sb .mid{text-align:center}.sb .q{font-weight:800;font-size:1.3rem}.sb .clk{font-size:1.1rem;opacity:.85}
.live-dot{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;background:#dc2626;margin-right:.35rem;animation:lp 1.4s infinite}
@keyframes lp{0%{opacity:1}50%{opacity:.25}100%{opacity:1}}
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


@st.cache_data(ttl=20, show_spinner=False)
def fetch_espn_cached(league):
    """Cache 20 giây: nhiều lần rerun (bấm nút, tự làm mới) không gọi ESPN dồn dập."""
    return fetch_espn_events(league)


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


def _pause_live_auto():
    """Gọi khi bấm cập nhật dự đoán: tắt tự làm mới để không ngắt quá trình cập nhật dài."""
    if st.session_state.get("live_auto"):
        st.session_state["live_auto"] = False
        st.session_state["live_auto_paused"] = True


def render_live_game(g, res, claude_key, model, jobs):
    mk = res["mk"] if res else None
    level = res["level"] if res else None
    h, a = g["home"], g["away"]
    with st.container(border=True):
        banner_cls = f"flag flag-{level}" if level else "flag"
        icon = "🔴" if level == "red" else ("🟡" if level == "yellow" else "")
        title = "NGHỊCH LÝ KÈO LIVE · " if level else ""
        st.markdown(
            f'<div class="{banner_cls}"><span>{icon} {title}{html.escape(a["name"])} @ {html.escape(h["name"])}'
            f'<span class="badge">{g["league"]}</span></span>'
            f'<span class="s"><span class="live-dot"></span>LIVE · {html.escape(g["detail"])}</span></div>',
            unsafe_allow_html=True)

        ball_a = " 🏈" if g["poss"] == a["abbr"] else ""
        ball_h = " 🏈" if g["poss"] == h["abbr"] else ""
        st.markdown(
            f'<div class="sb"><div class="tm"><div class="ab">{html.escape(a["abbr"])}{ball_a}</div><div class="sc">{a["score"]}</div></div>'
            f'<div class="mid"><div class="q">{period_label(g["period"])}</div><div class="clk">⏱ {html.escape(g["clock_txt"])}</div></div>'
            f'<div class="tm"><div class="ab">{html.escape(h["abbr"])}{ball_h}</div><div class="sc">{h["score"]}</div></div></div>',
            unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hiệp / đồng hồ", f"{period_label(g['period'])} · {g['clock_txt']}")
        c2.metric("Đội cầm bóng", g["poss"] or "—")
        if g["red_zone"]:
            c2.caption("🔴 Red zone")
        c3.metric("Down & Distance", g["down_dist"] or "—")
        c4.metric("Vị trí trên sân", g["spot"] or "—")
        if g["to_endzone"] is not None:
            c4.caption(f"Còn {g['to_endzone']} yards tới end zone")
        extra = []
        if g["last_play"]:
            extra.append(f"Vừa xảy ra: {g['last_play']}")
        if g["home_to"] is not None and g["away_to"] is not None:
            extra.append(f"Timeout còn lại: {a['abbr']} {g['away_to']} · {h['abbr']} {g['home_to']}")
        if extra:
            st.caption(" · ".join(extra))

        st.markdown("**📈 Đối chiếu kèo live**")
        if not mk:
            st.caption("Chưa ghép được kèo live từ The Odds API cho trận này (nhà cái có thể chưa mở kèo trong trận).")
        else:
            m1, m2, m3, m4 = st.columns(4)
            if mk["spread_home"] is not None:
                m1.metric(f"Spread live ({h['abbr']})", f"{mk['spread_home']:+.1f}")
                m1.caption(f"{mk['n_spread']} nhà cái: {mk['spread_min']:+.1f} → {mk['spread_max']:+.1f}")
            else:
                m1.metric("Spread live", "—")
            if mk["total"] is not None:
                m2.metric("Tài/Xỉu live", f"{mk['total']:.1f}")
                m2.caption(f"{mk['n_total']} nhà cái: {mk['total_min']:.1f} → {mk['total_max']:.1f}")
            else:
                m2.metric("Tài/Xỉu live", "—")
            m3.metric(f"Hiệu số hiện tại ({h['abbr']})", f"{res['cur_margin']:+d}")
            if res["rem_margin"] is not None:
                m3.caption(f"Kèo ngụ ý còn lại: {res['rem_margin']:+.1f}")
            m4.metric("Tổng điểm hiện tại", f"{res['cur_total']}")
            if res["rem_total"] is not None:
                m4.caption(f"Kèo ngụ ý còn lại: {res['rem_total']:+.1f}")
            age = mk["age_s"]
            st.caption("Độ mới của kèo: " + ("không rõ" if age is None else f"{age:.0f} giây trước"))
            if mk["stale"]:
                st.warning("Kèo từ The Odds API đã cũ (quá 10 phút), có thể là kèo trước trận hoặc nhà cái đã đóng kèo. "
                           "Không dùng để gắn cờ nghịch lý.")
            elif res["ti"] is None:
                st.caption("Đang hiệp phụ hoặc thiếu đồng hồ: không chạy phép thử nghịch lý.")
            for f in res["flags"]:
                st.markdown(f"{'🔴' if f['level'] == 'red' else '🟡'} {f['text']}")
            if not res["flags"] and not mk["stale"] and res["ti"] is not None:
                st.caption("✅ Kèo live khớp với diễn biến hiện tại.")
        if g["espn_odds"] and (g["espn_odds"].get("details") or g["espn_odds"].get("ou")):
            eo = g["espn_odds"]
            st.caption(f"Tham khảo kèo ESPN: {eo.get('details') or '—'} · O/U {eo.get('ou') or '—'}"
                       + (f" ({eo['provider']})" if eo.get("provider") else "") + " - có thể là kèo trước trận.")

        # --- Claude: chỉ chạy khi bấm nút ---
        st.markdown("---")
        job = jobs.get(g["key"])
        running = bool(job and job["status"] == "running")
        label = "🧠 AI Phân tích cục diện Live" if not job else "🔁 AI Phân tích lại cục diện Live"
        if st.button(label, key=f"live_ai_btn_{g['key']}", disabled=running or not claude_key,
                     help="Chỉ khi bấm nút này mới gọi Claude API (mỗi lần bấm = 1 lượt gọi, phân tích ảnh chụp tại thời điểm bấm)."):
            snap = (f"{a['abbr']} {a['score']}–{h['score']} {h['abbr']}, {period_label(g['period'])} {g['clock_txt']}")
            job = {"status": "running", "text": None, "err": None, "t": datetime.now(TZ_VN), "snap": snap}
            jobs[g["key"]] = job
            ai_executor().submit(_live_ai_worker, job, claude_key, model, live_payload(g, res))
            running = True
        if not claude_key:
            st.caption("Thiếu `ANTHROPIC_API_KEY` nên chưa dùng được phân tích AI.")
        elif not job:
            st.caption("Claude không tự chạy - chỉ phân tích khi bạn bấm nút.")
        if job:
            if job["status"] == "running":
                st.info("⏳ Claude đang phân tích ảnh chụp trận đấu ...")
            elif job["status"] == "done":
                st.caption(f"Phân tích lúc {job['t'].strftime('%H:%M:%S')} (giờ VN), theo ảnh chụp: {job['snap']}. "
                           "Trận có thể đã thay đổi kể từ đó.")
                st.markdown(job["text"])
            else:
                st.warning(f"Không tạo được phân tích. {job['err']}")


def render_live_tab():
    st.subheader("🔴 LIVE NFL & NCAAF (Real-Time)")
    if st.session_state.pop("live_auto_paused", False):
        st.info("Đã tự tắt làm mới tự động khi bạn cập nhật dự đoán (tránh làm gián đoạn quá trình cập nhật). Bật lại nếu cần.")

    c1, c2, c3 = st.columns([1.4, 1.2, 1])
    auto = c1.toggle("⏱️ Tự động làm mới sân đấu mỗi 60 giây", key="live_auto",
                     help="Dùng streamlit-autorefresh, miễn phí. Chỉ tải lại ESPN (miễn phí); kèo The Odds API có chu kỳ riêng bên cạnh.")
    interval = c2.selectbox("Chu kỳ lấy kèo (The Odds API)", [60, 120, 180, 300], index=2, key="live_odds_interval",
                            format_func=lambda s: f"{s // 60} phút",
                            help="Mỗi lần gọi tốn 2 credit cho mỗi giải đang có trận live (spreads + totals). Chỉ gọi khi có trận đang diễn ra.")
    force = c3.button("🔄 Làm mới ngay", help="Tải lại ESPN và ép lấy kèo mới (tốn credit The Odds API).")
    if force:
        fetch_espn_cached.clear()

    if auto:
        if st_autorefresh is None:
            st.warning("Chưa cài `streamlit-autorefresh` (xem requirements.txt) nên không tự làm mới được.")
        else:
            st_autorefresh(interval=60_000, key="live_autorefresh")

    odds_key, claude_key = get_secret("ODDS_API_KEY"), get_secret("ANTHROPIC_API_KEY")
    model = get_secret("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
    now = datetime.now(timezone.utc)

    # 1) ESPN: các trận đang diễn ra (state == "in")
    live_games, counts, fetched = [], {}, []
    for lg in LIVE_LEAGUES:
        try:
            events, ts = fetch_espn_cached(lg)
        except Exception as e:
            st.warning(str(e))
            continue
        fetched.append(ts)
        norm = [x for x in (norm_live_game(ev, lg) for ev in events) if x]
        counts[lg] = {s: sum(1 for x in norm if x["state"] == s) for s in ("in", "pre", "post")}
        live_games += [x for x in norm if x["state"] == "in"]
    if fetched:
        st.caption(f"Dữ liệu ESPN tải lúc {min(fetched).astimezone(TZ_VN).strftime('%H:%M:%S')} (giờ VN). "
                   "ESPN không phát từng giây: số liệu là ảnh chụp tại thời điểm tải, đồng hồ không chạy giữa hai lần làm mới.")
    if not live_games:
        st.info("Hiện không có trận NFL/NCAAF nào đang diễn ra. " +
                " · ".join(f"{lg}: {c['in']} đang đấu, {c['pre']} sắp diễn ra, {c['post']} đã kết thúc" for lg, c in counts.items()))
        return

    # 2) Kèo live: chỉ gọi The Odds API cho giải đang có trận live
    summaries, notes, credits = {}, [], []
    leagues_live = sorted({g["league"] for g in live_games})
    if not odds_key:
        notes.append("Chưa cấu hình `ODDS_API_KEY` trong st.secrets nên chỉ hiển thị diễn biến trận, chưa có kèo live.")
    else:
        for lg in leagues_live:
            ent = get_live_odds(odds_key, lg, interval, force=force)
            if ent["err"]:
                notes.append(f"{lg}: không lấy được kèo ({ent['err']}). Sẽ thử lại sau tối thiểu 1 phút.")
            summaries[lg] = [s for s in (summarize_live_event(e, now) for e in ent["events"]) if s]
            if ent.get("remaining") is not None:
                credits.append(f"{lg}: còn {ent['remaining']} credit (lần gọi gần nhất tốn {ent.get('last') or '?'})")
            st.session_state.setdefault("_odds_t", {})[lg] = ent["t"]
    for n in notes:
        st.warning(n)
    if credits:
        ages = [f"{lg} {int(time.time() - t)}s trước" for lg, t in st.session_state.get("_odds_t", {}).items() if lg in leagues_live]
        st.caption("The Odds API - " + " · ".join(credits) + " · kèo lấy lúc: " + ", ".join(ages))

    items = []
    for g in live_games:
        mk = match_live_odds(g, summaries.get(g["league"], [])) if summaries.get(g["league"]) else None
        items.append((g, evaluate_live_market(g, mk) if mk else None))
    items.sort(key=live_sort_key)

    # 3) Mục cảnh báo nghịch lý + danh sách trận
    flagged = [(g, r) for g, r in items if r and r["level"]]
    st.header("🚨 Cảnh báo Kèo Live Nghịch Lý")
    if flagged:
        n_red = sum(1 for _, r in flagged if r["level"] == "red")
        st.caption(f"{len(flagged)} trận bất thường ({n_red} cờ đỏ). Có thể do độ trễ giữa hai nguồn dữ liệu hoặc tin chấn thương "
                   "mà thị trường biết trước - không phải khuyến nghị đặt cược.")
        for g, r in flagged:
            box = st.error if r["level"] == "red" else st.warning
            box(f"**{g['away']['abbr']} @ {g['home']['abbr']}** ({g['league']}, {period_label(g['period'])} {g['clock_txt']}, "
                f"{g['away']['score']}–{g['home']['score']}): " + " | ".join(f["text"] for f in r["flags"]))
    else:
        st.success("Không có trận live nào có kèo bất thường so với diễn biến hiện tại."
                   if odds_key else "Cần `ODDS_API_KEY` để đối chiếu kèo live.")
    st.divider()
    st.subheader(f"Các trận đang diễn ra ({len(items)})")
    jobs = st.session_state.setdefault("live_ai", {})
    for g, r in items:
        render_live_game(g, r, claude_key, model, jobs)

    # 4) Có phân tích Claude đang chạy nền -> kiểm tra lại mỗi 3 giây (không tốn phí)
    if any(j["status"] == "running" for j in jobs.values()):
        if st_autorefresh is not None:
            st_autorefresh(interval=3000, limit=60, key="live_ai_poll")
        else:
            st.caption("Bấm “Làm mới ngay” để xem kết quả phân tích khi Claude xong.")




def render_pregame_tab(bundle, bundle_err):
    st.caption("Mô hình Machine Learning tính xác suất & tổng điểm → Claude viết phân tích tiếng Việt cho từng trận NBA.")
    if bundle is None:
        st.error(f"Không tải được mô hình: {bundle_err}")
        return
    today_et = datetime.now(TZ_ET).date()
    c1, c2 = st.columns([1, 3])
    target = c1.date_input("Ngày thi đấu (giờ Mỹ – ET)", value=today_et,
                           min_value=today_et, max_value=today_et + timedelta(days=7),
                           help="Lịch NBA/API tính theo giờ miền Đông Mỹ.")
    go = c2.button("🔄 Cập nhật trận đấu hôm nay", type="primary", on_click=_pause_live_auto)
    c2.caption("Dự đoán dùng dữ liệu các trận đã kết thúc đến hết ngày hôm qua (giờ ET).")

    if go:
        bdl_key, claude_key = get_secret("BALLDONTLIE_API_KEY"), get_secret("ANTHROPIC_API_KEY")
        odds_key = get_secret("ODDS_API_KEY")
        if not bdl_key or not claude_key:
            st.error("Thiếu `BALLDONTLIE_API_KEY` hoặc `ANTHROPIC_API_KEY` trong st.secrets.")
            return
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


def main():
    st.set_page_config(page_title="Sports AI Analyst", page_icon="🏟️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    require_password()

    st.title("🏟️ Sports AI Analyst")
    bundle, bundle_err = None, None
    try:
        bundle = load_bundle()
    except Exception as e:           # thiếu model_nba.pkl chỉ ảnh hưởng tab dự đoán, tab Live vẫn chạy
        bundle_err = str(e)
    if bundle is not None:
        sidebar_info(bundle)
    else:
        with st.sidebar:
            if st.button("Đăng xuất"):
                st.session_state.clear()
                st.rerun()

    tab_pre, tab_live = st.tabs(["📊 Dự đoán trước trận (NBA)", "🔴 LIVE NFL & NCAAF (Real-Time)"])
    with tab_pre:
        render_pregame_tab(bundle, bundle_err)
    with tab_live:
        render_live_tab()

if __name__ == "__main__":
    main()
