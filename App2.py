"""
NBA AI Analyst - ứng dụng Streamlit
-----------------------------------
Luồng: balldontlie.io (lịch thi đấu + kết quả mới) -> model_nba.pkl (xác suất thắng, tổng điểm)
       -> Claude API (bài phân tích tiếng Việt cho từng trận) -> hiển thị.

Cần có cạnh file này:  model_nba.pkl  (tạo bằng train_model.py)
Cần khai báo trong st.secrets: APP_PASSWORD, BALLDONTLIE_API_KEY, ANTHROPIC_API_KEY
(tùy chọn: CLAUDE_MODEL)
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
    Cần thiết vì file mô hình chỉ biết dữ liệu đến ngày huấn luyện (state['last_date'])."""
    if not state["last_date"]:
        return 0
    start = date.fromisoformat(state["last_date"]) + timedelta(days=1)
    if start > end_date:
        return 0
    games = [norm_game(g) for g in fetch_games(start, end_date, api_key, on_page)]
    games = [g for g in games if is_final(g) and g["home"] in state["teams"] and g["away"] in state["teams"]]
    games.sort(key=lambda g: (g["date"], g["id"] or 0))
    n = 0
    for g in games:
        if update_state(state, g["id"], int(g["season"]), g["date"], g["home"], g["away"], int(g["hs"]), int(g["vs"])):
            n += 1
    return n


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


def fmt_tip(dt):
    return dt.astimezone(TZ_VN).strftime("%H:%M %d/%m") + " (giờ VN)" if dt else "chưa rõ giờ"


def render_card(card, analysis, error):
    g = card["game"]
    ph, pa = card["p_home"] * 100, card["p_away"] * 100
    with st.container(border=True):
        status = ""
        if is_final(g):
            status = f'<span class="badge">Đã kết thúc: {html.escape(g["away"])} {g["vs"]} – {g["hs"]} {html.escape(g["home"])}</span>'
        elif g["postseason"]:
            status = '<span class="badge">Playoffs</span>'
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
        if not bdl_key or not claude_key:
            st.error("Thiếu `BALLDONTLIE_API_KEY` hoặc `ANTHROPIC_API_KEY` trong st.secrets.")
            st.stop()
        model_name = get_secret("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        try:
            with st.status("Đang cập nhật...", expanded=True) as status:
                # 1) đồng bộ trạng thái đội đến hết hôm qua
                live = st.session_state.get("live")
                through = today_et - timedelta(days=1)
                if live is None or live["through"] < through:
                    state = live["state"] if live else copy.deepcopy(bundle["state"])
                    st.write("📥 Đồng bộ kết quả các trận gần đây từ balldontlie.io ...")
                    holder = st.empty()
                    n_new = sync_state(state, bdl_key, through, on_page=lambda n: holder.write(f"   đã tải {n} trận"))
                    live = {"state": state, "through": through, "n_new": n_new}
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
                    st.write(f"🤖 Mô hình đã tính xác suất cho {len(cards)} trận. Đang nhờ Claude viết phân tích ...")
                    bar = st.progress(0.0)
                    done = {"n": 0}

                    def tick():
                        done["n"] += 1
                        bar.progress(min(done["n"] / len(cards), 1.0))

                    cache = st.session_state.setdefault("analysis_cache", {})
                    analyses = run_analyses(claude_key, model_name, cards, cache, on_done=tick)
                    st.session_state["results"] = {"date": target, "cards": cards, "analysis": analyses,
                                                   "made_at": datetime.now(TZ_VN)}
                    status.update(label=f"Hoàn tất: {len(cards)} trận", state="complete", expanded=False)
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

    st.subheader(f"Ngày {res['date'].strftime('%d/%m/%Y')} (ET) – {len(res['cards'])} trận")
    if res.get("made_at"):
        st.caption(f"Cập nhật lúc {res['made_at'].strftime('%H:%M %d/%m/%Y')} (giờ VN)")
    overview = pd.DataFrame([{
        "Trận": f"{c['game']['away']} @ {c['game']['home']}",
        "Giờ": fmt_tip(c["game"]["tip"]),
        "Xác suất khách thắng": f"{c['p_away']:.0%}",
        "Xác suất nhà thắng": f"{c['p_home']:.0%}",
        "Tổng điểm dự đoán": f"{c['total']:.0f}",
    } for c in res["cards"]])
    st.dataframe(overview, hide_index=True)
    for c in res["cards"]:
        text, err = res["analysis"].get(c["game"]["id"], (None, None))
        render_card(c, text, err)


if __name__ == "__main__":
    main()
