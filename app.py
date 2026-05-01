import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import pickle
import os
import time
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import streamlit.components.v1 as components  # type: ignore
from languages import LANGUAGES, LANG_LIST  # type: ignore
import sqlite3
import hashlib
from datetime import datetime

# ── Database Initialization ───────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('santhosh_ai.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    try:
        c.execute('''ALTER TABLE users ADD COLUMN email TEXT''')
    except:
        pass
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, sqft INTEGER,
                  bedrooms INTEGER, quality TEXT, est_price REAL, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_login(identifier, password):
    conn = sqlite3.connect('santhosh_ai.db')
    c = conn.cursor()
    c.execute('SELECT password, username FROM users WHERE username=? OR email=?', (identifier, identifier))
    data = c.fetchone()
    conn.close()
    if data and data[0] == make_hash(password):
        return True, data[1]
    return False, None

def add_user(username, password, email):
    conn = sqlite3.connect('santhosh_ai.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                  (username, make_hash(password), email))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def save_prediction(username, sqft, bedrooms, quality, price):
    conn = sqlite3.connect('santhosh_ai.db')
    c = conn.cursor()
    c.execute('INSERT INTO history (username, sqft, bedrooms, quality, est_price, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
              (username, sqft, bedrooms, quality, price, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="🏠 Santhosh AI – House Price Predictor", page_icon="🏠", layout="wide")

# ── Hide Streamlit Branding ───────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero Title ── */
.hero-wrapper {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f7971e44, #ffd20033);
    border: 1px solid #ffd20066;
    border-radius: 50px;
    padding: .35rem 1.1rem;
    font-size: .78rem;
    font-weight: 600;
    color: #ffd200;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #ffd200 50%, #f7971e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: .6rem;
    letter-spacing: -1px;
}
.hero-sub {
    color: #aaa;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
    max-width: 520px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}
.hero-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    border-radius: 99px;
    margin: 0 auto 2rem;
}

/* ── Price Card ── */
.price-card {
    background: linear-gradient(135deg, #f7971e18, #ffd20018);
    border: 1.5px solid #ffd200aa;
    border-radius: 20px;
    padding: 2.2rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 8px 40px #f7971e22;
}
.price-label {
    color: #ffd200;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: .5rem;
}
.price-value {
    font-size: 3.8rem;
    font-weight: 900;
    color: #fff;
    line-height: 1;
    letter-spacing: -2px;
}
.price-range {
    color: #aaa;
    font-size: .85rem;
    margin-top: .4rem;
}

/* ── Section Header ── */
.sec-hdr {
    font-size: 1rem;
    font-weight: 700;
    color: #ffd200;
    border-left: 4px solid #f7971e;
    padding-left: .8rem;
    margin: 1.8rem 0 .9rem;
    letter-spacing: .5px;
}

/* ── Cards ── */
.stage-card {
    background: #ffffff0a;
    border: 1px solid #ffffff15;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
    transition: border-color .2s;
}
.stage-card:hover { border-color: #ffd20055; }
.stage-title {
    font-size: .92rem;
    font-weight: 700;
    color: #ffd200;
    margin-bottom: .6rem;
}
.mat-row {
    color: #ccc;
    font-size: .82rem;
    padding: 2px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.qty-badge {
    background: #f7971e22;
    border: 1px solid #f7971e55;
    border-radius: 8px;
    padding: 1px 8px;
    color: #ffd200;
    font-weight: 600;
    font-size: .78rem;
}

/* ── Chips ── */
.chip {
    display: inline-block;
    background: #ffffff12;
    border: 1px solid #ffffff25;
    border-radius: 20px;
    padding: .25rem .8rem;
    margin: .2rem;
    font-size: .78rem;
    color: #ddd;
}
.chips-row { margin: .5rem 0 1rem; }

/* ── Sidebar polish ── */
section[data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #ffffff12;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label {
    font-size: .84rem;
    font-weight: 500;
    color: #ccc;
}
.sidebar-section-title {
    font-size: .72rem;
    font-weight: 700;
    color: #f7971e;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 1rem 0 .4rem;
}

/* ── Stat pills ── */
.stat-row {
    display: flex;
    gap: .6rem;
    flex-wrap: wrap;
    justify-content: center;
    margin: 1rem 0;
}
.stat-pill {
    background: #ffffff0e;
    border: 1px solid #ffffff18;
    border-radius: 12px;
    padding: .6rem 1.1rem;
    text-align: center;
    min-width: 90px;
}
.stat-pill-val {
    font-size: 1.3rem;
    font-weight: 800;
    color: #ffd200;
}
.stat-pill-lbl {
    font-size: .7rem;
    color: #888;
    margin-top: 1px;
}

/* ── Login page ── */
.login-hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.login-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #fff 0%, #ffd200 60%, #f7971e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.login-sub {
    color: #888;
    font-size: .95rem;
    margin-top: .4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Auth Flow ─────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.markdown("""
    <div class="login-hero">
        <div class="login-title">🏠 Santhosh AI</div>
        <div class="login-sub">House Price Predictor &amp; Construction Advisor</div>
    </div>
    """, unsafe_allow_html=True)

    col1, mid, col2 = st.columns([1, 1.4, 1])
    with mid:
        tab1, tab2 = st.tabs(["🔒 Login", "📝 Sign Up"])
        with tab1:
            u_login = st.text_input("Username or Gmail", key="l_u", placeholder="Enter username or email")
            p_login = st.text_input("Password", type="password", key="l_p", placeholder="Enter password")
            if st.button("Login →", width='stretch', type="primary"):
                success, real_username = verify_login(u_login, p_login)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = real_username
                    st.success("Welcome back! Redirecting…")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Invalid username/email or password.")
        with tab2:
            e_sign = st.text_input("Google Email", key="s_e", placeholder="example@gmail.com")
            u_sign = st.text_input("New Username", key="s_u", placeholder="Choose a username")
            p_sign = st.text_input("New Password", type="password", key="s_p", placeholder="Create a password")
            if st.button("Create Account →", width='stretch'):
                if u_sign and p_sign and e_sign:
                    if "@gmail.com" not in e_sign.lower():
                        st.error("Please provide a valid @gmail.com address!")
                    else:
                        if add_user(u_sign, p_sign, e_sign):
                            st.success("Account created! You can now login.")
                        else:
                            st.error("Username already exists!")
                else:
                    st.warning("Please fill all fields.")
    st.stop()

# ── Language selector ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<b style='color:#ffd200;font-size:1rem;'>👤 {st.session_state.username}</b>", unsafe_allow_html=True)
    if st.button("Logout", width='stretch'):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">🌐 Language</div>', unsafe_allow_html=True)
    sel_lang = st.selectbox("Language", LANG_LIST, index=0, label_visibility="collapsed")
    L = LANGUAGES[sel_lang]

    # ── SIDEBAR INPUTS ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">📐 House Details</div>', unsafe_allow_html=True)

    sqft = st.number_input(f"📐 {L['sqft']}", min_value=300, max_value=10000, value=1200, step=50)

    Q_LEVELS = {
        "Low":    {"mult": 0.78260869565, "label": "📉 Budget  ₹1800/sqft"},
        "Medium": {"mult": 1.0,           "label": "🏢 Standard  ₹2300/sqft"},
        "High":   {"mult": 1.21739130435, "label": "✨ Premium  ₹2800/sqft"},
    }
    quality_key = st.select_slider("🏗️ Construction Quality", options=["Low", "Medium", "High"], value="Medium")
    Q = Q_LEVELS[quality_key]

    st.markdown('<div class="sidebar-section-title">🏠 Rooms</div>', unsafe_allow_html=True)
    bedroom    = st.selectbox(f"🛏️ {L['bedrooms']}",  [1, 2, 3, 4, 5, 6], index=1)
    hall       = st.selectbox(f"🛋️ {L['halls']}",     [1, 2, 3, 4, 5],    index=0)
    kitchen    = st.selectbox(f"🍳 {L['kitchens']}",  [1, 2],              index=0)
    bathroom   = st.selectbox(f"🚿 {L['bathrooms']}", [1, 2, 3, 4, 5, 6], index=1)
    pooja_room = st.selectbox(f"🪔 {L['pooja']}",     [0, 1, 2, 3],        index=0)

    st.markdown('<div class="sidebar-section-title">🏢 Structure</div>', unsafe_allow_html=True)
    floor   = st.selectbox(f"🏢 {L['floors']}",  [1, 2, 3, 4, 5], index=0)
    parking = st.selectbox(f"🚗 {L['parking']}", [0, 1, 2, 3, 4], index=1)
    garden  = st.radio(f"🌿 {L['garden']}", [0, 1],
                       format_func=lambda x: L["yes"] if x else L["no"],
                       horizontal=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">📁 Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">🤖 AI Chat Key</div>', unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key (optional)", type="password",
                            key="gemini_api_key", help="Get free key at g.co/aistudio")

# Dataset upload handler
FEATURE_COLS = ['hall', 'bedroom', 'kitchen', 'sqft', 'floor', 'bathroom', 'garden', 'parking', 'pooja_room']
if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file)
        if all(col in new_df.columns for col in FEATURE_COLS + ['price']):
            new_df.to_csv("house_prediction.csv", index=False)
            if os.path.exists("house_model.pkl"):
                os.remove("house_model.pkl")
            st.cache_resource.clear()
            st.sidebar.success("✅ Dataset Updated! Model Retrained!")
        else:
            st.sidebar.error("❌ Missing required columns in CSV.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ── Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists('house_model.pkl'):
        try:
            with open('house_model.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    df = pd.read_csv('house_prediction.csv')
    model = LinearRegression()
    model.fit(df[FEATURE_COLS], df['price'])
    with open('house_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

model = load_model()

# ── Material estimation ────────────────────────────────────────────────────
def estimate_materials(sqft, hall, bedroom, kitchen, floor, bathroom, garden, parking, pooja_room, quality_key):
    s = sqft
    stages = [
        {"icon": "🧱", "title": "1. Foundation", "items": [
            ("Cement", f"{int(s*0.10)} Bags"), ("Sand", f"{int(s*0.35)} CFT"),
            ("Gravel (Jalli)", f"{int(s*.40)} CFT"), ("Steel Rods (TMT)", f"{int(s*1.0)} Kg"),
            ("Bricks / Stones", f"{int(s*4)} Nos"), ("Water", f"{int(s*10)} Litres")]},
        {"icon": "🏗️", "title": "2. Structure (Column, Beam, Slab)", "items": [
            ("Cement", f"{int(s*0.15)} Bags"), ("Sand", f"{int(s*0.55)} CFT"),
            ("Aggregate (Jalli)", f"{int(s*.60)} CFT"), ("Steel (TMT bars)", f"{int(s*3.0)} Kg"),
            ("Centering Sheets", f"{int(s*1.2)} Sqft")]},
        {"icon": "🧱", "title": "3. Walls", "items": [
            ("Bricks / AAC Blocks", f"{int(s*8)} Nos"),
            ("Cement", f"{int(s*0.08)} Bags"), ("Sand", f"{int(s*0.25)} CFT")]},
        {"icon": "🚪", "title": "4. Doors & Windows", "items": [
            ("Main Door", "1 Nos"), ("Bedroom Doors", f"{bedroom} Nos"),
            ("Bathroom Doors", f"{bathroom} Nos"),
            ("Window Frames", f"{(bedroom*2)+(bathroom)} Nos"),
            ("Glass Panels", f"{int(((bedroom*2)+bathroom)*6)} Sqft"),
            ("Locks", f"{1+bedroom} Nos")]},
        {"icon": "⚡", "title": "5. Electrical", "items": [
            ("Wires", f"{int(s*2.5)} Metres"), ("Switches & Sockets", f"{int(s/40 + bedroom*5)} Nos"),
            ("MCB Box", f"{floor} Nos"), ("Lights", f"{int(s/50 + bedroom*2)} Nos"),
            ("Fans", f"{hall+bedroom} Nos")]},
        {"icon": "🚿", "title": "6. Plumbing", "items": [
            ("PVC/CPVC Pipes", f"{int(s*0.4)} Metres"), ("Taps", f"{bathroom*3 + kitchen*2} Nos"),
            ("Shower Sets", f"{bathroom} Nos"), ("Toilet Fittings", f"{bathroom} Sets"),
            ("Water Tank", f"{bathroom*500} Litres")]},
        {"icon": "🧴", "title": "7. Plastering & Finishing", "items": [
            ("Cement", f"{int(s*0.04)} Bags"), ("Sand", f"{int(s*0.15)} CFT"),
            ("Wall Putty", f"{int(s*0.15)} Kg"), ("Primer", f"{int(s*0.025)} Litres"),
            ("Paint (2 coats)", f"{int(s*0.03)} Litres")]},
        {"icon": "🟦", "title": "8. Flooring", "items": [
            ("Tiles / Marble / Granite", f"{int(s*1.05)} Sqft"),
            ("Tile Adhesive", f"{int(s*0.02)} Bags"), ("Cement (base)", f"{int(s*0.03)} Bags")]},
        {"icon": "🍳", "title": "9. Kitchen", "items": [
            ("Granite Slab", f"{kitchen*22} Sqft"), ("Kitchen Sink", f"{kitchen} Nos"),
            ("Cabinets", f"{kitchen*3} Nos"), ("Wall Tiles", f"{kitchen*40} Sqft")]},
    ]
    if pooja_room > 0:
        stages.append({"icon": "🛕", "title": "10. Pooja Room", "items": [
            ("Marble / Tiles", f"{25*pooja_room} Sqft"), ("Wooden Door", f"{pooja_room} Nos"),
            ("Shelves", f"{3*pooja_room} Nos"), ("Lighting", f"{2*pooja_room} Nos")]})
    if parking > 0:
        stages.append({"icon": "🚗", "title": "11. Parking & Outside", "items": [
            ("Paver Blocks", f"{parking*180} Sqft"), ("Iron/Steel Gate", "1 Nos"),
            ("Compound Wall", f"{int((s**.5)*4*.5)} Sqft")]})
    return stages

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

# ── BIG HERO TITLE ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrapper">
    <div class="hero-badge">✦ AI Powered · ML Model</div>
    <div class="hero-title">{L['title']}</div>
    <div class="hero-sub">{L['subtitle']}</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Config summary chips ──────────────────────────────────────────────────
chips = [
    f"📐 {sqft} sqft", f"🏗️ {quality_key}", f"🛏️ {bedroom} Bed",
    f"🛋️ {hall} Hall", f"🍳 {kitchen} Kitchen", f"🏢 {floor} Floor",
    f"🚿 {bathroom} Bath", f"🚗 {parking} Park",
    "🌿 Garden" if garden else "🚫 No Garden",
    f"🪔 {pooja_room} Pooja" if pooja_room else "",
]
chips_html = " ".join(f'<span class="chip">{c}</span>' for c in chips if c)
st.markdown(f'<div class="chips-row">{chips_html}</div>', unsafe_allow_html=True)

# ── Location Map ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📍 Property Location</div>', unsafe_allow_html=True)
location_query = st.text_input("Enter your building location (e.g., Anna Nagar, Chennai):", "Chennai, Tamil Nadu")
if location_query:
    map_url = f"https://maps.google.com/maps?q={location_query.replace(' ', '%20')}&t=&z=13&ie=UTF8&iwloc=&output=embed"
    components.html(f'<iframe width="100%" height="280" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="{map_url}"></iframe>', height=280)

st.markdown("---")

# ── Predict Button ────────────────────────────────────────────────────────
predict_clicked = st.button(f"🔮 {L['predict_btn']}", width='stretch', type="primary")

if predict_clicked:
    inp = pd.DataFrame([[hall, bedroom, kitchen, sqft, floor, bathroom, garden, parking, pooja_room]], columns=FEATURE_COLS)
    base_pred = model.predict(inp)[0]
    floor_mult = 1.0 + ((floor - 1) * 0.1)
    pred = base_pred * Q["mult"] * floor_mult

    try:
        save_prediction(st.session_state.username, sqft, bedroom, Q['label'], pred)
    except Exception:
        pass

    # ── Price Result Card ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">✦ Estimated Construction Cost</div>
        <div class="price-value">₹{int(pred):,}</div>
        <div class="price-range">{Q['label']} &nbsp;·&nbsp; {sqft} sqft &nbsp;·&nbsp; {floor} Floor{'s' if floor > 1 else ''}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat pills ────────────────────────────────────────────────────────
    def material_breakdown(total_cost):
        return {
            "cement": total_cost * 0.12,
            "steel": total_cost * 0.18,
            "sand": total_cost * 0.10,
            "other_materials": total_cost * 0.25,
            "labor": total_cost * 0.35
        }

    breakdown = material_breakdown(pred)
    mat_total   = breakdown["cement"] + breakdown["steel"] + breakdown["sand"]
    inter_total = breakdown["other_materials"]
    labor_total = breakdown["labor"]
    per_sqft    = pred / sqft

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill"><div class="stat-pill-val">₹{int(mat_total/1e5):.0f}L</div><div class="stat-pill-lbl">Material Cost</div></div>
        <div class="stat-pill"><div class="stat-pill-val">₹{int(labor_total/1e5):.0f}L</div><div class="stat-pill-lbl">Labour Cost</div></div>
        <div class="stat-pill"><div class="stat-pill-val">₹{int(inter_total/1e5):.0f}L</div><div class="stat-pill-lbl">Other / Interior</div></div>
        <div class="stat-pill"><div class="stat-pill-val">₹{int(per_sqft):,}</div><div class="stat-pill-lbl">Per Sqft</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Cost Breakdown ────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">💰 Complete Cost Breakdown</div>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)

    mat_items = [
        ("Cement", breakdown["cement"]),
        ("Steel",  breakdown["steel"]),
        ("Sand",   breakdown["sand"]),
    ]
    mat_rows = "".join(f'<div class="mat-row"><span>🔹 <b>{n}</b></span><span class="qty-badge">₹{v:,.0f}</span></div>' for n, v in mat_items)
    cc1.markdown(f"""<div class="stage-card"><div class="stage-title">🧱 Material Cost &nbsp;<span class="qty-badge">₹{mat_total:,.0f}</span></div>
      <div class="mat-row" style="color:#888;font-size:.75rem;margin-bottom:.5rem;">40% of Total</div>{mat_rows}</div>""", unsafe_allow_html=True)

    labor_items = [
        ("Construction Labour", labor_total * 0.80),
        ("Finishing Labour",    labor_total * 0.20),
    ]
    lab_rows = "".join(f'<div class="mat-row"><span>🔹 <b>{n}</b></span><span class="qty-badge">₹{v:,.0f}</span></div>' for n, v in labor_items)
    cc2.markdown(f"""<div class="stage-card"><div class="stage-title">👷 Labour Cost &nbsp;<span class="qty-badge">₹{labor_total:,.0f}</span></div>
      <div class="mat-row" style="color:#888;font-size:.75rem;margin-bottom:.5rem;">35% of Total</div>{lab_rows}</div>""", unsafe_allow_html=True)

    inter_items = [
        ("Other Materials",   inter_total * 0.50),
        ("Interior & Polish", inter_total * 0.50),
    ]
    int_rows = "".join(f'<div class="mat-row"><span>🔹 <b>{n}</b></span><span class="qty-badge">₹{v:,.0f}</span></div>' for n, v in inter_items)
    cc3.markdown(f"""<div class="stage-card"><div class="stage-title">🛋️ Other / Interior &nbsp;<span class="qty-badge">₹{inter_total:,.0f}</span></div>
      <div class="mat-row" style="color:#888;font-size:.75rem;margin-bottom:.5rem;">25% of Total</div>{int_rows}</div>""", unsafe_allow_html=True)

    # ── Material Stages ───────────────────────────────────────────────────
    st.markdown(f'<div class="sec-hdr">{L["material_header"]}</div>', unsafe_allow_html=True)
    st.info(L["material_info"])

    est_stages = estimate_materials(sqft, hall, bedroom, kitchen, floor, bathroom, garden, parking, pooja_room, quality_key)
    for i in range(0, len(est_stages), 2):
        ca, cb = st.columns(2)
        stg1 = est_stages[i]
        rows1 = "".join(f'<div class="mat-row"><span>🔹 <b>{n}</b></span><span class="qty-badge">{q}</span></div>' for n, q in stg1["items"])
        ca.markdown(f'<div class="stage-card"><div class="stage-title">{stg1["icon"]} {stg1["title"]}</div>{rows1}</div>', unsafe_allow_html=True)
        if i + 1 < len(est_stages):
            stg2 = est_stages[i+1]
            rows2 = "".join(f'<div class="mat-row"><span>🔹 <b>{n}</b></span><span class="qty-badge">{q}</span></div>' for n, q in stg2["items"])
            cb.markdown(f'<div class="stage-card"><div class="stage-title">{stg2["icon"]} {stg2["title"]}</div>{rows2}</div>', unsafe_allow_html=True)

    # ── Grand Total Table ─────────────────────────────────────────────────
    st.markdown(f'<div class="sec-hdr">{L["grand_total"]}</div>', unsafe_allow_html=True)
    st.table(pd.DataFrame({
        L["material_col"]: [
            "🧱 Cement (Total)", "🏜️ Sand (Total)", "⚙️ Steel / TMT Bars",
            "🧱 Bricks / Blocks", "🟦 Tiles / Flooring", "🎨 Paint (2 coats)",
            "🧴 Wall Putty", "🧴 Wall Primer", "⚙️ Tile Adhesive",
            "🪵 Centering Sheets", "🚿 PVC/CPVC Pipes"
        ],
        L["qty_col"]: [
            f"{int(sqft*0.4)} Bags", f"{int(sqft*1.3)} CFT", f"{int(sqft*4)} Kg",
            f"{int(sqft*8)} Nos", f"{int(sqft*1.05)} Sqft", f"{int(sqft*0.03)} Litres",
            f"{int(sqft*0.15)} Kg", f"{int(sqft*0.025)} Litres", f"{int(sqft*0.02)} Bags",
            f"{int(sqft*1.2)} Sqft", f"{int(sqft*0.4)} Metres"
        ],
    }))

    st.success(L["success_msg"])

    # ── Download Report ───────────────────────────────────────────────────
    import io
    import openpyxl  # type: ignore
    from openpyxl.styles import Font, PatternFill, Alignment  # type: ignore

    def make_excel():
        wb = openpyxl.Workbook()
        hdr_font = Font(bold=True, color="FFFFFF")
        hdr_fill = PatternFill("solid", fgColor="302b63")
        gold_fill = PatternFill("solid", fgColor="f7971e")
        gold_font = Font(bold=True, color="FFFFFF", size=12)
        center = Alignment(horizontal="center")

        def style_header(ws, row, cols):
            for col in range(1, cols+1):
                cell = ws.cell(row=row, column=col)
                cell.font, cell.fill, cell.alignment = hdr_font, hdr_fill, center

        ws1 = wb.active
        ws1.title = "Summary"
        ws1.append(["Santhosh AI – House Price Report"])
        ws1.merge_cells("A1:C1")
        ws1["A1"].font, ws1["A1"].fill, ws1["A1"].alignment = gold_font, gold_fill, center
        ws1.append([])
        ws1.append(["House Details", "Value", ""])
        style_header(ws1, 3, 2)
        details = [("Total Area", f"{sqft} sqft"), ("Halls", hall), ("Bedrooms", bedroom),
                   ("Kitchens", kitchen), ("Floors", floor), ("Bathrooms", bathroom),
                   ("Parking", parking), ("Garden", "Yes" if garden else "No"), ("Pooja", pooja_room)]
        for n, v in details:
            ws1.append([n, v, ""])
        ws1.append([])
        ws1.append(["Cost Breakdown", "Amount (INR)", ""])
        style_header(ws1, 14, 2)
        ws1.append(["Total Predicted", f"₹{pred:,.0f}"])
        ws1.append(["Material Cost", f"₹{mat_total:,.0f}"])
        ws1.append(["Labor Cost", f"₹{labor_total:,.0f}"])
        ws1.append(["Interior Cost", f"₹{inter_total:,.0f}"])
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()

    st.download_button(
        label="📥 Download Full Report (Excel)", width='stretch',
        data=make_excel(),
        file_name="Santhosh_AI_House_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ── Data Analysis Dashboard ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">📊 Data Analysis Dashboard</div>', unsafe_allow_html=True)

try:
    raw_df = pd.read_csv('house_prediction.csv')
    t1, t2, t3, t4, t5 = st.tabs(["📈 Area vs Price", "📊 Distribution", "📦 Outliers", "🔥 Correlation", "📖 History"])

    with t1:
        st.write("### 📈 Area (Sqft) vs Price")
        st.line_chart(raw_df.sort_values(by="sqft").set_index("sqft")["price"])

    with t2:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
        sns.histplot(raw_df['price'], bins=20, kde=True, color="#f7971e", ax=ax_hist)
        ax_hist.set_facecolor("#111")
        fig_hist.patch.set_facecolor("#111")
        ax_hist.tick_params(colors='#aaa')
        ax_hist.set_title("Price Distribution", color="#ffd200", fontsize=12)
        st.pyplot(fig_hist)

    with t3:
        fig_box, ax_box = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=raw_df['price'], color="#ffd200", ax=ax_box)
        ax_box.set_facecolor("#111")
        fig_box.patch.set_facecolor("#111")
        ax_box.tick_params(colors='#aaa')
        ax_box.set_title("Price Outliers & Spread", color="#ffd200", fontsize=12)
        st.pyplot(fig_box)

    with t4:
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr = raw_df[FEATURE_COLS + ['price']].corr()
        sns.heatmap(corr, annot=True, cmap="YlOrBr", fmt=".2f", linewidths=.5, ax=ax_corr)
        ax_corr.set_facecolor("#111")
        fig_corr.patch.set_facecolor("#111")
        ax_corr.set_title("Feature Correlation Heatmap", color="#ffd200", fontsize=12)
        st.pyplot(fig_corr)

    with t5:
        st.write("### 📖 Your Prediction History")
        conn = sqlite3.connect('santhosh_ai.db')
        hist_df = pd.read_sql_query(
            'SELECT sqft as "Sq.Ft", bedrooms as "BHK", quality as "Quality", '
            'est_price as "Predicted Price (₹)", timestamp as "Date & Time" '
            'FROM history WHERE username=?', conn, params=(st.session_state.username,))
        conn.close()
        if not hist_df.empty:
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("No predictions yet. Use the Predict button to save your first estimate!")

except Exception as e:
    st.error(f"Error loading analytics: {e}")

# ── AI Chat Assistant ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="sec-hdr">🤖 {L.get("ai_chat", "Santhosh AI Assistant")}</div>', unsafe_allow_html=True)
st.info("💡 Enter your Gemini API key in the sidebar to enable real AI responses in Tamil.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": L.get("ai_welcome", "வணக்கம்! நான் Santhosh AI. வீட்டு கட்டுமானம் குறித்து எந்த கேள்வியும் கேளுங்கள்!")}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input(L.get("ai_placeholder", "Ask Santhosh AI…")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        if st.session_state.get("gemini_api_key"):
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=st.session_state.gemini_api_key)
                model_ai = genai.GenerativeModel('gemini-pro')
                convo_prompt = (
                    f"You are Santhosh AI, a friendly construction and real estate AI assistant. "
                    f"You MUST answer ONLY in Tamil. User asks: {prompt}"
                )
                response = model_ai.generate_content(convo_prompt, stream=True)
                for chunk in response:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"மன்னிக்கவும்! பிழை: {e}"
                placeholder.markdown(full_response)
        else:
            lower_prompt = prompt.lower()
            if any(k in lower_prompt for k in ["cement", "sand", "சிமெண்ட்"]):
                ai_response = "சிமெண்ட் மற்றும் மணலின் அளவு வீட்டின் sqft-ஐ பொறுத்தது. Material பட்டியலில் முழு விவரம் உள்ளது!"
            elif any(k in lower_prompt for k in ["price", "cost", "விலை"]):
                ai_response = "வீட்டின் விலை Sqft மற்றும் Construction Quality-ஐ பொறுத்து மாறுபடும். Sidebar-ல் மாற்றிப் பாருங்கள்!"
            elif any(k in lower_prompt for k in ["hello", "hi", "வணக்கம்"]):
                ai_response = "வணக்கம் நண்பரே! வீடு கட்டுவது குறித்து என்ன உதவி வேண்டும்?"
            else:
                ai_response = "Gemini API Key-ஐ Sidebar-ல் உள்ளிடுங்கள், நான் தமிழில் நேரடியாக உதவுவேன்!"

            for chunk in ai_response.split(" "):
                full_response += chunk + " "
                time.sleep(0.04)
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})