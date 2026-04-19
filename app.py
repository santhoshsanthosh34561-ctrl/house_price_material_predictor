import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import pickle
import os
import time
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import streamlit.components.v1 as components # type: ignore
from languages import LANGUAGES, LANG_LIST  # type: ignore
import sqlite3
import hashlib
from datetime import datetime

# ── Database Initialization ───────────────────
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
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', (username, make_hash(password), email))
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

# ── Page config ───────────────────────────────
st.set_page_config(page_title="🏠 Santhosh AI", page_icon="🏠", layout="wide")

# ── Auth Flow ─────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.markdown("<br><br><h1 style='text-align: center'>🏠 Santhosh AI<br><span style='font-size: 1.2rem;font-weight:400;color:#aaa'>Please login to access the application.</span></h1>", unsafe_allow_html=True)
    c1, mid, c2 = st.columns([1,2,1])
    with mid:
        st.markdown('<div class="price-card" style="padding:2rem">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔒 Login", "📝 Sign Up"])
        with tab1:
            u_login = st.text_input("Username or Gmail", key="l_u")
            p_login = st.text_input("Password", type="password", key="l_p")
            if st.button("Login", use_container_width=True, type="primary"):
                success, real_username = verify_login(u_login, p_login)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = real_username
                    st.success("Successfully logged in!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid Username/Gmail or Password")
        with tab2:
            e_sign = st.text_input("Google Email", key="s_e", placeholder="example@gmail.com")
            u_sign = st.text_input("New Username", key="s_u")
            p_sign = st.text_input("New Password", type="password", key="s_p")
            if st.button("Create Account", use_container_width=True):
                if u_sign and p_sign and e_sign:
                    if "@gmail.com" not in e_sign.lower():
                        st.error("Please provide a valid @gmail.com address!")
                    else:
                        if add_user(u_sign, p_sign, e_sign):
                            st.success("Account created successfully! You can now login.")
                        else:
                            st.error("Username already exists!")
                else:
                    st.warning("Please fill all fields.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ── CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.hero-title{font-size:2.5rem;font-weight:700;background:linear-gradient(90deg,#f7971e,#ffd200);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:.2rem;}
.hero-sub{text-align:center;color:#aaa;font-size:1rem;margin-bottom:1.5rem;}
.price-card{background:linear-gradient(135deg,#f7971e22,#ffd20022);border:1.5px solid #ffd200aa;
  border-radius:16px;padding:1.8rem;text-align:center;margin:1.2rem 0;}
.price-label{color:#ffd200;font-size:.9rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;}
.price-value{font-size:2.8rem;font-weight:700;color:#fff;margin:.3rem 0;}
.price-range{color:#aaa;font-size:.85rem;}
.sec-hdr{font-size:1.1rem;font-weight:700;color:#ffd200;border-left:4px solid #f7971e;
  padding-left:.8rem;margin:1.6rem 0 .8rem;}
.stage-card{background:#ffffff0d;border:1px solid #ffffff18;border-radius:12px;
  padding:1rem 1.2rem;margin-bottom:.8rem;}
.stage-title{font-size:.95rem;font-weight:700;color:#ffd200;margin-bottom:.5rem;}
.mat-row{color:#ddd;font-size:.85rem;padding:2px 0;}
.qty-badge{background:#f7971e33;border:1px solid #f7971e66;border-radius:8px;
  padding:1px 8px;color:#ffd200;font-weight:600;font-size:.82rem;}
.chip{display:inline-block;background:#ffffff18;border:1px solid #ffffff30;
  border-radius:20px;padding:.22rem .75rem;margin:.2rem;font-size:.8rem;color:#ddd;}
</style>
""", unsafe_allow_html=True)

# ── Language selector (sidebar) ───────────────
st.sidebar.markdown(f"**👤 Welcome, {st.session_state.username}!**")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.markdown("## 🌐 Language")
sel_lang = st.sidebar.selectbox("Choose Language", LANG_LIST, index=0)
L = LANGUAGES[sel_lang]

# ── Model ─────────────────────────────────────
FEATURE_COLS = ['hall', 'bedroom', 'kitchen', 'sqft', 'floor', 'bathroom', 'garden', 'parking', 'pooja_room']

@st.cache_resource
def load_model():
    if os.path.exists('house_model.pkl'):
        try:
            with open('house_model.pkl','rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    # Retrain on updated data
    df = pd.read_csv('house_prediction.csv')
    model = LinearRegression()
    model.fit(df[FEATURE_COLS], df['price'])
    with open('house_model.pkl','wb') as f:
        pickle.dump(model, f)
    return model

model = load_model()

# ── Material estimation engine ────────────────
def estimate_materials(sqft, hall, bedroom, kitchen, floor, bathroom, garden, parking, pooja_room, quality_key):
    s = sqft
    # Quality multiplier for quantities removed (user provided specific 1 sqft rates)
    qm = 1.0 

    stages = [
        {"icon":"🧱","title":"1. Foundation (Basement)","items":[
            ("Cement", f"{int(s*0.10)} Bags"), ("Sand", f"{int(s*0.35)} CFT"),
            ("Gravel (Jalli)", f"{int(s*.40)} CFT"), ("Steel Rods (TMT)", f"{int(s*1.0)} Kg"),
            ("Bricks / Stones", f"{int(s*4)} Nos"), ("Water", f"{int(s*10)} Litres")]},
        {"icon":"🧱","title":"2. Structure (Column, Beam, Slab)","items":[
            ("Cement", f"{int(s*0.15)} Bags"), ("Sand", f"{int(s*0.55)} CFT"),
            ("Aggregate (Jalli)", f"{int(s*.60)} CFT"), ("Steel (TMT bars)", f"{int(s*3.0)} Kg"),
            ("Centering Sheets", f"{int(s*1.2)} Sqft")]},
        {"icon":"🧱","title":"3. Walls","items":[
            ("Bricks / AAC Blocks", f"{int(s*8)} Nos"),
            ("Cement", f"{int(s*0.08)} Bags"), ("Sand", f"{int(s*0.25)} CFT")]},
        {"icon":"🚪","title":"4. Doors & Windows","items":[
            ("Main Door", "1 Nos"),
            ("Bedroom Doors", f"{bedroom} Nos"),
            ("Bathroom Doors", f"{bathroom} Nos"),
            ("Window Frames", f"{(bedroom*2)+(bathroom)} Nos"),
            ("Glass Panels", f"{int(((bedroom*2)+bathroom)*6)} Sqft"),
            ("Hinges", f"{int((1+bedroom+bathroom)*3+((bedroom*2)+bathroom)*2)} Nos"),
            ("Locks", f"{1+bedroom} Nos")]},
        {"icon":"⚡","title":"5. Electrical","items":[
            ("Wires", f"{int(s*2.5)} Metres"), ("Switches & Sockets", f"{int(s/40 + bedroom*5)} Nos"),
            ("Switch Boards", f"{int(s/80 + bedroom)} Nos"), ("MCB Box", f"{floor} Nos"),
            ("Lights", f"{int(s/50 + bedroom*2)} Nos"), ("Fans", f"{hall+bedroom} Nos")]},
        {"icon":"🚿","title":"6. Plumbing","items":[
            ("PVC/CPVC Pipes", f"{int(s*0.4)} Metres"), ("Taps", f"{bathroom*3 + kitchen*2} Nos"),
            ("Shower Sets", f"{bathroom} Nos"), ("Toilet Fittings", f"{bathroom} Sets"),
            ("Water Tank", f"{bathroom*500} Litres")]},
        {"icon":"🧴","title":"7. Plastering & Finishing","items":[
            ("Cement", f"{int(s*0.04)} Bags"), ("Sand", f"{int(s*0.15)} CFT"),
            ("Wall Putty", f"{int(s*0.15)} Kg"), ("Primer", f"{int(s*0.025)} Litres"),
            ("Paint (2 coats)", f"{int(s*0.03)} Litres")]},
        {"icon":"🧱","title":"8. Flooring","items":[
            ("Tiles / Marble / Granite", f"{int(s*1.05)} Sqft"),
            ("Tile Adhesive", f"{int(s*0.02)} Bags"), ("Cement (base)", f"{int(s*0.03)} Bags")]},
        {"icon":"🍳","title":"9. Kitchen","items":[
            ("Granite Slab", f"{kitchen*22} Sqft"), ("Kitchen Sink", f"{kitchen} Nos"),
            ("Cabinets", f"{kitchen*3} Nos"), ("Wall Tiles", f"{kitchen*40} Sqft")]},
    ]
    if pooja_room > 0:
        stages.append({"icon":"🛕","title":"10. Pooja Room","items":[
            ("Marble / Tiles", f"{25*pooja_room} Sqft"),("Wooden Door", f"{pooja_room} Nos"),("Shelves", f"{3*pooja_room} Nos"),("Lighting", f"{2*pooja_room} Nos")]})
    if parking > 0:
        stages.append({"icon":"🚗","title":"11. Parking & Outside","items":[
            ("Paver Blocks", f"{parking*180} Sqft"),("Iron/Steel Gate","1 Nos"),
            ("Compound Wall", f"{int((s**.5)*4*.5)} Sqft")]})
    return stages

# ── Header ────────────────────────────────────
st.markdown(f'<div class="hero-title">{L["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">{L["subtitle"]}</div>', unsafe_allow_html=True)



# ── Material Quality labels ───────────────────
Q_LEVELS = {
    "Low":    {"mult": 0.78260869565, "label": "📉 Budget (Low Cost: ₹1800/sqft)"},
    "Medium": {"mult": 1.0,           "label": "🏢 Standard (Medium: ₹2300/sqft)"},
    "High":   {"mult": 1.21739130435, "label": "✨ Premium (High End: ₹2800/sqft)"}
}

# ── Inputs ────────────────────────────────────
st.markdown(f'<div class="sec-hdr">{L["house_details"]}</div>', unsafe_allow_html=True)
sqft = st.number_input(f"📐 {L['sqft']}", min_value=300, max_value=10000, value=1200, step=50)

quality_key = st.select_slider("🏗️ Construction Quality", options=["Low", "Medium", "High"], value="Medium")
Q = Q_LEVELS[quality_key]

c1, c2, c3 = st.columns(3)
with c1:
    bedroom    = st.selectbox(f"🛏️ {L['bedrooms']}", [1,2,3,4,5,6], index=1)
    hall       = st.selectbox(f"🛋️ {L['halls']}",    [1,2,3,4,5], index=1)
    kitchen    = st.selectbox(f"🍳 {L['kitchens']}", [1,2],        index=0)
with c2:
    bathroom   = st.selectbox(f"🚿 {L['bathrooms']}",[1,2,3,4,5,6],index=1)
    floor      = st.selectbox(f"🏢 {L['floors']}",   [1,2,3,4,5],  index=0)
    parking    = st.selectbox(f"🚗 {L['parking']}",  [0,1,2,3,4],  index=1)
with c3:
    pooja_room = st.selectbox(f"🪔 {L['pooja']}",     [0,1,2,3],  index=0)
    garden     = st.radio(f"🌿 {L['garden']}",    [0,1], format_func=lambda x: L["yes"] if x else L["no"], horizontal=True)

# ── Property Configuration Summary ────────────
chips = [f"📐 {sqft} sqft", f"🏗️ {Q['label']}", f"🛋️ {hall} H", f"🛏️ {bedroom} B", f"🍳 {kitchen} K", f"🏢 {floor} F",
         f"🚿 {bathroom} Ba", f"🚗 {parking} P",
         "🌿 " + L["yes"] if garden else "🚫 " + L["no"],
         f"🪔 {pooja_room} Pj"]
st.markdown(" ".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)
st.divider()

# ── Property Location Map ──────────────────────
st.markdown('<div class="sec-hdr">📍 Property Location (Google Maps)</div>', unsafe_allow_html=True)
location_query = st.text_input("Enter your building location (e.g., Anna Nagar, Chennai):", "Chennai, Tamil Nadu")

if location_query:
    map_url = f"https://maps.google.com/maps?q={location_query.replace(' ', '%20')}&t=&z=13&ie=UTF8&iwloc=&output=embed"
    components.html(f'<iframe width="100%" height="300" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="{map_url}"></iframe>', height=300)

st.divider()

# ── Predict ───────────────────────────────────
if st.button(L["predict_btn"], use_container_width=True, type="primary"):
    inp = pd.DataFrame([[hall, bedroom, kitchen, sqft, floor, bathroom, garden, parking, pooja_room]], columns=FEATURE_COLS)
    try:
        base_pred = model.predict(inp)[0]
    except Exception:
        # Fallback if model doesn't match new features yet
        base_pred = model.predict(inp)[0]
        
    # Apply quality & floor multiplier to price
    # Based on the user's specified formula: price = base * (1 + (floors - 1) * 0.1)
    floor_mult = 1.0 + ((floor - 1) * 0.1)
    pred = base_pred * Q["mult"] * floor_mult
    
    try:
        save_prediction(st.session_state.username, sqft, bedroom, Q['label'], pred)
    except Exception:
        pass
        
    # ── Price card ────────────────────────────
    st.markdown(f'''
    <div class="price-card">
        <h2 style="margin:0;color:#f7971e;">{L["est_price"]}</h2>
        <h1 style="margin:10px 0;font-size:3.2rem;color:#ffd200;">₹{int(pred):,}</h1>
        <p style="margin:0;color:#aaa;">{L.get("based_on_current", "Based on current construction rates")}</p>
    </div>
    ''', unsafe_allow_html=True)

    # ── Cost breakdown ────────────────────────
    mat_total   = pred * 0.55
    labor_total = pred * 0.25
    inter_total = pred * 0.20

    st.markdown('<div class="sec-hdr">💰 Complete Cost Breakdown</div>', unsafe_allow_html=True)

    cc1, cc2, cc3 = st.columns(3)

    mat_items = [
        ("Foundation Materials",    mat_total * 0.15),
        ("Structure (Column/Slab)", mat_total * 0.25),
        ("Bricks / Blocks",         mat_total * 0.12),
        ("Doors & Windows",         mat_total * 0.10),
        ("Electrical Materials",    mat_total * 0.08),
        ("Plumbing Materials",      mat_total * 0.08),
        ("Plastering Materials",    mat_total * 0.07),
        ("Flooring Materials",      mat_total * 0.10),
        ("Miscellaneous",           mat_total * 0.05),
    ]
    mat_rows = "".join(f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">₹{v:,.0f}</span></div>' for n, v in mat_items)
    cc1.markdown(f"""<div class="stage-card"><div class="stage-title">🧱 Material Cost &nbsp;<span class="qty-badge">₹{mat_total:,.0f}</span></div>
      <div class="mat-row" style="color:#aaa;font-size:.78rem;margin-bottom:.5rem;">55% of Total Price</div>{mat_rows}</div>""", unsafe_allow_html=True)

    labor_items = [
        ("Foundation Labour",   labor_total * 0.18),
        ("Structure Labour",    labor_total * 0.28),
        ("Masonry (Walls)",     labor_total * 0.20),
        ("Plastering Labour",   labor_total * 0.14),
        ("Electrical Labour",   labor_total * 0.10),
        ("Plumbing Labour",     labor_total * 0.10),
    ]
    lab_rows = "".join(f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">₹{v:,.0f}</span></div>' for n, v in labor_items)
    cc2.markdown(f"""<div class="stage-card"><div class="stage-title">👷 Labor Cost &nbsp;<span class="qty-badge">₹{labor_total:,.0f}</span></div>
      <div class="mat-row" style="color:#aaa;font-size:.78rem;margin-bottom:.5rem;">25% of Total Price</div>{lab_rows}</div>""", unsafe_allow_html=True)

    inter_items = [
        ("Flooring (Tiles/Marble)",  inter_total * 0.30),
        ("Paint & Putty",            inter_total * 0.15),
        ("Kitchen Fitting",          inter_total * 0.25),
        ("False Ceiling",            inter_total * 0.10),
        ("Pooja Room",               inter_total * 0.05 * pooja_room),
        ("Furniture & Fixtures",     inter_total * 0.15),
    ]
    int_rows = "".join(f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">₹{v:,.0f}</span></div>' for n, v in inter_items)
    cc3.markdown(f"""<div class="stage-card"><div class="stage-title">🛋️ Interior Cost &nbsp;<span class="qty-badge">₹{inter_total:,.0f}</span></div>
      <div class="mat-row" style="color:#aaa;font-size:.78rem;margin-bottom:.5rem;">20% of Total Price</div>{int_rows}</div>""", unsafe_allow_html=True)

    # ── Material stages ───────────────────────
    st.markdown(f'<div class="sec-hdr">{L["material_header"]}</div>', unsafe_allow_html=True)
    st.info(L["material_info"])

    est_stages = estimate_materials(sqft, hall, bedroom, kitchen, floor, bathroom, garden, parking, pooja_room, quality_key)

    for i in range(0, len(est_stages), 2):
        ca, cb = st.columns(2)
        # First column
        stg1 = est_stages[i]
        rows1 = "".join(f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">{q}</span></div>' for n, q in stg1["items"])
        ca.markdown(f'<div class="stage-card"><div class="stage-title">{stg1["icon"]} {stg1["title"]}</div>{rows1}</div>', unsafe_allow_html=True)
        # Second column
        if i + 1 < len(est_stages):
            stg2 = est_stages[i+1]
            rows2 = "".join(f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">{q}</span></div>' for n, q in stg2["items"])
            cb.markdown(f'<div class="stage-card"><div class="stage-title">{stg2["icon"]} {stg2["title"]}</div>{rows2}</div>', unsafe_allow_html=True)

    # ── Grand total ───────────────────────────
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

    # ── Download Report ───────────────────────
    import io
    import openpyxl  # type: ignore
    from openpyxl.styles import Font, PatternFill, Alignment  # type: ignore

    def make_excel():
        wb = openpyxl.Workbook()
        hdr_font, hdr_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="302b63")
        gold_fill, gold_font = PatternFill("solid", fgColor="f7971e"), Font(bold=True, color="FFFFFF", size=12)
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
        details = [("Total Area", f"{sqft} sqft"), ("Halls", hall), ("Bedrooms", bedroom), ("Kitchens", kitchen),
                   ("Floors", floor), ("Bathrooms", bathroom), ("Parking", parking),
                   ("Garden", "Yes" if garden else "No"), ("Pooja", pooja_room)]
        for n, v in details: ws1.append([n, v, ""])
        
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

    st.download_button(label="📥 Download Full Report (Excel)", data=make_excel(),
                       file_name="Santhosh_AI_House_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

# ── Data Analysis Dashboard ─────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">📊 Data Analysis Dashboard</div>', unsafe_allow_html=True)
st.write("Explore the underlying dataset through visual analytics.")

try:
    raw_df = pd.read_csv('house_prediction.csv')
    t1, t2, t3, t4, t5 = st.tabs(["📈 Area vs Price", "📊 Price Distribution", "📦 Price Outliers", "🔥 Correlation Heatmap", "📖 History"])
    
    with t1:
        st.write("### 📈 Trend: Area (Sqft) vs Price")
        st.line_chart(raw_df.sort_values(by="sqft").set_index("sqft")["price"])
        
    with t2:
        st.write("### 📊 Histogram: Price Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        sns.histplot(raw_df['price'], bins=20, kde=True, color="#f7971e", ax=ax_hist)
        ax_hist.set_title("Distribution of House Prices", fontsize=12)
        ax_hist.set_xlabel("Price", fontsize=10)
        ax_hist.set_ylabel("Frequency", fontsize=10)
        st.pyplot(fig_hist)
        
    with t3:
        st.write("### 📦 Box Plot: Price Outliers")
        fig_box, ax_box = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=raw_df['price'], color="#ffd200", ax=ax_box)
        ax_box.set_title("House Price Outliers & Spread", fontsize=12)
        st.pyplot(fig_box)
        
    with t4:
        st.write("### 🔥 Heatmap: Feature Correlations")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr = raw_df[FEATURE_COLS + ['price']].corr()
        sns.heatmap(corr, annot=True, cmap="YlOrBr", fmt=".2f", linewidths=.5, ax=ax_corr)
        ax_corr.set_title("Feature Correlation Heatmap", fontsize=12)
        st.pyplot(fig_corr)
        
    with t5:
        st.write("### 📖 Your Prediction History")
        conn = sqlite3.connect('santhosh_ai.db')
        hist_df = pd.read_sql_query('SELECT sqft as "Sq.Ft", bedrooms as "BHK", quality as "Quality", est_price as "Predicted Price (₹)", timestamp as "Date & Time" FROM history WHERE username=?', conn, params=(st.session_state.username,))
        conn.close()
        if not hist_df.empty:
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("You haven't made any predictions yet. Predict a house price to save it here!")
            
except Exception as e:
    st.error(f"Error loading graphs: {e}")


# ── AI Chat Assistant (Real Gemini Integration) ─────────────────────────
st.markdown("---")
st.markdown(f'<div class="sec-hdr">🤖 {L.get("ai_chat", "Santhosh AI Assistant")}</div>', unsafe_allow_html=True)

st.info("Powered by Real AI. Please enter your Gemini API key in the sidebar to chat, or just try the preset questions without a key.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": L.get("ai_welcome", "வணக்கம்! நான் உங்கள் சந்தோஷ் AI Assistant. உங்கள் வீட்டின் கட்டமைப்பு அல்லது செலவுகள் குறித்து எந்த கேள்வியும் கேட்கலாம்!")}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input(L.get("ai_placeholder", "Ask Santhosh AI...")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Check if API key is provided
        if st.session_state.get("gemini_api_key"):
            try:
                import google.generativeai as genai # type: ignore
                genai.configure(api_key=st.session_state.gemini_api_key)
                # Initialize model
                model_ai = genai.GenerativeModel('gemini-pro')
                # Inject a brief system prompt context into the conversation implicitly 
                convo_prompt = f"You are Santhosh AI, a friendly construction and real estate AI assistant helping users estimate their house building costs. You MUST speak and explain everything ONLY in the Tamil language natively. Answer clearly and concisely in Tamil. User asks: {prompt}"
                response = model_ai.generate_content(convo_prompt, stream=True)
                for chunk in response:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"மன்னிக்கவும்! AI ஐ இணைப்பதில் பிழை: {e}"
                message_placeholder.markdown(full_response)
        else:
            # Fallback mock responses
            lower_prompt = prompt.lower()
            if "cement" in lower_prompt or "sand" in lower_prompt or "சிமெண்ட்" in lower_prompt:
                ai_response = "சிமெண்ட் மற்றும் மணலின் அளவு உங்கள் வீட்டின் அளவைப் பொறுத்தது. மேலே உள்ள Material பட்டியலில் முழு விவரங்களையும் நீங்கள் பார்க்கலாம்!"
            elif "price" in lower_prompt or "cost" in lower_prompt or "விலை" in lower_prompt:
                ai_response = "வீட்டின் விலை சதுர அடி (Sqft) மற்றும் நீங்கள் தேர்வு செய்யும் Construction Quality அளவைப் பொறுத்து மாறுபடும். அதனை மாற்றிப் பாருங்கள்!"
            elif "hello" in lower_prompt or "hi" in lower_prompt or "வணக்கம்" in lower_prompt:
                ai_response = "வணக்கம் நண்பரே! வீடு கட்டுவது குறித்து உங்களுக்கு என்ன உதவி வேண்டும்?"
            else:
                ai_response = "தற்போது நான் 'Mock Mode'-ல் உள்ளேன். செயற்கை நுண்ணறிவில் என்னுடன் நேரடியாகத் தமிழில் உரையாட, உங்கள் Gemini API Key-ஐ Sidebar-ல் அப்டேட் செய்யுங்கள்!"
                
            for chunk in ai_response.split(" "):
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})

# ── Sidebar info ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Update Dataset")
uploaded_file = st.sidebar.file_uploader("Upload New Dataset (CSV)", type=["csv"])
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

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Setup AI Chat")
api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password", key="gemini_api_key", help="Get your free API key at g.co/aistudio")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏠 Santhosh AI")
st.sidebar.info("Linear Regression Model · BHK Features Integrated")
st.sidebar.markdown("### 🚀 Deploy Free\n1. Push to [GitHub](https://github.com)\n2. [share.streamlit.io](https://share.streamlit.io)")