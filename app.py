import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import pickle
import os
from languages import LANGUAGES, LANG_LIST  # type: ignore

# ── Page config ───────────────────────────────
st.set_page_config(page_title="🏠 Santhosh AI", page_icon="🏠", layout="wide")

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
st.sidebar.markdown("## 🌐 Language / भाषा / மொழி")
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
    if pooja_room:
        stages.append({"icon":"🛕","title":"10. Pooja Room","items":[
            ("Marble / Tiles","25 Sqft"),("Wooden Door","1 Nos"),("Shelves","3 Nos"),("Lighting","2 Nos")]})
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

sqft = st.number_input(L["sqft"], min_value=300, max_value=10000, value=1200, step=50)

# Quality selector
quality_key = st.select_slider("🏗️ Construction Quality", options=["Low", "Medium", "High"], value="Medium")
Q = Q_LEVELS[quality_key]

c1, c2, c3 = st.columns(3)
with c1:
    hall       = st.selectbox(L["halls"],    [1,2,3,4,5], index=1)
    bedroom    = st.selectbox(L["bedrooms"], [1,2,3,4,5,6], index=1)
    kitchen    = st.selectbox(L["kitchens"], [1,2],        index=0)
with c2:
    floor      = st.selectbox(L["floors"],   [1,2,3,4,5],  index=0)
    bathroom   = st.selectbox(L["bathrooms"],[1,2,3,4,5,6],index=1)
    parking    = st.selectbox(L["parking"],  [0,1,2,3,4],  index=1)
with c3:
    garden     = st.radio(L["garden"],    [0,1], format_func=lambda x: L["yes"] if x else L["no"], horizontal=True)
    pooja_room = st.radio(L["pooja"],     [0,1], format_func=lambda x: L["yes"] if x else L["no"], horizontal=True)

chips = [f"📐 {sqft} sqft", f"🏗️ {Q['label']}", f"🛋️ {hall} H", f"🛏️ {bedroom} B", f"🍳 {kitchen} K", f"🏢 {floor} F",
         f"🚿 {bathroom} Ba", f"🚗 {parking} P",
         "🌿 " + L["yes"] if garden else "🚫 " + L["no"],
         "🪔 " + L["yes"] if pooja_room else "🚫 " + L["no"]]
st.markdown(" ".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)
st.divider()

# ── Predict ───────────────────────────────────
if st.button(L["predict_btn"], use_container_width=True, type="primary"):
    inp = pd.DataFrame([[hall, bedroom, kitchen, sqft, floor, bathroom, garden, parking, pooja_room]], columns=FEATURE_COLS)
    try:
        base_pred = model.predict(inp)[0]
    except Exception:
        # Fallback if model doesn't match new features yet
        model = load_model()
        base_pred = model.predict(inp)[0]
    
    # Apply quality multiplier to price
    pred = base_pred * Q["mult"]
    low, high = pred*.92, pred*1.08

    # ── Price card ────────────────────────────
    st.markdown(f"""
    <div class="price-card">
      <div class="price-label">{L["est_price"]}</div>
      <div class="price-value">₹{pred:,.0f}</div>
      <div class="price-range">{L["range_label"]}: ₹{low:,.0f} – ₹{high:,.0f}</div>
    </div>""", unsafe_allow_html=True)

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
        ("Pooja Room",               inter_total * 0.05 if pooja_room else 0),
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
                   ("Garden", "Yes" if garden else "No"), ("Pooja", "Yes" if pooja_room else "No")]
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

# ── Sidebar info ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏠 Santhosh AI")
st.sidebar.info("Decision Tree model · BHK Features Integrated")
st.sidebar.markdown("### 🚀 Deploy Free\n1. Push to [GitHub](https://github.com)\n2. [share.streamlit.io](https://share.streamlit.io)")