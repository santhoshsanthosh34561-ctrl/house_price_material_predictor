import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
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
FEATURE_COLS = ['hall','kitchen','sqft','floor','bathroom','garden','parking','pooja_room']

@st.cache_resource
def load_model():
    if os.path.exists('house_model.pkl'):
        with open('house_model.pkl','rb') as f:
            return pickle.load(f)
    df = pd.read_csv('house_prediction.csv')
    model = DecisionTreeRegressor(random_state=42)
    model.fit(df[FEATURE_COLS], df['price'])
    with open('house_model.pkl','wb') as f:
        pickle.dump(model, f)
    return model

model = load_model()

# ── Material estimation engine ────────────────
def estimate_materials(sqft, hall, kitchen, floor, bathroom, garden, parking, pooja_room, quality_key):
    s = sqft
    # Quality multiplier for quantities (High needs more steel/cement for strength)
    qm = 1.15 if quality_key == "High" else (0.90 if quality_key == "Low" else 1.0)

    stages = [
        {"icon":"🧱","title":"1. Foundation (Basement)","items":[
            ("Cement", f"{int(s*.12*qm)} Bags"), ("Sand", f"{int(s*.30*qm)} CFT"),
            ("Gravel (Jalli)", f"{int(s*.40*qm)} CFT"), ("Steel Rods (TMT)", f"{int(s*.80*qm)} Kg"),
            ("Bricks / Stones", f"{int(s*4)} Nos"), ("Water", f"{int(s*10)} Litres")]},
        {"icon":"🧱","title":"2. Structure (Column, Beam, Slab)","items":[
            ("Cement", f"{int(s*.20*qm)} Bags"), ("Sand", f"{int(s*.50*qm)} CFT"),
            ("Aggregate (Jalli)", f"{int(s*.60*qm)} CFT"), ("Steel (TMT bars)", f"{int(s*2.5*qm)} Kg"),
            ("Centering Sheets", f"{int(s*1.2)} Sqft")]},
        {"icon":"🧱","title":"3. Walls","items":[
            ("Bricks / AAC Blocks", f"{int(s*9*floor)} Nos"),
            ("Cement", f"{int(s*.10*qm)} Bags"), ("Sand", f"{int(s*.25*qm)} CFT")]},
        {"icon":"🚪","title":"4. Doors & Windows","items":[
            ("Wooden/Steel Doors", f"{hall+(1*floor)} Nos"),
            ("Window Frames", f"{(hall*2)+(bathroom)} Nos"),
            ("Glass Panels", f"{int(((hall*2)+bathroom)*6)} Sqft"),
            ("Hinges", f"{int((hall+(1*floor))*3+((hall*2)+bathroom)*2)} Nos"),
            ("Locks", f"{hall+(1*floor)} Nos")]},
        {"icon":"⚡","title":"5. Electrical","items":[
            ("Wires", f"{int(s*2.5)} Metres"), ("Switches & Sockets", f"{int(s/40)} Nos"),
            ("Switch Boards", f"{int(s/80)} Nos"), ("MCB Box", f"{floor} Nos"),
            ("Lights", f"{int(s/50)} Nos"), ("Fans", f"{hall+bathroom} Nos")]},
        {"icon":"🚿","title":"6. Plumbing","items":[
            ("PVC/CPVC Pipes", f"{int(s*1.5)} Metres"), ("Taps", f"{bathroom*3} Nos"),
            ("Shower Sets", f"{bathroom} Nos"), ("Toilet Fittings", f"{bathroom} Sets"),
            ("Water Tank", f"{bathroom*500} Litres")]},
        {"icon":"🧴","title":"7. Plastering & Finishing","items":[
            ("Cement", f"{int(s*.05)} Bags"), ("Sand", f"{int(s*.15)} CFT"),
            ("Wall Putty", f"{int(s*.05)} Kg"), ("Primer", f"{int(s*.08)} Litres"),
            ("Paint (2 coats)", f"{int(s*.12)} Litres")]},
        {"icon":"🧱","title":"8. Flooring","items":[
            ("Tiles / Marble / Granite", f"{int(s*1.1)} Sqft"),
            ("Tile Adhesive", f"{int(s*.03)} Bags"), ("Cement (base)", f"{int(s*.03)} Bags")]},
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

# ── Material Quality labels (local) ──────────
Q_LEVELS = {
    "Low":    {"mult": 0.85, "label": "📉 Budget (Low Cost)"},
    "Medium": {"mult": 1.00, "label": "🏢 Standard (Medium)"},
    "High":   {"mult": 1.35, "label": "✨ Premium (High End)"}
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
    kitchen    = st.selectbox(L["kitchens"], [1,2],        index=0)
    floor      = st.selectbox(L["floors"],   [1,2,3],      index=0)
with c2:
    bathroom   = st.selectbox(L["bathrooms"],[1,2,3,4],    index=1)
    parking    = st.selectbox(L["parking"],  [0,1,2,3],    index=1)
with c3:
    garden     = st.radio(L["garden"],    [0,1], format_func=lambda x: L["yes"] if x else L["no"], horizontal=True)
    pooja_room = st.radio(L["pooja"],     [0,1], format_func=lambda x: L["yes"] if x else L["no"], horizontal=True)

chips = [f"📐 {sqft} sqft", f"🏗️ {Q['label']}", f"🛋️ {hall}", f"🍳 {kitchen}", f"🏢 {floor}",
         f"🚿 {bathroom}", f"🚗 {parking}",
         "🌿 " + L["yes"] if garden else "🚫 " + L["no"],
         "🪔 " + L["yes"] if pooja_room else "🚫 " + L["no"]]
st.markdown(" ".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)
st.divider()

# ── Predict ───────────────────────────────────
if st.button(L["predict_btn"], use_container_width=True, type="primary"):
    inp = pd.DataFrame([[hall,kitchen,sqft,floor,bathroom,garden,parking,pooja_room]], columns=FEATURE_COLS)
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

    # ── Cost breakdown (Material / Labor / Interior) ──
    mat_total   = pred * 0.55   # 55%
    labor_total = pred * 0.25   # 25%
    inter_total = pred * 0.20   # 20%

    st.markdown('<div class="sec-hdr">💰 Complete Cost Breakdown</div>', unsafe_allow_html=True)

    cc1, cc2, cc3 = st.columns(3)

    # Material cost items
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
    mat_rows = "".join(
        f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">₹{v:,.0f}</span></div>'
        for n, v in mat_items
    )
    cc1.markdown(f"""
    <div class="stage-card">
      <div class="stage-title">🧱 Material Cost &nbsp;<span class="qty-badge">₹{mat_total:,.0f}</span></div>
      <div class="mat-row" style="color:#aaa;font-size:.78rem;margin-bottom:.5rem;">55% of Total Price</div>
      {mat_rows}
    </div>""", unsafe_allow_html=True)

    # Labor cost items
    labor_items = [
        ("Foundation Labour",   labor_total * 0.18),
        ("Structure Labour",    labor_total * 0.28),
        ("Masonry (Walls)",     labor_total * 0.20),
        ("Plastering Labour",   labor_total * 0.14),
        ("Electrical Labour",   labor_total * 0.10),
        ("Plumbing Labour",     labor_total * 0.10),
    ]
    lab_rows = "".join(
        f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">₹{v:,.0f}</span></div>'
        for n, v in labor_items
    )
    cc2.markdown(f"""
    <div class="stage-card">
      <div class="stage-title">👷 Labor Cost &nbsp;<span class="qty-badge">₹{labor_total:,.0f}</span></div>
      <div class="mat-row" style="color:#aaa;font-size:.78rem;margin-bottom:.5rem;">25% of Total Price</div>
      {lab_rows}
    </div>""", unsafe_allow_html=True)

    # Interior cost items
    inter_items = [
        ("Flooring (Tiles/Marble)",  inter_total * 0.30),
        ("Paint & Putty",            inter_total * 0.15),
        ("Kitchen Fitting",          inter_total * 0.25),
        ("False Ceiling",            inter_total * 0.10),
        ("Pooja Room",               inter_total * 0.05 if pooja_room else 0),
        ("Furniture & Fixtures",     inter_total * 0.15),
    ]
    int_rows = "".join(
        f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">₹{v:,.0f}</span></div>'
        for n, v in inter_items
    )
    cc3.markdown(f"""
    <div class="stage-card">
      <div class="stage-title">🛋️ Interior Cost &nbsp;<span class="qty-badge">₹{inter_total:,.0f}</span></div>
      <div class="mat-row" style="color:#aaa;font-size:.78rem;margin-bottom:.5rem;">20% of Total Price</div>
      {int_rows}
    </div>""", unsafe_allow_html=True)

    # ── Summary bar ───────────────────────────
    st.markdown(f"""
    <div style="background:#ffffff0d;border:1px solid #ffffff18;border-radius:12px;padding:1rem 1.5rem;margin-top:.5rem;">
      <b style="color:#ffd200;">💰 Total Cost Summary</b><br><br>
      <span style="color:#ddd;">🧱 Material Cost &nbsp;&nbsp;</span>
      <span style="color:#f7971e;font-weight:700;">₹{mat_total:,.0f}</span>
      <span style="color:#555;"> &nbsp;|&nbsp; </span>
      <span style="color:#ddd;">👷 Labor Cost &nbsp;&nbsp;</span>
      <span style="color:#f7971e;font-weight:700;">₹{labor_total:,.0f}</span>
      <span style="color:#555;"> &nbsp;|&nbsp; </span>
      <span style="color:#ddd;">🛋️ Interior Cost &nbsp;&nbsp;</span>
      <span style="color:#f7971e;font-weight:700;">₹{inter_total:,.0f}</span>
      <span style="color:#555;"> &nbsp;= &nbsp;</span>
      <span style="color:#ffd200;font-weight:700;font-size:1.1rem;">₹{pred:,.0f} Total</span>
    </div>""", unsafe_allow_html=True)

    # ── Material stages ───────────────────────
    st.markdown(f'<div class="sec-hdr">{L["material_header"]}</div>', unsafe_allow_html=True)
    st.info(L["material_info"])

    stages = estimate_materials(sqft, hall, kitchen, floor, bathroom, garden, parking, pooja_room, quality_key)

    for i in range(0, len(stages), 2):
        ca, cb = st.columns(2)
        for col_obj, idx in [(ca, i), (cb, i+1)]:
            if idx < len(stages):
                stg = stages[idx]  # type: ignore
                rows = "".join(
                    f'<div class="mat-row">🔹 <b>{n}</b> &nbsp;<span class="qty-badge">{q}</span></div>'
                    for n, q in stg["items"]
                )
                col_obj.markdown(
                    f'<div class="stage-card"><div class="stage-title">{stg["icon"]} {stg["title"]}</div>{rows}</div>',
                    unsafe_allow_html=True)

    # ── Grand total ───────────────────────────
    # Local multiplier for grand total table
    gtm = 1.15 if quality_key == "High" else (0.90 if quality_key == "Low" else 1.0)

    st.markdown(f'<div class="sec-hdr">{L["grand_total"]}</div>', unsafe_allow_html=True)
    st.table(pd.DataFrame({
        L["material_col"]: [
            "🧱 Cement (Total)", "🏜️ Sand (Total)", "⚙️ Steel / TMT Bars", 
            "🧱 Bricks / Blocks", "🟦 Tiles / Flooring", "🎨 Paint (2 coats)",
            "🧴 Wall Putty", "🧴 Wall Primer", "⚙️ Tile Adhesive",
            "🪵 Centering Sheets", "🚿 PVC/CPVC Pipes"
        ],
        L["qty_col"]: [
            f"{int(sqft*.50*gtm)} Bags", f"{int(sqft*1.20*gtm)} CFT", f"{int(sqft*3.3*gtm)} Kg",
            f"{int(sqft*9*floor)} Nos", f"{int(sqft*1.1)} Sqft", f"{int(sqft*.12)} Litres",
            f"{int(sqft*.05)} Kg", f"{int(sqft*.08)} Litres", f"{int(sqft*.03)} Bags",
            f"{int(sqft*1.2)} Sqft", f"{int(sqft*1.5)} Metres"
        ],
    }))

    st.success(L["success_msg"])

    # ── Download Report ───────────────────────
    st.markdown('<div class="sec-hdr">📥 Download Full Report</div>', unsafe_allow_html=True)

    import io
    import openpyxl  # type: ignore
    from openpyxl.styles import Font, PatternFill, Alignment  # type: ignore

    def make_excel():
        wb = openpyxl.Workbook()

        hdr_font  = Font(bold=True, color="FFFFFF", size=11)
        hdr_fill  = PatternFill("solid", fgColor="302b63")
        gold_fill = PatternFill("solid", fgColor="f7971e")
        gold_font = Font(bold=True, color="FFFFFF", size=12)
        center    = Alignment(horizontal="center")

        def style_header(ws, row, cols):
            for col in range(1, cols+1):
                cell = ws.cell(row=row, column=col)
                cell.font = hdr_font
                cell.fill = hdr_fill
                cell.alignment = center

        def auto_width(ws):
            for col in ws.columns:
                first = col[0]
                if not hasattr(first, "column_letter"):
                    continue
                max_len = max((len(str(c.value)) for c in col if hasattr(c, "value") and c.value), default=10)  # type: ignore
                ws.column_dimensions[first.column_letter].width = min(max_len + 4, 40)

        # ── Sheet 1: Summary ──────────────────
        ws1 = wb.active
        ws1.title = "Summary"

        ws1.append(["Santhosh AI – House Price Report"])
        ws1["A1"].font = gold_font
        ws1["A1"].fill = gold_fill
        ws1["A1"].alignment = center
        ws1.merge_cells("A1:C1")
        ws1.append([])

        ws1.append(["House Details", "Value", ""])
        style_header(ws1, 3, 3)
        ws1.append(["Total Area", f"{sqft} sqft", ""])
        ws1.append(["Halls", hall, ""])
        ws1.append(["Kitchens", kitchen, ""])
        ws1.append(["Floors", floor, ""])
        ws1.append(["Bathrooms", bathroom, ""])
        ws1.append(["Parking Spaces", parking, ""])
        ws1.append(["Garden", "Yes" if garden else "No", ""])
        ws1.append(["Pooja Room", "Yes" if pooja_room else "No", ""])
        ws1.append([])

        ws1.append(["Cost Breakdown", "Amount (INR)", "% of Total"])
        style_header(ws1, 13, 3)
        ws1.append(["Predicted Total Price", f"₹{pred:,.0f}", "100%"])
        ws1.append(["Material Cost", f"₹{mat_total:,.0f}", "55%"])
        ws1.append(["Labor Cost", f"₹{labor_total:,.0f}", "25%"])
        ws1.append(["Interior Cost", f"₹{inter_total:,.0f}", "20%"])
        ws1.append([])

        ws1.append(["Material Cost Sub-items", "Amount (INR)", ""])
        style_header(ws1, 19, 3)
        for n, v in mat_items:
            ws1.append([n, f"₹{v:,.0f}", ""])

        ws1.append([])
        ws1.append(["Labor Cost Sub-items", "Amount (INR)", ""])
        style_header(ws1, 30, 3)
        for n, v in labor_items:
            ws1.append([n, f"₹{v:,.0f}", ""])

        ws1.append([])
        ws1.append(["Interior Cost Sub-items", "Amount (INR)", ""])
        style_header(ws1, 38, 3)
        for n, v in inter_items:
            ws1.append([n, f"₹{v:,.0f}", ""])

        auto_width(ws1)

        # ── Sheet 2: Material Quantities ──────
        ws2 = wb.create_sheet("Material Quantities")
        ws2.append(["Stage", "Material", "Quantity"])
        style_header(ws2, 1, 3)
        for stg in stages:
            for name, qty in stg["items"]:
                ws2.append([stg["title"], name, qty])
        auto_width(ws2)

        # ── Sheet 3: Grand Total ──────────────
        ws3 = wb.create_sheet("Grand Total")
        ws3.append(["Material", "Total Quantity"])
        style_header(ws3, 1, 2)
        ws3.append(["Cement (all stages)", f"{int(sqft*.50)} Bags"])
        ws3.append(["Sand (all stages)",   f"{int(sqft*1.20)} CFT"])
        ws3.append(["Steel / TMT",         f"{int(sqft*3.3)} Kg"])
        ws3.append(["Bricks",              f"{int(sqft*9*floor)} Nos"])
        ws3.append(["Tiles / Flooring",    f"{int(sqft*1.1)} Sqft"])
        ws3.append(["Paint",               f"{int(sqft*.12)} Litres"])
        auto_width(ws3)

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.getvalue()

    excel_data = make_excel()
    st.download_button(
        label="📥 Download Full Report (Excel)",
        data=excel_data,
        file_name="Santhosh_AI_House_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ── Sidebar info ──────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏠 Santhosh AI")
st.sidebar.info("Decision Tree model · Standard civil engineering thumb rules")
st.sidebar.markdown("### 🚀 Deploy Free\n1. Push to [GitHub](https://github.com)\n2. [share.streamlit.io](https://share.streamlit.io) → Deploy")