import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import colorsys
import numpy as np

st.set_page_config(page_title="Raman Cleaner", layout="wide")
st.title("Raman Cleaner")

DASH_OPTIONS = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


# =========================
# Helpers
# =========================
def hsv_color(i: int, n: int) -> str:
    if n <= 0:
        n = 1
    h = (i / n) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.85)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def read_spectrum(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError("Plik musi zawierać co najmniej 2 kolumny: Raman shift i intensywność.")

    out = pd.DataFrame(
        {
            "x": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
            "y": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        }
    ).dropna()

    if out.empty:
        raise ValueError("Nie udało się wczytać danych liczbowych (sprawdź separator i format).")

    return out


def ensure_defaults(names: list[str]):
    st.session_state.setdefault("colors", {})
    st.session_state.setdefault("widths", {})
    st.session_state.setdefault("dashes", {})
    st.session_state.setdefault("legend_names", {})  # file_name -> display_name

    for i, name in enumerate(names):
        st.session_state["colors"].setdefault(name, hsv_color(i, len(names)))
        st.session_state["widths"].setdefault(name, 2)
        st.session_state["dashes"].setdefault(name, "solid")
        st.session_state["legend_names"].setdefault(name, name)

    # cleanup for removed spectra
    for key in list(st.session_state["colors"].keys()):
        if key not in names:
            st.session_state["colors"].pop(key, None)
            st.session_state["widths"].pop(key, None)
            st.session_state["dashes"].pop(key, None)
            st.session_state["legend_names"].pop(key, None)


def reset_style_for(name: str, idx: int, n: int):
    st.session_state["colors"][name] = hsv_color(idx, n)
    st.session_state["widths"][name] = 2
    st.session_state["dashes"][name] = "solid"


def closest_y_at_x(x: np.ndarray, y: np.ndarray, x_ref: float) -> float:
    idx = int(np.argmin(np.abs(x - x_ref)))
    return float(y[idx])


def baseline_als(y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = y.size
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)


# ---- Smoothing helpers ----
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window < 1:
        return y
    if window % 2 == 0:
        window += 1
    if window == 1:
        return y
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ypad, kernel, mode="valid")


def savgol_smooth(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    from scipy.signal import savgol_filter

    window = int(window)
    polyorder = int(polyorder)

    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    if polyorder < 0:
        polyorder = 0
    if polyorder >= window:
        polyorder = window - 1
    if window <= polyorder + 1:
        window = polyorder + 3
        if window % 2 == 0:
            window += 1

    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")


def apply_smoothing(
    y: np.ndarray,
    method: str,
    ma_window: int,
    sg_window: int,
    sg_poly: int,
) -> np.ndarray:
    if method == "Brak":
        return y
    if method == "Średnia krocząca":
        return moving_average(y, ma_window)
    if method == "Savitzky–Golay":
        return savgol_smooth(y, sg_window, sg_poly)
    return y


# ---- Stacking helper ----
def stack_spectrum(
    df: pd.DataFrame,
    x_stack: float,
    offset: float,
    idx: int,
    direction: str,
) -> pd.DataFrame:
    """
    y' = y - y(x_stack) + sign * idx * offset
    """
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    y_at = closest_y_at_x(x, y, float(x_stack))
    sign = 1.0 if direction == "Góra" else -1.0
    y2 = y - y_at + sign * float(idx) * float(offset)

    out = df.copy()
    out["y"] = y2
    return out


def apply_processing(
    df: pd.DataFrame,
    # smoothing
    smoothing_method: str,
    ma_window: int,
    sg_window: int,
    sg_poly: int,
    # baseline
    do_baseline: bool,
    lam: float,
    p: float,
    niter: int,
    # normalization
    do_norm: bool,
    norm_mode: str,
    x_ref: float,
    target: float,
    ref_y_at: float | None,
) -> pd.DataFrame:
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    # smoothing first
    try:
        y = apply_smoothing(y, smoothing_method, ma_window, sg_window, sg_poly)
    except Exception as e:
        st.warning(f"Wygładzanie nie zadziałało dla '{df['Spectrum'].iloc[0]}': {e}")

    # baseline
    if do_baseline:
        try:
            base = baseline_als(y, lam=lam, p=p, niter=niter)
            y = y - base
        except Exception as e:
            st.warning(f"Baseline ALS nie zadziałał dla '{df['Spectrum'].iloc[0]}': {e}")

    # normalization
    if do_norm:
        y_at = closest_y_at_x(x, y, x_ref)
        if np.isclose(y_at, 0.0):
            st.warning(
                f"Normalizacja: wartość ~0 w punkcie {x_ref} cm⁻¹ dla '{df['Spectrum'].iloc[0]}' — pomijam."
            )
        else:
            if norm_mode == "equalize":
                y = y * (target / y_at)
            else:  # preserve
                if ref_y_at is None or np.isclose(ref_y_at, 0.0):
                    st.warning("Normalizacja (preserve): problem z widmem referencyjnym — pomijam.")
                else:
                    scale = target / ref_y_at
                    y = y * scale

    out = df.copy()
    out["y"] = y
    return out


def display_name(fn: str) -> str:
    return st.session_state["legend_names"].get(fn, fn)


# =========================
# Upload
# =========================
files = st.file_uploader(
    "Wczytaj widma Raman (.txt, .csv) — możesz zaznaczyć wiele plików",
    type=["txt", "csv"],
    accept_multiple_files=True,
    key="uploader_files",
)

if not files:
    st.info("Wybierz co najmniej jeden plik, żeby zobaczyć wykres.")
    st.stop()

# =========================
# Load spectra
# =========================
spectra = []
errors = []

for f in files:
    try:
        spec = read_spectrum(f)
        spec["Spectrum"] = f.name
        spectra.append(spec)
    except Exception as e:
        errors.append(f"{f.name}: {e}")

if errors:
    st.warning("Niektóre pliki nie weszły:")
    for msg in errors:
        st.write("•", msg)

if not spectra:
    st.error("Nie udało się wczytać żadnego widma.")
    st.stop()

data = pd.concat(spectra, ignore_index=True)
names = list(dict.fromkeys(data["Spectrum"].tolist()))
ensure_defaults(names)

# =========================
# Sidebar: GLOBAL settings
# =========================
st.sidebar.header("Ustawienia globalne")

legend_position = st.sidebar.selectbox("Legenda", ["Prawa", "Dół"], index=0, key="legend_position")
reverse_x = st.sidebar.checkbox("Odwróć oś X (malejące cm⁻¹)", value=False, key="reverse_x")

# ROI typed (not slider)
st.sidebar.divider()
with st.sidebar.expander("Zakres osi X (ROI)", expanded=False):
    xmin = float(data["x"].min())
    xmax = float(data["x"].max())

    c1, c2 = st.columns(2)
    with c1:
        x_from = st.number_input(
            "X od (cm⁻¹)",
            value=float(st.session_state.get("x_from_val", xmin)),
            step=1.0,
            key="x_from_val",
        )
    with c2:
        x_to = st.number_input(
            "X do (cm⁻¹)",
            value=float(st.session_state.get("x_to_val", xmax)),
            step=1.0,
            key="x_to_val",
        )

    x_from = float(x_from)
    x_to = float(x_to)
    if x_from > x_to:
        x_from, x_to = x_to, x_from

# Axis labels + legend edit
st.sidebar.divider()
with st.sidebar.expander("Etykiety osi + nazwy legendy", expanded=False):
    x_label = st.text_input("Etykieta osi X", value="Raman shift (cm⁻¹)", key="x_label_free")
    y_label = st.text_input("Etykieta osi Y", value="Intensity", key="y_label_free")

    st.markdown("**Nazwy widm w legendzie**")
    sel_for_legend = st.selectbox("Wybierz widmo", names, key="legend_sel_one")
    new_name = st.text_input(
        "Nazwa wyświetlana",
        value=st.session_state["legend_names"].get(sel_for_legend, sel_for_legend),
        key="legend_new_name_one",
    )
    cA, cB = st.columns(2)
    with cA:
        if st.button("Zapisz nazwę", use_container_width=True, key="legend_save_one"):
            st.session_state["legend_names"][sel_for_legend] = new_name
    with cB:
        if st.button("Resetuj nazwę", use_container_width=True, key="legend_reset_one"):
            st.session_state["legend_names"][sel_for_legend] = sel_for_legend

# Smoothing
st.sidebar.divider()
with st.sidebar.expander("Wygładzanie danych", expanded=False):
    smoothing_method = st.selectbox(
        "Metoda wygładzania",
        ["Brak", "Średnia krocząca", "Savitzky–Golay"],
        index=0,
        key="smoothing_method",
    )

    ma_window = st.slider(
        "Okno średniej kroczącej (punkty)",
        min_value=1,
        max_value=301,
        value=11,
        step=2,
        disabled=(smoothing_method != "Średnia krocząca"),
        key="ma_window",
    )

    sg_window = st.slider(
        "Okno Savitzky–Golay (punkty)",
        min_value=3,
        max_value=301,
        value=11,
        step=2,
        disabled=(smoothing_method != "Savitzky–Golay"),
        key="sg_window",
    )
    sg_poly = st.slider(
        "Rząd wielomianu (polyorder)",
        min_value=1,
        max_value=7,
        value=3,
        step=1,
        disabled=(smoothing_method != "Savitzky–Golay"),
        key="sg_poly",
    )
    st.caption("Wygładzanie wykonywane jest przed baseline i normalizacją.")

# Baseline ALS
st.sidebar.divider()
with st.sidebar.expander("Korekcja linii bazowej (ALS)", expanded=False):
    do_baseline = st.checkbox("Włącz baseline ALS (dla wszystkich widm)", value=False, key="do_baseline")

    lam = st.number_input(
        "λ (lambda)",
        min_value=1.0,
        value=1e6,
        step=1e5,
        format="%.0f",
        disabled=not do_baseline,
        key="als_lambda",
    )
    p = st.number_input(
        "p (asymetria, 0–1)",
        min_value=0.0001,
        max_value=0.9999,
        value=0.001,
        step=0.0005,
        format="%.4f",
        disabled=not do_baseline,
        key="als_p",
    )
    niter = st.slider("Iteracje", 1, 50, 10, 1, disabled=not do_baseline, key="als_iter")

# Normalization
st.sidebar.divider()
with st.sidebar.expander("Normalizacja (do punktu na widmie)", expanded=False):
    do_norm = st.checkbox("Włącz normalizację", value=False, key="do_norm")

    norm_mode = st.selectbox(
        "Tryb normalizacji",
        [
            "Zrównać (wszystkie w punkcie mają target)",
            "Zachowaj różnice (skalowanie wspólne względem widma referencyjnego)",
        ],
        index=0,
        disabled=not do_norm,
        key="norm_mode",
    )

    apply_norm_to = st.selectbox(
        "Zastosuj normalizację do",
        ["Wszystkie widma", "Wybrane widma"],
        index=0,
        disabled=not do_norm,
        key="apply_norm_to",
    )

    norm_selected = []
    if do_norm and apply_norm_to == "Wybrane widma":
        norm_selected = st.multiselect(
            "Wybierz widma (normalizacja)",
            options=names,
            default=names[:1] if names else [],
            key="norm_selected",
        )

    x_ref = st.number_input(
        "Punkt odniesienia X (cm⁻¹)",
        value=1000.0,
        step=1.0,
        disabled=not do_norm,
        key="norm_xref",
    )
    target = st.number_input(
        "Do jakiej wartości skalować",
        value=1.0,
        step=0.1,
        disabled=not do_norm,
        key="norm_target",
    )

    ref_spectrum = None
    if do_norm and norm_mode.startswith("Zachowaj"):
        ref_spectrum = st.selectbox(
            "Widmo referencyjne (to ono ma mieć target w punkcie)",
            names,
            index=0,
            key="norm_ref_spectrum",
        )

# Stacking
st.sidebar.divider()
with st.sidebar.expander("Stack (rozsunięcie widm w pionie)", expanded=False):
    stack_mode = st.selectbox(
        "Tryb stack",
        ["Brak", "Stack (względem punktu)"],
        index=0,
        key="stack_mode",
    )

    stack_direction = st.selectbox(
        "Kierunek",
        ["Góra", "Dół"],
        index=0,
        disabled=(stack_mode == "Brak"),
        key="stack_direction",
    )

    x_stack = st.number_input(
        "Punkt odniesienia X (cm⁻¹) dla stack",
        value=1000.0,
        step=1.0,
        disabled=(stack_mode == "Brak"),
        key="x_stack",
    )

    stack_step = st.number_input(
        "Rozdzielenie między widmami (Δy)",
        value=0.2,
        step=0.1,
        disabled=(stack_mode == "Brak"),
        key="stack_step",
    )

    st.caption("Stack wykonywany jest po wygładzaniu, baseline i normalizacji.")

# Style
st.sidebar.divider()
with st.sidebar.expander("Styl widma (edycja wybranego)", expanded=False):
    selected = st.selectbox("Wybierz widmo do edycji", names, key="style_selected_spectrum")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Auto kolory", use_container_width=True, key="btn_auto_colors"):
            for i, name in enumerate(names):
                st.session_state["colors"][name] = hsv_color(i, len(names))
    with colB:
        if st.button("Reset wybranego", use_container_width=True, key="btn_reset_selected"):
            idx = names.index(selected)
            reset_style_for(selected, idx, len(names))

    st.caption("Ustawienia poniżej dotyczą tylko wybranego widma.")
    st.session_state["colors"][selected] = st.color_picker(
        "Kolor",
        value=st.session_state["colors"][selected],
        key=f"cp_{selected}",
    )
    st.session_state["widths"][selected] = st.slider(
        "Grubość linii",
        1,
        10,
        int(st.session_state["widths"][selected]),
        1,
        key=f"lw_{selected}",
    )
    st.session_state["dashes"][selected] = st.selectbox(
        "Styl linii",
        DASH_OPTIONS,
        index=DASH_OPTIONS.index(st.session_state["dashes"][selected]),
        key=f"dash_{selected}",
    )

    highlight_selected = st.checkbox("Podbij wybrane widmo (na wierzch)", value=True, key="highlight_selected")

# =========================
# PROCESSING
# =========================
ref_y_at = None
if do_norm and norm_mode.startswith("Zachowaj") and ref_spectrum is not None:
    df_ref = data[data["Spectrum"] == ref_spectrum].copy()

    df_ref2 = apply_processing(
        df_ref,
        smoothing_method=str(smoothing_method),
        ma_window=int(ma_window),
        sg_window=int(sg_window),
        sg_poly=int(sg_poly),
        do_baseline=bool(do_baseline),
        lam=float(lam),
        p=float(p),
        niter=int(niter),
        do_norm=False,
        norm_mode="equalize",
        x_ref=float(x_ref),
        target=float(target),
        ref_y_at=None,
    )
    x0 = df_ref2["x"].to_numpy(dtype=float)
    y0 = df_ref2["y"].to_numpy(dtype=float)
    ref_y_at = closest_y_at_x(x0, y0, float(x_ref))

processed_parts = []
for name in names:
    df_i = data[data["Spectrum"] == name].copy()

    norm_here = False
    if do_norm:
        norm_here = True if apply_norm_to == "Wszystkie widma" else (name in norm_selected)

    df_i2 = apply_processing(
        df_i,
        smoothing_method=str(smoothing_method),
        ma_window=int(ma_window),
        sg_window=int(sg_window),
        sg_poly=int(sg_poly),
        do_baseline=bool(do_baseline),
        lam=float(lam),
        p=float(p),
        niter=int(niter),
        do_norm=norm_here,
        norm_mode="preserve" if (do_norm and norm_mode.startswith("Zachowaj")) else "equalize",
        x_ref=float(x_ref),
        target=float(target),
        ref_y_at=ref_y_at,
    )
    processed_parts.append(df_i2)

data_proc = pd.concat(processed_parts, ignore_index=True)

# Apply stack AFTER processing
if stack_mode != "Brak":
    stacked_parts = []
    for i, name in enumerate(names):
        df_s = data_proc[data_proc["Spectrum"] == name].copy()
        try:
            df_s2 = stack_spectrum(
                df_s,
                x_stack=float(x_stack),
                offset=float(stack_step),
                idx=int(i),
                direction=str(stack_direction),
            )
        except Exception as e:
            st.warning(f"Stack nie zadziałał dla '{name}': {e}")
            df_s2 = df_s
        stacked_parts.append(df_s2)

    data_proc = pd.concat(stacked_parts, ignore_index=True)

    # ---- prevent negative intensities (FIX) ----
    min_y = float(data_proc["y"].min())
    if min_y < 0:
        data_proc["y"] = data_proc["y"] - min_y

# ROI after processing+stack
data_roi = data_proc[(data_proc["x"] >= x_from) & (data_proc["x"] <= x_to)].copy()

# =========================
# Plot
# =========================
fig = go.Figure()

for name in names:
    if highlight_selected and name == selected:
        continue
    df_plot = data_roi[data_roi["Spectrum"] == name]
    fig.add_trace(
        go.Scatter(
            x=df_plot["x"],
            y=df_plot["y"],
            name=display_name(name),
            mode="lines",
            line=dict(
                width=int(st.session_state["widths"][name]),
                dash=st.session_state["dashes"][name],
                color=st.session_state["colors"][name],
            ),
            hovertemplate=(f"<b>{display_name(name)}</b><br>" + "x=%{x}<br>y=%{y}<extra></extra>"),
        )
    )

if highlight_selected:
    df_plot = data_roi[data_roi["Spectrum"] == selected]
    fig.add_trace(
        go.Scatter(
            x=df_plot["x"],
            y=df_plot["y"],
            name=display_name(selected),
            mode="lines",
            line=dict(
                width=int(st.session_state["widths"][selected]),
                dash=st.session_state["dashes"][selected],
                color=st.session_state["colors"][selected],
            ),
            hovertemplate=(f"<b>{display_name(selected)}</b><br>" + "x=%{x}<br>y=%{y}<extra></extra>"),
        )
    )

fig.update_layout(
    template="simple_white",
    xaxis_title=x_label,
    yaxis_title=y_label,
    legend_title_text="Widma",
    margin=dict(l=20, r=20, t=20, b=20),
)

if legend_position == "Dół":
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0))
else:
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))

if reverse_x:
    fig.update_xaxes(autorange="reversed")

st.plotly_chart(fig, use_container_width=True)

# =========================
# Export CSV
# =========================
st.subheader("Eksport danych (CSV)")

export_names = st.multiselect(
    "Wybierz widma do zapisania",
    options=names,
    default=names,
    key="export_names",
)

if export_names:
    wide = None
    for name in export_names:
        df_e = data_roi[data_roi["Spectrum"] == name][["x", "y"]].copy()
        df_e = df_e.rename(columns={"y": display_name(name)}).drop_duplicates(subset=["x"]).set_index("x")
        wide = df_e if wide is None else wide.join(df_e, how="outer")

    wide = wide.sort_index().reset_index().rename(columns={"x": x_label})
    csv_bytes = wide.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Pobierz CSV",
        data=csv_bytes,
        file_name="raman_export.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_csv",
    )
else:
    st.info("Wybierz przynajmniej jedno widmo do eksportu CSV.")
