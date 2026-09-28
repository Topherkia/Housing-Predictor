"""
Melbourne Housing Map Demo
==========================

Loads an interactive map of Melbourne (Folium / Leaflet + OpenStreetMap) and
plots every property in the chosen Melbourne Housing dataset using its
latitude (``Lattitude``) and longitude (``Longtitude``) columns.

Features
--------
* Choose the dataset: raw, processed, processed (high prices removed) or any CSV path/URL
* One toggleable layer per property type (house / unit / townhouse)
* Markers are clustered so all ~13,500 properties stay fast to render
* Marker colour shows the price band; click a marker for full details
* Optional price heat-map layer, full-screen button and a colour legend

Usage (local)
-------------
    pip install pandas folium
    python map_demo.py                         # asks which dataset to use
    python map_demo.py --dataset processed     # non-interactive
    python map_demo.py --dataset data/raw/melb_data.csv --output my_map.html

The map is saved as an HTML file (default: ``melbourne_housing_map.html``)
that can be opened in any web browser.

Usage (Google Colab / Jupyter)
------------------------------
    !pip install -q folium
    !git clone -b map_branch https://github.com/Topherkia/Housing-Predictor.git
    %cd Housing-Predictor
    from map_demo import load_dataset, build_map
    df = load_dataset("processed")
    m = build_map(df)
    m            # the map renders directly in the notebook cell

If the repository is not cloned, the datasets are downloaded automatically
from GitHub, so ``map_demo.py`` also works when uploaded on its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import folium
    from folium.plugins import FastMarkerCluster, Fullscreen, HeatMap
except ImportError:  # pragma: no cover - helpful message for Colab / fresh envs
    sys.exit("Folium is not installed. Run:  pip install folium   (in Colab: !pip install -q folium)")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Topherkia/Housing-Predictor/map_branch/"

DATASETS = {
    "raw": "data/raw/melb_data.csv",
    "processed": "data/processed/melb_data_processed.csv",
    "processed_rm": "data/processed/melb_data_processed_rm.csv",
}

DATASET_DESCRIPTIONS = {
    "raw": "Original Melbourne Housing Snapshot (melb_data.csv)",
    "processed": "Cleaned dataset (melb_data_processed.csv)",
    "processed_rm": "Cleaned dataset, unusually high prices removed (melb_data_processed_rm.csv)",
}

LAT_COL = "Lattitude"   # spelling as in the original Kaggle dataset
LON_COL = "Longtitude"  # spelling as in the original Kaggle dataset

MELBOURNE_CBD = (-37.8136, 144.9631)

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
BASEMAPS = [
    ("Streets (Esri)", ESRI.format("World_Street_Map"),
     "Tiles &copy; Esri &mdash; Source: Esri, HERE, Garmin, USGS, NGA, EPA, NPS, OpenStreetMap contributors"),
    ("Light grey (Esri)", ESRI.format("Canvas/World_Light_Gray_Base"),
     "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, OpenStreetMap contributors"),
    ("Satellite (Esri)", ESRI.format("World_Imagery"),
     "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"),
    ("OpenStreetMap", "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
     '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'),
]

TYPE_LABELS = {"h": "House", "u": "Unit / Apartment", "t": "Townhouse"}

# Price bands (AUD) -> marker colour
PRICE_BANDS = [
    (0, 500_000, "#2b83ba", "< $500k"),
    (500_000, 800_000, "#abdda4", "$500k – $800k"),
    (800_000, 1_200_000, "#ffffbf", "$800k – $1.2M"),
    (1_200_000, 2_000_000, "#fdae61", "$1.2M – $2M"),
    (2_000_000, float("inf"), "#d7191c", "> $2M"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def resolve_dataset(choice: str) -> str:
    """Turn a dataset key, local path or URL into something pandas can read.

    Local files inside the repository are preferred. If they are missing
    (e.g. only this script was uploaded to Colab) the file is fetched from GitHub.
    """
    if choice in DATASETS:
        relative = DATASETS[choice]
        for base in (PROJECT_ROOT, Path.cwd()):
            local = base / relative
            if local.exists():
                return str(local)
        print(f"Local file '{relative}' not found – downloading it from GitHub...")
        return GITHUB_RAW_BASE + relative

    if choice.startswith(("http://", "https://")):
        return choice

    path = Path(choice).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return str(path)


def load_dataset(choice: str = "processed") -> pd.DataFrame:
    """Load the chosen dataset and keep only rows with valid Melbourne coordinates."""
    source = resolve_dataset(choice)
    df = pd.read_csv(source)

    missing = [c for c in (LAT_COL, LON_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing the coordinate columns: {missing}")

    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    before = len(df)
    # Keep coordinates in a sensible Greater-Melbourne / Victoria window
    df = df.dropna(subset=[LAT_COL, LON_COL])
    df = df[df[LAT_COL].between(-39.0, -37.0) & df[LON_COL].between(143.0, 146.0)]
    dropped = before - len(df)

    print(f"Loaded {len(df):,} properties from {source}"
          + (f" ({dropped:,} rows without valid coordinates skipped)" if dropped else ""))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Map building helpers
# ---------------------------------------------------------------------------
def price_colour(price) -> str:
    if pd.isna(price):
        return "#888888"
    for low, high, colour, _ in PRICE_BANDS:
        if low <= price < high:
            return colour
    return PRICE_BANDS[-1][2]


def _fmt(value, fmt="{:,.0f}", default="–"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


# Columns sent to the browser for each marker (kept compact so the map is
# small enough to render smoothly inside a Colab/Jupyter output cell).
POPUP_FIELDS = ["Address", "Suburb", "Postcode", "Price", "Type", "Rooms", "Bathroom",
                "Car", "Landsize", "BuildingArea", "YearBuilt", "Distance",
                "Regionname", "Date", "Method", "SellerG"]


def _clean(value):
    """Convert a cell to a compact JSON-friendly value (NaN -> None, 3.0 -> 3)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, float):
        return round(value, 1)
    return value


# JavaScript executed by FastMarkerCluster for every point.
# Each data row is [lat, lon, colour, Address, Suburb, Postcode, Price, ...].
# Building the popup in the browser keeps the HTML file small and the map
# responsive even with the full ~13.5k-row dataset.
_MARKER_CALLBACK = """
function (row) {
    var f = function (v, suffix) {
        if (v === null || v === undefined || v === '') return '–';
        if (typeof v === 'number') v = v.toLocaleString('en-AU');
        return v + (suffix || '');
    };
    var types = {h: 'House', u: 'Unit / Apartment', t: 'Townhouse'};
    var html = '<b>' + f(row[3]) + '</b><br>' + f(row[4]) + ' ' + (row[5] || '') +
        '<hr style="margin:4px 0">' +
        '<b>Price:</b> $' + f(row[6]) + '<br>' +
        '<b>Type:</b> ' + (types[row[7]] || f(row[7])) + '<br>' +
        '<b>Rooms:</b> ' + f(row[8]) + ' &nbsp; <b>Bath:</b> ' + f(row[9]) +
        ' &nbsp; <b>Car:</b> ' + f(row[10]) + '<br>' +
        '<b>Land:</b> ' + f(row[11], ' m²') + ' &nbsp; <b>Building:</b> ' + f(row[12], ' m²') + '<br>' +
        '<b>Year built:</b> ' + (row[13] || '–') + '<br>' +
        '<b>Distance to CBD:</b> ' + f(row[14], ' km') + '<br>' +
        '<b>Region:</b> ' + f(row[15]) + '<br>' +
        '<b>Sold:</b> ' + f(row[16]) + ' (' + f(row[17]) + ') by ' + f(row[18]) + '<br>' +
        '<span style="color:#888">' + row[0].toFixed(5) + ', ' + row[1].toFixed(5) + '</span>';
    var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
        radius: 6, color: '#333', weight: 1, fillColor: row[2], fillOpacity: 0.9
    });
    marker.bindPopup(html, {maxWidth: 320});
    marker.bindTooltip(f(row[4]) + ' – $' + f(row[6]));
    return marker;
}
"""


def _marker_rows(df: pd.DataFrame) -> list:
    cols = [c if c in df.columns else None for c in POPUP_FIELDS]
    rows = []
    for r in df.to_dict("records"):
        rows.append([round(r[LAT_COL], 5), round(r[LON_COL], 5), r["_colour"]]
                    + [_clean(r[c]) if c else None for c in cols])
    return rows


def _legend_html(n_points: int, dataset_label: str) -> str:
    items = "".join(
        f"<div><span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
        f"background:{c};border:1px solid #333;margin-right:6px'></span>{label}</div>"
        for _, _, c, label in PRICE_BANDS
    )
    return f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 10px 12px; border-radius: 6px;
                box-shadow: 0 1px 6px rgba(0,0,0,.3); font: 12px/1.5 Arial, sans-serif;">
        <b>Melbourne Housing</b><br>
        <span style="color:#555">{dataset_label}<br>{n_points:,} properties</span>
        <hr style="margin:6px 0">
        <b>Sale price</b>{items}
    </div>"""


def build_map(
    df: pd.DataFrame,
    dataset_label: str = "",
    cluster: bool = True,
    heatmap: bool = True,
    zoom_start: int = 11,
) -> folium.Map:
    """Create a Folium map of Melbourne with every property in ``df``."""
    m = folium.Map(location=MELBOURNE_CBD, zoom_start=zoom_start,
                   tiles=None, control_scale=True)
    # Base maps. Esri streets is the default because it needs no API key and
    # OpenStreetMap's own tile servers may block requests coming from notebook
    # or sandboxed iframes (e.g. some Colab outputs). Switch base maps in the layer control.
    for i, (name, url, attr) in enumerate(BASEMAPS):
        folium.TileLayer(tiles=url, attr=attr, name=name, max_zoom=19,
                         show=(i == 0)).add_to(m)

    folium.Marker(
        MELBOURNE_CBD, tooltip="Melbourne CBD",
        icon=folium.Icon(color="black", icon="star"),
    ).add_to(m)

    df = df.copy()
    df["_colour"] = df["Price"].apply(price_colour) if "Price" in df.columns else "#3388ff"

    type_col = df["Type"] if "Type" in df.columns else pd.Series("all", index=df.index)
    for ptype, group in df.groupby(type_col):
        label = f"{TYPE_LABELS.get(ptype, str(ptype))} ({len(group):,})"
        layer = folium.FeatureGroup(name=label, show=True)
        data = _marker_rows(group)

        if cluster:
            FastMarkerCluster(data=data, callback=_MARKER_CALLBACK).add_to(layer)
        else:
            for row in data:
                lat, lon, colour = row[:3]
                folium.CircleMarker(
                    (lat, lon), radius=4, color="#333", weight=1,
                    fill=True, fill_color=colour, fill_opacity=0.9,
                    tooltip=f"{row[4]} – ${_fmt(row[6])}",
                ).add_to(layer)
        layer.add_to(m)

    if heatmap and "Price" in df.columns:
        heat = df.dropna(subset=["Price"])
        weights = (heat["Price"] / heat["Price"].quantile(0.95)).clip(upper=1.0)
        HeatMap(
            [[round(a, 5), round(b, 5), round(w, 3)]
             for a, b, w in zip(heat[LAT_COL], heat[LON_COL], weights)],
            name="Price heat-map", radius=12, blur=15, min_opacity=0.3, show=False,
        ).add_to(m)

    Fullscreen().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html(len(df), dataset_label)))

    # Zoom to the actual extent of the data
    m.fit_bounds([[df[LAT_COL].min(), df[LON_COL].min()],
                  [df[LAT_COL].max(), df[LON_COL].max()]])
    return m


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def ask_for_dataset() -> str:
    keys = list(DATASETS)
    print("\n==============================================")
    print("Select a dataset to show on the Melbourne map:")
    print("==============================================")
    for i, key in enumerate(keys, 1):
        print(f"[{i}] {DATASET_DESCRIPTIONS[key]}")
    print(f"[{len(keys) + 1}] Custom CSV path or URL")
    print("==============================================")
    while True:
        answer = input(f"Enter choice [1-{len(keys) + 1}] (default 2): ").strip() or "2"
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            return keys[int(answer) - 1]
        if answer == str(len(keys) + 1):
            return input("CSV path or URL: ").strip()
        print("Invalid choice, please try again.")


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Show Melbourne housing data on an interactive map.")
    parser.add_argument("--dataset", "-d", default=None,
                        help="raw | processed | processed_rm | path/URL to a CSV (asked if omitted)")
    parser.add_argument("--output", "-o", default="melbourne_housing_map.html",
                        help="HTML file to write (default: melbourne_housing_map.html)")
    parser.add_argument("--no-cluster", action="store_true",
                        help="Draw every marker individually (slower for the full dataset)")
    parser.add_argument("--no-heatmap", action="store_true", help="Do not add the heat-map layer")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only plot a random sample of N properties")
    # parse_known_args ignores the extra '-f kernel.json' argument Jupyter/Colab adds
    args, _ = parser.parse_known_args(argv)

    choice = args.dataset or ask_for_dataset()
    df = load_dataset(choice)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42)
        print(f"Using a random sample of {len(df):,} properties")

    label = DATASET_DESCRIPTIONS.get(choice, Path(choice).name)
    m = build_map(df, dataset_label=label,
                  cluster=not args.no_cluster, heatmap=not args.no_heatmap)

    out = Path(args.output).expanduser().resolve()
    m.save(str(out))
    print(f"Map saved to: {out}")
    return m


if __name__ == "__main__":
    result = main()
    if not in_notebook():
        print("Open the HTML file in your web browser to explore the map.")
