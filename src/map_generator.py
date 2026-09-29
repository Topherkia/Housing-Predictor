from __future__ import annotations

import sys
import pandas as pd

try:
    import folium
    from branca.element import MacroElement
    from folium.plugins import FastMarkerCluster, Fullscreen, HeatMap
    from jinja2 import Template
except ImportError:
    sys.exit("Folium is required for MapGenerator. Please install via: pip install folium")

LAT_COL = "Lattitude"
LON_COL = "Longtitude"
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

PRICE_BANDS = [
    (0, 500_000, "#2b83ba", "< $500k"),
    (500_000, 800_000, "#abdda4", "$500k – $800k"),
    (800_000, 1_200_000, "#ffffbf", "$800k – $1.2M"),
    (1_200_000, 2_000_000, "#fdae61", "$1.2M – $2M"),
    (2_000_000, float("inf"), "#d7191c", "> $2M"),
]

ERROR_BANDS = [
    (float("-inf"), -30, "#2166ac", "Under-estimated > 30%"),
    (-30, -10, "#92c5de", "Under-estimated 10–30%"),
    (-10, 10, "#1a9850", "Within ±10%"),
    (10, 30, "#f4a582", "Over-estimated 10–30%"),
    (30, float("inf"), "#b2182b", "Over-estimated > 30%"),
]

CLUSTER_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"
]

POPUP_FIELDS = [
    "Address", "Suburb", "Postcode", "Price", "Type", "Rooms", "Bathroom",
    "Car", "Landsize", "BuildingArea", "YearBuilt", "Distance",
    "Regionname", "Date", "Method", "SellerG", "Cluster", "Predicted", "ErrorPct"
]

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
        (row[19] !== null ? '<b>K-Means cluster:</b> ' + row[19] + '<br>' : '') +
        (row[20] !== null ? '<hr style="margin:4px 0"><b>Predicted price:</b> $' + f(row[20]) +
            '<br><b>Error:</b> ' + (row[21] > 0 ? '+' : '') + f(row[21], '%') +
            (row[21] > 0 ? ' (over-estimated)' : ' (under-estimated)') + '<br>' : '') +
        '<span style="color:#888">' + row[0].toFixed(5) + ', ' + row[1].toFixed(5) + '</span>';
    var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
        radius: 6, color: '#333', weight: 1, fillColor: row[2], fillOpacity: 0.9
    });
    marker.bindPopup(html, {maxWidth: 320});
    marker.bindTooltip(f(row[4]) + ' – $' + f(row[6]));
    return marker;
}
"""

_REFIT_JS = """
(function () {
    var map = %s, bounds = %s, touched = false;
    function refit() {
        if (touched) return;
        map.invalidateSize();
        if (map.getSize().y > 0) map.fitBounds(bounds);
    }
    ['mousedown', 'wheel', 'touchstart', 'keydown'].forEach(function (ev) {
        map.getContainer().addEventListener(ev, function () { touched = true; });
    });
    if (window.ResizeObserver) new ResizeObserver(refit).observe(map.getContainer());
    window.addEventListener('load', function () { setTimeout(refit, 250); });
})();
"""


class MapGenerator:
    """Class that builds a Folium map and returns raw HTML string output."""

    def __init__(self, df: pd.DataFrame, dataset_label: str = "Melbourne Properties"):
        self.df = self._clean_coordinates(df)
        self.dataset_label = dataset_label

    @staticmethod
    def _clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if LAT_COL not in df.columns or LON_COL not in df.columns:
            return df
        df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
        df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
        if "Price" in df.columns:
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=[LAT_COL, LON_COL])
        return df[df[LAT_COL].between(-39.0, -37.0) & df[LON_COL].between(143.0, 146.0)].reset_index(drop=True)

    @staticmethod
    def _price_colour(price) -> str:
        if pd.isna(price):
            return "#888888"
        for low, high, colour, _ in PRICE_BANDS:
            if low <= price < high:
                return colour
        return PRICE_BANDS[-1][2]

    @staticmethod
    def _error_colour(error_pct) -> str:
        if error_pct is None or pd.isna(error_pct):
            return "#888888"
        for low, high, colour, _ in ERROR_BANDS:
            if low <= error_pct < high:
                return colour
        return ERROR_BANDS[-1][2]

    @staticmethod
    def _cluster_colour(cluster) -> str:
        if cluster is None or pd.isna(cluster):
            return "#888888"
        return CLUSTER_COLOURS[int(cluster) % len(CLUSTER_COLOURS)]

    @staticmethod
    def _clean_val(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, float):
            return round(value, 1)
        return value

    def _marker_rows(self, df: pd.DataFrame) -> list:
        cols = [c if c in df.columns else None for c in POPUP_FIELDS]
        rows = []
        for r in df.to_dict("records"):
            rows.append([round(r[LAT_COL], 5), round(r[LON_COL], 5), r["_colour"]]
                        + [self._clean_val(r[c]) if c else None for c in cols])
        return rows

    def _apply_colours(self, df: pd.DataFrame, color_by: str):
        if color_by == "cluster":
            if "Cluster" not in df.columns:
                raise ValueError("color_by='cluster' requires a 'Cluster' column.")
            df["_colour"] = df["Cluster"].apply(self._cluster_colour)
            stats = df.groupby("Cluster")["Price"].agg(["count", "median"]) if "Price" in df.columns \
                else df.groupby("Cluster").size().to_frame("count").assign(median=float("nan"))
            items = [
                (self._cluster_colour(c), f"Cluster {int(c)} – {int(r['count']):,} homes"
                 + (f", median ${r['median'] / 1e6:.2f}M" if pd.notna(r["median"]) else ""))
                for c, r in stats.iterrows()
            ]
            return "K-Means cluster", items

        if color_by == "error":
            if "ErrorPct" not in df.columns:
                raise ValueError("color_by='error' requires an 'ErrorPct' column.")
            df["_colour"] = df["ErrorPct"].apply(self._error_colour)
            return "Prediction error", [(c, label) for _, _, c, label in ERROR_BANDS]

        df["_colour"] = df["Price"].apply(self._price_colour) if "Price" in df.columns else "#3388ff"
        return "Sale price", [(c, label) for _, _, c, label in PRICE_BANDS]

    def _legend_html(self, n_points: int, title: str, items: list) -> str:
        rows = "".join(
            f"<div><span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
            f"background:{c};border:1px solid #333;margin-right:6px'></span>{label}</div>"
            for c, label in items
        )
        return f"""
        <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                    background: white; padding: 10px 12px; border-radius: 6px; max-width: 260px;
                    box-shadow: 0 1px 6px rgba(0,0,0,.3); font: 12px/1.5 Arial, sans-serif;">
            <b>Melbourne Housing</b><br>
            <span style="color:#555">{self.dataset_label}<br>{n_points:,} properties</span>
            <hr style="margin:6px 0">
            <b>{title}</b>{rows}
        </div>"""

    def generate_html(
        self,
        color_by: str = "price",
        cluster: bool = True,
        heatmap: bool = True,
        zoom_start: int = 11,
    ) -> str:
        """Builds the map and returns strictly its HTML representation."""
        if self.df.empty:
            raise ValueError("No valid coordinates found to construct map HTML.")

        m = folium.Map(location=MELBOURNE_CBD, zoom_start=zoom_start, tiles=None, control_scale=True)

        for i, (name, url, attr) in enumerate(BASEMAPS):
            folium.TileLayer(tiles=url, attr=attr, name=name, max_zoom=19, show=(i == 0)).add_to(m)

        folium.Marker(MELBOURNE_CBD, tooltip="Melbourne CBD", icon=folium.Icon(color="black", icon="star")).add_to(m)

        df = self.df.copy()
        legend_title, legend_items = self._apply_colours(df, color_by)

        type_col = df["Type"] if "Type" in df.columns else pd.Series("all", index=df.index)
        for ptype, group in df.groupby(type_col):
            label = f"{TYPE_LABELS.get(ptype, str(ptype))} ({len(group):,})"
            layer = folium.FeatureGroup(name=label, show=True)
            data = self._marker_rows(group)

            if cluster:
                FastMarkerCluster(data=data, callback=_MARKER_CALLBACK).add_to(layer)
            else:
                for row in data:
                    lat, lon, colour = row[:3]
                    folium.CircleMarker(
                        (lat, lon), radius=4, color="#333", weight=1,
                        fill=True, fill_color=colour, fill_opacity=0.9,
                        tooltip=f"{row[4]} – ${row[6]}",
                    ).add_to(layer)
            layer.add_to(m)

        if heatmap and "Price" in df.columns:
            heat = df.dropna(subset=["Price"])
            if not heat.empty:
                weights = (heat["Price"] / heat["Price"].quantile(0.95)).clip(upper=1.0)
                HeatMap(
                    [[round(a, 5), round(b, 5), round(w, 3)]
                     for a, b, w in zip(heat[LAT_COL], heat[LON_COL], weights)],
                    name="Price heat-map", radius=12, blur=15, min_opacity=0.3, show=False,
                ).add_to(m)

        m.get_root().header.add_child(folium.Element("""
        <style>
          .marker-cluster-small, .marker-cluster-medium, .marker-cluster-large
              { background-color: rgba(90, 90, 90, 0.25) !important; }
          .marker-cluster-small div, .marker-cluster-medium div, .marker-cluster-large div
              { background-color: rgba(60, 60, 60, 0.75) !important; color: #fff !important; }
        </style>"""))

        Fullscreen().add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        m.get_root().html.add_child(folium.Element(self._legend_html(len(df), legend_title, legend_items)))

        bounds = [[float(df[LAT_COL].min()), float(df[LON_COL].min())],
                  [float(df[LAT_COL].max()), float(df[LON_COL].max())]]
        m.fit_bounds(bounds)

        refit = MacroElement()
        refit._template = Template(
            "{% macro script(this, kwargs) %}" + _REFIT_JS % (m.get_name(), bounds) + "{% endmacro %}"
        )
        m.add_child(refit)

        # Render and return raw HTML string
        return m.get_root().render()