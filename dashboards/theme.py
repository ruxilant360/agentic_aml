"""
Chart palette and chrome tokens shared by the Streamlit dashboard.

Both modes are *selected*, not derived: the dark column is the same hues
re-stepped for the dark surface. The categorical order is fixed -- slot 1 is
always slot 1, so a filter that drops a series never repaints the survivors.

Palette provenance: validated with the data-viz palette validator for the
five slots used here.

    light (surface #fcfcfb): lightness band PASS, chroma floor PASS,
        adjacent CVD worst dE 9.1, adjacent normal-vision worst dE 19.6,
        contrast WARN -> three slots sit below 3:1 on the light surface, so
        every chart that uses them ships visible value labels and/or the
        "Show underlying rows" table (the relief rule).
    dark (surface #1a1a19): all six checks PASS.

Do not add a sixth categorical hue. Fold the tail into "Other" or facet.
"""

from __future__ import annotations

# --- categorical slots, fixed order -----------------------------------------
_CATEGORICAL = {
    "light": [
        "#2a78d6",  # 1 blue
        "#eb6834",  # 2 orange
        "#1baf7a",  # 3 aqua
        "#eda100",  # 4 yellow
        "#e87ba4",  # 5 magenta
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
    ],
}

# --- diverging poles (polarity), warm vs cool with a neutral midpoint -------
# Slot 8 red, NOT the `critical` status token: a status colour must never
# stand in for a non-status meaning (here, "this feature raised the score").
# Validated as a pair against both surfaces -- light dE 21.6 CVD / 32.3
# normal, dark 19.2 / 29.0, contrast >= 3:1 in both.
_DIVERGING_WARM = {"light": "#e34948", "dark": "#e66767"}

# --- single-hue sequential ramp (magnitude), light -> dark -------------------
_SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# --- status tokens: fixed, never themed, never reused as a series color -----
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# --- chrome / ink -----------------------------------------------------------
_CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "gridline": "#e1e0d9",
        "baseline": "#c3c2b7",
        "neutral_mid": "#f0efec",  # diverging midpoint
        "deemphasis": "#c3c2b7",   # the "rest" in an emphasis chart
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "gridline": "#2c2c2a",
        "baseline": "#383835",
        "neutral_mid": "#383835",
        "deemphasis": "#52514e",
    },
}

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Transaction types get a *stable* hue each, so `type` keeps its color
# across every chart and every filter state in the app.
TYPE_ORDER = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class Theme:
    """Resolved palette + chrome for one mode ('light' or 'dark')."""

    def __init__(self, mode: str = "light") -> None:
        self.mode = "dark" if str(mode).lower().startswith("d") else "light"
        self.categorical = _CATEGORICAL[self.mode]
        self.sequential = list(_SEQUENTIAL)
        if self.mode == "dark":
            # On the dark surface the lightest sequential steps disappear into
            # the background; drop them rather than invent new ones.
            self.sequential = _SEQUENTIAL[2:]
        self.chrome = _CHROME[self.mode]
        self.status = dict(STATUS)

    # ---- roles -------------------------------------------------------------
    @property
    def accent(self) -> str:
        """Slot 1 -- the single-series / emphasis hue."""
        return self.categorical[0]

    @property
    def contrast(self) -> str:
        """Slot 2 -- the second series when exactly two are on screen."""
        return self.categorical[1]

    def type_colors(self) -> dict:
        """Stable hue per transaction type (identity, not rank)."""
        return {t: self.categorical[i % len(self.categorical)] for i, t in enumerate(TYPE_ORDER)}

    def diverging(self) -> tuple[str, str, str]:
        """
        (cool pole, neutral midpoint, warm pole) for polarity encodings.

        Blue and red read as opposite, and the midpoint is gray so "zero"
        reads as nothing. The warm pole is categorical slot 8, deliberately
        not the `critical` status token -- polarity is not a status.
        """
        return (self.categorical[0], self.chrome["neutral_mid"], _DIVERGING_WARM[self.mode])

    # ---- plotly ------------------------------------------------------------
    def layout(self, **overrides) -> dict:
        """
        Baseline Plotly layout: recessive solid hairline grid, no vertical
        grid, generous padding, ink in text tokens rather than series colors.

        Overrides are merged one level deep, so a caller passing
        `title=dict(text="...")` keeps the title's size, colour and left
        alignment instead of replacing the whole block with a bare string.
        Pass a non-dict to replace a key outright.
        """
        c = self.chrome

        def axis_common() -> dict:
            # Built fresh per axis: the nested `title` dict must not be shared
            # between x and y, or setting one axis's label text sets both.
            return dict(
                # Let Plotly reserve whatever the tick labels and axis title
                # actually need. Fixed margins clip them as soon as a label grows.
                automargin=True,
                zeroline=False,
                tickfont=dict(color=c["muted"], size=12),
                title=dict(font=dict(color=c["muted"], size=12), standoff=12),
            )
        layout = dict(
            # Plotly's default template paints its own surface, grid and
            # colorway. Switch it off so this palette is the only thing
            # deciding how a chart looks.
            template="none",
            paper_bgcolor=c["surface"],
            plot_bgcolor=c["surface"],
            font=dict(family=FONT_STACK, size=13, color=c["text_secondary"]),
            title=dict(
                font=dict(size=15, color=c["text_primary"]),
                x=0, xanchor="left", y=0.97, yanchor="top",
            ),
            margin=dict(l=12, r=20, t=64, b=20),
            xaxis=dict(
                **axis_common(),
                showgrid=False,
                showline=True,
                linecolor=c["baseline"],
                linewidth=1,
            ),
            yaxis=dict(
                **axis_common(),
                showgrid=True,
                gridcolor=c["gridline"],
                gridwidth=1,
                griddash="solid",
                showline=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.0,
                xanchor="left",
                x=0,
                font=dict(color=c["text_secondary"], size=12),
                bgcolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(
                bgcolor=c["surface"],
                bordercolor=c["baseline"],
                font=dict(family=FONT_STACK, size=12, color=c["text_primary"]),
            ),
            hovermode="closest",
            # Thin marks: more surface than ink. Thick saturated blocks read loud.
            bargap=0.5,
        )
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(layout.get(key), dict):
                merged = dict(layout[key])
                merged.update(value)
                layout[key] = merged
            else:
                layout[key] = value
        return layout
