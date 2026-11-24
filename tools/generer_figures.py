"""Recrée les figures du README depuis les sorties locales du pricer."""

import argparse
from pathlib import Path
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


INK = "#142638"
MUTED = "#566b7e"
GRID = "#dce5ec"
PAGE = "#f5f8fb"
COLORS = {"3M": "#0c776e", "6M": "#4758a1"}
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": PAGE,
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.65,
    "text.color": INK,
}


def _load(folder: Path, name: str):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def _grid(result):
    rows = result["slices"]
    years = np.array([row["nominal_start_years"] for row in rows], dtype=float)
    strike_grid = sorted({point["strike"] for row in rows for point in row["curve"]})
    if not strike_grid:
        raise ValueError(f"Aucun point calculé pour {result['index_tenor']}.")
    vol_bp = np.full((len(rows), len(strike_grid)), np.nan)
    for i, row in enumerate(rows):
        by_strike = {point["strike"]: point["extended_normal_vol"] for point in row["curve"]}
        for j, strike in enumerate(strike_grid):
            vol = by_strike.get(strike)
            if vol is not None:
                vol_bp[i, j] = vol * 10_000
    strikes = np.array(strike_grid) * 100
    forward_percent = np.array([row["forward_rate"] for row in rows], dtype=float) * 100
    return years, strikes, vol_bp, forward_percent


def _heading(fig, title, description):
    fig.text(0.075, 0.937, title, fontsize=21, weight="bold", va="top")
    fig.text(0.075, 0.937 - 0.37 / fig.get_figheight(), description, fontsize=10.5, color=MUTED, va="top")


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(fig)


def _vol_limits(grids):
    values = np.concatenate([grid[2].ravel() for grid in grids.values()])
    level_min = np.floor(np.nanmin(values) / 10) * 10
    level_max = np.ceil(np.nanmax(values) / 10) * 10
    return level_min, level_max


def surface_figure(surfaces, as_of: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.876, top=0.752, bottom=0.164, wspace=0.14)
    _heading(fig, "Volatilités normales des caplets EUR", f"Rejeu du {as_of}  ·  Volatilités en points de base")
    panel_names = {"3M": "Euribor 3M prolongé", "6M": "Euribor 6M V3"}
    grids = {tenor: _grid(result) for tenor, result in surfaces.items()}
    level_min, level_max = _vol_limits(grids)
    image = None
    for ax, tenor in zip(axes, ("3M", "6M"), strict=True):
        years, strikes, vols, _ = grids[tenor]
        image = ax.pcolormesh(
            years, strikes, vols.T, shading="nearest", cmap="viridis",
            vmin=level_min, vmax=level_max, rasterized=True,
        )
        ax.set_title(panel_names[tenor], loc="left", pad=11)
        ax.set(xlim=(0, 30), ylim=(strikes[0] - 0.25, strikes[-1] + 0.25),
               xlabel="Début nominal du coupon (ans)", xticks=np.arange(0, 31, 5))
        ax.set_yticks(np.r_[strikes[0], np.arange(0, 6)])
        if tenor == "3M":
            first_extended = next(
                (row["nominal_start_years"] for row in surfaces[tenor]["slices"]
                 if any(pt["coverage"] is not None for pt in row["curve"])), None
            )
            if first_extended is not None:
                ax.axvline(first_extended, color="white", lw=1.05, ls=(0, (3, 3)))
                ax.text(first_extended + 0.65, -1.20, "Début du prolongement",
                        color="white", fontsize=8.2, weight="bold")
    axes[0].set_ylabel("Strike (%)")
    cax = fig.add_axes([0.904, 0.164, 0.014, 0.588])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Vol. normale (bp)", labelpad=10)
    bar.outline.set_visible(False)
    _save(fig, path)


def surface_3d_figures(surfaces, as_of: str, destination: Path):
    grids = {tenor: _grid(result) for tenor, result in surfaces.items()}
    level_min, level_max = _vol_limits(grids)
    names = {"3M": "Euribor 3M prolongé", "6M": "Euribor 6M V3"}
    french = FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))
    for tenor in ("3M", "6M"):
        years, strikes, vols, _ = grids[tenor]
        strike_mesh, year_mesh = np.meshgrid(strikes, years)

        fig = plt.figure(figsize=(10.8, 6.4))
        ax = fig.add_axes([0.115, 0.12, 0.70, 0.70], projection="3d")
        ax.set_facecolor(PAGE)
        _heading(
            fig,
            f"Surface 3D · {names[tenor]}",
            f"Rejeu du {as_of}  ·  {len(years)} fixings × {len(strikes)} strikes",
        )
        plot = ax.plot_surface(
            strike_mesh, year_mesh, vols, rcount=len(years), ccount=len(strikes),
            cmap="viridis", vmin=level_min, vmax=level_max, linewidth=0, antialiased=False,
        )
        ax.set(
            xlim=(strikes[0], strikes[-1]), ylim=(0, 30), zlim=(level_min, level_max),
            xlabel="Strike (%)", ylabel="Début du coupon (ans)", zlabel="Vol. normale (bp)",
        )
        ax.set_xticks([strikes[0], 0, 1, 2, 3, strikes[-1]])
        ax.set_yticks(np.arange(0, 31, 5))
        ax.xaxis.set_major_formatter(french)
        ax.tick_params(labelsize=10)
        ax.view_init(elev=26, azim=-55)
        ax.set_box_aspect((1.0, 1.30, 0.78), zoom=1.16)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0.88))

        cax = fig.add_axes([0.85, 0.25, 0.018, 0.46])
        bar = fig.colorbar(plot, cax=cax)
        bar.ax.set_title("bp", color=MUTED, fontsize=10, pad=8)
        bar.outline.set_visible(False)
        _save(fig, destination / f"surface-{tenor.lower()}-3d.png")


def projection_figure(surfaces, as_of: str, path: Path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 5.15))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.75, bottom=0.16, wspace=0.34)
    _heading(fig, "Des forwards au smile", f"Rejeu du {as_of}  ·  Exemple de coupe à dix ans")
    for tenor in ("3M", "6M"):
        years, strikes, vol_bp, forwards = _grid(surfaces[tenor])
        k = int(np.argmin(np.abs(years - 10)))
        style = {"color": COLORS[tenor], "label": f"Euribor {tenor}", "lw": 2.0}
        left.plot(years, forwards, **style)
        right.plot(strikes, vol_bp[k], marker="o", ms=3.5, **style)
    left.set_title("Forwards des coupons", loc="left", pad=11)
    left.set(xlabel="Début nominal du coupon (ans)", ylabel="Forward (%)", xlim=(0, 30))
    left.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))
    right.set_title("Smile de volatilité normale", loc="left", pad=11)
    right.set(xlabel="Strike (%)", ylabel="Volatilité normale (bp)")
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.legend(loc="best", frameon=False, fontsize=9.5)
    _save(fig, path)


def pricing_figure(prices, as_of: str, path: Path):
    fig, ax = plt.subplots(figsize=(11.8, 3.8))
    fig.subplots_adjust(left=0.17, right=0.90, top=0.69, bottom=0.22)
    _heading(fig, "Pricing Bachelier à 30 ans", f"Rejeu du {as_of}  ·  Notionnel de référence 10 M€  ·  Strike 3 %")
    labels, values = [], []
    for record in prices:
        price = record["resultat"]
        labels.append(f'{price["product_type"].capitalize()} {price["index_tenor"]}')
        values.append(price["price"] / 1_000_000)
    palette = [COLORS["3M"], "#237c8e", COLORS["6M"]]
    bars = ax.barh(labels, values, color=palette, height=0.57)
    for bar, value in zip(bars, values, strict=True):
        label = f"{value:.3f} M€".replace(".", ",")
        ax.text(value + 0.04, bar.get_y() + bar.get_height() / 2, label, va="center", weight="bold")
    ax.invert_yaxis()
    ax.set(xlabel="Prime actualisée (M€)", xlim=(0, max(values) * 1.23))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    _save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rapport", type=Path, help="Dossier de sorties du cas du 20 novembre")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    data = _load(args.rapport, "surfaces.json")
    metadata = _load(args.rapport, "calibration.json")
    prices = _load(args.rapport, "pricing.json")
    if len(prices) != 3 or any("resultat" not in item for item in prices):
        parser.error("Le rapport doit contenir les trois contrats de examples/pricing.json.")
    as_of = "/".join(metadata["date_marche"].split("-")[::-1])
    args.destination.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        surface_figure(data, as_of, args.destination / "surfaces-30-ans.png")
        surface_3d_figures(data, as_of, args.destination)
        projection_figure(data, as_of, args.destination / "forwards-et-smile.png")
        pricing_figure(prices, as_of, args.destination / "pricing-30-ans.png")


if __name__ == "__main__":
    main()
