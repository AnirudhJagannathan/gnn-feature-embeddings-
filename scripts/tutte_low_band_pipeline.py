"""Comprehensive reproduction of the Tutte low-frequency pipeline.

This script implements the full routine requested in the specification:

    A.  Build a planar grid graph, compute its Laplacian spectrum and
        identify a low-frequency band.
    B.  Compute the Tutte embedding via a Dirichlet boundary value solve and
        verify its spectral concentration.
    C.  Synthesize a smooth target by applying a low-degree polynomial filter
        to the Laplacian and a linear head.
    D.  Apply diffusive polynomial message passing for a depth sweep and
        measure representation error.
    E.  Study the conditioning of the Gram matrix and the convergence speed of
        gradient descent for fitting a linear head.
    F.  Repeat the depth/conditioning analysis with non-Tutte (random)
        features as a baseline ablation.
    G.  Provide lightweight helper utilities that mirror the pseudocode in the
        request.
    H.  Emit plots and textual logs that summarize each check.

The implementation purposefully avoids optional heavy dependencies (such as
PyTorch) so that it can run in a lightweight environment using only NumPy and
SciPy, while emitting SVG plots without relying on Matplotlib.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import numpy.linalg as npla
from numpy.typing import NDArray
from scipy import linalg as sla


Array = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Section G – helper utilities mirroring the requested pseudocode scaffold.
# ---------------------------------------------------------------------------


def build_grid_graph(m: int, n: int) -> Tuple[Array, Dict[int, Tuple[int, int]], int]:
    """Construct an ``m × n`` grid graph adjacency and coordinate map."""

    n_nodes = m * n
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    idx_to_coord: Dict[int, Tuple[int, int]] = {}
    edge_count = 0
    for i in range(m):
        for j in range(n):
            idx = i * n + j
            idx_to_coord[idx] = (i, j)
            if i + 1 < m:
                nbr = (i + 1) * n + j
                adjacency[idx, nbr] = adjacency[nbr, idx] = 1.0
                edge_count += 1
            if j + 1 < n:
                nbr = i * n + (j + 1)
                adjacency[idx, nbr] = adjacency[nbr, idx] = 1.0
                edge_count += 1
    return adjacency, idx_to_coord, edge_count


def boundary_cycle_order(m: int, n: int, idx_to_coord: Dict[int, Tuple[int, int]]) -> List[int]:
    """Return the perimeter nodes ordered along the outer cycle."""

    coord_to_idx = {coord: idx for idx, coord in idx_to_coord.items()}

    order: List[int] = []
    # top edge: (0, j)
    for j in range(n):
        order.append(coord_to_idx[(0, j)])
    # right edge: (i, n-1)
    for i in range(1, m):
        order.append(coord_to_idx[(i, n - 1)])
    # bottom edge: (m-1, j) descending
    for j in range(n - 2, -1, -1):
        order.append(coord_to_idx[(m - 1, j)])
    # left edge: (i, 0) descending, skipping corners
    for i in range(m - 2, 0, -1):
        order.append(coord_to_idx[(i, 0)])
    return order


def graph_laplacian(adj: Array) -> Array:
    """Return the dense combinatorial Laplacian ``L = D - A``."""

    deg = np.sum(adj, axis=1)
    lap = np.diag(deg) - adj
    return lap


def project_low_high(
    eigvecs: Array, eigvals: Array, lambda_c: float, x: Array
) -> Tuple[Array, Array]:
    mask = eigvals <= lambda_c + 1e-12
    u_low = eigvecs[:, mask]
    u_high = eigvecs[:, ~mask]
    return u_low.T @ x, u_high.T @ x


def tutte_embedding(
    lap: Array, boundary: Sequence[int], polygon: Array
) -> Array:
    """Solve the Tutte Dirichlet problem for a given boundary embedding."""

    n = lap.shape[0]
    phi = np.zeros((n, 2), dtype=np.float64)
    boundary = np.asarray(boundary)
    polygon = np.asarray(polygon)
    phi[boundary] = polygon

    mask = np.ones(n, dtype=bool)
    mask[boundary] = False
    interior = np.nonzero(mask)[0]

    l_ii = lap[np.ix_(interior, interior)]
    l_ib = lap[np.ix_(interior, boundary)]
    rhs = -l_ib @ polygon
    sol = sla.solve(l_ii, rhs, assume_a="sym")
    phi[interior] = sol
    return phi


def energy(lap: Array, x: Array) -> float:
    return float(np.trace(x.T @ lap @ x))


def frob_norm(x: Array) -> float:
    return float(npla.norm(x, ord="fro"))


def polynomial_filter(eigvecs: Array, eigvals: Array, coeffs: Sequence[float]) -> Array:
    powers = np.vstack([eigvals ** k for k in range(len(coeffs))])
    response = np.tensordot(coeffs, powers, axes=1)
    return eigvecs @ (response[:, None] * eigvecs.T)


def apply_polynomial_filter(
    eigvecs: Array, eigvals: Array, coeffs: Sequence[float], signal: Array
) -> Array:
    response = np.zeros_like(eigvals)
    for k, beta in enumerate(coeffs):
        response += beta * (eigvals ** k)
    projected = eigvecs.T @ signal
    return eigvecs @ (response[:, None] * projected)


def diffusion_features(
    lap: Array, features: Array, alpha: float, max_depth: int
) -> List[Array]:
    n = lap.shape[0]
    identity = np.eye(n)
    propagator = identity - alpha * lap
    current = features.copy()
    cumulative = current.copy()
    outputs = [cumulative.copy()]
    for _ in range(1, max_depth + 1):
        current = propagator @ current
        cumulative = cumulative + current
        outputs.append(cumulative.copy())
    return outputs


def gram_condition_number(z: Array, rtol: float = 1e-9) -> float:
    g = z @ z.T
    evals = sla.eigh(g, eigvals_only=True)
    evals = evals[evals > rtol]
    if evals.size == 0:
        return float("inf")
    return float(evals.max() / evals.min())


def gd_linear(
    z: Array,
    y: Array,
    step: float,
    max_iters: int = 512,
    tol: float = 1e-6,
) -> Tuple[int, Array]:
    gram = z.T @ z
    lipschitz = npla.norm(gram, ord=2)
    if not (0.0 < step < 2.0 / (lipschitz + 1e-12)):
        raise ValueError("Step size outside convergence interval")

    v = np.zeros(z.shape[1], dtype=np.float64)
    history: List[float] = []
    for it in range(max_iters):
        residual = z @ v - y
        history.append(float(npla.norm(residual)))
        if history[-1] <= tol:
            return it, np.asarray(history)
        grad = z.T @ residual
        v = v - step * grad
    return max_iters, np.asarray(history)


# ---------------------------------------------------------------------------
# Minimal SVG plotting helpers (replace Matplotlib functionality)
# ---------------------------------------------------------------------------


def _svg_canvas(width: int = 640, height: int = 400) -> ET.Element:
    return ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )


def _svg_text(svg: ET.Element, x: float, y: float, text: str, size: int = 12) -> None:
    ET.SubElement(
        svg,
        "text",
        {
            "x": f"{x:.2f}",
            "y": f"{y:.2f}",
            "font-size": str(size),
            "font-family": "Arial, sans-serif",
        },
    ).text = text


def _svg_line(svg: ET.Element, x1: float, y1: float, x2: float, y2: float, color: str = "black", width: float = 1.0, dasharray: str | None = None) -> None:
    attrs = {
        "x1": f"{x1:.2f}",
        "y1": f"{y1:.2f}",
        "x2": f"{x2:.2f}",
        "y2": f"{y2:.2f}",
        "stroke": color,
        "stroke-width": f"{width:.2f}",
    }
    if dasharray is not None:
        attrs["stroke-dasharray"] = dasharray
    ET.SubElement(svg, "line", attrs)


def _svg_circle(svg: ET.Element, cx: float, cy: float, r: float, color: str = "black") -> None:
    ET.SubElement(
        svg,
        "circle",
        {
            "cx": f"{cx:.2f}",
            "cy": f"{cy:.2f}",
            "r": f"{r:.2f}",
            "fill": color,
        },
    )


def _svg_rect(svg: ET.Element, x: float, y: float, width: float, height: float, color: str) -> None:
    ET.SubElement(
        svg,
        "rect",
        {
            "x": f"{x:.2f}",
            "y": f"{y:.2f}",
            "width": f"{width:.2f}",
            "height": f"{height:.2f}",
            "fill": color,
        },
    )


def _save_svg(svg: ET.Element, path: pathlib.Path) -> None:
    tree = ET.ElementTree(svg)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def save_svg_line_plot(
    path: pathlib.Path,
    x: Sequence[float],
    y_series: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    ylog: bool = False,
) -> None:
    svg = _svg_canvas()
    width, height, margin = 640, 400, 60
    plot_w, plot_h = width - 2 * margin, height - 2 * margin

    x_arr = np.asarray(x, dtype=np.float64)
    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0

    y_processed: List[np.ndarray] = []
    for series in y_series:
        arr = np.asarray(series, dtype=np.float64)
        if ylog:
            arr = np.log10(np.clip(arr, 1e-12, None))
        y_processed.append(arr)

    y_min = min(float(arr.min()) for arr in y_processed)
    y_max = max(float(arr.max()) for arr in y_processed)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    def map_x(val: float) -> float:
        return margin + (val - x_min) / (x_max - x_min) * plot_w

    def map_y(val: float) -> float:
        return height - margin - (val - y_min) / (y_max - y_min) * plot_h

    _svg_line(svg, margin, margin, margin, height - margin)
    _svg_line(svg, margin, height - margin, width - margin, height - margin)

    for frac in np.linspace(0.0, 1.0, num=5):
        x_val = x_min + frac * (x_max - x_min)
        x_pos = map_x(x_val)
        _svg_line(svg, x_pos, height - margin, x_pos, height - margin + 5)
        _svg_text(svg, x_pos - 10, height - margin + 20, f"{x_val:.0f}")

    if ylog:
        y_ticks = np.arange(math.floor(y_min), math.ceil(y_max) + 1)
        for log_val in y_ticks:
            y_pos = map_y(log_val)
            _svg_line(svg, margin - 5, y_pos, margin, y_pos)
            label = f"1e{int(log_val)}"
            _svg_text(svg, margin - 50, y_pos + 5, label)
    else:
        for frac in np.linspace(0.0, 1.0, num=5):
            y_val = y_min + frac * (y_max - y_min)
            y_pos = map_y(y_val)
            _svg_line(svg, margin - 5, y_pos, margin, y_pos)
            _svg_text(svg, margin - 45, y_pos + 5, f"{y_val:.2f}")

    for _orig, label, color, arr in zip(y_series, labels, colors, y_processed):
        points = []
        for xv, y_log in zip(x_arr, arr):
            y_val = y_log if not ylog else y_log
            points.append((map_x(float(xv)), map_y(float(y_val))))
        d = "M " + " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        ET.SubElement(
            svg,
            "path",
            {
                "d": d,
                "fill": "none",
                "stroke": color,
                "stroke-width": "2",
            },
        )

    legend_x = width - margin - 120
    legend_y = margin
    for idx, (label, color) in enumerate(zip(labels, colors)):
        y_pos = legend_y + idx * 18
        _svg_line(svg, legend_x, y_pos, legend_x + 20, y_pos, color=color, width=2)
        _svg_text(svg, legend_x + 25, y_pos + 4, label)

    _svg_text(svg, width / 2 - 40, margin / 2, title, size=14)
    _svg_text(svg, width / 2 - 20, height - 10, xlabel)
    _svg_text(svg, 5, height / 2, ylabel)

    _save_svg(svg, path)


def save_svg_scatter(
    path: pathlib.Path,
    points: Array,
    title: str,
    color: str = "royalblue",
) -> None:
    svg = _svg_canvas(420, 420)
    width, height, margin = 420, 420, 40
    plot_w, plot_h = width - 2 * margin, height - 2 * margin

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0

    def map_point(pt: Sequence[float]) -> Tuple[float, float]:
        x = margin + (pt[0] - x_min) / (x_max - x_min) * plot_w
        y = height - margin - (pt[1] - y_min) / (y_max - y_min) * plot_h
        return x, y

    _svg_line(svg, margin, margin, margin, height - margin)
    _svg_line(svg, margin, height - margin, width - margin, height - margin)

    for pt in points:
        cx, cy = map_point(pt)
        _svg_circle(svg, cx, cy, 2.2, color=color)

    _svg_text(svg, width / 2 - 80, margin / 2, title, size=14)
    _save_svg(svg, path)


def save_svg_bar(
    path: pathlib.Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    colors: Sequence[str],
) -> None:
    svg = _svg_canvas(420, 320)
    width, height, margin = 420, 320, 60
    plot_w, plot_h = width - 2 * margin, height - 2 * margin

    y_max = max(values) * 1.05 if max(values) > 0 else 1.0

    bar_width = plot_w / len(values)
    for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = margin + idx * bar_width + bar_width * 0.1
        bar_h = 0 if y_max == 0 else (value / y_max) * plot_h
        y = height - margin - bar_h
        _svg_rect(svg, x, y, bar_width * 0.8, bar_h, color)
        _svg_text(svg, x + bar_width * 0.2, height - margin + 20, label)
        _svg_text(svg, x + bar_width * 0.2, y - 5, f"{value:.3f}")

    _svg_line(svg, margin, margin, margin, height - margin)
    _svg_line(svg, margin, height - margin, width - margin, height - margin)
    _svg_text(svg, width / 2 - 80, margin / 2, title, size=14)
    _svg_text(svg, 10, height / 2, ylabel)
    _save_svg(svg, path)

# ---------------------------------------------------------------------------
# Section A – graph generation and spectral analysis.
# ---------------------------------------------------------------------------


def section_a(
    m: int = 20,
    n: int = 20,
) -> Tuple[Array, Dict[int, Tuple[int, int]], Array, Array, Array, float]:
    adj, idx_to_coord, edge_count = build_grid_graph(m, n)
    lap = graph_laplacian(adj)
    eigvals, eigvecs = sla.eigh(lap)

    n_nodes = lap.shape[0]
    low_count = max(1, int(math.floor(0.1 * n_nodes)))
    lambda_c = eigvals[low_count - 1]
    fraction_low = low_count / n_nodes

    print("[A] Grid graph stats:")
    print(f"    Nodes: {n_nodes}, Edges: {edge_count}")
    print(f"    λ_c (10% cutoff): {lambda_c:.6f} (low band fraction {fraction_low:.2%})")

    return adj, idx_to_coord, lap, eigvals, eigvecs, lambda_c


# ---------------------------------------------------------------------------
# Section B – Tutte embedding and spectral concentration checks.
# ---------------------------------------------------------------------------


def section_b(
    lap: Array,
    eigvals: Array,
    eigvecs: Array,
    lambda_c: float,
    idx_to_coord: Dict[int, Tuple[int, int]],
    grid_shape: Tuple[int, int],
) -> Tuple[Array, Dict[str, float]]:
    m, n = grid_shape
    boundary = boundary_cycle_order(m, n, idx_to_coord)
    k = len(boundary)
    angles = np.linspace(0.0, 2.0 * np.pi, num=k, endpoint=False)
    polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    phi = tutte_embedding(lap, boundary, polygon)
    e_phi = energy(lap, phi)
    u_low_t, u_high_t = project_low_high(eigvecs, eigvals, lambda_c, phi)
    low_mass = frob_norm(u_low_t)
    high_mass = frob_norm(u_high_t)
    phi_norm = frob_norm(phi)
    tail_ratio = (high_mass ** 2) / (phi_norm ** 2)
    bound_rhs = e_phi / max(lambda_c, 1e-12)

    print("[B] Tutte embedding checks:")
    print(f"    Energy tr(ΦᵀLΦ): {e_phi:.6f}")
    print(f"    Low-band Frobenius mass: {low_mass:.6f}")
    print(f"    Tail ratio ||U_>ᵀΦ||² / ||Φ||²: {tail_ratio:.6e}")
    print(
        "    Tail bound verification: ||U_>ᵀΦ||² = "
        f"{high_mass ** 2:.6e} ≤ E(Φ)/λ_c = {bound_rhs:.6e}"
    )

    metrics = {
        "energy": e_phi,
        "low_mass": low_mass,
        "high_mass": high_mass,
        "phi_norm": phi_norm,
        "tail_ratio": tail_ratio,
        "tail_bound_rhs": bound_rhs,
    }

    return phi, metrics


# ---------------------------------------------------------------------------
# Section C – synthetic smooth target generation.
# ---------------------------------------------------------------------------


def section_c(
    lap: Array,
    eigvals: Array,
    eigvecs: Array,
    phi: Array,
    rng: np.random.Generator,
) -> Tuple[Array, Array, Dict[str, float]]:
    coeffs = np.array([1.0, -0.4, 0.08, -0.005], dtype=np.float64)
    w = rng.normal(size=phi.shape[1])
    w = w / npla.norm(w)
    phi_w = phi @ w

    response = np.zeros_like(eigvals)
    for k, beta in enumerate(coeffs):
        response += beta * (eigvals ** k)
    y = eigvecs @ (response * (eigvecs.T @ phi_w))
    noise = 1e-3 * rng.normal(size=y.shape)
    y = y + noise

    metrics = {
        "head_norm": float(npla.norm(w)),
        "target_norm": float(npla.norm(y)),
    }
    return coeffs, w, y, metrics


def low_band_ratio(vector: Array, eigvecs: Array, eigvals: Array, lambda_c: float) -> float:
    u_low, u_high = project_low_high(eigvecs, eigvals, lambda_c, vector[:, None])
    low_mass = npla.norm(u_low)
    full_norm = npla.norm(vector)
    return float(low_mass / (full_norm + 1e-12))


# ---------------------------------------------------------------------------
# Section D/E/F – depth sweep, conditioning and GD dynamics.
# ---------------------------------------------------------------------------


@dataclass
class DepthMetrics:
    depth: int
    rep_error: float
    low_projection_error: float
    gram_kappa: float
    low_mass: float
    gd_iters: int
    gd_history: Array


def depth_analysis(
    lap: Array,
    eigvals: Array,
    eigvecs: Array,
    lambda_c: float,
    base_features: Array,
    target: Array,
    y_low: Array,
    max_depth: int,
) -> List[DepthMetrics]:
    lambda_max = eigvals.max()
    alpha = 0.9 / lambda_max

    z_list = diffusion_features(lap, base_features, alpha=alpha, max_depth=max_depth)

    metrics: List[DepthMetrics] = []
    for depth, z in enumerate(z_list):
        y_hat = z @ npla.pinv(z) @ target
        rep_error = float(npla.norm(y_hat - target))

        proj_low = eigvecs[:, eigvals <= lambda_c + 1e-12] @ (
            eigvecs[:, eigvals <= lambda_c + 1e-12].T @ target
        )
        low_proj_error = float(npla.norm(proj_low - target))

        gram_kappa = gram_condition_number(z)
        low_mass = frob_norm(eigvecs[:, eigvals <= lambda_c + 1e-12].T @ z)

        gram = z @ z.T
        eigvals_gram = sla.eigh(gram, eigvals_only=True)
        positive = eigvals_gram[eigvals_gram > 1e-9]
        lipschitz = positive.max() if positive.size else 1.0
        step = 1.0 / (lipschitz + 1e-12)
        try:
            gd_iters, gd_history = gd_linear(z, target, step=step, max_iters=256)
        except ValueError:
            step = 0.5 / (lipschitz + 1e-12)
            gd_iters, gd_history = gd_linear(z, target, step=step, max_iters=256)

        metrics.append(
            DepthMetrics(
                depth=depth,
                rep_error=rep_error,
                low_projection_error=float(npla.norm(y_low - target)),
                gram_kappa=gram_kappa,
                low_mass=low_mass,
                gd_iters=gd_iters,
                gd_history=gd_history,
            )
        )

    print("[D/E] Depth sweep summary (first five depths):")
    for entry in metrics[:5]:
        print(
            f"    T={entry.depth:2d}: rep_error={entry.rep_error:.4f}, "
            f"κ(G_T)={entry.gram_kappa:.2f}, m_≤={entry.low_mass:.4f}, GD iters={entry.gd_iters}"
        )

    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers (Section H expectations)
# ---------------------------------------------------------------------------


def plot_tail_ratios(
    tutte_tail: float,
    random_tail: float,
    output_dir: pathlib.Path,
) -> None:
    save_svg_bar(
        output_dir / "plot1_tail_ratio.svg",
        labels=["Tutte", "Random"],
        values=[tutte_tail, random_tail],
        title="High-frequency tail comparison",
        ylabel="Tail energy ratio",
        colors=["#4169E1", "#FFA500"],
    )


def plot_representation_error(
    tutte_metrics: Sequence[DepthMetrics],
    random_metrics: Sequence[DepthMetrics],
    output_dir: pathlib.Path,
) -> None:
    depths = [m.depth for m in tutte_metrics]
    tutte_errors = [m.rep_error for m in tutte_metrics]
    random_errors = [m.rep_error for m in random_metrics]
    low_baseline = [tutte_metrics[0].low_projection_error for _ in depths]
    save_svg_line_plot(
        output_dir / "plot2_representation_error.svg",
        x=depths,
        y_series=[tutte_errors, random_errors, low_baseline],
        labels=["Tutte", "Random", "Low-band projection"],
        colors=["#4169E1", "#FFA500", "#808080"],
        title="Representation error vs depth",
        xlabel="Depth T",
        ylabel="Error norm (log)",
        ylog=True,
    )


def plot_condition_numbers(
    tutte_metrics: Sequence[DepthMetrics],
    random_metrics: Sequence[DepthMetrics],
    output_dir: pathlib.Path,
) -> None:
    depths = [m.depth for m in tutte_metrics]
    save_svg_line_plot(
        output_dir / "plot3_condition_numbers.svg",
        x=depths,
        y_series=[[m.gram_kappa for m in tutte_metrics], [m.gram_kappa for m in random_metrics]],
        labels=["Tutte", "Random"],
        colors=["#4169E1", "#FFA500"],
        title="Gram matrix conditioning vs depth",
        xlabel="Depth T",
        ylabel="κ(G_T) (log)",
        ylog=True,
    )


def plot_gd_histories(
    tutte_metrics: Sequence[DepthMetrics],
    random_metrics: Sequence[DepthMetrics],
    output_dir: pathlib.Path,
    depths_to_plot: Sequence[int] = (0, 2, 5, 10),
) -> None:
    series = []
    labels: List[str] = []
    colors: List[str] = []
    for metrics, label, base_color in [
        (tutte_metrics, "Tutte", "#4169E1"),
        (random_metrics, "Random", "#FFA500"),
    ]:
        for depth in depths_to_plot:
            if depth >= len(metrics):
                continue
            history = metrics[depth].gd_history
            rel = history / (history[0] + 1e-12)
            series.append(rel)
            labels.append(f"{label} T={depth}")
            colors.append(base_color)
            kappa = max(metrics[depth].gram_kappa, 1.0)
            theo = (1.0 - 1.0 / kappa) ** np.arange(rel.size)
            series.append(theo)
            labels.append(f"{label} T={depth} (theory)")
            colors.append("#000000")
    max_len = max(len(s) for s in series)
    x = list(range(max_len))
    padded_series = []
    for s in series:
        if len(s) < max_len:
            padded = np.concatenate([s, np.full(max_len - len(s), s[-1])])
        else:
            padded = s
        padded_series.append(padded)

    save_svg_line_plot(
        output_dir / "plot4_gd_speed.svg",
        x=x,
        y_series=padded_series,
        labels=labels,
        colors=colors,
        title="Gradient descent convergence",
        xlabel="Iteration",
        ylabel="Relative residual (log)",
        ylog=True,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    output_dir = pathlib.Path("diagnostics/tutte_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    adj, idx_to_coord, lap, eigvals, eigvecs, lambda_c = section_a()
    save_svg_line_plot(
        output_dir / "plot0_spectrum.svg",
        x=list(range(len(eigvals))),
        y_series=[eigvals],
        labels=["Eigenvalues"],
        colors=["#4169E1"],
        title="Laplacian spectrum (grid)",
        xlabel="Index",
        ylabel="λ",
    )

    phi, tutte_metrics = section_b(
        lap, eigvals, eigvecs, lambda_c, idx_to_coord, (20, 20)
    )
    save_svg_scatter(
        output_dir / "plot_b_tutte_embedding.svg",
        phi,
        title="Tutte embedding (20×20 grid)",
    )

    coeffs, w, y, target_metrics = section_c(lap, eigvals, eigvecs, phi, rng)
    y_low = eigvecs[:, eigvals <= lambda_c + 1e-12] @ (
        eigvecs[:, eigvals <= lambda_c + 1e-12].T @ y
    )
    tutte_tail_ratio = float(
        npla.norm(
            project_low_high(eigvecs, eigvals, lambda_c, phi)[1], ord="fro"
        )
        ** 2
        / (frob_norm(phi) ** 2)
    )

    random_features = rng.normal(size=phi.shape)
    random_features *= frob_norm(phi) / (frob_norm(random_features) + 1e-12)
    random_tail_ratio = float(
        npla.norm(
            project_low_high(eigvecs, eigvals, lambda_c, random_features)[1], ord="fro"
        )
        ** 2
        / (frob_norm(random_features) ** 2)
    )

    max_depth = 15
    tutte_depth_metrics = depth_analysis(
        lap,
        eigvals,
        eigvecs,
        lambda_c,
        phi,
        y,
        y_low,
        max_depth,
    )
    random_depth_metrics = depth_analysis(
        lap,
        eigvals,
        eigvecs,
        lambda_c,
        random_features,
        y,
        y_low,
        max_depth,
    )

    plot_tail_ratios(tutte_tail_ratio, random_tail_ratio, output_dir)
    plot_representation_error(tutte_depth_metrics, random_depth_metrics, output_dir)
    plot_condition_numbers(tutte_depth_metrics, random_depth_metrics, output_dir)
    plot_gd_histories(tutte_depth_metrics, random_depth_metrics, output_dir)

    metrics_summary = {
        "lambda_c": lambda_c,
        "fraction_low": len(eigvals[eigvals <= lambda_c + 1e-12]) / eigvals.size,
        "tutte": {
            "embedding": tutte_metrics,
            "target": target_metrics,
            "tail_ratio": tutte_tail_ratio,
            "depth": [
                {
                    "depth": m.depth,
                    "rep_error": m.rep_error,
                    "gram_kappa": m.gram_kappa,
                    "low_mass": m.low_mass,
                    "gd_iters": m.gd_iters,
                }
                for m in tutte_depth_metrics
            ],
        },
        "random": {
            "tail_ratio": random_tail_ratio,
            "depth": [
                {
                    "depth": m.depth,
                    "rep_error": m.rep_error,
                    "gram_kappa": m.gram_kappa,
                    "low_mass": m.low_mass,
                    "gd_iters": m.gd_iters,
                }
                for m in random_depth_metrics
            ],
        },
        "polynomial_coeffs": coeffs.tolist(),
        "linear_head": w.tolist(),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_summary, fh, indent=2)

    print("[Summary] Metrics saved to", output_dir / "metrics.json")
    print("           Plots saved alongside the metrics file.")


if __name__ == "__main__":
    main()