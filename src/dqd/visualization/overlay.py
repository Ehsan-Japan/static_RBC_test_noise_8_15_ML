"""
OverlayRenderer — overlays scanned cells, peaks, and rays on charge-sensor images.
Absorbs the old plot_overlay.py, utility4.py, and utility6.py.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Tuple

from ..config.figure_style import (
    LABEL_SCANNED,
    MARKER_SCANNED,
    apply_voltage_axes,
    draw_ground_truth_map,
    new_map_figure,
    save_figure,
)

Coords = List[Tuple[float, float]]


def _load_grid(npy_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a [Vx, Vy, z] .npy file and return (z_2d, Vx_values, Vy_values)."""
    array = np.load(npy_file)
    Vx_values = np.unique(array[:, 0])
    Vy_values = np.unique(array[:, 1])
    z_2d = array[:, 2].reshape((len(Vy_values), len(Vx_values)))
    return z_2d, Vx_values, Vy_values


def _collect_sample_coords(sample_dir: str) -> Optional[Tuple[Coords, Coords]]:
    """Gather all scanned cells / peaks from every voltage_coordinates.txt
    under sample_dir/cropped_results/."""
    cropped_results_dir = os.path.join(sample_dir, "cropped_results")
    if not os.path.isdir(cropped_results_dir):
        print(f"No cropped_results folder found at {cropped_results_dir}.")
        return None

    all_scanned: Coords = []
    all_peaks: Coords = []
    for root, _, files in os.walk(cropped_results_dir):
        for fname in files:
            if fname.lower() == "voltage_coordinates.txt":
                s, p = OverlayRenderer.parse_voltage_coordinates(
                    os.path.join(root, fname)
                )
                all_scanned.extend(s)
                all_peaks.extend(p)
    return all_scanned, all_peaks


class OverlayRenderer:
    """
    Generates overlay images that highlight scanned cells, detected peaks,
    and ray paths on top of charge-sensor or double-dot background images.

    All methods are stateless; pass required parameters explicitly.
    """

    # ------------------------------------------------------------------
    # Voltage-coordinate file  (shared parser)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_voltage_coordinates(file_path: str) -> Tuple[Coords, Coords]:
        """
        Parse a voltage_coordinates.txt file.

        Returns
        -------
        (scanned_cells, peaks) — each is a list of (Vx, Vy) float tuples.
        """
        with open(file_path, "r") as f:
            content = f.read()

        scanned_match = re.search(
            r"Scanned Cells\s*\(Voltage Coordinates\):\s*(.+?)"
            r"(?:Peaks\s*\(Voltage Coordinates\):|$)",
            content,
            re.DOTALL,
        )
        peaks_match = re.search(
            r"Peaks\s*\(Voltage Coordinates\):\s*(.+)", content, re.DOTALL
        )

        def _extract(coord_string: str) -> Coords:
            coords = []
            for m in re.findall(
                r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", coord_string
            ):
                try:
                    coords.append((float(m[0]), float(m[1])))
                except Exception as e:
                    print(f"Error parsing coordinate {m}: {e}")
            return coords

        scanned = _extract(scanned_match.group(1)) if scanned_match else []
        peaks = _extract(peaks_match.group(1)) if peaks_match else []
        return scanned, peaks

    # ------------------------------------------------------------------
    # Single-file voltage-coordinate overlay  (from old plot_overlay.py)
    # ------------------------------------------------------------------

    @staticmethod
    def highlight_voltage_coordinates(
        npy_file: str,
        voltage_coords_file: str,
        output_file: str,
    ) -> None:
        """
        Replot the full charge-sensor image and overlay scanned cells / peaks
        read from *voltage_coords_file*.
        """
        z_2d, Vx_values, Vy_values = _load_grid(npy_file)
        extent = [Vx_values[0], Vx_values[-1], Vy_values[0], Vy_values[-1]]

        scanned_cells, peaks = OverlayRenderer.parse_voltage_coordinates(
            voltage_coords_file
        )

        fig, ax, cax = new_map_figure(with_colorbar=True)
        im = ax.imshow(
            z_2d, extent=extent, origin="lower", aspect="auto", cmap="hot"
        )

        if scanned_cells:
            sc = np.array(scanned_cells)
            ax.scatter(sc[:, 0], sc[:, 1], color="blue", alpha=0.5, s=10,
                       label="Scanned Cells")
        if peaks:
            pk = np.array(peaks)
            ax.scatter(pk[:, 0], pk[:, 1], marker="x", color="red", s=50,
                       linewidths=1, label="Peaks")

        apply_voltage_axes(ax, extent[0], extent[1], extent[2], extent[3])
        ax.legend(loc="upper right")
        fig.colorbar(im, cax=cax, label="Sensor Signal")
        save_figure(fig, output_file)

    # ------------------------------------------------------------------
    # All-rays overlay  (from old plot_overlay.py)
    # ------------------------------------------------------------------

    @staticmethod
    def highlight_all_rays_in_sample(
        npy_file: str,
        paired_dict: Dict,
        angles: List[float],
        output_file: str,
        vxmin: float,
        vxmax: float,
        vymin: float,
        vymax: float,
    ) -> None:
        """
        Plot the background image from *npy_file* and overlay each ray's peaks
        in a different colour.

        paired_dict : {angle_str: [[vx, vy], ...], ...}
        """
        if not os.path.isfile(npy_file):
            print(f"File not found: {npy_file}. Aborting.")
            return

        z_2d, _, _ = _load_grid(npy_file)

        fig, ax, cax = new_map_figure(with_colorbar=True)
        im = ax.imshow(
            z_2d,
            extent=[vxmin, vxmax, vymin, vymax],
            origin="lower",
            aspect="auto",
            cmap="hot",
        )
        apply_voltage_axes(ax, vxmin, vxmax, vymin, vymax)

        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(angles)))
        for i, angle in enumerate(angles):
            peak_coords = np.array(paired_dict.get(str(float(angle)), []))
            if len(peak_coords) > 0:
                ax.scatter(
                    peak_coords[:, 0], peak_coords[:, 1],
                    marker="x", color=color_cycle[i], s=50, linewidths=1,
                    label=f"Ray {round(angle)}°",
                )

        fig.colorbar(im, cax=cax, label="Sensor Signal")
        ax.legend(loc="upper right")
        save_figure(fig, output_file)

    # ------------------------------------------------------------------
    # Sample-wide scanned/peaks summary txt  (utility4.py)
    # ------------------------------------------------------------------

    @staticmethod
    def highlight_all_scanned_and_peaks_in_sample_in_binary(
        sample_dir: str,
        output_file: str,
    ) -> None:
        """
        Walk *sample_dir/cropped_results/* collecting every
        voltage_coordinates.txt and write a .txt summary of all scanned cells
        and peaks (no image).  Used for binary ground-truth workflows.

        The summary is written next to *output_file* with the same base name
        and a .txt extension.
        """
        collected = _collect_sample_coords(sample_dir)
        if collected is None:
            return
        all_scanned, all_peaks = collected

        txt_path = os.path.splitext(output_file)[0] + ".txt"
        with open(txt_path, "w") as f:
            f.write("[highlight_all_scanned_and_peaks_in_sample] Summary\n")
            f.write(f"Total scanned cells: {len(all_scanned)}\n")
            f.write(f"Total peaks:         {len(all_peaks)}\n\n")
            for title, coords in (
                ("Scanned Cells", all_scanned),
                ("Peaks", all_peaks),
            ):
                f.write(f"{title} (Voltage Coordinates):\n")
                if coords:
                    for vx, vy in coords:
                        f.write(f"({vx}, {vy})\n")
                else:
                    f.write("None\n")
                f.write("\n")

    # ------------------------------------------------------------------
    # Ground-truth binary array  (utility4.py)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_ground_truth_array(
        data_path: str,
        output_npy_path: str,
    ) -> np.ndarray:
        """
        Build a binary ground-truth array from double_dot_data.npy and save it
        as a .npy file.

        The z column of *data_path* is already an exact binary transition
        label (1 = charge-state change, 0 = background), which preserves
        interdot transition lines that have no measurable charge-sensor
        response.

        Returns
        -------
        binary_array : np.ndarray of shape (num_rows, num_cols), dtype uint8
        """
        z_2d, _, _ = _load_grid(data_path)
        binary_array = (z_2d > 0.5).astype(np.uint8)
        np.save(output_npy_path, binary_array)
        return binary_array

    # ------------------------------------------------------------------
    # 2-D ground-truth grid visualisations
    # ------------------------------------------------------------------

    @staticmethod
    def _ground_truth_axes(
        file_path: str,
        vxmin: float,
        vxmax: float,
        vymin: float,
        vymax: float,
    ):
        """
        Shared setup for the ground-truth grid plots: load the binary array,
        draw it, and read the sample-wide scanned/peak coordinates.

        Returns (fig, ax, snap, scanned, peaks) where *snap* maps a list of
        (Vx, Vy) coords to the (xs, ys) of their grid-cell centres.
        """
        data = np.load(file_path)
        res_y, res_x = data.shape

        # The sweep-stage txt is optional: without it (rays-only measurement)
        # the figures simply carry no sweep-scanned cells or sweep peaks.
        sample_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
        txt_path = os.path.join(sample_dir, "all_scanned_peaks_overlay_binary.txt")
        if os.path.isfile(txt_path):
            scanned, peaks = OverlayRenderer._parse_voltage_coordinates_global(txt_path)
        else:
            scanned, peaks = [], []

        fig, ax, _ = new_map_figure()
        draw_ground_truth_map(
            ax,
            np.linspace(vxmin, vxmax, res_x + 1),
            np.linspace(vymin, vymax, res_y + 1),
            data,
        )

        cell_w = (vxmax - vxmin) / res_x
        cell_h = (vymax - vymin) / res_y

        def snap(coords: Coords) -> Tuple[List[float], List[float]]:
            xs, ys = [], []
            for x, y in coords:
                col = int(np.clip((x - vxmin) / (vxmax - vxmin) * res_x, 0, res_x - 1))
                row = int(np.clip((y - vymin) / (vymax - vymin) * res_y, 0, res_y - 1))
                xs.append(vxmin + (col + 0.5) * cell_w)
                ys.append(vymin + (row + 0.5) * cell_h)
            return xs, ys

        return fig, ax, snap, scanned, peaks

    @staticmethod
    def visualize_grid_2d_with_rays(
        file_path: str,
        vxmin: float,
        vxmax: float,
        vymin: float,
        vymax: float,
        output_file: str,
        rays_dict: Dict,
        peaks_dict: Dict,
        ray_peak_color: Optional[str] = None,
        ray_peak_size: float = 80,
        ray_peak_marker: str = "x",
        ray_peak_linewidths: float = 2,
        ray_peak_edgecolor: Optional[str] = None,
        ray_peak_label: Optional[str] = None,
        legend_fontsize: int = 8,
    ) -> None:
        """
        Ground-truth grid with ray paths, ray peaks and detected transitions.

        Parameters
        ----------
        rays_dict      : {angle_float: {"X": ..., "Y": ..., "Current": ...}}
        peaks_dict     : {angle_float: {"vx": [...], "vy": [...]}}
        ray_peak_color : if given, EVERY ray-peak cross is drawn in this single
                         colour and gets one shared legend entry.  When ``None``
                         each ray angle keeps its own tab10 colour and the
                         crosses stay out of the legend (original behaviour).
        ray_peak_size / ray_peak_marker / ray_peak_linewidths /
        ray_peak_edgecolor  : marker styling for the ray-peak crosses.
        ray_peak_label : legend label used with ``ray_peak_color``.
        legend_fontsize : font size of the legend (bigger for publication).
        """
        fig, ax, snap, scanned, peaks = OverlayRenderer._ground_truth_axes(
            file_path, vxmin, vxmax, vymin, vymax
        )

        # ---- Sweeping results ----
        if scanned:
            ax.scatter(*snap(scanned), label=LABEL_SCANNED, **MARKER_SCANNED)
        if peaks:
            ax.scatter(*snap(peaks), color="red", s=50, marker="x",
                       linewidths=1, label="Detected Transitions")

        # ---- Ray traces and ray peaks (snapped to grid cell centres) ----
        sorted_angles = sorted(rays_dict.keys())
        color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(sorted_angles), 1)))
        uniform_ray_peaks: Coords = []
        for idx, angle in enumerate(sorted_angles):
            ray_data = rays_dict[angle]
            ray_coords = list(zip(ray_data["X"].tolist(), ray_data["Y"].tolist()))
            ax.scatter(*snap(ray_coords), color="blue", s=10, alpha=0.5,
                       label="_nolegend_")
            peak_coords = list(zip(
                peaks_dict.get(angle, {}).get("vx", []),
                peaks_dict.get(angle, {}).get("vy", []),
            ))
            if not peak_coords:
                continue
            if ray_peak_color is not None:
                # One colour for all of them -> collect and draw once so
                # the legend gets a single entry.
                uniform_ray_peaks.extend(peak_coords)
            else:
                ax.scatter(*snap(peak_coords), marker=ray_peak_marker,
                           color=color_cycle[idx], s=ray_peak_size,
                           linewidths=ray_peak_linewidths,
                           zorder=6, label="_nolegend_")

        if uniform_ray_peaks:
            scatter_kw = dict(
                marker=ray_peak_marker,
                s=ray_peak_size,
                linewidths=ray_peak_linewidths,
                zorder=7,
                label=ray_peak_label or "Sweep Start Points",
            )
            if ray_peak_edgecolor is not None:
                scatter_kw.update(facecolors=ray_peak_color,
                                  edgecolors=ray_peak_edgecolor)
            else:
                scatter_kw.update(color=ray_peak_color)
            ax.scatter(*snap(uniform_ray_peaks), **scatter_kw)

        apply_voltage_axes(ax, vxmin, vxmax, vymin, vymax)

        transition_patch = Patch(facecolor="black", edgecolor="black",
                                 label="Transition Lines (Ground Truth)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[transition_patch] + handles,
                  labels=["Transition Lines (Ground Truth)"] + labels,
                  loc="upper right", fontsize=legend_fontsize)

        save_figure(fig, output_file)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_voltage_coordinates_global(
        file_path: str,
    ) -> Tuple[Coords, Coords]:
        """Line-by-line parser for files where each coordinate is on its own line."""
        scanned: Coords = []
        peaks: Coords = []
        current: Optional[Coords] = None

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Scanned Cells (Voltage Coordinates):"):
                    current = scanned
                elif line.startswith("Peaks (Voltage Coordinates):"):
                    current = peaks
                elif current is not None and line.startswith("("):
                    try:
                        x_str, y_str = line[1:-1].split(",")
                        current.append((float(x_str.strip()), float(y_str.strip())))
                    except Exception:
                        continue

        return scanned, peaks
