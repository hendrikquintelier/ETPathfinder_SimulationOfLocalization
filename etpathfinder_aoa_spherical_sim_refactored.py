#!/usr/bin/env python3
"""ETpathfinder: two-antenna AoA error from ideal spherical-wave curvature.

Run:
    python etpathfinder_aoa_spherical_sim_refactored.py --step 0.1 --threshold 10

DEPENDENCIES: numpy, matplotlib

For human review, read the compact MATH MODEL section first. Plotting, output
and CLI code appear afterwards and do not affect the computed angles.

Coordinates: A1=(0,0), A2=(0,d), source=(x,y), x>=0. Positive angle
is counterclockwise from +x (array broadside). Delta r = r1-r2.
The phase is ideal and UNWRAPPED. Free-space scalar spherical propagation is
assumed; it is not a full electromagnetic near-field model.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ======================= MATH MODEL ========================
SPEED_OF_LIGHT_M_S = 299_792_458.0


def source_to_antenna_distances(x, y, d):
    """r1=sqrt(x²+y²), r2=sqrt(x²+(y-d)²) [m]."""
    return np.hypot(x, y), np.hypot(x, y - d)


def spherical_phase_difference(r1, r2, wavelength):
    """Delta phi = 2*pi*(r1-r2)/lambda [rad]; phase is unwrapped."""
    return (2.0 * np.pi / wavelength) * (r1 - r2)


def plane_wave_aoa_estimate(delta_phi, wavelength, d):
    """theta_hat=asin(Delta phi*lambda/(2*pi*d)) [rad].

    IMPORTANT: This is the *plane-wave* AoA estimator, deliberately applied
    to a spherical-wave phase difference to measure its model bias.
    Invalid inputs are NaN, not silently clipped to +/-90 degrees.
    """
    argument = delta_phi * wavelength / (2.0 * np.pi * d)
    # Only remove harmless floating-point overshoot at +/-1.
    eps = 1e-12
    argument = np.where(np.abs(argument) <= 1.0 + eps,
                        np.clip(argument, -1.0, 1.0), np.nan)
    return np.arcsin(argument)


def true_source_angle(x, y, d):
    """theta_true=atan2(y-d/2, x) [rad] from the antenna-array midpoint."""
    return np.arctan2(y - d / 2.0, x)


def angular_error(theta_estimated, theta_true):
    """Absolute AoA error [rad]. Both angles must share a convention."""
    return np.abs(theta_estimated - theta_true)


# ====================== GRID AND EVALUATION ============================
@dataclass(frozen=True)
class Config:
    length: float = 11.0 #in meter
    width: float = 2.8 #in meter
    spacing: float = 2.8 #in meter
    frequency: float = 14.5e6 #in Hz
    step: float = 0.1 #in meter
    threshold: float = 10.0 #in degrees

    def validate(self):
        if not (self.length > self.width > 0 and
                0 < self.spacing <= self.length and
                self.frequency > 0 and self.step > 0 and
                self.threshold >= 0):
            raise ValueError('Require length>width>0, 0<spacing<=length, '
                             'frequency>0, step>0, threshold>=0')


def make_source_grid(length, width, step):
    """Cell centres of a square grid, masked to both arms of the L."""
    edges = np.arange(0.0, length + step / 2.0, step)
    if not np.isclose(edges[-1], length):
        edges = np.append(edges, length)
    centres = (edges[:-1] + edges[1:]) / 2.0
    x, y = np.meshgrid(centres, centres)
    valid = (x <= width) | (y <= width)
    return edges, x, y, valid


def calculate_error_map(config):
    """No plotting here: compute quantities for every source-grid position."""
    config.validate()
    edges, x, y, valid = make_source_grid(config.length, config.width, config.step)
    wavelength = SPEED_OF_LIGHT_M_S / config.frequency
    r1, r2 = source_to_antenna_distances(x, y, config.spacing)
    phase = spherical_phase_difference(r1, r2, wavelength)
    estimated = plane_wave_aoa_estimate(phase, wavelength, config.spacing)
    actual = true_source_angle(x, y, config.spacing)
    error = angular_error(estimated, actual)
    # Antenna locations are on cell boundaries and therefore not sampled.
    valid = valid & np.isfinite(error)
    return dict(edges=edges, x=x, y=y, valid=valid, r1=r1, r2=r2,
                phase=phase, estimated_deg=np.degrees(estimated),
                actual_deg=np.degrees(actual), error_deg=np.degrees(error),
                wavelength=wavelength)


def summarize_error_map(data, threshold):
    """Equal weight for each grid-cell centre (not a source probability)."""
    samples = data['error_deg'][data['valid']]
    return dict(count=len(samples), median=float(np.median(samples)),
                mean=float(np.mean(samples)), p90=float(np.percentile(samples, 90)),
                maximum=float(np.max(samples)),
                percentage_above=float(100 * np.mean(samples > threshold)))


# ======================== PLOTTING ONLY ================================
def _room_axes(ax, config, *, annotate=False):
    from matplotlib.patches import Rectangle  # noqa: F401
    L, w = config.length, config.width
    ax.plot([0, L, L, w, w, 0, 0], [0, 0, w, w, L, L, 0],
            color='#232323', lw=1.9)
    ax.scatter([0, 0], [0, config.spacing], c='#05a59d', s=85,
               edgecolors='white', linewidths=1, zorder=5, clip_on=False)
    if annotate:
        for name, yy in [('A1', 0), ('A2', config.spacing)]:
            ax.annotate(name, (0, yy), (0.48, yy + 0.38), fontsize=10,
                        fontweight='bold', color='white',
                        bbox={'boxstyle': 'round,pad=0.2', 'facecolor': '#333333',
                              'edgecolor': 'none'},
                        arrowprops={'arrowstyle': '-', 'color': 'white'})
    ax.set(xlim=(-.25, L+.25), ylim=(-.25, L+.25),
           xlabel='x position (m)', ylabel='y position (m)')
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, L+.1, 2))
    ax.set_yticks(np.arange(0, L+.1, 2))


def _save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    fig.savefig(output / f'{name}.png', dpi=230)
    fig.savefig(output / f'{name}.pdf')
    plt.close(fig)


def plot_error_heatmap(data, config, output):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7.6), layout='constrained')
    masked = np.ma.masked_where(~data['valid'], data['error_deg'])
    vmax = max(35, np.ceil(np.nanmax(masked) / 5) * 5)
    mesh = ax.pcolormesh(data['edges'], data['edges'], masked,
                         cmap='magma', vmin=0, vmax=vmax,
                         shading='flat', rasterized=True)
    _room_axes(ax, config, annotate=True)
    ax.set_title('AoA bias caused by spherical wavefront curvature')
    fig.colorbar(mesh, ax=ax, shrink=.85, pad=.025).set_label(
        'Absolute angular error (degrees)')
    _save_figure(fig, output, 'aoa_spherical_error_map')


def plot_threshold_map(data, config, stats, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(8, 7.6), layout='constrained')
    classes = np.ma.masked_where(~data['valid'],
                                  (data['error_deg'] > config.threshold).astype(int))
    cmap = ListedColormap(['#a7d6c2', '#d87464'])
    ax.pcolormesh(data['edges'], data['edges'], classes, cmap=cmap,
                  norm=BoundaryNorm([-.5, .5, 1.5], 2),
                  shading='flat', rasterized=True)
    _room_axes(ax, config)
    ax.set_title(f'Source positions exceeding {config.threshold:g}° AoA error')
    handles = [Rectangle((0, 0), 1, 1, color='#a7d6c2',
                         label=f'Error ≤ {config.threshold:g}°'),
               Rectangle((0, 0), 1, 1, color='#d87464',
                         label=f'Error > {config.threshold:g}°')]
    ax.legend(handles=handles, loc='upper right', framealpha=.95)
    ax.text(.04, .03,
            f'{stats["percentage_above"]:.1f}% of evaluated positions exceed '
            f'{config.threshold:g}°', transform=ax.transAxes, fontsize=10,
            bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none',
                  'boxstyle': 'round,pad=.4'})
    _save_figure(fig, output, 'aoa_spherical_threshold_map')


def plot_error_histogram(data, config, stats, output):
    import matplotlib.pyplot as plt
    samples = data['error_deg'][data['valid']]
    fig, ax = plt.subplots(figsize=(7.7, 4.6), layout='constrained')
    ax.hist(samples, bins=np.arange(0, max(40, np.ceil(stats['maximum'])) + 1, 2),
            edgecolor='white', color='#4b7f9e')
    ax.axvline(config.threshold, color='#be554c', lw=2, ls='--',
               label=f'{config.threshold:g}° threshold')
    ax.axvline(stats['median'], color='#333333', lw=1.5, ls=':',
               label=f'Median {stats["median"]:.1f}°')
    ax.set(xlabel='Absolute angular error (degrees)', ylabel='Grid-cell count',
           title='AoA error distribution across the L-shaped room')
    ax.legend(frameon=False)
    _save_figure(fig, output, 'aoa_spherical_error_histogram')


# ========================== OUTPUT / CLI ===============================
def save_csv(data, config, output):
    with (output / 'aoa_spherical_grid_data.csv').open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['x_m', 'y_m', 'r1_m', 'r2_m', 'delta_phi_rad',
                     'estimated_angle_deg', 'true_angle_deg',
                     'absolute_error_deg', 'exceeds_threshold'])
        for j, i in zip(*np.where(data['valid'])):
            wr.writerow([data['x'][j, i], data['y'][j, i],
                         data['r1'][j, i], data['r2'][j, i],
                         data['phase'][j, i], data['estimated_deg'][j, i],
                         data['actual_deg'][j, i], data['error_deg'][j, i],
                         int(data['error_deg'][j, i] > config.threshold)])


def save_summary(config, stats, wavelength, output):
    summary = (f'ETpathfinder two-antenna ideal spherical-wave AoA bias\n'
               f'frequency = {config.frequency/1e6:g} MHz; '
               f'wavelength = {wavelength:.5f} m\n'
               f'L room: arms {config.length:g} m, width {config.width:g} m; '
               f'antennas (0,0) and (0,{config.spacing:g}) m\n'
               f'grid cell spacing = {config.step:g} m; '
               f'n positions = {stats["count"]}\n'
               f'median error = {stats["median"]:.4f} degrees\n'
               f'mean error = {stats["mean"]:.4f} degrees\n'
               f'p90 error = {stats["p90"]:.4f} degrees\n'
               f'max sampled error = {stats["maximum"]:.4f} degrees\n'
               f'positions with error > {config.threshold:g} degrees = '
               f'{stats["percentage_above"]:.3f}%\n'
               'Model: ideal unwrapped path phase; no noise/reflections; 2D geometry.\n')
    (output / 'aoa_spherical_results.txt').write_text(summary)
    print(summary)


def run(config, output):
    config.validate()
    output.mkdir(parents=True, exist_ok=True)
    data = calculate_error_map(config)
    stats = summarize_error_map(data, config.threshold)
    plot_error_heatmap(data, config, output)
    plot_threshold_map(data, config, stats, output)
    plot_error_histogram(data, config, stats, output)
    save_csv(data, config, output)
    save_summary(config, stats, data['wavelength'], output)
    return data, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=float, default=11)
    parser.add_argument('--width', type=float, default=2.8)
    parser.add_argument('--spacing', type=float, default=2.8)
    parser.add_argument('--frequency', type=float, default=14.5e6)
    parser.add_argument('--step', type=float, default=.1)
    parser.add_argument('--threshold', type=float, default=10)
    parser.add_argument('--output', type=Path, default=Path('.'))
    args = parser.parse_args()
    config = Config(args.length, args.width, args.spacing,
                    args.frequency, args.step, args.threshold)
    run(config, args.output)


if __name__ == '__main__':
    main()
