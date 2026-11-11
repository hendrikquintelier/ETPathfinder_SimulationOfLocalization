#!/usr/bin/env python3
"""ETpathfinder: sensitivity of two-antenna AoA to phase measurement noise.

Independent, reproducible Monte Carlo extension of the spherical-wave AoA
experiment in etpathfinder_aoa_spherical_sim_refactored.py.

Run: python etpathfinder_aoa_phase_uncertainty.py --output phase_results
Dependencies: numpy, matplotlib

Phase measurement uncertainty is assigned to the DIFFERENTIAL phase, not to
individual channels. Gaussian noise values are hypothetical sensitivity cases,
not calibrated specifications for Red Pitaya or the ETpathfinder installation.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = 299792458.0

def source_grid(length, width, step):
    edges = np.arange(0, length + step/2, step)
    if not np.isclose(edges[-1], length): edges = np.append(edges, length)
    ctr = (edges[:-1] + edges[1:]) / 2
    x, y = np.meshgrid(ctr, ctr)
    return x, y, (x <= width) | (y <= width)

def phase_difference(x, y, d, wavelength):
    r1, r2 = np.hypot(x, y), np.hypot(x, y-d)
    return 2*np.pi*(r1-r2)/wavelength

def estimate_aoa(phase, wavelength, d):
    u = phase*wavelength/(2*np.pi*d)
    # A physically impossible plane-wave phase input is an INVALID estimate.
    # Do not silently clip it to +/-90 deg.
    u = np.where(np.abs(u) <= 1+1e-12, np.clip(u, -1, 1), np.nan)
    return np.arcsin(u)

def room_plot(ax, values, x, y, valid, d, label, vmin=None, vmax=None, cmap='viridis'):
    pic = ax.pcolormesh(x, y, np.where(valid, values, np.nan), shading='nearest',
                        cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter([0,0],[0,d], s=30, marker='^', color='black', clip_on=False)
    ax.set_aspect('equal'); ax.set_xlim(0, 11); ax.set_ylim(0, 11)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title(label)
    return pic

def run(args):
    if args.spacing <= 0 or args.width <= 0 or args.length <= args.width or args.step <= 0 or args.trials < 2 or args.frequency <= 0:
        raise ValueError('Invalid geometry, frequency, step or trials')
    sigmas = [float(s) for s in args.sigmas.split(',')]
    if any(s < 0 for s in sigmas): raise ValueError('Sigma must be >= 0')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    lam = C/args.frequency
    x, y, valid = source_grid(args.length, args.width, args.step)
    phase0 = phase_difference(x, y, args.spacing, lam)
    theta_true = np.arctan2(y-args.spacing/2, x)
    theta0 = estimate_aoa(phase0, lam, args.spacing)
    valid = valid & np.isfinite(theta0)
    bias = np.degrees(np.abs(theta0-theta_true))
    rng = np.random.default_rng(args.seed)
    results = []
    detailed = {}
    # Trials are chunked so default 0.1m grid does not allocate huge arrays.
    for sigma_deg in sigmas:
        rms_num = np.zeros_like(phase0)
        exceed = np.zeros_like(phase0)
        invalid_count = np.zeros_like(phase0)
        additional_valid = np.zeros_like(phase0)
        samples_errors = []
        trials_done = 0
        for start in range(0, args.trials, 64):
            batch = min(64, args.trials-start)
            perturb = rng.normal(0, np.radians(sigma_deg), size=(batch,)+phase0.shape)
            theta = estimate_aoa(phase0[None,:,:]+perturb,lam,args.spacing)
            good = np.isfinite(theta) & valid[None,:,:]
            err = np.degrees(np.abs(theta-theta_true[None,:,:]))
            delta = np.degrees(theta-theta0[None,:,:])
            rms_num += np.nansum(np.where(good, delta**2, np.nan), axis=0)
            exceed += np.sum(good & (err > args.threshold), axis=0)
            invalid_count += np.sum((~np.isfinite(theta)) & valid[None,:,:], axis=0)
            additional_valid += np.sum(good, axis=0)
            samples_errors.append(err[:,valid].ravel())
            trials_done += batch
        with np.errstate(invalid='ignore', divide='ignore'):
            rms = np.sqrt(rms_num/additional_valid)
        p_exceed = exceed/args.trials  # invalid does NOT count as >threshold
        p_invalid = invalid_count/args.trials
        all_err = np.concatenate(samples_errors)
        all_err = all_err[np.isfinite(all_err)]
        row = dict(sigma_phase_deg=sigma_deg, trials=args.trials,
                   angle_error_median_deg=float(np.median(all_err)),
                   angle_error_p90_deg=float(np.percentile(all_err,90)),
                   fraction_above_threshold_all_trials=float(exceed[valid].sum()/(valid.sum()*args.trials)),
                   fraction_invalid=float(p_invalid[valid].mean()),
                   fraction_invalid_or_above_threshold=float((exceed[valid]+invalid_count[valid]).sum()/(valid.sum()*args.trials)),
                   mean_added_angle_rms_deg=float(np.nanmean(rms[valid])))
        results.append(row)
        detailed[sigma_deg] = (rms,p_exceed,p_invalid)
    # Main visualization: baseline geometric bias versus extra stochastic error.
    chosen = min(sigmas,key=lambda s:abs(s-args.show_sigma))
    rms, p_exceed, p_invalid = detailed[chosen]
    fig, axs = plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    images = [room_plot(axs[0],bias,x,y,valid,args.spacing,'A  Curvature-only angular error',vmin=0,cmap='magma'),
              room_plot(axs[1],rms,x,y,valid,args.spacing,f'B  Added angular RMS (phase σ={chosen:g}°)',vmin=0,cmap='viridis'),
              room_plot(axs[2],p_exceed*100,x,y,valid,args.spacing,f'C  P(error > {args.threshold:g}°), σ={chosen:g}°',vmin=0,vmax=100,cmap='magma')]
    for ax,im,label in zip(axs,images,['Angular bias [deg]','RMS increase [deg]','Monte Carlo exceedance [%]']):
        fig.colorbar(im,ax=ax,shrink=.8,label=label)
    fig.suptitle(f'ETpathfinder two-antenna AoA | {args.frequency/1e6:g} MHz | d={args.spacing:g} m | {args.trials} trials/location')
    fig.savefig(output/'aoa_phase_uncertainty_maps.png',dpi=180)
    fig.savefig(output/'aoa_phase_uncertainty_maps.pdf')
    plt.close(fig)
    fig,axs = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    s = [r['sigma_phase_deg'] for r in results]
    axs[0].plot(s,[r['angle_error_median_deg'] for r in results],'o-',label='Median total angular error')
    axs[0].plot(s,[r['angle_error_p90_deg'] for r in results],'s-',label='90th-percentile total error')
    axs[0].plot(s,[r['mean_added_angle_rms_deg'] for r in results],'^-',label='Mean added RMS vs noiseless estimate')
    axs[0].set(xlabel='Differential phase uncertainty σ [deg]',ylabel='Angular error [deg]')
    axs[0].grid(alpha=.3); axs[0].legend(fontsize=8)
    axs[1].plot(s,[100*r['fraction_invalid_or_above_threshold'] for r in results],'o-',label='Invalid OR error > threshold')
    axs[1].plot(s,[100*r['fraction_above_threshold_all_trials'] for r in results],'s--',label='Valid with error > threshold')
    axs[1].plot(s,[100*r['fraction_invalid'] for r in results],'^:',label='Invalid plane-wave estimate')
    axs[1].set(xlabel='Differential phase uncertainty σ [deg]',ylabel='Fraction of trials [%]',ylim=(0,100))
    axs[1].grid(alpha=.3); axs[1].legend(fontsize=8)
    fig.suptitle('AoA measurement sensitivity (uniform spatial grid; not a source probability model)')
    fig.savefig(output/'aoa_phase_uncertainty_sweep.png',dpi=180)
    fig.savefig(output/'aoa_phase_uncertainty_sweep.pdf')
    plt.close(fig)
    with open(output/'aoa_phase_uncertainty_summary.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
    with open(output/'aoa_phase_uncertainty_notes.txt','w',encoding='utf8') as f:
        f.write(f'Geometry: L-shaped arms {args.length}m by {args.width}m; antennas at (0,0),(0,{args.spacing})m\n')
        f.write(f'f={args.frequency/1e6:g} MHz; lambda={lam:.4f} m; grid={args.step:g}m; {valid.sum()} valid source locations\n')
        f.write(f'Monte Carlo: seed={args.seed}; {args.trials} trials/location; sigma differential phase (degrees): {sigmas}\n')
        f.write('Assumes Gaussian independent additive phase error for each trial and position; no reflections or calibration biases.\n')
        f.write('Crucial: invalid estimates are counted separately, not silently clipped or scored as angle errors.\n')
        f.write('P(error>threshold) counts ONLY valid estimates; invalid-or-exceed counts both failures.\n')
        f.write('Phase values deliberately remain unwrapped; baseline noise-free phase is inside unambiguous interval (d<lambda/2).\n')
        f.write('Neither sigma values nor error distributions are validated hardware performance predictions.\n')
    print(f'Grid points: {valid.sum()}; selected phase uncertainty: {chosen:g} deg; outputs: {output.resolve()}')
    for r in results:
        print(f"sigma={r['sigma_phase_deg']:>5g}°  median={r['angle_error_median_deg']:.2f}°  p90={r['angle_error_p90_deg']:.2f}°  invalid-or->threshold={100*r['fraction_invalid_or_above_threshold']:.1f}%")

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='phase_results')
    p.add_argument('--length',type=float,default=11)
    p.add_argument('--width',type=float,default=2.8)
    p.add_argument('--spacing',type=float,default=2.8)
    p.add_argument('--frequency',type=float,default=14.5e6)
    p.add_argument('--step',type=float,default=.1)
    p.add_argument('--threshold',type=float,default=10)
    p.add_argument('--sigmas',default='0,1,3,5,10,20',help='Comma-separated standard deviations of differential phase noise in degrees')
    p.add_argument('--show-sigma',type=float,default=5)
    p.add_argument('--trials',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261111)
    run(p.parse_args())
