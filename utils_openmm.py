"""
utils_openmm.py — GO-Model Analysis Utilities (OpenMM / OpenSMOG 1.2)
Roche Lab | Iowa State University

Analysis and simulation functions for GO-model folding landscape generation
using OpenMM/OpenSMOG output. Expects: .dcd trajectories + StateDataReporter CSVs.

Usage in Colab:
    !wget https://raw.githubusercontent.com/twatchorn/utils/main/utils_openmm.py
    import utils_openmm as utils
"""

import os
import glob
import pathlib
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import constants
import sys


# ── LOGGING ───────────────────────────────────────────────────────────────────

def log(wrkdir, message):
    """Append a message to the simulation log file."""
    os.makedirs(wrkdir, exist_ok=True)
    with open(f'{wrkdir}/simulation_log.txt', 'a') as f:
        f.write(message + '\n')
    print(message)


# ── CONTACT FILE LOADING ──────────────────────────────────────────────────────

def load_contacts(cont_file):
    """
    Load CA contact pairs from a SMOG .contacts file.
    Returns zero-indexed integer pair array of shape (n_contacts, 2).
    """
    df = pd.read_csv(cont_file, sep=r'\s+', skiprows=1, header=None)
    pairs = df.iloc[:, [1, 3]].values.astype(int) - 1
    return pairs


# ── Q(t) CALCULATION ──────────────────────────────────────────────────────────

def qplot(traj, cont_file, cut_off):
    """
    Calculate Q(t) — fraction of native contacts — for a trajectory.

    Parameters
    ----------
    traj      : md.Trajectory
    cont_file : str — path to .contacts file
    cut_off   : float — scaling factor (CA: ~1.2-1.5) or hard cutoff (AA: int nm)

    Returns
    -------
    qt : np.ndarray, shape (n_frames,)
    """
    pairs = load_contacts(cont_file)
    natcounts = len(pairs)
    natrng = md.compute_distances(traj[0:1], pairs)
    natmodu = natrng * cut_off if cut_off > 1 else cut_off
    natmodl = natrng * 0.8
    conts = md.compute_distances(traj, pairs)
    qt = ((conts < natmodu) & (conts > natmodl)).sum(axis=1) / natcounts
    return qt


def best_hummer_q(traj, native, cont_file):
    """
    Fraction of native contacts using the Best-Hummer-Eaton definition.
    Uses a sigmoidal switching function. Best, Hummer, and Eaton, PNAS 2013.
    """
    beta = 50
    lam = 1.2
    r_cut = 0.4
    pairs = load_contacts(cont_file)
    r0_all = md.compute_distances(native[0:1], pairs)
    native_mask = np.any(r0_all < r_cut, axis=0)
    native_pairs = pairs[native_mask]
    r0 = r0_all[:, native_mask]
    r = md.compute_distances(traj, native_pairs)
    q = (1.0 / (1.0 + np.exp(beta * (r - lam * r0)))).mean(axis=1)
    return q, native_pairs


# ── CONTACT PROBABILITY MAP ───────────────────────────────────────────────────

def contact_probability_map(traj, cont_file, cut_off):
    """
    Compute per-contact probability across all trajectory frames
    and return as a 2D square matrix.
    """
    pairs = load_contacts(cont_file)
    natrng = md.compute_distances(traj[0:1], pairs)
    natmodu = natrng * cut_off if cut_off > 1 else cut_off
    natmodl = natrng * 0.8
    conts = md.compute_distances(traj, pairs)
    prob = ((conts < natmodu) & (conts > natmodl)).mean(axis=0)
    q_sq, _ = md.geometry.squareform(prob, pairs)
    return q_sq, pairs


def plot_contact_map(cmap, wrkdir, label=''):
    """Plot and save a contact probability heatmap."""
    plt.figure(figsize=(8, 7))
    sns.heatmap(cmap, cmap='rocket_r', square=True, vmin=0, vmax=1)
    plt.gca().invert_yaxis()
    plt.xlabel('Residue #')
    plt.ylabel('Residue #')
    plt.title(f'Contact Probability Map {label}')
    plt.tight_layout()
    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    plt.savefig(f'{wrkdir}/MDOutputFiles/{label}_ContactMap.jpg', dpi=200)
    plt.close()


# ── TRAJECTORY PLOTS ──────────────────────────────────────────────────────────

def plot_qt(qt, wrkdir, label=''):
    """Plot Q(t) timeseries and save."""
    plt.figure(figsize=(10, 4))
    plt.plot(qt, linewidth=0.4, color='steelblue')
    plt.axhline(np.mean(qt), color='r', linestyle='--', linewidth=1, label=f'Mean Q = {np.mean(qt):.3f}')
    plt.ylim(0, 1)
    plt.xlabel('Frame')
    plt.ylabel('Q(t)')
    plt.title(f'Fraction of Native Contacts — {label}')
    plt.legend()
    plt.tight_layout()
    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    plt.savefig(f'{wrkdir}/MDOutputFiles/{label}_Qt.jpg', dpi=200)
    plt.close()


# ── REFERENCE STRUCTURE SELECTION ────────────────────────────────────────────

def get_reference_structure(traj, cont_file, method='max_q'):
    """Select a reference frame. method: 'max_q' | 'min_rg'"""
    if method == 'max_q':
        pairs = load_contacts(cont_file)
        mid = traj.n_frames // 2
        r0 = md.compute_distances(traj[mid:mid+1], pairs)
        dists = md.compute_distances(traj, pairs)
        q = ((dists < r0 * 1.2) & (dists > r0 * 0.8)).mean(axis=1)
        return int(np.argmax(q))
    elif method == 'min_rg':
        return int(np.argmin(md.compute_rg(traj)))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'max_q' or 'min_rg'.")


# ── FREE ENERGY LANDSCAPE ────────────────────────────────────────────────────

def free_energy_2d(x, y, n_bins=50, kT=1.0):
    """Compute 2D free energy surface F = -kT * ln P(x, y)."""
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    prob = hist / hist.sum()
    prob = np.maximum(prob, 1e-8)
    F = -kT * np.log(prob)
    F -= F.min()
    return F, x_edges, y_edges


def plot_landscape(F, x_edges, y_edges, x_label, y_label, title, save_path, cmap='viridis'):
    """Plot a 2D free energy landscape with contours."""
    fig, ax = plt.subplots(figsize=(7, 6))
    max_fe = np.percentile(F, 98)
    F_plot = np.minimum(F, max_fe)
    im = ax.imshow(F_plot.T, origin='lower', aspect='auto', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap=cmap, vmin=0, vmax=max_fe)
    X = (x_edges[:-1] + x_edges[1:]) / 2
    Y = (y_edges[:-1] + y_edges[1:]) / 2
    Xg, Yg = np.meshgrid(X, Y)
    ax.contour(Xg, Yg, F_plot.T, levels=np.linspace(0, max_fe, 8), colors='white', alpha=0.6, linewidths=0.8)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='Free Energy (kT)').ax.tick_params(labelsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def landscape(wrkdir, traj_file, top_file, cont_file, folding_temp, cut_off):
    """Full landscape pipeline for a single temperature. Expects .dcd trajectory."""
    log(wrkdir, f'Loading trajectory: {traj_file}')
    traj = md.load(traj_file, top=top_file)
    log(wrkdir, f'  {traj.n_frames} frames loaded')

    qt = qplot(traj, cont_file, cut_off)
    ref_frame = get_reference_structure(traj, cont_file, method='max_q')
    rmsd = md.rmsd(traj, traj, frame=ref_frame) * 10
    rg = md.compute_rg(traj) * 10

    plot_qt(qt, wrkdir, label=f'{folding_temp}K')
    cmap, _ = contact_probability_map(traj, cont_file, cut_off)
    plot_contact_map(cmap, wrkdir, label=f'{folding_temp}K')

    outdir = f'{wrkdir}/MDOutputFiles'
    for x, y, xl, yl, cm, tag in [(qt, rmsd, 'Q(t)', 'RMSD (A)', 'viridis', 'Q_RMSD'), (qt, rg, 'Q(t)', 'Rg (A)', 'magma', 'Q_Rg'), (rmsd, rg, 'RMSD (A)', 'Rg (A)', 'cividis', 'RMSD_Rg')]:
        F, xe, ye = free_energy_2d(x, y)
        save_path = f'{outdir}/{folding_temp}K_F({tag}).png'
        plot_landscape(F, xe, ye, xl, yl, f'F({tag}) T={folding_temp}K', save_path, cmap=cm)
        log(wrkdir, f'  Saved: {save_path}')

    log(wrkdir, f'Landscape complete for {folding_temp}K')
    return {'Q': qt, 'RMSD': rmsd, 'Rg': rg}


# ── BIMODALITY TEST ───────────────────────────────────────────────────────────

def bimodal_check(wrkdir, csv_files, plot=True):
    """
    Test whether potential energy histograms are bimodal.

    Returns
    -------
    results   : list of dicts — {file, temp, is_bimodal}
    best_temp : float or None
    """
    log(wrkdir, f'Bimodality test: {len(csv_files)} files')
    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    results = []
    best_temp = None

    for file in sorted(csv_files):
        df = pd.read_csv(file)
        if df.columns[0].startswith('#'):
            df.columns = [c.strip().lstrip('#').strip() for c in df.columns]

        energy_col = [c for c in df.columns if 'potential' in c.lower() or 'energy' in c.lower()]
        if not energy_col:
            log(wrkdir, f'  No energy column in {file}, skipping')
            continue

        data = df[energy_col[0]].dropna().values
        stem = pathlib.Path(file).stem
        try:
            temp_tag = stem.split('_')[1].lstrip('T')
            temp = float(temp_tag)
        except (IndexError, ValueError):
            temp = None

        log(wrkdir, f'  T={temp} | {len(data)} frames')

        # Skip temperatures with too few frames for a meaningful histogram
        MIN_FRAMES = 50
        if len(data) < MIN_FRAMES:
            log(wrkdir, f'  Skipping T={temp} -- fewer than {MIN_FRAMES} frames (NaN run?)')
            results.append({'file': file, 'temp': temp, 'is_bimodal': False})
            continue

        n_bins = min(50, max(10, len(data) // 5))
        counts, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        max_left_idx = int(np.argmax(counts))
        max_right_idx = int(len(counts) - 1 - np.argmax(counts[::-1]))

        if max_left_idx == max_right_idx:
            is_bimodal = False
        else:
            separation = abs(max_left_idx - max_right_idx) / len(counts)
            valley_region = counts[min(max_left_idx, max_right_idx):max(max_left_idx, max_right_idx)]
            has_valley = valley_region.min() < 0.6 * min(counts[max_left_idx], counts[max_right_idx])
            is_bimodal = (separation > 0.1) and has_valley

        if is_bimodal and best_temp is None:
            best_temp = temp
            log(wrkdir, f'  *** Bimodal detected — preliminary Tf: {temp} ***')

        results.append({'file': file, 'temp': temp, 'is_bimodal': is_bimodal})

        if plot:
            fig, ax = plt.subplots(figsize=(6, 4))
            color = 'tomato' if is_bimodal else 'steelblue'
            ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9, color=color, alpha=0.8)
            ax.axvline(bin_centers[max_left_idx], color='k', linestyle='--', linewidth=1, label='Peak L')
            if max_right_idx != max_left_idx:
                ax.axvline(bin_centers[max_right_idx], color='navy', linestyle='--', linewidth=1, label='Peak R')
            ax.set_xlabel('Potential Energy (kJ/mol)')
            ax.set_ylabel('Density')
            ax.set_title(f'PE Histogram T={temp}' + (' [BIMODAL]' if is_bimodal else ''))
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{wrkdir}/MDOutputFiles/PE_hist_T{temp}.png', dpi=150)
            plt.close()

    return results, best_temp


# ── POTENTIAL ENERGY SERIES PLOT ─────────────────────────────────────────────

def plot_pe_series(csv_files, wrkdir, label='PE_series'):
    """Plot potential energy timeseries for all temperatures on one figure."""
    fig, axes = plt.subplots(len(csv_files), 1, figsize=(10, 2.5 * len(csv_files)), sharex=False)
    if len(csv_files) == 1:
        axes = [axes]

    for ax, file in zip(axes, sorted(csv_files)):
        df = pd.read_csv(file)
        energy_col = [c for c in df.columns if 'potential' in c.lower() or 'energy' in c.lower()]
        if not energy_col:
            continue
        data = df[energy_col[0]].dropna().values
        ax.plot(data, linewidth=0.4, color='steelblue')
        ax.axhline(np.mean(data), color='r', linestyle='--', linewidth=0.8, label=f'mean={np.mean(data):.1f}')
        ax.set_ylabel('PE (kJ/mol)', fontsize=8)
        ax.set_title(pathlib.Path(file).stem, fontsize=9)
        ax.legend(fontsize=7)

    plt.xlabel('Frame')
    plt.tight_layout()
    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    plt.savefig(f'{wrkdir}/MDOutputFiles/{label}.png', dpi=150)
    plt.close()
    print(f'PE series saved: {wrkdir}/MDOutputFiles/{label}.png')


# ── WHAM ──────────────────────────────────────────────────────────────────────

def wham(wrkdir, csv_files, tlow, thigh, n_bins=50, max_iter=1000, tol=1e-8):
    """
    Weighted Histogram Analysis Method for temperature replica data.
    Returns dict: temps, heat_caps, energy_avgs, f_k, bin_centers, degeneracy, Tf
    """
    kB = constants.Boltzmann * constants.Avogadro / 1000.0

    temperatures = []
    all_energies = []
    per_window = []

    for file in sorted(csv_files):
        stem = pathlib.Path(file).stem
        try:
            temp_tag = stem.split('_')[1].lstrip('T')
            temp = float(temp_tag)
        except (IndexError, ValueError):
            try:
                temp = float(''.join(c for c in stem if c.isdigit() or c == '.'))
            except ValueError:
                log(wrkdir, f'  Could not parse temperature from {file}, skipping')
                continue

        df = pd.read_csv(file)
        energy_col = [c for c in df.columns if 'potential' in c.lower() or 'energy' in c.lower()]
        if not energy_col:
            log(wrkdir, f'  No energy column in {file}, skipping')
            continue

        energies = df[energy_col[0]].dropna().values[1:]
        temperatures.append(temp)
        all_energies.extend(energies)
        per_window.append(energies)
        log(wrkdir, f'  Loaded T={temp:.0f}K: {len(energies)} frames')

    temperatures = np.array(temperatures)
    beta_k = 1.0 / (kB * temperatures)
    N_k = np.array([len(e) for e in per_window])
    all_energies = np.array(all_energies)
    e_min, e_max = all_energies.min(), all_energies.max()
    bin_edges = np.linspace(e_min, e_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    histograms = np.array([np.histogram(e, bins=bin_edges)[0].astype(float) for e in per_window])

    log(wrkdir, f'WHAM: {len(temperatures)} windows, {n_bins} bins, E [{e_min:.1f}, {e_max:.1f}] kJ/mol')

    f_k = np.zeros(len(temperatures))
    for iteration in range(max_iter):
        f_k_old = f_k.copy()
        bias = np.outer(beta_k - beta_k[0], bin_centers)
        numerator = histograms.sum(axis=0)
        denominator = np.maximum((N_k[:, None] * np.exp(f_k[:, None] - bias)).sum(axis=0), 1e-300)
        rho = numerator / denominator
        weight = rho[None, :] * np.exp(-bias)
        sum_terms = weight.sum(axis=1)
        valid = (N_k > 0) & (sum_terms > 0)
        f_k[valid] = -np.log(sum_terms[valid])
        f_k -= f_k[0]
        if np.allclose(f_k, f_k_old, atol=tol):
            log(wrkdir, f'  WHAM converged after {iteration + 1} iterations')
            break
    else:
        log(wrkdir, '  WHAM did not converge — results may be approximate')

    bin_width = bin_centers[1] - bin_centers[0]
    beta_ref = 1.0 / (kB * temperatures[0])
    degeneracy = rho * np.exp(beta_ref * bin_centers + f_k[0])

    temp_range = np.linspace(tlow * 0.9, thigh * 1.1, 300)
    heat_caps, energy_avgs = [], []
    for T in temp_range:
        beta = 1.0 / (kB * T)
        bfact = np.exp(-beta * bin_centers)
        Z = np.sum(degeneracy * bfact) * bin_width
        if Z > 0:
            prob = degeneracy * bfact / Z
            E_avg = np.sum(bin_centers * prob) * bin_width
            E2_avg = np.sum(bin_centers**2 * prob) * bin_width
            Cv = (E2_avg - E_avg**2) / (kB * T**2)
        else:
            E_avg, Cv = 0.0, 0.0
        energy_avgs.append(E_avg)
        heat_caps.append(Cv)

    temp_range = np.array(temp_range)
    heat_caps = np.array(heat_caps)
    energy_avgs = np.array(energy_avgs)

    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(temp_range, energy_avgs, 'b-', linewidth=2, label='WHAM <E>')
    axes[0].scatter(temperatures, [np.mean(e) for e in per_window], c='red', s=40, zorder=5, label='Sim <E>')
    axes[0].set(xlabel='Temperature (K)', ylabel='<E> (kJ/mol)', title='Energy vs Temperature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    max_cv_idx = np.argmax(heat_caps)
    axes[1].plot(temp_range, heat_caps, 'g-', linewidth=2)
    axes[1].axvline(temp_range[max_cv_idx], color='r', linestyle='--', label=f'Tf = {temp_range[max_cv_idx]:.1f} K')
    axes[1].set(xlabel='Temperature (K)', ylabel='Cv (kJ/mol/K)', title='Heat Capacity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(temp_range, heat_caps**2 / heat_caps.max(), 'orange', linewidth=2)
    axes[2].set(xlabel='Temperature (K)', ylabel='Normalized Cv2', title='Cooperativity Indicator')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{wrkdir}/MDOutputFiles/WHAM.png', dpi=200)
    plt.close()

    tf = temp_range[max_cv_idx]
    log(wrkdir, f'  Tf (Cv max): {tf:.1f} K')
    log(wrkdir, '  WHAM complete.')

    return {'temps': temp_range, 'heat_caps': heat_caps, 'energy_avgs': energy_avgs, 'f_k': f_k, 'bin_centers': bin_centers, 'degeneracy': degeneracy, 'Tf': tf}


# ── OPENMM SIMULATION RUNNER (OpenSMOG 1.2) ──────────────────────────────────

def run_openmm_simulation(sbm, temperature_K, output_dir, n_steps=5_000_000,
                          report_interval=1000, timestep_ps=0.0005,
                          friction=1.0, platform='CPU'):
    """
    Run a single simulation at a given temperature.
    Features:
    - Completion detection: skips if a complete DCD already exists
    - Checkpoint save every 100k steps so runtime drops lose minimal progress
    - Checkpoint resume: picks up where it left off if a .chk file exists
    - Verbose stdout reporter with progress % and ns/day speed
    """
    from openmm.app import DCDReporter, StateDataReporter, CheckpointReporter
    from openmm import LangevinMiddleIntegrator, Platform, Vec3
    import openmm.unit as unit
    import gc

    os.makedirs(output_dir, exist_ok=True)
    _t_str   = f"{temperature_K:.4f}".rstrip('0').rstrip('.')
    tag      = f"T{_t_str}"
    dcd_file = os.path.join(output_dir, f"traj_{tag}.dcd")
    csv_file = os.path.join(output_dir, f"energy_{tag}.csv")
    chk_file = os.path.join(output_dir, f"checkpoint_{tag}.chk")

    # ── Completion check ─────────────────────────────────────────────────────
    expected_frames = int(n_steps / report_interval)
    if os.path.exists(dcd_file) and os.path.exists(csv_file):
        try:
            _df = pd.read_csv(csv_file)
            actual_frames = len(_df)
            if actual_frames >= int(expected_frames * 0.95):
                print(f'  T* = {temperature_K} already complete '
                      f'({actual_frames}/{expected_frames} frames) -- skipping.')
                return dcd_file, csv_file
        except Exception:
            pass

    # ── Check for partial run to resume ──────────────────────────────────────
    steps_done = 0
    resume     = False
    if os.path.exists(chk_file) and os.path.exists(csv_file):
        try:
            _df        = pd.read_csv(csv_file)
            steps_done = int(_df['Step'].iloc[-1]) if 'Step' in _df.columns else 0
            if 0 < steps_done < n_steps:
                resume = True
                pct    = steps_done / n_steps * 100
                print(f'  Checkpoint found for T* = {temperature_K} '
                      f'({pct:.1f}% complete, resuming from step {steps_done:,})')
        except Exception:
            pass

    # ── Temperature conversion ────────────────────────────────────────────────
    _kB      = 0.008314
    _epsilon = getattr(sbm, '_epsilon', 3.0)
    temperature_K_real = (temperature_K * _epsilon / _kB
                          if temperature_K < 10.0 else temperature_K)

    print(f'\n  Running T* = {temperature_K} (T = {temperature_K_real:.1f} K) ...')

    # ── Tear down existing context ────────────────────────────────────────────
    if hasattr(sbm, 'simulation') and sbm.simulation is not None:
        try:
            sbm.simulation.reporters.clear()
            del sbm.simulation.context
        except Exception:
            pass
        sbm.simulation = None
    gc.collect()

    # ── Build simulation ──────────────────────────────────────────────────────
    if hasattr(sbm, '_is_adapter'):
        box_vec    = sbm._box_vec
        integrator = LangevinMiddleIntegrator(
            temperature_K_real * unit.kelvin,
            friction / unit.picosecond,
            timestep_ps * unit.picoseconds)
        integrator.setConstraintTolerance(1e-5)
        plat = Platform.getPlatformByName(platform)
        from openmm.app import Simulation
        sim = Simulation(sbm.topology, sbm.system, integrator, plat)
        sim.context.setPositions(sbm._init_positions)
        sim.context.setPeriodicBoxVectors(*box_vec)
        if resume:
            sim.loadCheckpoint(chk_file)
            print(f'  Checkpoint loaded.')
        else:
            sim.minimizeEnergy()
            state = sim.context.getState(getPositions=True)
            pos   = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            bx    = float(box_vec[0][0])
            com   = pos.mean(axis=0)
            pos_c = pos - com + np.array([bx/2]*3)
            sim.context.setPositions(pos_c * unit.nanometer)
            sim.context.setPeriodicBoxVectors(*box_vec)
            _init_temp = max(1.0, temperature_K_real * 0.1)
            sim.context.setVelocitiesToTemperature(_init_temp * unit.kelvin)
            sim.step(1000)
            sim.context.setVelocitiesToTemperature(temperature_K_real * unit.kelvin)
        sbm.simulation = sim
    else:
        sbm.loaded      = False
        sbm.temperature = temperature_K_real
        sbm.dt          = timestep_ps
        sbm.gamma       = friction
        sbm.setup_openmm(platform=platform, precision='single',
                         integrator='lengevinMiddle')
        sbm.createSimulation()
        if resume:
            sbm.simulation.loadCheckpoint(chk_file)
            print(f'  Checkpoint loaded.')
        else:
            sbm.minimize()

    # ── Reporters ─────────────────────────────────────────────────────────────
    sbm.simulation.reporters.clear()
    _dcd_append = resume and os.path.exists(dcd_file)
    _csv_mode   = 'a' if (resume and os.path.exists(csv_file)) else 'w'

    sbm.simulation.reporters.append(
        DCDReporter(dcd_file, report_interval, append=_dcd_append))
    sbm.simulation.reporters.append(
        StateDataReporter(open(csv_file, _csv_mode), report_interval,
                          step=True, time=True, potentialEnergy=True,
                          temperature=True, separator=','))
    sbm.simulation.reporters.append(
        StateDataReporter(sys.stdout, report_interval * 10,
                          step=True, potentialEnergy=False,
                          kineticEnergy=False, totalEnergy=False,
                          temperature=False, progress=True,
                          remainingTime=True, speed=True,
                          totalSteps=n_steps, separator=','))
    # Checkpoint every 100k steps -- survives Colab runtime drops
    sbm.simulation.reporters.append(
        CheckpointReporter(chk_file, 100_000))

    # ── Run with NaN restart logic ────────────────────────────────────────────
    remaining    = n_steps - steps_done
    MAX_RESTARTS = 5
    restart      = 0

    while restart <= MAX_RESTARTS:
        try:
            sbm.simulation.step(remaining)
            break
        except Exception as exc:
            msg = str(exc)
            if 'NaN' in msg or 'nan' in msg.lower():
                restart += 1
                if restart > MAX_RESTARTS:
                    print(f'  WARNING: NaN after {MAX_RESTARTS} restarts -- '
                          f'saving partial trajectory.')
                    break
                print(f'  NaN -- restart {restart}/{MAX_RESTARTS} '
                      f'with fresh velocities...')
                if hasattr(sbm, '_is_adapter'):
                    sbm.simulation.context.setPositions(sbm._init_positions)
                    sbm.simulation.context.setPeriodicBoxVectors(*box_vec)
                    sbm.simulation.context.setVelocitiesToTemperature(
                        max(1.0, temperature_K_real * 0.1) * unit.kelvin)
                    sbm.simulation.step(1000)
                    sbm.simulation.context.setVelocitiesToTemperature(
                        temperature_K_real * unit.kelvin)
            else:
                raise

    print(f'  Done -> {dcd_file}')
    return dcd_file, csv_file


def run_temperature_screen(sbm, temperatures, output_dir, n_steps=5_000_000,
                           report_interval=1000, timestep_ps=0.0005,
                           friction=1.0, platform='CPU'):
    """Run independent simulations across a list of temperatures."""
    results = []
    for T in temperatures:
        dcd, csv = run_openmm_simulation(
            sbm, T, output_dir,
            n_steps=n_steps, report_interval=report_interval,
            timestep_ps=timestep_ps, friction=friction, platform=platform)
        results.append({'temperature_K': T, 'dcd': dcd, 'csv': csv})

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(output_dir, 'simulation_index.csv'), index=False)
    print(f'\nScreen complete. {len(results)} simulations finished.')
    return summary

def run_analysis(wrkdir, top_file, cont_file, cut_off, tlow, thigh):
    """Run complete analysis pipeline over all temperature simulations."""
    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    log(wrkdir, '=' * 60)
    log(wrkdir, 'STARTING FULL ANALYSIS (OpenMM)')
    log(wrkdir, '=' * 60)

    dcd_files = sorted(glob.glob(f'{wrkdir}/traj_T*.dcd'))
    csv_files = sorted(glob.glob(f'{wrkdir}/energy_T*.csv'))

    if not dcd_files:
        log(wrkdir, 'No .dcd trajectory files found')
        return

    log(wrkdir, f'Found {len(dcd_files)} trajectories')
    all_results = {}
    for dcd in dcd_files:
        temp = pathlib.Path(dcd).stem.replace('traj_T', '')
        log(wrkdir, f'\n--- T = {temp} K ---')
        try:
            results = landscape(wrkdir, dcd, top_file, cont_file, temp, cut_off)
            all_results[temp] = results
        except Exception as e:
            log(wrkdir, f'  ERROR in landscape for T={temp}: {e}')

    log(wrkdir, '\n--- Bimodality Test ---')
    bimodal_results, tf_prelim = bimodal_check(wrkdir, csv_files, plot=True)
    if tf_prelim:
        log(wrkdir, f'Preliminary Tf from bimodality: {tf_prelim} K')

    log(wrkdir, '\n--- WHAM ---')
    wham_results = wham(wrkdir, csv_files, tlow, thigh)
    log(wrkdir, f'Tf from Cv maximum: {wham_results["Tf"]:.1f} K')

    log(wrkdir, '\n' + '=' * 60)
    log(wrkdir, 'ANALYSIS COMPLETE')
    log(wrkdir, '=' * 60)

    return all_results, wham_results

