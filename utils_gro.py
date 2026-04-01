"""
utils_gro.py — GO-Model Analysis Utilities (GROMACS)
Roche Lab | Iowa State University

Analysis functions for GO-model folding landscape generation using GROMACS output.
Expects: .xtc trajectories, .edr energy files, GROMACS-style naming conventions.

Shared analysis core (Q, RMSD, Rg, landscapes, WHAM, bimodal) is imported from
utils_core.py — this module adds GROMACS-specific I/O and wrappers.

Usage in Colab:
    !wget https://raw.githubusercontent.com/twatchorn/utils/main/utils_gro.py
    import utils_gro as utils
"""

import os
import glob
import pathlib
import subprocess
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import constants


# ── LOGGING ───────────────────────────────────────────────────────────────────

def log(wrkdir, message):
    """Append a message to the GMPP log file."""
    with open(f'{wrkdir}/GMPP log.txt', 'a') as f:
        f.write(message + '\n')


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

    For Calpha GO-models (cut_off < 1): contact is formed if distance is within
    [0.8 * native, cut_off * native]. For all-atom simulations (cut_off >= 1):
    a hard distance cutoff (cut_off) is used as the upper bound.

    Parameters
    ----------
    traj     : md.Trajectory
    cont_file: str — path to .contacts file
    cut_off  : float — scaling factor (CA: ~1.2-1.5) or hard cutoff (AA: int nm)

    Returns
    -------
    qt : np.ndarray, shape (n_frames,)
    """
    pairs    = load_contacts(cont_file)
    natcounts = len(pairs)
    natrng   = md.compute_distances(traj[0:1], pairs)
    natmodu  = natrng * cut_off if cut_off > 1 else cut_off
    natmodl  = natrng * 0.8
    conts    = md.compute_distances(traj, pairs)
    qt       = ((conts < natmodu) & (conts > natmodl)).sum(axis=1) / natcounts
    return qt


def best_hummer_q(traj, native, cont_file):
    """
    Fraction of native contacts using the Best-Hummer-Eaton definition.
    Uses a sigmoidal switching function rather than a hard cutoff.
    Best, Hummer, and Eaton, PNAS 2013.
    """
    beta  = 50
    lam   = 1.2
    r_cut = 0.4
    pairs = load_contacts(cont_file)
    r0_all = md.compute_distances(native[0:1], pairs)
    native_mask  = np.any(r0_all < r_cut, axis=0)
    native_pairs = pairs[native_mask]
    r0 = r0_all[:, native_mask]
    r  = md.compute_distances(traj, native_pairs)
    q  = (1.0 / (1.0 + np.exp(beta * (r - lam * r0)))).mean(axis=1)
    return q, native_pairs


# ── CONTACT PROBABILITY MAP ───────────────────────────────────────────────────

def contact_probability_map(traj, cont_file, cut_off):
    """
    Compute per-contact probability across all trajectory frames
    and return as a 2D square matrix for heatmap plotting.
    """
    pairs   = load_contacts(cont_file)
    natrng  = md.compute_distances(traj[0:1], pairs)
    natmodu = natrng * cut_off if cut_off > 1 else cut_off
    natmodl = natrng * 0.8
    conts   = md.compute_distances(traj, pairs)
    prob    = ((conts < natmodu) & (conts > natmodl)).mean(axis=0)
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
    plt.axhline(np.mean(qt), color='r', linestyle='--', linewidth=1,
                label=f'Mean Q = {np.mean(qt):.3f}')
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
    """
    Select a reference frame from a trajectory.
    method : 'max_q' | 'min_rg'
    """
    if method == 'max_q':
        pairs = load_contacts(cont_file)
        mid   = traj.n_frames // 2
        r0    = md.compute_distances(traj[mid:mid+1], pairs)
        upper = r0 * 1.2
        lower = r0 * 0.8
        dists = md.compute_distances(traj, pairs)
        q     = ((dists < upper) & (dists > lower)).mean(axis=1)
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
    F    = -kT * np.log(prob)
    F   -= F.min()
    return F, x_edges, y_edges


def plot_landscape(F, x_edges, y_edges, x_label, y_label, title, save_path, cmap='viridis'):
    """Plot a 2D free energy landscape with contours."""
    fig, ax  = plt.subplots(figsize=(7, 6))
    max_fe   = np.percentile(F, 98)
    F_plot   = np.minimum(F, max_fe)
    im = ax.imshow(
        F_plot.T, origin='lower', aspect='auto',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap, vmin=0, vmax=max_fe
    )
    X  = (x_edges[:-1] + x_edges[1:]) / 2
    Y  = (y_edges[:-1] + y_edges[1:]) / 2
    Xg, Yg = np.meshgrid(X, Y)
    ax.contour(Xg, Yg, F_plot.T, levels=np.linspace(0, max_fe, 8),
               colors='white', alpha=0.6, linewidths=0.8)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='Free Energy (kT)').ax.tick_params(labelsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def landscape(wrkdir, traj_file, top_file, cont_file, folding_temp, cut_off):
    """
    Full landscape pipeline for a single temperature.
    Expects .xtc trajectory (GROMACS output).
    """
    log(wrkdir, f'Loading trajectory: {traj_file}')
    traj = md.load(traj_file, top=top_file)
    log(wrkdir, f'  {traj.n_frames} frames loaded')

    log(wrkdir, 'Computing Q(t)...')
    qt = qplot(traj, cont_file, cut_off)

    log(wrkdir, 'Computing RMSD...')
    ref_frame = get_reference_structure(traj, cont_file, method='max_q')
    rmsd = md.rmsd(traj, traj, frame=ref_frame) * 10   # nm -> Å

    log(wrkdir, 'Computing Rg...')
    rg = md.compute_rg(traj) * 10                       # nm -> Å

    plot_qt(qt, wrkdir, label=f'{folding_temp}K')

    cmap, _ = contact_probability_map(traj, cont_file, cut_off)
    plot_contact_map(cmap, wrkdir, label=f'{folding_temp}K')

    outdir = f'{wrkdir}/MDOutputFiles'
    combos = [
        (qt,   rmsd, 'Q(t)',      'RMSD (Å)',  'viridis',  'Q_RMSD'),
        (qt,   rg,   'Q(t)',      'Rg (Å)',    'magma',    'Q_Rg'),
        (rmsd, rg,   'RMSD (Å)', 'Rg (Å)',    'cividis',  'RMSD_Rg'),
    ]
    for x, y, xl, yl, cm, tag in combos:
        F, xe, ye = free_energy_2d(x, y)
        save_path = f'{outdir}/{folding_temp}K_F({tag}).png'
        plot_landscape(F, xe, ye, xl, yl, f'F({tag})  —  T = {folding_temp} K',
                       save_path, cmap=cm)
        log(wrkdir, f'  Saved: {save_path}')

    log(wrkdir, f'Landscape complete for {folding_temp}K')
    return {'Q': qt, 'RMSD': rmsd, 'Rg': rg}


# ── GROMACS ENERGY EXTRACTION ─────────────────────────────────────────────────

def extract_gromacs_energy(edr_file, output_csv, term='Potential'):
    """
    Extract an energy term from a GROMACS .edr file using gmx energy.
    Writes a CSV with columns [time, <term>].

    Parameters
    ----------
    edr_file   : str — path to .edr file
    output_csv : str — where to write the extracted CSV
    term       : str — GROMACS energy term (default: 'Potential')
    """
    xvg_tmp = output_csv.replace('.csv', '_raw.xvg')
    cmd = ['gmx', 'energy', '-f', edr_file, '-o', xvg_tmp]
    proc = subprocess.run(cmd, input=f'{term}\n0\n', text=True,
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f'gmx energy failed:\n{proc.stderr}')

    # Parse XVG → CSV (skip comment/label lines starting with # or @)
    rows = []
    with open(xvg_tmp) as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append({'time_ps': float(parts[0]),
                             'Potential Energy (kJ/mole)': float(parts[1])})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    os.remove(xvg_tmp)
    print(f'  Energy CSV written: {output_csv}  ({len(df)} frames)')
    return output_csv


# ── BIMODALITY TEST ───────────────────────────────────────────────────────────

def bimodal_check(wrkdir, csv_files):
    """
    Test whether potential energy histograms are bimodal — indicating a
    folding transition at that temperature.

    Parameters
    ----------
    wrkdir    : str
    csv_files : list of str — energy CSV files

    Returns
    -------
    (is_bimodal, folding_temp) or (False, None)
    """
    log(wrkdir, f'Bimodality test: {len(csv_files)} files')

    for file in csv_files:
        df = pd.read_csv(file)
        energy_col = [c for c in df.columns
                      if 'potential' in c.lower() or 'energy' in c.lower()]
        if not energy_col:
            log(wrkdir, f'  No energy column found in {file}, skipping')
            continue

        data = df[energy_col[0]].dropna().values
        log(wrkdir, f'  Processing {file}: {len(data)} frames')

        counts, bin_edges = np.histogram(data, bins=50, density=True)

        max_left_idx  = int(np.argmax(counts))
        max_right_idx = int(len(counts) - 1 - np.argmax(counts[::-1]))

        if max_left_idx == max_right_idx:
            is_bimodal = False
        else:
            separation   = abs(max_left_idx - max_right_idx) / len(counts)
            valley_region = counts[min(max_left_idx, max_right_idx):
                                   max(max_left_idx, max_right_idx)]
            has_valley   = valley_region.min() < 0.6 * min(
                counts[max_left_idx], counts[max_right_idx])
            is_bimodal   = (separation > 0.1) and has_valley

        if is_bimodal:
            folding_temp = pathlib.Path(file).stem.split('_')[0].replace('T', '')
            log(wrkdir, f'  Bimodal detected — preliminary Tf: {folding_temp} K')
            return True, folding_temp

    return False, None


# ── WHAM ──────────────────────────────────────────────────────────────────────

def wham(wrkdir, csv_files, tlow, thigh, n_bins=50, max_iter=1000, tol=1e-8):
    """
    Weighted Histogram Analysis Method for temperature replica data.
    Produces heat capacity curve and free energy estimates.

    Parameters
    ----------
    wrkdir    : str
    csv_files : list of str — energy CSV files, one per temperature
    tlow, thigh : float — temperature range for Cv curve
    n_bins    : int
    max_iter  : int
    tol       : float — convergence tolerance

    Returns
    -------
    dict with keys: temps, heat_caps, energy_avgs, f_k, bin_centers, degeneracy, Tf
    """
    kB = constants.Boltzmann * constants.Avogadro / 1000.0   # kJ/mol/K

    temperatures = []
    all_energies = []
    per_window   = []

    for file in sorted(csv_files):
        stem = pathlib.Path(file).stem
        try:
            temp = float(''.join(
                c for c in stem.split('_')[0] if c.isdigit() or c == '.'))
        except ValueError:
            log(wrkdir, f'  Could not parse temperature from {file}, skipping')
            continue

        df = pd.read_csv(file)
        energy_col = [c for c in df.columns
                      if 'potential' in c.lower() or 'energy' in c.lower()]
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
    N_k    = np.array([len(e) for e in per_window])

    all_energies = np.array(all_energies)
    e_min, e_max = all_energies.min(), all_energies.max()
    bin_edges    = np.linspace(e_min, e_max, n_bins + 1)
    bin_centers  = (bin_edges[:-1] + bin_edges[1:]) / 2

    histograms = np.array([
        np.histogram(e, bins=bin_edges)[0].astype(float)
        for e in per_window
    ])

    log(wrkdir, f'WHAM: {len(temperatures)} windows, {n_bins} bins, '
                f'E range [{e_min:.1f}, {e_max:.1f}] kJ/mol')

    f_k = np.zeros(len(temperatures))
    for iteration in range(max_iter):
        f_k_old = f_k.copy()
        bias    = np.outer(beta_k - beta_k[0], bin_centers)
        numerator   = histograms.sum(axis=0)
        denominator = np.maximum(
            (N_k[:, None] * np.exp(f_k[:, None] - bias)).sum(axis=0), 1e-300)
        rho    = numerator / denominator
        weight = rho[None, :] * np.exp(-bias)
        sum_terms = weight.sum(axis=1)
        valid  = (N_k > 0) & (sum_terms > 0)
        f_k[valid] = -np.log(sum_terms[valid])
        f_k -= f_k[0]
        if np.allclose(f_k, f_k_old, atol=tol):
            log(wrkdir, f'  WHAM converged after {iteration+1} iterations')
            break
    else:
        log(wrkdir, '  WHAM did not converge — results may be approximate')

    bin_width  = bin_centers[1] - bin_centers[0]
    beta_ref   = 1.0 / (kB * temperatures[0])
    degeneracy = rho * np.exp(beta_ref * bin_centers + f_k[0])

    temp_range  = np.linspace(tlow * 0.9, thigh * 1.1, 300)
    heat_caps, energy_avgs = [], []

    for T in temp_range:
        beta  = 1.0 / (kB * T)
        bfact = np.exp(-beta * bin_centers)
        Z     = np.sum(degeneracy * bfact) * bin_width
        if Z > 0:
            prob   = degeneracy * bfact / Z
            E_avg  = np.sum(bin_centers * prob) * bin_width
            E2_avg = np.sum(bin_centers**2 * prob) * bin_width
            Cv     = (E2_avg - E_avg**2) / (kB * T**2)
        else:
            E_avg, Cv = 0.0, 0.0
        energy_avgs.append(E_avg)
        heat_caps.append(Cv)

    temp_range  = np.array(temp_range)
    heat_caps   = np.array(heat_caps)
    energy_avgs = np.array(energy_avgs)

    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(temp_range, energy_avgs, 'b-', linewidth=2, label='WHAM <E>')
    axes[0].scatter(temperatures, [np.mean(e) for e in per_window],
                    c='red', s=40, zorder=5, label='Sim <E>')
    axes[0].set(xlabel='Temperature (K)', ylabel='<E> (kJ/mol)',
                title='Energy vs Temperature')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    max_cv_idx = np.argmax(heat_caps)
    axes[1].plot(temp_range, heat_caps, 'g-', linewidth=2)
    axes[1].axvline(temp_range[max_cv_idx], color='r', linestyle='--',
                    label=f'Tf ≈ {temp_range[max_cv_idx]:.1f} K')
    axes[1].set(xlabel='Temperature (K)', ylabel='Cv (kJ/mol/K)',
                title='Heat Capacity')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(temp_range, heat_caps**2 / heat_caps.max(), 'orange', linewidth=2)
    axes[2].set(xlabel='Temperature (K)', ylabel='Normalized Cv²',
                title='Cooperativity Indicator')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{wrkdir}/MDOutputFiles/WHAM.png', dpi=200)
    plt.close()

    tf = temp_range[max_cv_idx]
    log(wrkdir, f'  Tf (Cv max): {tf:.1f} K')
    log(wrkdir, '  WHAM complete. Plot saved.')

    return {
        'temps': temp_range, 'heat_caps': heat_caps,
        'energy_avgs': energy_avgs, 'f_k': f_k,
        'bin_centers': bin_centers, 'degeneracy': degeneracy, 'Tf': tf,
    }


# ── FULL ANALYSIS RUN (GROMACS) ───────────────────────────────────────────────

def run_analysis(wrkdir, top_file, cont_file, cut_off, tlow, thigh):
    """
    Run the complete analysis pipeline over all temperature simulations.

    Expects GROMACS output named: traj_T{temp}.xtc and energy_T{temp}.csv
    (use extract_gromacs_energy() to convert .edr → .csv first).

    Parameters
    ----------
    wrkdir   : str — simulation output directory
    top_file : str — topology PDB path
    cont_file: str — .contacts file path
    cut_off  : float
    tlow, thigh : float — temperature range for WHAM Cv curve
    """
    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    log(wrkdir, '=' * 60)
    log(wrkdir, 'STARTING FULL ANALYSIS (GROMACS)')
    log(wrkdir, '=' * 60)

    xtc_files = sorted(glob.glob(f'{wrkdir}/traj_T*.xtc'))
    csv_files = sorted(glob.glob(f'{wrkdir}/energy_T*.csv'))

    if not xtc_files:
        log(wrkdir, 'No .xtc trajectory files found — check wrkdir and naming')
        return

    log(wrkdir, f'Found {len(xtc_files)} trajectories')

    all_results = {}
    for xtc in xtc_files:
        temp = pathlib.Path(xtc).stem.replace('traj_T', '')
        log(wrkdir, f'\n--- T = {temp} K ---')
        try:
            results = landscape(wrkdir, xtc, top_file, cont_file, temp, cut_off)
            all_results[temp] = results
        except Exception as e:
            log(wrkdir, f'  ERROR in landscape for T={temp}: {e}')

    log(wrkdir, '\n--- Bimodality Test ---')
    is_bimodal, tf_prelim = bimodal_check(wrkdir, csv_files)
    if is_bimodal:
        log(wrkdir, f'Preliminary Tf from bimodality: {tf_prelim} K')

    log(wrkdir, '\n--- WHAM ---')
    wham_results = wham(wrkdir, csv_files, tlow, thigh)
    log(wrkdir, f'Tf from Cv maximum: {wham_results["Tf"]:.1f} K')

    log(wrkdir, '\n' + '=' * 60)
    log(wrkdir, 'ANALYSIS COMPLETE')
    log(wrkdir, '=' * 60)

    return all_results, wham_results
