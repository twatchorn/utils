"""
utils.py — GO-Model Analysis Utilities
Roche Lab | Iowa State University

Consolidated analysis functions for GO-model folding landscape generation.
Covers: Q(t), RMSD, Rg, contact probability maps, WHAM, heat capacity, and 2D free energy landscapes.

Usage in Colab:
    !wget https://raw.githubusercontent.com/YOURUSERNAME/YOURREPO/main/utils.py
    import utils
"""

import os
import glob
import pathlib
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
    pairs = df.iloc[:, [1, 3]].values.astype(int) - 1  # convert to 0-indexed
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
    traj : md.Trajectory
    cont_file : str — path to .contacts file
    cut_off : float — scaling factor (Calpha: ~1.2-1.5) or hard cutoff (all-atom: int nm)

    Returns
    -------
    qt : np.ndarray, shape (n_frames,) — Q values per frame
    """
    pairs = load_contacts(cont_file)
    natcounts = len(pairs)

    # Native distances from first frame
    natrng = md.compute_distances(traj[0:1], pairs)  # (1, n_contacts)

    if cut_off > 1:
        natmodu = natrng * cut_off
    else:
        natmodu = cut_off

    natmodl = natrng * 0.8

    # All-frame distances: (n_frames, n_contacts)
    conts = md.compute_distances(traj, pairs)

    # Vectorized contact counting — replaces double Python loop
    in_contact = (conts < natmodu) & (conts > natmodl)
    qt = in_contact.sum(axis=1) / natcounts  # (n_frames,)

    return qt


def best_hummer_q(traj, native, cont_file):
    """
    Fraction of native contacts using the Best-Hummer-Eaton definition.
    Uses a sigmoidal switching function rather than a hard cutoff.

    Best, Hummer, and Eaton, PNAS 2013.

    Parameters
    ----------
    traj : md.Trajectory — trajectory to analyze
    native : md.Trajectory — native/reference structure (first frame used)
    cont_file : str — path to .contacts file

    Returns
    -------
    q : np.ndarray, shape (n_frames,) — Q values per frame
    pairs : np.ndarray — contact pairs used
    """
    beta   = 50    # 1/nm
    lam    = 1.2   # lambda scaling constant
    r_cut  = 0.4   # nm — native contact cutoff

    pairs = load_contacts(cont_file)

    # Native distances
    r0_all = md.compute_distances(native[0:1], pairs)           # (1, n_contacts)
    native_mask = np.any(r0_all < r_cut, axis=0)                # contacts within cutoff
    native_pairs = pairs[native_mask]
    r0 = r0_all[:, native_mask]                                  # (1, n_native)

    # Trajectory distances
    r = md.compute_distances(traj, native_pairs)                 # (n_frames, n_native)

    # Sigmoid switching function
    q = (1.0 / (1.0 + np.exp(beta * (r - lam * r0)))).mean(axis=1)

    return q, native_pairs


# ── CONTACT PROBABILITY MAP ───────────────────────────────────────────────────

def contact_probability_map(traj, cont_file, cut_off):
    """
    Compute the per-contact probability across all trajectory frames
    and return as a 2D square matrix for heatmap plotting.

    Parameters
    ----------
    traj : md.Trajectory
    cont_file : str
    cut_off : float

    Returns
    -------
    cmap : np.ndarray — 2D square contact probability matrix
    pairs : np.ndarray — contact pairs
    """
    pairs = load_contacts(cont_file)
    natrng = md.compute_distances(traj[0:1], pairs)

    natmodu = natrng * cut_off if cut_off > 1 else cut_off
    natmodl = natrng * 0.8

    conts = md.compute_distances(traj, pairs)
    in_contact = (conts < natmodu) & (conts > natmodl)           # (n_frames, n_contacts)

    # Mean probability per contact
    prob = in_contact.mean(axis=0)                               # (n_contacts,)

    # Build square matrix
    q_sq, _ = md.geometry.squareform(prob, pairs)
    cmap = q_sq

    return cmap, pairs


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
    """
    Select a reference frame from a trajectory.

    Parameters
    ----------
    traj : md.Trajectory
    cont_file : str
    method : str — 'max_q' | 'min_rg' | 'min_energy' (requires energies kwarg)

    Returns
    -------
    frame_idx : int
    """
    if method == 'max_q':
        pairs = load_contacts(cont_file)
        mid = traj.n_frames // 2
        r0 = md.compute_distances(traj[mid:mid+1], pairs)
        upper = r0 * 1.2
        lower = r0 * 0.8
        dists = md.compute_distances(traj, pairs)
        q = ((dists < upper) & (dists > lower)).mean(axis=1)
        return int(np.argmax(q))

    elif method == 'min_rg':
        rg = md.compute_rg(traj)
        return int(np.argmin(rg))

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'max_q' or 'min_rg'.")


# ── FREE ENERGY LANDSCAPE ────────────────────────────────────────────────────

def free_energy_2d(x, y, n_bins=50, kT=1.0):
    """
    Compute 2D free energy surface F = -kT * ln P(x, y).

    Parameters
    ----------
    x, y : array-like — reaction coordinate arrays (same length)
    n_bins : int
    kT : float — temperature in energy units (default 1.0 → output in kT)

    Returns
    -------
    F : np.ndarray (n_bins, n_bins) — free energy surface, min set to 0
    x_edges, y_edges : bin edges
    """
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    prob = hist / hist.sum()
    prob = np.maximum(prob, 1e-8)           # avoid log(0)
    F = -kT * np.log(prob)
    F -= F.min()                            # set minimum to 0
    return F, x_edges, y_edges


def plot_landscape(F, x_edges, y_edges, x_label, y_label, title, save_path, cmap='viridis'):
    """
    Plot a 2D free energy landscape with contours.

    Parameters
    ----------
    F : np.ndarray — free energy surface from free_energy_2d()
    x_edges, y_edges : bin edges
    x_label, y_label, title : str
    save_path : str — full output file path
    cmap : str — matplotlib colormap
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    max_fe = np.percentile(F, 98)           # clip extreme high-energy unsampled bins
    F_plot = np.minimum(F, max_fe)

    im = ax.imshow(
        F_plot.T, origin='lower', aspect='auto',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap, vmin=0, vmax=max_fe
    )

    # Contours
    X = (x_edges[:-1] + x_edges[1:]) / 2
    Y = (y_edges[:-1] + y_edges[1:]) / 2
    Xg, Yg = np.meshgrid(X, Y)
    levels = np.linspace(0, max_fe, 8)
    ax.contour(Xg, Yg, F_plot.T, levels=levels, colors='white', alpha=0.6, linewidths=0.8)

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax, label='Free Energy (kT)')
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def landscape(wrkdir, traj_file, top_file, cont_file, folding_temp, cut_off):
    """
    Full landscape pipeline for a single temperature:
    computes Q, RMSD, Rg then generates all 3 pairwise 2D free energy surfaces.

    Parameters
    ----------
    wrkdir : str
    traj_file : str — path to .dcd (or .xtc) trajectory
    top_file : str — path to topology PDB
    cont_file : str — path to .contacts file
    folding_temp : float or int — temperature label for output filenames
    cut_off : float
    """
    log(wrkdir, f'Loading trajectory: {traj_file}')
    traj = md.load(traj_file, top=top_file)
    log(wrkdir, f'  {traj.n_frames} frames loaded')

    # Reaction coordinates
    log(wrkdir, 'Computing Q(t)...')
    qt = qplot(traj, cont_file, cut_off)

    log(wrkdir, 'Computing RMSD...')
    ref_frame = get_reference_structure(traj, cont_file, method='max_q')
    rmsd = md.rmsd(traj, traj, frame=ref_frame) * 10   # nm -> Angstrom

    log(wrkdir, 'Computing Rg...')
    rg = md.compute_rg(traj) * 10                       # nm -> Angstrom

    # Individual timeseries plots
    plot_qt(qt, wrkdir, label=f'{folding_temp}K')

    # Contact probability map
    cmap, _ = contact_probability_map(traj, cont_file, cut_off)
    plot_contact_map(cmap, wrkdir, label=f'{folding_temp}K')

    # 2D free energy landscapes
    outdir = f'{wrkdir}/MDOutputFiles'
    combos = [
        (qt,   rmsd, 'Q(t)',      'RMSD (Å)',  'viridis',  'Q_RMSD'),
        (qt,   rg,   'Q(t)',      'Rg (Å)',    'magma',    'Q_Rg'),
        (rmsd, rg,   'RMSD (Å)', 'Rg (Å)',    'cividis',  'RMSD_Rg'),
    ]

    for x, y, xl, yl, cm, tag in combos:
        F, xe, ye = free_energy_2d(x, y)
        save_path = f'{outdir}/{folding_temp}K_F({tag}).png'
        title = f'F({tag})  —  T = {folding_temp} K'
        plot_landscape(F, xe, ye, xl, yl, title, save_path, cmap=cm)
        log(wrkdir, f'  Saved: {save_path}')

    log(wrkdir, f'Landscape complete for {folding_temp}K')

    return {'Q': qt, 'RMSD': rmsd, 'Rg': rg}


# ── BIMODALITY TEST ───────────────────────────────────────────────────────────

def bimodal_check(wrkdir, csv_files):
    """
    Test whether potential energy histograms are bimodal — indicating a
    folding transition at that temperature.

    Replaces the broken loop logic in BimodalTest.py.

    Parameters
    ----------
    wrkdir : str
    csv_files : list of str — energy CSV files (StateDataReporter output)

    Returns
    -------
    (is_bimodal, folding_temp) or (False, None)
    """
    log(wrkdir, f'Bimodality test: {len(csv_files)} files')

    for file in csv_files:
        df = pd.read_csv(file)
        # StateDataReporter CSV has a 'Potential Energy (kJ/mole)' column
        energy_col = [c for c in df.columns if 'potential' in c.lower() or 'energy' in c.lower()]
        if not energy_col:
            log(wrkdir, f'  No energy column found in {file}, skipping')
            continue

        data = df[energy_col[0]].dropna().values
        log(wrkdir, f'  Processing {file}: {len(data)} frames')

        # Build histogram
        counts, bin_edges = np.histogram(data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find global max from left and from right — fixed logic
        max_left_idx  = int(np.argmax(counts))
        max_right_idx = int(len(counts) - 1 - np.argmax(counts[::-1]))

        if max_left_idx == max_right_idx:
            is_bimodal = False
        else:
            # Peaks must be meaningfully separated (> 30% of histogram width)
            separation = abs(max_left_idx - max_right_idx) / len(counts)
            # Check for a valley between the two peaks
            valley_region = counts[min(max_left_idx, max_right_idx):max(max_left_idx, max_right_idx)]
            has_valley = valley_region.min() < 0.6 * min(counts[max_left_idx], counts[max_right_idx])
            is_bimodal = (separation > 0.1) and has_valley

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
    wrkdir : str
    csv_files : list of str — energy CSV files, one per temperature
    tlow, thigh : float — temperature range for Cv curve
    n_bins : int
    max_iter : int
    tol : float — convergence tolerance

    Returns
    -------
    dict with keys: temps, heat_caps, energy_avgs, f_k, bin_centers, degeneracy
    """
    kB = constants.Boltzmann * constants.Avogadro / 1000.0   # kJ/mol/K

    # ── Load data in a single pass ────────────────────────────────────────────
    temperatures = []
    all_energies = []
    per_window   = []

    for file in sorted(csv_files):
        # Extract temperature from filename (expects T{temp} or {temp}_...)
        stem = pathlib.Path(file).stem
        try:
            temp = float(''.join(filter(lambda c: c.isdigit() or c == '.', stem.split('_')[0])))
        except ValueError:
            log(wrkdir, f'  Could not parse temperature from {file}, skipping')
            continue

        df = pd.read_csv(file)
        energy_col = [c for c in df.columns if 'potential' in c.lower() or 'energy' in c.lower()]
        if not energy_col:
            log(wrkdir, f'  No energy column in {file}, skipping')
            continue

        energies = df[energy_col[0]].dropna().values[1:]    # skip first frame
        temperatures.append(temp)
        all_energies.extend(energies)
        per_window.append(energies)
        log(wrkdir, f'  Loaded T={temp:.0f}K: {len(energies)} frames')

    temperatures = np.array(temperatures)
    beta_k = 1.0 / (kB * temperatures)
    N_k = np.array([len(e) for e in per_window])

    # ── Build histograms ──────────────────────────────────────────────────────
    all_energies = np.array(all_energies)
    e_min, e_max = all_energies.min(), all_energies.max()
    bin_edges   = np.linspace(e_min, e_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    histograms = np.array([
        np.histogram(e, bins=bin_edges)[0].astype(float)
        for e in per_window
    ])                                                        # (n_windows, n_bins)

    log(wrkdir, f'WHAM: {len(temperatures)} windows, {n_bins} bins, E range [{e_min:.1f}, {e_max:.1f}] kJ/mol')

    # ── WHAM iteration — vectorized ───────────────────────────────────────────
    f_k = np.zeros(len(temperatures))

    for iteration in range(max_iter):
        f_k_old = f_k.copy()

        # bias terms: (n_windows, n_bins) — temperature RE has no umbrella bias
        # bias_term[k,i] = E_i * (beta_k - beta_0)
        bias = np.outer(beta_k - beta_k[0], bin_centers)     # (n_windows, n_bins)

        # Unbiased probability — vectorized
        numerator   = histograms.sum(axis=0)                  # (n_bins,)
        denominator = (N_k[:, None] * np.exp(f_k[:, None] - bias)).sum(axis=0)  # (n_bins,)
        denominator = np.maximum(denominator, 1e-300)
        rho = numerator / denominator                         # (n_bins,)

        # Update free energies — vectorized
        weight = rho[None, :] * np.exp(-bias)                 # (n_windows, n_bins)
        sum_terms = weight.sum(axis=1)                        # (n_windows,)
        valid = (N_k > 0) & (sum_terms > 0)
        f_k[valid] = -np.log(sum_terms[valid])
        f_k -= f_k[0]                                         # reference first window

        if np.allclose(f_k, f_k_old, atol=tol):
            log(wrkdir, f'  WHAM converged after {iteration+1} iterations')
            break
    else:
        log(wrkdir, '  WHAM did not converge — results may be approximate')

    # ── Density of states ────────────────────────────────────────────────────
    bin_width  = bin_centers[1] - bin_centers[0]
    ref_temp   = temperatures[0]
    beta_ref   = 1.0 / (kB * ref_temp)
    degeneracy = rho * np.exp(beta_ref * bin_centers + f_k[0])

    # ── Heat capacity ────────────────────────────────────────────────────────
    temp_range  = np.linspace(tlow * 0.9, thigh * 1.1, 300)
    heat_caps   = []
    energy_avgs = []

    for T in temp_range:
        beta   = 1.0 / (kB * T)
        bfact  = np.exp(-beta * bin_centers)
        Z      = np.sum(degeneracy * bfact) * bin_width

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

    # ── Plots ────────────────────────────────────────────────────────────────
    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(temp_range, energy_avgs, 'b-', linewidth=2, label='WHAM <E>')
    sim_means = [np.mean(e) for e in per_window]
    axes[0].scatter(temperatures, sim_means, c='red', s=40, zorder=5, label='Sim <E>')
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('<E> (kJ/mol)')
    axes[0].set_title('Energy vs Temperature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(temp_range, heat_caps, 'g-', linewidth=2)
    max_cv_idx = np.argmax(heat_caps)
    axes[1].axvline(temp_range[max_cv_idx], color='r', linestyle='--',
                    label=f'Tf ≈ {temp_range[max_cv_idx]:.1f} K')
    axes[1].set_xlabel('Temperature (K)')
    axes[1].set_ylabel('Cv (kJ/mol/K)')
    axes[1].set_title('Heat Capacity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(temp_range, heat_caps**2 / heat_caps.max(), 'orange', linewidth=2)
    axes[2].set_xlabel('Temperature (K)')
    axes[2].set_ylabel('Normalized Cv²')
    axes[2].set_title('Cooperativity Indicator')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{wrkdir}/MDOutputFiles/WHAM.png', dpi=200)
    plt.close()

    tf = temp_range[max_cv_idx]
    log(wrkdir, f'  Tf (Cv max): {tf:.1f} K')
    log(wrkdir, f'  WHAM complete. Plot saved.')

    return {
        'temps':       temp_range,
        'heat_caps':   heat_caps,
        'energy_avgs': energy_avgs,
        'f_k':         f_k,
        'bin_centers': bin_centers,
        'degeneracy':  degeneracy,
        'Tf':          tf,
    }


# ── FULL ANALYSIS RUN ─────────────────────────────────────────────────────────

def run_analysis(wrkdir, top_file, cont_file, cut_off, tlow, thigh):
    """
    Run the complete analysis pipeline over all temperature simulations in wrkdir.

    Expects files named: traj_T{temp}.dcd and energy_T{temp}.csv
    (matching the go_model_pipeline.py output naming convention)

    Parameters
    ----------
    wrkdir : str — simulation output directory
    top_file : str — topology PDB path
    cont_file : str — .contacts file path
    cut_off : float
    tlow, thigh : float — temperature range for WHAM Cv curve
    """
    os.makedirs(f'{wrkdir}/MDOutputFiles', exist_ok=True)
    log(wrkdir, '='*60)
    log(wrkdir, 'STARTING FULL ANALYSIS')
    log(wrkdir, '='*60)

    dcd_files = sorted(glob.glob(f'{wrkdir}/traj_T*.dcd'))
    csv_files = sorted(glob.glob(f'{wrkdir}/energy_T*.csv'))

    if not dcd_files:
        log(wrkdir, 'No trajectory files found — check wrkdir and naming')
        return

    log(wrkdir, f'Found {len(dcd_files)} trajectories')

    # Per-temperature landscape analysis
    all_results = {}
    for dcd in dcd_files:
        temp = pathlib.Path(dcd).stem.replace('traj_T', '')
        log(wrkdir, f'\n--- T = {temp} K ---')
        try:
            results = landscape(wrkdir, dcd, top_file, cont_file, temp, cut_off)
            all_results[temp] = results
        except Exception as e:
            log(wrkdir, f'  ERROR in landscape for T={temp}: {e}')

    # Bimodality check
    log(wrkdir, '\n--- Bimodality Test ---')
    is_bimodal, tf_prelim = bimodal_check(wrkdir, csv_files)
    if is_bimodal:
        log(wrkdir, f'Preliminary Tf from bimodality: {tf_prelim} K')

    # WHAM
    log(wrkdir, '\n--- WHAM ---')
    wham_results = wham(wrkdir, csv_files, tlow, thigh)
    log(wrkdir, f'Tf from Cv maximum: {wham_results["Tf"]:.1f} K')

    log(wrkdir, '\n' + '='*60)
    log(wrkdir, 'ANALYSIS COMPLETE')
    log(wrkdir, '='*60)

    return all_results, wham_results
