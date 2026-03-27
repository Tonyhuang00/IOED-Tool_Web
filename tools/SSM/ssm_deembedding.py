"""
ssm_deembedding.py — Open/Short de-embedding (Steps 1a and 1b).

References: Gao, HBT for Circuit Design, Wiley 2015, §4.2
"""
from __future__ import annotations
import numpy as np
from .ssm_core import (s_to_y, y_to_z, z_to_y,
                       open_elem_Y, short_lead_Z, safe_median)


# ── Pad matrix builders (also used by forward simulators) ─────────────────────

def build_Y_pad(p: dict, w: float) -> np.ndarray:
    """
    2×2 admittance matrix for the three pad capacitors at angular freq w.
    Accounts for extended element models stored in p (mode / extra keys).

    Circuit:  Cpbe from B-node to GND,  Cpce from C-node to GND,
              Cpbc from B-node to C-node.
    """
    Ypbe = open_elem_Y(p["Cpbe"], p.get("Cpbe_mode","None"), p.get("Cpbe_extra",0.0), w)
    Ypce = open_elem_Y(p["Cpce"], p.get("Cpce_mode","None"), p.get("Cpce_extra",0.0), w)
    Ypbc = open_elem_Y(p["Cpbc"], p.get("Cpbc_mode","None"), p.get("Cpbc_extra",0.0), w)
    return np.array([[Ypbe+Ypbc, -Ypbc],
                     [-Ypbc,  Ypce+Ypbc]])

def build_Z_ser(p: dict, w: float) -> np.ndarray:
    """
    2×2 impedance matrix for the three series leads at angular freq w.
    Accounts for optional parallel capacitance per lead (Cpar_Lb/Lc/Le).

    Using port notation:  Z11 = Zb+Ze,  Z12=Z21 = Ze,  Z22 = Zc+Ze.
    """
    Zb = short_lead_Z(p["Rpb"], p["Lb"], p.get("Cpar_Lb", 0.0), w)
    Zc = short_lead_Z(p["Rpc"], p["Lc"], p.get("Cpar_Lc", 0.0), w)
    Ze = short_lead_Z(p["Rpe"], p["Le"], p.get("Cpar_Le", 0.0), w)
    return np.array([[Zb+Ze, Ze],
                     [Ze,    Zc+Ze]])


# ── Step 1a — Open dummy → pad capacitances ───────────────────────────────────

def step1a_open(open_data, n_low_frac=0.20):
    """
    Extract pad shunt capacitances from Open dummy.
    Also returns raw conductance arrays for diagnostic plots.

    Returns
    -------
    params : dict  {Cpbe, Cpce, Cpbc}  (SI units, Farads)
    arrays : dict  {Cpbe, Cpce, Cpbc, Gpbe, Gpce, Gpbc, omega}  (per-frequency)

    Formulas [Gao §4.2]:
        Cpbe = Im(Y11_open + Y12_open) / ω
        Cpce = Im(Y22_open + Y12_open) / ω
        Cpbc = −Im(Y12_open) / ω
        Gpbe = Re(Y11_open + Y12_open)   ← nonzero only if series R or parallel G
    """
    f, S_o, z0 = open_data
    omega = 2.0*np.pi*f
    n_low = max(3, int(len(f)*n_low_frac))
    Y_o = s_to_y(S_o, z0)

    # Implementation of the formulas above ↓
    Cpbe_arr = np.imag(Y_o[:,0,0] + Y_o[:,0,1]) / omega
    Cpce_arr = np.imag(Y_o[:,1,1] + Y_o[:,0,1]) / omega
    Cpbc_arr = -np.imag(Y_o[:,0,1]) / omega
    Gpbe_arr = np.real(Y_o[:,0,0] + Y_o[:,0,1])
    Gpce_arr = np.real(Y_o[:,1,1] + Y_o[:,0,1])
    Gpbc_arr = -np.real(Y_o[:,0,1])

    params = dict(
        Cpbe=safe_median(Cpbe_arr, n_low),
        Cpce=safe_median(Cpce_arr, n_low),
        Cpbc=safe_median(Cpbc_arr, n_low),
    )
    arrays = dict(Cpbe=Cpbe_arr, Cpce=Cpce_arr, Cpbc=Cpbc_arr,
                  Gpbe=Gpbe_arr, Gpce=Gpce_arr, Gpbc=Gpbc_arr, omega=omega)
    return params, arrays


# ── Step 1b — Short dummy → lead inductances & series resistances ──────────────

def step1b_short(short_data, freq, Cpbe, Cpce, Cpbc,
                 open_data=None, n_low_frac=0.20, measured_open=True,
                 Cpbe_mode="None", Cpbe_extra=0.0,
                 Cpce_mode="None", Cpce_extra=0.0,
                 Cpbc_mode="None", Cpbc_extra=0.0):
    """
    Extract lead inductances and series resistances from Short dummy.
    The Open pad admittance is subtracted first (measured or modelled).

    Returns
    -------
    params : dict  {Le, Lb, Lc, Rpe, Rpb, Rpc}  (SI units)
    arrays : dict  {Le, Lb, Lc, Rpe, Rpb, Rpc, warnings}  (per-frequency)

    Formulas [Gao §4.2]:
        Z_corr = [Y_short − Y_open]⁻¹
        Re = Re(Z12_corr)
        Rb = Re(Z11_corr − Z12_corr)
        Rc = Re(Z22_corr − Z21_corr)    ← Note: Gao text has erratum (Z11 vs Z22)
        Le = Im(Z12_corr) / ω,   Lb = Im(Z11−Z12) / ω,   Lc = Im(Z22−Z21) / ω
    """
    _, S_s, z0 = short_data
    omega = 2.0*np.pi*freq
    N = len(freq)
    n_low = max(3, int(N*n_low_frac))
    Y_s = s_to_y(S_s, z0)

    # Build Open admittance (measured or modelled)
    if measured_open and open_data is not None:
        _, S_o, z0_o = open_data
        Y_open_eff = s_to_y(S_o, z0_o)
    else:
        Y_open_eff = np.zeros((N,2,2), dtype=complex)
        for i, w in enumerate(omega):
            Ypbe = open_elem_Y(Cpbe, Cpbe_mode, Cpbe_extra, w)
            Ypce = open_elem_Y(Cpce, Cpce_mode, Cpce_extra, w)
            Ypbc = open_elem_Y(Cpbc, Cpbc_mode, Cpbc_extra, w)
            Y_open_eff[i] = np.array([[Ypbe+Ypbc, -Ypbc],
                                       [-Ypbc, Ypce+Ypbc]])

    # Implementation of the formulas above ↓
    Z_corr = y_to_z(Y_s - Y_open_eff)
    Rpe_arr = np.real(Z_corr[:,0,1])
    Rpb_arr = np.real(Z_corr[:,0,0] - Z_corr[:,0,1])
    Rpc_arr = np.real(Z_corr[:,1,1] - Z_corr[:,1,0])
    Le_arr  = np.imag(Z_corr[:,0,1]) / omega
    Lb_arr  = np.imag(Z_corr[:,0,0] - Z_corr[:,0,1]) / omega
    Lc_arr  = np.imag(Z_corr[:,1,1] - Z_corr[:,1,0]) / omega

    Le_raw  = safe_median(Le_arr,  n_low)
    Lb_raw  = safe_median(Lb_arr,  n_low)
    Lc_raw  = safe_median(Lc_arr,  n_low)
    Rpe_raw = safe_median(Rpe_arr, n_low)
    Rpb_raw = safe_median(Rpb_arr, n_low)
    Rpc_raw = safe_median(Rpc_arr, n_low)

    NOISE = 3e-12
    warnings_list = []
    if Le_raw  < -NOISE: warnings_list.append("Le significantly negative — Open caps may over-correct.")
    if Lb_raw  < -NOISE: warnings_list.append(f"⚠️ Lb negative ({Lb_raw*1e12:.1f} pH).")
    if Lc_raw  < -NOISE: warnings_list.append(f"⚠️ Lc negative ({Lc_raw*1e12:.1f} pH).")

    params = dict(Le=Le_raw, Lb=Lb_raw, Lc=Lc_raw,
                  Rpe=Rpe_raw, Rpb=Rpb_raw, Rpc=Rpc_raw)
    arrays = dict(Le=Le_arr, Lb=Lb_arr, Lc=Lc_arr,
                  Rpe=Rpe_arr, Rpb=Rpb_arr, Rpc=Rpc_arr,
                  warnings=warnings_list)
    return params, arrays


# ── Pad peeling (used by all models) ──────────────────────────────────────────

def peel_parasitics(S_raw, freq, z0, p: dict) -> np.ndarray:
    """
    Remove Open+Short pad parasitics from DUT S-parameters.
    Returns Y_ex1 (admittance after full de-embedding), ready for model extraction.

    p must contain: Cpbe/ce/bc (+ optional _mode/_extra),
                    Lb/Lc/Le, Rpb/Rpc/Rpe (+ optional Cpar_Lb/Lc/Le).
    """
    omega = 2.0*np.pi*freq
    Y_dut = s_to_y(S_raw, z0)

    # 1. Build and subtract pad shunt admittance (Open de-embedding)
    Y_pad = np.zeros((len(freq),2,2), dtype=complex)
    for i, w in enumerate(omega):
        Y_pad[i] = build_Y_pad(p, w)
    Z1 = y_to_z(Y_dut - Y_pad)

    # 2. Build and subtract series lead impedance (Short de-embedding)
    Z_ser = np.zeros((len(freq),2,2), dtype=complex)
    for i, w in enumerate(omega):
        Z_ser[i] = build_Z_ser(p, w)

    return z_to_y(Z1 - Z_ser)   # → Y_ex1
