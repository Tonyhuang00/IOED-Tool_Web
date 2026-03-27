"""
models/__init__.py — Model registry and AbstractSSMModel base class.

HOW TO ADD A NEW MODEL
──────────────────────
1. Create  models/mymodel.py  with a class  MyModel(AbstractSSMModel).
2. Implement all abstract methods (extract, simulate, render_*).
3. Set class attributes:  NAME, SHORT, TOPOLOGY_CHAR.
4. Import and add to REGISTRY below.

The orchestrator (ssm_extraction.py) iterates REGISTRY; no other changes needed.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class AbstractSSMModel(ABC):
    """
    Public API that every SSM model must implement.

    Class attributes (set on subclass, not instance)
    -------------------------------------------------
    NAME          : str   Full display name  e.g. "Cheng T-topology (2022)"
    SHORT         : str   Short key used in session-state namespacing, e.g. "T"
    TOPOLOGY_CHAR : str   Passed to make_topology_fig(): "T" | "pi" | "D"
    """

    NAME: str          = ""
    SHORT: str         = ""
    TOPOLOGY_CHAR: str = ""

    # ── Extraction ────────────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def extract(cls, Y_ex1: np.ndarray, freq: np.ndarray,
                n_low: int, **kwargs) -> tuple[dict, dict]:
        """
        Extract model parameters from fully de-embedded admittance matrix Y_ex1.

        Parameters
        ----------
        Y_ex1 : ndarray (N,2,2)  De-embedded DUT admittance (after Step 1a+1b).
        freq  : ndarray (N,)     Frequency in Hz.
        n_low : int              Number of low-frequency points used for medians.
        **kwargs                 Model-specific extras (e.g. Cbcx for Cheng Step 3).

        Returns
        -------
        params : dict   Scalar extracted parameters (SI units).
        arrays : dict   Per-frequency diagnostic arrays (for plots / debug).
        """

    # ── Forward simulation ────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def simulate(cls, params: dict, freq: np.ndarray,
                 z0: float = 50.0) -> np.ndarray:
        """
        Forward-simulate S-parameters from extracted params dict.

        params must contain intrinsic + extrinsic + pad/lead values.
        Returns S[N,2,2] complex array.
        """

    # ── Streamlit UI helpers ──────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def render_step_formulas(cls):
        """
        Render LaTeX formulas for the extraction steps in Streamlit.
        Called in the UI immediately before extract() so formulas
        sit next to the calculation they describe.
        """

    @classmethod
    @abstractmethod
    def render_results_table(cls, params: dict):
        """Render a Streamlit dataframe of the extracted scalar parameters."""

    @classmethod
    @abstractmethod
    def render_formula_trace(cls):
        """
        Render a collapsible expander with the full extraction + simulation
        formula dependency chain.
        """

    @classmethod
    @abstractmethod
    def render_override_and_smith(cls, fname: str,
                                  S_raw: np.ndarray, freq: np.ndarray,
                                  z0: float, para_eff: dict,
                                  extract_result: tuple[dict, dict],
                                  **kwargs) -> np.ndarray | None:
        """
        Render a fine-tune parameter override UI and a Smith chart.

        Parameters
        ----------
        fname          : str     File namespace for session-state keys.
        S_raw          : ndarray Measured DUT S-parameters.
        freq           : ndarray Frequency array.
        z0             : float   Reference impedance.
        para_eff       : dict    Current pad parameters (pre-extraction override).
        extract_result : tuple   (params, arrays) returned by extract().
        **kwargs                 Model-specific extras.

        Returns
        -------
        S_sim : ndarray (N,2,2) or None   Forward-simulated S from override values.
        """

    @classmethod
    def get_s2p_header_params(cls, params: dict, para_eff: dict) -> dict:
        """
        Return an ordered dict of human-readable parameter descriptions
        for the .s2p comment header.  Override to customise.
        """
        out = {}
        for k in ["Cpbe","Cpce","Cpbc"]:
            out[k] = f"{para_eff.get(k,0)*1e15:.4f} fF"
        for k in ["Rpb","Rpc","Rpe"]:
            out[{"Rpb":"Rb","Rpc":"Rc","Rpe":"Re"}[k]] = f"{para_eff.get(k,0):.4f} Ω"
        for k in ["Lb","Lc","Le"]:
            out[k] = f"{para_eff.get(k,0)*1e12:.4f} pH"
        return out


# ── Registry ──────────────────────────────────────────────────────────────────
# Import order sets default display order in the UI.

from .cheng   import ChengT, ChengPi   # noqa: E402
from .degachi import Degachi           # noqa: E402

REGISTRY: dict[str, type[AbstractSSMModel]] = {
    ChengT.SHORT:  ChengT,
    ChengPi.SHORT: ChengPi,
    Degachi.SHORT: Degachi,
}

# Default selection shown on first run (can be overridden in ssm_extraction.py)
DEFAULT_SELECTION: list[str] = ["T", "pi"]
