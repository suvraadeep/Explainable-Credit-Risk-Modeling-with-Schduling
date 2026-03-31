from __future__ import annotations
 
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import wandb
 
from river import (
    drift as river_drift,
    preprocessing as river_pp,
    tree as river_tree,
    metrics as river_metrics,
    stream as river_stream,
)
 
warnings.filterwarnings("ignore")
 
 
# ─── Drift Event dataclass ────────────────────────────────────────────────────
 
@dataclass
class DriftEvent:
    sample_index: int
    detector:     str          # "ADWIN" | "KSWIN"
    running_auc:  float
    action:       str          # "retrain" | "alert"
    timestamp:    float = field(default_factory=time.time)
 
 
# ─── DriftMonitor ─────────────────────────────────────────────────────────────
 
class DriftMonitor:
    """
    Wraps River's ADWIN and KSWIN detectors.
    Emits DriftEvent objects when drift is detected.
 
    Parameters
    ----------
    delta_adwin : ADWIN confidence (lower = more sensitive)
    alpha_kswin : KSWIN significance level
    window_size : KSWIN sliding window size
    use_adwin   : enable ADWIN detector
    use_kswin   : enable KSWIN detector
    """
 
    def __init__(
        self,
        delta_adwin:  float = 0.002,
        alpha_kswin:  float = 0.005,
        window_size:  int   = 100,
        use_adwin:    bool  = True,
        use_kswin:    bool  = True,
    ):
        self.use_adwin  = use_adwin
        self.use_kswin  = use_kswin
        self.adwin      = river_drift.ADWIN(delta=delta_adwin)  if use_adwin  else None
        self.kswin      = river_drift.KSWIN(alpha=alpha_kswin, window_size=window_size) if use_kswin else None
        self.events:    List[DriftEvent] = []
        self._n_adwin_resets = 0
        self._n_kswin_resets = 0
 
    def update(self, error: float, sample_idx: int, running_auc: float) -> Optional[DriftEvent]:
        """
        Feed one prediction error. Returns a DriftEvent if drift detected, else None.
        ADWIN takes priority; KSWIN fires if ADWIN didn't.
        """
        evt = None
 
        if self.use_adwin:
            self.adwin.update(error)
            if self.adwin.drift_detected:
                self._n_adwin_resets += 1
                evt = DriftEvent(
                    sample_index=sample_idx,
                    detector="ADWIN",
                    running_auc=running_auc,
                    action="retrain",
                )
                self.events.append(evt)
                return evt
 
        if self.use_kswin:
            self.kswin.update(error)
            if self.kswin.drift_detected:
                self._n_kswin_resets += 1
                evt = DriftEvent(
                    sample_index=sample_idx,
                    detector="KSWIN",
                    running_auc=running_auc,
                    action="alert",
                )
                self.events.append(evt)
                return evt
 
        return None
 
    @property
    def total_detections(self) -> int:
        return len(self.events)
 
    @property
    def adwin_detections(self) -> int:
        return self._n_adwin_resets
 
    @property
    def kswin_detections(self) -> int:
        return self._n_kswin_resets
 
    def summary(self) -> Dict:
        return {
            "total_detections": self.total_detections,
            "adwin_detections": self.adwin_detections,
            "kswin_detections": self.kswin_detections,
            "drift_sample_indices": [e.sample_index for e in self.events],
        }
 
 
# ─── OnlineLearner ────────────────────────────────────────────────────────────
 
class OnlineLearner:
    """
    Online learning wrapper around Hoeffding Adaptive Tree (HATR).
    Auto-retrains when DriftMonitor fires.
 
    Parameters
    ----------
    monitor : DriftMonitor instance
    grace_period_normal   : HATR grace period under normal stream
    grace_period_post_drift: HATR grace period right after drift (faster adapt)
    log_wandb : log metrics to W&B if True
    """
 
    def __init__(
        self,
        monitor:                DriftMonitor,
        grace_period_normal:    int  = 200,
        grace_period_post_drift:int  = 50,
        log_wandb:              bool = False,
        seed:                   int  = 42,
    ):
        self.monitor                  = monitor
        self.grace_period_normal      = grace_period_normal
        self.grace_period_post_drift  = grace_period_post_drift
        self.log_wandb                = log_wandb
        self.seed                     = seed
        self._build_model(grace_period_normal)
 
        self.auc_metric      = river_metrics.ROCAUC()
        self.errors:         List[float] = []
        self.running_aucs:   List[float] = []
        self.retrain_count:  int         = 0
 
    def _build_model(self, grace_period: int):
        self.pipeline = (
            river_pp.StandardScaler()
            | river_tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=grace_period,
                delta=1e-5,
                seed=self.seed,
            )
        )
 
    def run_stream(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        drift_inject_at: Optional[int] = None,
        drift_income_mult: float = 0.4,
        drift_label_noise: float = 0.12,
        drift_duration: int = 5000,
        verbose_every: int = 5000,
    ) -> Dict:
        """
        Stream all rows through the online learner.
 
        Parameters
        ----------
        drift_inject_at   : sample index to start injecting synthetic drift (None = no injection)
        drift_income_mult : income multiplier during drift window
        drift_label_noise : fraction of labels to flip during drift
        drift_duration    : how many samples the drift lasts
        verbose_every     : print progress every N samples
 
        Returns
        -------
        results dict with all tracked metrics
        """
        print(f"🌊 Streaming {len(X):,} samples through online learner...")
        if drift_inject_at:
            print(f"   Synthetic drift will be injected at sample {drift_inject_at:,} "
                  f"for {drift_duration:,} samples (income×{drift_income_mult})")
 
        start = time.time()
        income_col = "AMT_INCOME_TOTAL" if "AMT_INCOME_TOTAL" in X.columns else None
 
        for i, (xi, yi) in enumerate(river_stream.iter_pandas(X, y)):
 
            # ── Optional: synthetic drift injection ─────────────────────
            if drift_inject_at and drift_inject_at <= i < drift_inject_at + drift_duration:
                xi = dict(xi)
                if income_col:
                    xi[income_col] = xi[income_col] * drift_income_mult
                if np.random.random() < drift_label_noise:
                    yi = 1 - yi
 
            # ── Predict ─────────────────────────────────────────────────
            y_prob = self.pipeline.predict_proba_one(xi)
            p1     = y_prob.get(1, 0.5)
 
            # ── Update metric ────────────────────────────────────────────
            self.auc_metric.update(yi, p1)
            current_auc = self.auc_metric.get()
            self.running_aucs.append(current_auc)
 
            error = abs(yi - p1)
            self.errors.append(error)
 
            # ── Drift detection ──────────────────────────────────────────
            evt = self.monitor.update(error, i, current_auc)
            if evt is not None:
                self.retrain_count += 1
                gp = self.grace_period_post_drift if evt.action == "retrain" else self.grace_period_normal
                self._build_model(gp)
 
                if self.retrain_count <= 8:
                    print(f"  🚨 [{evt.detector}] Drift @ sample {i:,} | "
                          f"AUC: {current_auc:.4f} | Retrain #{self.retrain_count}")
 
                if self.log_wandb:
                    wandb.log({
                        "online/drift_detected_at": i,
                        "online/detector": evt.detector,
                        "online/auc_at_drift": current_auc,
                        "online/retrain_count": self.retrain_count,
                    })
 
            # ── Learn ────────────────────────────────────────────────────
            self.pipeline.learn_one(xi, yi)
 
            # ── Periodic logging ─────────────────────────────────────────
            if (i + 1) % verbose_every == 0:
                elapsed = time.time() - start
                print(f"  [{i+1:>7,}] AUC={current_auc:.4f} | "
                      f"Drifts={self.monitor.total_detections} | "
                      f"Elapsed={elapsed:.0f}s")
                if self.log_wandb:
                    wandb.log({"online/auc": current_auc, "online/sample": i + 1})
 
        elapsed = time.time() - start
        results = {
            "final_auc":         current_auc,
            "total_samples":     len(X),
            "elapsed_seconds":   elapsed,
            "throughput":        len(X) / elapsed,
            "total_retrains":    self.retrain_count,
            **self.monitor.summary(),
        }
        print(f"\n✅ Stream complete | Final AUC: {current_auc:.5f} | "
              f"Drift events: {self.monitor.total_detections} | "
              f"Time: {elapsed:.1f}s | Throughput: {results['throughput']:.0f} samples/s")
        return results
 
 
# ─── DriftSimulator ───────────────────────────────────────────────────────────
 
class DriftSimulator:
    """
    Simulates various economic shock scenarios on batch data.
    Useful for evaluating model degradation before deploying drift detection.
 
    Parameters
    ----------
    model_predict_fn : callable — takes a pd.DataFrame, returns probability array
    feature_cols     : list of feature column names
    """
 
    SCENARIOS = {
        "Baseline":                {"income_mult": 1.0,  "emp_mask": 0.00, "label_noise": 0.00},
        "Mild Income Shock -30%":  {"income_mult": 0.70, "emp_mask": 0.05, "label_noise": 0.02},
        "Severe Income Shock -60%":{"income_mult": 0.40, "emp_mask": 0.15, "label_noise": 0.05},
        "Mass Job Loss 20%":       {"income_mult": 0.50, "emp_mask": 0.20, "label_noise": 0.08},
        "Full Economic Collapse":  {"income_mult": 0.25, "emp_mask": 0.40, "label_noise": 0.15},
    }
 
    def __init__(self, model_predict_fn, feature_cols: List[str]):
        self.predict     = model_predict_fn
        self.feature_cols = feature_cols
 
    def _apply_shock(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        income_mult: float,
        emp_mask: float,
        label_noise: float,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        from sklearn.metrics import roc_auc_score
 
        rng      = np.random.RandomState(seed)
        X_shock  = X.copy()
        y_shock  = y.copy()
 
        # Income shock
        for col in [c for c in X_shock.columns if "INCOME" in c]:
            X_shock[col] *= income_mult
 
        # Employment shock — zero out employment columns for `emp_mask` fraction
        emp_cols = [c for c in X_shock.columns if "EMPLOY" in c or "DAYS_EMPLOYED" in c]
        mask     = rng.random(len(X_shock)) < emp_mask
        for col in emp_cols:
            X_shock.loc[mask, col] = 0
 
        # Label noise
        noise_idx = rng.choice(len(y_shock), int(label_noise * len(y_shock)), replace=False)
        y_shock[noise_idx] = 1 - y_shock[noise_idx]
 
        return X_shock, y_shock
 
    def run_all_scenarios(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        log_wandb: bool = False,
    ) -> pd.DataFrame:
        from sklearn.metrics import roc_auc_score
 
        results = []
        for name, params in self.SCENARIOS.items():
            X_s, y_s = self._apply_shock(X, y, **params)
            # Align columns
            for col in self.feature_cols:
                if col not in X_s.columns:
                    X_s[col] = 0.0
            X_s = X_s[self.feature_cols]
 
            preds = self.predict(X_s)
            auc   = roc_auc_score(y_s, preds)
 
            results.append({
                "scenario":    name,
                "auc":         auc,
                "income_mult": params["income_mult"],
                "emp_mask":    params["emp_mask"],
                "label_noise": params["label_noise"],
            })
 
            if log_wandb:
                wandb.log({"drift_sim/scenario": name, "drift_sim/auc": auc})
 
            print(f"  {name:<35s} | AUC: {auc:.5f}")
 
        df = pd.DataFrame(results)
        baseline_auc = df.loc[df["scenario"] == "Baseline", "auc"].values[0]
        df["auc_drop"]    = baseline_auc - df["auc"]
        df["pct_drop"]    = (df["auc_drop"] / baseline_auc * 100).round(2)
        return df
 
 
# ─── DriftDashboard ───────────────────────────────────────────────────────────
 
class DriftDashboard:
    """
    Generates publication-quality drift analysis plots.
    """
 
    @staticmethod
    def plot_error_stream(
        errors: List[float],
        drift_events: List[DriftEvent],
        drift_inject_at: Optional[int] = None,
        window: int = 500,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Smoothed error stream with drift markers."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        smoothed = pd.Series(errors).rolling(window).mean()
 
        ax1.plot(smoothed, color="#1565C0", linewidth=1.2, label=f"Error (rolling {window})")
        ax1.fill_between(range(len(smoothed)), smoothed, alpha=0.15, color="#1565C0")
 
        colors = {"ADWIN": "#F44336", "KSWIN": "#FF9800"}
        for evt in drift_events:
            c = colors.get(evt.detector, "#9C27B0")
            ax1.axvline(evt.sample_index, color=c, linewidth=0.7, alpha=0.8)
 
        if drift_inject_at:
            ax1.axvline(drift_inject_at, color="orange", linewidth=2.5,
                        linestyle="--", label="Synthetic Drift Injected")
 
        ax1.set_ylabel("Prediction Error")
        ax1.set_title("ADWIN + KSWIN Drift Detection — Error Stream", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right")
 
        # Cumulative detections
        if drift_events:
            indices = [e.sample_index for e in drift_events]
            ax2.step(indices, range(1, len(indices)+1), color="#F44336", linewidth=2)
            if drift_inject_at:
                ax2.axvline(drift_inject_at, color="orange", linewidth=2.5, linestyle="--")
        ax2.set_ylabel("Cumulative Detections")
        ax2.set_xlabel("Sample Index")
        ax2.set_title("Cumulative Drift Events", fontsize=11)
 
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved drift stream plot → {save_path}")
        return fig
 
    @staticmethod
    def plot_scenario_degradation(
        drift_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Plotly bar chart of AUC across drift scenarios."""
        PALETTE = ["#4CAF50", "#8BC34A", "#FF9800", "#F44336", "#B71C1C"]
 
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=drift_df["scenario"],
            y=drift_df["auc"],
            marker_color=PALETTE[:len(drift_df)],
            text=[f"{a:.4f}<br>({d:+.4f})" for a, d in zip(drift_df["auc"], -drift_df["auc_drop"])],
            textposition="outside",
        ))
        fig.add_hline(y=0.70, line_dash="dash", line_color="#F44336",
                      annotation_text="Min Acceptable AUC (0.70)")
        fig.update_layout(
            title="Model AUC Under Concept Drift Scenarios",
            xaxis_title="Scenario",
            yaxis_title="ROC-AUC",
            yaxis_range=[0.5, max(drift_df["auc"]) + 0.05],
            height=480,
            template="plotly_white",
        )
 
        if save_path:
            fig.write_image(save_path)
            print(f"✅ Saved scenario degradation plot → {save_path}")
        return fig
 
    @staticmethod
    def plot_income_sensitivity(
        drift_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Plotly line: AUC vs income multiplier."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drift_df["income_mult"],
            y=drift_df["auc"],
            mode="lines+markers",
            name="Ensemble AUC",
            line=dict(color="#F44336", width=3),
            marker=dict(size=10, color=[
                "#4CAF50" if a > 0.75 else "#FF9800" if a > 0.65 else "#F44336"
                for a in drift_df["auc"]
            ]),
        ))
        fig.add_hline(y=0.70, line_dash="dash", line_color="#666",
                      annotation_text="Min Acceptable AUC")
        fig.update_layout(
            title="AUC Degradation vs Income Shock Severity",
            xaxis_title="Remaining Income Fraction (1.0 = no shock)",
            yaxis_title="ROC-AUC",
            height=420,
            template="plotly_white",
        )
 
        if save_path:
            fig.write_image(save_path)
            print(f"✅ Saved income sensitivity plot → {save_path}")
        return fig