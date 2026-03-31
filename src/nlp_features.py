"""
src/nlp_features.py
──────────────────────────────────────────────────────────────────────────────
NLP / Sentence-BERT embedding pipeline for alternative credit signals.
 
Provides:
  • FinancialNarrativeBuilder  — synthesises a text description per applicant
  • SBERTEmbedder              — encodes texts → embeddings → PCA reduction
  • NLPFeaturePipeline         — end-to-end fit/transform orchestrator
 
In production, FinancialNarrativeBuilder would be replaced by real user
survey or app-usage text. Here we synthesise from tabular signals to
demonstrate the pipeline architecture.
 
Usage:
    from src.nlp_features import NLPFeaturePipeline
    nlp = NLPFeaturePipeline(cfg)
    train_nlp_df = nlp.fit_transform(train_df)
    test_nlp_df  = nlp.transform(test_df)
──────────────────────────────────────────────────────────────────────────────
"""
 
from __future__ import annotations
 
import os
import gc
import warnings
from typing import List, Optional
 
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
 
warnings.filterwarnings("ignore")
 
 
# ─── Narrative templates ──────────────────────────────────────────────────────
 
_LITERACY_LEVELS = {
    "high": [
        "demonstrates strong financial planning habits and consistently pays obligations on time",
        "shows excellent budgeting discipline and proactively manages debt obligations",
        "has a clear savings strategy and maintains low revolving credit utilisation",
    ],
    "medium": [
        "shows moderate financial awareness with occasional delayed payments",
        "manages debt adequately but has limited long-term financial planning",
        "meets minimum payment requirements but rarely pays ahead of schedule",
    ],
    "low": [
        "has limited formal financial experience and irregular payment patterns",
        "relies heavily on informal credit channels and lacks credit history",
        "demonstrates financial stress indicators with frequent payment shortfalls",
    ],
}
 
_EMPLOYMENT_TEMPLATES = {
    "long":   "Has stable employment of {years:.1f} years with the current employer.",
    "medium": "Currently employed for {years:.1f} years; career trajectory appears stable.",
    "short":  "Recently started employment ({years:.1f} years); income may not be fully stabilised.",
    "none":   "No current formal employment; income source requires verification.",
}
 
_ASSET_SENTENCES = {
    (True,  True):  "Applicant owns both a vehicle and residential property, indicating established assets.",
    (True,  False): "Applicant owns a vehicle but rents accommodation.",
    (False, True):  "Applicant owns residential property, a strong collateral signal.",
    (False, False): "No registered asset ownership; relies solely on income for repayment.",
}
 
 
class FinancialNarrativeBuilder:
    """
    Constructs a structured financial narrative text per applicant row.
 
    Parameters
    ----------
    ext_high_threshold  : EXT_SOURCE_MEAN above which → high financial literacy
    ext_low_threshold   : EXT_SOURCE_MEAN below which → low financial literacy
    include_enquiry     : whether to include credit enquiry paragraph
    include_bureau      : whether to include bureau summary paragraph
    random_template_seed: reproducibility for template sampling
    """
 
    def __init__(
        self,
        ext_high_threshold:   float = 0.60,
        ext_low_threshold:    float = 0.40,
        include_enquiry:      bool  = True,
        include_bureau:       bool  = True,
        random_template_seed: int   = 42,
    ):
        self.ext_high     = ext_high_threshold
        self.ext_low      = ext_low_threshold
        self.incl_enquiry = include_enquiry
        self.incl_bureau  = include_bureau
        self._rng         = np.random.RandomState(random_template_seed)
 
    def _literacy_sentence(self, ext_mean: float) -> str:
        if ext_mean >= self.ext_high:
            pool = _LITERACY_LEVELS["high"]
        elif ext_mean >= self.ext_low:
            pool = _LITERACY_LEVELS["medium"]
        else:
            pool = _LITERACY_LEVELS["low"]
        return self._rng.choice(pool)
 
    def _employment_sentence(self, emp_years: float) -> str:
        if emp_years > 10:
            tmpl = _EMPLOYMENT_TEMPLATES["long"]
        elif emp_years > 2:
            tmpl = _EMPLOYMENT_TEMPLATES["medium"]
        elif emp_years > 0:
            tmpl = _EMPLOYMENT_TEMPLATES["short"]
        else:
            return _EMPLOYMENT_TEMPLATES["none"]
        return tmpl.format(years=emp_years)
 
    def build_one(self, row: pd.Series) -> str:
        """Build a single narrative string from one applicant row."""
 
        income      = float(row.get("AMT_INCOME_TOTAL",  150_000))
        credit      = float(row.get("AMT_CREDIT",        300_000))
        age         = abs(float(row.get("DAYS_BIRTH",    -35*365))) / 365
        emp_years   = max(0, -float(row.get("DAYS_EMPLOYED", -3*365))) / 365
        ext1        = float(row.get("EXT_SOURCE_1",      0.5))
        ext2        = float(row.get("EXT_SOURCE_2",      0.5))
        ext3        = float(row.get("EXT_SOURCE_3",      0.5))
        ext_mean    = np.nanmean([ext1, ext2, ext3])
        has_realty  = bool(row.get("FLAG_OWN_REALTY", 0))
        has_car     = bool(row.get("FLAG_OWN_CAR",    0))
        n_children  = int(row.get("CNT_CHILDREN",     0))
        fam_size    = int(row.get("CNT_FAM_MEMBERS",  2))
        credit_income = credit / (income + 1)
 
        parts: List[str] = []
 
        # ── Core financial summary ──────────────────────────────────────
        parts.append(
            f"Applicant is {age:.0f} years old with a declared annual income of "
            f"{income:,.0f} currency units. "
            f"Requesting a credit facility of {credit:,.0f} units, "
            f"representing a credit-to-income ratio of {credit_income:.2f}x."
        )
 
        # ── Financial literacy level ────────────────────────────────────
        parts.append(f"Client {self._literacy_sentence(ext_mean)}.")
 
        # ── Employment ─────────────────────────────────────────────────
        parts.append(self._employment_sentence(emp_years))
 
        # ── Asset ownership ────────────────────────────────────────────
        parts.append(_ASSET_SENTENCES.get((has_car, has_realty), ""))
 
        # ── Family context ─────────────────────────────────────────────
        if n_children > 0:
            parts.append(
                f"Applicant has {n_children} dependent child{'ren' if n_children>1 else ''} "
                f"in a household of {fam_size}."
            )
        else:
            parts.append(f"No dependents; household size of {fam_size}.")
 
        # ── Credit bureau summary ──────────────────────────────────────
        if self.incl_bureau:
            bureau_count  = int(row.get("BUREAU_COUNT", 0))
            active_count  = int(row.get("BUREAU_ACTIVE_COUNT", 0))
            if bureau_count > 0:
                parts.append(
                    f"Bureau records show {bureau_count} historical credit lines, "
                    f"of which {active_count} are currently active."
                )
            else:
                parts.append(
                    "No external bureau credit history found — applicant is credit-invisible."
                )
 
        # ── Enquiry signals ────────────────────────────────────────────
        if self.incl_enquiry:
            enquiries = int(row.get("TOTAL_ENQUIRIES", 0))
            if enquiries > 5:
                parts.append(
                    f"High credit enquiry volume ({enquiries} enquiries) may indicate "
                    f"credit-seeking stress or rate shopping."
                )
            elif enquiries > 0:
                parts.append(f"Moderate enquiry activity ({enquiries} enquiries recorded).")
            else:
                parts.append("No recent credit enquiries recorded.")
 
        # ── External score summary ─────────────────────────────────────
        parts.append(
            f"External creditworthiness assessments: "
            f"bureau={ext1:.2f}, behavioural={ext2:.2f}, alternative={ext3:.2f} "
            f"(composite={ext_mean:.2f})."
        )
 
        return " ".join(p for p in parts if p)
 
    def build_batch(self, df: pd.DataFrame, verbose: bool = True) -> List[str]:
        """Build narratives for an entire DataFrame."""
        narratives = []
        n = len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            narratives.append(self.build_one(row))
            if verbose and (i + 1) % 50_000 == 0:
                print(f"  Narratives built: {i+1:,}/{n:,}")
        return narratives
 
 
# ─── SBERT Embedder ───────────────────────────────────────────────────────────
 
class SBERTEmbedder:
    """
    Encodes text narratives with Sentence-BERT and optionally reduces
    dimensionality with PCA.
 
    Parameters
    ----------
    model_name     : HuggingFace SBERT model name
    n_components   : PCA output dimension (None = no PCA)
    batch_size     : encoding batch size
    normalize      : L2-normalise embeddings before PCA
    device         : "cpu" | "cuda" | "mps" (auto if None)
    """
 
    def __init__(
        self,
        model_name:   str           = "all-MiniLM-L6-v2",
        n_components: Optional[int] = 32,
        batch_size:   int           = 512,
        normalize:    bool          = True,
        device:       Optional[str] = None,
    ):
        self.model_name   = model_name
        self.n_components = n_components
        self.batch_size   = batch_size
        self.normalize    = normalize
        self.device       = device
        self.pca: Optional[PCA] = None
        self._model: Optional[SentenceTransformer] = None
 
    def _load_model(self):
        if self._model is None:
            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
            print(f"✅ SBERT loaded: {self.model_name} "
                  f"(dim={self._model.get_sentence_embedding_dimension()})")
 
    def _encode(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
 
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Encode + fit PCA on train texts."""
        print(f"🤖 Encoding {len(texts):,} texts with SBERT...")
        emb = self._encode(texts)
        print(f"   Raw embedding shape: {emb.shape}")
 
        if self.n_components:
            n = min(self.n_components, emb.shape[0], emb.shape[1])
            self.pca = PCA(n_components=n, random_state=42)
            emb = self.pca.fit_transform(emb)
            print(f"   After PCA({n}): {emb.shape} | "
                  f"Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
 
        del self._model; self._model = None; gc.collect()
        return emb
 
    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode + apply fitted PCA on new texts."""
        print(f"🤖 Encoding {len(texts):,} texts (transform)...")
        emb = self._encode(texts)
        if self.pca is not None:
            emb = self.pca.transform(emb)
        del self._model; self._model = None; gc.collect()
        return emb
 
    def save(self, path: str):
        """Persist PCA object."""
        if self.pca is not None:
            joblib.dump(self.pca, path)
            print(f"✅ PCA saved → {path}")
 
    def load_pca(self, path: str):
        """Load a previously saved PCA object."""
        self.pca = joblib.load(path)
        print(f"✅ PCA loaded ← {path}")
 
 
# ─── End-to-end NLP pipeline ─────────────────────────────────────────────────
 
class NLPFeaturePipeline:
    """
    Orchestrates FinancialNarrativeBuilder + SBERTEmbedder.
 
    Parameters
    ----------
    cfg          : project config dataclass
    model_name   : SBERT model name
    n_components : PCA components
    batch_size   : SBERT batch size
    """
 
    def __init__(
        self,
        cfg,
        model_name:   str = "all-MiniLM-L6-v2",
        n_components: int = 32,
        batch_size:   int = 512,
    ):
        self.cfg           = cfg
        self.n_components  = n_components
        self.narrator      = FinancialNarrativeBuilder()
        self.embedder      = SBERTEmbedder(
            model_name   = model_name,
            n_components = n_components,
            batch_size   = batch_size,
        )
        self._emb_col_names = [f"NLP_EMB_{i}" for i in range(n_components)]
 
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build NLP features for training data.
        Fits PCA internally. Returns DataFrame with NLP_EMB_* columns.
        """
        texts = self.narrator.build_batch(df)
        emb   = self.embedder.fit_transform(texts)
        self.embedder.save(os.path.join(self.cfg.MODEL_DIR, "pca.pkl"))
        return pd.DataFrame(emb, columns=self._emb_col_names, index=df.index)
 
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build NLP features for new / test data.
        Uses already-fitted PCA. Returns DataFrame with NLP_EMB_* columns.
        """
        if self.embedder.pca is None:
            pca_path = os.path.join(self.cfg.MODEL_DIR, "pca.pkl")
            if os.path.exists(pca_path):
                self.embedder.load_pca(pca_path)
            else:
                raise FileNotFoundError(
                    f"PCA not found at {pca_path}. Run fit_transform first."
                )
        texts = self.narrator.build_batch(df, verbose=False)
        emb   = self.embedder.transform(texts)
        return pd.DataFrame(emb, columns=self._emb_col_names, index=df.index)
 
    def build_single_row(self, feature_dict: dict) -> pd.DataFrame:
        """
        Build NLP features for a single applicant (inference).
 
        Parameters
        ----------
        feature_dict : raw applicant features as a dict
 
        Returns
        -------
        DataFrame with NLP_EMB_* columns (1 row)
        """
        row = pd.Series(feature_dict)
        text = self.narrator.build_one(row)
        emb  = self.embedder.transform([text])
        return pd.DataFrame(emb, columns=self._emb_col_names)