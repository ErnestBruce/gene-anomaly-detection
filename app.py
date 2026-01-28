# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.neighbors import LocalOutlierFactor
from io import StringIO
import gzip

# Configure page
st.set_page_config(
    page_title="WIUC Gene Expression Anomaly Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit branding
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 🧬 Title & Instructions
# -----------------------------
st.title("🔍 WIUC Gene Expression Anomaly Detector")
st.markdown("""
Upload your **gene expression data** to identify unusual samples (e.g., potential disease cases or outliers).

### Supported Formats:
- NCBI GEO files: `.txt`, `.tsv`, `.txt.gz`
- Excel files: `.xlsx`, `.xls`
- Plain tables: `.csv`

> ✅ All files must have **genes as rows** and **samples as columns**.
""")

# -----------------------------
# 📤 File Uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📁 Choose a gene expression file",
    type=["txt", "tsv", "csv", "gz", "xlsx", "xls"],
    help="Genes as rows, samples as columns"
)

if uploaded_file is None:
    st.info("👆 Please upload a file to begin.")
    st.stop()

# -----------------------------
# 🔍 Load & Parse File
# -----------------------------
@st.cache_data
def load_data_from_upload(uploaded_file):
    filename = uploaded_file.name.lower()
    
    try:
        if filename.endswith('.gz'):
            # GEO .txt.gz file
            with gzip.open(uploaded_file, 'rt', encoding='utf-8') as f:
                lines = [line for line in f if not line.startswith(('!', '#'))]
            clean_content = '\n'.join(lines)
            sep = '\t' if '\t' in clean_content[:500] else ','
            df = pd.read_csv(StringIO(clean_content), sep=sep, index_col=0, low_memory=False)
            return df
            
        elif filename.endswith(('.xlsx', '.xls')):
            # Excel gene matrix
            df = pd.read_excel(uploaded_file, index_col=0)
            return df
            
        else:
            # Plain text: CSV/TSV
            content = uploaded_file.read().decode('utf-8')
            lines = [line for line in content.splitlines() if not line.startswith(('!', '#'))]
            clean_content = '\n'.join(lines)
            sep = '\t' if '\t' in clean_content[:500] else ','
            df = pd.read_csv(StringIO(clean_content), sep=sep, index_col=0, low_memory=False)
            return df
            
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

data = load_data_from_upload(uploaded_file)

# Validate
if data.empty:
    st.error("Uploaded file is empty!")
    st.stop()
if data.shape[0] < 2 or data.shape[1] < 2:
    st.error("Data must have at least 2 genes (rows) and 2 samples (columns).")
    st.stop()

st.success(f"✅ Loaded {data.shape[1]} samples and {data.shape[0]} genes.")

# -----------------------------
# ⚙️ Gene-Specific Pipeline
# -----------------------------
@st.cache_data
def run_gene_pipeline(data, _sample_ids):
    # Clean: convert to numeric, drop genes with any non-numeric values
    data_clean = data.apply(pd.to_numeric, errors='coerce')
    data_clean = data_clean.dropna()
    
    if data_clean.empty:
        return None, "No valid numeric gene data found."

    # Transpose: samples as rows, genes as columns
    data_T = data_clean.T

    # Log-transform (standard for gene expression)
    data_log = np.log2(data_T + 1)

    # Force column names to string (fixes sklearn feature name issue)
    data_log.columns = [str(col).strip() for col in data_log.columns]

    # Standardize genes
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_log)

    # PCA & FA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)

    fa = FactorAnalysis(n_components=2, random_state=42)
    fa_result = fa.fit_transform(data_scaled)

    # LOF anomaly detection
    n_neighbors = min(10, len(data_scaled) - 1)
    if n_neighbors < 2:
        return None, "Not enough samples for anomaly detection."

    lof_pca = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof_pca.fit(pca_result)
    lof_scores_pca = -lof_pca.negative_outlier_factor_

    lof_fa = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof_fa.fit(fa_result)
    lof_scores_fa = -lof_fa.negative_outlier_factor_

    # Adaptive threshold (95th percentile)
    threshold_pca = np.percentile(lof_scores_pca, 95)
    threshold_fa = np.percentile(lof_scores_fa, 95)

    anomaly_pca = np.where(lof_scores_pca > threshold_pca, -1, 1)
    anomaly_fa = np.where(lof_scores_fa > threshold_fa, -1, 1)
    consensus = np.where((anomaly_pca == -1) | (anomaly_fa == -1), -1, 1)

    # Result DataFrame
    plot_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'FA1': fa_result[:, 0],
        'FA2': fa_result[:, 1],
        'Sample': _sample_ids,
        'LOF_PCA_Score': lof_scores_pca,
        'LOF_FA_Score': lof_scores_fa,
        'Consensus_Anomaly': consensus
    })

    # Heatmap: top 50 most variable genes
    gene_var = data_log.var(axis=0)
    top_n = min(50, len(gene_var))
    top_genes = gene_var.nlargest(top_n).index

    anomalous_samples = plot_df[plot_df['Consensus_Anomaly'] == -1]['Sample']
    normal_samples = plot_df[plot_df['Consensus_Anomaly'] == 1]['Sample']
    ordered_samples = list(anomalous_samples) + list(normal_samples[:min(20, len(normal_samples))])
    heatmap_data = data_log.loc[ordered_samples, top_genes]

    return (plot_df, heatmap_data, pca, threshold_pca, threshold_fa, data_log), None

# Run pipeline
sample_ids = data.columns.tolist()
result, error = run_gene_pipeline(data, sample_ids)

if error:
    st.error(f"Analysis failed: {error}")
    st.stop()

plot_df, heatmap_data, pca, thr_pca, thr_fa, data_log = result
anomalous_ids = plot_df[plot_df['Consensus_Anomaly'] == -1]['Sample'].tolist()

# -----------------------------
# 📊 Visualizations (Gene-Focused Only)
# -----------------------------

# 1. PCA
st.markdown("## 🌐 1. Overall Pattern Map (PCA)")
st.markdown("""
Each dot is a sample (e.g., a patient or tissue).  
- **Blue**: Typical gene activity  
- **Red**: Unusual pattern (potential anomaly)  

This map simplifies thousands of genes into 2 main trends. Red dots stand out from the crowd.
""")
fig1, ax1 = plt.subplots(figsize=(6, 5))
colors = plot_df['Consensus_Anomaly'].map({1: 'steelblue', -1: 'crimson'})
ax1.scatter(plot_df['PC1'], plot_df['PC2'], c=colors, alpha=0.8, s=50, edgecolor='k', linewidth=0.3)
ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax1.set_title("PCA Pattern Map")
st.pyplot(fig1)

# 2. FA
st.markdown("## 🔍 2. Hidden Factor Map (Factor Analysis)")
st.markdown("""
Another view focusing on hidden biological signals.  
- **Green**: Typical  
- **Red**: Unusual  

Samples red in **both maps** are strong anomaly candidates.
""")
fig2, ax2 = plt.subplots(figsize=(6, 5))
colors_fa = plot_df['Consensus_Anomaly'].map({1: 'seagreen', -1: 'crimson'})
ax2.scatter(plot_df['FA1'], plot_df['FA2'], c=colors_fa, alpha=0.8, s=50, edgecolor='k', linewidth=0.3)
ax2.set_xlabel("Factor 1")
ax2.set_ylabel("Factor 2")
ax2.set_title("Factor Analysis Map")
st.pyplot(fig2)

# 3. LOF Scores
st.markdown("## 📈 3. How Unusual Is Each Sample?")
st.markdown("""
"Weirdness score" for every sample:  
- **Higher score** = more unusual  
- **Red dashed line** = threshold (top 5% most unusual)  
- Samples **right of the line** are flagged as anomalies
""")
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))
ax3.hist(plot_df['LOF_PCA_Score'], bins=20, color='skyblue', edgecolor='k', alpha=0.7)
ax3.axvline(thr_pca, color='red', linestyle='--', label=f'Threshold')
ax3.set_title("LOF Score (PCA)")
ax3.set_xlabel("Score")
ax3.legend()

ax4.hist(plot_df['LOF_FA_Score'], bins=20, color='lightgreen', edgecolor='k', alpha=0.7)
ax4.axvline(thr_fa, color='red', linestyle='--', label=f'Threshold')
ax4.set_title("LOF Score (FA)")
ax4.set_xlabel("Score")
ax4.legend()

plt.tight_layout()
st.pyplot(fig3)

# 4. Heatmap
st.markdown("## 🧬 4. Gene Activity Snapshot")
st.markdown("""
Think of this as a **"gene activity snapshot"** — like a thermal camera for biology!

### 🔍 How to Read:
- **Each row** = a specific gene  
- **Each column** = a sample  
- **Color tells the story**:
  - 🟡 **Yellow** = **high activity** (gene is "turned ON")
  - 🟣 **Purple** = **low activity** (gene is "turned OFF")

### 👀 What You’re Seeing:
- **Left**: Unusual (anomalous) samples  
- **Right**: Typical samples (for comparison)


### 💡 Why This Matters:
If the **left side shows a consistent pattern** (e.g., a block of yellow in certain genes),  
it could mean:  
> “These unusual samples share something in common — maybe a rare disease, a treatment response, or a technical error.”

Scientists use patterns like this to **generate new hypotheses** or **double-check data quality**.

> ✨ **Tip**: Look for vertical stripes or blocks of color on the left that *don’t appear* on the right!
""")

fig4, ax4 = plt.subplots(figsize=(12, 7))
sns.heatmap(
    heatmap_data.T,
    cmap='viridis',
    xticklabels=False,
    yticklabels=True,
    cbar_kws={'label': 'Gene Activity'}
)
ax4.set_title("Gene Activity: Anomalous (Left) vs. Typical (Right)", fontsize=13, pad=20)
ax4.set_xlabel("Samples")
ax4.set_ylabel("Genes (Top 50 Most Variable)")
plt.tight_layout()
st.pyplot(fig4)

st.info("💡 This view shows the **50 most variable genes** — often the most biologically informative.")

# -----------------------------
# 📥 Results
# -----------------------------
st.markdown("## ✅ Detected Anomalies")
st.markdown(f"Found **{len(anomalous_ids)} unusual samples** out of {len(plot_df)} total.")

if len(anomalous_ids) > 0:
    st.write("**Anomalous Sample IDs:**")
    st.text("\n".join(anomalous_ids))
    csv = plot_df[['Sample', 'Consensus_Anomaly']].to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Anomaly Report (CSV)",
        csv,
        "gene_anomaly_report.csv",
        "text/csv"
    )
else:
    st.info("No anomalies detected. All samples appear typical.")

st.markdown("---")
st.caption("💡 WIUC Gene Expression Anomaly Detector | Accepts Excel, CSV, and GEO .txt.gz files")