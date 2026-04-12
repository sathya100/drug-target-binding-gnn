"""
app.py — Gradio interface for Drug-Target Binding Affinity Prediction
EGN6217 | Sathyadharini Srinivasan | Spring 2026
"""

import sys, os, io, base64
import torch
import numpy as np
import gradio as gr

# Allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from graph_utils import smiles_to_graph, encode_protein
from model import DTAModel

# ── Load model ────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'results', 'dta_model.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = DTAModel().to(device)

if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    MODEL_LOADED = True
else:
    MODEL_LOADED = False

# ── Prediction logic ─────────────────────────────────────────────────────────
def predict(smiles: str, protein_seq: str):
    if not MODEL_LOADED:
        return "⚠️ Model checkpoint not found. Run training first.", "", None

    smiles      = smiles.strip()
    protein_seq = protein_seq.strip().upper()

    if not smiles:
        return "⚠️ Please enter a SMILES string.", "", None
    if not protein_seq:
        return "⚠️ Please enter a protein sequence.", "", None

    # Convert SMILES → molecular graph
    graph = smiles_to_graph(smiles)
    if graph is None:
        return "❌ Invalid SMILES string — could not parse with RDKit.", "", None

    # Encode protein
    prot_tensor = encode_protein(protein_seq, max_len=1000).unsqueeze(0).to(device)

    # Batch the graph (single item)
    from torch_geometric.data import Batch
    graph_batch = Batch.from_data_list([graph]).to(device)

    with torch.no_grad():
        pkd_pred = model(graph_batch, prot_tensor).item()

    # Convert pKd → Kd (nM)
    kd_nm = 10 ** (-pkd_pred) * 1e9

    # Binding strength label
    if kd_nm < 100:
        strength = "🟢 STRONG BINDING"
        color_html = "#22c55e"
    elif kd_nm < 1000:
        strength = "🟡 MODERATE BINDING"
        color_html = "#eab308"
    else:
        strength = "🔴 WEAK BINDING"
        color_html = "#ef4444"

    result_text = (
        f"Predicted pKd    : {pkd_pred:.3f}\n"
        f"Predicted Kd     : {kd_nm:,.1f} nM\n"
        f"Binding Strength : {strength}"
    )

    badge_html = f"""
    <div style='text-align:center; padding:20px; border-radius:12px;
                background:#f8fafc; border: 2px solid {color_html}; margin-top:10px;'>
        <div style='font-size:2em; font-weight:bold; color:{color_html};'>
            {kd_nm:,.1f} nM
        </div>
        <div style='font-size:1.1em; color:#334155; margin-top:6px;'>
            pKd = {pkd_pred:.3f}
        </div>
        <div style='font-size:1.3em; margin-top:10px; font-weight:600; color:{color_html};'>
            {strength}
        </div>
        <div style='font-size:0.85em; color:#64748b; margin-top:8px;'>
            &lt; 100 nM = Strong &nbsp;|&nbsp; 100–1000 nM = Moderate &nbsp;|&nbsp; &gt; 1000 nM = Weak
        </div>
    </div>"""

    # 2D molecule visualization
    mol_img = None
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(300, 250))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            mol_img = buf.getvalue()
    except Exception:
        pass

    return result_text, badge_html, mol_img


# ── Example inputs ────────────────────────────────────────────────────────────
EXAMPLES = [
    [
        "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=CC=C3)C(F)(F)F",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
    ],
    [
        "C1=CC=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C(=O)C4=CC=CC=C4",
        "MAAVILESIFLKRSQQKKKTSPLNFKKRLFLLTVHKLSYYEYDFERDDIGKKQVTSGTVEADNLYPEVNPNVLQEIGEQLKKEIFNLEKALDVKVYLNQNLVPEQKQLQVGQKIPKGGTRMSKLKVIASNTKLWGRPQKPAMITDAGKTHFNLQHFLGFHWPYLHEVNLLNFETFDVNLEFRNLKKAVSEILFPDDTPRIIFNQNLKELPKNFKGRNHIFAQRGGVKLNHVV",
    ],
]

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="DTA Predictor") as demo:
    gr.Markdown("""
    # Drug-Target Binding Affinity Predictor
    **EGN6217 | Sathyadharini Srinivasan | University of Florida | Spring 2026**

    Predict the binding affinity (Kd in nM) between a **drug molecule** (SMILES) and a
    **protein target** (amino acid sequence) using a Graph Neural Network + 1D CNN architecture.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            smiles_input = gr.Textbox(
                label="Drug SMILES String",
                placeholder="e.g. CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=CC=C3)C(F)(F)F",
                lines=3,
            )
            protein_input = gr.Textbox(
                label="Protein Amino Acid Sequence (max 1000 chars)",
                placeholder="e.g. MKTAYIAKQRQISFVK...",
                lines=6,
            )
            predict_btn = gr.Button("Predict Binding Affinity", variant="primary")

        with gr.Column(scale=1):
            result_text  = gr.Textbox(label="Prediction Output", lines=4)
            badge_output = gr.HTML(label="Binding Strength")
            mol_image    = gr.Image(label="2D Molecule Structure", type="numpy",
                                    height=250)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[smiles_input, protein_input],
        label="Example Drug-Protein Pairs",
    )

    predict_btn.click(
        fn=predict,
        inputs=[smiles_input, protein_input],
        outputs=[result_text, badge_output, mol_image],
    )

    gr.Markdown("""
    ---
    **Model:** GCN (3-layer, 5→64→128 dim) + Conv1D (3-layer, 96 dim) + MLP regressor
    **Dataset:** DeepDTA Davis — 442 drugs × 68 proteins — 30,056 pairs
    **Metrics:** Pearson r = 0.606 | CI = 0.801 | Test MSE = 0.479 (pKd)
    """)


if __name__ == "__main__":
    demo.launch(share=False)
