# Vision Transformer (ViT) Comparison & Analysis Project

This project fulfills all requirements specified in `REQ.txt` and the university lab task. It provides a comprehensive comparison between three Vision Transformer models, defines the architecture mathematically, and presents the findings in a formal paper and presentation.

## Project Structure & Requirements Mapping

| Requirement (from REQ.txt) | Implementation File | Description |
| :--- | :--- | :--- |
| **Compare between three models** | `Compare_Three_ViT_Models.py` | Python script that runs inference on ViT-B/32, ViT-B/16, and ViT-L/14. |
| **Define model architecture** | `Scientific_Paper_and_Architecture_Definition.md` | Detailed breakdown of patches, embeddings, and transformer blocks. |
| **All equations** | `Scientific_Paper_and_Architecture_Definition.md` | Mathematical definitions for Attention, MSA, MLP, and Embeddings. |
| **Drawbacks & Advantages** | `Scientific_Paper_and_Architecture_Definition.md` | Comprehensive list of ViT pros (scaling) and cons (data hunger). |
| **Vision for enhancement** | `Scientific_Paper_and_Architecture_Definition.md` | Section on Hybrid models, Linear Attention, and Self-Supervision. |
| **Write a Scientific paper** | `Scientific_Paper_and_Architecture_Definition.md` | A complete, formally structured academic paper (Markdown). |
| **High-Quality LaTeX Paper** | `Vision_Transformer_Scientific_Paper.tex` | Deep, academically rigorous LaTeX source with advanced derivations. |
| **Compiled PDF Paper** | `Vision_Transformer_Scientific_Paper.pdf` | The professional PDF version of the research paper. |
| **Valuable presentation** | `Vision_Transformer_Presentation.md` | A slide-based summary of the project (Marp compatible). |
| **Experimental Analysis** | `Final_Comparison_Report.md` | The generated output comparing the three models' accuracy and confidence. |

## How to Run

1.  **Setup Environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Comparison Experiment:**
    ```bash
    python Compare_Three_ViT_Models.py
    ```
3.  **Generate Analysis Report:**
    ```bash
    python Analyze_and_Generate_Report.py
    ```

## Models Compared
1.  **ViT-B/32:** Base model with 32x32 patches.
2.  **ViT-B/16:** Base model with 16x16 patches (higher granularity).
3.  **ViT-L/14:** Large model with 14x14 patches (maximum depth and detail).
