# Vision Transformer (ViT) Comparison & Analysis Project

This project fulfills all requirements specified in `REQ.txt` and the university lab task. It provides a comprehensive comparison between modern Vision Transformer architectures, defines their internal mechanics mathematically, and presents the findings in a cutting-edge scientific paper along with tested PyTorch implementations of novel enhancements.

## Project Structure & Requirements Mapping

| Requirement (from REQ.txt) | Implementation File | Description |
| :--- | :--- | :--- |
| **Compare between three models** | `Meta_ViT_Comparative_Analysis.pdf` | A detailed academic comparison of SAM 2, DeiT, and DINOv3. |
| **Define model architecture** | `Meta_ViT_Comparative_Analysis.pdf` | Detailed breakdown of patches, distillation, streaming memory, and contrastive losses. |
| **All equations** | `Meta_ViT_Comparative_Analysis.pdf` | Mathematical definitions for Attention, MSA, MLP, and advanced objectives. |
| **Drawbacks & Advantages** | `Meta_ViT_Comparative_Analysis.pdf` | Comprehensive list of pros and cons across supervised and self-supervised regimes. |
| **Vision for enhancement** | `Meta_ViT_Enhancements.py` | PyTorch code implementing three novel hybrid architectures. |
| **Write a Scientific paper** | `Meta_ViT_Comparative_Analysis.tex` | Deep, academically rigorous LaTeX source with advanced derivations. |
| **Compiled PDF Paper** | `Meta_ViT_Comparative_Analysis.pdf` | The professional PDF version of the research paper. |
| **Experimental Analysis** | `Test_Meta_ViT_on_Dataset.py` | Tests the novel enhancements against real image data (CIFAR-10). |
| **Basic ViT Comparison** | `Compare_Three_ViT_Models.py` | Python script that runs base ViT inference comparisons. |
| **Basic Comparison Report** | `Final_Comparison_Report.md` | Output report comparing ViT-B/32, ViT-B/16, and ViT-L/14. |

## How to Run

1.  **Setup Environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Test Meta ViT Enhancements on Dataset (Novel Contribution):**
    ```bash
    python Test_Meta_ViT_on_Dataset.py
    ```
3.  **Run Basic ViT Comparison Experiment:**
    ```bash
    python Compare_Three_ViT_Models.py
    ```
