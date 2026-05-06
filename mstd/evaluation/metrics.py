import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1, "precision_macro": p, "recall_macro": r}


def print_comparison_table(all_results: list):
    print(f"\n{'='*70}")
    print(f"  BACKBONE COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(
            f"  {r['model']:<12} {r['accuracy']:<12.4f} "
            f"{r['precision_macro']:<12.4f} {r['recall_macro']:<12.4f} "
            f"{r['f1_macro']:<12.4f}"
        )
    print(f"{'='*70}\n")
