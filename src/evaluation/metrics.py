import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the sense of python)
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold = roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

def compute_metrics(labels, preds, threshold=0.5):
    labels = np.array(labels)
    preds = np.array(preds)
    pred_labels = (preds > threshold).astype(int)
    
    acc = accuracy_score(labels, pred_labels)
    try:
        roc_auc = roc_auc_score(labels, preds)
    except ValueError:
        roc_auc = 0.0
        
    prec = precision_score(labels, pred_labels, zero_division=0)
    rec = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)
    
    try:
        eer = compute_eer(labels, preds)
    except Exception:
        eer = 0.0
        
    cm = confusion_matrix(labels, pred_labels)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "eer": float(eer),
        "confusion_matrix": cm.tolist()
    }
