# evaluation_gt.py
import csv
from collections import Counter
import pandas as pd

VALID_ROLES = [
    "heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow", "heat_source_power",
    "terminal_supply_air_temp", "terminal_return_air_temp", "terminal_air_volume", "fan_power"
]
OTHER = "other"  # for unknown/missing predictions

def load_ground_truth_labels(path):
    """
    CSV columns: column,role
    Returns: dict {column_name: role}
    """
    gt = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            col = (row.get("column") or "").strip()
            role = (row.get("role") or "").strip()
            if not col or not role:
                continue
            gt[col] = role
    return gt

def evaluate_predictions(pred_roles: dict, gt_map: dict, valid_roles=VALID_ROLES):
    """
    pred_roles: {column_name: role_predicted}
    gt_map:     {column_name: role_gt}
    Returns:
      report_df: per-class precision/recall/f1/support
      macro_avg: dict
      micro_avg: dict
      conf_mat:  pd.DataFrame confusion matrix (rows=GT, cols=PRED)
      coverage:  float (share of GT columns that received a prediction)
      missing_cols: list (GT columns not in predictions)
    """
    y_true, y_pred = [], []
    missing_cols = []

    for col, gt_role in gt_map.items():
        gt_role = gt_role if gt_role in valid_roles else OTHER
        pred_role = pred_roles.get(col, OTHER)
        if pred_role not in valid_roles:
            pred_role = OTHER
        y_true.append(gt_role)
        y_pred.append(pred_role)
        if col not in pred_roles:
            missing_cols.append(col)

    classes = valid_roles + [OTHER]
    tp = Counter(); fp = Counter(); fn = Counter(); support = Counter()

    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    rows = []
    for c in valid_roles:
        P = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        R = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
        rows.append(dict(role=c, precision=P, recall=R, f1=F1, support=support[c]))

    report_df = pd.DataFrame(rows).sort_values("role")

    valid = report_df[report_df["support"] > 0]
    macro = dict(
        precision=float(valid["precision"].mean()) if len(valid) else 0.0,
        recall=float(valid["recall"].mean()) if len(valid) else 0.0,
        f1=float(valid["f1"].mean()) if len(valid) else 0.0,
        support=int(valid["support"].sum())
    )

    total_tp = sum(tp[c] for c in VALID_ROLES)
    total_fp = sum(fp[c] for c in VALID_ROLES)
    total_fn = sum(fn[c] for c in VALID_ROLES)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2*micro_prec*micro_rec/(micro_prec+micro_rec) if (micro_prec+micro_rec)>0 else 0.0
    micro = dict(precision=micro_prec, recall=micro_rec, f1=micro_f1, support=int(sum(support[c] for c in VALID_ROLES)))

    conf_labels = VALID_ROLES + [OTHER]
    conf = pd.crosstab(
        pd.Series(y_true, name="GT", dtype="category"),
        pd.Series(y_pred, name="PRED", dtype="category"),
        dropna=False
    ).reindex(index=conf_labels, columns=conf_labels, fill_value=0)

    coverage = 1.0 - (len(missing_cols) / len(gt_map) if gt_map else 0.0)
    return report_df, macro, micro, conf, coverage, missing_cols
