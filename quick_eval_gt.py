# quick_eval_gt.py
import os
import argparse
import pandas as pd

from hvac_power_col_memory import batch_physical_role_review
from evaluation_gt import load_ground_truth_labels, evaluate_predictions

def sniff_dataframe_from_excel(xlsx_path, head_rows=10):
    xls = pd.ExcelFile(xlsx_path)
    dfs = []
    for s in xls.sheet_names:
        try:
            dfs.append(pd.read_excel(xlsx_path, sheet_name=s, nrows=head_rows))
        except Exception:
            pass
    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()




def main():
    ap = argparse.ArgumentParser(description="Evaluate semantic recognition with manual ground truth.")
    ap.add_argument("--excel", required=True, help="Path to Excel data (e.g., Netsugen3_202405.xlsx)")
    ap.add_argument("--gt", required=True, help="Path to ground truth CSV (column,role)")
    ap.add_argument("--rows", type=int, default=10, help="Header sniff rows per sheet (default: 10)")
    ap.add_argument("--outdir", default="output", help="Directory to save reports (default: output)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 只需列名即可做语义识别（提高速度）
    df = sniff_dataframe_from_excel(args.excel, head_rows=args.rows)

    # 跑列角色识别（AI+兜底）。无 OpenAI client 时 client=None 也能跑兜底逻辑
    _, ai_roles,_ = batch_physical_role_review(df, client=None)

    # 评测
    gt_map = load_ground_truth_labels(args.gt)
    report_df, macro, micro, conf, coverage, missing_cols = evaluate_predictions(ai_roles, gt_map)

    print("\n=== Semantic recognition (Ground Truth) ===")
    print(f"[Coverage] {coverage*100:.1f}% of labeled columns received predictions.")
    print(f"[Macro]  P={macro['precision']:.3f} R={macro['recall']:.3f} F1={macro['f1']:.3f} (support={macro['support']})")
    print(f"[Micro]  P={micro['precision']:.3f} R={micro['recall']:.3f} F1={micro['f1']:.3f} (support={micro['support']})")
    if missing_cols:
        print(f"[WARN] {len(missing_cols)} labeled columns not predicted (name mismatch or filtered).")

    # 保存报表
    per_class_csv = os.path.join(args.outdir, "semantic_report_per_class.csv")
    conf_csv = os.path.join(args.outdir, "semantic_confusion_matrix.csv")
    report_df.to_csv(per_class_csv, index=False)
    conf.to_csv(conf_csv)
    print(f"[Saved] {per_class_csv}")
    print(f"[Saved] {conf_csv}")

if __name__ == "__main__":
    main()
