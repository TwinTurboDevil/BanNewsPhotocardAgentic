# evaluate_mcnemar.py
import re
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def clean_image_id(val):
    """Strips file extensions (.jpg, .png, etc.) and leading/trailing spaces for clean merging."""
    s = str(val).strip()
    return re.sub(r'\.(jpg|jpeg|png|webp)$', '', s, flags=re.IGNORECASE)


def evaluate_thesis_results(
    ground_truth_csv,
    baseline_csv,
    agentic_csv,
    output_txt_path="mcnemar_report.txt",
):
    # 1. Load CSV files
    df_ground = pd.read_csv(ground_truth_csv)
    df_base = pd.read_csv(baseline_csv)
    df_agent = pd.read_csv(agentic_csv)

    # 2. Generate clean ID for matching across all datasets
    df_ground['clean_id'] = (
        df_ground['image_id'].apply(clean_image_id)
        if 'image_id' in df_ground.columns
        else df_ground['image_name'].apply(clean_image_id)
    )
    df_base['clean_id'] = df_base['image_name'].apply(clean_image_id)
    df_agent['clean_id'] = df_agent['image_name'].apply(clean_image_id)

    # 3. Select required columns to avoid merge conflicts
    df_ground_sub = df_ground[['clean_id', 'actual']].rename(
        columns={'actual': 'true_label'}
    )
    df_base_sub = df_base[['clean_id', 'prediction']].rename(
        columns={'prediction': 'pred_base'}
    )
    df_agent_sub = df_agent[['clean_id', 'prediction']].rename(
        columns={'prediction': 'pred_agent'}
    )

    # 4. Merge datasets on clean_id
    merged = df_ground_sub.merge(df_base_sub, on='clean_id').merge(
        df_agent_sub, on='clean_id'
    )

    # 5. Clean string values for accurate comparison
    merged['true_label'] = (
        merged['true_label'].astype(str).str.strip().str.capitalize()
    )
    merged['pred_base'] = (
        merged['pred_base'].astype(str).str.strip().str.capitalize()
    )
    merged['pred_agent'] = (
        merged['pred_agent'].astype(str).str.strip().str.capitalize()
    )

    # 6. Compute correctness (1 if correct, 0 if incorrect)
    merged['correct_base'] = (
        merged['pred_base'] == merged['true_label']
    ).astype(int)
    merged['correct_agent'] = (
        merged['pred_agent'] == merged['true_label']
    ).astype(int)

    # 7. Construct 2x2 Contingency Matrix
    n11 = (
        (merged['correct_base'] == 1) & (merged['correct_agent'] == 1)
    ).sum()  # Both Correct
    n10 = (
        (merged['correct_base'] == 1) & (merged['correct_agent'] == 0)
    ).sum()  # Baseline Correct, Agentic Wrong
    n01 = (
        (merged['correct_base'] == 0) & (merged['correct_agent'] == 1)
    ).sum()  # Agentic Correct, Baseline Wrong
    n00 = (
        (merged['correct_base'] == 0) & (merged['correct_agent'] == 0)
    ).sum()  # Both Wrong

    contingency_table = [[n11, n10], [n01, n00]]

    # 8. Run McNemar's Test (uses Exact Binomial test if discordant pairs < 25)
    discordant_pairs = n10 + n01
    use_exact = discordant_pairs < 25
    result = mcnemar(contingency_table, exact=use_exact, correction=True)

    # 9. Build and print comprehensive report
    acc_base = merged['correct_base'].mean()
    acc_agent = merged['correct_agent'].mean()

    report_lines = []

    def log(text=""):
        report_lines.append(text)
        print(text)

    log("=" * 65)
    log("          MCNEMAR STATISTICAL SIGNIFICANCE TEST REPORT")
    log("=" * 65)
    log(f"Total Samples Evaluated               : {len(merged)}")
    log(f"Baseline Model Accuracy               : {acc_base:.4%}")
    log(f"Agentic Model Accuracy                : {acc_agent:.4%}")
    log("-" * 65)
    log(f"Both Correct (n11)                    : {n11}")
    log(f"Baseline Correct, Agentic Wrong (n10) : {n10}")
    log(f"Agentic Correct, Baseline Wrong (n01) : {n01}")
    log(f"Both Wrong (n00)                      : {n00}")
    log("-" * 65)
    log(
        f"Test Variant Used                     : {'Exact Binomial' if use_exact else 'Chi-Square with Continuity Correction'}"
    )
    log(f"McNemar Test Statistic                : {result.statistic:.4f}")
    log(f"p-value                               : {result.pvalue:.6e}")
    log("=" * 65)

    if result.pvalue < 0.05:
        log(
            "VERDICT: STATISTICALLY SIGNIFICANT IMPROVEMENT (p < 0.05)\n"
            "The Agentic model shows a statistically significant performance gain over the Baseline."
        )
    else:
        log(
            "VERDICT: NO STATISTICAL SIGNIFICANCE (p >= 0.05)\n"
            "The performance difference between models is not statistically significant."
        )

    # Save output to text file using 'with open()'
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[+] Full report successfully saved to '{output_txt_path}'.")


if __name__ == "__main__":
    # Update these file paths to match your local directory structure
    evaluate_thesis_results(
        ground_truth_csv="all_annotations.csv",
        baseline_csv="qwen_baseline_result.csv",
        agentic_csv="qwen_agentic_verification_results.csv",
        output_txt_path="mcnemar_report_qwen.txt",
    )