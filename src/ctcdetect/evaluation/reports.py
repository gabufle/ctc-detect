"""Report generation for CTC-Detect.

Produces clinical summary reports with key statistics,
threshold-based CTC calls, and QC metrics.
"""

from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()


def generate_report(
    results_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    """Generate a clinical summary report from detection results.

    Args:
        results_path: Path to detection results (CSV with scores).
        output_path: Path to write the report (HTML or text).
        threshold: Probability score threshold for calling a cell a CTC.
    """
    # Load results
    results_df = pd.read_csv(results_path)

    # Calculate statistics
    total_cells = len(results_df)
    ctc_count = int((results_df['predicted_label'] == 1).sum())
    non_ctc_count = int((results_df['predicted_label'] == 0).sum())
    uncertain_count = int(results_df['uncertain'].sum())

    mean_score = results_df['ctc_probability'].mean()
    median_score = results_df['ctc_probability'].median()
    std_score = results_df['ctc_probability'].std()

    # Create plain text summary
    report_lines = [
        "=" * 50,
        "CTC-DETECT SUMMARY REPORT",
        "=" * 50,
        f"Total cells analyzed: {total_cells}",
        f"CTC calls (predicted label = 1): {ctc_count} ({ctc_count/total_cells*100:.1f}%)",
        f"Non-CTC calls (predicted label = 0): {non_ctc_count} ({non_ctc_count/total_cells*100:.1f}%)",
        f"Uncertain predictions: {uncertain_count} ({uncertain_count/total_cells*100:.1f}%)",
        "",
        "CTC Probability Score Statistics:",
        f"  Mean: {mean_score:.4f}",
        f"  Median: {median_score:.4f}",
        f"  Standard deviation: {std_score:.4f}",
        f"  Minimum: {results_df['ctc_probability'].min():.4f}",
        f"  Maximum: {results_df['ctc_probability'].max():.4f}",
        "",
        f"Threshold used for CTC calls: {threshold}",
        f"Cells above threshold: {(results_df['ctc_probability'] > threshold).sum()}",
        f"Cells below or equal to threshold: {(results_df['ctc_probability'] <= threshold).sum()}",
        "",
        "=" * 50,
        "END OF REPORT",
        "=" * 50
    ]

    # Write report
    report_text = "\n".join(report_lines)

    # Ensure output directory exists
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / "summary.txt"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_file = output_path

    with open(report_file, 'w') as f:
        f.write(report_text)

    console.print(f"[green]✓[/green] Summary report saved to {report_file}")


def generate_html_report(
    results_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    """Generate an HTML clinical summary report from detection results.

    Args:
        results_path: Path to detection results (CSV with scores).
        output_path: Path to write the HTML report.
        threshold: Probability score threshold for calling a cell a CTC.
    """
    # Load results
    results_df = pd.read_csv(results_path)

    # Calculate statistics
    total_cells = len(results_df)
    ctc_count = int((results_df['predicted_label'] == 1).sum())
    non_ctc_count = int((results_df['predicted_label'] == 0).sum())
    uncertain_count = int(results_df['uncertain'].sum())

    mean_score = results_df['ctc_probability'].mean()
    median_score = results_df['ctc_probability'].median()
    std_score = results_df['ctc_probability'].std()

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CTC-Detect Summary Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px auto; max-width: 800px; padding: 0 20px; color: #333; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            h1 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 8px; }}
            h2 {{ color: #444; margin-top: 32px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; flex-wrap: wrap; }}
            .stat-box {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; min-width: 150px; margin: 8px; background: #fafafa; }}
            .stat-number {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            .section {{ margin: 30px 0; }}
            .section-title {{ border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>CTC-Detect Summary Report</h1>
            <p>Circulating Tumor Cell Detection from scRNA-seq</p>
        </div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{total_cells}</div>
                <div class="stat-label">Total Cells</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #dc3545;">{ctc_count}</div>
                <div class="stat-label">CTC Calls</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #28a745;">{non_ctc_count}</div>
                <div class="stat-label">Non-CTC Calls</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #ffc107;">{uncertain_count}</div>
                <div class="stat-label">Uncertain</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">CTC Probability Score Statistics</h2>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Mean</td><td>{mean_score:.4f}</td></tr>
                <tr><td>Median</td><td>{median_score:.4f}</td></tr>
                <tr><td>Standard Deviation</td><td>{std_score:.4f}</td></tr>
                <tr><td>Minimum</td><td>{results_df['ctc_probability'].min():.4f}</td></tr>
                <tr><td>Maximum</td><td>{results_df['ctc_probability'].max():.4f}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Threshold Analysis</h2>
            <p>Threshold used for CTC calls: {threshold}</p>
            <table>
                <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
                <tr><td>Above Threshold</td><td>{(results_df['ctc_probability'] > threshold).sum()}</td><td>{(results_df['ctc_probability'] > threshold).sum()/total_cells*100:.1f}%</td></tr>
                <tr><td>At or Below Threshold</td><td>{(results_df['ctc_probability'] <= threshold).sum()}</td><td>{(results_df['ctc_probability'] <= threshold).sum()/total_cells*100:.1f}%</td></tr>
            </table>
        </div>

        <div class="footer">
            <p>Generated by CTC-Detect - Powered by Geneformer</p>
        </div>
    </body>
    </html>
    """

    # Write HTML report
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / "summary.html"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == "":
            report_file = output_path.with_suffix(".html")
        else:
            report_file = output_path

    with open(report_file, 'w') as f:
        f.write(html_content)

    console.print(f"[green]✓[/green] HTML report saved to {report_file}")


def generate_eval_report(metrics: dict, output_path: Path) -> None:
    """Generate a text evaluation report from metrics.

    Parameters
    ----------
    metrics : dict
        Output from ``compute_metrics``.
    output_path : Path
        Directory to write ``eval_report.txt``.
    """
    m = metrics
    report_lines = [
        "=" * 60,
        "CTC-DETECT EVALUATION REPORT",
        "=" * 60,
        "",
        f"Total cells evaluated: {m['n_total']}",
        f"Ground truth CTCs: {m['n_positive']} ({m['prevalence']*100:.1f}%)",
        f"Ground truth non-CTCs: {m['n_negative']} ({(1-m['prevalence'])*100:.1f}%)",
        "",
        f"Threshold: {m['threshold']}",
        "",
        "Metrics:",
        f"  AUROC:        {m['auroc']:.4f}",
        f"  AUPRC:        {m['auprc']:.4f}",
        f"  F1:           {m['f1']:.4f}",
        f"  Sensitivity:  {m['sensitivity']:.4f}",
        f"  Specificity:  {m['specificity']:.4f}",
        f"  PPV:          {m['ppv']:.4f}",
        f"  NPV:          {m['npv']:.4f}",
        "",
        f"Confusion Matrix (threshold={m['threshold']}):",
        "                 Predicted",
        "                 non-CTC    CTC",
        f"  Actual non-CTC  {m['tn']:6d}  {m['fp']:6d}",
        f"  Actual CTC      {m['fn']:6d}  {m['tp']:6d}",
        "",
        f"  TP: {m['tp']}  FP: {m['fp']}",
        f"  FN: {m['fn']}  TN: {m['tn']}",
        "",
        "=" * 60,
    ]
    report_file = output_path / "eval_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines) + "\n")


def generate_eval_html_report(metrics: dict, output_path: Path) -> None:
    """Generate an HTML evaluation report.

    Parameters
    ----------
    metrics : dict
        Output from ``compute_metrics``.
    output_path : Path
        Directory to write ``eval_report.html``.
    """
    m = metrics
    prev_pct = m['prevalence'] * 100
    non_prev_pct = (1 - m['prevalence']) * 100

    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>CTC-Detect Evaluation Report</title>\n'
        '<style>\n'
        'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }\n'
        'h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 8px; }\n'
        'h2 { color: #444; margin-top: 32px; }\n'
        'table { border-collapse: collapse; width: 100%; margin: 16px 0; }\n'
        'th, td { border: 1px solid #ddd; padding: 10px 14px; text-align: left; }\n'
        'th { background: #f8f9fa; font-weight: 600; }\n'
        'tr:nth-child(even) { background: #fafafa; }\n'
        '.value { font-weight: bold; color: #1a73e8; font-size: 1.2em; }\n'
        '.cm-cell { text-align: center; font-size: 1.3em; font-weight: bold; }\n'
        '.cm-header { background: #e8f0fe; }\n'
        '.img-container { text-align: center; margin: 20px 0; }\n'
        '.img-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '<h1>CTC-Detect Evaluation Report</h1>\n'
        '\n'
        '<h2>Dataset Summary</h2>\n'
        '<table>\n'
        '  <tr><th>Metric</th><th>Value</th></tr>\n'
        f'  <tr><td>Total cells</td><td class="value">{m["n_total"]}</td></tr>\n'
        f'  <tr><td>Ground truth CTCs</td><td>{m["n_positive"]} ({prev_pct:.1f}%)</td></tr>\n'
        f'  <tr><td>Ground truth non-CTCs</td><td>{m["n_negative"]} ({non_prev_pct:.1f}%)</td></tr>\n'
        f'  <tr><td>Threshold</td><td>{m["threshold"]}</td></tr>\n'
        '</table>\n'
        '\n'
        '<h2>Performance Metrics</h2>\n'
        '<table>\n'
        '  <tr><th>Metric</th><th>Value</th></tr>\n'
        f'  <tr><td>AUROC</td><td class="value">{m["auroc"]:.4f}</td></tr>\n'
        f'  <tr><td>AUPRC</td><td class="value">{m["auprc"]:.4f}</td></tr>\n'
        f'  <tr><td>F1 Score</td><td>{m["f1"]:.4f}</td></tr>\n'
        f'  <tr><td>Sensitivity (Recall)</td><td>{m["sensitivity"]:.4f}</td></tr>\n'
        f'  <tr><td>Specificity</td><td>{m["specificity"]:.4f}</td></tr>\n'
        f'  <tr><td>PPV (Precision)</td><td>{m["ppv"]:.4f}</td></tr>\n'
        f'  <tr><td>NPV</td><td>{m["npv"]:.4f}</td></tr>\n'
        '</table>\n'
        '\n'
        f'<h2>Confusion Matrix (threshold={m["threshold"]})</h2>\n'
        '<table>\n'
        '  <tr><th></th><th class="cm-header">Predicted non-CTC</th><th class="cm-header">Predicted CTC</th></tr>\n'
        f'  <tr><th class="cm-header">Actual non-CTC</th><td class="cm-cell">{m["tn"]}</td><td class="cm-cell">{m["fp"]}</td></tr>\n'
        f'  <tr><th class="cm-header">Actual CTC</th><td class="cm-cell">{m["fn"]}</td><td class="cm-cell">{m["tp"]}</td></tr>\n'
        '</table>\n'
        '\n'
        '<h2>ROC Curve</h2>\n'
        '<div class="img-container">\n'
        '  <img src="roc.png" alt="ROC Curve">\n'
        '</div>\n'
        '\n'
        '<h2>Precision-Recall Curve</h2>\n'
        '<div class="img-container">\n'
        '  <img src="pr.png" alt="Precision-Recall Curve">\n'
        '</div>\n'
        '\n'
        '</body>\n'
        '</html>'
    )

    html_file = output_path / "eval_report.html"
    with open(html_file, "w") as f:
        f.write(html)


__all__ = [
    "generate_report",
    "generate_html_report",
    "generate_eval_report",
    "generate_eval_html_report",
]
