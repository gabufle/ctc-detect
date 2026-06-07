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
        output_path: Path to write the report (HTML or PDF).
        threshold: Probability score threshold for calling a cell a CTC.
    """
    # Load results
    results_df = pd.read_csv(results_path)
    
    # Calculate statistics
    total_cells = len(results_df)
    ctc_count = (results_df['predicted_label'] == 1).sum()
    non_ctc_count = (results_df['predicted_label'] == 0).sum()
    uncertain_count = results_df['uncertain'].sum()
    
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
    ctc_count = (results_df['predicted_label'] == 1).sum()
    non_ctc_count = (results_df['predicted_label'] == 0).sum()
    uncertain_count = results_df['uncertain'].sum()
    
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
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-box {{ border: 1px solid #ddd; padding: 15px; text-align: center; min-width: 150px; }}
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
                <div class="stat-number">{ctc_count}</div>
                <div class="stat-label">CTC Calls</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{non_ctc_count}</div>
                <div class="stat-label">Non-CTC Calls</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{uncertain_count}</div>
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