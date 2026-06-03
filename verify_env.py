import sys
print(f"Python {sys.version}")

packages = [
    ("scanpy", "scanpy"),
    ("anndata", "anndata"),
    ("geneformer", "geneformer"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("umap", "umap-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("jupyter_core", "jupyter"),
    ("datasets", "datasets (HuggingFace)"),
]

ok = 0
fail = 0
failed = []
for mod, name in packages:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK] {name:30s} v{ver}")
        ok += 1
    except ImportError as e:
        print(f"  [FAIL] {name:30s} -> {e}")
        fail += 1
        failed.append(name)

print(f"\nResults: {ok} OK, {fail} failed")
if failed:
    print("FAILED:", ", ".join(failed))
    sys.exit(1)
else:
    print("ALL PACKAGES IMPORT SUCCESSFULLY")
