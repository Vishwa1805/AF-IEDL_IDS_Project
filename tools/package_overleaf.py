"""Create a self-contained ZIP archive for direct upload to Overleaf."""
import zipfile
from pathlib import Path

LATEX_DIR = Path("D:/Academics/7th-Sem/Computer And Network Security/Project/IEDL_IDS_Project/Paper/latex")
FIGURES_SRC = Path("C:/Users/vishw/.gemini/antigravity/brain/8f104935-84d6-46aa-bfaf-d8cfc1d61a5f/scratch/figures")
OUTPUT_ZIP_SCRATCH = Path("C:/Users/vishw/.gemini/antigravity/brain/8f104935-84d6-46aa-bfaf-d8cfc1d61a5f/scratch/AF_IEDL_Overleaf_Package.zip")

def build_zip():
    print(f"Building Overleaf package at: {OUTPUT_ZIP_SCRATCH}")
    with zipfile.ZipFile(OUTPUT_ZIP_SCRATCH, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add main.tex
        main_tex = LATEX_DIR / "main.tex"
        zf.write(main_tex, arcname="main.tex")
        print(f"Added: main.tex ({main_tex.stat().st_size} bytes)")
        
        # 2. Add references.bib
        ref_bib = LATEX_DIR / "references.bib"
        zf.write(ref_bib, arcname="references.bib")
        print(f"Added: references.bib ({ref_bib.stat().st_size} bytes)")
        
        # 3. Add all generated figures
        for fig in FIGURES_SRC.glob("*.png"):
            zf.write(fig, arcname=f"figures/{fig.name}")
            print(f"Added: figures/{fig.name} ({fig.stat().st_size} bytes)")
            
    print(f"Overleaf package successfully created! Total size: {OUTPUT_ZIP_SCRATCH.stat().st_size:,} bytes")

if __name__ == "__main__":
    build_zip()
