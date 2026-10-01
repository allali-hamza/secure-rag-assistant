#!/usr/bin/env python3
import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def ensure_pdftotext_available():
    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "pdftotext pas installé"
        )


def convert_pdf(pdf_path: Path, output_md: Path, layout: bool = True) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    text_path = output_md.with_suffix(".txt")

    commands = []
    if layout:
        commands.append(["pdftotext", "-layout", str(pdf_path), str(text_path)])
    commands.append(["pdftotext", str(pdf_path), str(text_path)])

    last_error = None
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"Avertissement : pdftotext a échoué pour {pdf_path} avec {cmd}.")
            if exc.stderr:
                print(exc.stderr.decode("utf-8", errors="ignore").strip())
    else:
        raise RuntimeError(
            f"Impossible de convertir {pdf_path} en texte avec pdftotext."
        )

    text = text_path.read_text(encoding="utf-8", errors="ignore")
    text_path.unlink()

    # Minimal markdown conversion: preserve paragraph breaks
    md_text = text.strip()
    output_md.write_text(md_text + "\n", encoding="utf-8")


def convert_folder(source_dir: Path, target_dir: Path, recurse: bool = True) -> None:
    ensure_pdftotext_available()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Le dossier source n'existe pas: {source_dir}")

    pdf_files = sorted(source_dir.rglob("*.pdf")) if recurse else sorted(source_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Aucun fichier PDF trouvé dans {source_dir}")
        return

    processed = 0
    failed = 0
    for pdf_file in pdf_files:
        relative = pdf_file.relative_to(source_dir)
        output_md = target_dir / relative.with_suffix(".md")
        output_md.parent.mkdir(parents=True, exist_ok=True)
        print(f"Conversion {pdf_file} -> {output_md}")
        try:
            convert_pdf(pdf_file, output_md)
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"Erreur : impossible de convertir {pdf_file} ({exc})")

    print(f"Conversion terminée : {processed} fichier(s) traité(s), {failed} fichier(s) en échec")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convertit les PDF de la KB France en fichiers Markdown dans un dossier converted."
    )
    parser.add_argument(
        "--source",
        default="config/kb",
        help="Dossier source contenant les PDF (par défaut config/kb)",
    )
    parser.add_argument(
        "--target",
        default="config/kb",
        help="Dossier de sortie Markdown (par défaut config/kb)",
    )
    parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="Ne pas parcourir les sous-dossiers",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)
    convert_folder(source_dir, target_dir, recurse=not args.no_recurse)


if __name__ == "__main__":
    main()
