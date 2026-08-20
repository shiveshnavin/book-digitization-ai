import os
import sys
import argparse
from extractor import MathPdfExtractor

def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect & extract mathematical questions, graphics, and explanations from PDF."
    )
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, default="extracted_output", help="Directory where results will be saved")
    parser.add_argument("--dpi", type=int, default=300, help="Rendering resolution (default: 300 DPI for crisp math/diagrams)")
    parser.add_argument("--header_pct", type=float, default=0.04, help="Header margin to ignore as fraction of height (default: 0.04)")
    parser.add_argument("--footer_pct", type=float, default=0.04, help="Footer margin to ignore as fraction of height (default: 0.04)")

    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: Specified PDF file does not exist: {args.pdf}")
        sys.exit(1)

    extractor = MathPdfExtractor(
        dpi=args.dpi
    )
    extractor.process_pdf(args.pdf, output_dir=args.output)

if __name__ == "__main__":
    main()