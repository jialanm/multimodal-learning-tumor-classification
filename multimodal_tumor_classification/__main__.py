"""CLI entry point: python -m multimodal_tumor_classification <command> [options]"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="multimodal_tumor_classification",
        description="Multimodal breast cancer tumor grading from DCE-MRI and clinical features",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ovis2 subcommand ---
    ovis2_parser = subparsers.add_parser(
        "ovis2", help="Run Ovis2-4B VLM few-shot classification")
    ovis2_parser.add_argument(
        "--crop", choices=["proportional", "none", "fixed256"],
        default="proportional",
        help="Crop mode for DCE composites (default: proportional)")
    ovis2_parser.add_argument(
        "--num-patients", type=int, default=None,
        help="Limit number of patients (default: all)")
    ovis2_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: output/ovis2_<crop>)")

    # --- swin subcommand ---
    swin_parser = subparsers.add_parser(
        "swin", help="Run Swin-Tiny + clinical MLP baseline")
    swin_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: output/swin_baseline)")
    swin_parser.add_argument(
        "--composites-dir", type=str, default=None,
        help="Path to composites directory (default: output/ovis2_fixed256_crop/composites)")
    swin_parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max training epochs (default: 200)")
    swin_parser.add_argument(
        "--patient-list", type=str, default=None,
        help="Path to text file with one patient ID per line to filter patients")

    args = parser.parse_args()

    if args.command == "ovis2":
        from .ovis2_pipeline import run_ovis2_pipeline
        run_ovis2_pipeline(
            crop_mode=args.crop,
            num_patients=args.num_patients,
            output_dir=args.output_dir,
        )

    elif args.command == "swin":
        from .swin_pipeline import run_swin_pipeline
        run_swin_pipeline(
            output_dir=args.output_dir,
            composites_dir=args.composites_dir,
            num_epochs=args.epochs,
            patient_list=args.patient_list,
        )


if __name__ == "__main__":
    main()
