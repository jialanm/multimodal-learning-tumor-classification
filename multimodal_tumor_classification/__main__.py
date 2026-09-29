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

    # --- dmgi subcommand ---
    dmgi_parser = subparsers.add_parser(
        "dmgi", help="Run DMGI multiplex graph classification")
    dmgi_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: output/dmgi_baseline)")
    dmgi_parser.add_argument(
        "--composites-dir", type=str, default=None,
        help="Path to composites directory (default: output/ovis2_fixed256_crop/composites)")
    dmgi_parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max training epochs (default: 2000)")
    dmgi_parser.add_argument(
        "--patient-list", type=str, default=None,
        help="Path to text file with one patient ID per line to filter patients")

    # --- graph-aug subcommand ---
    ga_parser = subparsers.add_parser(
        "graph-aug", help="Run graph-augmented Swin+MLP with smoothness regularization")
    ga_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: output/graph_augmented)")
    ga_parser.add_argument(
        "--composites-dir", type=str, default=None,
        help="Path to composites directory (default: output/ovis2_fixed256_crop/composites)")
    ga_parser.add_argument(
        "--epochs", type=int, default=None,
        help="Max training epochs (default: 200)")
    ga_parser.add_argument(
        "--patient-list", type=str, default=None,
        help="Path to text file with one patient ID per line to filter patients")
    ga_parser.add_argument(
        "--split", choices=["60-10-30", "70-10-20", "80-20"], default="60-10-30",
        help="Data split strategy (default: 60-10-30)")
    ga_parser.add_argument(
        "--mode", choices=["full-batch", "hybrid"], default="hybrid",
        help="Training mode: full-batch (original) or hybrid (mini-batch CE + graph smoothness, default: hybrid)")
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

    elif args.command == "dmgi":
        from .dmgi_pipeline import run_dmgi_pipeline
        run_dmgi_pipeline(
            output_dir=args.output_dir,
            composites_dir=args.composites_dir,
            num_epochs=args.epochs,
            patient_list=args.patient_list,
        )

    elif args.command == "graph-aug":
        from .graph_augmented_pipeline import run_graph_augmented_pipeline
        run_graph_augmented_pipeline(
            output_dir=args.output_dir,
            composites_dir=args.composites_dir,
            num_epochs=args.epochs,
            patient_list=args.patient_list,
            split=args.split,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
