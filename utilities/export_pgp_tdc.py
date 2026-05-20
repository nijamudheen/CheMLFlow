import argparse
import sys


def main(output_path: str) -> int:
    try:
        from tdc.single_pred import ADME
    except Exception as exc:
        print(
            "pytdc is not installed. Create a small export env and install pytdc to run this script.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    data = ADME(name="Pgp_Broccatelli")
    df = data.get_data()
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Pgp_Broccatelli from TDC to CSV.")
    parser.add_argument("output_path", type=str, help="Output CSV path.")
    args = parser.parse_args()
    raise SystemExit(main(args.output_path))
