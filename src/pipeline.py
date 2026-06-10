# src/pipeline.py
import argparse
import subprocess
import sys


def run_module(module: str, args: list[str] | None = None):
    if args is None:
        args = []

    cmd = [sys.executable, "-m", module] + args

    print()
    print("=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Falló {module} con código {result.returncode}")


def cmd_predict(args):
    run_module("src.run_daily", ["--date", args.date])


def cmd_evaluate(args):
    run_module("src.evaluator", ["--date", args.date])


def cmd_sync_results(args):
    cli_args = []

    if args.print:
        cli_args.append("--print")

    run_module("src.sync_worldcup_results", cli_args)


def cmd_retrain(args):
    run_module("src.retrain")

def cmd_fetch_odds(args):
    run_module("src.odds_api", ["--date", args.date])
    
def cmd_telegram(args):
    cli_args = ["send"]

    if args.date:
        cli_args += ["--date", args.date]

    run_module("src.telegram_notify", cli_args)

def cmd_full(args):
    if args.eval_date:
        run_module("src.evaluator", ["--date", args.eval_date])
        run_module("src.sync_worldcup_results", ["--print"])
        run_module("src.retrain")

    if args.fetch_odds:
        run_module("src.odds_api", ["--date", args.predict_date])

    run_module("src.run_daily", ["--date", args.predict_date])

    if args.telegram:
        run_module("src.telegram_notify", ["send", "--date", args.predict_date])


def main():
    parser = argparse.ArgumentParser(
        description="World Cup predictor pipeline orchestrator"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_predict = subparsers.add_parser("predict")
    p_predict.add_argument("--date", required=True)
    p_predict.set_defaults(func=cmd_predict)

    p_evaluate = subparsers.add_parser("evaluate")
    p_evaluate.add_argument("--date", required=True)
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_sync = subparsers.add_parser("sync-results")
    p_sync.add_argument("--print", action="store_true")
    p_sync.set_defaults(func=cmd_sync_results)

    p_retrain = subparsers.add_parser("retrain")
    p_retrain.set_defaults(func=cmd_retrain)
    
    p_odds = subparsers.add_parser("fetch-odds")
    p_odds.add_argument("--date", required=True)
    p_odds.set_defaults(func=cmd_fetch_odds)
    
    p_telegram = subparsers.add_parser("telegram")
    p_telegram.add_argument("--date", required=False)
    p_telegram.set_defaults(func=cmd_telegram)

    p_full = subparsers.add_parser("full")
    p_full.add_argument("--eval-date", required=False)
    p_full.add_argument("--predict-date", required=True)
    p_full.add_argument("--fetch-odds", action="store_true")
    p_full.add_argument("--telegram", action="store_true")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()