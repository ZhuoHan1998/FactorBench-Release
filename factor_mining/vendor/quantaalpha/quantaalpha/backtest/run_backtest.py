#!/usr/bin/env python3
"""
Backtest entry script. Usage:
  quantaalpha backtest --factor-source alpha158_20
  quantaalpha backtest --factor-source custom --factor-json /path/to/factors.json
  python -m quantaalpha.backtest.run_backtest -c configs/backtest.yaml --factor-source alpha158_20
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded env: {env_file}")
else:
    print(f".env not found: {env_file}, using system env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Backtest V2 - full-featured backtest tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py -c config.yaml --factor-source alpha158_20
  python run_backtest.py -c config.yaml --factor-source custom --factor-json /path/to/factors.json
  python run_backtest.py -c config.yaml --factor-source combined --factor-json f1.json --factor-json f2.json
        """
    )
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path (YAML)')
    parser.add_argument('-s', '--factor-source', type=str,
                        choices=['alpha158', 'alpha158_20', 'alpha360', 'custom', 'combined'],
                        default=None, help='Factor source type (overrides config)')
    parser.add_argument('-j', '--factor-json', type=str, action='append', default=None,
                        help='Custom factor JSON path (can repeat)')
    parser.add_argument('-e', '--experiment', type=str, default=None, help='Experiment name (overrides config)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--dry-run', action='store_true', help='Load factors only, no backtest')
    parser.add_argument('--skip-uncached', action='store_true',
                        help='Skip uncached factors; use only cached factors for backtest')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    if args.factor_source == 'custom' and not args.factor_json:
        parser.error("--factor-source custom requires --factor-json")
    if args.factor_source == 'combined' and not args.factor_json:
        parser.error("--factor-source combined requires --factor-json")
    
    try:
        from quantaalpha.backtest.runner import BacktestRunner
        
        runner = BacktestRunner(str(config_path))
        
        if args.dry_run:
            print("\nDry Run - load factors only\n")
            from quantaalpha.backtest.factor_loader import FactorLoader
            if args.factor_source:
                runner.config['factor_source']['type'] = args.factor_source
            if args.factor_json:
                runner.config['factor_source']['custom']['json_files'] = args.factor_json
            
            loader = FactorLoader(runner.config)
            qlib_factors, custom_factors = loader.load_factors()
            
            print(f"\nFactor load result: Qlib {len(qlib_factors)}, custom (LLM) {len(custom_factors)}")
            if args.verbose:
                for name in list(qlib_factors.keys())[:10]:
                    print(f"  - {name}")
                if len(qlib_factors) > 10:
                    print(f"  ... and {len(qlib_factors) - 10} more")
                if custom_factors:
                    for factor in custom_factors[:5]:
                        print(f"  - {factor.get('factor_name', 'unknown')}")
                    if len(custom_factors) > 5:
                        print(f"  ... and {len(custom_factors) - 5} more")
        else:
            runner.run(
                factor_source=args.factor_source,
                factor_json=args.factor_json,
                experiment_name=args.experiment,
                skip_uncached=args.skip_uncached,
            )
            
    except KeyboardInterrupt:
        print("\nUser interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nBacktest failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

