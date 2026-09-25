import os
import yaml
import argparse
import torch
from datetime import datetime

from fqf_iqn_qrdqn.agent import QRDQNAgent, IQNAgent, FQFAgent
from alphagen.data.expression import Feature, FeatureType, Ref, StockData
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv


def run(args):

    # torch.cuda.set_device(args.cuda)
    config_path = os.path.join('config', f'{args.model}.yaml')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    device = torch.device(f'cuda')
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    instruments: float = 'csi300'

    data_train = StockData(instrument=instruments,
                           start_time='2010-01-01',
                           end_time='2019-12-31')
    data_valid = StockData(instrument=instruments,
                           start_time='2020-01-01',
                           end_time='2020-12-31')
    data_test = StockData(instrument=instruments,
                          start_time='2021-01-01',
                          end_time='2022-12-31')
    train_calculator = QLibStockDataCalculator(data_train, target)
    valid_calculator = QLibStockDataCalculator(data_valid, target)
    test_calculator = QLibStockDataCalculator(data_test, target)
    train_pool = AlphaPool(capacity=args.pool,
                           calculator=train_calculator,
                           ic_lower_bound=None,
                           l1_alpha=5e-3)
    train_env = AlphaEnv(pool=train_pool, device=device, print_expr=True)

    # Specify the directory to log.
    name = args.model
    time = datetime.now().strftime("%Y%m%d-%H%M")
    if name == 'qrdqn':
        log_dir = os.path.join('AlphaQCM_data/csi300_logs',
                           f"pool_{args.pool}",
                           f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}")
    elif name == 'iqn':
        log_dir = os.path.join('AlphaQCM_data/csi300_logs',
                           f"pool_{args.pool}",
                           f"{name}-seed{args.seed}-{time}-N{config['K']}-lr{config['lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}")
    elif name == 'fqf':
        log_dir = os.path.join('AlphaQCM_data/csi300_logs',
                           f"pool_{args.pool}",
                           f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['quantile_lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}")

    # Create the agent and run.
    if name == 'qrdqn':
        agent = QRDQNAgent(env=train_env,
                           valid_calculator=valid_calculator,
                           test_calculator=test_calculator,
                           log_dir=log_dir,
                           seed=args.seed,
                           cuda=True, **config)
    elif name == 'iqn':
        agent = IQNAgent(env=train_env,
                         valid_calculator=valid_calculator,
                         test_calculator=test_calculator,
                         log_dir=log_dir,
                         seed=args.seed,
                         cuda=True, **config)
    elif name == 'fqf':
        agent = FQFAgent(env=train_env,
                         valid_calculator=valid_calculator,
                         test_calculator=test_calculator,
                         log_dir=log_dir,
                         seed=args.seed,
                         cuda=True, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qrdqn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool', type=int, default=20)
    args = parser.parse_args()
    run(args)