"""Run official AlphaForge stage 1 (generative-predictive mining) on FactorBench data.

Mirrors ``train_AFF.py::main`` from the official repository (commit d0cfc27),
with qlib initialization replaced by the shared data contract. All search
hyperparameters are the official defaults.

Two research-semantics deviations, both documented in the vendor notes:

- **Target.** The official target is ``Ref(vwap,-21)/Ref(vwap,-1) - 1``
  (vwap-to-vwap with a one-day execution delay). FactorBench mines against the
  shared contract target ``Ref(close,-20)/close - 1`` so that every method
  optimizes the quantity it is later scored on.
- **Round cap.** The official loop spins until the zoo is full with no upper
  bound on rounds; ``max_rounds`` bounds it so a sweep cannot hang forever.

Usage:
    python -m factor_mining.methods.alphaforge.run --market sp100 --seed 0
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import fire
import numpy as np
import torch

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alphaforge.data import PanelStockData
from factor_mining.methods.alphaforge.vendored import use_alphaforge

use_alphaforge()

from alphagen.data.expression import Feature, Ref  # noqa: E402
from alphagen.rl.env.wrapper import SIZE_ACTION  # noqa: E402
from alphagen.utils.correlation import batch_pearsonr, batch_ret  # noqa: E402
from alphagen.utils.pytorch_utils import normalize_by_day  # noqa: E402
from alphagen.utils.random import reseed_everything  # noqa: E402
from alphagen_qlib.stock_data import FeatureType  # noqa: E402
from gan.dataset import Collector  # noqa: E402
from gan.network.generater import NetG_DCGAN, train_network_generator  # noqa: E402
from gan.network.masker import NetM  # noqa: E402
from gan.network.predictor import NetP, train_net_p_with_weight  # noqa: E402
from gan.utils import Builders, blds_list_to_tensor, filter_valid_blds, save_blds  # noqa: E402

OFFICIAL_REPO = "https://github.com/DulyHao/AlphaForge"
OFFICIAL_COMMIT = "d0cfc27df23c60f271bc885fd43027b86b787746"


def pre_process_y(y: torch.Tensor) -> torch.Tensor:
    """Official ``train_AFF.py::pre_process_y`` — rescale scores to [0, 100]."""
    min_y = 0
    max_y = y.flatten().max()
    return (y - min_y) / (max_y - min_y) * 100


def get_metric(zoo_blds, device, corr_thresh: float = 0.5, metric_target: str = "ic"):
    """Official ``train_AFF.py::get_metric``, verbatim in behavior.

    Returns a scorer that maps (factor, target) to the search score: |mean IC|,
    zeroed when the factor is mostly NaN, near-constant, or its daily-return
    series correlates above ``corr_thresh`` with anything already in the zoo.
    """
    n_blds = len(zoo_blds)
    if n_blds > 0:
        n_days = len(zoo_blds.ret_list[0])
        existed = np.vstack(zoo_blds.ret_list)  # (n_blds, n_days)
        existed = torch.from_numpy(existed).to(device)
        assert existed.shape == (n_blds, n_days)
        print(f"existed n_blds == {n_blds}")
    else:
        print("n_blds == 0")

    def get_score(fct, tgt):
        # Upstream re-binds metric_target inside the closure, so the outer
        # argument is inert; kept as-is to preserve official behavior.
        metric_target = "ic"
        ret = batch_ret(fct, tgt)
        ic = batch_pearsonr(fct, tgt)

        ic_mean = ic.mean().abs().item()
        icir = (ic_mean / ic.std()).item()
        ret_mean = ret.mean().abs().item()
        ret_ir = (ret_mean / ret.std()).item()
        sharpe = ((ret_mean - 0.03 / 252) / ret.std() * np.sqrt(252)).item()

        def invalid_to_zero(x):
            if not np.isfinite(x):
                return 0.0
            return max(x, 0.0)

        multi_score = {
            "ic": ic_mean,
            "icir": icir,
            "ret": ret_mean,
            "sharpe": sharpe,
            "retir": ret_ir,
        }
        multi_score = {k: invalid_to_zero(v) for k, v in multi_score.items()}
        score = multi_score[metric_target]

        # too many nan
        if torch.isfinite(fct[0]).sum() / torch.isfinite(tgt[0]).sum() < 0.8:
            score = 0.0
        # unique ratio too small
        elif len(torch.unique(fct[0])) / len(torch.unique(tgt[0])) < 0.01:
            score = 0.0

        if n_blds > 0 and score > 0.0:
            assert len(ret.shape) == 1, f"{ret.shape},{n_days}"
            assert len(ret) == n_days, f"{ret.shape},{n_days}"
            all_matrix = torch.concatenate([existed, ret[None]], dim=0)
            assert all_matrix.shape == (n_blds + 1, n_days), f"{all_matrix.shape}"
            corr_score = torch.corrcoef(all_matrix)[-1, :-1].abs().max().item()
            if corr_score > corr_thresh:
                score = 0.0

        return {
            "score": score,
            "ret": ret.detach().cpu().numpy(),
            "multi_score": multi_score,
        }

    return get_score


def _signed_train_ics(zoo_blds, data, target_expr) -> list[float]:
    """Signed mean IC per zoo factor on the training window.

    The search score is |mean IC| (``get_metric``), which would make the
    ``train.ic_delta`` fidelity check in factor_bench.eval.compare meaningless
    for negative-IC factors. Recomputed here with the same normalization and
    the same ``batch_pearsonr``, just without the absolute value.
    """
    target_value = normalize_by_day(target_expr.evaluate(data))
    ics: list[float] = []
    for expr in zoo_blds.exprs:
        factor = normalize_by_day(expr.evaluate(data))
        ics.append(batch_pearsonr(factor, target_value).mean().item())
    return ics


def run_alphaforge(
    market: str = "csi300",
    seed: int = 0,
    zoo_size: int = 100,
    corr_thresh: float = 0.7,
    ic_thresh: float = 0.03,
    icir_thresh: float = 0.1,
    max_rounds: int = 50,
    device: str = "cpu",
    out_dir: str = "out/mining",
    verbose: int = 1,
) -> Path:
    """Mine one AlphaForge factor zoo and write factors.json. Returns its path."""
    t0 = time.time()
    reseed_everything(seed)

    panel = load_panel(market)
    TARGET.validate(panel)  # leakage guard: cached target matches its definition

    train_start, train_end = DEFAULT_SPLIT.train
    data = PanelStockData(panel, train_start, train_end, device=torch.device(device),
                          market=market)
    # Shared contract target (see module docstring for the deviation note).
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    save_path = Path(out_dir) / "alphaforge" / f"{market}_{zoo_size}_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    class cfg:  # noqa: N801 — official config object, name kept for traceability
        name = f"{market}_{zoo_size}_{seed}"

        max_len = 20

        batch_size = 256
        potential_size = 100
        n_layers = 2
        d_model = 128
        dropout = 0.2
        num_factors = zoo_size

        # generator configuration
        num_epochs_g = 200
        g_es_score = "max"  # max mean std combined
        g_es = 10
        g_hidden = 128
        g_lr = 1e-3

        # predictor configuration
        p_hidden = 128
        p_lr = 1e-3
        es_p = 10
        batch_size_p = 64
        num_epochs_p = 100
        data_keep_p = 20000

        f_corr_thresh = corr_thresh  # threshold to penalize the correlation
        f_add_thresh = corr_thresh  # threshold to add new exprs to the zoo
        f_score_thresh = ic_thresh  # threshold to filter exprs in the zoo
        f_multi_score_thresh = {"icir": icir_thresh}

        # loss configuration
        l_pred = 1.0
        l_simi = 10.0
        l_simi_thresh = 0.4

        l_potential = 10.0
        l_potential_thresh = 0.4
        l_potential_epsilon = 1e-7

        l_entropy = 0

    cfg.device = device  # official cfg hardcodes 'cuda:0'; FactorBench parameterizes

    print(f"seed:{seed},name:{cfg.name}")

    def random_call(z):
        return z.normal_()

    netG = NetG_DCGAN(  # noqa: N806 — official variable names
        n_chars=SIZE_ACTION,
        latent_size=cfg.potential_size,
        seq_len=cfg.max_len,
        hidden=cfg.g_hidden,
    ).to(cfg.device)
    netM = NetM(max_len=cfg.max_len, size_action=SIZE_ACTION).to(cfg.device)  # noqa: N806
    netP = NetP(  # noqa: N806
        n_chars=SIZE_ACTION, seq_len=cfg.max_len, hidden=cfg.p_hidden
    ).to(cfg.device)

    z = torch.zeros([cfg.batch_size, cfg.potential_size]).to(cfg.device)
    random_call(z)

    # initialize the zoo
    zoo_blds = Builders(0, max_len=cfg.max_len, n_actions=SIZE_ACTION)
    metric = get_metric(zoo_blds, device=cfg.device, corr_thresh=cfg.f_corr_thresh)
    empty_metric = get_metric(
        Builders(0, max_len=cfg.max_len, n_actions=SIZE_ACTION),
        device=cfg.device,
        corr_thresh=cfg.f_corr_thresh,
    )

    coll = Collector(seq_len=cfg.max_len, n_actions=SIZE_ACTION)
    coll.reset(data, target, metric)
    coll.collect_target_num(netG, netM, z, data, target, metric,
                            target_num=10000, reset_net=True, drop_invalid=False,
                            randomly=False, random_method=random_call, max_iter=200)

    # train and mine until the zoo is full
    t = 0
    while len(zoo_blds) < cfg.num_factors and t < max_rounds:
        if not zoo_blds.examined:
            print(" zoo_blds not examined")
            zoo_blds.evaluate(data, target, empty_metric, verbose=True)

        # update the metric for the current zoo
        metric = get_metric(zoo_blds, device=cfg.device, corr_thresh=cfg.f_corr_thresh)

        # prepare data to train predictor
        coll.blds.evaluate(data, target, metric, verbose=True)
        if coll.blds_bak.batch_size > cfg.data_keep_p:
            print(f"sample datas to keep {coll.blds_bak.batch_size}to{cfg.data_keep_p}")
            indices = np.random.choice(
                np.arange(coll.blds_bak.batch_size), cfg.data_keep_p, replace=False
            )
            coll.blds_bak = coll.blds_bak.filter_by_index(indices)
        coll.blds_bak.evaluate(data, target, metric, verbose=True)

        if coll.blds_bak.batch_size > 0:
            # give the current builders more weight when training the predictor
            blds_list = [coll.blds_bak, coll.blds]
            weight_list = [1.0, 2.0]
        else:
            blds_list = [coll.blds]
            weight_list = [1.0]

        x, y, weights = blds_list_to_tensor(blds_list, weight_list)
        y = pre_process_y(y)

        # train predictor
        netP.initialize_parameters()
        train_net_p_with_weight(cfg, netP, x, y, weights, lr=cfg.p_lr)

        # train generator
        netG.initialize_parameters()
        blds_in_train = train_network_generator(
            netG, netM, netP, cfg, data, target, t, random_method=random_call,
            metric=metric, lr=cfg.g_lr, n_actions=SIZE_ACTION,
        )

        # generate new alpha factors from the current generator
        coll.reset(data, target, metric)
        coll.collect_target_num(netG, netM, z, data, target, metric,
                                target_num=1000, reset_net=False, drop_invalid=False,
                                randomly=False, random_method=random_call, max_iter=100)

        lengh_s = {"train": len(blds_in_train), "new": len(coll.blds)}
        coll.blds = coll.blds + blds_in_train
        coll.blds.drop_duplicated()
        lengh_s["all_new"] = len(coll.blds)
        print(f"{lengh_s['train']} (train) + {lengh_s['new']} (new)  =  "
              f"{lengh_s['all_new']} (all_new)")

        # keep the valid factors found during training and generation
        new_zoo = filter_valid_blds(
            coll.blds,
            corr_thresh=cfg.f_add_thresh,
            score_thresh=cfg.f_score_thresh,
            multi_score_thresh=cfg.f_multi_score_thresh,
            device=cfg.device,
            verbose=True,
        )
        lengh_s["zoo_prev"] = len(zoo_blds)
        zoo_blds = zoo_blds + new_zoo
        print(f" zoo_prev:{lengh_s['zoo_prev']},all_new:{len(new_zoo)},current:{len(zoo_blds)}")
        zoo_blds.evaluate(data, target, empty_metric, verbose=True)
        if t % 5 == 2:
            print("#" * 20, "zoo_rebalance")
            zoo_blds = filter_valid_blds(
                zoo_blds,
                corr_thresh=cfg.f_add_thresh,
                score_thresh=cfg.f_score_thresh,
                multi_score_thresh=cfg.f_multi_score_thresh,
                device=cfg.device,
                verbose=False,
            )
        save_blds(zoo_blds, str(save_path), "zoo_final")

        # randomly generate some factors to promote exploration
        coll.collect_target_num(netG, netM, z, data, target, metric,
                                target_num=1000, reset_net=False, drop_invalid=False,
                                randomly=True, random_method=random_call, max_iter=100)

        del x, y, weights
        gc.collect()
        torch.cuda.empty_cache()
        t += 1

    if len(zoo_blds) < cfg.num_factors:
        print(f"[alphaforge] max_rounds={max_rounds} reached with "
              f"{len(zoo_blds)}/{cfg.num_factors} factors; exporting what was mined")

    empty_blds = Builders(0, max_len=cfg.max_len, n_actions=SIZE_ACTION)
    metric = get_metric(empty_blds, device=cfg.device, corr_thresh=cfg.f_corr_thresh)
    zoo_blds.evaluate(data, target, metric, verbose=True)
    save_blds(zoo_blds, str(save_path), "zoo_final")

    signed_ics = _signed_train_ics(zoo_blds, data, target)
    factors = [
        ExpressionFactor(
            name=f"alphaforge_{market}_s{seed}_{i:02d}",
            expression=zoo_blds.exprs_str[i],
            grammar="alphaforge-v1",
            # Stage 1 emits an unweighted zoo; combination is stage 2, whose
            # weights are re-fit every day and so cannot live in this contract.
            weight=None,
            train_metrics={
                "ic": signed_ics[i],
                "ic_abs": float(zoo_blds.scores[i]),
                **{k: float(v) for k, v in zoo_blds.multi_scores[i].items() if k != "ic"},
            },
        )
        for i in range(len(zoo_blds))
    ]
    run = MiningRun(
        method="alphaforge",
        paradigm="generative",
        output_form="expression",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget={
            "zoo_size": zoo_size,
            "rounds": t,
            "max_rounds": max_rounds,
            "max_len": cfg.max_len,
            "num_epochs_g": cfg.num_epochs_g,
            "num_epochs_p": cfg.num_epochs_p,
            "corr_thresh": corr_thresh,
            "ic_thresh": ic_thresh,
            "icir_thresh": icir_thresh,
        },
        wall_clock_seconds=time.time() - t0,
        provenance=Provenance(
            repo=OFFICIAL_REPO,
            commit=OFFICIAL_COMMIT,
            entry="train_AFF.py::main",
            notes="Official implementation vendored unmodified; qlib data layer "
            "replaced by the FactorBench shared contract, and the official "
            "vwap-to-vwap target replaced by the shared close-to-close "
            "forward_return_20d so all methods optimize what they are scored "
            "on. Stage 2 (dynamic combination) runs separately via "
            "factor_mining.methods.alphaforge.combine.",
        ),
        factors=factors,
    )
    path = run.save(save_path / "factors.json")
    if verbose:
        print(f"[alphaforge] {len(factors)} factors -> {path}")
    return path


if __name__ == "__main__":
    fire.Fire(run_alphaforge)
