import argparse
import datetime
import os
import pickle
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch_geometric.data as gd
import wandb
import yaml
from rdkit.Chem import QED
from rdkit.Chem.rdchem import Mol as RDMol
from tdc import Oracle
from torch import Tensor
from torch_scatter import scatter_add

from synflownet import GFNTask, LogScalar, ObjectProperties
from synflownet.algo.reaction_sampling import SynthesisSampler
from synflownet.config import Config, init_empty, load_yaml, override_config
from synflownet.envs.synthesis_building_env import ReactionTemplateEnv, ReactionTemplateEnvContext
from synflownet.models import bengio2021flow
from synflownet.online_trainer import StandardOnlineTrainer
from synflownet.utils import metrics, sascore
from synflownet.utils.conditioning import MultiObjectiveWeightedPreferences, TemperatureConditional
from synflownet.utils.gpu_vina import QuickVina2GPU
from synflownet.utils.misc import get_worker_device
from synflownet.utils.transforms import thermometer, to_logreward

# Write any local overrides required here
LOCAL_OVERRIDES = "local_overrides.yaml"
FINAL_PREF = 0.5
PIN_PREF = False
FINAL_BETA = 16.0
PIN_BETA = False
PREF_REWARDS = ("seh_cyp_pref",)
CYP_REWARDS = ("seh_cyp1a2", "seh_cyp1a2_3a4", "seh_cyp_product") + PREF_REWARDS

# Set by --reward / --sample_dist / --dist_params.
REWARD = "seh_cyp_pref"
SAMPLE_DIST = "constant"
DIST_PARAMS = [32.0]
SCALE = 0.5
from synflownet.envs.graph_building_env import Graph


class ReactionTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `synflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        cfg: Config,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.cfg = cfg
        self.reward = cfg.reward
        self.is_pc = self.reward in PREF_REWARDS
        if self.reward in CYP_REWARDS:
            from predict import CypPredictor

            isoforms = ["cyp1a2"] if self.reward == "seh_cyp1a2" else ["cyp1a2", "cyp3a4"]
            self.cyp = CypPredictor(isoforms=isoforms, device="cpu")
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg)
        self.pref_conditional = MultiObjectiveWeightedPreferences(cfg) if self.is_pc else None
        self.num_cond_dim = self.temperature_conditional.encoding_size() + (
            self.pref_conditional.encoding_size() if self.is_pc else 0
        )

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> Dict[str, Tensor]:
        if PIN_BETA:
            tcfg = self.cfg.cond.temperature
            beta = torch.full((n,), FINAL_BETA)
            if tcfg.sample_dist == "constant":
                enc = torch.zeros((n, tcfg.num_thermometer_dim))
            else:
                enc = thermometer(beta, tcfg.num_thermometer_dim, 0, self.temperature_conditional.upper_bound)
            temp = {"beta": beta, "encoding": enc}
        else:
            temp = self.temperature_conditional.sample(n)

        if not self.is_pc:
            # single-objective reward: temperature is the whole conditioning
            return temp

        if PIN_PREF:
            prefs = torch.tensor([[FINAL_PREF, 1.0 - FINAL_PREF]]).repeat(n, 1).float()
            pref = {"preferences": prefs, "encoding": self.pref_conditional.encode(prefs)}
        else:
            pref = self.pref_conditional.sample(n)
        # both dicts carry an "encoding" key, so concatenate explicitly
        return {**temp, **pref, "encoding": torch.cat([temp["encoding"], pref["encoding"]], dim=1)}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        if self.reward == "seh_cyp_pref":
            logr = flat_reward.log()                               
            scalar_log = (logr * cond_info["preferences"]).sum(1)   
            return LogScalar(self.temperature_conditional.transform(cond_info, LogScalar(scalar_log)))
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def seh_proxy_reward_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol], traj_lens: Tensor) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device())
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu() / 8
        if self.cfg.task.reactions_task.reward is None:
            # Using only affinity proxy
            pass
        elif self.cfg.task.reactions_task.reward == "efficiency":
            mol_sizes = scatter_add(torch.ones_like(batch.batch), batch.batch, dim=0).cpu()
            avg_sizes = traj_lens * 13  # 13 is the average size of the BBs from large set
            norm = (mol_sizes / avg_sizes) + 1e-5
            preds = preds / norm
        elif self.cfg.task.reactions_task.reward.startswith("size_penalty"):
            _, _, max_size, penalty = self.cfg.task.reactions_task.reward.split("_")
            mol_sizes = scatter_add(torch.ones_like(batch.batch), batch.batch, dim=0).cpu()
            penalties = (mol_sizes > int(max_size)) * float(penalty)
            preds = preds - penalties
        else:
            raise NotImplementedError

        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))

    def safe(self, f, x, default):
        try:
            return f(x)
        except Exception:
            return default

    def vina_rewards_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol]) -> Tensor:

        print("Calculating Vina rewards...")
        vina = QuickVina2GPU(vina_path=self.cfg.vina.vina_path, target=self.cfg.vina.target)
        # Convert smiles to mols
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        _, _, vina_rewards = vina.calculate_rewards(smiles)

        return torch.tensor(vina_rewards).float().clip(1e-4, 100)

    def qed_rewards_from_mols(self, mols: List[RDMol]) -> Tensor:
        return torch.tensor([QED.qed(m) for m in mols])

    def cyp_safe_from_mols(self, mols: List[RDMol], isoforms) -> Tensor:
        """1 - mean(p) over the given isoforms, as a standalone objective."""
        smiles = [Chem.MolToSmiles(m) for m in mols]
        probs = self.cyp.predict(smiles)
        p = np.nanmean(np.vstack([np.nan_to_num(probs[i], nan=1.0) for i in isoforms]), axis=0)
        return torch.tensor(1.0 - p).float().clip(1e-4, 100)

    def cyp_panel_rewards_from_mols(self, mols: List[RDMol], isoforms, scale: float = None) -> Tensor:
        """1 - scale * mean(p) over the given isoforms.
        scale defaults to the module-level SCALE, set by --scale.
        """
        if scale is None:
            scale = SCALE
        smiles = [Chem.MolToSmiles(m) for m in mols]
        probs = self.cyp.predict(smiles)
        p = np.nanmean(np.vstack([np.nan_to_num(probs[i], nan=1.0) for i in isoforms]), axis=0)
        return torch.tensor(1.0 - scale * p).float().clip(1e-4, 100)

    def cyp1a2_rewards_from_mols(self, mols: List[RDMol]) -> Tensor:
        """1 - p(inhibits CYP1A2), from the GCN proxy in CYP/."""
        smiles = [Chem.MolToSmiles(m) for m in mols]
        p = self.cyp.predict(smiles)["cyp1a2"]
        p = np.nan_to_num(p, nan=1.0)  # unparseable: treat as an inhibitor, never reward it
        return torch.tensor(1.0 - p).float().clip(1e-4, 100)

    def gsk_rewards_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol]) -> Tensor:
        gsk = Oracle("GSK3B")
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        gsk_rewards = [gsk(s) for s in smiles]

        return torch.tensor(gsk_rewards).float().clip(1e-4, 100)

    def drd2_rewards_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol]) -> Tensor:
        drd2 = Oracle("DRD2")
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        drd2_rewards = [drd2(s) for s in smiles]

        return torch.tensor(drd2_rewards).float().clip(1e-4, 100)

    def mol2sas(self, mols: list[RDMol], default=10):
        sas = torch.tensor([self.safe(sascore.calculateScore, i, default) for i in mols])
        # sas = (10 - sas) / 9  # Turn into a [0-1] reward
        sas = ((10 - sas) / (10 - 3.5)).clamp(0, 1)
        return torch.tensor(sas).float().clip(1e-4, 100)

    def compute_reward_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol], traj_lens: Tensor) -> Tensor:
        if self.reward == "seh_reaction":
            return self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
        elif self.reward == "vina":
            return self.vina_rewards_from_graph(graphs, mols)
        elif self.reward == "qed":
            return self.qed_rewards_from_mols(mols)
        elif self.reward == "gsk":
            return self.gsk_rewards_from_graph(graphs, mols)
        elif self.reward == "drd2":
            return self.drd2_rewards_from_graph(graphs, mols)
        elif self.reward == "reward_one":
            return torch.ones(len([g for g in graphs if g is not None]))
        elif self.reward == "seh_qed":
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            qed_scores = self.qed_rewards_from_mols(mols)
            qed_scores = (qed_scores / 0.7).clamp(0, 1)
            return seh_scores * qed_scores
        elif self.reward == "seh_cyp1a2":
            valid_mols = [m for m, g in zip(mols, graphs) if g is not None]
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            return seh_scores * self.cyp1a2_rewards_from_mols(valid_mols)
        elif self.reward == "seh_cyp_pref":
            # returns (n, 2): the preference conditional scalarises it later
            valid_mols = [m for m, g in zip(mols, graphs) if g is not None]
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            safe = self.cyp_safe_from_mols(valid_mols, ["cyp1a2", "cyp3a4"])
            return torch.stack([seh_scores, safe], dim=1)
        elif self.reward == "seh_cyp_product":
            # Raw product: the CYP term spans [0, 1] rather than [1 - scale, 1], so it
            # separates clean from liable far harder at matched concentration and needs
            valid_mols = [m for m, g in zip(mols, graphs) if g is not None]
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            return seh_scores * self.cyp_panel_rewards_from_mols(valid_mols, ["cyp1a2", "cyp3a4"], scale=1.0)
        elif self.reward == "seh_cyp1a2_3a4":
            valid_mols = [m for m, g in zip(mols, graphs) if g is not None]
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            return seh_scores * self.cyp_panel_rewards_from_mols(valid_mols, ["cyp1a2", "cyp3a4"])
        elif self.reward == "seh_sa":
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            sas = self.mol2sas(mols)
            return seh_scores * sas
        elif self.reward == "seh_mol_wt":
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            for i, mol in enumerate(mols):
                # Penalize molecules with more than reference number of atoms
                if mol.GetNumHeavyAtoms() > 37:
                    seh_scores[i] += -0.4
            return seh_scores
        else:
            raise ValueError(f"Unknown task {self.task}")

    def compute_obj_properties(self, mols: List[RDMol], traj_lens: Tensor, **kwargs) -> Tuple[ObjectProperties, Tensor]:
        graphs = []
        for i in mols:
            try:
                g = bengio2021flow.mol2graph(i)
            except IndexError:
                # NOTE: affinity model does not have all atom types found in BBs
                # so for now we mark such molecules as invalid
                g = None
            graphs.append(g)

        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds = self.compute_reward_from_graph(graphs=graphs, mols=mols, traj_lens=traj_lens)
        if preds.ndim == 1:  # single-objective rewards; the preference branch is already (n, k)
            preds = preds.reshape((-1, 1))
        assert len(preds) == is_valid.sum(), f"len(preds)={len(preds)}, is_valid.sum()={is_valid.sum()}"

        return ObjectProperties(preds), is_valid


class ReactionTrainer(StandardOnlineTrainer):
    task: ReactionTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_num_from_policy = 64
        cfg.num_validation_gen_steps = 10
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = False

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 64
        cfg.replay.num_from_replay = 0
        cfg.replay.num_new_samples = 64

    def setup_task(self):
        self.task = ReactionTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)

        # Load building blocks
        building_blocks_path = os.path.join(
            repo_root, "data/building_blocks", self.cfg.task.reactions_task.building_blocks_filename
        )
        with open(building_blocks_path, "r") as file:
            building_blocks = file.read().splitlines()

        # Load templates
        templates_path = os.path.join(repo_root, "data/templates", self.cfg.task.reactions_task.templates_filename)
        with open(templates_path, "r") as file:
            reaction_templates = file.read().splitlines()

        precomputed_bb_masks_path = os.path.join(
            repo_root, "data/building_blocks", self.cfg.task.reactions_task.precomputed_bb_masks_filename
        )
        with open(precomputed_bb_masks_path, "rb") as f:
            precomputed_bb_masks = pickle.load(f)

        self.ctx = ReactionTemplateEnvContext(
            num_cond_dim=self.task.num_cond_dim,
            building_blocks=building_blocks,
            reaction_templates=reaction_templates,
            precomputed_bb_masks=precomputed_bb_masks,
            fp_type=self.cfg.model.graph_transformer.fingerprint_type,
            fp_path=self.cfg.model.graph_transformer.fingerprint_path,
            strict_bck_masking=self.cfg.algo.strict_bck_masking,
        )
        self.env = ReactionTemplateEnv(
            reaction_templates=reaction_templates,
            building_blocks=building_blocks,
            precomputed_bb_masks=precomputed_bb_masks,
            fp_type=self.cfg.model.graph_transformer.fingerprint_type,
            fp_path=self.cfg.model.graph_transformer.fingerprint_path,
        )

    def setup_sampler(self):
        self.sampler = SynthesisSampler(
            cfg=self.cfg,
            ctx=self.ctx,
            env=self.env,
            max_len=self.cfg.algo.max_len,
            correct_idempotent=self.cfg.algo.tb.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.algo.tb.do_parameterize_p_b,
        )

    def build_callbacks(self):

        from rdkit.Chem.Scaffolds import MurckoScaffold

        graph_to_obj = self.ctx.graph_to_obj
        # mean over objectives so the callback's reward_thresh comparison stays scalar
        reward_fn = lambda rdmols, traj_lens: self.task.compute_obj_properties(rdmols, traj_lens)[0].mean(1, keepdim=True)

        class UniqueMurckoScaffoldsCallback:
            def __init__(self, reward_thresh):
                self._reward_thresh = reward_thresh

            def on_validation_end(self, step_outputs):

                mols = []
                rewards = []
                for out in step_outputs:
                    batch = out["batch"]
                    final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1
                    final_graphs = [batch.nx_graphs[i] for i in final_graph_idx]
                    batch_mols = [graph_to_obj(g) for g in final_graphs]
                    mols.extend(batch_mols)
                    rewards.append(reward_fn(batch_mols, batch.traj_lens))

                murcko_scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
                rewards = torch.cat(rewards)
                assert len(murcko_scaffolds) == len(rewards)

                scaffolds_above_thresh = [smi for smi, r in zip(murcko_scaffolds, rewards) if r > self._reward_thresh]
                unique_scaffolds = set(scaffolds_above_thresh)

                return {f"unique_murcko_r_gt_{self._reward_thresh}": len(unique_scaffolds)}

        return {"murcko_scaffolds": UniqueMurckoScaffoldsCallback(reward_thresh=0.5)}


def main(
    wandb_run_name,
    backoff=False,
    final_pref=None,
    final_beta=None,
    reward=None,
    scale=None,
    sample_dist=None,
    dist_params=None,
):
    """Example of how this trainer can be run"""
    global FINAL_PREF, PIN_PREF, FINAL_BETA, PIN_BETA, REWARD, SCALE, SAMPLE_DIST, DIST_PARAMS
    if reward is not None:
        REWARD = reward
    if scale is not None:
        SCALE = float(scale)
    if sample_dist is not None:
        SAMPLE_DIST = sample_dist
    if dist_params is not None:
        DIST_PARAMS = [float(x) for x in dist_params]
    if final_pref is not None:
        FINAL_PREF = float(final_pref)
        PIN_PREF = True
        print(f"pinning omega = ({FINAL_PREF}, {1 - FINAL_PREF}) for all sampling")
    if final_beta is not None:
        FINAL_BETA = float(final_beta)
        PIN_BETA = True
        print(f"pinning beta={FINAL_BETA} for all sampling")
    # scale never reaches the checkpoint, so log it on every run including --backoff
    print(
        f"reward={REWARD}  scale={SCALE}  sample_dist={SAMPLE_DIST}  dist_params={DIST_PARAMS}",
        flush=True,
    )

    name = None
    checkpoint_path = None
    if backoff:
        assert wandb_run_name is not None
        logroot = "./logs"
        logdirs = [d for d in os.listdir(logroot) if d.startswith(wandb_run_name)]
        if len(logdirs) > 0:
            name = sorted(logdirs)[0]
            checkpoint_path = os.path.join(logroot, name, "model_state.pt")
            wandb.init(project="synflownet", name=name, id=name, resume="must")
            print(f"Found checkpoint at {checkpoint_path}")

    if name is None and checkpoint_path is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if wandb_run_name is not None:
            name = f"{wandb_run_name}_{now}"
            wandb.init(project="synflownet", name=name, id=name)
        else:
            name = f"debug_{now}"
        print(f"No checkpoint found, starting a new run {name}")

    # trainer is loaded from checkpoint using StandardOnlineTrainer.load_from_checkpoint
    if checkpoint_path is not None:
        trial = ReactionTrainer.load_from_checkpoint(checkpoint_path)
    else:
        config = init_empty(Config())
        config.reward = REWARD
        if REWARD in PREF_REWARDS:
            # only for the multi-objective rewards: this widens num_cond_dim
            config.cond.moo.num_objectives = 2
            config.cond.weighted_prefs.preference_type = "dirichlet"
            config.cond.weighted_prefs.preference_param = 1.0  # Dirichlet(1,1) = uniform on [0,1]
        config.print_every = 1
        config.log_dir = f"./logs/{name}"
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.overwrite_existing_exp = True
        config.num_training_steps = 5000
        config.validate_every = 500
        config.num_final_gen_steps = 100
        config.num_workers = 8
        config.opt.learning_rate = 1e-4
        config.opt.lr_decay = 2_000
        config.algo.sampling_tau = 0.99
        config.cond.temperature.sample_dist = SAMPLE_DIST
        config.cond.temperature.dist_params = list(DIST_PARAMS)
        config.model.graph_transformer.continuous_action_embs = True
        config.model.graph_transformer.fingerprint_type = "morgan_1024"

        # For Vina experiments
        config.vina.target = "kras"

        # Load local overrides
        if os.path.exists(LOCAL_OVERRIDES):
            overrides = load_yaml(LOCAL_OVERRIDES)
            config = override_config(config, overrides)

        # Activate WandB here if not running experiment through a sweep
        if hasattr(config, "wandb"):
            wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config)
        config.algo.max_len = 3
        config.algo.tb.backward_policy = "MaxLikelihood"
        config.algo.tb.do_parameterize_p_b = True
        config.algo.tb.do_sample_p_b = False
        config.replay.use = False
        config.algo.synthesis_cost_as_bck_reward = False
        config.algo.tb.reinforce_loss_multiplier = 1.0
        config.algo.tb.bck_entropy_loss_multiplier = 1.0
        config.algo.num_from_policy = 64
        config.algo.num_from_buffer_for_pb = 64

        config.algo.strict_forward_policy = False  # If True, a fwd action is allowed only if in reverse it produces the exact same reaction (identical reactants and products)
        config.algo.strict_bck_masking = (
            False  # If True, bimolecular bck actions are masked if they don't produce at least one bb
        )
        config.algo.tb.do_correct_idempotent = False

        trial = ReactionTrainer(config)
    trial.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_run_name", type=str, required=False, default=None)
    p.add_argument("--backoff", action="store_true", required=False, default=False)
    p.add_argument(
        "--reward",
        type=str,
        default=None,
        choices=[
            "seh_reaction",
            "seh_cyp1a2",
            "seh_cyp1a2_3a4",
            "seh_cyp_product",
            "seh_cyp_pref",
            "seh_qed",
            "seh_sa",
            "seh_mol_wt",
            "vina",
            "qed",
            "gsk",
            "drd2",
            "reward_one",
        ],
        help=f"reward to train (default {REWARD}). Ignored with --backoff, which takes "
        "the reward from the checkpoint's config.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help=f"CYP penalty strength for seh_cyp1a2_3a4: sEH * (1 - scale * mean p) "
        f"(default {SCALE}). Not stored in the checkpoint, so pass the same value "
        "again on the --backoff generation pass.",
    )
    p.add_argument(
        "--sample_dist",
        type=str,
        default=None,
        choices=["constant", "uniform", "loguniform", "gamma", "beta"],
        help=f"beta sampling distribution used during training (default {SAMPLE_DIST}).",
    )
    p.add_argument(
        "--dist_params",
        type=float,
        nargs="+",
        default=None,
        help=f"parameters for --sample_dist: one value for 'constant', two for "
        f"'uniform' (default {DIST_PARAMS}).",
    )
    p.add_argument(
        "--final_beta",
        type=float,
        default=None,
        help=f"temperature for the final generation pass (default {FINAL_BETA}). "
        "Only has an effect when sample_dist is not 'constant'.",
    )
    p.add_argument(
        "--final_pref",
        type=float,
        default=None,
        help=f"weight on sEH for the final generation pass, in [0,1] (default {FINAL_PREF}). "
        "1.0 = pure sEH, 0.0 = pure CYP safety. Only used by the *_pref rewards.",
    )
    args = p.parse_args()
    main(
        args.wandb_run_name,
        args.backoff,
        final_pref=args.final_pref,
        final_beta=args.final_beta,
        reward=args.reward,
        scale=args.scale,
        sample_dist=args.sample_dist,
        dist_params=args.dist_params,
    )
