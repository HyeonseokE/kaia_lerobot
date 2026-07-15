"""Apply the chunk-wise-delta-action patch to an installed lerobot, in place.

Run against site-packages at Docker build time (see cluster/Dockerfile), so the image
carries the patch and training jobs need no runtime bind-mounts.

lerobot 0.6.0 ships the Relative/AbsoluteActionsProcessorStep machinery -- pi0.5 uses it --
but three things stop SmolVLA from using it:

  1. SmolVLAConfig has no use_relative_actions / relative_exclude_joints fields, and
     make_smolvla_pre_post_processors never builds the steps.
  2. make_pre_post_processors(pretrained_path=...) rebuilds the pipeline from the JSON the
     checkpoint shipped with. lerobot/pi05_base carries a relative_actions_processor step;
     lerobot/smolvla_base and lerobot/pi0 do not, so overriding it is a hard
         KeyError: Override keys ['relative_actions_processor'] do not match any step
     The steps have to be INSERTED, not overridden.
  3. SmolVLA's observation_delta_indices is [0], so observation.state arrives as
     (B, n_obs, D) where pi0/pi0.5 give (B, D). to_relative_actions assumed the latter and
     broadcast a (B,1,D) state against a (B,T,D) action chunk into (B,B,T,D).

All replacements are anchored and must match exactly once; anything else aborts the build
rather than half-applying. Re-running on an already-patched tree is a no-op.

    python delta_patch.py            # patch the installed lerobot in place
    python delta_patch.py --check    # verify the patch is live, change nothing
"""

import argparse
import ast
import importlib.util
import os
import pathlib
import sys


def locate(module):
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise SystemExit(f"FATAL: cannot locate {module}.")
    return pathlib.Path(spec.origin)


def sub1(text, anchor, replacement, what):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit(
            f"FATAL: delta patch anchor for '{what}' matched {n} times, expected 1.\n"
            f"       This lerobot differs from what the patch was written against\n"
            f"       (v0.6.0). Refusing to patch rather than half-apply it."
        )
    return text.replace(anchor, replacement)


# ------------------------------------------------ 1. configuration_smolvla.py
def patch_cfg(src):
    return sub1(
        src,
        "    use_delta_joint_actions_aloha: bool = False\n",
        "    use_delta_joint_actions_aloha: bool = False\n"
        "\n"
        "    # Relative actions: converts absolute actions to relative (relative to state).\n"
        "    use_relative_actions: bool = False\n"
        "    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.\n"
        '    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])\n'
        "    # Populated at runtime from dataset metadata by make_policy.\n"
        "    action_feature_names: list[str] | None = None\n",
        "config: relative action fields",
    )


# ---------------------------------------------------- 2. processor_smolvla.py
def patch_proc(src):
    src = sub1(
        src,
        "from lerobot.processor import (\n    AddBatchDimensionProcessorStep,",
        "from lerobot.processor import (\n    AbsoluteActionsProcessorStep,\n    AddBatchDimensionProcessorStep,",
        "processor: import AbsoluteActionsProcessorStep",
    )
    src = sub1(
        src,
        "    PolicyProcessorPipeline,\n    RenameObservationsProcessorStep,",
        "    PolicyProcessorPipeline,\n    RelativeActionsProcessorStep,\n    RenameObservationsProcessorStep,",
        "processor: import RelativeActionsProcessorStep",
    )
    src = sub1(
        src,
        "    input_steps = [\n",
        "    relative_step = RelativeActionsProcessorStep(\n"
        "        enabled=config.use_relative_actions,\n"
        '        exclude_joints=getattr(config, "relative_exclude_joints", []),\n'
        '        action_names=getattr(config, "action_feature_names", None),\n'
        "    )\n"
        "\n"
        "    # OpenPI order: raw -> relative -> normalize -> model -> unnormalize -> absolute\n"
        "    input_steps = [\n",
        "processor: build relative_step",
    )
    src = sub1(
        src,
        "        DeviceProcessorStep(device=config.device),\n        NormalizerProcessorStep(",
        "        DeviceProcessorStep(device=config.device),\n        relative_step,\n        NormalizerProcessorStep(",
        "processor: insert relative before Normalizer",
    )
    src = sub1(
        src,
        '        ),\n        DeviceProcessorStep(device="cpu"),\n    ]',
        "        ),\n"
        "        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),\n"
        '        DeviceProcessorStep(device="cpu"),\n'
        "    ]",
        "processor: insert absolute after Unnormalizer",
    )
    return src


# ---------------------------------------------------------------- 3. factory.py
ENSURE_FN = '''def _ensure_relative_absolute_steps(
    policy_cfg: PreTrainedConfig,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> None:
    """Insert the relative/absolute action steps into pipelines loaded from a checkpoint lacking them.

    PolicyProcessorPipeline.from_pretrained rebuilds the pipeline from the JSON the checkpoint
    shipped with, so a base model uploaded before relative actions existed has no
    relative_actions_processor step at all. lerobot/pi05_base does carry one; lerobot/pi0 and
    lerobot/smolvla_base do not. Overriding a step that is not there raises
        KeyError: Override keys ['relative_actions_processor'] do not match any step
    so we insert them here instead, positioned exactly as the from-scratch factories do:
    relative immediately before the normalizer, absolute immediately after the unnormalizer
    (raw -> relative -> normalize -> model -> unnormalize -> absolute).
    """
    if not getattr(policy_cfg, "use_relative_actions", False):
        return

    relative_step = next((s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None)
    if relative_step is None:
        relative_step = RelativeActionsProcessorStep()
        norm_idx = next(
            (i for i, s in enumerate(preprocessor.steps) if isinstance(s, NormalizerProcessorStep)), None
        )
        if norm_idx is None:
            raise ValueError(
                "Cannot enable relative actions: the loaded preprocessor has no NormalizerProcessorStep, "
                "so there is no defined place to insert the relative conversion. Actions must be made "
                "relative BEFORE normalization, since the stats are computed in relative space."
            )
        preprocessor.steps.insert(norm_idx, relative_step)

    relative_step.enabled = True
    relative_step.exclude_joints = list(getattr(policy_cfg, "relative_exclude_joints", []) or [])
    relative_step.action_names = getattr(policy_cfg, "action_feature_names", None)

    absolute_step = next(
        (s for s in postprocessor.steps if isinstance(s, AbsoluteActionsProcessorStep)), None
    )
    if absolute_step is None:
        absolute_step = AbsoluteActionsProcessorStep()
        unnorm_idx = next(
            (i for i, s in enumerate(postprocessor.steps) if isinstance(s, UnnormalizerProcessorStep)), None
        )
        if unnorm_idx is None:
            raise ValueError(
                "Cannot enable relative actions: the loaded postprocessor has no "
                "UnnormalizerProcessorStep, so predicted deltas could not be turned back into "
                "absolute actions for execution."
            )
        postprocessor.steps.insert(unnorm_idx + 1, absolute_step)

    absolute_step.enabled = True
    absolute_step.relative_step = relative_step


'''


def patch_factory(src):
    src = sub1(
        src,
        "from lerobot.processor import (\n    AbsoluteActionsProcessorStep,\n    PolicyProcessorPipeline,\n    RelativeActionsProcessorStep,\n",
        "from lerobot.processor import (\n"
        "    AbsoluteActionsProcessorStep,\n"
        "    NormalizerProcessorStep,\n"
        "    PolicyProcessorPipeline,\n"
        "    RelativeActionsProcessorStep,\n"
        "    UnnormalizerProcessorStep,\n",
        "factory: imports",
    )
    src = sub1(
        src,
        "def get_policy_class(name: str) -> type[PreTrainedPolicy]:",
        ENSURE_FN + "def get_policy_class(name: str) -> type[PreTrainedPolicy]:",
        "factory: _ensure_relative_absolute_steps",
    )
    src = sub1(
        src,
        '            overrides=kwargs.get("preprocessor_overrides", {}),\n',
        "            overrides=pre_overrides,\n",
        "factory: preprocessor overrides",
    )
    src = sub1(
        src,
        '            overrides=kwargs.get("postprocessor_overrides", {}),\n',
        "            overrides=post_overrides,\n",
        "factory: postprocessor overrides",
    )
    src = sub1(
        src,
        "        preprocessor = PolicyProcessorPipeline.from_pretrained(\n            pretrained_model_name_or_path=pretrained_path,\n",
        "        # The relative/absolute overrides are applied by _ensure_relative_absolute_steps below,\n"
        "        # not by from_pretrained: a base checkpoint predating relative actions has no such step\n"
        "        # in its saved pipeline, and overriding a step that is not there is a hard KeyError.\n"
        "        pre_overrides = {\n"
        "            k: v\n"
        '            for k, v in (kwargs.get("preprocessor_overrides") or {}).items()\n'
        '            if k != "relative_actions_processor"\n'
        "        }\n"
        "        post_overrides = {\n"
        "            k: v\n"
        '            for k, v in (kwargs.get("postprocessor_overrides") or {}).items()\n'
        '            if k != "absolute_actions_processor"\n'
        "        }\n"
        "\n"
        "        preprocessor = PolicyProcessorPipeline.from_pretrained(\n"
        "            pretrained_model_name_or_path=pretrained_path,\n",
        "factory: build filtered overrides",
    )
    src = sub1(
        src,
        "        _reconnect_relative_absolute_steps(preprocessor, postprocessor)\n",
        "        _ensure_relative_absolute_steps(policy_cfg, preprocessor, postprocessor)\n"
        "        _reconnect_relative_absolute_steps(preprocessor, postprocessor)\n",
        "factory: call _ensure",
    )
    return src


# ------------------------------------------- 4. relative_action_processor.py
CURRENT_STATE_FN = '''def _current_state(state: Tensor) -> Tensor:
    """Reduce an observation-history state to the single reference step, (B, state_dim).

    Policies differ in whether observation.state carries a time axis: SmolVLA's
    observation_delta_indices is [0], so its batches hold (B, n_obs_steps, state_dim), while
    pi0/pi0.5 return None and hold (B, state_dim). A chunk of relative actions is defined
    against ONE reference state -- the current one -- so take the last observed step. Without
    this, a (B, 1, D) state broadcasts against the (B, T, D) action chunk into (B, B, T, D)
    and the subtraction fails.
    """
    if state.ndim == 3:
        return state[:, -1]
    return state


'''


def patch_relative(src):
    src = sub1(
        src,
        "def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:",
        CURRENT_STATE_FN + "def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:",
        "relative: _current_state helper",
    )
    n = src.count("    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)\n")
    if n != 2:
        raise SystemExit(f"FATAL: relative_action_processor.py: expected 2 mask_t sites, found {n}.")
    src = src.replace(
        "    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)\n",
        "    state = _current_state(state)\n"
        "    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)\n",
    )
    return src


TARGETS = [
    ("lerobot.policies.smolvla.configuration_smolvla", "use_relative_actions", patch_cfg),
    ("lerobot.policies.smolvla.processor_smolvla", "RelativeActionsProcessorStep", patch_proc),
    ("lerobot.policies.factory", "_ensure_relative_absolute_steps", patch_factory),
    ("lerobot.processor.relative_action_processor", "_current_state", patch_relative),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    args = ap.parse_args()

    missing = []
    wrote = False
    for module, marker, patch_fn in TARGETS:
        path = locate(module)
        src = path.read_text()
        if marker in src:
            print(f"[delta] {path.name}: already patched")
            continue
        if args.check:
            missing.append(path.name)
            continue
        patched = patch_fn(src)
        ast.parse(patched)
        path.write_text(patched)
        wrote = True
        print(f"[delta] {path.name}: patched")

    if args.check and missing:
        raise SystemExit(f"FATAL: delta patch NOT applied to: {', '.join(missing)}")

    if wrote:
        # locate() calls find_spec, which imports the parent packages -- and those import the
        # very modules we then rewrite on disk. sys.modules still holds the PRE-patch classes,
        # so verifying in this interpreter would test the old code and fail. Re-exec to get a
        # clean import of what we just wrote.
        os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), "--check"])

    verify()


def verify():
    """Prove the patch actually works, not merely that the text is there."""
    import torch

    from lerobot.policies import factory
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.processor.relative_action_processor import to_relative_actions

    assert "use_relative_actions" in SmolVLAConfig.__dataclass_fields__, "config fields missing"

    import inspect

    src = inspect.getsource(make_smolvla_pre_post_processors)
    assert "RelativeActionsProcessorStep" in src and "AbsoluteActionsProcessorStep" in src, (
        "processor pipeline not wired"
    )
    assert hasattr(factory, "_ensure_relative_absolute_steps"), "factory cannot insert the steps"

    # SmolVLA hands in a (B, n_obs, D) state; pi0/pi0.5 hand in (B, D). Both must work.
    out = to_relative_actions(torch.zeros(2, 50, 12), torch.ones(2, 1, 12), [True] * 12)
    assert out.shape == (2, 50, 12), f"(B,n_obs,D) state broke: got {tuple(out.shape)}"
    assert torch.allclose(out, torch.full((2, 50, 12), -1.0)), "relative conversion is wrong"

    out = to_relative_actions(torch.zeros(2, 50, 12), torch.ones(2, 12), [True] * 12)
    assert out.shape == (2, 50, 12), f"(B,D) state broke: got {tuple(out.shape)}"

    print("[delta] verified: config fields, processor wiring, factory insertion, both state shapes")


if __name__ == "__main__":
    sys.exit(main())
