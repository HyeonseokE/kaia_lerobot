# CSI Lab GPU cluster image for SmolVLA finetuning with DELTA joint actions.
#
#   docker build -f cluster/Dockerfile -t hyeonseoke/lerobot_delta:v1 .
#   docker push hyeonseoke/lerobot_delta:v1
#
# This is hyeonseoke/lerobot:v1 plus the chunk-wise delta-action patch for SmolVLA
# (see cluster/delta_patch.py). The absolute-action baseline (cluster/train.sbatch)
# still runs on the unpatched hyeonseoke/lerobot:v1 and does not need this image.
#
# Use an explicit version tag, not :latest -- Apptainer caches the SIF by tag and
# will happily serve a stale :latest after a rebuild ("Using cached SIF image"),
# which is exactly the silent-wrong-image trap to avoid. Bump to :v2 on the next
# rebuild and update cluster/train_delta.sbatch to match.
#
# No conda: the container is already the isolation boundary.
# Everything lives outside $HOME, because Apptainer bind-mounts the cluster
# home over the container's home at runtime.

FROM python:3.12-slim

# torchcodec links against system FFmpeg shared libs (they are not bundled).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Official package. 0.6.0 is the latest PyPI release (the local checkout is an
# unreleased 0.6.1 dev tree); v0.6.0 already carries --rename_map, which the
# training runs need. torch comes in as a dependency; recent PyPI wheels are
# CUDA 12.8 builds that include sm_120 (Blackwell / RTX PRO 6000).
#   smolvla  -> the policy itself (transformers, num2words, accelerate)
#   training -> dataset extra (torchcodec/av for video decoding) + wandb
# The dataset extra is not optional here: every episode is stored as video, so
# without torchcodec the dataloader cannot read a single frame.
RUN pip install --no-cache-dir "lerobot[smolvla,training]==0.6.0"

# ---------------------------------------------------------------- delta actions
# Rewrite four files in the installed lerobot so SmolVLA can train on chunk-wise
# delta joint actions (a_t - s_0, one s_0 per chunk). 0.6.0 already ships the
# Relative/AbsoluteActionsProcessorStep machinery -- pi0.5 uses it -- but SmolVLA
# cannot reach it: the config has no use_relative_actions field, the factory cannot
# insert the steps into a base checkpoint whose saved pipeline predates them
# (smolvla_base and pi0 are both in that boat; pi05_base is not), and the shared
# relative helper mis-broadcasts SmolVLA's (B, n_obs, D) observation.state.
#
# Every edit is anchored and must match exactly once, so a future lerobot that has
# moved on FAILS THE BUILD rather than producing an image that half-works. The
# script then re-execs itself to verify the patch against a clean import.
COPY cluster/delta_patch.py /tmp/delta_patch.py
RUN python /tmp/delta_patch.py && rm /tmp/delta_patch.py

# NOTE: torch.cuda.get_arch_list() returns [] whenever no GPU is visible, so the
# Blackwell (sm_120) check cannot run here -- `docker build` has no GPU. The job
# script does it as a preflight instead, on the real GPU, before training starts.
RUN python -c "import torch, torchcodec, lerobot; from importlib.metadata import version; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| lerobot', version('lerobot'))" \
    && lerobot-train --help > /dev/null

# Fail the build if the delta patch is not live in a fresh interpreter. This is the
# check that a silent half-apply would have to get past.
RUN python -c "\
import inspect, torch; \
from lerobot.policies import factory; \
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig; \
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors; \
from lerobot.processor.relative_action_processor import to_relative_actions; \
assert 'use_relative_actions' in SmolVLAConfig.__dataclass_fields__, 'config fields missing'; \
src = inspect.getsource(make_smolvla_pre_post_processors); \
assert 'RelativeActionsProcessorStep' in src and 'AbsoluteActionsProcessorStep' in src, 'pipeline not wired'; \
assert hasattr(factory, '_ensure_relative_absolute_steps'), 'factory cannot insert the steps'; \
out = to_relative_actions(torch.zeros(2, 50, 12), torch.ones(2, 1, 12), [True] * 12); \
assert out.shape == (2, 50, 12), out.shape; \
assert torch.allclose(out, torch.full((2, 50, 12), -1.0)), 'relative conversion is wrong'; \
print('delta patch verified in image')"

ENV PYTHONUNBUFFERED=1
