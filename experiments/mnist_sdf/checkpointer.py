from typing import Any

import optax
import orbax.checkpoint
from flax import core, struct
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint.utils import get_save_directory
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class EMATrainState(TrainState):
    ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    batch_stats: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


def new_train_state(rng_key, model, init_batch, config):
    variables = model.init(
        {"params": rng_key, "sample": rng_key},
        method="loss",
        y0=init_batch,
        is_training=False,
    )
    if config.optimizer.do_warmup and config.optimizer.do_decay:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optimizer.learning_rate,
            warmup_steps=config.optimizer.warmup_steps,
            decay_steps=config.optimizer.decay_steps,
            end_value=config.optimizer.end_learning_rate,
        )
    elif config.optimizer.do_warmup:
        lr = optax.linear_schedule(
            init_value=0.0,
            end_value=config.optimizer.learning_rate,
            transition_steps=config.optimizer.warmup_steps,
        )
    elif config.optimizer.do_decay:
        lr = optax.cosine_decay_schedule(
            init_value=config.optimizer.learning_rate,
            decay_steps=config.optimizer.decay_steps,
            alpha=config.optimizer.end_learning_rate
            / config.optimizer.learning_rate,
        )
    else:
        lr = config.optimizer.learning_rate

    tx = optax.adamw(lr, weight_decay=config.optimizer.weight_decay)
    if config.optimizer.do_gradient_clipping:
        tx = optax.chain(
            optax.clip_by_global_norm(config.optimizer.gradient_clipping), tx
        )

    return EMATrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        ema_params=variables["params"].copy(),
        batch_stats=variables["batch_stats"],
        tx=tx,
    )


def get_checkpointer_fns(outfolder):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=5,
        save_interval_steps=20,
        create=True,
        best_fn=lambda x: x["train_loss"],
        best_mode="min",
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        outfolder,
        checkpointer,
        options,
    )

    def save_fn(epoch, ckpt, metrics):
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            epoch, ckpt, save_kwargs={"save_args": save_args}, metrics=metrics
        )

    def restore_fn():
        return checkpoint_manager.restore(checkpoint_manager.best_step())

    def path_best_ckpt_fn():
        return get_save_directory(
            checkpoint_manager.best_step(), checkpoint_manager.directory
        )

    return save_fn, restore_fn, path_best_ckpt_fn
