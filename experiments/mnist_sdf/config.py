import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.rng_key = 1
    config.model = new_dict(
        score_model=new_dict(
            n_blocks=2,
            n_channels=32,
            channel_multipliers=(1, 2, 4, 8),
            dropout_rate=0.1,
            do_pooling_via_fno=True,
        ),
        diffusion_model=new_dict(
            alpha_schedule="linear",
            n_timesteps=1000,
            clip_when_sampling=False,
        ),
    )

    config.training = new_dict(
        optimizer=new_dict(
            name="adamw",
            learning_rate=1e-3,
            weight_decay=1e-4,
            do_warmup=True,
            warmup_steps=100,
            do_decay=True,
            decay_steps=500_000,
            end_learning_rate=1e-5,
            do_gradient_clipping=True,
            gradient_clipping=1.0,
        ),
        ema_rate=0.999,
        n_epochs=1000,
        batch_size=128,
    )

    return config
