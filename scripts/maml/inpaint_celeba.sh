PYTHONPATH=. python3 src/functional_diffusion_processes/run.py --multirun +experiments_maml=exp_inpainting \
predictors=euler \
correctors=langevin \
correctors.snr=0.06 \
datasets.train.data_config.batch_size=4 \
datasets.train.data_config.image_height_size=64 \
datasets.train.data_config.image_width_size=64 \
samplers.sampler_config.N=100 \
samplers.sampler_config.k=1 \
sdes.sde_config.probability_flow=True \
sdes.sde_config.factor=2 \
samplers.sampler_config.denoise=True
