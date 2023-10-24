PYTHONPATH=. python3 src/functional_diffusion_processes/run.py --multirun +experiments_maml=exp_deblurring \
predictors=euler \
correctors=langevin \
correctors.snr=0.19 \
datasets.train.data_config.seed=4 \
datasets.train.data_config.batch_size=4 \
datasets.train.data_config.image_height_size=64 \
datasets.train.data_config.image_width_size=64 \
samplers.sampler_config.N=1000 \
samplers.sampler_config.k=1 \
sdes.sde_config.probability_flow=False \
sdes.sde_config.factor=1 \
samplers.sampler_config.denoise=True
