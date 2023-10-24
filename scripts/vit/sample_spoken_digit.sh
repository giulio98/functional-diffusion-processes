PYTHONPATH=. python3 src/functional_diffusion_processes/run.py --multirun +experiments_vit=exp_sampling_spoken_digit \
correctors.snr=0.17 \
sdes.sde_config.probability_flow=True,False \
sdes.sde_config.factor=0.6 \
samplers.sampler_config.N=1000
