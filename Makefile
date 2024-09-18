.PHONY: exo-1
exo-1:
	uv run python tabular_marl/train_iql.py

.PHONY: exo-2.1
exo-2.1:
	cd marl-book-codebase/marlbase && uv run python run.py +algorithm=ia2c env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=100000

.PHONY: exo-2.2
exo-2.2:
	cd marl-book-codebase/marlbase && uv run python run.py +algorithm=ia2c env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=1000000 algorithm.name=ia2c env.standardise_rewards=True algorithm.video_interval=200000 seed=0


.PHONY: exo-2.3
exo-2.3:
	cd marl-book-codebase && uv run python -m marlbase.utils.postprocessing.plot_runs --source marlbase/outputs


.PHONY: exo-2.4
exo-2.4:
	cd marl-book-codebase && uv run python -m marlbase.utils.postprocessing.plot_runs --source marlbase/outputs --metric entropy

.PHONY: exo-2.5
exo-2.5:
	cd marl-book-codebase/marlbase && uv run python run.py -m +algorithm=ia2c env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=1000000 env.standardise_rewards=True seed=0 algorithm.entropy_coef=0.001,0.01,0.1

.PHONY: exo-2.6
exo-2.6:
	cd marl-book-codebase && uv run python -m marlbase.utils.postprocessing.find_best_hyperparams --source marlbase/multirun

.PHONY: exo-2.7
exo-2.7:
	cd marl-book-codebase && uv run python -m marlbase.utils.postprocessing.plot_runs --source marlbase/multirun

.PHONY: exo-2.8
exo-2.8:
	cd marl-book-codebase/marlbase && uv run python run.py -m +algorithm=ia2c,maa2c env.name="rware:rware-tiny-4ag-v2" env.time_limit=500 algorithm.total_steps=20000000 env.standardise_rewards=True seed=0,1,2

.PHONY: exo-2.9
exo-2.9:
	cd marl-book-codebase && uv run python -m marlbase.utils.postprocessing.plot_runs --source ../deep_marl_data/rware_tiny_4ag --metric value_loss

.PHONY: exo-2.10
exo-2.10:
	cd marl-book-codebase/marlbase && uv run python run.py +algorithm=idqn env.name="lbforaging:Foraging-8x8-2p-3f-v3" env.time_limit=50 algorithm.total_steps=4000000 algorithm.eval_interval=100000 algorithm.log_interval=100000 env.standardise_rewards=True env.wrappers="[CooperativeReward]"

.PHONY: exo-2.11
exo-2.11:
	cd marl-book-codebase/marlbase && uv run python run.py +algorithm=idqn env.name="smaclite/2s3z-v0" env.time_limit=150 algorithm.total_steps=1000000 algorithm.eval_interval=1000 algorithm.log_interval=1000 env.standardise_rewards=True

