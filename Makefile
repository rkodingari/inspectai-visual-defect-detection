.PHONY: install data train train-all evaluate test api ui quick

install:
	python -m pip install -e '.[dev]'

data:
	python -m inspectai.data.prepare --dataset mvtec --category bottle

quick:
	python -m inspectai.data.prepare --dataset synthetic --output data/processed/synthetic --force
	python -m inspectai.training.train --data-dir data/processed/synthetic --model baseline_cnn --epochs 2 --quick

train:
	python -m inspectai.training.train --data-dir data/processed/mvtec_bottle --model mobilenet_v3_small

train-all:
	python -m inspectai.training.train_all --data-dir data/processed/mvtec_bottle

evaluate:
	python -m inspectai.evaluation.evaluate --checkpoint artifacts/best_model.pt --data-dir data/processed/mvtec_bottle

test:
	pytest

api:
	uvicorn inspectai.api.main:app --reload

ui:
	streamlit run src/inspectai/ui/app.py
