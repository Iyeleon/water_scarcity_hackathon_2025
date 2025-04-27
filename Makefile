.PHONY: prep_dataset
prep_dataset: src/preprocessing.py ./data/dataset/
	@python -m src.preprocessing

.PHONY: prep_mini_dataset
prep_mini_dataset: src/preprocessing.py ./data/dataset ./data/dataset_mini_challenge/
	@python -m src.preprocessing --is-mini	

.PHONY: final_data
final_data: src/feature_engineering.py ./data/preprocessed
	@python -m src.feature_engineering

.PHONY: feature_selection
feature_selection: src/feature_engineering.py data/final/train.csv
	@python -m src.feature_selection