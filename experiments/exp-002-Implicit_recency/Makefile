.DEFAULT: help

help:
	@echo "     USAGE"
	@echo "==============="
	@echo "make -B dashboard"
	@echo "       Starts MLFlow UI on 0.0.0.0, port 8011"
	@echo "make -B search_ab"
	@echo "       Starts grid search 'alpha' and 'beta' recency params for Implicit model."  
	@echo "make -B search_pruning"
	@echo "       Starts grid search 'pruning' param for Implicit model."  

grid_search_ab:
	@python3 scripts/train_and_search.py

grid_search_pruning:
	@python3 scripts/train_and_search.py --alpha 1200 --beta 800

dashboard:
	@mlflow ui -h 0.0.0.0 -p 8011
