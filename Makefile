ECHO_SUCCESS=@echo " \033[1;32m✔\033[0m  "

all: download, learn

.PHONY: learn

download:
	@rm -rf data
	@mkdir data
	@wget -O data/whites.csv http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

learn:
	cd learn && node learn.js