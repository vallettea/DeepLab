ECHO_SUCCESS=@echo " \033[1;32m✔\033[0m  "

all: download, learn

.PHONY: learn

download:
	@rm -rf data
	@mkdir data
	@wget -O data/servo.csv https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data

learn:
	node learn.js