ECHO_SUCCESS=@echo " \033[1;32m✔\033[0m  "

all: download, learn

.PHONY: learn

download:
	@rm -rf data
	@mkdir data
	@wget -O data/titanic.csv http://lib.stat.cmu.edu/S/Harrell/data/ascii/titanic.txt

learn:
	node learn.js