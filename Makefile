.PHONY: setup collect analyze visualize blog all clean test ui

setup:
	pip install -r requirements.txt
	python -m textblob.download_corpora lite

collect:
	python run.py collect

analyze:
	python run.py analyze

visualize:
	python run.py visualize

blog:
	python run.py blog

all: collect analyze visualize blog

test:
	pytest tests/ -v

ui:
	streamlit run app.py

clean:
	rm -rf data/ output/
