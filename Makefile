PYTHON ?= python3

.PHONY: install run quick test dashboard

install:
	$(PYTHON) -m pip install -r requirements.txt

# full run: 150 students x 24 months (about 20-60 seconds)
run:
	$(PYTHON) -m spendinsights.pipeline

# small run for a quick sanity check
quick:
	$(PYTHON) -m spendinsights.pipeline --students 40 --months 16 --test-months 3 --no-deck

test:
	$(PYTHON) -m pytest -q

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py
