.PHONY: install-deps run

install-deps:
	@echo "Installing Python dependencies from requirements.txt..."
	pip install -r requirements.txt
	@echo "Dependencies installed."

run:
	@echo "Running the Signature Predictor application..."
	python app/main.py
