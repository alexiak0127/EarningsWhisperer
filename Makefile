# Makefile for EarningsWhisperer Project

# Virtual environment
VENV = earningsenv

# Default target
all: install data process model neural visualize

# Setup virtual environment and install dependencies
install:
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

# Generate subset of data for testing
data: install
	./$(VENV)/bin/python3 create_sample_data.py

# Process data
process: data
	./$(VENV)/bin/python3 data_processing.py

# Run traditional ML models
model: process
	./$(VENV)/bin/python3 modeling.py

# # Run neural network models
# neural: model
# 	./$(VENV)/bin/python3 neural_network_model.py

# Run Visualizations
visualize:
	./$(VENV)/bin/python3 visualizations.py

# Clean up
clean:
	rm -rf $(VENV)

# Reinstall everything
reinstall: clean install

# Run everything without virtual environment (for GitHub Actions / CI)
run:
	python3 create_sample_data.py
	python3 enhanced_sentiment_analysis.py
	python3 data_processing.py
	python3 modeling.py
