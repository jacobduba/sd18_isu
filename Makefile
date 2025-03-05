PYTHON = python3
VENV = venv

help:
	@echo "Commands:"
	@echo "  make venv      - Create a virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make create    - Run the create_data.py script"
	@echo "  make clean     - Remove temporary files"
	@echo "  make search    - Remove temporary files"
	@echo "  make env       - Set the OpenRouter API key"
venv:
	$(PYTHON) -m venv $(VENV)
install: venv
	$(VENV)/Scripts/pip install -r requirements.txt  	    
create: install
	$(VENV)/Scripts/python CodeSearch/create_data.py  	      
clean:
	rm -rf __pycache__ $(VENV)
search: install
	$(VENV)/Scripts/python CodeSearch/search.py
env:
	@echo "Setting OpenRouter API Key..."
	@echo 'export OPENROUTER_API_KEY=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' >> .env
	@echo "Done. Restart your shell or run 'source .env' to apply changes."