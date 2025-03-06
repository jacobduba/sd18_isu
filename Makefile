help:
	@echo "Commands:"
	@echo "  make venv      - Create a virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make create    - Run the create_data.py script"
	@echo "  make clean     - Remove temporary files"
	@echo "  make search    - Remove temporary files"
	@echo "  make env       - Set the OpenRouter API key"

install:
	cd UniXcoder && uv pip install -e .
	uv pip install -r requirements.txt

create:
	python3 CodeSearch/create_data.py

clean:
	rm -rf __pycache__ $(VENV)

search:
	OPENROUTER_API_KEY=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 python3 CodeSearch/search.py
