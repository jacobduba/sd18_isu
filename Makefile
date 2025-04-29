
help:
	@echo "Commands:"
	@echo "  make venv      - Create a virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make create    - Run the create_data.py script"
	@echo "  make clean     - Remove temporary files"
	@echo "  make search    - Remove temporary files"
	@echo "  make search-dev- search but utilizes the small model"
	@echo "  make env       - Set the OpenRouter API key"
	@echo "  make ollama    - Start Ollama Model"
	@echo "  make small     - Start 1.1GB Ollama Model"

install:
	cd UniXcoder && uv pip install -e .
	uv pip install -r requirements.txt

create:
	python3 CodeSearch/create_data.py

clean:
	rm -rf __pycache__ $(VENV)
	rm -rf embeddings*

search:
	python3 CodeSearch/search.py 

search-dev:
	python3 CodeSearch/search.py 1

ollama:
	ollama run deepseek-coder-v2:latest

small:
	ollama run deepseek-r1:1.5b
