PYTHON = python3
VENV = venv

help:
	@echo "Commands:"
	@echo "  make venv      - Create a virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make run       - Run the create_data.py script"
	@echo "  make clean     - Remove temporary files"

venv:
	$(PYTHON) -m venv $(VENV)
install: venv
	$(VENV)/Scripts/pip install -r requirements.txt  	    
run: install
	$(VENV)/Scripts/python CodeSearch/create_data.py  	      
clean:
	rm -rf __pycache__ $(VENV)
