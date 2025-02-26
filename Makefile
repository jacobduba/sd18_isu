PYTHON = python3
VENV = venv

help:
	@echo "Commands:"
	@echo "  make venv      - Create a virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make create       - Run the create_data.py script"
	@echo "  make clean     - Remove temporary files"
	@echo "  make search     - Remove temporary files"
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