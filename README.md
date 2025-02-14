# Domain-Specific AI Assistant Enhancement Project

## Overview
This project focuses on enhancing the quality of Large Language Models (LLMs) for domain-specific AI assistants.
The implementation leverages UniXcoder for traditional code search capabilities and works with the CodeSearchNet dataset, with plans to expand to custom repositories.

## Team Members
- Carter Cutsforth (cvcuts@iastate.edu)
- Jacob Duba (jduba@iastate.edu)
- Keenan Jacobs (kcjacobs@iastate.edu)
- Conor O'Shea (coshea@iastate.edu)
- Diego Perez (joceo@iastate.edu)

## Getting Started
*Instructions for setting up and running the project will be added as development progresses*

### Prerequisites
- UV package installer ([Installation guide](https://github.com/astral-sh/uv))

### Installation
1. Create and activate UV environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. Install local UniXcoder package:
   ```bash
   cd UniXcoder/
   uv pip install -e .
   cd ../
   ```

### Running
Run the code search application:
```bash
python CodeSearch/run.py
```

## Contributing
1. Clone the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
*License information to be added*

## Contact
For questions or collaboration inquiries, please contact any team member listed above.

---
*This project is part of COM S 4020 at Iowa State University*
