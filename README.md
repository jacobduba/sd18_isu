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

### Prerequisites
- UV package installer ([Installation guide](https://github.com/astral-sh/uv))

### Installation
1. Create and activate UV environment:
   ```bash
   uv venv
   # If using bash/zsh
   source .venv/bin/activate
   # If using fish
   source .venv/bin/activate.fish
   ```
2. Install dependencies:
   ```bash
   make install
   ```
3. Export your API key:
   ```bash
   make env
   ```
   Then copy and paste the printed line into your terminal.

   - For **Bash/Zsh**, just paste the `export` command.
   - For **Fish shell**, run:
     ```fish
     set -x OPENROUTER_API_KEY your-key-here
     ```

### Running
Run the script that creates the data:
```bash
make create
```
Run the code search script:
```bash
make search
```

## License
*License information to be added*

## Contact
For questions or collaboration inquiries, please contact any team member listed above.

---
*This project is part of COM S 4020 at Iowa State University*
