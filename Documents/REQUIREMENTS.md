# Project Requirements: Domain-Specific AI Code Search


## Functional Requirements

1. **Natural Language Input**
   - The system must accept a natural language query describing a desired code behavior.

2. **Comment-Code Embedding**
   - Code comments and queries must be embedded using UniXCoder or an equivalent model.

3. **Vector Similarity Search**
   - The backend must compute dot product similarity between the query embedding and all comment embeddings to identify relevant matches.

4. **LLM-Based Ranking**
   - The top N results from the vector search must be re-ranked using an LLM via OpenRouter to enhance relevance.

5. **Searchable Dataset**
   - The system must support searching over datasets like CodeSearchNet, with the ability to substitute domain-specific datasets.

6. **Code Snippet Output**
   - The system must display code snippets ranked by relevance, with associated confidence scores.

7. **Basic UI**
   - A user-friendly web-based UI must allow inputting queries and viewing results.

---

## Non-Functional Requirements

1. **Performance**
   - Average query-response time should be under 5 seconds for small to medium datasets.

2. **Scalability**
   - The system must scale to larger datasets, including enterprise and HPC environments.

3. **Robustness**
   - The application must handle API failures and provide fallback behavior or error messages.

4. **Security**
   - Internal codebases or user data must be protected and not exposed to external systems unnecessarily.

5. **Extensibility**
   - The system should be modular to allow future replacement of components (e.g., swapping LLMs or embedding models).

6. **Usability**
   - The UI should be simple and intuitive for software developers.

---


## Tech Stack & Tools

- **Languages**: Python  
- **Models**: UniXCoder, DeepSeek LLM
- **Frameworks**: Flask, vector search libraries
- **Version Control**: GitLab with merge requests, code reviews, and issue tracking  


