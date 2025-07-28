# Adobe Problem 1B — Persona-Driven Document Intelligence

## 🔍 Problem Overview

This solution reads PDF documents, segments them into meaningful sections, ranks them by relevance to a given persona and job, and then generates concise summaries tailored to that role.

---

## 📁 Folder Structure
```bash
Problem1b/
├── process.py
├── input/
├── output/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 💡 Approach

- **Semantic Chunking**: PDFs are split into sections based on font size and layout structure.
- **Ranking**: SentenceTransformers (`all-MiniLM-L6-v2`) ranks the relevance of each section to the persona’s task.
- **Summarization**: Uses a T5 model (`google/flan-t5-small`) to summarize top-ranked sections tailored to the user's goal.

---

## 🧠 Models Used

| Task               | Model                         |
|--------------------|-------------------------------|
| Sentence Embedding | `all-MiniLM-L6-v2`             |
| Summarization      | `google/flan-t5-small`         |

- Model size < 200MB each
- Runs fully offline (no internet required)

---

## ⚙️ How to Build

```bash
docker build --platform linux/amd64 -t adobe_problem_1b:layout .
```

# Run it (mount input/output folders)
For Windows PowerShell:
```bash
docker run --rm `
  -v ${PWD}\input:/app/input `
  -v ${PWD}\output:/app/output `
  adobe_problem_1b:layout `
  python process.py --input_json /app/input/challenge1b_input.json
```
For Windows Git Bash:
```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  adobe_problem_1b:layout \
  python process.py --input_json /app/input/challenge1b_input.json
```