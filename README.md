# 🧠 Plagiarism Detector for Code Submissions (Python)

A powerful and scalable plagiarism detection tool for student code submissions. Built using the Rabin-Karp string matching algorithm, graph-based clustering, greedy representative selection, and efficient B+ Tree indexing.

## 🎯 Objective

The goal is to identify groups of code submissions that are suspiciously similar and help instructors efficiently review potential plagiarism cases.

---

## 🛠 Technologies Used

- **Language**: Python
- **Algorithms**:
  - Rabin-Karp for string matching
  - Graph traversal using BFS/DFS
  - Greedy algorithm for representative selection
- **Data Structures**:
  - B+ Tree (simulated if library not allowed)
  - Merge Sort for sorting similarity scores

---

## 📋 Key Features

- 📄 Tokenizes and parses student code for comparison
- 🔍 Detects code similarities using Rabin-Karp
- 📈 Builds a weighted similarity graph
- 🧱 Clusters similar submissions using BFS/DFS
- 🌟 Selects best representative(s) using a Greedy algorithm
- 📂 Stores and retrieves metadata (Student ID, Timestamp, etc.) via B+ Tree
- ⚙️ Supports real-time submission updates (optional)

---

## 📌 Functional Requirements

1. **Parsing & Tokenization**: Breaks down code into tokens for standardized comparison.
2. **Rabin-Karp Matching**: Efficient substring comparison to detect shared sequences.
3. **Similarity Score**: Computes a normalized score based on matching sequences.
4. **Similarity Graph**: Nodes are submissions, edges show similarity above a threshold.
5. **Clustering via BFS/DFS**: Groups similar submissions together.
6. **Greedy Representative Selection**: Picks the most “central” or “suspicious” submissions.
7. **B+ Tree for Metadata**: Enables fast lookup of student information.
8. **Report Generation**: Shows clusters, representative files, and optional similarity scores.
9. **(Optional)** Real-time comparison as new submissions arrive.

---

## 🧪 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/plagiarism-detector-python.git
cd plagiarism-detector-python
