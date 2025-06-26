import hashlib
import os
from collections import defaultdict, deque
import re
import uuid
from typing import Dict, List, Set, Tuple
import time

# B+ Tree Node for metadata storage
class BPlusTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.values = []
        self.children = []
        self.next = None

class BPlusTree:
    def __init__(self, order=3):
        self.root = BPlusTreeNode(leaf=True)
        self.order = order

    def insert(self, key: str, value: dict):
        node = self.root
        if len(node.keys) >= 2 * self.order - 1:
            new_root = BPlusTreeNode()
            self.root = new_root
            new_root.children.append(node)
            self._split_child(new_root, 0)
            self._insert_non_full(new_root, key, value)
        else:
            self._insert_non_full(node, key, value)

    def _split_child(self, parent: BPlusTreeNode, index: int):
        order = self.order
        node = parent.children[index]
        new_node = BPlusTreeNode(leaf=node.leaf)
        mid = order - 1

        new_node.keys = node.keys[mid:]
        if node.leaf:
            new_node.values = node.values[mid:]
        else:
            new_node.children = node.children[mid:]
        node.keys = node.keys[:mid]
        if node.leaf:
            node.values = node.values[:mid]
            new_node.next = node.next
            node.next = new_node
        else:
            node.children = node.children[:mid + 1]

        parent.keys.insert(index, new_node.keys[0])
        parent.children.insert(index + 1, new_node)

    def _insert_non_full(self, node: BPlusTreeNode, key: str, value: dict):
        if node.leaf:
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, key)
            node.values.insert(i + 1, value)
        else:
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            child = node.children[i]
            if len(child.keys) >= 2 * self.order - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, value)

    def search(self, key: str) -> dict:
        node = self.root
        while True:
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if node.leaf:
                for j, val in enumerate(node.keys):
                    if val == key:
                        return node.values[j]
                return None
            node = node.children[i]

# Rabin-Karp for similarity detection
def rabin_karp(text1: str, text2: str, k: int = 3, q: int = 101) -> int:
    if not text1 or not text2:
        return 0
    d = 256  # Number of characters in input alphabet
    h = pow(d, k-1) % q
    matches = 0
    text1 = text1.lower()
    text2 = text2.lower()

    # Preprocess to remove comments and normalize whitespace
    text1 = re.sub(r'#.*$', '', text1, flags=re.MULTILINE)  # Remove Python comments
    text2 = re.sub(r'#.*$', '', text2, flags=re.MULTILINE)  # Remove Python comments
    text1 = re.sub(r'\s+', ' ', text1.strip())
    text2 = re.sub(r'\s+', ' ', text2.strip())

    # Create k-grams
    kgrams1 = [text1[i:i+k] for i in range(len(text1) - k + 1)]
    kgrams2 = [text2[i:i+k] for i in range(len(text2) - k + 1)]

    # Compute hash for first k-gram of text1
    p = 0
    for i in range(k):
        p = (d * p + ord(text1[i])) % q

    # Store hashes for text2 k-grams
    t_hashes = set()
    t = 0
    for i in range(k):
        t = (d * t + ord(text2[i])) % q
    t_hashes.add(t)

    for i in range(len(text2) - k):
        t = (d * (t - ord(text2[i]) * h) + ord(text2[i + k])) % q
        if t < 0:
            t += q
        t_hashes.add(t)

    # Check for matches
    for i in range(len(kgrams1)):
        if p in t_hashes:
            # Verify match to avoid hash collisions
            if kgrams1[i] in kgrams2:
                matches += 1
        if i < len(kgrams1) - 1:
            p = (d * (p - ord(text1[i]) * h) + ord(text1[i + k])) % q
            if p < 0:
                p += q

    # Calculate similarity score
    total_kgrams = max(len(kgrams1), len(kgrams2))
    return (matches / total_kgrams) * 100 if total_kgrams > 0 else 0

# Graph for clustering
class Graph:
    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u: str, v: str, weight: float):
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))

    def bfs_clusters(self, threshold: float) -> List[Set[str]]:
        visited = set()
        clusters = []
        for node in self.adj:
            if node not in visited:
                cluster = set()
                queue = deque([node])
                visited.add(node)
                while queue:
                    curr = queue.popleft()
                    cluster.add(curr)
                    for neighbor, weight in self.adj[curr]:
                        if neighbor not in visited and weight >= threshold:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if cluster:
                    clusters.append(cluster)
        return clusters

# Greedy algorithm for selecting representatives
def select_representative(cluster: Set[str], similarity_scores: Dict[Tuple[str, str], float]) -> str:
    max_score = -1
    representative = None
    for file in cluster:
        total_similarity = sum(
            similarity_scores.get((min(file, other), max(file, other)), 0)
            for other in cluster if other != file
        )
        avg_similarity = total_similarity / (len(cluster) - 1) if len(cluster) > 1 else 0
        if avg_similarity > max_score:
            max_score = avg_similarity
            representative = file
    return representative

# Plagiarism Detector
class PlagiarismDetector:
    def __init__(self, similarity_threshold: float = 70.0, k: int = 3):
        self.files = {}
        self.metadata = BPlusTree()
        self.graph = Graph()
        self.similarity_scores = {}
        self.threshold = similarity_threshold
        self.k = k

    def parse_file(self, filepath: str, metadata: dict):
        try:
            if os.path.getsize(filepath) == 0:
                return False
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            file_id = str(uuid.uuid4())
            self.files[file_id] = content
            self.metadata.insert(file_id, {'filepath': filepath, **metadata})
            return file_id
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return False

    def compute_similarities(self):
        for i, (id1, content1) in enumerate(self.files.items()):
            for id2, content2 in list(self.files.items())[i+1:]:
                score = rabin_karp(content1, content2, k=self.k)
                self.similarity_scores[(min(id1, id2), max(id2, id1))] = score
                if score >= self.threshold:
                    self.graph.add_edge(id1, id2, score)

    def detect_plagiarism(self) -> Dict[str, dict]:
        self.compute_similarities()
        clusters = self.graph.bfs_clusters(self.threshold)
        results = {}
        for cluster in clusters:
            if len(cluster) > 1:  # Only consider clusters with multiple files
                rep = select_representative(cluster, self.similarity_scores)
                rep_metadata = self.metadata.search(rep)
                # Create a mapping of similarity scores with file paths instead of IDs
                similarity_scores_with_paths = {}
                for other in cluster:
                    if other != rep:
                        score = self.similarity_scores.get((min(rep, other), max(rep, other)), 0)
                        other_metadata = self.metadata.search(other)
                        other_path = other_metadata['filepath'] if other_metadata else other
                        similarity_scores_with_paths[other_path] = round(score, 2)
                results[rep] = {
                    'cluster': list(cluster),
                    'metadata': rep_metadata,
                    'similarity_scores': similarity_scores_with_paths
                }
        return results

    def add_file(self, filepath: str, metadata: dict) -> dict:
        file_id = self.parse_file(filepath, metadata)
        if not file_id:
            return {'error': 'Failed to parse file'}
        # Recompute similarities for new file
        for id2, content2 in self.files.items():
            if id2 != file_id:
                score = rabin_karp(self.files[file_id], content2, k=self.k)
                self.similarity_scores[(min(file_id, id2), max(file_id, id2))] = score
                if score >= self.threshold:
                    self.graph.add_edge(file_id, id2, score)
        return self.detect_plagiarism()

# Pretty print results
def print_results(test_case: str, results: Dict[str, dict], execution_time: float = None):
    print("\n" + "=" * 50)
    print(f"📊 {test_case.upper()}")
    if execution_time is not None:
        print(f"⏱️ Execution Time: {execution_time:.2f} seconds")
    print("=" * 50 + "\n")

    if not results:
        print("  No plagiarism clusters detected.\n")
        return

    for idx, (rep, data) in enumerate(results.items(), 1):
        print(f"🔍 Cluster {idx}")
        print(f"  Representative File: {data['metadata']['filepath']}")
        print(f"  Author: {data['metadata'].get('author', 'Unknown')}")
        print(f"  Files in Cluster ({len(data['cluster'])} files):")
        for fid in data['cluster']:
            file_metadata = detector.metadata.search(fid)
            file_path = file_metadata['filepath'] if file_metadata else fid
            file_author = file_metadata.get('author', 'Unknown') if file_metadata else 'Unknown'
            print(f"    - {file_path} (Author: {file_author})")
        print("  Similarity Scores (Compared to Representative):")
        for file_path, score in data['similarity_scores'].items():
            print(f"    - {file_path}: {score}%")
        print("\n" + "-" * 50 + "\n")

# Example usage and testing
if __name__ == "__main__":
    # Test Case 1: Basic Functionality
    detector = PlagiarismDetector(similarity_threshold=70.0, k=3)
    files = {
        'Testcase 1/A.py': ('def sum(a, b): return a + b', {'author': 'Ali'}),
        'Testcase 1/B.py': ('def sum(a, b): return a + b', {'author': 'Bisam'}),
        'Testcase 1/C.py': ('def add(x, y): return x + y', {'author': 'Hamza'}),
        'Testcase 1/D.py': ('def product(a, b): return a * b', {'author': 'Farhan'}),
    }
    for filepath, (content, metadata) in files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        detector.parse_file(filepath, metadata)

    results = detector.detect_plagiarism()
    print_results("Test Case 1: Basic Functionality", results)

    # Test Case 2: Edge Cases
    detector = PlagiarismDetector(similarity_threshold=70.0, k=3)
    edge_files = {
        'Testcase 2/E.py': ('def compute(x): return x * 2', {'author': 'Emma'}),
        'Testcase 2/F.py': ('# Comment\ndef compute(x):\n    return x * 2', {'author': 'Faizan'}),
        'Testcase 2/G.py': ('def compute(x): y = x * 2; return y', {'author': 'Gwen'}),
        'Testcase 2/H.java': ('public class Test { int test() { return 0; } }', {'author': 'Hanan'}),
        'Testcase 2/I.py': ('', {'author': 'Imman'}),
    }
    # # Add j1.py to j50.py for B+ Tree efficiency testing
    # for i in range(1, 51):
    #     edge_files[f'Testcase 2/j{i}.py'] = (f'def calc(x): return x + {i}', {'author': f'Author{i}'})

    for filepath, (content, metadata) in edge_files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        detector.parse_file(filepath, metadata)

    results = detector.detect_plagiarism()
    print_results("Test Case 2: Edge Cases", results)

    # Test Case 3: Algorithm Integration and Complexity
    detector = PlagiarismDetector(similarity_threshold=70.0, k=3)
    large_files = {
        'Testcase 3/group 1/j.py': ('def process(x): return x * 3', {'author': 'AuthorJ'}),
        'Testcase 3/group 1/k.py': ('def process(x): return x * 3', {'author': 'AuthorK'}),
        'Testcase 3/group 1/l.py': ('def proc(y): return y * 3', {'author': 'AuthorL'}),
        'Testcase 3/group 2/m.py': ('def double(x): return x + x', {'author': 'AuthorM'}),
        'Testcase 3/group 2/n.py': ('def double(x): return x + x', {'author': 'AuthorN'}),
        'Testcase 3/group 2/o.py': ('def double(x): return x + x', {'author': 'AuthorN'}),
        'greedy testing/file_high_sim.py': ('def process(x): return x * 3', {'author': 'AuthorHigh'}),
        'greedy testing/file_mod_sim.py': ('def process(x): return x * 3 + 1', {'author': 'AuthorMod'}),
    }
    # # Add unique files to reach 20+ total files
    # for i, letter in enumerate('OPQRSTUVWXYZ', start=1):
    #     large_files[f'Testcase 3/{letter}.py'] = (f'def unique{i}(x): return x ** {i}', {'author': f'Author{letter}'})
    # for i in range(1, 9):
    #     large_files[f'Testcase 3/unique{i}.py'] = (f'def func{i}(x): return x + {i}', {'author': f'AuthorUnique{i}'})

    for filepath, (content, metadata) in large_files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        detector.parse_file(filepath, metadata)

    start_time = time.time()
    results = detector.detect_plagiarism()
    execution_time = time.time() - start_time
    print_results("Test Case 3: Algorithm Integration and Complexity", results, execution_time)