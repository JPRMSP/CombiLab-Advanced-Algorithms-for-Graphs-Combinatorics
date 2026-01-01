import streamlit as st
import itertools
import random
import networkx as nx
import numpy as np
from sympy import Matrix

st.set_page_config(page_title="CombiLab+", layout="centered")

st.title("🧠 CombiLab+ — Advanced Algorithms for Graphs & Combinatorics")
st.caption("Pure algorithmic exploration — no datasets, no AI models")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Permutations (Gen / Rank / Unrank)",
        "k-Subsets + Ranking / Unranking",
        "Integer Partitions & Gray Walk",
        "Generating Functions",
        "Prüfer Codes & Tree Visualizer",
        "Matrix Tree Theorem",
        "Graph Certificates (Isomorphism Hint)"
    ]
)

# ---------------- PERMUTATIONS ----------------
if menu == "Permutations (Gen / Rank / Unrank)":
    st.header("🔁 Permutations")

    n = st.number_input("Choose n (1–8 recommended)", 1, 9, 4)

    # Lexicographic permutations
    perms = list(itertools.permutations(range(1, n+1)))

    st.subheader("Lexicographic Permutations")
    st.write(perms)

    # Ranking
    k = st.number_input("Rank (0-based)", 0, len(perms)-1, 0)
    st.success(f"Permutation at rank {k}: {perms[k]}")

    # Unranking
    st.subheader("Unranking Example")
    input_perm = st.text_input(f"Enter a permutation of 1..{n} (comma separated)")
    if input_perm:
        try:
            p = tuple(map(int, input_perm.split(",")))
            rank = perms.index(p)
            st.info(f"Rank of {p} = {rank}")
        except:
            st.error("Invalid permutation")

    st.subheader("Johnson–Trotter Animation")
    def johnson_trotter(n):
        perm = list(range(1, n+1))
        dir = [-1]*n
        result = [perm.copy()]

        def largest_mobile():
            best = -1
            idx = -1
            for i in range(n):
                j = i + dir[i]
                if 0 <= j < n and perm[i] > perm[j] and perm[i] > best:
                    best = perm[i]
                    idx = i
            return idx

        while True:
            idx = largest_mobile()
            if idx == -1:
                break
            j = idx + dir[idx]
            perm[idx], perm[j] = perm[j], perm[idx]
            dir[idx], dir[j] = dir[j], dir[idx]
            for i in range(n):
                if perm[i] > perm[j]:
                    dir[i] *= -1
            result.append(perm.copy())
        return result

    st.write(johnson_trotter(n))


# ---------------- K SUBSETS ----------------
elif menu == "k-Subsets + Ranking / Unranking":
    st.header("📦 k-Subsets — Generation, Rank & Unrank")

    n = st.number_input("n", 1, 12, 6)
    k = st.number_input("k", 0, n, 3)

    all_k = list(itertools.combinations(range(1, n+1), k))
    st.write(all_k)

    idx = st.number_input("Rank index", 0, len(all_k)-1, 0)
    st.success(f"k-subset at rank {idx}: {all_k[idx]}")

    st.subheader("Unranking")
    entered = st.text_input("Enter subset (comma separated)")
    if entered:
        try:
            s = tuple(sorted(map(int, entered.split(","))))
            r = all_k.index(s)
            st.info(f"Rank of {s}: {r}")
        except:
            st.error("Invalid subset")

    st.subheader("Gray-code k-subsets (simple idea)")
    def gray_k(n, k):
        result = []
        for mask in range(1 << n):
            if bin(mask).count("1") == k:
                subset = tuple(i+1 for i in range(n) if mask & (1 << i))
                result.append(subset)
        return result
    st.write(gray_k(n, k))


# ---------------- PARTITIONS ----------------
elif menu == "Integer Partitions & Gray Walk":
    st.header("🍰 Integer Partitions")

    n = st.number_input("Enter integer", 1, 30, 7)

    def partitions(num, max_val=None):
        if max_val is None:
            max_val = num
        if num == 0:
            return [[]]
        result = []
        for i in range(min(num, max_val), 1-1, -1):
            for p in partitions(num-i, i):
                result.append([i] + p)
        return result

    P = partitions(n)
    st.write(P)
    st.success(f"Total partitions: {len(P)}")

    st.subheader("Gray-style Walk (small changes)")
    for i in range(min(len(P)-1, 6)):
        st.write(f"{P[i]}  →  {P[i+1]}")


# ---------------- GENERATING FUNCTIONS ----------------
elif menu == "Generating Functions":
    st.header("🧮 Generating Functions")

    st.subheader("Fibonacci via Generating Function")
    fib_n = st.number_input("Terms", 1, 20, 8)

    fib = [0, 1]
    for _ in range(fib_n-2):
        fib.append(fib[-1] + fib[-2])
    st.write(fib)

    st.subheader("Coin Change (Generating Function idea)")
    total = st.number_input("Amount", 1, 30, 6)
    coins = [1, 2, 5]

    ways = [0]*(total+1)
    ways[0] = 1
    for c in coins:
        for i in range(c, total+1):
            ways[i] += ways[i-c]

    st.success(f"Ways to make {total} using {coins}: {ways[total]}")


# ---------------- PRUFER ----------------
elif menu == "Prüfer Codes & Tree Visualizer":
    st.header("🌳 Prüfer Codes")

    n = st.number_input("Vertices", 2, 12, 6)
    mode = st.radio("Choose", ["Random Tree → Prüfer", "Prüfer → Tree"])

    if mode == "Random Tree → Prüfer":
        T = nx.random_tree(n)
        code = list(nx.to_prufer_sequence(T))
        st.write("Edges:", sorted(T.edges()))
        st.success(f"Prüfer code: {code}")

        st.graphviz_chart(nx.nx_pydot.to_pydot(T).to_string())

    else:
        s = st.text_input("Enter Prüfer code")
        if s:
            code = list(map(int, s.split(",")))
            T = nx.from_prufer_sequence(code)
            st.write("Edges:", sorted(T.edges()))
            st.graphviz_chart(nx.nx_pydot.to_pydot(T).to_string())


# ---------------- MATRIX TREE ----------------
elif menu == "Matrix Tree Theorem":
    st.header("🌉 Counting Spanning Trees")

    n = st.number_input("Nodes", 2, 8, 5)
    p = st.slider("Edge probability", 0.1, 1.0, 0.5)

    G = nx.gnp_random_graph(n, p)
    L = nx.laplacian_matrix(G).todense()
    minor = Matrix(L[1:, 1:])
    count = int(round(minor.det()))

    st.write("Edges:", sorted(G.edges()))
    st.success(f"Number of spanning trees: {count}")


# ---------------- GRAPH CERTIFICATES ----------------
elif menu == "Graph Certificates (Isomorphism Hint)":
    st.header("🔍 Simple Graph Certificate")

    n = st.number_input("Nodes (same for both)", 2, 9, 6)

    G1 = nx.gnp_random_graph(n, 0.5)
    G2 = nx.gnp_random_graph(n, 0.5)

    cert1 = sorted(dict(G1.degree()).values())
    cert2 = sorted(dict(G2.degree()).values())

    st.write("Certificate G1 (degree sorted):", cert1)
    st.write("Certificate G2 (degree sorted):", cert2)

    if cert1 == cert2:
        st.warning("Certificates match — graphs MAY be isomorphic (not guaranteed).")
    else:
        st.success("Certificates differ — graphs are NOT isomorphic.")
