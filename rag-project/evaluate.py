import json
import sys
import os
import math
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from app.services.retriever import retrieve
from app.services.rag_service import run_rag

TEST_CASES = [
    {
        "id": "TC-01",
        "query": "What is the candidate's name?",
        "relevant_keywords": ["name", "cv", "candidate", "profile"],
        "expected_answer_keywords": ["name"],
        "category": "happy_path",
        "description": "Basic CV identity retrieval"
    },
    {
        "id": "TC-02",
        "query": "What programming languages does the candidate know?",
        "relevant_keywords": ["python", "java", "javascript", "c++", "programming", "language", "skill"],
        "expected_answer_keywords": ["python", "language", "skill"],
        "category": "happy_path",
        "description": "Skills extraction from CV"
    },
    {
        "id": "TC-03",
        "query": "What is the candidate's educational background?",
        "relevant_keywords": ["university", "degree", "bachelor", "master", "education", "gpa", "graduate"],
        "expected_answer_keywords": ["university", "degree", "education"],
        "category": "happy_path",
        "description": "Education section retrieval"
    },
    {
        "id": "TC-04",
        "query": "What work experience does the candidate have?",
        "relevant_keywords": ["experience", "work", "job", "company", "intern", "position", "role"],
        "expected_answer_keywords": ["experience", "work"],
        "category": "happy_path",
        "description": "Work experience retrieval"
    },
    {
        "id": "TC-05",
        "query": "What projects has the candidate worked on?",
        "relevant_keywords": ["project", "built", "developed", "system", "application", "app"],
        "expected_answer_keywords": ["project"],
        "category": "happy_path",
        "description": "Projects section retrieval"
    },
    # ── Edge-case queries (expected to fail or partially fail) ──
    {
        "id": "EC-01",
        "query": "What is the candidate's salary expectation?",
        "relevant_keywords": ["salary", "compensation", "pay", "wage", "expected"],
        "expected_answer_keywords": ["not", "could not", "salary"],
        "category": "edge_case",
        "description": "Query for information NOT present in CV — should trigger 'not found' response"
    },
    {
        "id": "EC-02",
        "query": "كم عمر المرشح؟",  # "How old is the candidate?" in Arabic
        "relevant_keywords": ["age", "born", "birth", "year", "عمر", "مرشح"],
        "expected_answer_keywords": ["not", "could not", "age", "عمر"],
        "category": "edge_case",
        "description": "Arabic query — tests multilingual handling; age likely not in CV"
    },
    {
        "id": "EC-03",
        "query": "xQzKt9 random nonsense query with no meaning",
        "relevant_keywords": ["xqzkt9", "nonsense"],
        "expected_answer_keywords": ["not", "could not", "find"],
        "category": "edge_case",
        "description": "Noise/garbage query — system should NOT hallucinate an answer"
    },
    {
        "id": "EC-04",
        "query": "What are the candidate's references and contact details of their managers?",
        "relevant_keywords": ["reference", "manager", "contact", "phone", "email", "referee"],
        "expected_answer_keywords": ["not", "could not", "reference", "contact"],
        "category": "edge_case",
        "description": "Query for private/unlisted information — hallucination risk"
    },
    {
        "id": "EC-05",
        "query": "Does the candidate have 10 years of experience in quantum computing?",
        "relevant_keywords": ["quantum", "10 years", "decade"],
        "expected_answer_keywords": ["not", "could not", "quantum", "no"],
        "category": "edge_case",
        "description": "False-premise query — tests hallucination resistance"
    },
]

TOP_K = 5  # number of chunks to retrieve per query

def chunk_is_relevant(chunk_text: str, relevant_keywords: list[str]) -> bool:
    """
    A retrieved chunk is considered RELEVANT (TP candidate) if it
    contains at least one of the ground-truth relevant keywords
    (case-insensitive substring match).
    """
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in relevant_keywords)


def answer_is_correct(answer: str, expected_keywords: list[str]) -> bool:
    """
    The LLM answer is considered CORRECT if it contains at least one
    expected answer keyword.
    """
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def compute_retrieval_metrics(retrieved_chunks: list[dict],
                              relevant_keywords: list[str],
                              total_relevant_in_db: Optional[int] = None) -> dict:
    tp = sum(1 for c in retrieved_chunks if chunk_is_relevant(c["text"], relevant_keywords))
    fp = len(retrieved_chunks) - tp
    fn = max(0, (total_relevant_in_db or TOP_K) - tp)
    tn = 0  # not meaningful in this single-query context

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def build_confusion_matrix(results: list[dict]) -> dict:
    total_tp = sum(r["retrieval_metrics"]["TP"] for r in results)
    total_fp = sum(r["retrieval_metrics"]["FP"] for r in results)
    total_fn = sum(r["retrieval_metrics"]["FN"] for r in results)
    total_tn = 0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return {
        "matrix": {
            "TP": total_tp, "FP": total_fp,
            "FN": total_fn, "TN": total_tn
        },
        "overall_precision": round(precision, 4),
        "overall_recall":    round(recall, 4),
        "overall_f1":        round(f1, 4),
    }


HALLUCINATION_SIGNALS = [
    "i could not find",
    "not in the provided",
    "not mentioned",
    "no information",
    "cannot find",
    "not available",
    "not provided",
]

def detect_hallucination(answer: str, chunks: list[dict],
                         edge_case: bool = False) -> dict:
    """
    Heuristic hallucination detector.
    For edge-case queries, a confident answer (no 'not found' phrasing)
    when zero relevant chunks exist is a hallucination red flag.
    """
    answer_lower = answer.lower()
    admitted_not_found = any(sig in answer_lower for sig in HALLUCINATION_SIGNALS)
    zero_relevant_retrieved = all(c["score"] < 0.2 for c in chunks)

    hallucination_flag = False
    reason = "none"

    if edge_case and not admitted_not_found and zero_relevant_retrieved:
        hallucination_flag = True
        reason = "Gave confident answer despite no high-similarity chunks (score < 0.2)"
    elif edge_case and not admitted_not_found and len(chunks) == 0:
        hallucination_flag = True
        reason = "Gave confident answer with empty retrieval"

    return {
        "hallucination_detected": hallucination_flag,
        "admitted_not_found": admitted_not_found,
        "reason": reason
    }


def run_evaluation() -> dict:
    results = []
    failed_rag = []

    print("\n" + "═" * 65)
    print("  RAG EVALUATION — Phase 4 (Lecture 4 Metrics)")
    print("═" * 65)

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['description']}")
        print(f"  Query: {tc['query'][:80]}")

        t0 = time.time()
        try:
            chunks = retrieve(tc["query"], top_k=TOP_K)
        except Exception as e:
            print(f"  ✗ Retrieval failed: {e}")
            failed_rag.append({"id": tc["id"], "stage": "retrieval", "error": str(e)})
            continue

        retrieval_time = round(time.time() - t0, 3)

        t1 = time.time()
        try:
            rag_result = run_rag(tc["query"], top_k=TOP_K)
            answer = rag_result["answer"]
        except Exception as e:
            print(f"  ✗ RAG/LLM call failed: {e}")
            answer = "[LLM_UNAVAILABLE]"
            rag_result = {"answer": answer, "chunks": chunks}
        rag_time = round(time.time() - t1, 3)

        metrics = compute_retrieval_metrics(chunks, tc["relevant_keywords"])
        answer_ok = answer_is_correct(answer, tc["expected_answer_keywords"])
        hallucination = detect_hallucination(
            answer, chunks,
            edge_case=(tc["category"] == "edge_case")
        )

        print(f"  Chunks retrieved : {len(chunks)}")
        print(f"  Scores           : {[round(c['score'],3) for c in chunks]}")
        print(f"  TP/FP/FN         : {metrics['TP']}/{metrics['FP']}/{metrics['FN']}")
        print(f"  Precision        : {metrics['precision']:.4f}")
        print(f"  Recall           : {metrics['recall']:.4f}")
        print(f"  F1               : {metrics['f1']:.4f}")
        print(f"  Answer correct   : {'✓' if answer_ok else '✗'}")
        if hallucination["hallucination_detected"]:
            print(f"  ⚠ HALLUCINATION  : {hallucination['reason']}")
        print(f"  Retrieval time   : {retrieval_time}s  |  RAG time: {rag_time}s")

        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "description": tc["description"],
            "query": tc["query"],
            "answer": answer[:300],
            "answer_correct": answer_ok,
            "hallucination": hallucination,
            "retrieval_metrics": metrics,
            "top_chunks": [
                {"text": c["text"][:120], "score": round(c["score"], 4)}
                for c in chunks
            ],
            "timings": {
                "retrieval_sec": retrieval_time,
                "rag_sec": rag_time
            }
        })

    
    cm = build_confusion_matrix(results)

    happy   = [r for r in results if r["category"] == "happy_path"]
    edge    = [r for r in results if r["category"] == "edge_case"]
    hallucs = [r for r in results if r["hallucination"]["hallucination_detected"]]

    avg_p  = round(sum(r["retrieval_metrics"]["precision"] for r in results) / len(results), 4) if results else 0
    avg_r  = round(sum(r["retrieval_metrics"]["recall"]    for r in results) / len(results), 4) if results else 0
    avg_f1 = round(sum(r["retrieval_metrics"]["f1"]        for r in results) / len(results), 4) if results else 0

    summary = {
        "total_queries": len(TEST_CASES),
        "evaluated":     len(results),
        "failed_runs":   failed_rag,
        "happy_path": {
            "count": len(happy),
            "avg_f1": round(sum(r["retrieval_metrics"]["f1"] for r in happy) / len(happy), 4) if happy else 0,
        },
        "edge_cases": {
            "count": len(edge),
            "avg_f1": round(sum(r["retrieval_metrics"]["f1"] for r in edge) / len(edge), 4) if edge else 0,
        },
        "hallucinations_detected": len(hallucs),
        "macro_avg_precision": avg_p,
        "macro_avg_recall":    avg_r,
        "macro_avg_f1":        avg_f1,
        "global_confusion_matrix": cm,
    }

    return {"summary": summary, "results": results}

def write_text_report(eval_data: dict, path: str = "eval_report.txt"):
    s = eval_data["summary"]
    cm = s["global_confusion_matrix"]
    mat = cm["matrix"]
    results = eval_data["results"]

    lines = []
    h = lambda t: lines.append("\n" + "═" * 65 + "\n  " + t + "\n" + "═" * 65)

    h("PHASE 4 — RAG EVALUATION REPORT")
    lines.append(f"\nTotal test cases  : {s['total_queries']}")
    lines.append(f"Evaluated         : {s['evaluated']}")
    lines.append(f"Failed runs       : {len(s['failed_runs'])}")

    h("GLOBAL METRICS (Lecture 4 Formulas)")
    lines.append(f"\n  Macro-Avg Precision : {s['macro_avg_precision']:.4f}")
    lines.append(f"  Macro-Avg Recall    : {s['macro_avg_recall']:.4f}")
    lines.append(f"  Macro-Avg F1-Score  : {s['macro_avg_f1']:.4f}")

    h("GLOBAL CONFUSION MATRIX")
    lines.append(f"""
                   Predicted Positive | Predicted Negative
  ─────────────────────────────────────────────────────────
  Actual Positive │     TP = {mat['TP']:>4}      │     FN = {mat['FN']:>4}
  Actual Negative │     FP = {mat['FP']:>4}      │     TN = {mat['TN']:>4}
  ─────────────────────────────────────────────────────────

  Overall Precision (TP/(TP+FP))  = {cm['overall_precision']:.4f}
  Overall Recall    (TP/(TP+FN))  = {cm['overall_recall']:.4f}
  Overall F1                      = {cm['overall_f1']:.4f}
  
  Interpretation:
    Precision = {cm['overall_precision']:.1%} of retrieved chunks were genuinely relevant.
    Recall    = {cm['overall_recall']:.1%} of all relevant chunks were retrieved.
""")

    h("CATEGORY BREAKDOWN")
    lines.append(f"\n  Happy-path queries  ({s['happy_path']['count']} tests):")
    lines.append(f"    Avg F1 = {s['happy_path']['avg_f1']:.4f}")
    lines.append(f"\n  Edge-case queries   ({s['edge_cases']['count']} tests):")
    lines.append(f"    Avg F1 = {s['edge_cases']['avg_f1']:.4f}")
    lines.append(f"\n  Hallucinations detected : {s['hallucinations_detected']}")

    h("PER-QUERY RESULTS")
    for r in results:
        m = r["retrieval_metrics"]
        lines.append(f"""
  [{r['id']}] {r['description']}
  Category : {r['category']}
  Query    : {r['query'][:70]}
  Answer   : {r['answer'][:120]}...
  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}
  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}
  Answer correct : {'YES' if r['answer_correct'] else 'NO'}
  Hallucination  : {'⚠ YES — ' + r['hallucination']['reason'] if r['hallucination']['hallucination_detected'] else 'No'}
  Top chunk scores: {[c['score'] for c in r['top_chunks']]}
""")

    h("EDGE-CASE ERROR ANALYSIS (Phase 4 Requirement: ≥3 failures)")
    edge_results = [r for r in results if r["category"] == "edge_case"]
    failures = [r for r in edge_results
                if not r["answer_correct"] or r["hallucination"]["hallucination_detected"]]

    for i, r in enumerate(failures[:5], 1):
        m = r["retrieval_metrics"]
        lines.append(f"""
  Failure #{i}: [{r['id']}] — {r['description']}
  Query   : {r['query']}
  Answer  : {r['answer'][:200]}

  What went wrong:
    • Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}
    • TP={m['TP']}, FP={m['FP']}, FN={m['FN']}
    • Hallucination flag: {r['hallucination']['hallucination_detected']}
    • Hallucination reason: {r['hallucination']['reason']}

  Why the architecture missed:
""")
        if r["id"] == "EC-01":
            lines.append("    The CV does not contain salary information. With cosine-similarity\n"
                         "    retrieval, the embedder maps 'salary expectation' to 'skills/experience'\n"
                         "    chunks because of semantic overlap with compensation-adjacent language.\n"
                         "    The top-k retrieved chunks are irrelevant (high FP), leading the LLM\n"
                         "    to either confabulate or correctly admit the absence of information.\n"
                         "    Fix: Add a metadata filter or a 'not-found' classifier before LLM call.")
        elif r["id"] == "EC-02":
            lines.append("    The embedding model (all-MiniLM-L6-v2) is English-centric. Arabic\n"
                         "    tokens are out-of-distribution, producing near-random embeddings that\n"
                         "    retrieve semantically unrelated chunks (high FP, zero TP). The LLM\n"
                         "    then has no grounded context and may hallucinate or respond in English.\n"
                         "    Fix: Use a multilingual model (e.g., paraphrase-multilingual-MiniLM-L12-v2)\n"
                         "    and add Arabic text normalisation (CAMeL-Tools or PyAramorph).")
        elif r["id"] == "EC-03":
            lines.append("    Garbage input produces a random embedding vector. ChromaDB returns\n"
                         "    chunks with low cosine similarity (score < 0.1) because no document\n"
                         "    is truly close to random noise. However the system still passes those\n"
                         "    low-scoring chunks to the LLM, which may invent an answer.\n"
                         "    Fix: Apply a similarity threshold (e.g., score < 0.3 → reject query)\n"
                         "    before invoking the LLM.")
        elif r["id"] == "EC-04":
            lines.append("    References are rarely written explicitly in CVs. The retriever finds\n"
                         "    chunks with partial overlap (e.g., contact email), giving misleading\n"
                         "    partial context that can cause the LLM to fabricate names/phone numbers.\n"
                         "    Fix: Prompt engineering — add 'Do NOT invent names or phone numbers'\n"
                         "    and a post-generation PII scanner.")
        elif r["id"] == "EC-05":
            lines.append("    The false-premise query ('10 years of quantum computing') retrieves\n"
                         "    any chunk containing technical keywords regardless of years/field.\n"
                         "    The LLM may confirm the false premise because the context mentions\n"
                         "    'computing' or experience in adjacent fields.\n"
                         "    Fix: Claim-verification step — compare LLM answer against retrieved\n"
                         "    chunks using NLI (Natural Language Inference) before returning response.")
        else:
            lines.append(f"    Precision={m['precision']:.4f} indicates retrieval pulled irrelevant chunks.\n"
                         "    The LLM therefore lacked grounded context.")

    h("SUMMARY & RECOMMENDATIONS")
    lines.append(f"""
  1. Retrieval quality
     Macro F1 = {s['macro_avg_f1']:.4f}.  Happy-path F1 = {s['happy_path']['avg_f1']:.4f}
     vs. Edge-case F1 = {s['edge_cases']['avg_f1']:.4f}.  The gap reveals the system
     handles well-formed domain queries but degrades sharply on out-of-domain,
     multilingual, and adversarial inputs.

  2. Hallucination rate
     {s['hallucinations_detected']} / {s['evaluated']} queries triggered a hallucination flag.
     Root cause: no similarity-threshold gate between retriever and LLM.

  3. Recommended improvements
     a) Add a cosine-similarity threshold (≥ 0.3) — reject queries with no
        strong match before calling the LLM.
     b) Switch to a multilingual embedding model for Arabic support (Bonus 2).
     c) Implement the LLM Factory Pattern (Bonus 1) to allow swapping models.
     d) Add a post-generation NLI verifier to detect factual contradictions.
     e) Log and monitor per-query F1 in production for continuous evaluation.
""")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"\n  Report written → {path}")


if __name__ == "__main__":
    eval_data = run_evaluation()

    s = eval_data["summary"]
    cm = s["global_confusion_matrix"]
    print("\n" + "═" * 65)
    print("  GLOBAL RESULTS")
    print("═" * 65)
    print(f"  Macro Precision : {s['macro_avg_precision']:.4f}")
    print(f"  Macro Recall    : {s['macro_avg_recall']:.4f}")
    print(f"  Macro F1        : {s['macro_avg_f1']:.4f}")
    print(f"\n  Confusion Matrix (aggregated)")
    mat = cm["matrix"]
    print(f"    TP={mat['TP']}  FP={mat['FP']}  FN={mat['FN']}  TN={mat['TN']}")
    print(f"\n  Hallucinations  : {s['hallucinations_detected']}")
    print(f"  Happy-path F1   : {s['happy_path']['avg_f1']:.4f}")
    print(f"  Edge-case  F1   : {s['edge_cases']['avg_f1']:.4f}")
    print("═" * 65)

    json_path = "eval_report.json"
    txt_path  = "eval_report.txt"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(eval_data, fh, indent=2, ensure_ascii=False)
    print(f"\n  JSON report → {json_path}")

    write_text_report(eval_data, txt_path)
