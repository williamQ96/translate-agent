"""
RAG store: dense + lexical hybrid retrieval with optional rerank/self-correction/web fallback.

Backward compatible with previous interface:
- index_document(text)
- retrieve_context(query_text, k=3) -> str
- clear()
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import chromadb

from src.utils.config_loader import load_config


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _trim(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    return text if len(text) <= max_chars else text[:max_chars]


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


@dataclass
class _Candidate:
    doc_id: str
    text: str
    score: float
    source: str  # dense|lexical|web
    meta: dict[str, Any]


class RAGStore:
    def __init__(self):
        config = load_config()
        rag_config = config.get("rag", {})
        retrieval_cfg = config.get("retrieval", {})
        rerank_cfg = config.get("rerank", {})
        self_correction_cfg = config.get("self_correction", {})
        web_cfg = config.get("web_search", {})

        persist_dir = rag_config.get("persist_directory", "data/chroma_db")
        collection_name = rag_config.get("collection_name", "document_context")

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Retrieval settings
        self.mode = str(retrieval_cfg.get("mode", "dense")).lower()  # dense|hybrid
        self.dense_k = int(retrieval_cfg.get("dense_k", 6))
        self.lexical_k = int(retrieval_cfg.get("lexical_k", 6))
        self.rrf_k = int(retrieval_cfg.get("rrf_k", 60))
        self.context_max_chars = int(retrieval_cfg.get("context_max_chars", 2200))
        self.min_confidence = float(retrieval_cfg.get("min_confidence", 0.25))
        self.log_trace = bool(retrieval_cfg.get("log_trace", True))

        # Rerank settings
        self.rerank_enabled = bool(rerank_cfg.get("enabled", False))
        self.rerank_provider = str(rerank_cfg.get("provider", "heuristic")).lower()
        self.rerank_top_n = int(rerank_cfg.get("top_n", 10))
        self.rerank_keep_k = int(rerank_cfg.get("keep_k", 3))

        # Self-correction settings
        self.self_correction_enabled = bool(self_correction_cfg.get("enabled", False))
        self.self_correction_retries = int(self_correction_cfg.get("max_retrieval_retries", 1))

        # Web search settings
        self.web_enabled = bool(web_cfg.get("enabled", False))
        self.web_provider = str(web_cfg.get("provider", "off")).lower()  # off|wikipedia|tavily|bing
        self.web_timeout = int(web_cfg.get("timeout", 8))
        self.web_max_results = int(web_cfg.get("max_results", 2))
        self.web_cache_path = str(web_cfg.get("cache_path", "data/cache/web_context.json"))
        self.web_allowed_domains = [str(x).lower() for x in web_cfg.get("allowed_domains", [])]
        self.web_fallback_min_confidence = float(web_cfg.get("fallback_min_confidence", 0.22))

        # Lexical index
        self._docs_by_id: dict[str, str] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_tf: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._avg_len: float = 1.0

        # Optional local cross encoder
        self._cross_encoder = None

        # Diagnostics
        self.last_trace: dict[str, Any] = {}

    def index_document(self, text: str, min_paragraph_length: int = 50):
        """Split text into paragraphs and index into ChromaDB."""
        if not text.strip():
            print("    WARN RAG: Empty document, nothing to index")
            return

        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_paragraph_length]
        if not paragraphs:
            print("    WARN RAG: No paragraphs found to index")
            return

        # Rebuild dense collection each run to keep retrieval deterministic with current source.
        existing = self.collection.count()
        if existing:
            try:
                existing_ids = self.collection.get(include=[])["ids"]
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
            except Exception:
                pass

        ids = [f"p_{i}" for i in range(len(paragraphs))]
        batch_size = 100
        for i in range(0, len(paragraphs), batch_size):
            batch_docs = paragraphs[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.collection.add(documents=batch_docs, ids=batch_ids)

        self._build_lexical_index(ids, paragraphs)
        print(f"    RAG: Indexed {len(paragraphs)} paragraphs into ChromaDB (mode={self.mode})")

    def _build_lexical_index(self, ids: list[str], docs: list[str]) -> None:
        self._docs_by_id = {}
        self._doc_tokens = {}
        self._doc_tf = {}
        self._doc_len = {}
        self._df = {}

        for doc_id, text in zip(ids, docs):
            tokens = _tokenize(text)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._docs_by_id[doc_id] = text
            self._doc_tokens[doc_id] = tokens
            self._doc_tf[doc_id] = tf
            self._doc_len[doc_id] = max(len(tokens), 1)
            for term in tf.keys():
                self._df[term] = self._df.get(term, 0) + 1

        self._avg_len = (
            sum(self._doc_len.values()) / max(len(self._doc_len), 1)
            if self._doc_len
            else 1.0
        )

    def _retrieve_dense(self, query_text: str, k: int) -> list[_Candidate]:
        if not query_text.strip():
            return []
        try:
            result = self.collection.query(query_texts=[query_text], n_results=max(k, 1))
        except Exception:
            return []

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        dists = (result.get("distances") or [[]])[0] if result.get("distances") else [None] * len(ids)

        out: list[_Candidate] = []
        for i, doc_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            dist = dists[i] if i < len(dists) else None
            score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.2
            out.append(_Candidate(doc_id=doc_id, text=text, score=score, source="dense", meta={"distance": dist}))
        return out

    def _bm25(self, tf: int, df: int, doc_len: int, n_docs: int, k1: float = 1.5, b: float = 0.75) -> float:
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1.0 - b + b * (doc_len / max(self._avg_len, 1e-9)))
        return idf * (tf * (k1 + 1.0) / max(denom, 1e-9))

    def _retrieve_lexical(self, query_text: str, k: int) -> list[_Candidate]:
        q_terms = _tokenize(query_text)
        if not q_terms or not self._doc_tf:
            return []

        n_docs = len(self._doc_tf)
        scored: list[tuple[str, float]] = []
        for doc_id, tf_map in self._doc_tf.items():
            score = 0.0
            for term in q_terms:
                tf = tf_map.get(term, 0)
                if tf <= 0:
                    continue
                df = self._df.get(term, 1)
                score += self._bm25(tf=tf, df=df, doc_len=self._doc_len.get(doc_id, 1), n_docs=n_docs)
            if score > 0:
                scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        out: list[_Candidate] = []
        for doc_id, score in scored[: max(k, 1)]:
            out.append(
                _Candidate(
                    doc_id=doc_id,
                    text=self._docs_by_id.get(doc_id, ""),
                    score=score,
                    source="lexical",
                    meta={},
                )
            )
        return out

    def _rrf_fuse(self, dense: list[_Candidate], lexical: list[_Candidate]) -> list[_Candidate]:
        rank_maps: dict[str, dict[str, int]] = {}
        docs_by_id: dict[str, _Candidate] = {}

        for idx, cand in enumerate(dense):
            docs_by_id[cand.doc_id] = cand
            rank_maps.setdefault(cand.doc_id, {})["dense"] = idx + 1

        for idx, cand in enumerate(lexical):
            if cand.doc_id not in docs_by_id:
                docs_by_id[cand.doc_id] = cand
            rank_maps.setdefault(cand.doc_id, {})["lexical"] = idx + 1

        fused: list[_Candidate] = []
        for doc_id, ranks in rank_maps.items():
            score = 0.0
            for _, rank in ranks.items():
                score += 1.0 / (self.rrf_k + rank)
            base = docs_by_id[doc_id]
            fused.append(
                _Candidate(
                    doc_id=doc_id,
                    text=base.text,
                    score=score,
                    source="hybrid",
                    meta={"ranks": ranks},
                )
            )
        fused.sort(key=lambda c: c.score, reverse=True)
        return fused

    def _heuristic_rerank(self, query: str, candidates: list[_Candidate], keep_k: int) -> list[_Candidate]:
        q_terms = set(_tokenize(query))
        if not q_terms:
            return candidates[:keep_k]

        rescored: list[_Candidate] = []
        for cand in candidates:
            d_terms = set(_tokenize(cand.text))
            overlap = len(q_terms & d_terms) / max(len(q_terms), 1)
            # blend previous score + lexical overlap
            score = 0.7 * cand.score + 0.3 * overlap
            rescored.append(_Candidate(cand.doc_id, cand.text, score, cand.source, cand.meta))
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:keep_k]

    def _cross_encoder_rerank(self, query: str, candidates: list[_Candidate], keep_k: int) -> list[_Candidate]:
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder

                model_name = load_config().get("rerank", {}).get(
                    "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self._cross_encoder = CrossEncoder(model_name)
            except Exception:
                return self._heuristic_rerank(query, candidates, keep_k)

        pairs = [[query, cand.text] for cand in candidates]
        try:
            scores = self._cross_encoder.predict(pairs)
        except Exception:
            return self._heuristic_rerank(query, candidates, keep_k)

        rescored: list[_Candidate] = []
        for cand, s in zip(candidates, scores):
            rescored.append(_Candidate(cand.doc_id, cand.text, float(s), cand.source, cand.meta))
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:keep_k]

    def _rerank(self, query: str, candidates: list[_Candidate]) -> list[_Candidate]:
        if not self.rerank_enabled or not candidates:
            return candidates[: self.rerank_keep_k]
        pool = candidates[: max(self.rerank_top_n, self.rerank_keep_k)]
        if self.rerank_provider in {"cross_encoder", "local_cross_encoder"}:
            return self._cross_encoder_rerank(query, pool, self.rerank_keep_k)
        return self._heuristic_rerank(query, pool, self.rerank_keep_k)

    def _confidence(self, candidates: list[_Candidate]) -> float:
        if not candidates:
            return 0.0
        return max(0.0, min(1.0, float(candidates[0].score)))

    def _rewrite_query(self, query_text: str) -> str:
        # Lightweight self-correction query rewrite: keep entities/nouns-ish tokens.
        lines = [ln.strip() for ln in query_text.splitlines() if ln.strip()]
        base = " ".join(lines)
        terms = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", base)
        if not terms:
            terms = _tokenize(base)[:30]
        dedup: list[str] = []
        seen = set()
        for t in terms:
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(t)
        if not dedup:
            return query_text
        return " ".join(dedup[:24])

    def _prepare_web_query(self, query_text: str) -> str:
        """
        Prepare shorter, less noisy query for web search.
        Long raw chunk text often yields poor search quality.
        """
        rewritten = self._rewrite_query(query_text)
        tokens = _tokenize(rewritten)
        if not tokens:
            tokens = _tokenize(query_text)
        if not tokens:
            return query_text.strip()

        # keep early unique terms; enough for entity/topic matching
        seen = set()
        keep: list[str] = []
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            keep.append(t)
            if len(keep) >= 12:
                break
        return " ".join(keep) if keep else rewritten.strip()

    def _load_web_cache(self) -> dict[str, Any]:
        path = self.web_cache_path
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def _save_web_cache(self, cache: dict[str, Any]) -> None:
        path = self.web_cache_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(cache, file, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _http_get_json(self, url: str, headers: dict[str, str] | None = None) -> Any:
        merged_headers = {
            "User-Agent": "translate-agent/0.32 (+local-rag-web-fallback)",
            "Accept": "application/json",
        }
        if headers:
            merged_headers.update(headers)
        req = urllib.request.Request(url, headers=merged_headers)
        with urllib.request.urlopen(req, timeout=self.web_timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return json.loads(body)

    def _web_search_wikipedia(self, query: str) -> list[_Candidate]:
        q = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit={self.web_max_results}"
        data = self._http_get_json(url)
        results = data.get("query", {}).get("search", []) if isinstance(data, dict) else []
        out: list[_Candidate] = []
        for idx, item in enumerate(results):
            title = str(item.get("title", "")).strip()
            snippet = re.sub(r"<[^>]+>", "", str(item.get("snippet", ""))).strip()
            if not title and not snippet:
                continue
            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}" if title else ""
            text = f"{title}: {snippet}".strip(": ")
            out.append(
                _Candidate(
                    doc_id=f"web_wiki_{idx}",
                    text=text,
                    score=max(0.05, 0.1 - idx * 0.01),
                    source="web",
                    meta={"url": page_url or "https://en.wikipedia.org", "provider": "wikipedia"},
                )
            )
        return out

    def _web_search_tavily(self, query: str) -> list[_Candidate]:
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            return []
        payload = json.dumps({"query": query, "max_results": self.web_max_results}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.web_timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(body)
        out: list[_Candidate] = []
        for idx, item in enumerate(data.get("results", [])[: self.web_max_results]):
            url = str(item.get("url", "")).strip()
            if not self._allow_domain(url):
                continue
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            text = f"{title}: {content}".strip(": ")
            out.append(
                _Candidate(
                    doc_id=f"web_tavily_{idx}",
                    text=text,
                    score=max(0.05, 0.1 - idx * 0.01),
                    source="web",
                    meta={"url": url, "provider": "tavily"},
                )
            )
        return out

    def _web_search_bing(self, query: str) -> list[_Candidate]:
        api_key = os.environ.get("BING_SEARCH_API_KEY", "").strip()
        if not api_key:
            return []
        endpoint = os.environ.get("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
        params = urllib.parse.urlencode({"q": query, "count": self.web_max_results, "textDecorations": False})
        url = f"{endpoint}?{params}"
        data = self._http_get_json(url, headers={"Ocp-Apim-Subscription-Key": api_key})
        values = data.get("webPages", {}).get("value", []) if isinstance(data, dict) else []
        out: list[_Candidate] = []
        for idx, item in enumerate(values[: self.web_max_results]):
            link = str(item.get("url", "")).strip()
            if not self._allow_domain(link):
                continue
            name = str(item.get("name", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            text = f"{name}: {snippet}".strip(": ")
            out.append(
                _Candidate(
                    doc_id=f"web_bing_{idx}",
                    text=text,
                    score=max(0.05, 0.1 - idx * 0.01),
                    source="web",
                    meta={"url": link, "provider": "bing"},
                )
            )
        return out

    def _allow_domain(self, url: str) -> bool:
        if not self.web_allowed_domains:
            return True
        try:
            domain = urllib.parse.urlparse(url).netloc.lower()
        except Exception:
            return False
        return any(domain.endswith(allowed) for allowed in self.web_allowed_domains)

    def _web_search(self, query: str) -> list[_Candidate]:
        if not self.web_enabled or self.web_provider in {"off", "none", ""}:
            return []

        prepared_query = self._prepare_web_query(query)
        key = f"{self.web_provider}::{prepared_query.strip().lower()}"
        cache = self._load_web_cache()
        if key in cache:
            items = cache.get(key, [])
            out: list[_Candidate] = []
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    out.append(
                        _Candidate(
                            doc_id=item.get("doc_id", f"web_cached_{idx}"),
                            text=str(item.get("text", "")),
                            score=float(item.get("score", 0.05)),
                            source="web",
                            meta=item.get("meta", {}),
                        )
                    )
            return out

        try:
            if self.web_provider == "wikipedia":
                found = self._web_search_wikipedia(prepared_query)
            elif self.web_provider == "tavily":
                found = self._web_search_tavily(prepared_query)
            elif self.web_provider == "bing":
                found = self._web_search_bing(prepared_query)
            else:
                found = []
        except Exception:
            found = []

        cache[key] = [
            {
                "doc_id": c.doc_id,
                "text": c.text,
                "score": c.score,
                "meta": c.meta,
            }
            for c in found
        ]
        self._save_web_cache(cache)
        return found

    def _retrieve_once(self, query_text: str, k: int) -> tuple[list[_Candidate], float]:
        dense = self._retrieve_dense(query_text, max(k, self.dense_k))
        if self.mode != "hybrid":
            ranked = dense
        else:
            lexical = self._retrieve_lexical(query_text, self.lexical_k)
            ranked = self._rrf_fuse(dense, lexical)

        reranked = self._rerank(query_text, ranked)
        reranked = reranked[: max(k, 1)]
        conf = self._confidence(reranked)
        return reranked, conf

    def retrieve_context(self, query_text: str, k: int = 3) -> str:
        """Retrieve top-k relevant paragraphs for a query chunk."""
        if not query_text.strip():
            return ""

        candidates, conf = self._retrieve_once(query_text, k)
        used_query = query_text
        corrected = False

        if self.self_correction_enabled and conf < self.min_confidence:
            retries = max(self.self_correction_retries, 0)
            for _ in range(retries):
                rewritten_query = self._rewrite_query(used_query)
                if rewritten_query.strip() == used_query.strip():
                    break
                nxt, nxt_conf = self._retrieve_once(rewritten_query, k)
                if nxt_conf >= conf:
                    candidates, conf = nxt, nxt_conf
                    used_query = rewritten_query
                    corrected = True
                if conf >= self.min_confidence:
                    break

        web_candidates: list[_Candidate] = []
        if conf < self.web_fallback_min_confidence:
            web_candidates = self._web_search(used_query)[: self.web_max_results]

        context_parts: list[str] = []
        for i, cand in enumerate(candidates, start=1):
            text = _trim(cand.text.strip(), self.context_max_chars)
            if not text:
                continue
            context_parts.append(f"[RAG#{i} {cand.source} score={cand.score:.3f}] {text}")

        for i, cand in enumerate(web_candidates, start=1):
            text = _trim(cand.text.strip(), max(200, self.context_max_chars // 2))
            if not text:
                continue
            cite = cand.meta.get("url", "")
            context_parts.append(f"[WEB#{i} {cand.meta.get('provider', 'web')}] {text}\n来源: {cite}")

        self.last_trace = {
            "timestamp": _now_iso(),
            "mode": self.mode,
            "query_rewritten": corrected,
            "confidence": conf,
            "candidate_count": len(candidates),
            "web_count": len(web_candidates),
        }
        if self.log_trace:
            print(
                "    RAG TRACE: "
                f"confidence={conf:.3f} "
                f"query_rewritten={corrected} "
                f"web_count={len(web_candidates)} "
                f"mode={self.mode}"
            )
        return "\n\n".join(context_parts)

    def clear(self):
        """Clear ChromaDB collection and in-memory lexical index."""
        try:
            ids = self.collection.get(include=[])["ids"]
            if ids:
                self.collection.delete(ids=ids)
        except Exception:
            pass
        self._docs_by_id.clear()
        self._doc_tokens.clear()
        self._doc_tf.clear()
        self._doc_len.clear()
        self._df.clear()
