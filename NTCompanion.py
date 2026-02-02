"""
NTCompanion Pro - High-Performance Scraper for NTTuner
======================================================
Uses the proven urllib scraping approach from a.py + ThreadPoolExecutor for concurrency
- Working HTTP stack with reliable content extraction
- Concurrent scraping (10-50x faster than single-threaded)
- Bloom filter deduplication
- Enhanced proxy management
- Priority crawl queue
- Simple, reliable link extraction and crawling
"""

import dearpygui.dearpygui as dpg
import json
import os
import time
import threading
import urllib.request
import urllib.parse
import ssl
import re
import random
import tkinter as tk
import http.cookiejar
from tkinter import filedialog
from urllib.error import URLError, HTTPError
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, Tuple
import heapq
import queue

# Windows sound (optional)
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Optional: Bloom filter for memory-efficient dedup
try:
    import mmh3
    HAS_MMH3 = True
except ImportError:
    HAS_MMH3 = False
    print("Note: Install mmh3 for memory-efficient deduplication: pip install mmh3")

# ================================================================
# CONSTANTS & CONFIG
# ================================================================
CONFIG_FILE = "nttuner_config_pro.json"
INI_FILE = "ntcompanion_pro.ini"
VERSION = "build.2026.05.Pro-ThreadPool"
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Concurrency defaults
DEFAULT_WORKERS = 10
MAX_WORKERS = 50

# Expanded User Agent Pool - Chrome dominates, match it exactly
USER_AGENTS = [
    # Chrome on Windows (65%+ of traffic)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",

    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",

    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",

    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",

    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",

    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",

    # Mobile
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36",
]

# System Prompt Presets
SYSTEM_PROMPTS = {
    "Blank (No System Context)": "",
    "Helpful Assistant": "You are a helpful and honest assistant.",
    "Data Summarizer": "Summarize the following content concisely into a JSON object.",
    "Code Expert": "You are an expert programmer. Analyze the code snippets found in the text.",
    "Creative Writer": "Rewrite the following text in a more engaging, narrative style.",
    "NTTuner Default": "You are an AI assistant trained for reasoning and clarity."
}

# Chat Templates
MODEL_TEMPLATES = {
    "Meta Llama-3.1 / 3.2 / 3.3 Instruct": {
        "begin": "<|begin_of_text|>",
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
    },
    "Mistral Nemo / Large Instruct": {
        "begin": "<|begin_of_text|>",
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
    },
    "Qwen2.5 Instruct": {
        "begin": "",
        "system": "<|im_start|>system\n{system}<|im_end|>\n",
        "user": "<|im_start|>user\n{user}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{assistant}<|im_end|>\n",
    },
    "Phi-4 Instruct": {
        "begin": "",
        "system": "<|system|>\n{system}<|end|>\n",
        "user": "<|user|>\n{user}<|end|>\n",
        "assistant": "<|assistant|>\n{assistant}<|end|>\n",
    },
    "Gemma-2 Instruct": {
        "begin": "<bos>",
        "system": "<start_of_turn>system\n{system}<end_of_turn>\n",
        "user": "<start_of_turn>user\n{user}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n{assistant}<end_of_turn>\n",
    },
}

DEFAULT_TEMPLATE_KEY = "Meta Llama-3.1 / 3.2 / 3.3 Instruct"
DEFAULT_SYSTEM = "You are a helpful and honest assistant."

# Expanded Proxy Sources
PROXY_SOURCES = {
    "ProxyScrape HTTPS": "https://api.proxyscrape.com/v2/?request=getproxies&protocol=https&timeout=10000&country=all&ssl=yes&anonymity=all",
    "ProxyScrape HTTP": "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
    "ProxyScrape SOCKS4": "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks4&timeout=10000&country=all",
    "ProxyScrape SOCKS5": "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=all",
    "Geonode Free": "https://proxylist.geonode.com/api/proxy-list?limit=500&page=1&sort_by=lastChecked&sort_type=desc",
    "TheSpeedX HTTP": "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "TheSpeedX SOCKS4": "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks4.txt",
    "TheSpeedX SOCKS5": "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
    "Monosans HTTP": "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "Monosans SOCKS4": "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks4.txt",
    "Monosans SOCKS5": "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt",
    "ShiftyTR HTTP": "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt",
    "ShiftyTR HTTPS": "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/https.txt",
    "Clarketm HTTP": "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "JetKai HTTP": "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
    "JetKai HTTPS": "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-https.txt",
    "Spys.me HTTP": "https://spys.me/proxy.txt",
    "Free-Proxy-List.net": "https://free-proxy-list.net/",
    "Proxy-List-Download HTTP": "https://www.proxy-list.download/api/v1/get?type=http",
    "Proxy-List-Download HTTPS": "https://www.proxy-list.download/api/v1/get?type=https",
}


# ================================================================
# BLOOM FILTER (Memory-efficient deduplication)
# ================================================================
class BloomFilter:
    """Memory-efficient probabilistic deduplication"""

    def __init__(self, expected_items: int = 100000, false_positive_rate: float = 0.01):
        if HAS_MMH3:
            import math
            self.size = int(-expected_items * math.log(false_positive_rate) / (math.log(2) ** 2))
            self.hash_count = max(1, int((self.size / expected_items) * math.log(2)))
            self.bit_array = bytearray((self.size + 7) // 8)
        else:
            # Fallback to regular set
            self.fallback_set = set()
        self.count = 0

    def add(self, item: str):
        if HAS_MMH3:
            for i in range(self.hash_count):
                pos = mmh3.hash(item, i) % self.size
                self.bit_array[pos // 8] |= (1 << (pos % 8))
        else:
            self.fallback_set.add(item)
        self.count += 1

    def __contains__(self, item: str) -> bool:
        if HAS_MMH3:
            for i in range(self.hash_count):
                pos = mmh3.hash(item, i) % self.size
                if not (self.bit_array[pos // 8] & (1 << (pos % 8))):
                    return False
            return True
        else:
            return item in self.fallback_set

    def __len__(self) -> int:
        return self.count


# ================================================================
# PROXY MANAGER
# ================================================================
@dataclass
class ProxyStats:
    address: str
    success_count: int = 0
    fail_count: int = 0
    total_time: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)
    quarantine_until: Optional[datetime] = None

    @property
    def score(self) -> float:
        if self.quarantine_until and datetime.now() < self.quarantine_until:
            return -1000
        total = self.success_count + self.fail_count
        if total == 0:
            return 50
        success_rate = self.success_count / total
        return success_rate * 100


class ProxyManager:
    def __init__(self):
        self.proxies: Dict[str, ProxyStats] = {}
        self.lock = threading.Lock()

    def add_proxies(self, proxy_list: List[str]):
        with self.lock:
            for proxy in proxy_list:
                proxy = proxy.strip()
                if ':' in proxy and proxy not in self.proxies:
                    self.proxies[proxy] = ProxyStats(address=proxy)

    def get_best_proxy(self) -> Optional[str]:
        with self.lock:
            now = datetime.now()
            candidates = [(stats.score, addr) for addr, stats in self.proxies.items()
                          if not stats.quarantine_until or now >= stats.quarantine_until]
            if not candidates:
                return None
            candidates.sort(reverse=True, key=lambda x: x[0])
            top_count = max(1, len(candidates) // 5)
            return random.choice(candidates[:top_count])[1]

    def report_success(self, proxy: str, response_time: float):
        with self.lock:
            if proxy in self.proxies:
                self.proxies[proxy].success_count += 1
                self.proxies[proxy].total_time += response_time
                self.proxies[proxy].quarantine_until = None

    def report_failure(self, proxy: str):
        with self.lock:
            if proxy in self.proxies:
                stats = self.proxies[proxy]
                stats.fail_count += 1
                total = stats.success_count + stats.fail_count
                if total >= 3:
                    fail_rate = stats.fail_count / total
                    if fail_rate > 0.7:
                        stats.quarantine_until = datetime.now() + timedelta(minutes=30)
                    elif fail_rate > 0.5:
                        stats.quarantine_until = datetime.now() + timedelta(minutes=10)

    def get_stats(self) -> Dict:
        with self.lock:
            now = datetime.now()
            total = len(self.proxies)
            active = sum(1 for s in self.proxies.values()
                         if not s.quarantine_until or now >= s.quarantine_until)
            return {"total": total, "active": active, "quarantined": total - active}

    def clear_quarantine(self):
        with self.lock:
            for stats in self.proxies.values():
                stats.quarantine_until = None

    def clear_all(self):
        with self.lock:
            self.proxies.clear()


# ================================================================
# CRAWL QUEUE WITH PRIORITY
# ================================================================
class CrawlQueue:
    """Priority-based crawl queue with domain tracking"""

    def __init__(self):
        self.queue = []  # heap: (priority, counter, url, depth, source_domain)
        self.counter = 0
        self.seen = BloomFilter(expected_items=500000)
        self.domain_counts = defaultdict(int)
        self.lock = threading.Lock()
        self.seed_domains = set()

    def set_seed_domains(self, urls: List[str]):
        """Track which domains were in the initial seed"""
        for url in urls:
            domain = get_domain(url)
            if domain:
                self.seed_domains.add(domain)

    def _calculate_priority(self, url: str, depth: int, prioritize_content: bool) -> int:
        """Lower number = higher priority"""
        priority = depth * 100

        if prioritize_content:
            # Boost content-like URLs
            url_lower = url.lower()
            if any(x in url_lower for x in ['/article', '/post', '/blog', '/news', '/story', '/content']):
                priority -= 30
            if any(x in url_lower for x in ['/wiki/', '/docs/', '/guide/', '/tutorial/']):
                priority -= 20

            # Penalize non-content URLs
            if any(x in url_lower for x in ['/tag/', '/category/', '/author/', '/page/', '/archive/']):
                priority += 40
            if any(x in url_lower for x in ['/login', '/signup', '/register', '/cart', '/checkout']):
                priority += 100
            if '?' in url:
                priority += 20
            if url.count('/') > 6:
                priority += 15

        return priority

    def push(self, url: str, depth: int, config: Dict) -> bool:
        """Add URL to queue. Returns True if added."""
        with self.lock:
            # Normalize
            url = normalize_url(url)

            # Check if seen
            if url in self.seen:
                return False

            # Check depth limit
            max_depth = config.get('max_depth', 3)
            if depth > max_depth:
                return False

            # Check domain limit
            domain = get_domain(url)
            max_per_domain = config.get('max_per_domain', 100)
            if self.domain_counts[domain] >= max_per_domain:
                return False

            # Check same-domain restriction
            if config.get('same_domain', True) and self.seed_domains:
                if domain not in self.seed_domains:
                    return False

            # Calculate priority
            prioritize = config.get('prioritize_content', True)
            priority = self._calculate_priority(url, depth, prioritize)

            # Add to queue
            self.seen.add(url)
            self.domain_counts[domain] += 1
            self.counter += 1
            heapq.heappush(self.queue, (priority, self.counter, url, depth, domain))

            return True

    def push_many(self, urls: List[str], depth: int, config: Dict) -> int:
        """Add multiple URLs. Returns count added."""
        added = 0
        links_per_page = config.get('links_per_page', 20)

        # Shuffle to avoid bias toward top-of-page links
        urls_shuffled = list(urls)
        random.shuffle(urls_shuffled)

        for url in urls_shuffled[:links_per_page * 2]:  # Consider 2x, add up to limit
            if added >= links_per_page:
                break
            if self.push(url, depth, config):
                added += 1

        return added

    def pop(self) -> Optional[Tuple[str, int]]:
        """Get next URL. Returns (url, depth) or None."""
        with self.lock:
            while self.queue:
                priority, counter, url, depth, domain = heapq.heappop(self.queue)
                return (url, depth)
            return None

    def size(self) -> int:
        with self.lock:
            return len(self.queue)

    def get_stats(self) -> Dict:
        with self.lock:
            return {
                "queued": len(self.queue),
                "seen": len(self.seen),
                "domains": len(self.domain_counts)
            }

    def clear(self):
        with self.lock:
            self.queue.clear()
            self.seen = BloomFilter(expected_items=500000)
            self.domain_counts.clear()
            self.counter = 0
            self.seed_domains.clear()


# Global crawl queue
crawl_queue = CrawlQueue()


# ================================================================
# DOMAIN RATE LIMITER
# ================================================================
class DomainRateLimiter:
    """Prevent hammering single domains"""

    def __init__(self, min_delay: float = 1.0):
        self.last_request: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.min_delay = min_delay

    def wait_for_domain(self, domain: str):
        with self.lock:
            now = time.time()
            if domain in self.last_request:
                elapsed = now - self.last_request[domain]
                if elapsed < self.min_delay:
                    time.sleep(self.min_delay - elapsed + random.uniform(0.1, 0.5))
            self.last_request[domain] = time.time()

    def set_delay(self, delay: float):
        self.min_delay = delay


# ================================================================
# GLOBAL STATE
# ================================================================
proxy_manager = ProxyManager()
rate_limiter = DomainRateLimiter()
crawl_queue = CrawlQueue()

is_running = False
stop_requested = False
force_stop = False
last_stop_press = 0
output_file = "nttuner_dataset.jsonl"
log_lines = []
write_lock = threading.Lock()

# Statistics
stats = {
    "total_processed": 0,
    "success": 0,
    "failed": 0,
    "skipped": 0,
    "total_chars": 0,
    "speed": 0.0
}


# ================================================================
# UTILITIES
# ================================================================
def log(msg, color=[200, 200, 200]):
    ts = time.strftime("%H:%M:%S")
    formatted_msg = f"[NT] [{ts}] {msg}"
    log_lines.append(formatted_msg)
    if len(log_lines) > 500:
        log_lines.pop(0)

    try:
        if dpg.does_item_exist("log_text"):
            dpg.set_value("log_text", "\n".join(log_lines))
            dpg.set_y_scroll("log_group", -1.0)
    except:
        pass

    # Log to file if enabled
    try:
        if dpg.does_item_exist("chk_log_file") and dpg.get_value("chk_log_file"):
            with open("scraper_log.txt", "a", encoding="utf-8") as f:
                f.write(formatted_msg + "\n")
    except:
        pass


def update_stats_ui():
    try:
        if dpg.does_item_exist("stat_success"):
            dpg.set_value("stat_success", str(stats["success"]))
            dpg.set_value("stat_failed", str(stats["failed"]))
            dpg.set_value("stat_skipped", str(stats["skipped"]))
            dpg.set_value("stat_chars", f"{stats['total_chars'] / 1000:.1f}k")
            dpg.set_value("stat_speed", f"{stats['speed']:.1f}/s")
    except:
        pass


def get_domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except:
        return ""


def normalize_url(url: str) -> str:
    """Remove query params and fragments"""
    try:
        parsed = urllib.parse.urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
    except:
        return url


def build_text(system_override: str, user_content: str, template_key: str) -> str:
    """Build formatted training text"""
    tpl = MODEL_TEMPLATES.get(template_key, MODEL_TEMPLATES[DEFAULT_TEMPLATE_KEY])
    system_text = system_override.strip()

    text_out = tpl.get("begin", "")

    if system_text and "system" in tpl:
        text_out += tpl["system"].format(system=system_text)

    if "user" in tpl:
        text_out += tpl["user"].format(user=user_content.strip())

    if "assistant" in tpl:
        text_out += tpl["assistant"].format(assistant="[Detailed answer based on content]")

    return text_out


# ================================================================
# HTTP FETCHER (using working urllib approach)
# ================================================================
def extract_text_content(html: str, url: str, config: Dict) -> Tuple[str, List[str]]:
    """
    Simple, reliable content extraction (from working a.py).
    Returns (clean_text, links)
    """
    # Extract links first (before cleaning)
    found_links = []
    if config.get('crawler_enabled', False):
        found_links = extract_links(html, url)

    # Cleaning - remove problematic tags completely
    text = re.sub(r"<(script|style|nav|footer|header|iframe|noscript)[^>]*>.*?</\1>", "", html,
                  flags=re.I | re.S)

    # Remove code blocks if configured
    if config.get('clean_code', False):
        text = re.sub(r"<pre>.*?</pre>", "", text, flags=re.I | re.S)
        text = re.sub(r"<code>.*?</code>", "", text, flags=re.I | re.S)

    # Strip all HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Clean whitespace
    if config.get('clean_whitespace', True):
        text = re.sub(r"\s{2,}", " ", text).strip()
    else:
        text = re.sub(r"\n\s*\n", "\n\n", text).strip()  # Preserve paragraphs

    return text[:200000], found_links


def extract_links(html: str, base_url: str) -> List[str]:
    """Simple, reliable link extraction (from working a.py)"""
    links = set()
    matches = re.findall(r'href=["\'](http[s]?://[^"\']+)["\']', html)
    for link in matches:
        if link != base_url:
            links.add(link)
    return list(links)


def fetch_url(url: str, config: Dict) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Fetch URL using the working urllib approach from a.py.
    Returns: (content, error, links)
    """
    domain = get_domain(url)
    proxy_addr = None
    start_time = time.time()

    # Rate limit per domain
    rate_limiter.wait_for_domain(domain)

    # Setup handlers
    handlers = []

    # Proxy if enabled
    if config.get('use_proxy', False):
        proxy_addr = proxy_manager.get_best_proxy()
        if proxy_addr:
            # Check for SOCKS proxy
            if "socks" in proxy_addr.lower():
                return None, "SOCKS not supported via urllib", []
            handlers.append(urllib.request.ProxyHandler({'http': proxy_addr, 'https': proxy_addr}))

    opener = urllib.request.build_opener(*handlers)

    # Feature: User Agent Rotation (from a.py)
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NTTuner/1.0"
    if config.get('rotate_ua', True):
        ua = random.choice(USER_AGENTS)

    headers = [("User-Agent", ua)]
    opener.addheaders = headers

    timeout = config.get('timeout', 25)
    max_retries = config.get('max_retries', 3)

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                backoff = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(backoff)

            req = urllib.request.Request(url)
            with opener.open(req, timeout=timeout) as resp:
                raw_data = resp.read().decode("utf-8", errors="ignore")

            # Success - report proxy
            if proxy_addr:
                proxy_manager.report_success(proxy_addr, time.time() - start_time)

            # Extract meaningful text content
            clean_text, links = extract_text_content(raw_data, url, config)

            if not clean_text or len(clean_text) < 50:
                return None, "No extractable content", []

            return clean_text, None, links

        except HTTPError as e:
            if proxy_addr:
                if e.code in (403, 429, 503):
                    proxy_manager.report_failure(proxy_addr)
            if attempt < max_retries - 1:
                continue
            return None, f"HTTP {e.code}", []

        except Exception as e:
            if proxy_addr:
                proxy_manager.report_failure(proxy_addr)
            if attempt < max_retries - 1:
                continue
            return None, f"Net: {type(e).__name__}", []

    return None, "Max retries exceeded", []


# ================================================================
# WORKER FUNCTION
# ================================================================
def process_url(url: str, depth: int, config: Dict) -> Tuple[str, bool, int, List[str]]:
    """
    Process a single URL. Returns (url, success, char_count, new_links)
    Legacy function for compatibility
    """
    result = process_url_v2(url, depth, config)
    return (url, result.get('success', False), result.get('chars', 0), result.get('links', []))


def process_url_v2(url: str, depth: int, config: Dict) -> Dict:
    """
    Process a single URL. Returns dict with results.
    """
    if stop_requested or force_stop:
        return {'success': False, 'error': None, 'chars': 0, 'links': [], 'filtered': True}

    # Domain blacklist
    domain_bl = config.get('domain_blacklist', [])
    if any(bl in url.lower() for bl in domain_bl):
        return {'success': False, 'error': None, 'chars': 0, 'links': [], 'filtered': True}

    # Fetch
    content, error, links = fetch_url(url, config)

    if error:
        log(f"  [-] [D{depth}] {url[:50]}... : {error}", [255, 100, 100])
        return {'success': False, 'error': error, 'chars': 0, 'links': [], 'filtered': False}

    if not content:
        return {'success': False, 'error': None, 'chars': 0, 'links': [], 'filtered': True}

    # Apply filters
    content_len = len(content)
    text_lower = content.lower()

    min_chars = config.get('min_chars', 300)
    max_chars = config.get('max_chars', 50000)
    kw_in = config.get('keywords_in', [])
    kw_out = config.get('keywords_out', [])

    if content_len < min_chars:
        log(f"  [!] Too short ({content_len}): {url[:40]}...", [150, 150, 100])
        return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

    if content_len > max_chars:
        log(f"  [!] Too long ({content_len}): {url[:40]}...", [150, 150, 100])
        return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

    if kw_in and not any(k in text_lower for k in kw_in):
        return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

    if kw_out and any(k in text_lower for k in kw_out):
        return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

    # Build training text
    system_prompt = config.get('system_prompt', '')
    template_key = config.get('template', DEFAULT_TEMPLATE_KEY)
    final_text = build_text(system_prompt, content, template_key)

    # Write to file (thread-safe)
    with write_lock:
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"text": final_text}, ensure_ascii=False) + "\n")
        except Exception as e:
            log(f"  [!] Write error: {e}", [255, 100, 100])
            return {'success': False, 'error': str(e), 'chars': 0, 'links': [], 'filtered': False}

    log(f"  [+] [D{depth}] {url[:50]}... ({content_len} chars)", [0, 255, 150])

    # Return links for crawling (even if we succeed, pass links for discovery)
    return {'success': True, 'error': None, 'chars': content_len, 'links': links, 'filtered': False}


# ================================================================
# MAIN ENGINE
# ================================================================
def scrape_worker():
    global is_running, stop_requested, force_stop, stats, crawl_queue

    # Get URLs
    raw_urls = dpg.get_value("urls_input").splitlines()
    initial_urls = [normalize_url(u.strip()) for u in raw_urls if u.strip().startswith("http")]

    if not initial_urls:
        log("Engine Init Failed: No valid URLs", [255, 100, 100])
        is_running = False
        return

    # Reset crawl queue
    crawl_queue.clear()
    for k in stats:
        stats[k] = 0

    # Build config
    config = {
        "use_proxy": dpg.get_value("use_proxy_checkbox"),
        "rotate_ua": dpg.get_value("chk_ua_rotate"),
        "max_retries": dpg.get_value("inp_max_retries"),
        "timeout": dpg.get_value("timeout_sec"),
        "system_prompt": dpg.get_value("system_prompt_input"),
        "template": dpg.get_value("template_combo"),
        "min_chars": dpg.get_value("inp_min_chars"),
        "max_chars": dpg.get_value("inp_max_chars"),
        "keywords_in": [k.strip().lower() for k in dpg.get_value("inp_kw_in").split(",") if k.strip()],
        "keywords_out": [k.strip().lower() for k in dpg.get_value("inp_kw_out").split(",") if k.strip()],
        "domain_blacklist": [d.strip().lower() for d in dpg.get_value("inp_domain_bl").split(",") if d.strip()],
        "crawler_enabled": dpg.get_value("chk_crawler"),
        "max_depth": dpg.get_value("inp_max_depth"),
        "links_per_page": dpg.get_value("inp_links_per_page"),
        "max_per_domain": dpg.get_value("inp_max_per_domain"),
        "same_domain": dpg.get_value("chk_same_domain"),
        "prioritize_content": dpg.get_value("chk_prioritize_content"),
        "clean_code": dpg.get_value("chk_clean_code"),
        "clean_whitespace": dpg.get_value("chk_clean_whitespace"),
    }

    # Set rate limiter
    rate_limiter.set_delay(dpg.get_value("inp_domain_delay"))

    # Workers
    num_workers = dpg.get_value("inp_workers")
    stop_limit = dpg.get_value("inp_stop_limit")

    # Initialize crawl queue with seed domains
    crawl_queue.set_seed_domains(initial_urls)
    for url in initial_urls:
        crawl_queue.push(url, 0, config)

    log(f"Engine Online. Seeds: {len(initial_urls)}, Workers: {num_workers}", [100, 200, 255])
    if config['crawler_enabled']:
        log(f"Crawler: Depth={config['max_depth']}, Links/Page={config['links_per_page']}, PerDomain={config['max_per_domain']}", [100, 200, 255])
        log(f"         SameDomain={config['same_domain']}, Prioritize={config['prioritize_content']}", [100, 200, 255])

    start_time = time.time()
    processed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        while not force_stop and not stop_requested:
            # Submit new tasks from crawl queue
            while len(futures) < num_workers:
                item = crawl_queue.pop()
                if item is None:
                    break
                url, depth = item
                future = executor.submit(process_url_v2, url, depth, config)
                futures[future] = (url, depth)

            if not futures:
                if crawl_queue.size() == 0:
                    break
                time.sleep(0.1)
                continue

            # Process completed futures
            done_futures = [f for f in futures if f.done()]

            for future in done_futures:
                url, depth = futures.pop(future)
                processed_count += 1

                try:
                    result = future.result()
                    success = result.get('success', False)
                    char_count = result.get('chars', 0)
                    new_links = result.get('links', [])
                    error = result.get('error', None)
                    filtered = result.get('filtered', False)

                    if success:
                        stats["success"] += 1
                        stats["total_chars"] += char_count

                        # Add discovered links to crawl queue
                        if config['crawler_enabled'] and new_links and depth < config['max_depth']:
                            added = crawl_queue.push_many(new_links, depth + 1, config)
                            if added > 0:
                                log(f"    [>] +{added} links queued (depth {depth+1})", [100, 180, 255])
                    else:
                        if error:
                            stats["failed"] += 1
                        else:
                            stats["skipped"] += 1

                except Exception as e:
                    stats["failed"] += 1
                    log(f"  [!] Worker error: {e}", [255, 100, 100])

                # Update speed
                elapsed = time.time() - start_time
                if elapsed > 0:
                    stats["speed"] = processed_count / elapsed

                update_stats_ui()

                # Update progress with queue info
                queue_size = crawl_queue.size()
                total = processed_count + queue_size
                progress = processed_count / max(1, total)
                try:
                    dpg.set_value("progress_bar", progress)
                    queue_stats = crawl_queue.get_stats()
                    label = f"Done: {processed_count} | Queue: {queue_size} | Domains: {queue_stats['domains']} | OK: {stats['success']}"
                    dpg.configure_item("progress_bar", label=label)
                except:
                    pass

                # Check stop limit
                if stop_limit > 0 and stats["success"] >= stop_limit:
                    log(f"Stop limit reached ({stop_limit})", [0, 255, 0])
                    stop_requested = True
                    break

            time.sleep(0.05)

    # Finish
    elapsed = time.time() - start_time
    queue_stats = crawl_queue.get_stats()
    log(f"Engine Offline. Success: {stats['success']}, Failed: {stats['failed']}, Time: {elapsed:.1f}s", [0, 255, 200])
    log(f"Crawl Summary: Discovered {queue_stats['seen']} URLs across {queue_stats['domains']} domains", [0, 180, 255])

    if dpg.get_value("chk_sound") and HAS_WINSOUND:
        try:
            winsound.MessageBeep()
        except:
            pass

    is_running = stop_requested = force_stop = False
    try:
        dpg.set_value("progress_bar", 0.0)
        dpg.configure_item("progress_bar", label="IDLE")
    except:
        pass


def start_scrape():
    global is_running, stop_requested, force_stop
    if is_running:
        return

    is_running = True
    stop_requested = force_stop = False
    threading.Thread(target=scrape_worker, daemon=True).start()


def handle_stop():
    global stop_requested, force_stop, last_stop_press
    if time.time() - last_stop_press < 1.5:
        force_stop = True
        log("FORCE STOP", [255, 50, 50])
    else:
        stop_requested = True
        log("Stopping... (double-click to force)", [255, 200, 50])
    last_stop_press = time.time()


# ================================================================
# PROXY FUNCTIONS
# ================================================================
def fetch_proxies_selected():
    source = dpg.get_value("proxy_source_combo")
    url = PROXY_SOURCES.get(source)
    if not url:
        return

    log(f"Fetching: {source}...", [255, 200, 100])

    def fetch():
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = resp.read().decode("utf-8")

            # Parse proxies
            proxies = []
            # Standard IP:PORT
            found = re.findall(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[^\d]{1,5}(\d{2,5})', data)
            proxies.extend([f"{ip}:{port}" for ip, port in found])

            # JSON format
            if '{' in data:
                try:
                    json_data = json.loads(data)
                    if isinstance(json_data, dict) and 'data' in json_data:
                        for item in json_data['data']:
                            if 'ip' in item and 'port' in item:
                                proxies.append(f"{item['ip']}:{item['port']}")
                except:
                    pass

            proxy_manager.add_proxies(list(set(proxies)))
            stats = proxy_manager.get_stats()
            log(f"Added {len(proxies)} proxies. Pool: {stats['active']}/{stats['total']}", [0, 255, 150])
            dpg.set_value("proxy_status", f"Pool: {stats['active']}/{stats['total']}")
        except Exception as e:
            log(f"Fetch failed: {e}", [255, 100, 100])

    threading.Thread(target=fetch, daemon=True).start()


def fetch_all_proxies():
    log("Fetching from ALL sources...", [255, 200, 100])

    def fetch_all():
        total = 0
        for name, url in PROXY_SOURCES.items():
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = resp.read().decode("utf-8")
                found = re.findall(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[^\d]{1,5}(\d{2,5})', data)
                proxies = [f"{ip}:{port}" for ip, port in found]
                proxy_manager.add_proxies(proxies)
                total += len(proxies)
                log(f"  [{name}] +{len(proxies)}", [100, 200, 100])
            except:
                pass

        stats = proxy_manager.get_stats()
        log(f"Fetch complete. Pool: {stats['active']}/{stats['total']}", [0, 255, 150])
        dpg.set_value("proxy_status", f"Pool: {stats['active']}/{stats['total']}")

    threading.Thread(target=fetch_all, daemon=True).start()


def import_custom_proxies():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    root.destroy()

    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                proxies = [l.strip() for l in f.readlines() if l.strip() and ':' in l]
            proxy_manager.add_proxies(proxies)
            stats = proxy_manager.get_stats()
            log(f"Imported {len(proxies)} proxies", [0, 180, 255])
            dpg.set_value("proxy_status", f"Pool: {stats['active']}/{stats['total']}")
        except Exception as e:
            log(f"Import failed: {e}", [255, 100, 100])


def clear_proxy_quarantine():
    proxy_manager.clear_quarantine()
    stats = proxy_manager.get_stats()
    log(f"Quarantine cleared. Active: {stats['active']}", [0, 255, 150])
    dpg.set_value("proxy_status", f"Pool: {stats['active']}/{stats['total']}")


# ================================================================
# CONFIG FUNCTIONS
# ================================================================
def save_config():
    config = {
        "workers": dpg.get_value("inp_workers"),
        "domain_delay": dpg.get_value("inp_domain_delay"),
        "max_retries": dpg.get_value("inp_max_retries"),
        "timeout": dpg.get_value("timeout_sec"),
        "system_prompt": dpg.get_value("system_prompt_input"),
        "use_proxy": dpg.get_value("use_proxy_checkbox"),
        "rotate_ua": dpg.get_value("chk_ua_rotate"),
        "min_chars": dpg.get_value("inp_min_chars"),
        "max_chars": dpg.get_value("inp_max_chars"),
        "keywords_in": dpg.get_value("inp_kw_in"),
        "keywords_out": dpg.get_value("inp_kw_out"),
        "crawler_enabled": dpg.get_value("chk_crawler"),
        "max_depth": dpg.get_value("inp_max_depth"),
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        log("Config saved", [0, 255, 100])
    except Exception as e:
        log(f"Save failed: {e}", [255, 100, 100])


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

        dpg.set_value("inp_workers", config.get("workers", 10))
        dpg.set_value("inp_domain_delay", config.get("domain_delay", 1.0))
        dpg.set_value("inp_max_retries", config.get("max_retries", 3))
        dpg.set_value("timeout_sec", config.get("timeout", 25))
        dpg.set_value("system_prompt_input", config.get("system_prompt", DEFAULT_SYSTEM))
        dpg.set_value("use_proxy_checkbox", config.get("use_proxy", False))
        dpg.set_value("chk_ua_rotate", config.get("rotate_ua", True))
        dpg.set_value("inp_min_chars", config.get("min_chars", 300))
        dpg.set_value("inp_max_chars", config.get("max_chars", 50000))
        dpg.set_value("inp_kw_in", config.get("keywords_in", ""))
        dpg.set_value("inp_kw_out", config.get("keywords_out", ""))
        dpg.set_value("chk_crawler", config.get("crawler_enabled", False))
        dpg.set_value("inp_max_depth", config.get("max_depth", 2))

        log("Config loaded", [0, 255, 100])
    except Exception as e:
        log(f"Load failed: {e}", [255, 100, 100])


def update_output_file():
    global output_file
    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(
        defaultextension=".jsonl",
        initialfile=output_file,
        filetypes=[("JSONL", "*.jsonl"), ("JSON", "*.json"), ("TXT", "*.txt")]
    )
    root.destroy()
    if path:
        output_file = path
        dpg.set_value("out_file", output_file)
        log(f"Output: {output_file}")


def set_preset_prompt(sender, app_data):
    if app_data in SYSTEM_PROMPTS:
        dpg.set_value("system_prompt_input", SYSTEM_PROMPTS[app_data])


def copy_log():
    try:
        import pyperclip
        pyperclip.copy("\n".join(log_lines))
        log("Log copied", [0, 200, 255])
    except:
        log("Install pyperclip: pip install pyperclip", [255, 200, 100])


# ================================================================
# GUI
# ================================================================
dpg.create_context()

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [15, 15, 18])
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [22, 22, 26])
        dpg.add_theme_color(dpg.mvThemeCol_Border, [45, 45, 52])
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [28, 28, 34])
        dpg.add_theme_color(dpg.mvThemeCol_Button, [35, 35, 42])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [50, 50, 65])
        dpg.add_theme_color(dpg.mvThemeCol_Text, [220, 220, 225])
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, [0, 180, 255])
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)

with dpg.window(tag="PrimaryWindow"):
    dpg.add_text("NTCompanion Pro - ThreadPool Scraper", color=[100, 200, 255])
    dpg.add_text(f"Version: {VERSION} | urllib + ThreadPoolExecutor", color=[150, 150, 150])
    dpg.add_separator()

    # === SOURCE MANIFEST ===
    with dpg.collapsing_header(label="Source Manifest", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_text("URL LIST", color=[150, 150, 160])
            dpg.add_button(label="Clear", small=True, callback=lambda: dpg.set_value("urls_input", ""))

        dpg.add_input_text(tag="urls_input", multiline=True, height=150, width=-1,
                           hint="Enter URLs (http/https), one per line...")

        dpg.add_text("LIVE STATS:", color=[150, 150, 150])
        with dpg.group(horizontal=True):
            dpg.add_text("OK:", color=[0, 255, 0]); dpg.add_text("0", tag="stat_success")
            dpg.add_text("  Fail:", color=[255, 50, 50]); dpg.add_text("0", tag="stat_failed")
            dpg.add_text("  Skip:", color=[255, 200, 0]); dpg.add_text("0", tag="stat_skipped")
            dpg.add_text("  Vol:", color=[0, 200, 255]); dpg.add_text("0k", tag="stat_chars")
            dpg.add_text("  Speed:", color=[200, 100, 255]); dpg.add_text("0/s", tag="stat_speed")

    # === CONCURRENCY ===
    with dpg.collapsing_header(label="Concurrency Settings", default_open=False):
        dpg.add_text("THREAD POOL", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="Workers", tag="inp_workers", default_value=10, min_value=1, max_value=50, width=100)
            dpg.add_input_float(label="Domain Delay (s)", tag="inp_domain_delay", default_value=1.0, width=100)
            dpg.add_input_int(label="Max Retries", tag="inp_max_retries", default_value=3, width=100)

        dpg.add_input_float(label="Timeout (s)", tag="timeout_sec", default_value=25.0, width=100)

        dpg.add_spacer(height=5)
        dpg.add_text("Recommended: 5-10 workers, 1-2s domain delay", color=[100, 100, 120])

    # === PROXY ===
    with dpg.collapsing_header(label="Proxy Configuration", default_open=False):
        dpg.add_checkbox(label="Enable Proxies", tag="use_proxy_checkbox", default_value=False)
        dpg.add_checkbox(label="User-Agent Rotation", tag="chk_ua_rotate", default_value=True)

        dpg.add_spacer(height=10)
        dpg.add_text("PROXY SOURCES (20+)", color=[150, 150, 160])
        dpg.add_combo(list(PROXY_SOURCES.keys()), default_value="ProxyScrape HTTPS",
                      tag="proxy_source_combo", width=-1)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Fetch Selected", callback=fetch_proxies_selected, width=120)
            dpg.add_button(label="Fetch ALL", callback=fetch_all_proxies, width=100)
            dpg.add_button(label="Import File", callback=import_custom_proxies, width=100)
            dpg.add_button(label="Clear Quarantine", callback=clear_proxy_quarantine, width=120)

        dpg.add_text("Status: No proxies", tag="proxy_status", color=[0, 200, 150])

    # === CRAWLER ===
    with dpg.collapsing_header(label="Crawler Configuration", default_open=False):
        dpg.add_text("CRAWL SETTINGS", color=[150, 150, 160])
        dpg.add_checkbox(label="Enable Crawler (Follow Links)", tag="chk_crawler", default_value=False)

        with dpg.group(horizontal=True):
            dpg.add_input_int(label="Max Depth", tag="inp_max_depth", default_value=3, min_value=1, max_value=10, width=100)
            dpg.add_input_int(label="Links Per Page", tag="inp_links_per_page", default_value=20, min_value=1, max_value=100, width=100)
            dpg.add_input_int(label="Max Per Domain", tag="inp_max_per_domain", default_value=100, min_value=1, max_value=1000, width=100)

        dpg.add_checkbox(label="Stay On Same Domain", tag="chk_same_domain", default_value=True)
        dpg.add_checkbox(label="Prioritize Content Pages", tag="chk_prioritize_content", default_value=True)

        dpg.add_spacer(height=5)
        dpg.add_text("Depth Guide: 1=seed only, 2=seed+links, 3+=deep crawl", color=[100, 100, 120])
        dpg.add_text("Same Domain: Only follow links to the original domain(s)", color=[100, 100, 120])

    # === FILTERS ===
    with dpg.collapsing_header(label="Filter Configuration", default_open=False):
        dpg.add_text("CONTENT CLEANING", color=[150, 150, 160])
        dpg.add_checkbox(label="Remove Code Blocks", tag="chk_clean_code", default_value=False)
        dpg.add_checkbox(label="Collapse Whitespace", tag="chk_clean_whitespace", default_value=True)

        dpg.add_spacer(height=10)
        dpg.add_text("SIZE", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="Min Chars", tag="inp_min_chars", default_value=300, width=120)
            dpg.add_input_int(label="Max Chars", tag="inp_max_chars", default_value=50000, width=120)
            dpg.add_input_int(label="Stop After N", tag="inp_stop_limit", default_value=0, width=120)

        dpg.add_spacer(height=10)
        dpg.add_text("KEYWORDS (comma separated)", color=[150, 150, 160])
        dpg.add_input_text(label="Must Contain", tag="inp_kw_in", width=-1)
        dpg.add_input_text(label="Exclude If", tag="inp_kw_out", width=-1)
        dpg.add_input_text(label="Domain Blacklist", tag="inp_domain_bl", width=-1,
                           hint="facebook.com, twitter.com")

    # === PROMPT ===
    with dpg.collapsing_header(label="Prompt & Template", default_open=False):
        dpg.add_text("SYSTEM PROMPT", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_combo(list(SYSTEM_PROMPTS.keys()), label="Preset", width=200, callback=set_preset_prompt)
            dpg.add_text("'Blank' = no system", color=[100, 200, 100])

        dpg.add_input_text(tag="system_prompt_input", multiline=True, height=100, width=-1,
                           default_value=DEFAULT_SYSTEM)

        dpg.add_spacer(height=10)
        dpg.add_text("TEMPLATE", color=[150, 150, 160])
        dpg.add_combo(list(MODEL_TEMPLATES.keys()), default_value=DEFAULT_TEMPLATE_KEY,
                      tag="template_combo", width=-1)

    # === OUTPUT ===
    with dpg.collapsing_header(label="Output Settings", default_open=False):
        dpg.add_text("OUTPUT FILE", color=[150, 150, 160])
        dpg.add_input_text(default_value=output_file, tag="out_file", width=-1, readonly=True)
        dpg.add_button(label="Select...", callback=update_output_file, width=100)

        dpg.add_spacer(height=10)
        dpg.add_checkbox(label="Log to File", tag="chk_log_file", default_value=False)
        dpg.add_checkbox(label="Sound on Finish", tag="chk_sound", default_value=True)

    dpg.add_separator()

    # === CONTROLS ===
    with dpg.group(horizontal=True):
        start_btn = dpg.add_button(label="START", width=120, height=35, callback=start_scrape)
        with dpg.theme() as start_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, [20, 80, 40])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [30, 120, 60])
        dpg.bind_item_theme(start_btn, start_theme)

        stop_btn = dpg.add_button(label="STOP", width=100, height=35, callback=handle_stop)
        with dpg.theme() as stop_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, [100, 30, 30])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [150, 50, 50])
        dpg.bind_item_theme(stop_btn, stop_theme)

        dpg.add_button(label="Save Config", small=True, callback=save_config)
        dpg.add_button(label="Copy Log", small=True, callback=copy_log)

    dpg.add_separator()

    # === PROGRESS ===
    dpg.add_progress_bar(tag="progress_bar", label="IDLE", width=-1, height=18)
    dpg.add_spacer(height=8)
    dpg.add_text("Console:")
    with dpg.child_window(tag="log_group", height=-35, border=True):
        dpg.add_text("", tag="log_text", wrap=0)

    dpg.add_separator()
    dpg.add_text(f"NTCompanion Pro | {VERSION}", color=[50, 50, 60])

# Init
load_config()
if os.path.exists(INI_FILE):
    dpg.load_init_file(INI_FILE)
dpg.set_exit_callback(lambda: dpg.save_init_file(INI_FILE))

dpg.create_viewport(title=f"NTCompanion Pro - {VERSION}", width=1050, height=950, resizable=True)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("PrimaryWindow", True)
dpg.bind_theme(global_theme)
dpg.start_dearpygui()
dpg.destroy_context()