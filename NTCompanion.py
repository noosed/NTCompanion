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
import shutil
import tempfile
import subprocess

# GitHub repository support
try:
    import git

    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False
    print("Note: Install gitpython for GitHub repo support: pip install gitpython")

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
VERSION = "build.2026.05"
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# GitHub repository processing - CODEBASE FEATURE
MAX_REPO_SIZE_MB = 500  # Skip repos larger than this
GITHUB_CLONE_TIMEOUT = 300  # 5 minutes per repo clone
CONCURRENT_REPO_CLONES = 4  # Parallel repo cloning

# RAG chunking defaults - CODEBASE FEATURE
DEFAULT_CHUNK_SIZE = 50  # lines per chunk
MIN_CHUNK_SIZE = 20
MAX_CHUNK_SIZE = 200

# Code file extensions to process - CODEBASE FEATURE
CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.m', '.mm', '.sh', '.bash', '.ps1', '.sql', '.html', '.css', '.scss',
    '.less', '.vue', '.svelte', '.dart', '.lua', '.perl', '.pl', '.hs',
    '.ml', '.fs', '.ex', '.exs', '.clj', '.lisp', '.scm', '.erl', '.elm'
}

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

# ================================================================
# CONTENT TYPE CONFIGURATIONS
# ================================================================
CONTENT_TYPES = {
    "Recipe": {
        "user_prompt_template": "How do I make {title}?",
        "detail_sections": ["Ingredients", "Instructions"],
        "system_prompt": "You are a helpful and honest assistant.",
        "example_titles": ["Chocolate Cake", "Spaghetti Carbonara", "Thai Green Curry"]
    },
    "Tutorial": {
        "user_prompt_template": "How do I {title}?",
        "detail_sections": ["Requirements", "Steps", "Tips"],
        "system_prompt": "You are a helpful technical assistant.",
        "example_titles": ["set up a web server", "configure SSH keys", "build a Docker container"]
    },
    "Product Info": {
        "user_prompt_template": "Tell me about {title}",
        "detail_sections": ["Features", "Specifications", "Reviews"],
        "system_prompt": "You are a helpful product information assistant.",
        "example_titles": ["iPhone 15 Pro", "Sony WH-1000XM5", "Tesla Model 3"]
    },
    "Article/Blog": {
        "user_prompt_template": "Please provide detailed information about {title}",
        "detail_sections": ["Summary", "Key Points", "Conclusion"],
        "system_prompt": "You are a helpful and honest assistant.",
        "example_titles": ["Climate Change Effects", "Machine Learning Basics", "Remote Work Tips"]
    },
    "Documentation": {
        "user_prompt_template": "Explain {title}",
        "detail_sections": ["Overview", "Usage", "Examples", "Parameters"],
        "system_prompt": "You are a technical documentation assistant.",
        "example_titles": ["React Hooks API", "Python asyncio", "Git branching"]
    },
    "FAQ": {
        "user_prompt_template": "How do I {title}",
        "detail_sections": ["Questions", "Answers"],
        "system_prompt": "You are a helpful and honest assistant.",
        "example_titles": ["reset my password", "contact support", "cancel subscription"]
    },
    "News": {
        "user_prompt_template": "Summarize: {title}",
        "detail_sections": ["Summary", "Details", "Context"],
        "system_prompt": "You are a news summarization assistant.",
        "example_titles": ["Latest Tech Announcements", "Stock Market Update", "Political Summit"]
    },
    "Research Paper": {
        "user_prompt_template": "Explain the research paper: {title}",
        "detail_sections": ["Abstract", "Methods", "Results", "Conclusions"],
        "system_prompt": "You are an academic research assistant.",
        "example_titles": ["Attention Is All You Need", "Deep Learning for NLP"]
    },
    "Medical Info": {
        "user_prompt_template": "Provide information about {title}",
        "detail_sections": ["Overview", "Symptoms", "Treatment", "Prevention"],
        "system_prompt": "You are a medical information assistant. Provide factual information but remind users to consult healthcare professionals.",
        "example_titles": ["Type 2 Diabetes", "Hypertension Management"]
    },
    "Code File": {
        "user_prompt_template": "Explain this code from {title}",
        "detail_sections": ["Purpose", "Key Functions", "Usage"],
        "system_prompt": "You are an expert programmer who explains code clearly and concisely.",
        "example_titles": ["main.py", "UserController.java", "utils.js"]
    },
    "Custom": {
        "user_prompt_template": "{title}",
        "detail_sections": [],
        "system_prompt": "You are a helpful and honest assistant.",
        "example_titles": [""]
    }
}

DEFAULT_CONTENT_TYPE = "Recipe"

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

# Common subdomains for discovery
COMMON_SUBDOMAINS = [
    'www', 'blog', 'shop', 'store', 'api', 'dev', 'docs', 'support', 'help',
    'portal', 'app', 'mobile', 'cdn', 'static', 'images', 'media', 'news',
    'community', 'forum', 'wiki', 'learn', 'training', 'academy', 'education',
    'status', 'dashboard', 'admin', 'mail', 'webmail', 'secure', 'my',
    'download', 'files', 'ftp', 'data', 'test', 'demo', 'beta', 'stage',
]


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
            # Get full domain
            domain = get_domain(url)
            if domain:
                self.seed_domains.add(domain)

                # Also add root domain for subdomain support
                # e.g., www.example.com -> example.com
                parts = domain.split('.')
                if len(parts) >= 2:
                    root_domain = '.'.join(parts[-2:])
                    self.seed_domains.add(root_domain)

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
            if depth >= max_depth:
                return False

            # Check domain limit
            domain = get_domain(url)
            max_per_domain = config.get('max_per_domain', 100)
            if self.domain_counts[domain] >= max_per_domain:
                return False

            # Check same-domain restriction
            if config.get('same_domain', True) and self.seed_domains:
                domain = get_domain(url)
                # Check both full domain and root domain
                is_allowed = domain in self.seed_domains

                # Also check root domain (e.g., themealdb.com for www.themealdb.com)
                if not is_allowed:
                    parts = domain.split('.')
                    if len(parts) >= 2:
                        root_domain = '.'.join(parts[-2:])
                        is_allowed = root_domain in self.seed_domains

                if not is_allowed:
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
# CODEBASE SCANNER - NEW FEATURE
# ================================================================
def scan_codebase_folder(folder_path: str) -> List[Tuple[str, str]]:
    """Recursively scan a folder for code files."""
    code_files = []

    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in {
            '.git', '.svn', '.hg', '__pycache__', 'node_modules',
            '.venv', 'venv', 'env', 'build', 'dist', '.idea',
            '.vscode', 'target', 'bin', 'obj', '.gradle'
        }]

        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in CODE_EXTENSIONS:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                code_files.append((full_path, rel_path))

    return code_files


def read_code_file(file_path: str, max_size: int = 100000) -> Optional[str]:
    """Read a code file with proper encoding detection."""
    try:
        if not os.path.exists(file_path):
            return None

        file_size = os.path.getsize(file_path)
        if file_size > max_size or file_size == 0:
            return None

        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()

                if '\x00' in content:
                    return None

                if len(content.strip()) < 10:
                    return None

                return content
            except (UnicodeDecodeError, UnicodeError, PermissionError):
                continue
            except Exception:
                return None

        return None
    except Exception:
        return None


def extract_code_metadata(content: str, file_path: str) -> Dict[str, any]:
    """Extract metadata from code content."""
    metadata = {
        'functions': [],
        'classes': [],
        'imports': [],
        'comments': 0,
        'lines': len(content.split('\n'))
    }

    func_patterns = [
        r'def\s+(\w+)\s*\(',
        r'function\s+(\w+)\s*\(',
        r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*{',
    ]

    for pattern in func_patterns:
        metadata['functions'].extend(re.findall(pattern, content))

    class_patterns = [r'class\s+(\w+)']
    for pattern in class_patterns:
        metadata['classes'].extend(re.findall(pattern, content))

    comment_patterns = [r'#[^\n]*', r'//[^\n]*', r'/\*.*?\*/']
    for pattern in comment_patterns:
        metadata['comments'] += len(re.findall(pattern, content, re.DOTALL))

    return metadata


def format_code_for_training(file_path: str, rel_path: str, content: str,
                             metadata: Dict, config: Dict) -> str:
    """Format code file content for training data."""
    parts = []
    parts.append(f"File: {rel_path}")
    parts.append(f"Language: {os.path.splitext(file_path)[1][1:]}")
    parts.append(f"Lines: {metadata['lines']}")

    if metadata['functions']:
        parts.append(f"Functions: {', '.join(metadata['functions'][:10])}")
    if metadata['classes']:
        parts.append(f"Classes: {', '.join(metadata['classes'][:10])}")

    parts.append("\nCode:\n")
    parts.append(content)

    return "\n".join(parts)


def process_code_file(file_path: str, rel_path: str, config: Dict) -> Dict:
    """Process a single code file for dataset generation."""
    if stop_requested or force_stop:
        return {'success': False, 'error': None, 'chars': 0, 'filtered': True}

    content = read_code_file(file_path, config.get('max_chars', 100000))

    if content is None:
        log(f"  [SKIP] Could not read: {rel_path}", [150, 150, 150])
        return {'success': False, 'error': 'Could not read file', 'chars': 0, 'filtered': True}

    content_len = len(content)
    min_chars = config.get('min_chars', 50)

    if content_len < min_chars:
        log(f"  [SKIP] Too short ({content_len}): {rel_path}", [150, 150, 150])
        return {'success': False, 'error': None, 'chars': 0, 'filtered': True}

    metadata = extract_code_metadata(content, file_path)
    formatted_content = format_code_for_training(file_path, rel_path, content, metadata, config)

    system_prompt = config.get('system_prompt', '')
    template_key = config.get('template', DEFAULT_TEMPLATE_KEY)
    content_type_cfg = CONTENT_TYPES.get("Code File", CONTENT_TYPES["Custom"])
    final_text = build_text(system_prompt, formatted_content, template_key, content_type_cfg)

    with write_lock:
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"text": final_text}, ensure_ascii=False) + "\n")
        except Exception as e:
            log(f"  [!] Write error: {e}", [255, 100, 100])
            return {'success': False, 'error': str(e), 'chars': 0, 'filtered': False}

    funcs = len(metadata['functions'])
    classes = len(metadata['classes'])
    log(f"  [+] {rel_path} ({content_len} chars, {funcs}F/{classes}C)", [0, 255, 150])

    return {'success': True, 'error': None, 'chars': content_len, 'filtered': False}


def process_codebase_worker():
    """Worker function for processing code files from a folder"""
    global is_running, stop_requested, force_stop, stats

    try:
        folder_path = dpg.get_value("codebase_folder_path")

        if not folder_path or not os.path.isdir(folder_path):
            log("Invalid folder path", [255, 100, 100])
            is_running = False
            return

        for k in stats:
            stats[k] = 0

        config = {
            "system_prompt": dpg.get_value("system_prompt_input"),
            "template": dpg.get_value("template_combo"),
            "min_chars": dpg.get_value("inp_min_chars"),
            "max_chars": dpg.get_value("inp_max_chars"),
        }

        log(f"Scanning codebase: {folder_path}", [100, 200, 255])

        try:
            code_files = scan_codebase_folder(folder_path)
        except PermissionError as e:
            log(f"Permission denied: {e}", [255, 100, 100])
            is_running = False
            return
        except Exception as e:
            log(f"Error scanning folder: {e}", [255, 100, 100])
            is_running = False
            return

        if not code_files:
            log("No code files found", [255, 150, 100])
            is_running = False
            return

        log(f"Found {len(code_files)} code files", [100, 255, 150])

        num_workers = dpg.get_value("inp_workers")
        start_time = time.time()
        processed_count = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}

            for fp, rp in code_files:
                try:
                    future = executor.submit(process_code_file, fp, rp, config)
                    futures[future] = (fp, rp)
                except Exception as e:
                    log(f"Error submitting task for {rp}: {e}", [255, 150, 100])
                    continue

            for future in as_completed(futures):
                if force_stop or stop_requested:
                    log("[STOP] Cancelling remaining tasks...", [255, 200, 100])
                    for f in futures:
                        f.cancel()
                    break

                file_path, rel_path = futures[future]
                processed_count += 1

                try:
                    result = future.result(timeout=30)

                    if result.get('success'):
                        stats["success"] += 1
                        stats["total_chars"] += result.get('chars', 0)
                    else:
                        if result.get('error'):
                            stats["failed"] += 1
                        else:
                            stats["skipped"] += 1

                except TimeoutError:
                    stats["failed"] += 1
                    log(f"  [!] Timeout processing {rel_path}", [255, 100, 100])
                except Exception as e:
                    stats["failed"] += 1
                    log(f"  [!] Error processing {rel_path}: {e}", [255, 100, 100])

                elapsed = time.time() - start_time
                if elapsed > 0:
                    stats["speed"] = processed_count / elapsed

                update_stats_ui()

                progress = processed_count / len(code_files)
                try:
                    dpg.set_value("progress_bar", progress)
                    label = f"Done: {processed_count}/{len(code_files)} | OK: {stats['success']}"
                    dpg.configure_item("progress_bar", label=label)
                except:
                    pass

        elapsed = time.time() - start_time
        log(f"Codebase processing complete. Success: {stats['success']}, Failed: {stats['failed']}, Time: {elapsed:.1f}s",
            [0, 255, 200])

        if dpg.get_value("chk_sound") and HAS_WINSOUND:
            try:
                winsound.MessageBeep()
            except:
                pass

    except Exception as e:
        log(f"Critical error in worker: {e}", [255, 100, 100])
        import traceback
        log(f"Traceback: {traceback.format_exc()}", [255, 100, 100])

    finally:
        is_running = stop_requested = force_stop = False
        try:
            dpg.set_value("progress_bar", 0.0)
            dpg.configure_item("progress_bar", label="IDLE")
        except:
            pass


def select_codebase_folder():
    """Open folder selection dialog"""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Codebase Folder")
    root.destroy()

    if folder:
        dpg.set_value("codebase_folder_path", folder)
        log(f"Selected folder: {folder}")


def start_codebase_processing():
    """Start processing codebase"""
    global is_running, stop_requested, force_stop

    if is_running:
        return

    is_running = True
    stop_requested = force_stop = False
    threading.Thread(target=process_codebase_worker, daemon=True).start()


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
# INTELLIGENT CONTENT QUALITY SCORER (For NTTuner)
# ================================================================
class NTTunerContentScorer:
    """
    Intelligent content quality assessment for training data.
    Scores content based on information density, structure, and educational value.
    """

    def __init__(self):
        # High-value content indicators
        self.high_value_patterns = {
            'educational': [
                r'\b(how to|tutorial|guide|learn|explanation|introduction to|basics of)\b',
                r'\b(step \d+|first|second|third|finally|conclusion)\b',
                r'\b(example|for instance|such as|including|specifically)\b',
            ],
            'informational': [
                r'\b(definition|meaning|refers to|is defined as|consists of)\b',
                r'\b(overview|summary|description|details|information about)\b',
                r'\b(history|background|context|origin|development)\b',
            ],
            'technical': [
                r'\b(algorithm|method|technique|approach|process|procedure)\b',
                r'\b(function|variable|parameter|argument|return|output)\b',
                r'\b(implementation|usage|syntax|structure|format)\b',
            ],
            'analytical': [
                r'\b(analysis|comparison|evaluation|assessment|review)\b',
                r'\b(advantage|disadvantage|benefit|drawback|limitation)\b',
                r'\b(pros and cons|strengths and weaknesses)\b',
            ]
        }

        # Low-value content indicators
        self.low_value_patterns = {
            'navigation': [
                r'\b(home|about us|contact|privacy policy|terms of service)\b',
                r'\b(sitemap|search|menu|navigation|subscribe|newsletter)\b',
                r'\b(follow us|social media|share|tweet|facebook)\b',
            ],
            'commercial': [
                r'\b(buy now|add to cart|checkout|purchase|sale|discount)\b',
                r'\b(price|\$\d+|order now|free shipping|limited offer)\b',
                r'\b(advertisement|sponsored|promoted|affiliate)\b',
            ],
            'generic': [
                r'\b(click here|read more|learn more|see more|view all)\b',
                r'\b(loading|please wait|error|404|page not found)\b',
                r'\b(cookies|gdpr|accept|decline|agree)\b',
            ],
            'placeholder': [
                r'\b(lorem ipsum|placeholder|coming soon|under construction)\b',
                r'\b(test|debug|temp|temporary|example\.com)\b',
            ]
        }

        # Content structure indicators
        self.structure_patterns = {
            'lists': r'(?:\n\s*[-•*]\s+.+){3,}',  # Bullet points
            'numbered': r'(?:\n\s*\d+[.)]\s+.+){3,}',  # Numbered lists
            'headings': r'(?:^|\n)#+\s+.+|(?:^|\n)[A-Z][^.!?]*:',  # Headers
            'code': r'```[\s\S]*?```|`[^`]+`',  # Code blocks
            'quotes': r'(?:^|\n)\s*>.*',  # Blockquotes
        }

    def score_content(self, text: str, url: str) -> Dict[str, float]:
        """
        Score content across multiple dimensions.
        Returns dict with scores and overall quality rating.
        """
        text_lower = text.lower()

        scores = {
            'length': self._score_length(text),
            'information_density': self._score_information_density(text, text_lower),
            'structure': self._score_structure(text),
            'educational_value': self._score_educational_value(text_lower),
            'noise_level': self._score_noise_level(text_lower),
            'url_quality': self._score_url_quality(url),
        }

        # Calculate weighted overall score (0-100)
        weights = {
            'length': 0.10,
            'information_density': 0.30,
            'structure': 0.15,
            'educational_value': 0.25,
            'noise_level': 0.15,
            'url_quality': 0.05,
        }

        overall = sum(scores[k] * weights[k] for k in weights)
        scores['overall'] = overall

        # Add quality rating
        if overall >= 80:
            scores['rating'] = 'excellent'
        elif overall >= 65:
            scores['rating'] = 'good'
        elif overall >= 50:
            scores['rating'] = 'fair'
        else:
            scores['rating'] = 'poor'

        return scores

    def _score_length(self, text: str) -> float:
        """Score based on content length - more lenient for dense content"""
        length = len(text)

        # Very short content can still be valuable (tips, quick recipes)
        if length < 100:
            return 0
        elif length < 200:
            # Check if it's actually useful despite being short
            # Count instructional indicators
            useful_patterns = ['ingredients?:', 'instructions?:', 'steps?:', 'how to',
                               'tip:', 'note:', '\d+\.', 'first', 'then', 'finally']
            useful_count = sum(1 for p in useful_patterns if re.search(p, text.lower()))

            if useful_count >= 2:  # Has at least 2 instructional indicators
                return 70  # Good score for short instructional content
            return 20
        elif length < 300:
            # 200-300: Acceptable for quick tips/recipes
            return 55
        elif length < 500:
            return 65
        elif length < 800:
            return 75
        elif 800 <= length <= 5000:
            return 100  # Sweet spot
        elif 5000 < length <= 10000:
            return 85
        elif 10000 < length <= 20000:
            return 70
        else:
            return 50  # Very long content might be diluted

    def _score_information_density(self, text: str, text_lower: str) -> float:
        """Score based on information-rich content patterns"""
        score = 50  # Base score

        # Count sentences
        sentences = len(re.findall(r'[.!?]+', text))
        if sentences == 0:
            return 0

        # Words per sentence (ideal: 15-25)
        words = len(text.split())
        wps = words / sentences if sentences > 0 else 0

        if 15 <= wps <= 25:
            score += 20
        elif 10 <= wps < 15 or 25 < wps <= 35:
            score += 10
        elif wps < 5 or wps > 50:
            score -= 20

        # Check for high-value patterns
        high_value_count = 0
        for category, patterns in self.high_value_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    high_value_count += 1

        score += min(30, high_value_count * 3)

        # Penalize excessive repetition
        unique_words = len(set(text_lower.split()))
        if words > 0:
            uniqueness = unique_words / words
            if uniqueness < 0.3:
                score -= 20

        return max(0, min(100, score))

    def _score_structure(self, text: str) -> float:
        """Score based on document structure (lists, headers, formatting)"""
        score = 40  # Base score

        for structure_type, pattern in self.structure_patterns.items():
            if re.search(pattern, text):
                score += 12

        # Check for paragraph structure
        paragraphs = text.split('\n\n')
        if 2 <= len(paragraphs) <= 20:
            score += 15
        elif len(paragraphs) > 20:
            score += 5

        return min(100, score)

    def _score_educational_value(self, text_lower: str) -> float:
        """Score based on educational/tutorial content"""
        score = 30  # Base score

        # Check each category
        for category, patterns in self.high_value_patterns.items():
            matches = sum(1 for p in patterns if re.search(p, text_lower))
            if matches > 0:
                score += 15

        # Bonus for multiple categories
        categories_found = sum(
            1 for patterns in self.high_value_patterns.values()
            if any(re.search(p, text_lower) for p in patterns)
        )

        if categories_found >= 3:
            score += 20
        elif categories_found >= 2:
            score += 10

        return min(100, score)

    def _score_noise_level(self, text_lower: str) -> float:
        """Score based on noise/unwanted content (higher score = less noise)"""
        score = 100  # Start at 100, deduct for noise

        for category, patterns in self.low_value_patterns.items():
            matches = sum(1 for p in patterns if re.search(p, text_lower))
            score -= matches * 8  # Penalize each noise pattern match

        return max(0, score)

    def _score_url_quality(self, url: str) -> float:
        """Score URL structure for content quality indicators"""
        url_lower = url.lower()
        score = 50

        # VERY BAD URL patterns (instant low score)
        very_bad_patterns = [
            '/api', '/api.php', '.css', '.js', '.json', '.xml',
            '/feed', '/rss', '/wp-json/', '/graphql',
            '/css/', '/js/', '/images/', '/icons/', '/static/', '/assets/',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.pdf',
            '/privacy', '/terms', '/about', '/contact', '/faq',
            '/browse/letter', '/browse/name', '/browse/category',
        ]

        for pattern in very_bad_patterns:
            if pattern in url_lower:
                return 0  # Instant fail

        # Good URL indicators
        good_patterns = [
            '/blog/', '/article/', '/post/', '/tutorial/', '/guide/',
            '/docs/', '/documentation/', '/learn/', '/wiki/', '/news/',
            '/recipe/', '/meal/', '/ingredient/', '/category/',
            '/technique/', '/method/', '/how-to/', '/tips/',
        ]

        for pattern in good_patterns:
            if pattern in url_lower:
                score += 30
                break

        # Bad URL indicators
        bad_patterns = [
            '/tag/', '/author/', '/page/', '/search',
            '/login', '/signup', '/register', '/cart', '/checkout',
            '?page=', '?sort=', '?filter=', '&ref=', '&utm_',
            '/browse/area/', '/browse/letter/',  # Browse pages
        ]

        for pattern in bad_patterns:
            if pattern in url_lower:
                score -= 25
                break

        # Penalize very long/complex URLs
        if len(url) > 150 or url.count('?') > 1:
            score -= 15

        # Penalize URLs with too many path segments
        path_segments = url.count('/')
        if path_segments > 8:
            score -= 10

        return max(0, min(100, score))

    def should_accept(self, scores: Dict[str, float], threshold: float = 50.0) -> bool:
        """Determine if content should be accepted based on overall score"""
        return scores['overall'] >= threshold

    def get_rejection_reason(self, scores: Dict[str, float]) -> str:
        """Get human-readable reason for rejection"""
        if scores['overall'] >= 50:
            return None

        # Find weakest dimension
        dimensions = ['length', 'information_density', 'structure',
                      'educational_value', 'noise_level', 'url_quality']
        weakest = min(dimensions, key=lambda d: scores[d])

        reasons = {
            'length': 'content too short or too long',
            'information_density': 'low information density',
            'structure': 'poor structure/formatting',
            'educational_value': 'lacks educational value',
            'noise_level': 'too much noise/irrelevant content',
            'url_quality': 'URL indicates low-quality page'
        }

        return reasons.get(weakest, 'low overall quality')


# ================================================================
# SUBDOMAIN DISCOVERY ENGINE
# ================================================================
class SubdomainDiscovery:
    """Intelligent subdomain enumeration and validation"""

    def __init__(self):
        self.discovered = set()
        self.verified = set()
        self.failed = set()
        self.lock = threading.Lock()

    def extract_root_domain(self, url: str) -> str:
        """Get root domain from URL"""
        try:
            netloc = urllib.parse.urlparse(url).netloc.lower()
            # Remove port if present
            netloc = netloc.split(':')[0]
            parts = netloc.split('.')
            if len(parts) >= 2:
                return '.'.join(parts[-2:])
            return netloc
        except:
            return ""

    def generate_candidates(self, domain: str) -> List[str]:
        """Generate subdomain candidates"""
        candidates = []
        for sub in COMMON_SUBDOMAINS:
            candidates.append(f"https://{sub}.{domain}")
        return candidates

    def extract_from_html(self, html: str, base_domain: str) -> Set[str]:
        """Extract subdomains from HTML content"""
        subdomains = set()
        # Find all URLs in the HTML
        urls = re.findall(r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', html)
        for url in urls:
            if base_domain in url.lower():
                # Reconstruct full URL
                subdomains.add(f"https://{url}")
        return subdomains

    def verify_subdomain(self, url: str, timeout: float = 5.0) -> bool:
        """Quick check if subdomain exists"""
        with self.lock:
            if url in self.verified:
                return True
            if url in self.failed:
                return False

        try:
            req = urllib.request.Request(url, headers={'User-Agent': random.choice(USER_AGENTS)})
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
                with self.lock:
                    self.verified.add(url)
                return True
        except:
            with self.lock:
                self.failed.add(url)
            return False

    def discover_from_page(self, html: str, base_url: str) -> List[str]:
        """Discover new subdomains from page content"""
        root = self.extract_root_domain(base_url)
        if not root:
            return []

        found = self.extract_from_html(html, root)

        new_subdomains = []
        with self.lock:
            for sub in found:
                if sub not in self.discovered:
                    self.discovered.add(sub)
                    new_subdomains.append(sub)

        return new_subdomains

    def get_stats(self) -> Dict:
        """Get discovery statistics"""
        with self.lock:
            return {
                "discovered": len(self.discovered),
                "verified": len(self.verified),
                "failed": len(self.failed)
            }


# ================================================================
# GLOBAL STATE
# ================================================================
proxy_manager = ProxyManager()
rate_limiter = DomainRateLimiter()
crawl_queue = CrawlQueue()
subdomain_engine = SubdomainDiscovery()
content_scorer = NTTunerContentScorer()

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
    "junk_filtered": 0,
    "total_chars": 0,
    "speed": 0.0,
    "subdomains_found": 0,
    "avg_quality": 0.0,
    "quality_filtered": 0
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
            dpg.set_value("stat_junk", str(stats.get("junk_filtered", 0)))
            dpg.set_value("stat_chars", f"{stats['total_chars'] / 1000:.1f}k")
            dpg.set_value("stat_speed", f"{stats['speed']:.1f}/s")
        if dpg.does_item_exist("stat_quality"):
            dpg.set_value("stat_quality", f"{stats.get('avg_quality', 0):.0f}%")
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


def get_current_content_config():
    """Get the current content type configuration"""
    content_type = DEFAULT_CONTENT_TYPE

    try:
        if dpg.does_item_exist("content_type_combo"):
            content_type = dpg.get_value("content_type_combo")
    except:
        pass

    config = CONTENT_TYPES.get(content_type, CONTENT_TYPES["Custom"]).copy()

    # Override with custom settings if provided
    try:
        if dpg.does_item_exist("custom_sections_input"):
            custom_sections = dpg.get_value("custom_sections_input").strip()
            if custom_sections:
                config["detail_sections"] = [s.strip() for s in custom_sections.split(",") if s.strip()]

        if dpg.does_item_exist("custom_prompt_input"):
            custom_prompt = dpg.get_value("custom_prompt_input").strip()
            if custom_prompt:
                config["user_prompt_template"] = custom_prompt
    except:
        pass

    return config


def on_content_type_change(sender, app_data):
    """Callback when content type changes"""
    content_type = app_data
    config = CONTENT_TYPES[content_type]

    # Update system prompt suggestion
    if dpg.does_item_exist("system_prompt_input"):
        current_system = dpg.get_value("system_prompt_input").strip()
        # Only auto-update if it's empty or matches a known preset
        if not current_system or current_system in SYSTEM_PROMPTS.values():
            dpg.set_value("system_prompt_input", config["system_prompt"])

    # Show/hide custom fields
    is_custom = content_type == "Custom"
    if dpg.does_item_exist("custom_fields_group"):
        dpg.configure_item("custom_fields_group", show=is_custom)

    # Update preview
    if dpg.does_item_exist("prompt_preview_text"):
        example = config["example_titles"][0] if config["example_titles"] and config["example_titles"][
            0] else "Example Title"
        preview = config["user_prompt_template"].format(title=example)
        dpg.set_value("prompt_preview_text", f"Preview: {preview}")

    log(f"Content type changed to: {content_type}")


def format_assistant_response(title, content, sections):
    """
    Format the assistant's response with optional sections
    """
    response_parts = [title]

    if sections:
        # Add section markers - in future could use ML to extract actual sections
        response_parts.append(content)
    else:
        response_parts.append(content)

    return "\n".join(response_parts)


def build_text(system_override: str, user_content: str, template_key: str, content_type_config: dict = None) -> str:
    """Build formatted training text using content type configuration"""
    tpl = MODEL_TEMPLATES.get(template_key, MODEL_TEMPLATES[DEFAULT_TEMPLATE_KEY])
    system_text = system_override.strip()

    text_out = tpl.get("begin", "")

    if system_text and "system" in tpl:
        text_out += tpl["system"].format(system=system_text)

    # Get content type configuration
    if content_type_config is None:
        content_type_config = CONTENT_TYPES[DEFAULT_CONTENT_TYPE]

    # Extract title from content - IMPROVED LOGIC
    title = "Untitled Content"
    content_lines = user_content.split('\n')

    if content_lines:
        # Try to find the first substantial line as title
        for line in content_lines[:5]:  # Check first 5 lines
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip obvious non-title patterns
            if line.lower().startswith(('recipe api', 'themealdb', 'ingredient api')):
                continue
            if line.startswith(('http://', 'https://', 'www.')):
                continue
            if len(line) < 3:  # Too short
                continue
            if len(line) > 150:  # Too long
                continue

            # This looks like a potential title
            title = line
            break

    # Format user prompt based on content type
    user_prompt = content_type_config["user_prompt_template"].format(title=title)

    if "user" in tpl:
        text_out += tpl["user"].format(user=user_prompt)

    if "assistant" in tpl:
        # Clean up the content before adding
        cleaned_content = user_content.strip()
        
        # Newlines are already correct from extract_text_content()

        # Remove common noise patterns from the start
        lines = cleaned_content.split('\n')
        cleaned_lines = []
        skip_count = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Skip the first occurrence of site branding
            if i < 3 and any(x in line_stripped.lower() for x in ['recipe api', 'themealdb', '- free']):
                skip_count += 1
                continue

            # Skip repeated title (often happens)
            if i > 0 and skip_count < 2 and line_stripped == title:
                skip_count += 1
                continue

            cleaned_lines.append(line)

        cleaned_content = '\n'.join(cleaned_lines)

        # Format the final response
        text_out += tpl["assistant"].format(assistant=cleaned_content)

    # CRITICAL FIX: Ensure all newlines are actual newlines, not escaped strings\n    # This prevents the model from learning literal \\n patterns\n    text_out = text_out.replace('\\\\n', '\n')  # Fix any double-escaped newlines\n    return text_out


# ================================================================
# HTTP FETCHER (using working urllib approach)
# ================================================================
def extract_text_content(html: str, url: str, config: Dict) -> Tuple[str, List[str]]:
    """
    Intelligent content extraction - extracts ONLY main content, filters noise.
    Returns (clean_text, links)
    """
    # Extract links first (before cleaning)
    found_links = []
    if config.get('crawler_enabled', False):
        found_links = extract_links(html, url)

    # Step 1: Remove entire sections that are pure noise
    noise_sections = [
        r'<nav[^>]*>.*?</nav>',
        r'<header[^>]*>.*?</header>',
        r'<footer[^>]*>.*?</footer>',
        r'<aside[^>]*>.*?</aside>',
        r'<script[^>]*>.*?</script>',
        r'<style[^>]*>.*?</style>',
        r'<iframe[^>]*>.*?</iframe>',
        r'<noscript[^>]*>.*?</noscript>',
        r'<svg[^>]*>.*?</svg>',
        r'<!--.*?-->',
    ]

    for pattern in noise_sections:
        html = re.sub(pattern, ' ', html, flags=re.I | re.S)

    # Step 2: Remove common noise patterns
    noise_patterns = [
        r'<button[^>]*>.*?</button>',
        r'<select[^>]*>.*?</select>',
        r'<input[^>]*>',
        r'<form[^>]*>.*?</form>',
    ]

    for pattern in noise_patterns:
        html = re.sub(pattern, ' ', html, flags=re.I | re.S)

    # Step 3: Try to extract main content area
    main_content_patterns = [
        r'<main[^>]*>(.*?)</main>',
        r'<article[^>]*>(.*?)</article>',
        r'<div[^>]*class=["\'][^"\']*content[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]*id=["\']content["\'][^>]*>(.*?)</div>',
    ]

    extracted_content = None
    for pattern in main_content_patterns:
        matches = re.findall(pattern, html, flags=re.I | re.S)
        if matches:
            extracted_content = max(matches, key=len)
            break

    if extracted_content and len(extracted_content) > 200:
        html = extracted_content

    # Step 4: Preserve structure before removing tags
    # Convert headers to text markers
    html = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n\1\n', html, flags=re.I)

    # Convert lists to formatted text
    html = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', html, flags=re.I)

    # Convert paragraphs to double newlines
    html = re.sub(r'</p>', r'\n\n', html, flags=re.I)
    html = re.sub(r'<br\s*/?>', r'\n', html, flags=re.I)

    # Step 5: Remove code blocks if configured
    if config.get('clean_code', False):
        html = re.sub(r"<pre[^>]*>.*?</pre>", "", html, flags=re.I | re.S)
        html = re.sub(r"<code[^>]*>.*?</code>", "", html, flags=re.I | re.S)

    # Step 6: Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Step 7: Decode HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&mdash;', '—')
    text = text.replace('&ndash;', '–')

    # Step 8: Remove common noise text patterns
    noise_text_patterns = [
        r'Recipe API - TheMealDB',
        r'- TheMealDB\.com',
        r'TheMealDB\.com',
        r'No Tags',
        r'Tags:?\s*$',
        r'Category:?\s*$',
        r'Skip to (?:main )?content',
        r'(?:Toggle|Open|Close) (?:navigation|menu)',
        r'(?:Share|Like|Tweet|Pin) (?:this|on)',
        r'Subscribe to (?:our )?newsletter',
        r'Follow us on (?:Facebook|Twitter|Instagram)',
        r'Copyright \d{4}',
        r'All rights reserved',
        r'Terms (?:of Service|and Conditions)',
        r'Privacy Policy',
        r'Cookie (?:Policy|Settings)',
        r'Click here to',
        r'Loading\.\.\.',
        r'Please enable JavaScript',
        r'Browse (?:all |More)?[A-Z\s/]*',
        r'Browse (?:By Name|Country|Category)',
        r'Latest Meals?',
        r'Popular Ingredients?',
        r'Random (?:Meals?|Ingredients?)',
        r'Welcome to \w+',
        r'\d+ premium supporters?',
    ]

    for pattern in noise_text_patterns:
        text = re.sub(pattern, '', text, flags=re.I)

    # Step 9: Better organization - add section headers if missing
    lines = text.split('\n')
    organized_lines = []
    in_ingredients = False
    in_instructions = False

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Detect ingredient sections (lines with measurements)
        is_ingredient = bool(
            re.search(r'\d+\s*(cup|tsp|tbsp|oz|lb|g|kg|ml|l|medium|large|small|pinch|dash)', line, re.I))

        # Detect instruction sections (lines starting with verbs or numbers)
        is_instruction = bool(
            re.search(r'^(\d+[\.)]\s+|Mix|Add|Stir|Bake|Cook|Heat|Pour|Blend|Combine|Place|Put)', line, re.I))

        # Add section headers
        if is_ingredient and not in_ingredients and not in_instructions:
            if organized_lines and organized_lines[-1] != '':
                organized_lines.append('')
            organized_lines.append('Ingredients:')
            in_ingredients = True
            in_instructions = False
        elif is_instruction and not in_instructions:
            if organized_lines and organized_lines[-1] != '':
                organized_lines.append('')
            organized_lines.append('Instructions:')
            in_instructions = True
            in_ingredients = False

        # Format ingredient lines
        if in_ingredients and is_ingredient:
            if not line.startswith('-'):
                line = '- ' + line

        # Format instruction lines
        if in_instructions and is_instruction:
            # Make sure numbered instructions are formatted consistently
            line = re.sub(r'^(\d+)\.?\s*', r'\1. ', line)

        organized_lines.append(line)

    text = '\n'.join(organized_lines)

    # Step 9: Clean whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)

    # Step 10: Remove very short lines that are likely navigation
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip lines that are too short or look like navigation
        if len(line) < 3:
            continue
        # Skip single-word lines unless they're part of a list
        words = line.split()
        if len(words) == 1 and not line.startswith('-'):
            continue
        filtered_lines.append(line)

    text = '\n'.join(filtered_lines)

    return text[:200000], found_links


def extract_links(html: str, base_url: str) -> List[str]:
    """Extract all links from HTML, converting relative URLs to absolute"""
    links = set()

    # Extract absolute URLs (http:// or https://)
    absolute_pattern = r'href=["\'](https?://[^"\']+)["\']'
    for match in re.findall(absolute_pattern, html, re.IGNORECASE):
        if match != base_url:
            links.add(match)

    # Extract relative URLs (starting with /)
    relative_pattern = r'href=["\'](/[^"\']*?)["\']'
    for match in re.findall(relative_pattern, html, re.IGNORECASE):
        # Convert relative to absolute
        absolute_url = urllib.parse.urljoin(base_url, match)
        if absolute_url != base_url:
            links.add(absolute_url)

    # Extract protocol-relative URLs (//example.com/path)
    protocol_relative_pattern = r'href=["\'](//[^"\']+)["\']'
    for match in re.findall(protocol_relative_pattern, html, re.IGNORECASE):
        # Add protocol from base_url
        parsed_base = urllib.parse.urlparse(base_url)
        absolute_url = f"{parsed_base.scheme}:{match}"
        if absolute_url != base_url:
            links.add(absolute_url)

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

            # Discover subdomains if enabled
            if config.get('discover_subdomains', False):
                new_subdomains = subdomain_engine.discover_from_page(raw_data, url)
                if new_subdomains:
                    log(f"  [SUBDOMAIN] Discovered {len(new_subdomains)} subdomains from page", [150, 200, 255])

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


def validate_content_quality(content: str, url: str) -> Tuple[bool, str]:
    """
    Enhanced validation to detect and reject junk pages, list pages, and non-recipe content.
    OPTIMIZED FOR THEMEALDB - prioritizes /meal/ URLs and uses more lenient thresholds.
    Returns (is_valid, reason_if_invalid)
    """
    content_lower = content.lower()
    url_lower = url.lower()

    # === CRITICAL FIX 1: PRIORITY BYPASS for /meal/ URLs ===
    # Always accept /meal/ URLs - these are actual recipe pages on TheMealDB
    if '/meal/' in url_lower:
        return True, ""

    # Check 1: Detect FAQ/About/Privacy pages by keywords
    junk_keywords = [
        'privacy policy', 'terms of service', 'terms of use', 'cookie policy',
        'about us', 'contact us', 'gdpr', 'data protection', 'email protected',
        'all rights reserved', 'copyright', '© 20', 'terms and conditions',
        'patreon supporter', 'api key', 'how do i add', 'sign up on patreon'
    ]

    junk_count = sum(1 for kw in junk_keywords if kw in content_lower)
    # === FIX 2: Increased threshold from 2 to 3 to reduce false positives ===
    if junk_count >= 3:
        return False, "Junk page (FAQ/About/Privacy)"

    # Check 2: Detect list/index pages (lots of recipe names, no instructions)
    lines = content.split('\n')
    non_empty_lines = [l.strip() for l in lines if l.strip()]

    # === FIX 3: Increased threshold from 15 to 20 ===
    if len(non_empty_lines) > 20:
        # Count lines that look like recipe titles (short, capitalized)
        title_like_lines = 0
        for line in non_empty_lines[:30]:  # Check first 30 lines
            # Recipe title pattern: 15-60 chars, starts with capital, no numbers at start
            if (15 < len(line) < 60 and
                    line[0].isupper() and
                    not line[0].isdigit() and
                    line.count('\n') == 0):
                title_like_lines += 1

        # === FIX 4: Increased threshold from 60% to 70% ===
        if title_like_lines > len(non_empty_lines[:30]) * 0.85:
            return False, "List/index page (many recipe names)"

    # Check 3: Detect recipe content indicators
    recipe_indicators = {
        'ingredients': ['ingredient', 'cup', 'tablespoon', 'teaspoon', 'tsp', 'tbsp',
                        'oz', 'lb', 'gram', 'kg', 'ml', 'liter', 'pinch', 'dash', 'clove'],
        'instructions': ['cook', 'bake', 'mix', 'stir', 'heat', 'add', 'pour', 'chop',
                         'dice', 'slice', 'preheat', 'simmer', 'boil', 'fry', 'roast',
                         'step 1', 'step 2', 'step', 'minutes', 'until', 'serve'],
    }

    ingredient_count = sum(1 for word in recipe_indicators['ingredients'] if word in content_lower)
    instruction_count = sum(1 for word in recipe_indicators['instructions'] if word in content_lower)

    # === FIX 5: More lenient thresholds and smart filtering ===
    # Reduced from ingredient_count < 3 and instruction_count < 5
    # to ingredient_count < 2 and instruction_count < 3
    if ingredient_count < 2 and instruction_count < 3:
        # Only reject if it's a known browse/category page
        if any(pat in url_lower for pat in ['/browse/', '/area/', '/category/', '/tag/', '/letter/']):
            return False, "Missing recipe indicators (browse/filter page)"

    # Check 4: Detect ingredient/category description pages

    # Check 5: Number density check (recipes have measurements/times)
    number_count = len(re.findall(r'\d+', content))
    words = len(content.split())

    # === FIX 6: More lenient word count threshold (100 -> 150) ===
    if words > 150:
        number_density = number_count / words
        # === FIX 7: Lower threshold from 0.02 to 0.015 ===
        if number_density < 0.015:
            return False, "Too few numbers (likely not a recipe)"

    # Check 6: Sentence structure (lists have many short lines)
    avg_line_length = sum(len(l) for l in non_empty_lines) / max(len(non_empty_lines), 1)
    # === FIX 8: More lenient thresholds (25 -> 20, 20 -> 25) ===
    if avg_line_length < 20 and len(non_empty_lines) > 25:
        return False, "Too many short lines (likely list page)"

    return True, ""


def process_url_v2(url: str, depth: int, config: Dict) -> Dict:
    """
    Process a single URL. Returns dict with results.
    """
    if stop_requested or force_stop:
        return {'success': False, 'error': None, 'chars': 0, 'links': [], 'filtered': True}

    # URL pattern filtering - skip non-content URLs
    skip_url_patterns = [
        r'/api[/\.]',  # API endpoints
        r'/api\.php',  # API files
        r'\.css',  # CSS files
        r'\.js',  # JavaScript files
        r'\.json',  # JSON files
        r'\.xml',  # XML files
        r'\.jpg', r'\.jpeg', r'\.png', r'\.gif', r'\.svg', r'\.webp',  # Images
        r'\.pdf', r'\.zip', r'\.tar', r'\.gz',  # Documents/Archives
        r'/feed', r'/rss',  # RSS feeds
        r'/wp-json/',  # WordPress API
        r'/graphql',  # GraphQL endpoints
        r'\?format=json',  # JSON format queries
        r'/css/',  # CSS directories
        r'/js/',  # JS directories
        r'/images/',  # Image directories
        r'/icons/',  # Icon directories
        r'/fonts/',  # Font directories
        r'/static/',  # Static assets
        r'/assets/',  # Asset directories
    ]

    url_lower = url.lower()

    # === THEMEALDB SMART FILTERING ===
    # Block browse/navigation pages that waste crawl budget
    browse_patterns = [
        # r'/browse/letter/',  # Alphabetical browse (COMMENTED - allow as seeds)
        # r'/browse/area/',  # Area browse (COMMENTED - these are valuable recipe list pages)
        # r'/browse/category/',  # Category browse (COMMENTED - these are valuable recipe list pages)
        r'/api\.php',  # API page
        r'/privacy',  # Privacy pages
        r'/terms',  # Terms pages
        r'/contact',  # Contact pages
        r'/refunds',  # Refunds pages
    ]

    for pattern in browse_patterns:
        if re.search(pattern, url_lower):
            log(f"  [SKIP-URL] Browse/nav page: {url[:60]}...", [150, 150, 150])
            return {'success': False, 'error': None, 'chars': 0, 'links': [], 'filtered': True}

    for pattern in skip_url_patterns:
        if re.search(pattern, url_lower):
            log(f"  [SKIP-URL] Non-content URL: {url[:60]}...", [150, 150, 150])
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

    # SMART CONTENT VALIDATION - Skip junk pages and lists
    is_valid, invalid_reason = validate_content_quality(content, url)
    if not is_valid:
        log(f"  [SKIP-JUNK] {invalid_reason}: {url[:50]}...", [200, 150, 100])
        stats['junk_filtered'] += 1
        return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

    # Log links found
    if config.get('crawler_enabled', False) and links:
        log(f"  [LINKS] Found {len(links)} links on page", [120, 180, 220])

    # Apply basic filters first
    content_len = len(content)
    text_lower = content.lower()
    quality_scores = None  # Initialize to None

    min_chars = config.get('min_chars', 150)
    max_chars = config.get('max_chars', 50000)
    allow_short_quality = config.get('allow_short_quality', False)
    short_min_chars = config.get('short_min_chars', 100)
    ignore_quality_short = config.get('ignore_quality_short', False)

    # Length check with bypass options
    if content_len < min_chars:
        # Check if content meets the absolute minimum for short content
        if content_len < short_min_chars:
            log(f"  [!] Too short ({content_len}, min={short_min_chars}): {url[:40]}...", [150, 150, 100])
            return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

        # Content is between short_min_chars and min_chars
        # Apply short content rules
        if ignore_quality_short:
            # Accept ALL short content without quality check
            log(f"  [SHORT-ACCEPT] Accepted short content ({content_len} chars, quality ignored): {url[:40]}...",
                [100, 255, 100])
            quality_scores = content_scorer.score_content(content, url)  # Calculate for stats only
        elif allow_short_quality:
            # Accept short content only if high quality
            quality_scores = content_scorer.score_content(content, url)
            if quality_scores['overall'] >= 70:
                log(f"  [SHORT-OK] Short but high quality ({content_len} chars, Q{quality_scores['overall']:.0f}): {url[:40]}...",
                    [100, 200, 255])
            else:
                log(f"  [!] Too short ({content_len} chars, Q{quality_scores['overall']:.0f}): {url[:40]}...",
                    [150, 150, 100])
                return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}
        else:
            # Reject short content (neither option enabled)
            log(f"  [!] Too short ({content_len} chars, min={min_chars}): {url[:40]}...", [150, 150, 100])
            return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

    if content_len > max_chars:
        content = content[:max_chars]
        content_len = max_chars

    # Apply keyword filters
    kw_in = config.get('keywords_in', [])
    kw_out = config.get('keywords_out', [])

    if kw_in and not any(k in text_lower for k in kw_in):
        log(f"  [!] Missing required keywords: {url[:40]}...", [150, 150, 100])
        return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

    if kw_out and any(k in text_lower for k in kw_out):
        log(f"  [!] Contains excluded keywords: {url[:40]}...", [150, 150, 100])
        return {'success': False, 'error': None, 'chars': 0, 'links': links, 'filtered': True}

    # Quality scoring (if enabled and not already scored from short content check)
    if not quality_scores and config.get('use_quality_filter', False):
        quality_scores = content_scorer.score_content(content, url)
        quality_threshold = config.get('quality_threshold', 50.0)

        if not content_scorer.should_accept(quality_scores, quality_threshold):
            reason = content_scorer.get_rejection_reason(quality_scores)
            log(f"  [SKIP-Q] Quality: {quality_scores['overall']:.0f}/100 ({reason}): {url[:40]}...",
                [200, 150, 100])
            stats['quality_filtered'] += 1
            return {'success': False, 'error': None, 'chars': 0, 'links': links,
                    'filtered': True, 'quality': quality_scores['overall']}

    # Build training text
    system_prompt = config.get('system_prompt', '')
    template_key = config.get('template', DEFAULT_TEMPLATE_KEY)
    content_type_cfg = config.get('content_type_config', CONTENT_TYPES[DEFAULT_CONTENT_TYPE])
    final_text = build_text(system_prompt, content, template_key, content_type_cfg)

    # Write to file (thread-safe)
    with write_lock:
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"text": final_text}, ensure_ascii=False) + "\n")
        except Exception as e:
            log(f"  [!] Write error: {e}", [255, 100, 100])
            return {'success': False, 'error': str(e), 'chars': 0, 'links': [], 'filtered': False}

    # Log success with quality score
    if quality_scores:
        quality_str = f"Q{quality_scores['overall']:.0f}"
        rating_colors = {
            'excellent': [50, 255, 50],
            'good': [100, 255, 100],
            'fair': [200, 200, 100],
            'poor': [255, 150, 50]
        }
        color = rating_colors.get(quality_scores['rating'], [100, 255, 150])
        log(f"  [+] [D{depth}] [{quality_str}] {url[:50]}... ({content_len} chars)", color)
    else:
        log(f"  [+] [D{depth}] {url[:50]}... ({content_len} chars)", [0, 255, 150])

    # Return quality score for stats
    quality_val = quality_scores['overall'] if quality_scores else 0
    return {'success': True, 'error': None, 'chars': content_len, 'links': links,
            'filtered': False, 'quality': quality_val}


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
        "allow_short_quality": dpg.get_value("chk_allow_short_quality"),
        "short_min_chars": dpg.get_value("inp_short_min_chars"),
        "ignore_quality_short": dpg.get_value("chk_ignore_quality_short"),
        "crawler_enabled": dpg.get_value("chk_crawler"),
        "max_depth": dpg.get_value("inp_max_depth"),
        "links_per_page": dpg.get_value("inp_links_per_page"),
        "max_per_domain": dpg.get_value("inp_max_per_domain"),
        "same_domain": dpg.get_value("chk_same_domain"),
        "prioritize_content": dpg.get_value("chk_prioritize_content"),
        "clean_code": dpg.get_value("chk_clean_code"),
        "clean_whitespace": dpg.get_value("chk_clean_whitespace"),
        "discover_subdomains": dpg.get_value("chk_subdomain_discovery"),
        "use_quality_filter": dpg.get_value("chk_quality_filter"),
        "quality_threshold": dpg.get_value("inp_quality_threshold"),
        "content_type_config": get_current_content_config(),  # Add content type config
    }

    # Set rate limiter
    rate_limiter.set_delay(dpg.get_value("inp_domain_delay"))

    # Workers
    num_workers = dpg.get_value("inp_workers")
    stop_limit = dpg.get_value("inp_stop_limit")

    # Initialize crawl queue with seed domains
    crawl_queue.set_seed_domains(initial_urls)

    # Subdomain Discovery Phase
    if config.get('discover_subdomains', False):
        log("[*] Subdomain Discovery Enabled - Scanning...", [150, 200, 255])

        all_discovered_subdomains = []
        for seed_url in initial_urls:
            root_domain = subdomain_engine.extract_root_domain(seed_url)
            if not root_domain:
                continue

            log(f"  Scanning {root_domain} for subdomains...", [120, 180, 220])
            candidates = subdomain_engine.generate_candidates(root_domain)

            # Verify subdomains (in parallel for speed)
            verified_count = 0
            with ThreadPoolExecutor(max_workers=10) as verifier:
                verification_futures = {verifier.submit(subdomain_engine.verify_subdomain, candidate): candidate
                                        for candidate in candidates}

                for future in as_completed(verification_futures):
                    candidate = verification_futures[future]
                    try:
                        if future.result():
                            all_discovered_subdomains.append(candidate)
                            verified_count += 1
                            stats['subdomains_found'] += 1
                            log(f"    [+] Found: {candidate}", [100, 255, 150])
                    except:
                        pass

            log(f"  [OK] Discovered {verified_count} subdomains for {root_domain}", [100, 255, 100])

        # Add verified subdomains to crawl queue
        for subdomain in all_discovered_subdomains:
            crawl_queue.push(subdomain, 0, config)

        if all_discovered_subdomains:
            log(f"[SUBDOMAIN] Total: {len(all_discovered_subdomains)} subdomains added to queue", [100, 255, 200])
        else:
            log("  No additional subdomains found", [150, 150, 160])

    # Add seed URLs to queue
    for url in initial_urls:
        crawl_queue.push(url, 0, config)

    log(f"Engine Online. Seeds: {len(initial_urls)}, Workers: {num_workers}", [100, 200, 255])
    log(f"Seed domains tracked: {sorted(crawl_queue.seed_domains)}", [120, 180, 220])
    if config['crawler_enabled']:
        log(f"Crawler: Depth={config['max_depth']}, Links/Page={config['links_per_page']}, PerDomain={config['max_per_domain']}",
            [100, 200, 255])
        log(f"         SameDomain={config['same_domain']}, Prioritize={config['prioritize_content']}", [100, 200, 255])
    else:
        log(f"[WARNING] Crawler is DISABLED - will only process seed URLs!", [255, 200, 100])

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
                    quality = result.get('quality', 0)

                    if success:
                        stats["success"] += 1
                        stats["total_chars"] += char_count

                        # Track quality
                        if quality > 0:
                            current_avg = stats.get("avg_quality", 0)
                            count = stats["success"]
                            stats["avg_quality"] = ((current_avg * (count - 1)) + quality) / count
                    else:
                        if error:
                            stats["failed"] += 1
                        else:
                            stats["skipped"] += 1

                    # Add discovered links to crawl queue (even from filtered/browse pages)
                    if config['crawler_enabled'] and new_links and (depth + 1) < config['max_depth']:
                        added = crawl_queue.push_many(new_links, depth + 1, config)
                        if added > 0:
                            log(f"    [>] +{added} links queued (depth {depth + 1}, max={config['max_depth']})",
                                [100, 180, 255])

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

            # Force stop check - cancel all pending futures
            if force_stop:
                log("[FORCE STOP] Cancelling remaining tasks...", [255, 100, 100])
                for future in list(futures.keys()):
                    future.cancel()
                break

    # Finish
    elapsed = time.time() - start_time
    queue_stats = crawl_queue.get_stats()
    subdomain_stats = subdomain_engine.get_stats()

    if force_stop:
        log(f"Engine FORCE STOPPED. Processed: {processed_count}, Success: {stats['success']}, Time: {elapsed:.1f}s",
            [255, 100, 100])
    else:
        log(f"Engine Offline. Success: {stats['success']}, Failed: {stats['failed']}, Time: {elapsed:.1f}s",
            [0, 255, 200])

    log(f"Crawl Summary: Discovered {queue_stats['seen']} URLs across {queue_stats['domains']} domains", [0, 180, 255])

    if stats['subdomains_found'] > 0:
        log(f"Subdomain Discovery: Found {stats['subdomains_found']} subdomains ({subdomain_stats['verified']} verified, {subdomain_stats['failed']} unreachable)",
            [150, 200, 255])

    if config.get('use_quality_filter', False):
        avg_quality = stats.get('avg_quality', 0)
        quality_filtered = stats.get('quality_filtered', 0)
        log(f"Quality Stats: Average {avg_quality:.1f}/100 | Filtered {quality_filtered} low-quality pages",
            [255, 200, 50])

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
    global stop_requested, force_stop, last_stop_press, is_running

    if not is_running:
        return

    if time.time() - last_stop_press < 1.5:
        # Force stop - cancel all futures
        force_stop = True
        log("[FORCE STOP] Cancelling all tasks...", [255, 50, 50])
    else:
        stop_requested = True
        log("[STOPPING] Graceful stop... (click again within 1.5s to force)", [255, 200, 50])

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
    # Add content type settings
    if dpg.does_item_exist("content_type_combo"):
        config["content_type"] = dpg.get_value("content_type_combo")
    if dpg.does_item_exist("custom_prompt_input"):
        config["custom_prompt"] = dpg.get_value("custom_prompt_input")
    if dpg.does_item_exist("custom_sections_input"):
        config["custom_sections"] = dpg.get_value("custom_sections_input")

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

        # Load content type settings
        if "content_type" in config and dpg.does_item_exist("content_type_combo"):
            dpg.set_value("content_type_combo", config["content_type"])
            on_content_type_change(None, config["content_type"])  # Trigger update
        if "custom_prompt" in config and dpg.does_item_exist("custom_prompt_input"):
            dpg.set_value("custom_prompt_input", config["custom_prompt"])
        if "custom_sections" in config and dpg.does_item_exist("custom_sections_input"):
            dpg.set_value("custom_sections_input", config["custom_sections"])

        log("Config loaded", [0, 255, 100])
    except Exception as e:
        log(f"Load failed: {e}", [255, 100, 100])



# ================================================================
# OUTPUT FORMATTING UTILITIES
# ================================================================
def convert_jsonl_to_readable(input_file: str = None):
    """
    Convert JSONL file to human-readable text format.
    Creates a new file with '_readable.txt' suffix.
    """
    global output_file
    
    if input_file is None:
        input_file = output_file
    
    if not os.path.exists(input_file):
        log(f"[ERROR] File not found: {input_file}", [255, 100, 100])
        return None
    
    base, ext = os.path.splitext(input_file)
    readable_file = f"{base}_readable.txt"
    
    try:
        entry_count = 0
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(readable_file, 'w', encoding='utf-8') as f_out:
                for line_num, line in enumerate(f_in, 1):
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        # Write with proper formatting
                        f_out.write(f"{'='*80}\n")
                        f_out.write(f"Entry {line_num}\n")
                        f_out.write(f"{'='*80}\n")
                        f_out.write(text)
                        f_out.write(f"\n\n")
                        entry_count += 1
                    except json.JSONDecodeError:
                        continue
        
        log(f"[OK] Created readable output: {readable_file} ({entry_count} entries)", [100, 255, 100])
        return readable_file
    except Exception as e:
        log(f"[ERROR] Failed to convert: {e}", [255, 100, 100])
        return None


def preview_first_entry():
    """Preview first entry with proper formatting"""
    global output_file
    
    if not os.path.exists(output_file):
        log(f"[ERROR] File not found: {output_file}", [255, 100, 100])
        return
        
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            if line:
                data = json.loads(line)
                text = data.get('text', '')
                log(f"\n{'='*70}\nPreview of First Entry\n{'='*70}", [100, 200, 255])
                # Show first 2000 chars
                preview = text[:2000]
                for line in preview.split('\n')[:40]:
                    log(line, [200, 200, 200])
                if len(text) > 2000:
                    log(f"\n... ({len(text)-2000} more characters)", [150, 150, 150])
            else:
                log("[INFO] Output file is empty", [255, 200, 100])
    except Exception as e:
        log(f"[ERROR] Failed to preview: {e}", [255, 100, 100])


def filter_existing_dataset():
    """Filter existing JSONL dataset to remove junk/list pages"""
    global output_file

    if not os.path.exists(output_file):
        log(f"No dataset found at: {output_file}", [255, 100, 100])
        return

    log(f"Filtering dataset: {output_file}", [255, 200, 100])

    def worker():
        try:
            # Read all entries
            entries = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass

            log(f"Read {len(entries)} total entries", [150, 200, 255])

            # Filter entries
            good_entries = []
            filtered_reasons = {}

            for entry in entries:
                text = entry.get('text', '')

                # Extract content from the formatted text
                # Find the assistant's response
                if '<|start_header_id|>assistant<|end_header_id|>' in text:
                    content = text.split('<|start_header_id|>assistant<|end_header_id|>')[1]
                    content = content.split('<|eot_id|>')[0].strip()
                else:
                    content = text

                # Validate content
                is_valid, reason = validate_content_quality(content, entry.get('url', ''))

                if is_valid:
                    good_entries.append(entry)
                else:
                    filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1

            # Create backup
            backup_file = output_file.replace('.jsonl', '_backup.jsonl')
            import shutil
            shutil.copy(output_file, backup_file)
            log(f"Backup saved: {backup_file}", [100, 200, 100])

            # Write filtered data
            filtered_file = output_file.replace('.jsonl', '_filtered.jsonl')
            with open(filtered_file, 'w', encoding='utf-8') as f:
                for entry in good_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            # Log results
            log(f"✓ Filtered dataset saved: {filtered_file}", [100, 255, 100])
            log(f"  Kept: {len(good_entries)}/{len(entries)} ({len(good_entries) / len(entries) * 100:.1f}%)",
                [100, 255, 150])
            log(f"  Removed: {len(entries) - len(good_entries)}", [255, 150, 100])

            # Log reasons
            if filtered_reasons:
                log("Filtering breakdown:", [150, 200, 255])
                for reason, count in sorted(filtered_reasons.items(), key=lambda x: x[1], reverse=True):
                    log(f"  - {reason}: {count}", [150, 150, 200])

        except Exception as e:
            log(f"Filter error: {e}", [255, 100, 100])

    threading.Thread(target=worker, daemon=True).start()


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
            dpg.add_text("OK:", color=[0, 255, 0]);
            dpg.add_text("0", tag="stat_success")
            dpg.add_text("  Fail:", color=[255, 50, 50]);
            dpg.add_text("0", tag="stat_failed")
            dpg.add_text("  Skip:", color=[255, 200, 0]);
            dpg.add_text("0", tag="stat_skipped")
            dpg.add_text("  Junk:", color=[255, 150, 0]);
            dpg.add_text("0", tag="stat_junk")
            dpg.add_text("  Vol:", color=[0, 200, 255]);
            dpg.add_text("0k", tag="stat_chars")
            dpg.add_text("  Speed:", color=[200, 100, 255]);
            dpg.add_text("0/s", tag="stat_speed")
            dpg.add_text("  Quality:", color=[255, 200, 50]);
            dpg.add_text("0%", tag="stat_quality")

    # === CODEBASE MODE - NEW FEATURE ===
    with dpg.collapsing_header(label="Codebase Dataset Builder", default_open=False):
        dpg.add_text("BUILD DATASET FROM CODE FOLDER", color=[255, 200, 100])
        dpg.add_text("Recursively scan a codebase folder and create training data from code files",
                     color=[150, 150, 160])

        dpg.add_spacer(height=8)
        dpg.add_text("FOLDER PATH", color=[150, 150, 160])

        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="codebase_folder_path", width=-100,
                               hint="Select folder containing code...")
            dpg.add_button(label="Browse...", callback=select_codebase_folder, width=90)

        dpg.add_spacer(height=8)
        dpg.add_text("SUPPORTED LANGUAGES", color=[150, 150, 160])
        dpg.add_text("Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Rust, Ruby, PHP,",
                     color=[100, 100, 120])
        dpg.add_text("Swift, Kotlin, Scala, R, Shell, SQL, HTML, CSS, and 30+ more...",
                     color=[100, 100, 120])

        dpg.add_spacer(height=10)

        codebase_btn = dpg.add_button(label="START CODEBASE PROCESSING",
                                      callback=start_codebase_processing,
                                      width=250, height=35)
        with dpg.theme() as codebase_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, [40, 80, 120])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [60, 100, 150])
        dpg.bind_item_theme(codebase_btn, codebase_theme)

        dpg.add_spacer(height=8)
        dpg.add_text("FEATURES:", color=[150, 200, 255])
        dpg.add_text("  • Automatically skips .git, node_modules, __pycache__, build folders",
                     color=[100, 100, 120])
        dpg.add_text("  • Extracts functions, classes, and code structure",
                     color=[100, 100, 120])
        dpg.add_text("  • Multi-threaded processing for speed",
                     color=[100, 100, 120])
        dpg.add_text("  • Uses same output file and settings as web scraping",
                     color=[100, 100, 120])

        dpg.add_spacer(height=8)
        dpg.add_text("TIP: Set Content Type to 'Code File' for best results",
                     color=[255, 200, 100])

    # === CONTENT TYPE ===
    with dpg.collapsing_header(label="Content Type Configuration", default_open=False):
        dpg.add_text("CONTENT TYPE", color=[255, 200, 100])
        dpg.add_text("Select what type of content you're scraping:", color=[150, 150, 160])
        dpg.add_spacer(height=5)

        dpg.add_combo(
            list(CONTENT_TYPES.keys()),
            label="Content Type",
            tag="content_type_combo",
            default_value=DEFAULT_CONTENT_TYPE,
            callback=on_content_type_change,
            width=300
        )

        dpg.add_spacer(height=8)
        dpg.add_text("PROMPT PREVIEW", color=[150, 150, 160])
        example_title = CONTENT_TYPES[DEFAULT_CONTENT_TYPE]["example_titles"][0]
        preview = CONTENT_TYPES[DEFAULT_CONTENT_TYPE]["user_prompt_template"].format(title=example_title)
        dpg.add_text(f"Preview: {preview}", tag="prompt_preview_text", color=[100, 255, 150])

        dpg.add_spacer(height=10)

        # Section info
        with dpg.collapsing_header(label="About Sections", default_open=False):
            dpg.add_text("Content will be organized into these sections:", color=[150, 150, 160])
            dpg.add_spacer(height=5)
            dpg.add_text("  Recipe: Ingredients, Instructions", color=[100, 100, 120])
            dpg.add_text("  Tutorial: Requirements, Steps, Tips", color=[100, 100, 120])
            dpg.add_text("  Product: Features, Specifications, Reviews", color=[100, 100, 120])
            dpg.add_text("  Article: Summary, Key Points, Conclusion", color=[100, 100, 120])
            dpg.add_text("  Documentation: Overview, Usage, Examples, Parameters", color=[100, 100, 120])
            dpg.add_text("  FAQ: Questions, Answers", color=[100, 100, 120])
            dpg.add_text("  News: Summary, Details, Context", color=[100, 100, 120])

        # Custom fields (hidden by default)
        with dpg.group(tag="custom_fields_group", show=False):
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_text("CUSTOM CONFIGURATION", color=[255, 200, 100])
            dpg.add_text("Define your own prompt template and sections:", color=[150, 150, 160])

            dpg.add_spacer(height=5)
            dpg.add_text("User Prompt Template:")
            dpg.add_input_text(
                tag="custom_prompt_input",
                hint="Use {title} as placeholder, e.g. 'Explain {title} in detail'",
                width=-1
            )
            dpg.add_text("  Use {title} where the page title should appear", color=[100, 100, 120])

            dpg.add_spacer(height=8)
            dpg.add_text("Content Sections (comma-separated):")
            dpg.add_input_text(
                tag="custom_sections_input",
                hint="Overview, Details, Examples, Notes",
                width=-1
            )
            dpg.add_text("  Leave empty for no specific sections", color=[100, 100, 120])

            dpg.add_spacer(height=5)
            dpg.add_text("EXAMPLE:", color=[150, 200, 255])
            dpg.add_text("  Prompt: 'What are the main features of {title}?'", color=[100, 100, 120])
            dpg.add_text("  Sections: Key Features, Pros, Cons, Price", color=[100, 100, 120])

    # === CONCURRENCY ===
    with dpg.collapsing_header(label="Concurrency Settings", default_open=False):
        dpg.add_text("THREAD POOL", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="Workers", tag="inp_workers", default_value=10, min_value=1, max_value=50,
                              width=100)
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
            dpg.add_input_int(label="Max Depth", tag="inp_max_depth", default_value=3, min_value=1, max_value=10,
                              width=100)
            dpg.add_input_int(label="Links Per Page", tag="inp_links_per_page", default_value=20, min_value=1,
                              max_value=100, width=100)
            dpg.add_input_int(label="Max Per Domain", tag="inp_max_per_domain", default_value=100, min_value=1,
                              max_value=1000, width=100)

        dpg.add_checkbox(label="Stay On Same Domain", tag="chk_same_domain", default_value=True)
        dpg.add_checkbox(label="Prioritize Content Pages", tag="chk_prioritize_content", default_value=True)

        dpg.add_spacer(height=10)
        dpg.add_text("SMART FEATURES", color=[150, 150, 160])
        dpg.add_checkbox(label="[*] Automatic Subdomain Discovery", tag="chk_subdomain_discovery", default_value=False)
        dpg.add_text("  Discovers & crawls www, blog, docs, api, shop, etc.", color=[100, 100, 120])

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
            dpg.add_input_int(label="Min Chars", tag="inp_min_chars", default_value=150, width=120)
            dpg.add_input_int(label="Max Chars", tag="inp_max_chars", default_value=50000, width=120)
            dpg.add_input_int(label="Stop After N", tag="inp_stop_limit", default_value=0, width=120)

        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Allow Short High-Quality Content",
                             tag="chk_allow_short_quality", default_value=True)
            dpg.add_input_int(label="Short Min", tag="inp_short_min_chars", default_value=50,
                              min_value=10, max_value=500, width=100)
        dpg.add_text("  If enabled: content above 'Short Min' accepted if quality 70+ (ignores regular Min Chars)",
                     color=[100, 100, 120])

        dpg.add_spacer(height=5)
        dpg.add_checkbox(label="Ignore Quality Filter for Short Content",
                         tag="chk_ignore_quality_short", default_value=False)
        dpg.add_text("  If enabled: accepts ALL content above 'Short Min' regardless of quality", color=[100, 100, 120])
        dpg.add_text("  WARNING: Will include navigation/junk. Only use if you need everything.", color=[255, 150, 100])

        dpg.add_spacer(height=10)
        with dpg.collapsing_header(label="Keywords & Blacklist", default_open=False):
            dpg.add_text("KEYWORDS (comma separated)", color=[150, 150, 160])
            dpg.add_input_text(label="Must Contain", tag="inp_kw_in", width=-1,
                               hint="recipe, ingredient, cooking")
            dpg.add_input_text(label="Exclude If", tag="inp_kw_out", width=-1,
                               hint="privacy policy, terms of use, faq, about us, patreon")
            dpg.add_spacer(height=5)
            dpg.add_text("DOMAIN BLACKLIST (comma separated)", color=[150, 150, 160])
            dpg.add_input_text(tag="inp_domain_bl", width=-1,
                               hint="facebook.com, twitter.com, pinterest.com")
            dpg.add_text("  Pages from these domains will be skipped", color=[100, 100, 120])

        dpg.add_spacer(height=10)
        dpg.add_text("[QUALITY] NTTuner QUALITY FILTER", color=[255, 200, 50])
        dpg.add_checkbox(label="Enable Intelligent Quality Filtering", tag="chk_quality_filter", default_value=True)
        dpg.add_text("  Filters content based on information density & educational value", color=[100, 100, 120])

        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_text("Quality Threshold:", color=[200, 200, 210])
            dpg.add_slider_int(tag="inp_quality_threshold", default_value=35, min_value=0, max_value=100, width=200)
            dpg.add_text("(0=accept all, 100=only excellent)", color=[100, 100, 120])

        dpg.add_spacer(height=5)
        dpg.add_text("  Scoring factors:", color=[120, 180, 220])
        dpg.add_text("    - Information density (30%): how-to, tutorials, explanations", color=[100, 100, 120])
        dpg.add_text("    - Educational value (25%): technical, analytical content", color=[100, 100, 120])
        dpg.add_text("    - Structure quality (15%): lists, headers, paragraphs", color=[100, 100, 120])
        dpg.add_text("    - Noise level (15%): filters ads, navigation, placeholders", color=[100, 100, 120])
        dpg.add_text("    - Length (10%): sweet spot 800-5000 chars", color=[100, 100, 120])
        dpg.add_text("    - URL quality (5%): /blog/, /article/, /recipe/ patterns", color=[100, 100, 120])

        dpg.add_spacer(height=8)
        dpg.add_text("  Quality Score Ranges:", color=[150, 200, 255])

        # Visual quality range indicators
        with dpg.group(horizontal=True):
            dpg.add_text("    0-49:", color=[255, 100, 100])
            dpg.add_text("Poor (filtered)", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_text("   50-64:", color=[255, 200, 100])
            dpg.add_text("Fair (basic content)", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_text("   65-79:", color=[150, 255, 100])
            dpg.add_text("Good (quality content)", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_text("  80-100:", color=[50, 255, 100])
            dpg.add_text("Excellent (information-rich)", color=[150, 150, 160])

        dpg.add_spacer(height=5)
        dpg.add_text("  [TIP] Recommended: 50 for general, 65 for high-quality only", color=[150, 200, 255])

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
        dpg.add_text("POST-PROCESSING", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_button(label="Filter Dataset", callback=lambda: filter_existing_dataset(), width=150)
            dpg.add_button(label="Convert to Readable", callback=lambda: convert_jsonl_to_readable(), width=170)
            dpg.add_button(label="Preview First", callback=lambda: preview_first_entry(), width=130)
        dpg.add_text("  Filter: Remove junk | Convert: Create .txt | Preview: Show first entry", color=[100, 100, 120])

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
    dpg.add_text("NTCompanion", color=[50, 50, 60])

# Init
load_config()
if os.path.exists(INI_FILE):
    dpg.load_init_file(INI_FILE)
dpg.set_exit_callback(lambda: dpg.save_init_file(INI_FILE))

dpg.create_viewport(title="NTCompanion", width=1050, height=950, resizable=True)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("PrimaryWindow", True)
dpg.bind_theme(global_theme)
dpg.start_dearpygui()
dpg.destroy_context()
