# NTCompanion Pro - High-Performance Web Scraper for NTTuner

[![Version](https://img.shields.io/badge/version-build.2026.05.Pro-blue.svg)](https://github.com/noosed/NTTuner)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

**Professional-grade web scraper with concurrent processing, intelligent crawling, and proxy support for building high-quality LLM training datasets.**

---

## Features

### Core Capabilities
- **Concurrent Scraping**: 10-50x faster than single-threaded with ThreadPoolExecutor
- **Intelligent Crawler**: Follow links with configurable depth and domain restrictions
- **Proxy Support**: 20+ proxy sources with automatic rotation and health tracking
- **Memory-Efficient**: Bloom filter deduplication for handling millions of URLs
- **Rate Limiting**: Per-domain delays to avoid overwhelming servers
- **Real-Time Stats**: Live monitoring of success/failure rates and processing speed

### Content Processing
- **Smart Text Extraction**: Proven regex-based content cleaning
- **Template Support**: Built-in chat templates for LLaMA, Mistral, Qwen, Phi, Gemma
- **Flexible Filtering**: Character limits, keyword inclusion/exclusion, domain blacklists
- **Code Block Handling**: Optional removal of code snippets
- **Whitespace Cleaning**: Configurable text normalization

### Advanced Features
- **Priority Queue**: Content-focused URLs processed first
- **Automatic Chunking**: Splits output files at 500MB
- **Session Persistence**: Saves/loads configuration between runs
- **Comprehensive Logging**: Optional file logging with timestamps
- **Sound Notifications**: Audio alert when scraping completes

---

## Requirements

### Python Dependencies
```bash
pip install dearpygui pyperclip
```

### Optional (Recommended)
```bash
pip install mmh3  # For memory-efficient Bloom filter deduplication
```

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows, Linux, macOS
- **RAM**: 2GB minimum, 4GB+ recommended for large crawls
- **Storage**: Varies by dataset size (plan for 1-10GB+)

---

## Quick Start

### 1. Installation
```bash
# Clone or download NTCompanion.py
git clone https://github.com/noosed/NTTuner.git
cd NTTuner

# Install dependencies
pip install dearpygui pyperclip mmh3
```

### 2. Basic Usage
```bash
python NTCompanion.py
```

### 3. Configure Your First Scrape
1. **Add URLs**: Paste URLs (one per line) in the "Source Manifest" section
2. **Set Workers**: Start with 5-10 workers in "Concurrency Settings"
3. **Choose Template**: Select your LLM's chat template (e.g., "Meta Llama-3.1")
4. **Click START**: Monitor progress in the console

### 4. Output
- Default: `nttuner_dataset.jsonl`
- Format: One JSON object per line
- Structure: `{"text": "<formatted_conversation>"}`

---

## Detailed Guide

### Source Manifest
Enter URLs to scrape, one per line:
```
https://example.com/article1
https://example.com/article2
https://blog.example.com/post/123
```

**Tips:**
- Use full URLs including `http://` or `https://`
- Mix domains freely - rate limiting handles per-domain delays
- Paste thousands of URLs - deduplication prevents re-processing

### Concurrency Settings

| Setting | Recommended | Description |
|---------|-------------|-------------|
| **Workers** | 5-10 | Number of concurrent threads |
| **Domain Delay** | 1-2s | Delay between requests to same domain |
| **Max Retries** | 3 | Retry attempts for failed requests |
| **Timeout** | 25s | Maximum wait time per request |

**Performance Guide:**
- **Conservative**: 5 workers, 2s delay (safe for most sites)
- **Balanced**: 10 workers, 1s delay (good speed, respectful)
- **Aggressive**: 20+ workers, 0.5s delay (use with proxies only)

### Proxy Configuration

**20+ Built-in Sources:**
- ProxyScrape (HTTP, HTTPS, SOCKS4, SOCKS5)
- TheSpeedX GitHub lists
- Monosans proxy lists
- Geonode free proxies
- And many more...

**Usage:**
1. Select a proxy source from dropdown
2. Click "Fetch Selected" or "Fetch ALL" for maximum pool
3. Enable "Enable Proxies" checkbox
4. Optional: Import custom proxy list (IP:PORT format)

**Proxy Features:**
- Automatic health tracking
- Bad proxy quarantine (15-minute cooldown)
- Best-proxy selection based on success rate
- Clear quarantine to retry failed proxies

### Crawler Configuration

Transform your scraper into a web crawler:

| Setting | Default | Description |
|---------|---------|-------------|
| **Max Depth** | 3 | How deep to follow links (1=seeds only) |
| **Links Per Page** | 20 | Maximum links to extract per page |
| **Max Per Domain** | 100 | Maximum pages to scrape per domain |
| **Stay On Same Domain** | Yes | Only follow links to original domains |
| **Prioritize Content** | Yes | Process content-rich URLs first |

**Depth Guide:**
- **Depth 1**: Only scrape the URLs you provide
- **Depth 2**: Scrape seeds + links found on those pages
- **Depth 3+**: Deep crawl (can generate thousands of URLs)

**Example:**
```
Seed: https://blog.example.com/
Depth 1: Just the blog homepage
Depth 2: Homepage + all linked articles
Depth 3: Homepage + articles + linked resources
```

### Filter Configuration

**Content Cleaning:**
- **Remove Code Blocks**: Strip `<pre>` and `<code>` tags
- **Collapse Whitespace**: Normalize spacing (recommended)

**Size Constraints:**
- **Min Chars**: 300 (default) - Skip content that's too short
- **Max Chars**: 50,000 (default) - Skip content that's too long
- **Stop After N**: 0 (disabled) - Auto-stop after N successful scrapes

**Keyword Filtering:**
```
Must Contain: python, machine learning, tutorial
Exclude If: advertisement, sponsored, cookies
Domain Blacklist: facebook.com, twitter.com, instagram.com
```

**Filter Logic:**
- **Must Contain**: Page must include at least ONE keyword (comma-separated)
- **Exclude If**: Page is rejected if it contains ANY keyword
- **Domain Blacklist**: Skip URLs from these domains entirely

### Prompt & Template

**System Prompt Presets:**
- **Blank**: No system context (for base models)
- **Helpful Assistant**: General-purpose AI assistant
- **Data Summarizer**: For data extraction tasks
- **Code Expert**: For programming content
- **Creative Writer**: For narrative content
- **NTTuner Default**: Reasoning and clarity focused

**Custom System Prompt:**
```
You are an expert in data science and machine learning. 
Provide detailed, accurate explanations with examples.
```

**Supported Templates:**
- Meta LLaMA-3.1 / 3.2 / 3.3 Instruct
- Mistral Nemo / Large Instruct
- Qwen2.5 Instruct
- Phi-4 Instruct
- Gemma-2 Instruct

**Output Format:**
```json
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n[Scraped content here]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[Detailed answer based on content]<|eot_id|>"}
```

### Output Settings

**Output File:**
- Default: `nttuner_dataset.jsonl`
- Click "Select..." to choose custom location
- Automatic chunking at 500MB (creates `_part2`, `_part3`, etc.)

**Advanced Options:**
- **Log to File**: Save console output to `scraper_log.txt`
- **Sound on Finish**: Play notification when complete (Windows only)

---

## Usage Examples

### Example 1: Simple Article Scraping
```
URLs: 100 blog posts
Workers: 10
Domain Delay: 1s
Crawler: Disabled
Filters: Min 500 chars, Max 20000 chars
Output: 85 successful articles, 15 filtered/failed
Time: ~2 minutes
```

### Example 2: Deep Domain Crawl
```
URLs: 1 seed URL
Workers: 15
Crawler: Enabled (Depth 3, Same Domain)
Links Per Page: 30
Output: 847 pages discovered and scraped
Time: ~45 minutes
```

### Example 3: Multi-Domain with Proxies
```
URLs: 5000 mixed domain URLs
Workers: 25
Proxies: Enabled (ProxyScrape HTTP)
Domain Delay: 0.5s
Output: 4273 successful, 727 failed
Time: ~3 hours
```

### Example 4: Keyword-Filtered Technical Content
```
URLs: 1000 documentation URLs
Must Contain: python, tutorial, example
Exclude If: deprecated, legacy
Min Chars: 1000
Output: 312 high-quality tutorials
```

---

## Understanding Statistics

### Live Stats Display
```
OK: 1,247        # Successfully scraped and saved
Fail: 183        # Network/parsing errors
Skip: 89         # Filtered (size/keywords/duplicates)
Vol: 12.3M       # Total characters scraped (in thousands)
Speed: 8.2/s     # Current processing rate (pages/second)
```

### Progress Bar
- Shows percentage complete
- Displays current URL being processed
- Updates in real-time

### Console Log
```
[NT] [14:23:45] Engine Online. Queue: 1000.
[NT] [14:23:46] Scanning: https://example.com/page1
[NT] [14:23:47]   [+] [D1] https://example.com/page1... (4523 chars)
[NT] [14:23:48]   [>] Crawler: Added 12 new links.
[NT] [14:23:49] Scanning: https://example.com/page2
[NT] [14:23:50]   [-] [D1] https://example.com/page2... : HTTP 404
```

**Log Prefixes:**
- `[+]` Success - content saved
- `[-]` Error - request failed
- `[!]` Filtered - content rejected by filters
- `[>]` Crawler - new links discovered

---

## Troubleshooting

### Issue: "No valid URLs" error
**Solution:** Ensure URLs start with `http://` or `https://`

### Issue: Too many failures
**Solutions:**
- Reduce worker count (try 5-10)
- Increase domain delay (try 2-3s)
- Enable proxies
- Increase timeout (try 30-40s)

### Issue: Crawler not finding links
**Solutions:**
- Ensure "Enable Crawler" is checked
- Increase "Links Per Page" limit
- Disable "Stay On Same Domain" if you want external links
- Check if target sites use JavaScript rendering (not supported)

### Issue: Content quality is poor
**Solutions:**
- Increase Min Chars (try 500-1000)
- Add keyword filters (Must Contain)
- Enable "Remove Code Blocks" if scraping documentation
- Check system prompt is appropriate for content type

### Issue: Proxies not working
**Solutions:**
- Try different proxy source (some are more reliable)
- Click "Clear Quarantine" to retry banned proxies
- Use "Fetch ALL" to get maximum proxy pool
- Disable SOCKS proxies (not supported by urllib)

### Issue: Out of memory
**Solutions:**
- Reduce worker count
- Enable Stop After N limit
- Process in batches (split URL list)
- Restart between large scraping sessions

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────┐
│           DearPyGUI Interface                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Settings │  │ Controls │  │ Stats    │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│         Scrape Engine (Main Thread)             │
│  • URL Queue Management                         │
│  • ThreadPoolExecutor Coordination              │
│  • Statistics Aggregation                       │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │Worker 1 │  │Worker 2 │  │Worker N │
   └─────────┘  └─────────┘  └─────────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│              Shared Components                  │
│  • Proxy Manager (health tracking)              │
│  • Rate Limiter (per-domain delays)             │
│  • Bloom Filter (deduplication)                 │
│  • Crawl Queue (priority scheduling)            │
│  • Write Lock (thread-safe file I/O)            │
└─────────────────────────────────────────────────┘
                      │
                      ▼
               ┌─────────────┐
               │ Output File │
               │  (.jsonl)   │
               └─────────────┘
```

### Request Flow

```
1. URL → Rate Limiter (check domain delay)
2. Rate Limiter → Proxy Manager (get best proxy)
3. Proxy Manager → urllib.request (fetch URL)
4. urllib.request → Content Extractor (clean HTML)
5. Content Extractor → Filters (validate content)
6. Filters → Template Builder (format for LLM)
7. Template Builder → File Writer (thread-safe write)
8. File Writer → Stats Updater (increment counters)
9. Stats Updater → Crawler (extract links if enabled)
10. Crawler → URL Queue (add discovered links)
```

---

## File Structure

```
NTTuner/
├── bIG.py                          # Main application
├── nttuner_config_pro.json         # Auto-saved settings
├── ntcompanion_pro.ini            # Window layout/position
├── nttuner_dataset.jsonl          # Output dataset
├── nttuner_dataset_part2.jsonl    # Chunked output (if >500MB)
├── scraper_log.txt                # Optional log file
└── README.md                       # This file
```

---

## Best Practices

### Performance Optimization
1. **Start conservative**: 5-10 workers, test on small URL set
2. **Monitor failures**: If >20% fail, reduce workers or add proxies
3. **Use proxies for large jobs**: Avoids IP bans, enables higher concurrency
4. **Batch processing**: Split 100K URLs into 10K chunks

### Content Quality
1. **Tight filters**: Better to reject marginal content than pollute dataset
2. **Domain-specific keywords**: "tutorial", "documentation", "guide"
3. **Minimum character counts**: 500-1000 for substantive content
4. **Review samples**: Spot-check output quality early

### Crawler Strategy
1. **Start shallow**: Depth 2 for initial exploration
2. **Same-domain only**: Prevents explosive link growth
3. **Content prioritization**: Enable for better quality-to-volume ratio
4. **Domain limits**: Prevents single domain from dominating dataset

### Proxy Usage
1. **Fetch ALL**: Maximize pool size before starting
2. **Monitor quarantine**: Clear periodically to recycle proxies
3. **Expect failures**: Even good proxies fail ~10-30% of time
4. **Public vs Private**: Public proxies are free but less reliable

---

## Rate Limiting & Ethics

### Built-in Protections
- **Per-domain delays**: Never hammer a single server
- **Configurable timeouts**: Respect slow-responding servers
- **User-Agent rotation**: Identify as scraper, not disguised bot
- **Retry backoff**: Exponential delays on retries

### Recommended Settings
```
Domain Delay: 1-2 seconds (minimum)
Workers: ≤20 for public sites
Timeout: 25-30 seconds
Max Retries: 2-3 attempts
```

### Respectful Scraping
1. **Check robots.txt**: Honor site policies (not auto-enforced)
2. **Reasonable rate limits**: Don't overwhelm servers
3. **Off-peak hours**: Scrape during low-traffic times
4. **Contact site owners**: For large-scale scraping, ask permission
5. **Cache results**: Don't re-scrape unnecessarily

### Legal Considerations
- This tool is for **educational and research purposes**
- Respect copyright, terms of service, and privacy laws
- Public data ≠ legal to scrape (jurisdiction-dependent)
- Commercial use may require permissions/licenses
- Use for personal learning and open research
- Cite sources in published datasets
- Remove sensitive content if discovered

---

## Known Limitations

### Technical
- **No JavaScript rendering**: Can't scrape SPAs or dynamic content
- **SOCKS proxy**: Not supported by urllib (HTTP/HTTPS only)
- **Character encoding**: Rare encoding errors on exotic charsets
- **Binary detection**: Occasionally misses non-text content

### Scale
- **Memory growth**: Large crawls (100K+ URLs) may consume 1-2GB RAM
- **Bloom filter**: False positive rate ~0.001% at 1M URLs
- **Concurrent writes**: Occasional file lock contention on Windows

### Content
- **Paywalls**: Can't access subscriber-only content
- **CAPTCHAs**: No automatic solving
- **Rate limits**: Some sites enforce strict limits regardless of delays
- **Cloudflare/Bot detection**: May block urllib user-agents

---

## Updates & Changelog

### v2026.05.Pro-ThreadPool (Current)
- Integrated proven scraper from a.py
- Simplified content extraction for better reliability
- Improved link discovery algorithm
- Removed complex, unnecessary code paths
- Enhanced error handling and logging
- Fixed encoding issues with non-UTF8 content

### v2026.03.Pro (Previous)
- Initial ThreadPool-based concurrent scraper
- Advanced proxy management
- Priority crawl queue
- Bloom filter deduplication

---

## Contributing

This tool is part of the **NTTuner** ecosystem. Contributions welcome!

### Report Issues
- Clear description of problem
- Steps to reproduce
- Sample URLs (if applicable)
- System info (OS, Python version)

### Feature Requests
- Describe use case
- Expected behavior
- Alternative solutions considered

### Pull Requests
- Follow existing code style
- Test thoroughly
- Update README if needed
- One feature per PR

---

## Support

### Documentation
- **This README**: Comprehensive guide
- **NTTuner Repo**: https://github.com/noosed/NTTuner
- **Code Comments**: Inline documentation

### Community
- **GitHub Issues**: Bug reports and features
- **Discussions**: General questions and tips

### Quick Help
```python
# Enable debug logging
dpg.set_value("chk_log_file", True)

# Check logs for errors
tail -f scraper_log.txt

# Verify output format
head -n 5 nttuner_dataset.jsonl | python -m json.tool
```

---

## License

This project is part of NTTuner and follows its licensing terms.

**For educational and research use.** Commercial use may require additional permissions.

---

## Acknowledgments

- **DearPyGUI**: Modern GPU-accelerated Python GUI framework
- **urllib**: Python's reliable HTTP library
- **Proxy Sources**: Free proxy list providers
- **NTTuner Community**: Feedback and testing

---

## Learning Resources

### Web Scraping
- Respect robots.txt and rate limits
- Understand HTTP headers and user-agents
- Handle encoding and character sets properly

### Concurrent Programming
- ThreadPoolExecutor for I/O-bound tasks
- Thread-safe file writing with locks
- Rate limiting in multi-threaded contexts

### LLM Training Data
- Quality over quantity for better models
- Diverse sources prevent overfitting
- Proper formatting for instruction tuning
- Deduplication prevents memorization

---

## Performance Benchmarks

### Single Domain (Blog)
```
URLs: 500 articles
Workers: 10
Time: ~8 minutes
Success Rate: 94%
Output: 487 articles, 8.2MB
```

### Multi-Domain Crawl
```
Seeds: 10 domains
Depth: 3
Workers: 15
Time: ~2 hours
Discovered: 5,847 URLs
Success Rate: 73%
Output: 4,271 pages, 124MB
```

### Large-Scale Proxy Scrape
```
URLs: 50,000 mixed
Workers: 30
Proxies: Enabled
Time: ~6 hours
Success Rate: 68%
Output: 34,000 pages, 892MB
```

*Benchmarks vary based on target sites, network speed, and content size.*

---

## Version Compatibility

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| Python | 3.8 | 3.10+ | 3.11 |
| DearPyGUI | 1.9 | Latest | 1.11 |
| Windows | 10 | 11 | 11 |
| Linux | Ubuntu 20.04 | 22.04 | 22.04 |
| macOS | 11 | 13+ | 13 |

---

**Built for the LLM training community**

*Happy Scraping!*
