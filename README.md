

<img width="503" height="305" alt="image" src="https://github.com/user-attachments/assets/66f421ec-dc25-4a9a-8cba-e5e846829ef7" />


# NTCompanion Enhanced

**Professional dataset engine for NTTuner with codebase scanning capability**

NTCompanion is a powerful web scraper and codebase analyzer designed to create high-quality training datasets for fine-tuning language models. Build datasets from websites, documentation, or entire codebases with intelligent content filtering and quality scoring.

## What's New

**Codebase Dataset Builder** - Point the tool at any code repository and automatically generate training data from your codebase. Supports 40+ programming languages with intelligent metadata extraction.

## Features

### Web Scraping
- Multi-threaded crawling with configurable depth and workers
- Intelligent content extraction that filters navigation, ads, and noise
- Quality scoring system (6-factor analysis: information density, educational value, structure, noise level, length, URL patterns)
- Smart link discovery with priority queuing
- Domain-aware rate limiting to be respectful
- Subdomain discovery and enumeration
- Proxy support with automatic rotation and health tracking
- User agent rotation with 15+ authentic browser profiles

### Codebase Analysis
- Recursive folder scanning for code files
- Support for Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, Scala, and 30+ more languages
- Automatic detection and skipping of .git, node_modules, build folders, etc.
- Extracts functions, classes, imports, and code structure
- Multi-threaded processing for large codebases
- Configurable file size and character limits

### Content Processing
- Multiple chat templates (Llama 3, Mistral, Qwen, Phi-4, Gemma-2)
- Configurable system prompts with presets
- Content type configurations (Recipe, Tutorial, Documentation, Code, etc.)
- Keyword filtering (include/exclude patterns)
- Length constraints with quality-based overrides
- Real-time statistics and progress tracking

### Quality Control
- NTTuner-optimized content scoring algorithm
- Junk page detection (FAQ, privacy policy, list pages)
- Recipe/instruction validation for food content
- Configurable quality thresholds (0-100 scale)
- Smart short content handling
- Duplicate detection with Bloom filters

## Installation

### Requirements

```bash
pip install dearpygui beautifulsoup4
```

Optional (for memory-efficient large crawls):
```bash
pip install mmh3
```

### Quick Setup

1. Clone or download NTCompanion.py
2. Install dependencies
3. Run the application

```bash
git clone https://github.com/noosed/NTCompanion.git
cd NTCompanion
pip install -r requirements.txt
python NTCompanion.py
```

## Usage

### Web Scraping Mode

1. Enter URLs in the Source Manifest section (one per line)
2. Configure crawl settings (depth, workers, quality threshold)
3. Set content filters (keywords, character limits)
4. Choose your chat template and system prompt
5. Click START SCRAPING
6. Monitor progress and check output in nttuner_dataset.jsonl

### Codebase Mode

1. Open the "Codebase Dataset Builder" section
2. Click Browse and select your code folder
3. Configure settings (workers, character limits, template)
4. Set Content Type to "Code File" for best results
5. Click START CODEBASE PROCESSING
6. Output saved to the same dataset file

## Configuration

### Crawl Settings
- **Workers**: 1-50 concurrent threads (recommended: 10-20)
- **Domain Delay**: Minimum seconds between requests to same domain (recommended: 1-2s)
- **Max Depth**: How many link layers to follow (1=seed only, 2=seed+links, 3+=deep crawl)
- **Max Retries**: Number of retry attempts on failure
- **Timeout**: Request timeout in seconds

### Quality Filters
- **Min/Max Characters**: Content length constraints
- **Quality Threshold**: 0-100 score (50=general, 65=high quality, 80=excellent)
- **Keywords In/Out**: Comma-separated filtering terms
- **Domain Blacklist**: Skip specific domains entirely

### Content Types

Pre-configured templates for different content:
- Recipe (ingredients, instructions)
- Tutorial (requirements, steps, tips)
- Product Info (features, specs, reviews)
- Article/Blog (summary, key points)
- Documentation (overview, usage, examples)
- FAQ (questions, answers)
- Code File (purpose, functions, usage)
- Custom (define your own)

## Quality Scoring

The intelligent content scorer evaluates pages across six dimensions:

**Information Density (30%)** - Identifies how-to content, tutorials, explanations  
**Educational Value (25%)** - Detects technical and analytical content  
**Structure Quality (15%)** - Evaluates lists, headers, paragraphs  
**Noise Filtering (15%)** - Removes ads, navigation, placeholders  
**Length Optimization (10%)** - Sweet spot: 800-5000 characters  
**URL Quality (5%)** - Recognizes quality URL patterns

Score ranges:
- 0-49: Poor (filtered)
- 50-64: Fair (basic content)
- 65-79: Good (quality content)
- 80-100: Excellent (information-rich)

## Output Format

NTCompanion generates JSONL files compatible with NTTuner:

```json
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow do I make Chocolate Cake?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nChocolate Cake\nIngredients:\n- 2 cups flour\n...<|eot_id|>"}
```

Each line is a complete conversation ready for fine-tuning.

## File Structure

```
NTCompanion/
├── NTCompanion.py              # Main application
├── nttuner_config_pro.json     # Auto-saved configuration
├── ntcompanion_pro.ini         # Window layout/position
├── nttuner_dataset.jsonl       # Output dataset
├── scraper_log.txt             # Optional log file
└── README.md                   # This file
```

## Advanced Features

### Proxy Support
- Multiple proxy sources (20+ built-in)
- Automatic health tracking and quarantine
- Score-based selection (success rate weighted)
- Import custom proxy lists

### Subdomain Discovery
- Automatically discovers www, blog, docs, api, shop subdomains
- Verifies subdomain existence before crawling
- Expands crawl scope intelligently

### Bloom Filters
- Memory-efficient deduplication for large crawls
- Handles 100,000+ URLs with minimal overhead
- Optional (requires mmh3 package)

### Session Persistence
- Saves configuration between runs
- Remembers window position and size
- Maintains proxy pool state

## Tips for Best Results

**For general scraping:**
- Start with depth 2 and 10-15 workers
- Use quality threshold of 50-65
- Enable quality filtering
- Set domain delay to 1-2 seconds

**For high-quality datasets:**
- Increase quality threshold to 70+
- Use keyword filtering to focus content
- Enable "Allow Short High-Quality Content"
- Review and iterate on results

**For large crawls:**
- Install mmh3 for Bloom filters
- Use 20-30 workers for speed
- Enable proxy rotation if needed
- Monitor failed requests and adjust

**For code analysis:**
- Set min characters low (50-100) for small files
- Use "Code File" content type
- Process one repository at a time
- Check logs for encoding issues

## Troubleshooting

**Too many low-quality pages**
- Increase quality threshold to 65-70
- Add exclusion keywords: privacy, terms, about, contact
- Enable junk filtering

**Not enough data**
- Decrease quality threshold to 40-50
- Increase crawl depth to 3-4
- Disable "Same Domain Only"

**Scraping too slow**
- Increase workers to 20-30
- Reduce crawl depth to 2
- Check domain delay setting

**Getting blocked**
- Reduce workers to 5-10
- Increase domain delay to 2-3 seconds
- Enable proxy rotation

**Code files not processing**
- Check file permissions
- Verify encoding (UTF-8 preferred)
- Check console for specific errors
- Adjust min/max character limits

## Integration with NTTuner

NTCompanion is designed to work seamlessly with NTTuner for fine-tuning:

```bash
# After scraping, use with NTTuner
python -m nttuner.train \
  --data nttuner_dataset.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --output my_finetuned_model \
  --epochs 3
```

## Contributing

Contributions are welcome. If you find bugs or have feature requests, please open an issue on GitHub.

## License

This tool is provided for educational and research purposes. Users are responsible for complying with website terms of service and applicable laws when scraping content.

## Changelog

### v2026.05 - Pro+Enhanced+ContentTypes+Codebase
- Added codebase dataset builder
- Support for 40+ programming languages
- Intelligent code metadata extraction
- Multi-threaded code processing
- Enhanced error handling for file operations
- Fixed DearPyGUI deprecation warnings

### v2026.05 - Pro+Enhanced+ContentTypes
- Added content type configurations
- Recipe, tutorial, documentation templates
- Improved content extraction
- Better junk page detection
- Enhanced quality scoring

### v2026.05 - Pro+Enhanced
- Intelligent quality filtering
- 6-factor content scoring
- Subdomain discovery
- Bloom filter support
- Enhanced proxy management

---

**Made by [@noosed](https://github.com/noosed)**
### Data Quality

* The quality filter helps exclude low-value content
* Keyword filters prevent scraping unwanted sections
* Domain blacklisting prevents social media scraping

## Changelog

### v2026.06 - Enhanced+Universal+BugFix

* Added BeautifulSoup support for better parsing
* Implemented content-type aware extraction
* Enhanced URL normalization and link discovery
* Added Bloom filter for memory efficiency
* Improved quality scoring algorithm
* Better error handling and retries
* Domain-based rate limiting
* Multiple user agent rotation
* Enhanced text cleaning and extraction
* Fixed encoding detection issues
* Added comprehensive documentation

## Credits

* Built for [NTTuner](https://github.com/noosed/nttuner) by noosed
* Uses [DearPyGUI](https://github.com/hoffstadt/DearPyGui) for the interface
* Optional [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) for parsing
* Optional [mmh3](https://github.com/hajimes/mmh3) for Bloom filters

## License

This tool is provided as-is for educational and research purposes. Users are responsible for complying with website terms of service and applicable laws.

## Reporting Issues

If you encounter bugs or have feature requests:

1. Check the console log for error messages
2. Note your configuration (content type, depth, etc.)
3. Provide example URLs if possible
4. Describe expected vs actual behavior

## Learn More

* **NTTuner Documentation**: https://github.com/noosed/nttuner
* **Fine-tuning Guide**: Check NTTuner's README
* **Best Practices**: See Tips & Best Practices section above

---

**Happy Scraping!**

Build high-quality datasets for your fine-tuning projects!
