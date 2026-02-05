

<div align="center"><img width="503" height="305" alt="image" src="https://github.com/user-attachments/assets/66f421ec-dc25-4a9a-8cba-e5e846829ef7" /></div>

# NTCompanion Pro

**Professional Dataset Engine for Training Better AI Models**

> Your AI is only as good as the data it learns from. Let's fix that.

A powerful web scraping and dataset creation tool designed specifically to work with [NTTuner](https://github.com/noosed/NTTuner). Built for people who want to fine-tune language models but don't want to spend days manually collecting training data.

---

## What is NTCompanion?

Ever tried to fine-tune an AI model and realized you need thousands of high-quality training examples? Yeah, that's painful. 

NTCompanion automates the entire process. Point it at websites you want to learn from, and it'll scrape, filter, and format everything into a clean dataset ready for fine-tuning with NTTuner.

**What it does:**
- Collects training data from any website automatically
- Filters out junk using intelligent quality scoring
- Formats everything perfectly for NTTuner
- Builds specialized datasets for recipes, tutorials, docs, code, and more

---

## Features

### Smart Web Scraping

- **Multi-threaded crawling** - Process 10-50 pages simultaneously
- **Intelligent link discovery** - Automatically finds and follows relevant links
- **Depth control** - Crawl seed pages only, or go 2-3 levels deep
- **Same-domain mode** - Stay focused on one site or explore freely
- **Rate limiting** - Be respectful to websites (automatic throttling)
- **User agent rotation** - Uses 15+ realistic browser signatures

### Quality Filtering (The Important Part)

Not all web content is worth training on. NTCompanion includes a 6-factor quality scoring system:

1. **Information Density (30%)** - Finds how-tos, tutorials, explanations
2. **Educational Value (25%)** - Detects technical, analytical content  
3. **Structure Quality (15%)** - Values proper formatting with lists, headers
4. **Noise Level (15%)** - Filters out ads, navigation, placeholders
5. **Length (10%)** - Prefers the sweet spot of 800-5000 characters
6. **URL Quality (5%)** - Recognizes quality patterns like /blog/, /article/

**Quality Score Ranges:**
- **0-49**: Poor - Filtered out
- **50-64**: Fair - Basic content
- **65-79**: Good - Quality content
- **80-100**: Excellent - Information-rich

### Content Type Presets

Pre-optimized configurations for different types of content:

| Content Type | Best For | Example Sites |
|-------------|----------|---------------|
| **Generic/Auto** | Any website | Universal fallback |
| **Recipe** | Cooking sites | AllRecipes, FoodNetwork |
| **Tutorial** | Learning content | WikiHow, tutorials |
| **Article/Blog** | News/opinions | Medium, blogs |
| **Documentation** | Technical docs | ReadTheDocs, API docs |
| **FAQ** | Q&A pages | Support sites |
| **Product Info** | E-commerce | Product pages |
| **Code File** | GitHub repos | Source code |

### NTTuner Integration

Outputs data in exactly the format NTTuner expects:

- Proper JSONL formatting
- Chat template support (Llama, Mistral, Qwen, Phi, Gemma)
- System prompt configuration
- Custom user prompt templates per content type
- Ready for immediate fine-tuning

### Code & GitHub Support

Process entire codebases:

- Local folder scanning (auto-skips node_modules, .git, etc.)
- GitHub repository cloning and processing
- Support for 40+ programming languages
- RAG-style chunking for better code understanding
- Function and class extraction

---

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Disk Space**: 500MB for dependencies

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/noosed/NTCompanion.git
cd NTCompanion

# 2. Install required dependencies
pip install dearpygui

# 3. That's it (Optional dependencies below)
```

### Optional (But Recommended) Dependencies

```bash
# For better HTML parsing
pip install beautifulsoup4 lxml

# For memory-efficient large crawls (10,000+ pages)
pip install mmh3

# For GitHub repository support
pip install gitpython

# All at once
pip install beautifulsoup4 lxml mmh3 gitpython
```

### Verify Installation

```bash
python NTCompanion.py
```

You should see the GUI window pop up. If it does, you're ready to go.

---

## Quick Start

### Your First Dataset in 5 Minutes

Let's scrape a cooking website to create a recipe dataset:

1. **Launch NTCompanion**
   ```bash
   python NTCompanion.py
   ```

2. **Add Your URLs**
   - In the "URL LIST" box, paste:
   ```
   https://www.allrecipes.com/recipes/
   https://www.allrecipes.com/recipes/desserts/
   ```

3. **Configure Settings**
   - **Content Type**: Select "Recipe"
   - **Crawl Depth**: Set to "2"
   - **Quality Threshold**: Set to "50"
   - **Workers**: Set to "10"

4. **Set Filters (Optional but Recommended)**
   - **Keywords Must Contain**: `recipe, ingredient, instructions`
   - **Keywords Exclude**: `subscribe, newsletter, advertisement`
   - **Min Chars**: `200`
   - **Max Chars**: `10000`

5. **Click START**
   - Watch the console log fill with progress
   - Wait for "Scraping complete!" message
   - Your data is saved to `scraped_data.jsonl`

That's it. You now have a recipe dataset ready for NTTuner.

---

## Complete Tutorial

### Understanding Crawl Depth

**Depth** controls how far the scraper explores from your seed URLs:

- **Depth 1**: Only scrapes the exact URLs you provide
  - Use when: You have a specific list of pages
  - Result: Fast, precise, around 10-50 pages

- **Depth 2**: Scrapes seeds plus all links found on those pages (RECOMMENDED)
  - Use when: You want to explore a section thoroughly
  - Result: Medium speed, around 50-500 pages

- **Depth 3+**: Goes deeper into the site hierarchy
  - Use when: You want comprehensive coverage
  - Result: Slower, can find 500-10,000+ pages

**Pro Tip**: Start with depth 2. You can always run it again deeper if you need more data.

### Setting Up Quality Filters

Quality filters are your best friend. Here's how to use them effectively:

#### Example 1: High-Quality Technical Blog

```
Content Type: Article/Blog
Keywords Must Contain: tutorial, guide, how-to, explained
Keywords Exclude: subscribe, buy now, ad, promotion
Quality Threshold: 70
Min Chars: 500
Same Domain Only: Enabled
```

**Result**: Only keeps well-written, informative articles.

#### Example 2: Documentation Scraping

```
Content Type: Documentation  
Keywords Must Contain: API, function, parameter, example
Keywords Exclude: pricing, contact, enterprise
Quality Threshold: 75
Min Chars: 300
Ignore Quality for Short Content: Disabled
```

**Result**: Technical documentation with code examples.

#### Example 3: Recipe Collection

```
Content Type: Recipe
Keywords Must Contain: ingredients, instructions, servings
Keywords Exclude: sponsored, affiliate, advertisement
Quality Threshold: 60
Allow Short High-Quality Content: Enabled
Short Min: 150
```

**Result**: Complete recipes, including shorter ones if high-quality.

### Content Type Selection Guide

Not sure which content type to use? Here's a decision tree:

```
What are you scraping?
├─ Food/cooking website? → Recipe
├─ How-to guides/tutorials? → Tutorial/How-To
├─ News/opinion articles? → Article/Blog
├─ Technical documentation? → Documentation
├─ FAQ/support pages? → FAQ
├─ Product descriptions? → Product Info
├─ GitHub/source code? → Code File
└─ Not sure? → Generic/Auto-detect
```

### Working with Code Repositories

NTCompanion can process entire codebases:

#### Processing a Local Folder

1. Click **"Browse"** under "Codebase Dataset Builder"
2. Select your code folder
3. Configure:
   - **RAG Chunking**: Enable for better semantic chunking
   - **Chunk Size**: 50 lines (default is good)
4. Click **"START CODEBASE PROCESSING"**

#### Processing GitHub Repositories

1. Create a text file with GitHub URLs:
   ```
   https://github.com/microsoft/vscode
   https://github.com/python/cpython
   https://github.com/torvalds/linux
   ```

2. In NTCompanion:
   - Click **"Browse"** next to "GitHub List File"
   - Select your text file
   - (Optional) Add personal access token for private repos
   - Click **"START CODEBASE PROCESSING"**

**Supported Languages**:
Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, Scala, R, Shell, SQL, HTML, CSS, and 30+ more.

---

## Working with NTTuner

The entire point of NTCompanion is to create data for [NTTuner](https://github.com/noosed/NTTuner). Here's the complete workflow:

### Step 1: Create Your Dataset with NTCompanion

```bash
# 1. Scrape your data
python NTCompanion.py
# Configure and scrape as shown above
# Output: scraped_data.jsonl
```

### Step 2: Verify Your Dataset

Quick check:
```bash
# On Windows
type scraped_data.jsonl | find /c "{"

# On Linux/Mac  
wc -l scraped_data.jsonl
```

This tells you how many training examples you have.

**How much data do you need?**
- **Minimum**: 100 examples (very small, targeted task)
- **Good**: 500-1,000 examples (solid fine-tune)
- **Great**: 2,000-5,000 examples (comprehensive dataset)
- **Overkill**: 10,000+ examples (diminishing returns)

### Step 3: Configure NTTuner

1. **Open NTTuner**:
   ```bash
   python NTTuner.py
   ```

2. **Load Your Dataset**:
   - Click "Browse" next to Dataset Path
   - Select your `scraped_data.jsonl` file

3. **Choose a Base Model**:
   - For 8GB GPU: `meta-llama/Llama-3.2-3B-Instruct`
   - For 12GB GPU: `meta-llama/Llama-3.1-8B-Instruct`  
   - For 16GB+ GPU: `mistralai/Mistral-Nemo-Instruct-2407`

4. **Set Training Parameters**:
   - **LoRA Rank**: 16-32 (start with 16)
   - **Epochs**: 1-3 (start with 1)
   - **Batch Size**: 4 (reduce if OOM)
   - **Learning Rate**: 2e-4 (default is good)

5. **Configure Output**:
   - **Model Name**: Give it a meaningful name
   - **Quantization**: q5_k_m (good balance)
   - **Import to Ollama**: Enabled

6. **Click "Start Training"**

### Step 4: Use Your Fine-Tuned Model

Once training completes:

```bash
# Test your model
ollama run your-model-name

# Example conversation
>>> Hello! I need help with a recipe.
>>> Can you explain how to make chocolate chip cookies?
```

Your model should now respond with knowledge from your scraped dataset.

### Template Matching

**IMPORTANT**: The chat template in NTCompanion must match your NTTuner base model:

| NTTuner Base Model | NTCompanion Template |
|-------------------|---------------------|
| meta-llama/Llama-3.2-* | Meta Llama-3.1/3.2/3.3 Instruct |
| mistralai/Mistral-* | Mistral Nemo/Large Instruct |
| Qwen/Qwen2.5-* | Qwen2.5 Instruct |
| microsoft/Phi-4 | Phi-4 Instruct |
| google/gemma-2-* | Gemma-2 Instruct |

**If they don't match**, NTTuner won't be able to parse your dataset correctly.

---

## Configuration Guide

### Saving & Loading Configs

**Save your configuration**:
1. Set up all your parameters
2. Click "Save Config"
3. Choose a location (e.g., `recipe_scraper.json`)

**Load a saved configuration**:
1. Click "Load Config"
2. Select your JSON file
3. All settings populate automatically

**Pro Tip**: Create multiple config files for different projects:
- `recipes.json`
- `tech_docs.json`
- `tutorials.json`

### System Prompts

System prompts define how the AI should behave. NTCompanion includes presets:

- **Blank**: No system prompt (rare use case)
- **Helpful Assistant**: General purpose (RECOMMENDED)
- **Data Summarizer**: For summarization tasks
- **Code Expert**: For technical/code content
- **Creative Writer**: For narrative, story-like content
- **NTTuner Default**: Optimized for reasoning

**Custom System Prompts**:

```
You are a knowledgeable cooking assistant specialized in 
international cuisine. Provide detailed, step-by-step 
instructions and explain cooking techniques clearly.
```

Click dropdown, select "Helpful Assistant", modify the text box, done.

### Understanding User Prompts

Each content type has a **user prompt template** that creates the question part of the training data.

**Example for Recipes**:
```
Template: "How do I make {title}?"
Scraped Title: "Chocolate Chip Cookies"
Result: "How do I make Chocolate Chip Cookies?"
```

The AI learns to respond to questions like this with the scraped content.

---

## Best Practices

### Do's

1. **Start small** - Test with 2-3 URLs before scaling up
2. **Use quality filtering** - Don't disable it unless you have a reason
3. **Match content types** - Select the right preset for your data
4. **Enable "Same Domain Only"** - When building focused datasets
5. **Set keyword filters** - Be specific about what you want/don't want
6. **Save your configs** - Reuse successful configurations
7. **Monitor the console** - Watch for patterns in failures
8. **Iterate** - Adjust filters based on initial results

### Don'ts

1. **Don't skip quality filtering** - You'll get lots of junk
2. **Don't use depth 5+** - You'll scrape forever (and maybe get blocked)
3. **Don't set workers too high** - Above 30 is asking for trouble
4. **Don't ignore robots.txt** - The scraper respects it automatically
5. **Don't scrape paywalled content** - It won't work and it's unethical
6. **Don't forget to match templates** - NTTuner won't understand mismatched data
7. **Don't collect more than you need** - 5,000 examples is plenty for most tasks

### Performance Tips

**For speed**:
```
Workers: 20-30
Depth: 2
Quality Threshold: 50
Crawl Strategy: BFS (Breadth-first)
```

**For quality**:
```
Workers: 5-10
Depth: 2-3
Quality Threshold: 70
Enable all keyword filters
```

**For discovery**:
```
Workers: 10
Depth: 3-4
Same Domain: Disabled
Subdomain Discovery: Enabled
```

---

## Troubleshooting

### Common Issues

**Problem: "No content extracted" for most pages**

**Solution**:
- Lower quality threshold to 40
- Check if keywords are too restrictive
- Disable "Ignore Quality Filter for Short Content"
- Try "Generic/Auto-detect" content type

---

**Problem: Too much junk data collected**

**Solution**:
- Increase quality threshold to 65-70
- Add exclusion keywords: `privacy, terms, about, subscribe, newsletter`
- Enable "Same Domain Only"
- Reduce crawl depth to 2

---

**Problem: Scraping is very slow**

**Solution**:
- Increase workers to 20-30
- Reduce crawl depth
- Check your internet connection
- Some sites have rate limiting (be patient)

---

**Problem: Getting blocked/403 errors**

**Solution**:
- Reduce workers to 5-10
- The scraper already rotates user agents and respects rate limits
- Some sites simply don't allow scraping
- Try adding delays between requests (edit `CRAWL_DELAY` in code)

---

**Problem: NTTuner can't load the dataset**

**Solution**:
- Check that chat template matches base model
- Verify JSONL file isn't corrupted:
  ```bash
  python -m json.tool scraped_data.jsonl
  ```
- Make sure each line is valid JSON
- Check file encoding is UTF-8

---

**Problem: Out of memory during large crawls**

**Solution**:
- Install mmh3: `pip install mmh3` (enables Bloom filter)
- Reduce workers to 10
- Set "Stop After N" limit
- Process in smaller batches

---

## Example Workflows

### Workflow 1: Recipe Dataset for Cooking Chatbot

**Goal**: Train a model to answer cooking questions

```
NTCompanion Settings:
  Content Type: Recipe
  URLs: 
    - https://www.allrecipes.com/
    - https://www.foodnetwork.com/
  Depth: 2
  Workers: 15
  Quality Threshold: 60
  Keywords Include: recipe, ingredients, instructions
  Keywords Exclude: sponsored, advertisement
  Template: Meta Llama-3.2 Instruct
  System Prompt: "You are a helpful cooking assistant."

Expected Result: 500-2000 recipe examples

NTTuner Settings:
  Base Model: meta-llama/Llama-3.2-3B-Instruct
  Epochs: 2
  LoRA Rank: 16
  Batch Size: 4
```

---

### Workflow 2: Technical Documentation Assistant

**Goal**: Create an AI expert in a specific API/framework

```
NTCompanion Settings:
  Content Type: Documentation
  URLs:
    - https://docs.python.org/3/
    - https://pytorch.org/docs/
  Depth: 3
  Workers: 10
  Quality Threshold: 75
  Keywords Include: function, method, parameter, example
  Keywords Exclude: download, install
  Same Domain: Enabled
  Template: Qwen2.5 Instruct

Expected Result: 1000-3000 documentation pages

NTTuner Settings:
  Base Model: Qwen/Qwen2.5-7B-Instruct
  Epochs: 3
  LoRA Rank: 32
  Batch Size: 2
```

---

### Workflow 3: Code Understanding Model

**Goal**: Fine-tune a model on a codebase

```
NTCompanion Settings:
  Mode: Codebase Dataset Builder
  Source: Local Folder
  Path: /path/to/your/project
  Content Type: Code File
  RAG Chunking: Enabled
  Chunk Size: 50 lines
  Template: Meta Llama-3.2 Instruct

Expected Result: 500-5000 code chunks

NTTuner Settings:
  Base Model: meta-llama/Llama-3.2-3B-Instruct
  Epochs: 1
  LoRA Rank: 16
  Batch Size: 4
  Max Seq Length: 2048
```

---

## Legal & Ethical Usage

### Legal Considerations

**Before scraping a website**:
1. Check their `robots.txt` file (scraper respects this automatically)
2. Review their Terms of Service
3. Look for an official API (always prefer APIs when available)
4. Consider seeking permission for commercial use

**Copyright**:
- Scraped data may be copyrighted
- Fine-tuning on copyrighted data is a gray area legally
- Use for research/personal projects is generally safer
- Commercial use requires more caution

### Be Respectful

**NTCompanion includes built-in protections**:
- Rate limiting per domain
- User agent rotation
- Respects robots.txt
- Exponential backoff on errors

**But you should still**:
- Keep workers reasonable (10-20 is plenty)
- Don't hammer small sites
- Respect paywalls
- Don't scrape personal/private data
- Consider the website's resources

### Good Use Cases

- Publicly accessible content
- Educational research
- Personal learning projects
- Non-commercial fine-tuning
- Your own content scattered across sites
- Open-source documentation

### Bad Use Cases

- Circumventing paywalls
- Collecting personal information
- Commercial scraping without permission
- Social media at scale
- Financial/trading data (use official APIs)
- Anything explicitly forbidden in ToS

**Bottom line**: If you're unsure, ask permission or use an official API.

---

## Performance Benchmarks

Tested on typical websites with 10 workers, quality threshold 50:

| Content Type | Pages/Minute | Memory Usage | Avg Quality Score |
|-------------|-------------|--------------|------------------|
| Blog Posts | 30-50 | ~100MB | 65-75 |
| Documentation | 20-40 | ~150MB | 70-85 |
| Recipes | 40-60 | ~80MB | 60-70 |
| News Articles | 25-45 | ~120MB | 55-70 |
| GitHub Repos | 10-30 files/min | ~200MB | 75-85 |

*Your mileage may vary based on internet speed, site structure, and settings*

---

## Getting Help

### Debug Checklist

Before asking for help, try these steps:

1. **Check the console log**
   - Look for error messages
   - Note which URLs are failing
   - See if there's a pattern

2. **Verify your settings**
   - Is depth reasonable? (2-3 is safe)
   - Are workers too high? (>30 can cause issues)
   - Are keywords too restrictive?

3. **Test with a simple case**
   - Try 1-2 URLs only
   - Use "Generic/Auto-detect"
   - Lower quality threshold to 30

4. **Check the output file**
   - Does `scraped_data.jsonl` exist?
   - Is it empty or does it have data?
   - Is each line valid JSON?

### Reporting Bugs

If you found a genuine bug, open an issue on GitHub with:

1. Exact error message from console
2. Your configuration (JSON file)
3. Example URLs (if possible)
4. What you expected vs. what happened
5. Python version and OS

---

## Acknowledgments

**Built With**:
- [DearPyGUI](https://github.com/hoffstadt/DearPyGui) - GPU-accelerated GUI framework
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing
- [urllib](https://docs.python.org/3/library/urllib.html) - HTTP requests
- [GitPython](https://github.com/gitpython-developers/GitPython) - Repository handling

**Inspired By**:
- [Common Crawl](https://commoncrawl.org/) - Large-scale web archiving
- [Hugging Face Datasets](https://huggingface.co/datasets) - Dataset standards
- The need for better fine-tuning data

---

## Changelog

### v2026.05.Pro+Enhanced+ContentTypes+Codebase

**Added**:
- Codebase processing (local folders + GitHub repos)
- RAG-style code chunking
- Content-type specific configurations
- Enhanced quality filtering algorithm
- Bloom filter for memory efficiency
- Subdomain discovery feature
- Short high-quality content support

**Improved**:
- Better HTML parsing with multiple strategies
- Enhanced URL normalization
- More robust error handling
- Smarter link discovery

**Fixed**:
- Relative URL handling
- Encoding detection
- Memory leaks on large crawls
- Rate limiting issues

---

## License

This tool is provided as-is for **educational and research purposes**. 

**You are responsible for**:
- Complying with website terms of service
- Respecting copyright laws
- Using scraped data ethically
- Following applicable regulations

No warranty provided. Use at your own risk.

---

## Contributing

Contributions welcome. Whether it's bug fixes, new features, documentation improvements, test cases, or ideas and suggestions - open an issue or submit a pull request on GitHub.

---

## Support This Project

If NTCompanion helped you build better AI models:

- Star the repo on GitHub
- Share it with others working on fine-tuning
- Report bugs and suggest improvements
- Improve documentation
- Join discussions in Issues

---

**Built for the AI community**

*Making fine-tuning accessible, one dataset at a time.*


</div>
