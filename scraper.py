"""
Computer Parts Web Scraper Suite
Scrapes data from multiple sources for training your LLM
"""

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os

# CONSTANTS
OUTPUT_FILE = "data.txt"
HTML_PARSER = "html.parser"
DEFAULT_RATE_LIMIT = 2

class ComputerPartsScraper:
    def __init__(self, output_file=OUTPUT_FILE):
        self.output_file = output_file
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def save_to_file(self, text, source):
        """Append scraped text to data.txt with source annotation"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Source: {source} ---\n")
            f.write(text)
            f.write("\n\n")
        print(f"✓ Saved {len(text)} characters from {source}")
    
    def rate_limit(self, seconds=DEFAULT_RATE_LIMIT):
        """Be respectful with rate limiting"""
        time.sleep(seconds)
    
    def _parse_pcpartpicker_product(self, product, category):
        """Helper function to parse a single product - Fix for S3776 (complexity)"""
        try:
            name_elem = product.find('td', class_='td__name')
            if not name_elem:
                return None
                
            name = name_elem.get_text(strip=True)
            
            # Get specs if available
            specs_elem = product.find('td', class_='td__spec')
            specs = specs_elem.get_text(strip=True) if specs_elem else ""
            
            # Get price if available
            price_elem = product.find('td', class_='td__price')
            price = price_elem.get_text(strip=True) if price_elem else ""
            
            text = f"Product: {name}\n"
            if specs:
                text += f"Specifications: {specs}\n"
            if price:
                text += f"Price: {price}\n"
            
            return text
        except Exception as e:
            print(f"Error parsing product: {e}")
            return None
    
    # ===== PCPARTPICKER SCRAPER =====
    def scrape_pcpartpicker_products(self, category="cpu", max_pages=50):
        """Scrape product listings from PCPartPicker"""
        print(f"\n🔍 Scraping PCPartPicker {category}...")
        
        categories = {
            "cpu": "https://pcpartpicker.com/products/cpu/",
            "video-card": "https://pcpartpicker.com/products/video-card/",
            "memory": "https://pcpartpicker.com/products/memory/",
            "motherboard": "https://pcpartpicker.com/products/motherboard/",
            "power-supply": "https://pcpartpicker.com/products/power-supply/",
            "case": "https://pcpartpicker.com/products/case/",
            "internal-hard-drive": "https://pcpartpicker.com/products/internal-hard-drive/"
        }
        
        base_url = categories.get(category, categories["cpu"])
        
        for page in range(1, max_pages + 1):
            try:
                url = f"{base_url}#page={page}"
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.content, HTML_PARSER)
                
                # Find product rows
                products = soup.find_all('tr', class_='tr__product')
                
                for product in products[:10]:  # Limit per page
                    product_text = self._parse_pcpartpicker_product(product, category)
                    if product_text:
                        self.save_to_file(product_text, f"PCPartPicker-{category}")
                
                print(f"✓ Scraped page {page}/{max_pages}")
                self.rate_limit()
                
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
                continue
    
    # ===== REDDIT SCRAPER =====
    def scrape_reddit_buildapc(self, max_posts=100):
        """Scrape r/buildapc posts and comments"""
        print("\n🔍 Scraping Reddit r/buildapc...")
        
        try:
            # Use Reddit's JSON API (no auth needed for public posts)
            url = "https://www.reddit.com/r/buildapc/hot.json?limit=100"
            response = self.session.get(url, timeout=10)
            data = response.json()
            
            posts = data['data']['children'][:max_posts]
            
            for post in posts:
                try:
                    post_data = post['data']
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    
                    if selftext and len(selftext) > 50:  # Skip short posts
                        text = f"Title: {title}\n\nQuestion/Post: {selftext}\n"
                        self.save_to_file(text, "Reddit-buildapc")
                    
                except Exception as e:
                    print(f"Error parsing Reddit post: {e}")
                    continue
            
            print(f"✓ Scraped {len(posts)} Reddit posts")
            self.rate_limit(3)
            
        except Exception as e:
            print(f"Error scraping Reddit: {e}")
    
    def _parse_newegg_item(self, item, search_term):
        """Helper function to parse a single Newegg item - Fix for S3776"""
        try:
            title_elem = item.find('a', class_='item-title')
            if not title_elem:
                return None
                
            title = title_elem.get_text(strip=True)
            
            # Get price
            price_elem = item.find('li', class_='price-current')
            price = price_elem.get_text(strip=True) if price_elem else ""
            
            # Get specs list
            specs_list = item.find('ul', class_='item-features')
            specs = ""
            if specs_list:
                specs = " ".join([li.get_text(strip=True) for li in specs_list.find_all('li')])
            
            text = f"Product: {title}\n"
            if price:
                text += f"Price: {price}\n"
            if specs:
                text += f"Features: {specs}\n"
            
            return text
        except Exception as e:
            print(f"Error parsing Newegg item: {e}")
            return None
    
    # ===== NEWEGG SCRAPER =====
    def scrape_newegg_products(self, search_term="graphics card", max_results=15):
        """Scrape Newegg product listings"""
        print(f"\n🔍 Scraping Newegg for '{search_term}'...")
        
        try:
            # Newegg search results
            search_url = f"https://www.newegg.com/p/pl?d={search_term.replace(' ', '+')}"
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.content, HTML_PARSER)
            
            # Find product items
            items = soup.find_all('div', class_='item-cell')[:max_results]
            
            for item in items:
                item_text = self._parse_newegg_item(item, search_term)
                if item_text:
                    self.save_to_file(item_text, f"Newegg-{search_term}")
            
            print("✓ Scraped Newegg results")
            self.rate_limit()
            
        except Exception as e:
            print(f"Error scraping Newegg: {e}")
    
    # ===== WIKIPEDIA TECH ARTICLES =====
    def scrape_wikipedia_tech(self, topics=None):
        """Scrape Wikipedia articles about computer hardware"""
        print("\n🔍 Scraping Wikipedia tech articles...")
        
        if topics is None:
            topics = [
                "Graphics_processing_unit",
                "Central_processing_unit", 
                "Random-access_memory",
                "Solid-state_drive",
                "Computer_cooling",
                "Motherboard",
                "Power_supply_unit_(computer)"
            ]
        
        for topic in topics:
            try:
                url = f"https://en.wikipedia.org/wiki/{topic}"
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.content, HTML_PARSER)
                
                # Get main content
                content = soup.find('div', {'id': 'mw-content-text'})
                if content:
                    # Get paragraphs
                    paragraphs = content.find_all('p', limit=100)  # First 10 paragraphs
                    text = f"Article: {topic.replace('_', ' ')}\n\n"
                    text += "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 100])
                    
                    if len(text) > 200:
                        self.save_to_file(text, f"Wikipedia-{topic}")
                
                print(f"✓ Scraped Wikipedia: {topic}")
                self.rate_limit()
                
            except Exception as e:
                print(f"Error scraping Wikipedia {topic}: {e}")
                continue
    
    # ===== SAMPLE DATA GENERATOR =====
    def generate_sample_data(self):
        """Generate synthetic Q&A data about computer parts"""
        print("\n🔍 Generating sample Q&A data...")
        
        sample_data = [
            {
                "q": "What's the difference between DDR4 and DDR5 RAM?",
                "a": "DDR5 RAM offers higher speeds (up to 6400MHz+) compared to DDR4 (typically 2133-3200MHz). DDR5 also has improved power efficiency with lower voltage (1.1V vs 1.2V) and higher bandwidth. However, DDR5 typically has higher latency and is more expensive. For most gaming and everyday use, DDR4 is still excellent value."
            },
            {
                "q": "How much power supply wattage do I need for an RTX 4090?",
                "a": "NVIDIA recommends an 850W power supply for the RTX 4090, but many users opt for 1000W or higher for headroom and efficiency. The 4090 has a 450W TDP, so you need to account for your CPU (100-250W), motherboard, drives, and other components. A quality 1000W PSU from brands like Corsair, EVGA, or Seasonic is recommended."
            },
            {
                "q": "Is 16GB RAM enough for gaming in 2024?",
                "a": "16GB RAM is still adequate for most gaming in 2024, but 32GB is becoming the recommended amount for new builds. Some modern games like Star Citizen, Cyberpunk 2077, and flight simulators can use more than 16GB. If you multitask (gaming while streaming/browsing), 32GB provides much better headroom. DDR4 16GB kits are affordable, making 32GB a good value upgrade."
            },
            {
                "q": "What CPU should I pair with an RTX 4070?",
                "a": "For an RTX 4070, good CPU pairings include: Intel Core i5-13600K or i5-14600K for excellent gaming performance, AMD Ryzen 5 7600X or 7700X for great value, or Intel Core i7-13700K/i7-14700K if you also do productivity work. Avoid pairing with older CPUs like the i5-10400 or Ryzen 3600 as they may bottleneck the GPU in CPU-intensive games at 1080p."
            },
            {
                "q": "Do I need a CPU cooler or can I use the stock one?",
                "a": "Intel K-series CPUs and AMD Ryzen 5 7600X and above don't include stock coolers. Intel non-K chips include basic coolers that work but are loud under load. AMD's stock coolers (Wraith) are better but still noisy. For best results, invest in a tower cooler like the Thermalright Peerless Assassin 120 ($35) or Arctic Freezer 34 eSports. High-end CPUs benefit from 280mm+ AIOs."
            }
        ]
        
        for item in sample_data:
            text = f"Question: {item['q']}\n\nAnswer: {item['a']}\n"
            self.save_to_file(text, "Sample-QA")
        
        print("✓ Generated sample data")


def main():
    print("=" * 60)
    print(" COMPUTER PARTS DATA SCRAPER")
    print("=" * 60)
    
    # Initialize scraper
    scraper = ComputerPartsScraper(output_file=OUTPUT_FILE)
    
    # Backup existing data.txt if it exists
    if os.path.exists(OUTPUT_FILE):
        backup_name = f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.rename(OUTPUT_FILE, backup_name)
        print(f"Backed up existing data.txt to {backup_name}")
    
    print("\nStarting data collection...\n")
    
    # 1. Generate sample Q&A data first (quick and useful)
    scraper.generate_sample_data()
    
    # 2. Scrape Wikipedia articles (reliable, factual)
    scraper.scrape_wikipedia_tech()
    
    # 3. Scrape Reddit (conversational data)
    scraper.scrape_reddit_buildapc(max_posts=200)
    
    # 4. Scrape PCPartPicker (structured product data)
    categories = ["cpu", "video-card", "memory", "motherboard"]
    for cat in categories:
        scraper.scrape_pcpartpicker_products(category=cat, max_pages=3)
    
    # 5. Scrape Newegg
    search_terms = ["RTX 4090", "Intel Core i9", "DDR5 RAM", "NVMe SSD"]
    for term in search_terms:
        scraper.scrape_newegg_products(search_term=term, max_results=200)
    
    # Check final file size
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            char_count = len(content)
            word_count = len(content.split())
        
        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE!")
        print("=" * 60)
        print(f"File size: {file_size / 1024:.2f} KB ({file_size / (1024*1024):.2f} MB)")
        print(f"Characters: {char_count:,}")
        print(f"Words: {word_count:,}")
        print(f"Estimated tokens: {word_count * 1.3:.0f}")
        print("\nNext steps:")
        # print("   1. Run 'python train_tokenizer.py' to retrain your tokenizer")
        # print("   2. Run 'python preprocess_data.py' to tokenize the new data")
        # print("   3. Run 'python train_model.py' to train your model")
    
if __name__ == "__main__":
    main()