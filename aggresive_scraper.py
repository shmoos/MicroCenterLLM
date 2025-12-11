"""
Aggressive Computer Parts Data Scraper
Goal: Collect 500KB-2MB of quality training data
"""

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os

OUTPUT_FILE = "data.txt"
HTML_PARSER = "html.parser"

class AggressivePartsScraper:
    def __init__(self, output_file=OUTPUT_FILE):
        self.output_file = output_file
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.total_chars = 0
        
    def save_to_file(self, text, source):
        """Append scraped text to data.txt"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Source: {source} ---\n")
            f.write(text)
            f.write("\n\n")
        self.total_chars += len(text)
        print(f"✓ Saved {len(text)} chars from {source} | Total: {self.total_chars/1024:.1f}KB")
    
    def rate_limit(self, seconds=2):
        """Rate limiting"""
        time.sleep(seconds)
    
    # ===== REDDIT SCRAPER (BEST SOURCE) =====
    def scrape_reddit_comprehensive(self):
        """Scrape multiple PC building subreddits"""
        print("\n🔍 Scraping Reddit comprehensively...")
        
        subreddits = [
            "buildapc",
            "pcmasterrace", 
            "buildapcforme",
            "hardware",
            "nvidia",
            "Amd",
            "intel"
        ]
        
        for subreddit in subreddits:
            print(f"\n  → Scraping r/{subreddit}...")
            try:
                # Get hot posts
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=100"
                response = self.session.get(url, timeout=10)
                data = response.json()
                
                posts = data['data']['children']
                
                for post in posts:
                    try:
                        post_data = post['data']
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        
                        # Save posts with substantial content
                        if selftext and len(selftext) > 100:
                            text = f"Title: {title}\n\nPost: {selftext}\n"
                            self.save_to_file(text, f"Reddit-r/{subreddit}")
                        
                    except Exception as e:
                        continue
                
                self.rate_limit(3)
                
            except Exception as e:
                print(f"  ✗ Error scraping r/{subreddit}: {e}")
                continue
    
    # ===== WIKIPEDIA COMPREHENSIVE =====
    def scrape_wikipedia_comprehensive(self):
        """Scrape extensive Wikipedia articles"""
        print("\n🔍 Scraping Wikipedia comprehensively...")
        
        topics = [
            # Core components
            "Graphics_processing_unit",
            "Central_processing_unit",
            "Random-access_memory",
            "Solid-state_drive",
            "Hard_disk_drive",
            "Motherboard",
            "Power_supply_unit_(computer)",
            "Computer_case",
            
            # Specific technologies
            "DDR4_SDRAM",
            "DDR5_SDRAM",
            "PCI_Express",
            "NVMe",
            "SATA",
            "USB",
            "Thunderbolt_(interface)",
            
            # Manufacturers/Products
            "Nvidia",
            "AMD",
            "Intel",
            "GeForce",
            "Radeon",
            
            # Gaming/Performance
            "PC_game",
            "Computer_cooling",
            "Overclocking",
            "Computer_performance",
            "Gaming_computer",
            "Workstation"
        ]
        
        for topic in topics:
            try:
                url = f"https://en.wikipedia.org/wiki/{topic}"
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.content, HTML_PARSER)
                
                content = soup.find('div', {'id': 'mw-content-text'})
                if content:
                    # Get MORE paragraphs
                    paragraphs = content.find_all('p', limit=20)
                    text = f"Article: {topic.replace('_', ' ')}\n\n"
                    text += "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 100])
                    
                    if len(text) > 500:
                        self.save_to_file(text, f"Wikipedia-{topic}")
                
                print(f"  ✓ {topic}")
                self.rate_limit(2)
                
            except Exception as e:
                print(f"  ✗ {topic}: {e}")
                continue
    
    # ===== SYNTHETIC Q&A GENERATOR =====
    def generate_comprehensive_qa(self):
        """Generate extensive synthetic Q&A data"""
        print("\n🔍 Generating comprehensive Q&A data...")
        
        qa_data = [
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
                "a": "16GB RAM is still adequate for most gaming in 2024, but 32GB is becoming the recommended amount for new builds. Some modern games like Star Citizen, Cyberpunk 2077, and flight simulators can use more than 16GB. If you multitask (gaming while streaming/browsing), 32GB provides much better headroom."
            },
            {
                "q": "What CPU should I pair with an RTX 4070?",
                "a": "For an RTX 4070, good CPU pairings include Intel Core i5-13600K or i5-14600K for excellent gaming performance, AMD Ryzen 5 7600X or 7700X for great value, or Intel Core i7-13700K/i7-14700K if you also do productivity work. Avoid pairing with older CPUs as they may bottleneck the GPU."
            },
            {
                "q": "Do I need a CPU cooler or can I use the stock one?",
                "a": "Intel K-series CPUs and AMD Ryzen 5 7600X and above don't include stock coolers. Intel non-K chips include basic coolers that work but are loud under load. AMD's stock coolers are better but still noisy. For best results, invest in a tower cooler like the Thermalright Peerless Assassin 120 or Arctic Freezer 34 eSports."
            },
            {
                "q": "What's the difference between NVMe and SATA SSDs?",
                "a": "NVMe SSDs use the PCIe interface and offer much faster speeds (3500-7000 MB/s) compared to SATA SSDs (up to 550 MB/s). NVMe drives are ideal for OS installation, frequently used programs, and content creation. SATA SSDs are still great for mass storage and games, where the speed difference is less noticeable."
            },
            {
                "q": "How do I choose a motherboard?",
                "a": "Choose a motherboard based on: 1) CPU socket compatibility (LGA1700 for Intel 12th-14th gen, AM5 for AMD Ryzen 7000), 2) Form factor (ATX, Micro-ATX, Mini-ITX), 3) RAM support (DDR4 vs DDR5), 4) Expansion slots (PCIe for GPU, M.2 for storage), 5) Connectivity (USB, networking), and 6) VRM quality for overclocking."
            },
            {
                "q": "What's the best GPU for 1440p gaming?",
                "a": "For 1440p gaming, the RTX 4070 or AMD RX 7800 XT offer excellent performance at around $500-600. For higher framerates or ray tracing, consider the RTX 4070 Ti or RX 7900 XT. Budget options include the RTX 4060 Ti or RX 7700 XT. These can handle most games at high-ultra settings at 60+ fps."
            },
            {
                "q": "How much storage do I need for a gaming PC?",
                "a": "A typical gaming PC needs at least 500GB-1TB for the OS and frequently played games. Modern AAA games can be 50-150GB each. Recommended setup: 500GB-1TB NVMe SSD for OS and main games, plus 1-2TB SATA SSD or HDD for additional game storage and files. Power users might want 2TB+ of NVMe storage."
            },
            {
                "q": "What's thermal paste and do I need to replace it?",
                "a": "Thermal paste is a heat-conductive compound applied between the CPU and cooler to improve heat transfer. Most coolers come with pre-applied paste or include a tube. You should replace thermal paste when: 1) Installing a new cooler, 2) CPU temps increase over time (usually after 3-5 years), or 3) After removing the cooler for any reason."
            },
            {
                "q": "Is liquid cooling better than air cooling?",
                "a": "Liquid cooling (AIO or custom loop) can offer better cooling for high-end CPUs and looks cleaner, but quality air coolers like the Noctua NH-D15 or Thermalright Peerless Assassin perform similarly to 240-280mm AIOs at lower cost. Liquid cooling advantages: better aesthetics, works in smaller cases. Air cooling advantages: more reliable, quieter, cheaper, easier maintenance."
            },
            {
                "q": "What's the difference between Intel and AMD CPUs?",
                "a": "Intel CPUs (12th-14th gen Core) generally offer slightly better gaming performance and single-thread speeds, while AMD Ryzen (5000/7000 series) often provides better multi-core performance and value. AMD's AM5 platform promises longer upgrade path. Both are excellent for gaming. Choose based on budget, use case (gaming vs productivity), and motherboard ecosystem."
            },
            {
                "q": "How important is RAM speed for gaming?",
                "a": "RAM speed matters, but with diminishing returns. For Intel, 3200-3600MHz DDR4 or 5600MHz+ DDR5 is ideal. AMD Ryzen benefits more from faster RAM due to Infinity Fabric; aim for 3600MHz CL16 DDR4 or 6000MHz DDR5. Going beyond these speeds shows minimal gaming improvement (1-5 fps) but helps in productivity workloads."
            },
            {
                "q": "What's the best CPU for gaming and streaming?",
                "a": "For gaming and streaming, you need strong multi-core performance. Best options: AMD Ryzen 7 7700X/7800X3D (8 cores), Intel Core i7-13700K/14700K (16 cores with E-cores), or Ryzen 9 7900X (12 cores) for heavy streaming. The extra cores handle encoding while gaming. Alternatively, use NVENC (NVIDIA GPU encoding) with a mid-range CPU like i5-13600K or Ryzen 5 7600X."
            },
            {
                "q": "What's PCIe 4.0 vs 5.0?",
                "a": "PCIe 5.0 offers double the bandwidth of PCIe 4.0 (32 GT/s vs 16 GT/s). For GPUs, PCIe 4.0 x16 is still more than enough for even RTX 4090. PCIe 5.0 is most relevant for next-gen NVMe SSDs reaching 10-14 GB/s speeds. Most users don't need PCIe 5.0 yet, but it provides future-proofing for upcoming hardware."
            }
        ]
        
        for item in qa_data:
            text = f"Question: {item['q']}\n\nAnswer: {item['a']}\n"
            self.save_to_file(text, "Synthetic-QA")


def main():
    print("=" * 70)
    print("AGGRESSIVE COMPUTER PARTS DATA COLLECTION")
    print("=" * 70)
    print("\nGoal: Collect 500KB-2MB of training data")
    print("This will take 15-30 minutes...\n")
    
    scraper = AggressivePartsScraper(output_file=OUTPUT_FILE)
    
    # Backup existing file
    if os.path.exists(OUTPUT_FILE):
        backup = f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.rename(OUTPUT_FILE, backup)
        print(f" Backed up existing data to {backup}\n")
    
    # 1. Generate synthetic Q&A (quick, high quality)
    scraper.generate_comprehensive_qa()
    
    # 2. Wikipedia (reliable, substantial)
    scraper.scrape_wikipedia_comprehensive()
    
    # 3. Reddit (conversational, real-world)
    scraper.scrape_reddit_comprehensive()
    
    # Final stats
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            word_count = len(content.split())
            char_count = len(content)
        
        print("\n" + "=" * 70)
        print(" DATA COLLECTION COMPLETE!")
        print("=" * 70)
        print(f" File size: {file_size/1024:.1f} KB ({file_size/(1024*1024):.2f} MB)")
        print(f" Characters: {char_count:,}")
        print(f" Words: {word_count:,}")
        print(f" Estimated tokens: ~{int(word_count * 1.3):,}")
        
        # Recommendations
        print("\n💡 ASSESSMENT:")
        if file_size < 200_000:
            print("⚠️  < 200KB: Consider running scraper again or adding more sources")
        elif file_size < 500_000:
            print("✓ 200-500KB: Good for a small model (20-40M params)")
        else:
            print("✓✓ 500KB+: Excellent! Can train a 50-100M param model")
        
        print("\n NEXT STEPS:")
        print("1. python train_tokenizer.py")
        print("2. python preprocess_data.py")
        print("3. python train_model_optimized.py")

if __name__ == "__main__":
    main()