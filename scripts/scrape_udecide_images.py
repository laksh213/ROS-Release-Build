import asyncio
import json
import os
import re
import urllib.parse
from pathlib import Path
import requests
from PIL import Image
from playwright.async_api import async_playwright

# Setup directories
REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = REPO_ROOT / "data" / "udecide_images"
MANIFEST_PATH = IMAGE_DIR / "manifest.json"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Search queries
QUERIES = [
    "UDecide Sri Lanka mediation",
    "UDecideSL mediation",
    "UDecide Mediation platform",
    "udecide_sl",
    "Saranee Gunathilaka UDecide"
]

async def scrape_images_from_bing(queries):
    metadata_list = []
    seen_urls = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # We can run parallel pages or sequential
        for query in queries:
            print(f"Scraping query: '{query}'...")
            try:
                page = await browser.new_page()
                await page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                })
                url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}"
                await page.goto(url)
                await page.wait_for_timeout(4000)
                
                # Scroll down to load more images
                for _ in range(3):
                    await page.mouse.wheel(0, 3000)
                    await page.wait_for_timeout(1000)
                
                # Parse image metadata from a.iusc elements
                results = await page.evaluate("""
                    () => {
                        const links = Array.from(document.querySelectorAll('a.iusc'));
                        return links.map(link => {
                            try {
                                const metadata = JSON.parse(link.getAttribute('m'));
                                return {
                                    murl: metadata.murl, // original image
                                    turl: metadata.turl, // thumbnail
                                    page_url: metadata.purl, // page containing image
                                    title: metadata.title || ""
                                };
                            } catch (e) {
                                return null;
                            }
                        }).filter(x => x !== null);
                    }
                """)
                
                print(f"  Query '{query}' returned {len(results)} image metadata objects.")
                for r in results:
                    murl = r['murl']
                    if murl not in seen_urls:
                        seen_urls.add(murl)
                        metadata_list.append(r)
                
                await page.close()
            except Exception as e:
                print(f"  Error scraping query '{query}': {e}")
                
        await browser.close()
        
    return metadata_list

def download_images(metadata_list):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    manifest = []
    counter = 1
    
    for i, meta in enumerate(metadata_list, 1):
        url = meta['murl']
        print(f"[{i}/{len(metadata_list)}] Downloading {url} ...")
        try:
            # We must be careful not to crash on network timeouts or invalid content
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"  Failed with status {resp.status_code}")
                continue
                
            # Deduce extension from Content-Type or URL
            content_type = resp.headers.get("Content-Type", "")
            ext = ".jpg" # default
            if "png" in content_type:
                ext = ".png"
            elif "gif" in content_type:
                ext = ".gif"
            elif "webp" in content_type:
                ext = ".webp"
            elif "jpeg" in content_type:
                ext = ".jpg"
            else:
                # Try getting from URL
                path_part = urllib.parse.urlparse(url).path
                _, url_ext = os.path.splitext(path_part)
                if url_ext.lower() in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                    ext = url_ext.lower()
            
            temp_filename = f"image_{counter}{ext}"
            temp_path = IMAGE_DIR / temp_filename
            
            # Save the image content
            temp_path.write_bytes(resp.content)
            
            # Verify and identify with Pillow
            try:
                with Image.open(temp_path) as img:
                    width, height = img.size
                    img_format = img.format
                
                # Rename to match counter + verified extension
                final_ext = f".{img_format.lower()}" if img_format else ext
                if final_ext == ".jpeg":
                    final_ext = ".jpg"
                final_filename = f"image_{counter}{final_ext}"
                final_path = IMAGE_DIR / final_filename
                
                if temp_path != final_path:
                    if final_path.exists():
                        final_path.unlink()
                    temp_path.rename(final_path)
                
                print(f"  Successfully saved {final_filename} ({img_format}, {width}x{height})")
                
                # Add to manifest
                manifest.append({
                    "id": counter,
                    "local_filename": final_filename,
                    "title": meta['title'],
                    "original_url": url,
                    "source_page_url": meta['page_url'],
                    "width": width,
                    "height": height,
                    "format": img_format
                })
                counter += 1
                
            except Exception as pillow_err:
                print(f"  Invalid image file downloaded: {pillow_err}")
                if temp_path.exists():
                    temp_path.unlink()
                    
        except Exception as e:
            print(f"  Error downloading or validating image: {e}")
            
    # Save manifest file
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"\nScraping complete. Downloaded {len(manifest)} valid images.")
    print(f"Manifest saved to: {MANIFEST_PATH}")

async def main():
    metadata = await scrape_images_from_bing(QUERIES)
    print(f"Found {len(metadata)} unique image URLs.")
    # Download images
    download_images(metadata)

if __name__ == "__main__":
    asyncio.run(main())
