import requests
from bs4 import BeautifulSoup
import re
import time
import os
import tempfile
import pandas as pd
import html

def clean_detik_url(url):
    """
    Cleans an article URL and appends ?single=1 or &single=1 to load the full article on one page.
    """
    url = url.split('#')[0]
    if '?' in url:
        if 'single=1' not in url:
            url += '&single=1'
    else:
        url += '?single=1'
    return url

def is_valid_detik_news_link(href):
    """
    Validates if a URL is a Detik news article link containing 'berita' or '/d-' article ID,
    excluding video, photo, tv, and infografis URLs.
    """
    if not href or not isinstance(href, str):
        return False
    
    # Must be absolute http/https link
    if not (href.startswith('http://') or href.startswith('https://')):
        return False

    # Skip non-article sections
    bad_keywords = ['/detiktv/', '/foto/', '/fotohealth/', '/infografis/', '/video/', '/tv/', '/wawancara-khusus/']
    if any(bad in href for bad in bad_keywords):
        return False

    # Must contain berita or article ID /d-
    if ('/berita' in href or 'berita-' in href or '/d-' in href):
        return True

    return False

def discover_detik_tag_links(tag, target_count=100, progress_callback=None):
    """
    Crawls tag pages (https://www.detik.com/tag/{tag}/?sortby=time&page={num})
    and extracts unique news article links up to target_count.
    """
    clean_tag = tag.strip().lower().replace(' ', '-')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    found_links = []
    page = 1
    max_pages = 50 # Safeguard max pages
    
    if isinstance(target_count, str) and 'all' in target_count.lower():
        max_articles = 500
    else:
        try:
            max_articles = int(target_count)
        except Exception:
            max_articles = 100

    while len(found_links) < max_articles and page <= max_pages:
        page_url = f"https://www.detik.com/tag/{clean_tag}/?sortby=time&page={page}"
        if progress_callback:
            progress_callback(f"Crawling Detik.com Tag Page {page}...", min(0.3, (page / 15)))

        try:
            res = requests.get(page_url, headers=headers, timeout=10)
            if res.status_code != 200:
                break
            
            soup = BeautifulSoup(res.content, 'html.parser')
            article_anchors = soup.find_all('a', href=True)
            
            new_links_in_page = 0
            for a in article_anchors:
                href = a['href']
                if is_valid_detik_news_link(href):
                    cleaned_link = clean_detik_url(href)
                    if cleaned_link not in found_links:
                        found_links.append(cleaned_link)
                        new_links_in_page += 1
                        if len(found_links) >= max_articles:
                            break

            # If no new links found on this page, end pagination
            if new_links_in_page == 0:
                break
                
            page += 1
            time.sleep(0.3) # Polite crawl delay
        except Exception as e:
            print(f"[WARN] Error crawling page {page}: {e}")
            break

    return found_links

def scrape_detik_article(url, headers=None):
    """
    Scrapes title, subtitle, author, date, and body paragraphs from a single Detik article URL.
    """
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    clean_url = clean_detik_url(url)
    
    try:
        res = requests.get(clean_url, headers=headers, timeout=12)
        if res.status_code != 200:
            return None
        
        soup = BeautifulSoup(res.content, 'html.parser')
        
        title_el = soup.find('h1', class_=lambda c: c and 'detail__title' in c) or soup.find('h1')
        subtitle_el = soup.find(class_=lambda c: c and 'detail__subtitle' in c)
        author_el = soup.find(class_=lambda c: c and 'detail__author' in c)
        date_el = soup.find(class_=lambda c: c and 'detail__date' in c)

        title = title_el.get_text(strip=True) if title_el else ''
        subtitle = subtitle_el.get_text(strip=True) if subtitle_el else ''
        author = author_el.get_text(strip=True) if author_el else ''
        date = date_el.get_text(strip=True) if date_el else ''

        body = soup.find('div', class_=lambda c: c and 'detail__body-text' in c) or soup.find('div', class_=lambda c: c and 'itp_bodycontent' in c)
        paragraphs = []
        if body:
            # Remove scripts, styles, embedded divs, ads
            for el in body.find_all(['script', 'style', 'iframe', 'ins']):
                el.decompose()
            for el in body.find_all('div'):
                if hasattr(el, 'attrs') and el.attrs and 'class' in el.attrs:
                    cls_list = el.attrs['class']
                    cls_str = ' '.join(cls_list) if isinstance(cls_list, list) else str(cls_list)
                    if any(bad in cls_str for bad in ['sisipan', 'detail__media', 'link_baca_juga', 'parallax', 'banner', 'detail__body-tag']):
                        el.decompose()
            for p in body.find_all('p'):
                txt = p.get_text(strip=True)
                if txt and not txt.startswith('Simak Video') and not txt.startswith('Saksikan Video'):
                    paragraphs.append(txt)

        return {
            'url': clean_url,
            'title': title,
            'subtitle': subtitle,
            'author': author,
            'date': date,
            'paragraphs': paragraphs
        }
    except Exception as e:
        print(f"[ERROR] Failed to scrape article {url}: {e}")
        return None

def build_detik_corpus_xml(tag, target_count=100, progress_callback=None):
    """
    Crawls Detik tag, scrapes articles, and packages them into an annotated XML string.
    Returns: (xml_content, df_summary, total_articles)
    """
    links = discover_detik_tag_links(tag, target_count=target_count, progress_callback=progress_callback)
    if not links:
        return None, pd.DataFrame(), 0

    articles_data = []
    xml_parts = []
    escaped_tag = html.escape(str(tag), quote=True)
    xml_parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
    xml_parts.append(f'<corpus tag="{escaped_tag}" source="Detik.com">')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    total_links = len(links)
    for idx, link in enumerate(links):
        if progress_callback:
            pct = 0.3 + (0.65 * ((idx + 1) / total_links))
            progress_callback(f"Scraping article {idx + 1}/{total_links}: {link[:60]}...", pct)
            
        art = scrape_detik_article(link, headers=headers)
        if art and art['paragraphs']:
            escaped_url = html.escape(art['url'], quote=True)
            escaped_title = html.escape(art['title'], quote=True)
            escaped_subtitle = html.escape(art['subtitle'], quote=True)
            escaped_author = html.escape(art['author'], quote=True)
            escaped_date = html.escape(art['date'], quote=True)

            paragraphs_xml = "\n".join([f"    <p>{html.escape(p)}</p>" for p in art['paragraphs']])
            
            art_xml = f"""  <detik id="detik_{idx+1}" article="{escaped_title}" title="{escaped_title}" author="{escaped_author}" date="{escaped_date}" subtitle="{escaped_subtitle}" url="{escaped_url}">
    <subtitle>{html.escape(art['subtitle'])}</subtitle>
    <title>{html.escape(art['title'])}</title>
    <author>{html.escape(art['author'])}</author>
    <date>{html.escape(art['date'])}</date>
    <text>
{paragraphs_xml}
    </text>
  </detik>"""
            xml_parts.append(art_xml)
            
            full_text = ""
            if art['subtitle']: full_text += art['subtitle'] + " "
            if art['title']: full_text += art['title'] + " "
            full_text += " ".join(art['paragraphs'])
            words_count = len(full_text.split())

            articles_data.append({
                'ID': f"detik_{idx+1}",
                'Title': art['title'],
                'Subtitle': art['subtitle'],
                'Author': art['author'],
                'Date': art['date'],
                'Words': words_count,
                'URL': art['url']
            })
        
        time.sleep(0.2) # Polite delay between articles

    xml_parts.append('</corpus>')
    full_xml = "\n".join(xml_parts)
    df_summary = pd.DataFrame(articles_data)

    if progress_callback:
        progress_callback("Detik.com corpus build complete!", 1.0)

    return full_xml, df_summary, len(articles_data)
