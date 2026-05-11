"""
get_tickers.py
Fetches all available US stock tickers from multiple free sources
and saves a deduplicated list to a text file.

Usage:
    pip install requests pandas
    python get_tickers.py --out tickers.txt
"""

import argparse
import requests
import pandas as pd
import re
import time

def clean_tickers(lst):
    cleaned = []
    for t in lst:
        t = re.sub(r"\[.*?\]", "", str(t)).strip()
        if 1 <= len(t) <= 5 and t.replace("-", "").replace(".", "").isalpha():
            cleaned.append(t.upper())
    return cleaned

def from_nasdaq_trader():
    """
    NASDAQ trader FTP — lists all NASDAQ, NYSE, AMEX tickers.
    This is the most reliable free source, ~10,000+ tickers.
    """
    tickers = []
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            for line in lines[1:]:          # skip header
                parts = line.split("|")
                if len(parts) > 0:
                    t = parts[0].strip()
                    tickers.extend(clean_tickers([t]))
            print(f"  NASDAQ trader ({url.split('/')[-1]}): {len(tickers)} so far")
        except Exception as e:
            print(f"  Warning: NASDAQ trader fetch failed: {e}")
    return tickers

def from_github_symbols():
    """
    GitHub repo that maintains a clean list of all US stock symbols.
    ~8,000 tickers, updated regularly.
    """
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        tickers = clean_tickers(r.text.strip().split("\n"))
        print(f"  GitHub symbols: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"  Warning: GitHub symbols fetch failed: {e}")
        return []

def from_sec_edgar():
    """
    SEC EDGAR company tickers JSON — all companies registered with the SEC.
    ~13,000 entries but includes many OTC/pink sheet names.
    """
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "ticker-fetcher contact@example.com"}
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()
        data = r.json()
        tickers = clean_tickers([v["ticker"] for v in data.values()])
        print(f"  SEC EDGAR: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"  Warning: SEC EDGAR fetch failed: {e}")
        return []

def from_iex_cloud():
    """
    IEX Cloud public symbols endpoint — no API key needed for this endpoint.
    """
    try:
        url = "https://api.iex.cloud/v1/data/core/ref_data_symbols?token=pk_test"
        r = requests.get(url, timeout=15)
        data = r.json()
        if isinstance(data, list):
            tickers = clean_tickers([d.get("symbol","") for d in data])
            print(f"  IEX Cloud: {len(tickers)} tickers")
            return tickers
    except Exception as e:
        print(f"  Warning: IEX Cloud fetch failed: {e}")
    return []

def validate_with_yfinance(tickers, sample_size=50):
    """
    Quick sanity check — validate a random sample before saving.
    Full validation happens in fetch_data.py via NaN filter.
    """
    import yfinance as yf
    import random
    sample = random.sample(tickers, min(sample_size, len(tickers)))
    data = yf.download(sample, period="5d", progress=False, auto_adjust=True)["Close"]
    valid = [t for t in sample if t in data.columns and data[t].notna().any()]
    hit_rate = len(valid) / len(sample) * 100
    print(f"  Validation sample: {len(valid)}/{len(sample)} valid ({hit_rate:.0f}% hit rate)")
    return hit_rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",      type=str,  default="tickers.txt", help="Output file")
    ap.add_argument("--validate", action="store_true",              help="Run yfinance sanity check")
    ap.add_argument("--exchange", type=str,  default="all",
                    help="Filter by exchange: all | nasdaq | nyse | amex")
    args = ap.parse_args()

    print("[get_tickers] Fetching from all sources...")
    all_tickers = []

    # Source 1: NASDAQ trader (most reliable)
    all_tickers.extend(from_nasdaq_trader())
    time.sleep(1)

    # Source 2: GitHub maintained list
    all_tickers.extend(from_github_symbols())
    time.sleep(1)

    # Source 3: SEC EDGAR
    all_tickers.extend(from_sec_edgar())
    time.sleep(1)

    # Deduplicate
    seen = set()
    unique = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"\n[get_tickers] Total unique tickers: {len(unique)}")

    # Optional exchange filter using NASDAQ trader data
    if args.exchange != "all":
        print(f"  (Exchange filtering not yet applied — all tickers kept)")

    # Optional validation
    if args.validate:
        print("[get_tickers] Running yfinance validation sample...")
        try:
            import yfinance as yf
            validate_with_yfinance(unique)
        except ImportError:
            print("  yfinance not installed — skipping validation")

    # Save
    with open(args.out, "w") as f:
        f.write("\n".join(unique))
    print(f"[get_tickers] Saved {len(unique)} tickers → {args.out}")
    print(f"\nNext step:")
    print(f"  python fetch_data.py --tickers {args.out} --n 1000 --years 5 --out returns.csv")

if __name__ == "__main__":
    main()