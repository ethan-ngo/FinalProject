"""
fetch_data.py
Downloads historical daily returns for S&P 500 constituents (or a subset)
and saves them as a CSV: rows = trading days, columns = ticker symbols.
 
Usage:
    pip install yfinance pandas
    python fetch_data.py --n 200 --years 5 --out returns.csv
"""
 
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
 
# A representative 500-ticker list would be very long; we ship a
# hard-coded 500-symbol list.  For the demo we default to --n 200 of them.
SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
    "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH",
    "ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG",
    "AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC",
    "BK","BBWI","BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX",
    "BA","BKNG","BWA","BSX","BMY","AVGO","BR","BF-B","BLDR","BG","CDNS","CZR",
    "CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW",
    "CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB",
    "CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH",
    "CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA",
    "CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DE",
    "DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV",
    "DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA",
    "ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS",
    "EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS",
    "FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FLT","FMC","F",
    "FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN",
    "GNRC","GD","GIS","GPC","GWW","HAL","HIG","HAS","HCA","DOC","HSIC","HSY",
    "HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM",
    "HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE",
    "IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY",
    "J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM",
    "KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LNC",
    "LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX",
    "MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META",
    "MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ",
    "MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM",
    "NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE",
    "NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS",
    "PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG",
    "PM","PSX","PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR","PRU",
    "PLD","PRU","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX",
    "O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL",
    "SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA",
    "SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS",
    "SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER",
    "TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL",
    "TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO",
    "VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB",
    "GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST",
    "WDC","WHR","WRK","WY","WHR","WLTW","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int,   default=500, help="Number of assets")
    ap.add_argument("--years",type=int,   default=5,   help="Years of history")
    ap.add_argument("--out",  type=str,   default="returns.csv")
    args = ap.parse_args()
 
    tickers = list(dict.fromkeys(SP500_TICKERS))[:args.n]   # dedupe, trim
    period  = f"{args.years}y"
 
    print(f"[fetch_data] Downloading {len(tickers)} tickers, period={period} …")
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=True)["Close"]
 
    # Drop columns with too many NaNs (newly listed, etc.)
    raw = raw.dropna(axis=1, thresh=int(0.9 * len(raw)))
    print(f"[fetch_data] {raw.shape[1]} tickers survived NaN filter.")
 
    # Daily log-returns
    returns = np.log(raw / raw.shift(1)).dropna()
    returns.to_csv(args.out)
    print(f"[fetch_data] Saved {returns.shape} return matrix → {args.out}")
    print(f"  rows (days) = {returns.shape[0]},  cols (assets) = {returns.shape[1]}")
 
if __name__ == "__main__":
    main()