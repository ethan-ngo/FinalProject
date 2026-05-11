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
TICKERS = [
    # S&P 500
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
    "PLD","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX",
    "O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL",
    "SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA",
    "SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS",
    "SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER",
    "TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL",
    "TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO",
    "VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB",
    "WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST",
    "WDC","WHR","WRK","WY","XEL","XYL","YUM","ZBRA","ZBH","ZTS",

    # S&P 400 MidCap
    "AAN","ACC","ACHC","ACM","AFG","AGCO","AGL","AIR","AIT","AKR",
    "ALK","ALKS","ALLY","ALSN","AM","AMG","AMKR","ANF","APAM","APG",
    "APPF","APPN","AR","ARCB","ARMK","ARW","ASB","ASH","ASO","ATI",
    "ATKR","AUB","AVT","AWI","AX","AYI","AZZ","BC","BCO","BFH",
    "BGS","BHF","BJ","BKH","BLKB","BMI","BOH","BOKF","BRC","BRKR",
    "BTU","BV","BYD","CABO","CADE","CALM","CAR","CATY","CC","CCCS",
    "CCS","CDK","CDMO","CDRE","CECO","CENX","CFR","CHCO","CHDN","CHE",
    "CHRD","CIR","CIVI","CKH","CLB","CLF","CLVT","CMC","CMCO","CMP",
    "CNA","CNK","CNO","CNXC","COHU","COLB","COLM","COOP","CORT","CPF",
    "CPK","CPRX","CR","CRC","CRGY","CRI","CRK","CRUS","CRY","CSL",
    "CSTM","CSWI","CUZ","CVI","CVLT","CW","CWT","CXW","DAN","DBD",
    "DDS","DEI","DFIN","DKS","DKNG","DLNG","DLX","DOCN","DV","DVN",
    "DY","EAT","EBC","EGP","EIG","ELF","ENVA","ENS","EPRT","ESI",
    "ESNT","ETRN","EVTC","EXP","EXPI","EXAS","EXEL","EXLS","FFIN",
    "FLT","FLO","FORM","FOUR","FRPT","FRSH","FSS","FTRE","GBCI","GEF",
    "GFF","GKOS","GLBE","GME","GMS","GNRC","GPI","GPOR","GRND","GSHD",
    "GTLB","HAE","HALO","HBI","HCC","HHH","HIW","HLNE","HLX","HMST",
    "HNI","HOMB","HUBG","IART","IBCP","IBTX","ICFI","IDCC","IDYA",
    "IESC","IGT","IMVT","INDB","INMD","INSP","INST","INTA","IOSP",
    "IPGP","IRTC","ITCI","ITGR","JACK","JAMF","JOBY","KALU","KFRC",
    "KLIC","KMPR","KNSL","KROS","KRYS","KSS","KTOS","LANC","LAUR",
    "LCII","LCNB","LEA","LESL","LGF-A","LGF-B","LGND","LKQ","LMAT",
    "LNTH","LOPE","LPLA","LSTR","LUMN","MARA","MAT","MATX","MCY",
    "MDGL","MEDP","MERI","MHO","MIDD","MKSI","MLI","MMSI","MMS","MNR",
    "MOD","MODV","MOFG","MSGE","MSGS","MTH","MTRN","MTUS","MUR","NARI",
    "NATL","NBR","NBTB","NDAQ","NEOG","NFBK","NHI","NJR","NNN","NOMD",
    "NPO","NRC","NRG","NSP","NUS","NVT","NYCB","NYCR","NYT","OGE",
    "OGS","OHI","OII","OIS","OLN","OMCL","ONB","OPCH","ORI","OSCR",
    "OTEX","PACB","PATK","PAYLOCITY","PBF","PBFX","PBPB","PCH","PDCO",
    "PDM","PENN","PFS","PGNY","PJT","PKE","PLAB","PLMR","PLXS","PMT",
    "POWL","PRGO","PRIM","PRK","PRKS","PSN","PTEN","PTGX","PVH","QLYS",
    "QNST","RBC","RCM","RDNT","RELY","REZI","RHP","RLI","RLJ","RNG",
    "ROAD","RPM","RRR","RSGX","RXO","SAFE","SAIA","SANM","SBCF","SBH",
    "SBSI","SCI","SEIC","SFBS","SFM","SGBX","SHAK","SHO","SIG","SITC",
    "SKX","SLG","SM","SMAR","SMG","SNX","SOUN","SPNT","SPR","SPSC",
    "SSD","SSTK","STC","STEP","STKL","STNG","STRS","SUM","SWI","SWX",
    "SYNA","TAHOE","TALO","TBBK","TCBI","TCBK","TGNA","THRM","TIGO",
    "TILE","TMHC","TNDM","TPH","TREX","TRMK","TRN","TRNO","TROW",
    "TRS","TRUP","TTGT","TWNK","TXG","UBER","UCBI","UNF","UNFI","UNM",
    "UNUM","UPBD","USTR","VCEL","VCNX","VICR","VIRT","VIST","VIVO",
    "VLTO","VNET","VRTS","VSAT","VSH","WAFD","WABC","WASH","WD","WDFC",
    "WEN","WERN","WEX","WINA","WING","WK","WKC","WLK","WMS","WOLF",
    "WOR","WPC","WRK","WSBC","WTFC","WTS","WULF","WWD","XPEL","XPOF",

    # S&P 600 SmallCap
    "AAON","ABG","ABM","ACAD","ACNB","ACSF","ADUS","AEIS","AESI","AGYS",
    "AHCO","AHH","AIRC","AJRD","ALEX","ALGT","ALNY","ALPN","ALTA","ALTG",
    "AMBC","AMED","AMPH","AMRX","AMSC","AMWD","ANGI","ANGO","AORT","APEI",
    "APOG","APTI","AQST","ARCH","ARCO","AROC","ARLO","ARWR","ASIX","ASND",
    "ASTH","ATEC","ATNI","ATRO","ATSG","ATXI","AVAL","AVAV","AVNS","AVNW",
    "AXNX","AXSM","AZTA","BANF","BANR","BBIO","BCAL","BCBP","BCPC","BCYC",
    "BDTX","BEAM","BGFV","BHB","BHLB","BJRI","BKSC","BKSY","BLDP","BLFS",
    "BLKB","BLX","BMBL","BMRC","BNED","BNGO","BPMC","BPOP","BRBR","BRBS",
    "BRKL","BSIG","BSVN","BTAI","BUSE","BV","BVS","BWFG","BWIN","CABO",
    "CAKE","CALX","CAMP","CARE","CARG","CASH","CBRL","CCNE","CDMO","CDXS",
    "CEIX","CENT","CENX","CERC","CEVA","CFFI","CFFN","CFFI","CGBD","CGEM",
    "CHCO","CHUY","CIFR","CLAR","CLDX","CLFD","CLNE","CLPT","CLSK","CLVS",
    "CMRE","CNCE","CNDT","CNMD","CNOB","CNXN","COHU","COLL","COMM","CONN",
    "COOP","CORR","CORT","CPRI","CPRX","CPSS","CRAI","CRCT","CRDF","CRDO",
    "CRVL","CSGP","CSGS","CSTM","CSWI","CTBI","CTLP","CVBF","CVCO","CVGW",
    "CVLT","CVLY","CWCO","CWT","CYBE","CYCN","CYRX","DAKT","DCOM","DCTH",
    "DFIN","DGII","DKNG","DLHC","DLNG","DLTH","DMRC","DNOW","DOCN","DOMO",
    "DORM","DSGN","DSGX","DSGR","DTIL","DXPE","DY","EAST","EBIX","ECAT",
    "ECPG","EDUC","EFSC","EGHT","EIG","ELME","EMBC","EMKR","ENOV","ENSG",
    "ENTA","ENTG","EPAC","EPIX","EPRT","ERIC","ERII","ESNT","ESOA","ESRT",
    "ESSA","ESTA","ESTE","ETD","ETWO","EVBG","EVER","EVGO","EVRI","EVTC",
    "EWBC","EXPI","EXTR","EZPW","FBNC","FBNK","FBRC","FBSS","FCFS","FCNCA",
    "FELE","FENC","FFBC","FFIC","FFIN","FFNW","FGBI","FISI","FITBI","FIXX",
    "FKWL","FLGT","FLIC","FLNC","FLNG","FLNX","FLXS","FMAO","FMBH","FMBI",
    "FMNB","FMST","FOLD","FONR","FORM","FORR","FOSL","FRBA","FRME","FRMEP",
    "FRST","FRWK","FSBC","FSBW","FSTR","FTLF","FULT","FUNC","FUSB","FWRG",
    "GBL","GCBC","GENC","GENI","GEOS","GEVI","GHLD","GIII","GLAD","GLDD",
    "GLNG","GLRE","GLSI","GMRE","GNE","GNLN","GNSS","GOOD","GRBK","GRFS",
    "GRNT","GROW","GRWG","GSBC","GSHD","GSIT","GTBP","GTLS","GURE","HAIN",
    "HALL","HALO","HARP","HAYN","HBCP","HBIO","HBMD","HBNC","HCAT","HCCI",
    "HCSG","HGBL","HGTY","HIBB","HIFS","HIMX","HLIT","HLNE","HLSS","HMN",
    "HMST","HOFT","HONE","HOPE","HOTH","HROW","HRTS","HSII","HSTM","HTBI",
    "HTBK","HTGM","HTLD","HTLF","HURN","HWBK","HWKN","HWM","HYLN","HYMC",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years",type=int,   default=5,   help="Years of history")
    ap.add_argument("--out",  type=str,   default="returns.csv")
    ap.add_argument("--tickers", type=str, default=None, help="Path to tickers.txt from get_tickers.py")
    
    args = ap.parse_args()

    # In main(), replace the TICKERS list with:
    if args.tickers:
        with open(args.tickers) as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = TICKERS  # fallback to hardcoded list

    tickers = list(dict.fromkeys(tickers))   # dedupe, trim
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