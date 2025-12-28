"""
Portabull - Indian Stock Database for Search
Contains popular NSE/BSE stocks for fuzzy search
"""

from typing import List, Dict, Any
import re

# Popular Indian stocks database
# Format: (symbol, company_name, sector, aliases)
INDIAN_STOCKS = [
    # NIFTY 50 Companies
    ("RELIANCE", "Reliance Industries Limited", "Energy", ["RIL", "JIO", "MUKESH"]),
    ("TCS", "Tata Consultancy Services", "IT", ["TATA", "CONSULTANCY"]),
    ("HDFCBANK", "HDFC Bank Limited", "Banking", ["HDFC"]),
    ("INFY", "Infosys Limited", "IT", ["INFOSYS"]),
    ("ICICIBANK", "ICICI Bank Limited", "Banking", ["ICICI"]),
    ("HINDUNILVR", "Hindustan Unilever Limited", "FMCG", ["HUL", "UNILEVER"]),
    ("SBIN", "State Bank of India", "Banking", ["SBI", "STATE BANK"]),
    ("BHARTIARTL", "Bharti Airtel Limited", "Telecom", ["AIRTEL", "BHARTI"]),
    ("BAJFINANCE", "Bajaj Finance Limited", "Finance", ["BAJAJ", "BAJFIN"]),
    ("KOTAKBANK", "Kotak Mahindra Bank", "Banking", ["KOTAK"]),
    ("LT", "Larsen & Toubro Limited", "Infrastructure", ["LARSEN", "TOUBRO"]),
    ("ITC", "ITC Limited", "FMCG", ["CIGARETTE", "TOBACCO"]),
    ("AXISBANK", "Axis Bank Limited", "Banking", ["AXIS"]),
    ("ASIANPAINT", "Asian Paints Limited", "Consumer", ["ASIAN", "PAINTS"]),
    ("MARUTI", "Maruti Suzuki India", "Auto", ["SUZUKI", "CAR"]),
    ("HCLTECH", "HCL Technologies", "IT", ["HCL"]),
    ("SUNPHARMA", "Sun Pharmaceutical", "Pharma", ["SUN", "PHARMA"]),
    ("WIPRO", "Wipro Limited", "IT", []),
    ("TITAN", "Titan Company Limited", "Consumer", ["TANISHQ", "WATCH"]),
    ("ULTRACEMCO", "UltraTech Cement", "Cement", ["ULTRATECH"]),
    ("TATAMOTORS", "Tata Motors Limited", "Auto", ["TATA", "MOTORS", "JLR"]),
    ("ONGC", "Oil & Natural Gas Corp", "Energy", ["OIL", "GAS"]),
    ("NTPC", "NTPC Limited", "Power", ["POWER"]),
    ("POWERGRID", "Power Grid Corp", "Power", ["GRID"]),
    ("M&M", "Mahindra & Mahindra", "Auto", ["MAHINDRA"]),
    ("JSWSTEEL", "JSW Steel Limited", "Steel", ["JSW"]),
    ("TATASTEEL", "Tata Steel Limited", "Steel", ["TATA"]),
    ("BAJAJFINSV", "Bajaj Finserv Limited", "Finance", ["BAJAJ", "FINSERV"]),
    ("ADANIENT", "Adani Enterprises", "Diversified", ["ADANI"]),
    ("ADANIPORTS", "Adani Ports & SEZ", "Infrastructure", ["ADANI", "PORTS"]),
    ("TECHM", "Tech Mahindra Limited", "IT", ["TECH", "MAHINDRA"]),
    ("NESTLEIND", "Nestle India Limited", "FMCG", ["NESTLE", "MAGGI"]),
    ("INDUSINDBK", "IndusInd Bank Limited", "Banking", ["INDUSIND"]),
    ("DIVISLAB", "Divi's Laboratories", "Pharma", ["DIVIS"]),
    ("GRASIM", "Grasim Industries", "Diversified", ["ADITYA BIRLA"]),
    ("DRREDDY", "Dr. Reddy's Labs", "Pharma", ["REDDY"]),
    ("BRITANNIA", "Britannia Industries", "FMCG", ["BISCUIT"]),
    ("COALINDIA", "Coal India Limited", "Mining", ["COAL"]),
    ("BAJAJ-AUTO", "Bajaj Auto Limited", "Auto", ["BAJAJ"]),
    ("CIPLA", "Cipla Limited", "Pharma", []),
    ("EICHERMOT", "Eicher Motors", "Auto", ["ROYAL ENFIELD", "BULLET"]),
    ("SBILIFE", "SBI Life Insurance", "Insurance", ["SBI", "LIFE"]),
    ("APOLLOHOSP", "Apollo Hospitals", "Healthcare", ["APOLLO", "HOSPITAL"]),
    ("BPCL", "Bharat Petroleum", "Energy", ["BHARAT", "PETROL"]),
    ("HEROMOTOCO", "Hero MotoCorp", "Auto", ["HERO", "HONDA"]),
    ("UPL", "UPL Limited", "Chemicals", []),
    ("HINDALCO", "Hindalco Industries", "Metals", ["NOVELIS"]),
    ("TATACONSUM", "Tata Consumer Products", "FMCG", ["TATA", "TEA"]),
    
    # Other Popular Stocks
    ("ZOMATO", "Zomato Limited", "Food Delivery", ["FOOD"]),
    ("PAYTM", "One97 Communications", "Fintech", ["ONE97"]),
    ("NYKAA", "FSN E-Commerce Ventures", "E-Commerce", ["FSN"]),
    ("POLICYBZR", "PB Fintech Limited", "Insurance", ["POLICY BAZAAR"]),
    ("DELHIVERY", "Delhivery Limited", "Logistics", []),
    ("IRCTC", "Indian Railway Catering", "Travel", ["RAILWAY"]),
    ("HAL", "Hindustan Aeronautics", "Defence", ["AERONAUTICS"]),
    ("BEL", "Bharat Electronics", "Defence", ["ELECTRONICS"]),
    ("VEDL", "Vedanta Limited", "Mining", ["VEDANTA"]),
    ("ADANIGREEN", "Adani Green Energy", "Power", ["ADANI", "GREEN"]),
    ("ADANIPOWER", "Adani Power Limited", "Power", ["ADANI"]),
    ("BANKBARODA", "Bank of Baroda", "Banking", ["BOB", "BARODA"]),
    ("PNB", "Punjab National Bank", "Banking", ["PUNJAB"]),
    ("CANBK", "Canara Bank", "Banking", ["CANARA"]),
    ("UNIONBANK", "Union Bank of India", "Banking", ["UNION"]),
    ("IDFCFIRSTB", "IDFC First Bank", "Banking", ["IDFC"]),
    ("YESBANK", "Yes Bank Limited", "Banking", ["YES"]),
    ("FEDERALBNK", "Federal Bank", "Banking", ["FEDERAL"]),
    ("RBLBANK", "RBL Bank Limited", "Banking", ["RBL"]),
    ("BANDHANBNK", "Bandhan Bank", "Banking", ["BANDHAN"]),
    ("AUBANK", "AU Small Finance Bank", "Banking", ["AU"]),
    ("HDFCLIFE", "HDFC Life Insurance", "Insurance", ["HDFC", "LIFE"]),
    ("ICICIGI", "ICICI Lombard GIC", "Insurance", ["ICICI", "GENERAL"]),
    ("ICICIPRULI", "ICICI Prudential Life", "Insurance", ["ICICI", "PRUDENTIAL"]),
    ("SBICARD", "SBI Cards & Payment", "Finance", ["SBI", "CARD"]),
    ("PEL", "Piramal Enterprises", "Diversified", ["PIRAMAL"]),
    ("MUTHOOTFIN", "Muthoot Finance", "Finance", ["MUTHOOT", "GOLD"]),
    ("CHOLAFIN", "Cholamandalam Investment", "Finance", ["CHOLA"]),
    ("SHRIRAMFIN", "Shriram Finance", "Finance", ["SHRIRAM"]),
    ("MANAPPURAM", "Manappuram Finance", "Finance", ["GOLD"]),
    ("LICI", "Life Insurance Corp", "Insurance", ["LIC"]),
    ("DABUR", "Dabur India Limited", "FMCG", []),
    ("MARICO", "Marico Limited", "FMCG", ["PARACHUTE"]),
    ("GODREJCP", "Godrej Consumer Products", "FMCG", ["GODREJ"]),
    ("COLPAL", "Colgate-Palmolive India", "FMCG", ["COLGATE"]),
    ("PIDILITIND", "Pidilite Industries", "Consumer", ["FEVICOL"]),
    ("BERGEPAINT", "Berger Paints India", "Consumer", ["BERGER"]),
    ("HAVELLS", "Havells India Limited", "Consumer", []),
    ("VOLTAS", "Voltas Limited", "Consumer", ["AC", "AIR CONDITIONER"]),
    ("WHIRLPOOL", "Whirlpool of India", "Consumer", []),
    ("TRENT", "Trent Limited", "Retail", ["WESTSIDE", "ZUDIO"]),
    ("DMART", "Avenue Supermarts", "Retail", ["AVENUE"]),
    ("JUBLFOOD", "Jubilant FoodWorks", "Food", ["DOMINOS", "PIZZA"]),
    ("TATAELXSI", "Tata Elxsi Limited", "IT", ["ELXSI"]),
    ("LTIM", "LTIMindtree Limited", "IT", ["MINDTREE"]),
    ("MPHASIS", "Mphasis Limited", "IT", []),
    ("COFORGE", "Coforge Limited", "IT", []),
    ("PERSISTENT", "Persistent Systems", "IT", []),
    ("HAPPSTMNDS", "Happiest Minds Tech", "IT", ["HAPPIEST"]),
    ("BIOCON", "Biocon Limited", "Pharma", []),
    ("LUPIN", "Lupin Limited", "Pharma", []),
    ("AUROPHARMA", "Aurobindo Pharma", "Pharma", ["AUROBINDO"]),
    ("TORNTPHARM", "Torrent Pharma", "Pharma", ["TORRENT"]),
    ("ALKEM", "Alkem Laboratories", "Pharma", []),
    ("GLENMARK", "Glenmark Pharma", "Pharma", []),
    ("MAXHEALTH", "Max Healthcare", "Healthcare", ["MAX", "HOSPITAL"]),
    ("FORTIS", "Fortis Healthcare", "Healthcare", []),
    ("METROPOLIS", "Metropolis Healthcare", "Healthcare", ["DIAGNOSTICS"]),
    ("LALPATHLAB", "Dr Lal PathLabs", "Healthcare", ["LAL", "PATHLAB"]),
    ("INDIGO", "InterGlobe Aviation", "Aviation", ["INTERGLOBE"]),
    ("SPICEJET", "SpiceJet Limited", "Aviation", ["SPICE"]),
    ("PIIND", "PI Industries", "Chemicals", ["PI"]),
    ("ATUL", "Atul Limited", "Chemicals", []),
    ("DEEPAKNTR", "Deepak Nitrite", "Chemicals", ["DEEPAK"]),
    ("SRF", "SRF Limited", "Chemicals", []),
    ("TATAPOWER", "Tata Power Company", "Power", ["TATA"]),
    ("NHPC", "NHPC Limited", "Power", ["HYDRO"]),
    ("SJVN", "SJVN Limited", "Power", []),
    ("RECLTD", "REC Limited", "Finance", ["REC", "RURAL"]),
    ("PFC", "Power Finance Corp", "Finance", ["POWER"]),
    ("IREDA", "Indian Renewable Energy", "Finance", ["RENEWABLE"]),
    ("IOC", "Indian Oil Corporation", "Energy", ["INDIAN OIL"]),
    ("GAIL", "GAIL India Limited", "Energy", ["GAS"]),
    ("PETRONET", "Petronet LNG Limited", "Energy", ["LNG"]),
    ("MRF", "MRF Limited", "Auto", ["TYRE"]),
    ("APOLLOTYRE", "Apollo Tyres", "Auto", ["TYRE"]),
    ("CEAT", "CEAT Limited", "Auto", ["TYRE"]),
    ("BALKRISIND", "Balkrishna Industries", "Auto", ["BKT", "TYRE"]),
    ("MOTHERSON", "Samvardhana Motherson", "Auto", []),
    ("BHARATFORG", "Bharat Forge", "Auto", ["FORGE"]),
    ("BOSCHLTD", "Bosch Limited", "Auto", ["BOSCH"]),
    ("ASHOKLEY", "Ashok Leyland", "Auto", ["LEYLAND", "TRUCK"]),
    ("TVSMOTOR", "TVS Motor Company", "Auto", ["TVS"]),
    ("ACC", "ACC Limited", "Cement", []),
    ("AMBUJACEM", "Ambuja Cements", "Cement", ["AMBUJA"]),
    ("SHREECEM", "Shree Cement", "Cement", ["SHREE"]),
    ("RAMCOCEM", "Ramco Cements", "Cement", ["RAMCO"]),
    ("JKCEMENT", "JK Cement Limited", "Cement", ["JK"]),
    ("SAIL", "Steel Authority of India", "Steel", ["STEEL"]),
    ("NMDC", "NMDC Limited", "Mining", []),
    ("NATIONALUM", "National Aluminium", "Metals", ["NALCO"]),
    ("MOIL", "MOIL Limited", "Mining", ["MANGANESE"]),
    ("JIOFIN", "Jio Financial Services", "Finance", ["JIO"]),
    ("LODHA", "Macrotech Developers", "Real Estate", ["MACROTECH"]),
    ("DLF", "DLF Limited", "Real Estate", []),
    ("GODREJPROP", "Godrej Properties", "Real Estate", ["GODREJ"]),
    ("OBEROIRLTY", "Oberoi Realty", "Real Estate", ["OBEROI"]),
    ("PRESTIGE", "Prestige Estates", "Real Estate", []),
    ("PHOENIXLTD", "Phoenix Mills", "Real Estate", ["PHOENIX", "MALL"]),
    ("BRIGADE", "Brigade Enterprises", "Real Estate", []),
    
    # US Stocks (for US exchange)
    ("AAPL", "Apple Inc", "Technology", ["APPLE", "IPHONE"]),
    ("MSFT", "Microsoft Corporation", "Technology", ["MICROSOFT", "WINDOWS"]),
    ("GOOGL", "Alphabet Inc", "Technology", ["GOOGLE"]),
    ("AMZN", "Amazon.com Inc", "E-Commerce", ["AMAZON"]),
    ("META", "Meta Platforms Inc", "Technology", ["FACEBOOK", "FB"]),
    ("TSLA", "Tesla Inc", "Auto", ["TESLA", "MUSK"]),
    ("NVDA", "NVIDIA Corporation", "Technology", ["NVIDIA", "GPU"]),
    ("AMD", "Advanced Micro Devices", "Technology", []),
    ("NFLX", "Netflix Inc", "Entertainment", ["NETFLIX"]),
    ("DIS", "Walt Disney Company", "Entertainment", ["DISNEY"]),
    ("PYPL", "PayPal Holdings", "Fintech", ["PAYPAL"]),
    ("V", "Visa Inc", "Finance", ["VISA"]),
    ("MA", "Mastercard Inc", "Finance", ["MASTERCARD"]),
    ("JPM", "JPMorgan Chase", "Banking", ["JPMORGAN"]),
    ("BAC", "Bank of America", "Banking", []),
    ("WMT", "Walmart Inc", "Retail", ["WALMART"]),
    ("KO", "Coca-Cola Company", "FMCG", ["COCA", "COKE"]),
    ("PEP", "PepsiCo Inc", "FMCG", ["PEPSI"]),
    ("JNJ", "Johnson & Johnson", "Healthcare", []),
    ("PFE", "Pfizer Inc", "Pharma", ["PFIZER"]),
]


def search_stocks(query: str, exchange: str = "NSE", limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search stocks with fuzzy matching
    
    Matches against:
    - Symbol (exact and partial)
    - Company name (partial)
    - Sector (partial)
    - Aliases
    """
    if not query or len(query) < 1:
        return []
    
    query = query.upper().strip()
    query_lower = query.lower()
    results = []
    
    # Filter by exchange
    if exchange.upper() == "US":
        # US stocks only
        stock_list = [(s, n, sec, a) for s, n, sec, a in INDIAN_STOCKS 
                      if s in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", 
                              "NVDA", "AMD", "NFLX", "DIS", "PYPL", "V", "MA", 
                              "JPM", "BAC", "WMT", "KO", "PEP", "JNJ", "PFE"]]
    else:
        # Indian stocks
        stock_list = [(s, n, sec, a) for s, n, sec, a in INDIAN_STOCKS 
                      if s not in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", 
                                  "NVDA", "AMD", "NFLX", "DIS", "PYPL", "V", "MA", 
                                  "JPM", "BAC", "WMT", "KO", "PEP", "JNJ", "PFE"]]
    
    for symbol, name, sector, aliases in stock_list:
        score = 0
        name_lower = name.lower()
        sector_lower = sector.lower()
        
        # Exact symbol match (highest score)
        if symbol == query:
            score = 100
        # Symbol starts with query
        elif symbol.startswith(query):
            score = 80 + (len(query) / len(symbol)) * 10
        # Symbol contains query
        elif query in symbol:
            score = 60 + (len(query) / len(symbol)) * 10
        # Company name starts with query
        elif name_lower.startswith(query_lower):
            score = 70
        # Company name contains query word
        elif query_lower in name_lower:
            score = 50 + (len(query) / len(name)) * 10
        # Sector matches
        elif query_lower in sector_lower:
            score = 40
        # Check aliases
        else:
            for alias in aliases:
                if alias == query:
                    score = 75
                    break
                elif query in alias:
                    score = 55
                    break
        
        if score > 0:
            results.append({
                'symbol': symbol,
                'name': name,
                'sector': sector,
                'exchange': exchange,
                'score': score
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Remove score from results and limit
    return [
        {
            'symbol': r['symbol'],
            'name': r['name'],
            'sector': r['sector'],
            'exchange': r['exchange']
        }
        for r in results[:limit]
    ]


def get_popular_stocks(exchange: str = "NSE", limit: int = 10) -> List[Dict[str, Any]]:
    """Get popular stocks for a given exchange"""
    popular_nse = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "BAJFINANCE", "KOTAKBANK"
    ]
    
    popular_us = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "NFLX", "V", "JPM"
    ]
    
    symbols = popular_us if exchange.upper() == "US" else popular_nse
    
    results = []
    for symbol in symbols[:limit]:
        for s, n, sec, _ in INDIAN_STOCKS:
            if s == symbol:
                results.append({
                    'symbol': s,
                    'name': n,
                    'sector': sec,
                    'exchange': exchange
                })
                break
    
    return results
