"""
Index constituent ticker lists for Yahoo Finance.
Updated: 2026-05-15

Sources:
- S&P 100: Wikipedia (as of Sep 2025)
- FTSE 100: Wikipedia (as of Mar 2026)
- Nikkei 225: Wikipedia
- Hang Seng Index: Wikipedia (as of Jan 2026)
- CSI 300: Wikipedia (as of Mar 2024)

Suffixes follow Yahoo Finance conventions:
- US tickers: no suffix
- UK tickers: .L (London Stock Exchange)
- Japanese tickers: .T (Tokyo Stock Exchange)
- Hong Kong tickers: .HK (4-digit zero-padded)
- China A-shares: .SS (Shanghai) or .SZ (Shenzhen)
"""

# S&P 100 (OEX) - 100 tickers
SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GEV", "GILD",
    "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG",
    "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD",
    "MDLZ", "MDT", "META", "MMM", "MO", "MRK", "MS", "MSFT", "MU", "NEE",
    "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TMO", "TMUS", "TSLA",
    "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
]

# FTSE 100 - 100 tickers (with .L suffix for Yahoo Finance)
FTSE100 = [
    "III.L", "ADM.L", "AAF.L", "ALW.L", "AAL.L", "ANTO.L", "ABF.L", "AZN.L",
    "AUTO.L", "AV.L", "BAB.L", "BA.L", "BARC.L", "BTRW.L", "BEZ.L", "BKG.L",
    "BP.L", "BATS.L", "BLND.L", "BT-A.L", "BNZL.L", "BRBY.L", "CNA.L", "CCEP.L",
    "CCH.L", "CPG.L", "CTEC.L", "CRDA.L", "DCC.L", "DGE.L", "DPLM.L", "EDV.L",
    "ENT.L", "EXPN.L", "FCIT.L", "FRES.L", "GAW.L", "GLEN.L", "GSK.L", "HLN.L",
    "HLMA.L", "HSX.L", "HWDN.L", "HSBA.L", "ICG.L", "IGG.L", "IHG.L", "IMI.L",
    "IMB.L", "INF.L", "IAG.L", "ITRK.L", "JD.L", "BGEO.L", "KGF.L", "LAND.L",
    "LGEN.L", "LLOY.L", "LMP.L", "LSEG.L", "MNG.L", "MKS.L", "MRO.L", "MTLN.L",
    "MNDI.L", "NG.L", "NWG.L", "NXT.L", "PSON.L", "PSH.L", "PSN.L", "PCT.L",
    "PRU.L", "RKT.L", "REL.L", "RTO.L", "RMV.L", "RIO.L", "RR.L", "SGE.L",
    "SBRY.L", "SDR.L", "SMT.L", "SGRO.L", "SVT.L", "SHEL.L", "SMIN.L", "SN.L",
    "SPX.L", "SSE.L", "STAN.L", "SDLF.L", "STJ.L", "TSCO.L", "BBOX.L", "ULVR.L",
    "UU.L", "VOD.L", "WEIR.L", "WTB.L",
]

# Nikkei 225 - 225 tickers (with .T suffix for Yahoo Finance)
NIKKEI225 = [
    # Air Transport
    "9202.T", "9201.T",
    # Automotive
    "7267.T", "7202.T", "7261.T", "7211.T", "7201.T", "7270.T",
    "7269.T", "7203.T", "7272.T",
    # Banking
    "8304.T", "8331.T", "8354.T", "8306.T", "8411.T", "8308.T", "5831.T",
    "8316.T", "8309.T", "7186.T",
    # Chemicals
    "3407.T", "4061.T", "4901.T", "4452.T", "3405.T", "4188.T", "4183.T",
    "4021.T", "6988.T", "4004.T", "4063.T", "4911.T", "4005.T", "4043.T",
    "4042.T", "4208.T",
    # Communications
    "9433.T", "9432.T", "9434.T", "9984.T",
    # Construction
    "1721.T", "1925.T", "1808.T", "1963.T", "1812.T", "1802.T", "1928.T",
    "1803.T", "1801.T",
    # Electric Machinery
    "6857.T", "6770.T", "7751.T", "6902.T", "6954.T", "6504.T", "6702.T",
    "6501.T", "6861.T", "285A.T", "6971.T", "6920.T", "6479.T", "6503.T",
    "6981.T", "6701.T", "6594.T", "6645.T", "6752.T", "6723.T", "7752.T",
    "6963.T", "7735.T", "6724.T", "6753.T", "6758.T", "6526.T", "6976.T",
    "6762.T", "8035.T", "6506.T", "6841.T",
    # Electric Power
    "9502.T", "9503.T", "9501.T",
    # Fishery
    "1332.T",
    # Foods
    "2802.T", "2502.T", "2914.T", "2801.T", "2503.T", "2269.T", "2282.T",
    "2871.T", "2002.T", "2501.T",
    # Gas
    "9532.T", "9531.T",
    # Glass & Ceramics
    "5201.T", "5333.T", "5214.T", "5233.T", "5301.T", "5332.T",
    # Insurance
    "8750.T", "8725.T", "8630.T", "8795.T", "8766.T",
    # Land Transport
    "9147.T", "9064.T",
    # Machinery
    "6113.T", "6367.T", "6361.T", "6305.T", "7004.T", "7013.T", "5631.T",
    "6473.T", "6301.T", "6326.T", "7011.T", "6471.T", "6472.T", "6103.T",
    "6302.T", "6273.T",
    # Marine Transport
    "9107.T", "9104.T", "9101.T",
    # Mining
    "1605.T",
    # Nonferrous Metals
    "5714.T", "5803.T", "5801.T", "5711.T", "5706.T", "3436.T", "5802.T",
    "5713.T",
    # Other Financial Services
    "8253.T", "8697.T", "8591.T",
    # Other Manufacturing
    "7832.T", "7912.T", "7911.T", "7951.T",
    # Petroleum
    "5020.T", "5019.T",
    # Pharmaceuticals
    "4503.T", "4519.T", "4568.T", "4523.T", "4151.T", "4578.T", "4506.T",
    "4507.T", "4502.T",
    # Precision Instruments
    "6146.T", "7741.T", "4902.T", "7731.T", "7733.T", "4543.T",
    # Pulp & Paper
    "3861.T",
    # Railway/Bus
    "9022.T", "9020.T", "9008.T", "9009.T", "9007.T", "9001.T", "9005.T",
    "9021.T",
    # Real Estate
    "8802.T", "8801.T", "8830.T", "8804.T", "3289.T",
    # Retail
    "8267.T", "9983.T", "3099.T", "3086.T", "8252.T", "7453.T", "9843.T",
    "7532.T", "3382.T", "8233.T", "3092.T",
    # Rubber
    "5108.T", "5101.T",
    # Securities
    "8601.T", "8604.T",
    # Services
    "6532.T", "4751.T", "2432.T", "4324.T", "6178.T", "9766.T", "4689.T",
    "4385.T", "2413.T", "3659.T", "7974.T", "4307.T", "4661.T", "4755.T",
    "6098.T", "9735.T", "3697.T", "9602.T", "4704.T",
    # Shipbuilding
    "7012.T",
    # Steel
    "5411.T", "5406.T", "5401.T",
    # Textiles & Apparel
    "3401.T", "3402.T",
    # Trading Companies
    "8001.T", "8002.T", "8058.T", "8031.T", "2768.T", "8053.T", "8015.T",
    # Warehousing & Transport
    "9301.T",
]

# Hang Seng Index (HSI) - 85 tickers (with .HK suffix, 4-digit zero-padded)
HSI = [
    # Finance
    "0005.HK", "0388.HK", "0939.HK", "1299.HK", "1398.HK", "2318.HK",
    "2388.HK", "2628.HK", "3968.HK", "3988.HK",
    # Utilities
    "0002.HK", "0003.HK", "0006.HK", "0836.HK", "1038.HK", "2688.HK",
    # Properties
    "0012.HK", "0016.HK", "0101.HK", "0688.HK", "0823.HK", "0960.HK",
    "1109.HK", "1113.HK", "1209.HK", "1997.HK",
    # Commerce & Industry
    "0001.HK", "0027.HK", "0066.HK", "0175.HK", "0241.HK", "0267.HK",
    "0285.HK", "0288.HK", "0291.HK", "0300.HK", "0316.HK", "0322.HK",
    "0386.HK", "0669.HK", "0700.HK", "0762.HK", "0857.HK", "0868.HK",
    "0881.HK", "0883.HK", "0941.HK", "0968.HK", "0981.HK", "0992.HK",
    "1024.HK", "1044.HK", "1088.HK", "1093.HK", "1099.HK", "1177.HK",
    "1211.HK", "1378.HK", "1810.HK", "1876.HK", "1928.HK", "1929.HK",
    "2015.HK", "2020.HK", "2057.HK", "2269.HK", "2313.HK", "2319.HK",
    "2331.HK", "2359.HK", "2382.HK", "2618.HK", "2899.HK", "3690.HK",
    "3692.HK", "6618.HK", "6690.HK", "6862.HK", "9618.HK", "9633.HK",
    "9888.HK", "9961.HK", "9988.HK", "9992.HK", "9999.HK",
]

# CSI 300 - 272 tickers (.SS for Shanghai, .SZ for Shenzhen)
# Shanghai stocks start with 6 (including 688xxx for STAR Market)
# Shenzhen stocks start with 0 or 3 (300xxx for ChiNext)
CSI300 = [
    # Shanghai Stock Exchange (.SS)
    "600000.SS", "600009.SS", "600010.SS", "600011.SS", "600015.SS", "600016.SS",
    "600018.SS", "600019.SS", "600023.SS", "600025.SS", "600028.SS", "600029.SS",
    "600030.SS", "600031.SS", "600036.SS", "600039.SS", "600048.SS", "600050.SS",
    "600061.SS", "600085.SS", "600089.SS", "600104.SS", "600111.SS", "600115.SS",
    "600132.SS", "600150.SS", "600183.SS", "600188.SS", "600196.SS", "600219.SS",
    "600233.SS", "600276.SS", "600309.SS", "600346.SS", "600362.SS", "600406.SS",
    "600426.SS", "600436.SS", "600438.SS", "600460.SS", "600489.SS", "600515.SS",
    "600519.SS", "600547.SS", "600570.SS", "600584.SS", "600585.SS", "600588.SS",
    "600600.SS", "600606.SS", "600660.SS", "600674.SS", "600690.SS", "600732.SS",
    "600741.SS", "600745.SS", "600754.SS", "600760.SS", "600795.SS", "600803.SS",
    "600809.SS", "600845.SS", "600875.SS", "600886.SS", "600887.SS",
    "600893.SS", "600900.SS", "600905.SS", "600918.SS", "600919.SS", "600926.SS",
    "600938.SS", "600941.SS", "600958.SS", "600989.SS", "600999.SS", "601006.SS",
    "601009.SS", "601012.SS", "601021.SS", "601059.SS", "601066.SS", "601088.SS",
    "601100.SS", "601111.SS", "601117.SS", "601138.SS", "601155.SS", "601166.SS",
    "601169.SS", "601186.SS", "601211.SS", "601225.SS", "601229.SS", "601236.SS",
    "601238.SS", "601288.SS", "601318.SS", "601319.SS", "601336.SS", "601377.SS",
    "601390.SS", "601398.SS", "601600.SS", "601601.SS", "601607.SS", "601615.SS",
    "601628.SS", "601633.SS", "601658.SS", "601668.SS", "601669.SS", "601688.SS",
    "601689.SS", "601698.SS", "601699.SS", "601728.SS", "601766.SS", "601788.SS",
    "601799.SS", "601800.SS", "601808.SS", "601816.SS", "601818.SS", "601838.SS",
    "601857.SS", "601865.SS", "601868.SS", "601877.SS", "601878.SS", "601881.SS",
    "601888.SS", "601916.SS", "601919.SS", "601939.SS", "601985.SS", "601988.SS",
    "601995.SS", "601998.SS", "603019.SS", "603195.SS", "603259.SS",
    "603260.SS", "603288.SS", "603290.SS", "603369.SS", "603392.SS", "603486.SS",
    "603501.SS", "603659.SS", "603799.SS", "603806.SS", "603833.SS", "603899.SS",
    "603986.SS", "605117.SS", "605499.SS", "688008.SS", "688012.SS", "688036.SS",
    "688041.SS", "688065.SS", "688111.SS", "688126.SS", "688187.SS", "688223.SS",
    "688256.SS", "688271.SS", "688303.SS", "688363.SS", "688561.SS", "688599.SS",
    "688981.SS",
    # Shenzhen Stock Exchange (.SZ)
    "000001.SZ", "000002.SZ", "000063.SZ", "000069.SZ", "000100.SZ", "000157.SZ",
    "000301.SZ", "000333.SZ", "000408.SZ", "000538.SZ", "000568.SZ", "000596.SZ",
    "000617.SZ", "000625.SZ", "000651.SZ", "000708.SZ", "000725.SZ", "000733.SZ",
    "000768.SZ", "000786.SZ", "000792.SZ", "000800.SZ", "000858.SZ", "000877.SZ",
    "000895.SZ", "000938.SZ", "000963.SZ", "000977.SZ", "000983.SZ", "000999.SZ",
    "001289.SZ", "001979.SZ", "002001.SZ", "002007.SZ", "002027.SZ", "002049.SZ",
    "002074.SZ", "002129.SZ", "002142.SZ", "002179.SZ", "002180.SZ", "002202.SZ",
    "002230.SZ", "002241.SZ", "002252.SZ", "002304.SZ", "002311.SZ", "002352.SZ",
    "002371.SZ", "002410.SZ", "002415.SZ", "002460.SZ", "002466.SZ", "002475.SZ",
    "002493.SZ", "002555.SZ", "002594.SZ", "002603.SZ", "002648.SZ", "002709.SZ",
    "002714.SZ", "002736.SZ", "002812.SZ", "002821.SZ", "002841.SZ", "002920.SZ",
    "002938.SZ", "002916.SZ", "003816.SZ", "300015.SZ", "300033.SZ", "300059.SZ",
    "300124.SZ", "300142.SZ", "300223.SZ", "300274.SZ", "300308.SZ", "300316.SZ",
    "300347.SZ", "300408.SZ", "300413.SZ", "300433.SZ", "300450.SZ", "300454.SZ",
    "300496.SZ", "300498.SZ", "300628.SZ", "300661.SZ", "300750.SZ", "300751.SZ",
    "300759.SZ", "300760.SZ", "300763.SZ", "300782.SZ", "300896.SZ", "300919.SZ",
    "300957.SZ", "300979.SZ", "300999.SZ",
]
