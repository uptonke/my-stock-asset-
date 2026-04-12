import os
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from supabase import create_client, Client
import re
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 安全地讀取 GitHub Secrets (環境變數)
# ==========================================
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    raise ValueError("❌ 找不到環境變數！請確認 GitHub Secrets 是否設定正確。")

supabase: Client = create_client(url, key)

# ==========================================
# 2. 以下邏輯與 Colab 完全相同
# ==========================================
response = supabase.table("portfolio_db").select("stock_meta").eq("id", 1).execute()
data = response.data

if not data:
    print("❌ 資料庫沒東西或連線失敗！")
else:
    stock_meta = data[0].get("stock_meta", {})
    tickers = list(stock_meta.keys())
    
    if tickers:
        benchmark = "SPY"
        yf_tickers = [f"{t}.TW" if re.match(r'^\d+[A-Za-z]?$', t) else t for t in tickers]
        download_list = yf_tickers + [benchmark]

        print(f"⏳ 正在向 Yahoo Finance 請求歷史資料...")
        prices = yf.download(download_list, period="1y")["Close"]
        
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=download_list[0])

        returns = prices.pct_change()

        print("🧠 開始計算 多因子 風險與動能參數...")
        for i, original_ticker in enumerate(tickers):
            yf_ticker = yf_tickers[i]

            if yf_ticker not in returns.columns:
                continue

            stock_close = prices[yf_ticker].dropna()
            stock_ret = returns[yf_ticker].dropna()
            
            if len(stock_ret) < 60: 
                continue

            bench_ret = returns[benchmark]
            aligned_data = pd.concat([stock_ret, bench_ret], axis=1).dropna()
            aligned_stock = aligned_data.iloc[:, 0]
            aligned_bench = aligned_data.iloc[:, 1]

            ewma_var = aligned_stock.ewm(alpha=0.06).var().iloc[-1]
            ann_std = np.sqrt(ewma_var * 252) * 100
            
            cov = aligned_stock.cov(aligned_bench)
            bench_var = aligned_bench.var()
            beta = cov / bench_var if bench_var > 0 else 1.0

            rsi_series = ta.rsi(stock_close, length=14)
            rsi_14 = rsi_series.iloc[-1] if rsi_series is not None and not rsi_series.empty else 50.0

            macd_df = ta.macd(stock_close, fast=12, slow=26, signal=9)
            macd_hist = macd_df.iloc[-1, 1] if macd_df is not None and not macd_df.empty else 0.0

            mom_6m = stock_close.pct_change(periods=126).iloc[-1] * 100

            stock_meta[original_ticker]["std"] = round(ann_std, 2)
            stock_meta[original_ticker]["beta"] = round(beta, 2)
            stock_meta[original_ticker]["rsi"] = round(rsi_14, 2) if pd.notna(rsi_14) else 50.0
            stock_meta[original_ticker]["macd_h"] = round(macd_hist, 4) if pd.notna(macd_hist) else 0.0
            stock_meta[original_ticker]["mom_6m"] = round(mom_6m, 2) if pd.notna(mom_6m) else 0.0

            print(f"✅ [{original_ticker}] 更新 -> Beta:{beta:.2f}, Std:{ann_std:.1f}%, RSI:{rsi_14:.1f}, MACD_H:{macd_hist:.2f}, 6M_Mom:{mom_6m:.1f}%")

        print("🚀 正在將多因子參數打回 Supabase...")
        supabase.table("portfolio_db").update({"stock_meta": stock_meta}).eq("id", 1).execute()
        print("🎉 雲端自動化任務完成！")
