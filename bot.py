import asyncio
import logging
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import talib
from typing import Dict, List, Tuple, Optional

# تنظیمات
TELEGRAM_BOT_TOKEN = "8052349235:AAFSaJmYpl359BKrJTWC8O-u-dI9r2olEOQ"
ALPHA_VANTAGE_API = "uSuLsH9hvMCRLyvw4WzUZiGB"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# لیست ارزهای مورد نظر
CRYPTO_SYMBOLS = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'DOT', 'AVAX']

# وزن‌های اندیکاتورها برای استراتژی‌های مختلف
INDICATOR_WEIGHTS = {
    'conservative': {
        'RSI': 5, 'MACD': 5, 'BBANDS': 4, 'STOCH': 4, 'ADX': 3,
        'OBV': 4, 'CCI': 3, 'ATR': 3, 'WILLR': 3, 'MFI': 3,
        'EMA12_26': 4, 'EMA50_200': 5, 'SAR': 3, 'ROC': 3,
        'AD': 3, 'VWAP': 4, 'ICHIMOKU': 4, 'MOM': 3,
        'PPO': 3, 'SLOPE': 3, 'SUPERTREND': 5, 'FIBONACCI': 4,
        'VOLUME_PROFILE': 4, 'WHALE_ACTIVITY': 5, 'MARKET_SENTIMENT': 4
    },
    'moderate': {
        'RSI': 6, 'MACD': 6, 'BBANDS': 5, 'STOCH': 5, 'ADX': 4,
        'OBV': 5, 'CCI': 4, 'ATR': 4, 'WILLR': 4, 'MFI': 4,
        'EMA12_26': 5, 'EMA50_200': 6, 'SAR': 4, 'ROC': 4,
        'AD': 4, 'VWAP': 5, 'ICHIMOKU': 5, 'MOM': 4,
        'PPO': 4, 'SLOPE': 4, 'SUPERTREND': 6, 'FIBONACCI': 5,
        'VOLUME_PROFILE': 5, 'WHALE_ACTIVITY': 6, 'MARKET_SENTIMENT': 5
    },
    'aggressive': {
        'RSI': 7, 'MACD': 7, 'BBANDS': 6, 'STOCH': 6, 'ADX': 5,
        'OBV': 6, 'CCI': 5, 'ATR': 5, 'WILLR': 5, 'MFI': 5,
        'EMA12_26': 6, 'EMA50_200': 7, 'SAR': 5, 'ROC': 5,
        'AD': 5, 'VWAP': 6, 'ICHIMOKU': 6, 'MOM': 5,
        'PPO': 5, 'SLOPE': 5, 'SUPERTREND': 7, 'FIBONACCI': 6,
        'VOLUME_PROFILE': 6, 'WHALE_ACTIVITY': 7, 'MARKET_SENTIMENT': 6
    }
}

class AdvancedCryptoAnalyzer:
    def __init__(self):
        self.session = None
        self.data_cache = {}
        self.last_request_time = {}
        self.whale_activity_cache = {}
        self.market_sentiment_cache = {}
    
    async def safe_init_session(self):
        """ایجاد session با مدیریت خطا"""
        try:
            if not self.session or self.session.closed:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self.session = aiohttp.ClientSession(timeout=timeout)
            return True
        except Exception as e:
            print(f"خطا در ایجاد session: {e}")
            return False
    
    async def safe_close_session(self):
        """بستن session با مدیریت خطا"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
        except Exception as e:
            print(f"خطا در بستن session: {e}")
    
    async def safe_api_request(self, url, symbol, cache_key=None, cache_duration=300):
        """درخواست API با مدیریت کامل خطا و کش"""
        try:
            # بررسی کش
            if cache_key and cache_key in self.data_cache:
                cached_data, cache_time = self.data_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < cache_duration:
                    return cached_data
            
            # کنترل rate limiting
            current_time = datetime.now()
            if symbol in self.last_request_time:
                time_diff = (current_time - self.last_request_time[symbol]).total_seconds()
                if time_diff < 1.2:
                    await asyncio.sleep(1.2 - time_diff)
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.last_request_time[symbol] = datetime.now()
                    
                    # ذخیره در کش
                    if cache_key:
                        self.data_cache[cache_key] = (data, datetime.now())
                    
                    return data
                else:
                    print(f"خطای HTTP {response.status} برای {symbol}")
                    return None
        except asyncio.TimeoutError:
            print(f"Timeout برای {symbol}")
            return None
        except aiohttp.ClientError as e:
            print(f"خطای شبکه برای {symbol}: {e}")
            return None
        except Exception as e:
            print(f"خطای غیرمنتظره برای {symbol}: {e}")
            return None
    
    async def get_crypto_data(self, symbol, days=90):
        """دریافت داده‌های تاریخی ارز از CoinGecko"""
        try:
            cache_key = f"{symbol}_data_{days}"
            if cache_key in self.data_cache:
                cached_data, cache_time = self.data_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < 600:  # 10 دقیقه کش
                    return cached_data
            
            if not await self.safe_init_session():
                return None
            
            # پیدا کردن ID ارز برای CoinGecko
            coin_id = await self.get_coin_id(symbol)
            if not coin_id:
                return None
            
            url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
            
            data = await self.safe_api_request(url, symbol, cache_key, 600)
            if not data or 'prices' not in data:
                print(f"داده‌ای برای {symbol} دریافت نشد")
                return None
            
            # پردازش داده‌ها
            prices = data['prices']
            volumes = data['total_volumes']
            market_caps = data['market_caps']
            
            df_data = []
            for i in range(len(prices)):
                try:
                    timestamp = datetime.fromtimestamp(prices[i][0] / 1000)
                    row_data = {
                        'date': timestamp,
                        'timestamp': prices[i][0],
                        'open': prices[i][1] if i == 0 else prices[i-1][1],
                        'high': prices[i][1] * 1.02,  # تقریب برای high
                        'low': prices[i][1] * 0.98,   # تقریب برای low
                        'close': prices[i][1],
                        'volume': volumes[i][1] if i < len(volumes) else 0,
                        'market_cap': market_caps[i][1] if i < len(market_caps) else 0
                    }
                    df_data.append(row_data)
                except (ValueError, TypeError):
                    continue
            
            if len(df_data) < 30:
                print(f"داده‌های ناکافی برای {symbol}")
                return None
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna().sort_values('date').reset_index(drop=True)
            
            # محاسبه high/low واقعی‌تر بر اساس نوسانات
            for i in range(1, len(df)):
                df.at[i, 'high'] = max(df.at[i, 'open'], df.at[i, 'close']) * 1.01
                df.at[i, 'low'] = min(df.at[i, 'open'], df.at[i, 'close']) * 0.99
            
            self.data_cache[cache_key] = (df, datetime.now())
            return df
            
        except Exception as e:
            print(f"خطای کلی در دریافت داده‌های {symbol}: {e}")
            return None
    
    async def get_coin_id(self, symbol):
        """دریافت ID ارز از CoinGecko"""
        try:
            cache_key = "coin_list"
            if cache_key in self.data_cache:
                coin_list, cache_time = self.data_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < 3600:  # 1 ساعت کش
                    return next((coin['id'] for coin in coin_list if coin['symbol'].upper() == symbol.upper()), None)
            
            if not await self.safe_init_session():
                return None
            
            url = f"{COINGECKO_API_URL}/coins/list"
            coin_list = await self.safe_api_request(url, "coin_list", cache_key, 3600)
            
            if coin_list:
                return next((coin['id'] for coin in coin_list if coin['symbol'].upper() == symbol.upper()), None)
            
            return None
        except Exception as e:
            print(f"خطا در دریافت ID ارز {symbol}: {e}")
            return None
    
    async def get_whale_activity(self, symbol):
        """دریافت اطلاعات فعالیت نهنگ‌ها (شبیه‌سازی شده)"""
        try:
            cache_key = f"{symbol}_whale"
            if cache_key in self.whale_activity_cache:
                cached_data, cache_time = self.whale_activity_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < 1800:  # 30 دقیقه کش
                    return cached_data
            
            # در واقعیت، این داده‌ها از APIهای تخصصی مانند Glassnode یا CryptoQuant دریافت می‌شوند
            # اینجا برای نمونه، داده‌های شبیه‌سازی شده تولید می‌کنیم
            
            # شبیه‌سازی فعالیت نهنگ‌ها
            import random
            whale_activity = {
                'large_transactions': random.randint(5, 50),
                'total_volume': random.uniform(1000000, 50000000),
                'buy_ratio': random.uniform(0.3, 0.7),
                'sentiment': random.choice(['very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish'])
            }
            
            self.whale_activity_cache[cache_key] = (whale_activity, datetime.now())
            return whale_activity
            
        except Exception as e:
            print(f"خطا در دریافت فعالیت نهنگ‌ها برای {symbol}: {e}")
            return {'large_transactions': 0, 'total_volume': 0, 'buy_ratio': 0.5, 'sentiment': 'neutral'}
    
    async def get_market_sentiment(self, symbol):
        """دریافت احساسات بازار (شبیه‌سازی شده)"""
        try:
            cache_key = f"{symbol}_sentiment"
            if cache_key in self.market_sentiment_cache:
                cached_data, cache_time = self.market_sentiment_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < 1800:  # 30 دقیقه کش
                    return cached_data
            
            # در واقعیت، این داده‌ها از APIهای تخصصی دریافت می‌شوند
            # اینجا برای نمونه، داده‌های شبیه‌سازی شده تولید می‌کنیم
            
            import random
            sentiment = {
                'fear_greed': random.randint(0, 100),
                'social_volume': random.randint(1000, 100000),
                'positive_sentiment': random.uniform(0.2, 0.8),
                'volatility': random.uniform(1.0, 10.0)
            }
            
            self.market_sentiment_cache[cache_key] = (sentiment, datetime.now())
            return sentiment
            
        except Exception as e:
            print(f"خطا در دریافت احساسات بازار برای {symbol}: {e}")
            return {'fear_greed': 50, 'social_volume': 0, 'positive_sentiment': 0.5, 'volatility': 2.0}
    
    def calculate_advanced_indicators(self, df):
        """محاسبه 25 اندیکاتور پیشرفته"""
        indicators = {}
        
        if df is None or len(df) < 50:
            return self.get_default_indicators()
        
        try:
            close = np.array(df['close'].values, dtype=np.float64)
            high = np.array(df['high'].values, dtype=np.float64)
            low = np.array(df['low'].values, dtype=np.float64)
            volume = np.array(df['volume'].values, dtype=np.float64)
            open_price = np.array(df['open'].values, dtype=np.float64)
            
            # 1. RSI
            indicators['RSI'] = self.safe_rsi(close, 14)
            
            # 2. MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['MACD'] = macd[-1] if not np.isnan(macd[-1]) else 0
            indicators['MACD_Signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            indicators['MACD_Hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            
            # 3. Bollinger Bands
            upper_bb, middle_bb, lower_bb = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            indicators['BB_Upper'] = upper_bb[-1] if not np.isnan(upper_bb[-1]) else 0
            indicators['BB_Middle'] = middle_bb[-1] if not np.isnan(middle_bb[-1]) else 0
            indicators['BB_Lower'] = lower_bb[-1] if not np.isnan(lower_bb[-1]) else 0
            indicators['BB_Percent'] = (close[-1] - lower_bb[-1]) / (upper_bb[-1] - lower_bb[-1]) if upper_bb[-1] != lower_bb[-1] else 0.5
            
            # 4. Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            indicators['STOCH_K'] = slowk[-1] if not np.isnan(slowk[-1]) else 50
            indicators['STOCH_D'] = slowd[-1] if not np.isnan(slowd[-1]) else 50
            
            # 5. ADX
            indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 25
            
            # 6. OBV
            indicators['OBV'] = talib.OBV(close, volume)[-1] if len(close) > 0 else 0
            
            # 7. CCI
            indicators['CCI'] = talib.CCI(high, low, close, timeperiod=20)[-1] if len(close) >= 20 else 0
            
            # 8. ATR
            indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            
            # 9. Williams %R
            indicators['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else -50
            
            # 10. Money Flow Index
            indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1] if len(close) >= 14 else 50
            
            # 11. EMA Cross (12 & 26)
            ema12 = talib.EMA(close, timeperiod=12)
            ema26 = talib.EMA(close, timeperiod=26)
            indicators['EMA12'] = ema12[-1] if not np.isnan(ema12[-1]) else close[-1]
            indicators['EMA26'] = ema26[-1] if not np.isnan(ema26[-1]) else close[-1]
            indicators['EMA12_26_Cross'] = 1 if ema12[-1] > ema26[-1] else -1
            
            # 12. EMA Cross (50 & 200)
            ema50 = talib.EMA(close, timeperiod=50)
            ema200 = talib.EMA(close, timeperiod=200)
            indicators['EMA50'] = ema50[-1] if not np.isnan(ema50[-1]) else close[-1]
            indicators['EMA200'] = ema200[-1] if not np.isnan(ema200[-1]) else close[-1]
            indicators['EMA50_200_Cross'] = 1 if ema50[-1] > ema200[-1] else -1
            
            # 13. Parabolic SAR
            sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            indicators['SAR'] = sar[-1] if not np.isnan(sar[-1]) else close[-1]
            indicators['SAR_Direction'] = 1 if sar[-1] < close[-1] else -1
            
            # 14. Rate of Change
            indicators['ROC'] = talib.ROC(close, timeperiod=10)[-1] if len(close) >= 10 else 0
            
            # 15. Accumulation/Distribution
            indicators['AD'] = talib.AD(high, low, close, volume)[-1] if len(close) > 0 else 0
            
            # 16. VWAP (شبیه‌سازی شده)
            typical_price = (high + low + close) / 3
            vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
            indicators['VWAP'] = vwap[-1] if not np.isnan(vwap[-1]) else close[-1]
            indicators['VWAP_Signal'] = 1 if close[-1] > vwap[-1] else -1
            
            # 17. Ichimoku Cloud (شبیه‌سازی شده)
            tenkan_sen = (np.max(high[-9:]) + np.min(low[-9:])) / 2 if len(close) >= 9 else close[-1]
            kijun_sen = (np.max(high[-26:]) + np.min(low[-26:])) / 2 if len(close) >= 26 else close[-1]
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            senkou_span_b = (np.max(high[-52:]) + np.min(low[-52:])) / 2 if len(close) >= 52 else close[-1]
            
            indicators['Ichimoku_Tenkan'] = tenkan_sen
            indicators['Ichimoku_Kijun'] = kijun_sen
            indicators['Ichimoku_Senkou_A'] = senkou_span_a
            indicators['Ichimoku_Senkou_B'] = senkou_span_b
            indicators['Ichimoku_Cloud'] = 1 if close[-1] > max(senkou_span_a, senkou_span_b) else -1
            
            # 18. Momentum
            indicators['MOM'] = talib.MOM(close, timeperiod=10)[-1] if len(close) >= 10 else 0
            
            # 19. PPO
            indicators['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)[-1] if len(close) >= 26 else 0
            
            # 20. Slope (روند خطی)
            if len(close) >= 20:
                x = np.arange(20)
                y = close[-20:]
                slope = np.polyfit(x, y, 1)[0]
                indicators['SLOPE'] = slope
            else:
                indicators['SLOPE'] = 0
            
            # 21. Supertrend (شبیه‌سازی شده)
            atr_multiplier = 3
            hl2 = (high + low) / 2
            supertrend_upper = hl2 + (atr_multiplier * indicators['ATR'])
            supertrend_lower = hl2 - (atr_multiplier * indicators['ATR'])
            
            if close[-1] > supertrend_upper:
                indicators['SUPERTREND'] = 1  # روند صعودی
            elif close[-1] < supertrend_lower:
                indicators['SUPERTREND'] = -1  # روند نزولی
            else:
                indicators['SUPERTREND'] = 0  # خنثی
            
            # 22. Fibonacci Retracement
            max_price = np.max(close[-30:]) if len(close) >= 30 else close[-1]
            min_price = np.min(close[-30:]) if len(close) >= 30 else close[-1]
            diff = max_price - min_price
            
            indicators['FIB_0'] = max_price
            indicators['FIB_0.236'] = max_price - 0.236 * diff
            indicators['FIB_0.382'] = max_price - 0.382 * diff
            indicators['FIB_0.5'] = max_price - 0.5 * diff
            indicators['FIB_0.618'] = max_price - 0.618 * diff
            indicators['FIB_1'] = min_price
            indicators['FIB_Level'] = self.get_fib_level(close[-1], max_price, min_price)
            
            # 23. Volume Profile (شبیه‌سازی شده)
            price_range = np.linspace(min_price, max_price, 10)
            volume_profile = []
            for i in range(len(price_range)-1):
                mask = (close >= price_range[i]) & (close <= price_range[i+1])
                volume_profile.append(np.sum(volume[mask]))
            
            indicators['VOLUME_PROFILE_POC'] = price_range[np.argmax(volume_profile)]  # Point of Control
            indicators['VOLUME_PROFILE_Value'] = np.max(volume_profile) / np.mean(volume_profile) if np.mean(volume_profile) > 0 else 1
            
            # قیمت فعلی و اطلاعات پایه
            indicators['close'] = float(close[-1]) if len(close) > 0 else 0
            indicators['high'] = float(high[-1]) if len(high) > 0 else 0
            indicators['low'] = float(low[-1]) if len(low) > 0 else 0
            indicators['open'] = float(open_price[-1]) if len(open_price) > 0 else 0
            indicators['volume'] = float(volume[-1]) if len(volume) > 0 else 0
            
            return indicators
            
        except Exception as e:
            print(f"خطا در محاسبه اندیکاتورها: {e}")
            return self.get_default_indicators()
    
    def get_fib_level(self, current_price, max_price, min_price):
        """تعیین سطح فیبوناچی"""
        diff = max_price - min_price
        if diff == 0:
            return "N/A"
        
        retracement = (max_price - current_price) / diff
        
        if retracement < 0.236:
            return "0-0.236"
        elif retracement < 0.382:
            return "0.236-0.382"
        elif retracement < 0.5:
            return "0.382-0.5"
        elif retracement < 0.618:
            return "0.5-0.618"
        else:
            return "0.618-1"
    
    def get_default_indicators(self):
        """مقادیر پیش‌فرض برای زمانی که محاسبه ممکن نیست"""
        return {
            'RSI': 50.0, 'MACD': 0.0, 'MACD_Signal': 0.0, 'MACD_Hist': 0.0,
            'BB_Upper': 0.0, 'BB_Middle': 0.0, 'BB_Lower': 0.0, 'BB_Percent': 0.5,
            'STOCH_K': 50.0, 'STOCH_D': 50.0, 'ADX': 25.0, 'OBV': 0.0,
            'CCI': 0.0, 'ATR': 0.0, 'WILLR': -50.0, 'MFI': 50.0,
            'EMA12': 0.0, 'EMA26': 0.0, 'EMA12_26_Cross': 0,
            'EMA50': 0.0, 'EMA200': 0.0, 'EMA50_200_Cross': 0,
            'SAR': 0.0, 'SAR_Direction': 0, 'ROC': 0.0, 'AD': 0.0,
            'VWAP': 0.0, 'VWAP_Signal': 0, 'Ichimoku_Tenkan': 0.0,
            'Ichimoku_Kijun': 0.0, 'Ichimoku_Senkou_A': 0.0, 'Ichimoku_Senkou_B': 0.0,
            'Ichimoku_Cloud': 0, 'MOM': 0.0, 'PPO': 0.0, 'SLOPE': 0.0,
            'SUPERTREND': 0, 'FIB_Level': "N/A", 'VOLUME_PROFILE_POC': 0.0,
            'VOLUME_PROFILE_Value': 1.0, 'close': 0.0, 'high': 0.0,
            'low': 0.0, 'open': 0.0, 'volume': 0.0
        }
    
    def safe_rsi(self, prices, period=14):
        """محاسبه RSI با مدیریت خطا"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
            
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return max(0.0, min(100.0, rsi))
        except:
            return 50.0
    
    async def generate_advanced_signal(self, symbol, indicators, whale_activity, market_sentiment, strategy_type='moderate'):
        """تولید سیگنال پیشرفته با استفاده از 25 اندیکاتور"""
        try:
            if not indicators:
                return "HOLD", 0.5, 0, 0, 1, "داده ناکافی"
            
            weights = INDICATOR_WEIGHTS.get(strategy_type, INDICATOR_WEIGHTS['moderate'])
            total_score = 0.0
            max_score = float(sum(weights.values()))
            
            if max_score <= 0:
                return "HOLD", 0.5, 0, 0, 1, "وزن‌های نامعتبر"
            
            current_price = indicators.get('close', 0)
            
            # تحلیل RSI
            rsi = indicators.get('RSI', 50)
            if rsi < 30:
                total_score += weights['RSI'] * 1.0
                rsi_signal = "اشباع فروش"
            elif rsi > 70:
                total_score -= weights['RSI'] * 1.0
                rsi_signal = "اشباع خرید"
            else:
                rsi_factor = (rsi - 50) / 20
                total_score += weights['RSI'] * rsi_factor
                rsi_signal = "خنثی"
            
            # تحلیل MACD
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_Signal', 0)
            if macd > macd_signal:
                total_score += weights['MACD'] * 0.8
                macd_status = "صعودی"
            else:
                total_score -= weights['MACD'] * 0.8
                macd_status = "نزولی"
            
            # تحلیل Bollinger Bands
            bb_percent = indicators.get('BB_Percent', 0.5)
            if bb_percent < 0.2:
                total_score += weights['BBANDS'] * 0.7
                bb_signal = "نزدیک کف"
            elif bb_percent > 0.8:
                total_score -= weights['BBANDS'] * 0.7
                bb_signal = "نزدیک سقف"
            else:
                bb_signal = "در محدوده میانی"
            
            # تحلیل Stochastic
            stoch_k = indicators.get('STOCH_K', 50)
            stoch_d = indicators.get('STOCH_D', 50)
            if stoch_k < 20 and stoch_d < 20:
                total_score += weights['STOCH'] * 0.6
                stoch_signal = "اشباع فروش"
            elif stoch_k > 80 and stoch_d > 80:
                total_score -= weights['STOCH'] * 0.6
                stoch_signal = "اشباع خرید"
            else:
                stoch_signal = "خنثی"
            
            # تحلیل ADX (قدرت روند)
            adx = indicators.get('ADX', 25)
            if adx > 25:
                total_score += weights['ADX'] * 0.5
                adx_signal = "روند قوی"
            else:
                adx_signal = "روند ضعیف"
            
            # تحلیل EMA Crosses
            ema12_26_cross = indicators.get('EMA12_26_Cross', 0)
            if ema12_26_cross > 0:
                total_score += weights['EMA12_26'] * 0.6
                ema12_26_signal = "صعودی"
            else:
                total_score -= weights['EMA12_26'] * 0.6
                ema12_26_signal = "نزولی"
            
            ema50_200_cross = indicators.get('EMA50_200_Cross', 0)
            if ema50_200_cross > 0:
                total_score += weights['EMA50_200'] * 0.8
                ema50_200_signal = "صعودی"
            else:
                total_score -= weights['EMA50_200'] * 0.8
                ema50_200_signal = "نزولی"
            
            # تحلیل Ichimoku Cloud
            ichimoku_cloud = indicators.get('Ichimoku_Cloud', 0)
            if ichimoku_cloud > 0:
                total_score += weights['ICHIMOKU'] * 0.7
                ichimoku_signal = "صعودی"
            elif ichimoku_cloud < 0:
                total_score -= weights['ICHIMOKU'] * 0.7
                ichimoku_signal = "نزولی"
            else:
                ichimoku_signal = "خنثی"
            
            # تحلیل Supertrend
            supertrend = indicators.get('SUPERTREND', 0)
            if supertrend > 0:
                total_score += weights['SUPERTREND'] * 0.7
                supertrend_signal = "صعودی"
            elif supertrend < 0:
                total_score -= weights['SUPERTREND'] * 0.7
                supertrend_signal = "نزولی"
            else:
                supertrend_signal = "خنثی"
            
            # تحلیل فعالیت نهنگ‌ها
            whale_buy_ratio = whale_activity.get('buy_ratio', 0.5)
            if whale_buy_ratio > 0.6:
                total_score += weights['WHALE_ACTIVITY'] * 0.8
                whale_signal = "خرید نهنگ‌ها"
            elif whale_buy_ratio < 0.4:
                total_score -= weights['WHALE_ACTIVITY'] * 0.8
                whale_signal = "فروش نهنگ‌ها"
            else:
                whale_signal = "خنثی"
            
            # تحلیل احساسات بازار
            fear_greed = market_sentiment.get('fear_greed', 50)
            if fear_greed > 60:
                total_score += weights['MARKET_SENTIMENT'] * 0.5
                sentiment_signal = "طمع"
            elif fear_greed < 40:
                total_score -= weights['MARKET_SENTIMENT'] * 0.5
                sentiment_signal = "ترس"
            else:
                sentiment_signal = "خنثی"
            
            # تحلیل سایر اندیکاتورها
            # (می‌توانید تحلیل سایر اندیکاتورها را اینجا اضافه کنید)
            
            # نرمال‌سازی امتیاز
            normalized_score = total_score / max_score
            
            # تولید سیگنال
            if normalized_score > 0.15:
                signal = "BUY"
                confidence = min(max(normalized_score, 0.0), 0.95)
                
                # محاسبه حد ضرر و حد سود بر اساس ATR و سطوح کلیدی
                atr = indicators.get('ATR', current_price * 0.02)
                support = indicators.get('BB_Lower', current_price * 0.95)
                resistance = indicators.get('BB_Upper', current_price * 1.05)
                
                # تعیین حد ضرر (حداقل 1.5 برابر ATR یا 3% زیر قیمت)
                stop_loss = min(current_price - (1.5 * atr), support)
                stop_loss_pct = max(2.0, (1 - (stop_loss / current_price)) * 100)
                
                # تعیین حد سود (حداقل 3 برابر ATR یا 6% بالای قیمت)
                take_profit = max(current_price + (3 * atr), resistance)
                take_profit_pct = max(4.0, ((take_profit / current_price) - 1) * 100)
                
                # تعیین اهرم بر اساس اعتماد و نوسانات
                volatility = market_sentiment.get('volatility', 2.0)
                leverage = min(5, max(1, int(3 / volatility * confidence)))
                
                # تولید پیام تحلیلی
                analysis_msg = (
                    f"📈 سیگنال خرید برای {symbol}\n\n"
                    f"🔍 تحلیل فنی:\n"
                    f"• RSI: {rsi:.2f} ({rsi_signal})\n"
                    f"• MACD: {macd_status}\n"
                    f"• بولینگر: {bb_signal}\n"
                    f"• استوکاستیک: {stoch_signal}\n"
                    f"• ADX: {adx:.2f} ({adx_signal})\n"
                    f"• EMA12/26: {ema12_26_signal}\n"
                    f"• EMA50/200: {ema50_200_signal}\n"
                    f"• ابر ایچیموکو: {ichimoku_signal}\n"
                    f"• سوپرترند: {supertrend_signal}\n\n"
                    f"🐋 تحلیل نهنگ‌ها: {whale_signal}\n"
                    f"📊 احساسات بازار: {sentiment_signal}\n\n"
                    f"💰 قیمت فعلی: ${current_price:.4f}\n"
                    f"🛑 حد ضرر: ${stop_loss:.4f} ({stop_loss_pct:.2f}%)\n"
                    f"🎯 حد سود: ${take_profit:.4f} ({take_profit_pct:.2f}%)\n"
                    f"⚖️ اهرم پیشنهادی: {leverage}x\n"
                    f"🔒 اطمینان: {confidence*100:.1f}%"
                )
                
                return signal, confidence, stop_loss, take_profit, leverage, analysis_msg
                
            elif normalized_score < -0.15:
                signal = "SELL"
                confidence = min(max(abs(normalized_score), 0.0), 0.95)
                
                # محاسبه حد ضرر و حد سود برای فروش
                atr = indicators.get('ATR', current_price * 0.02)
                support = indicators.get('BB_Lower', current_price * 0.95)
                resistance = indicators.get('BB_Upper', current_price * 1.05)
                
                # تعیین حد ضرر برای فروش (بالای قیمت)
                stop_loss = max(current_price + (1.5 * atr), resistance)
                stop_loss_pct = max(2.0, ((stop_loss / current_price) - 1) * 100)
                
                # تعیین حد سود برای فروش (زیر قیمت)
                take_profit = min(current_price - (3 * atr), support)
                take_profit_pct = max(4.0, (1 - (take_profit / current_price)) * 100)
                
                # تعیین اهرم
                volatility = market_sentiment.get('volatility', 2.0)
                leverage = min(5, max(1, int(3 / volatility * confidence)))
                
                # تولید پیام تحلیلی
                analysis_msg = (
                    f"📉 سیگنال فروش برای {symbol}\n\n"
                    f"🔍 تحلیل فنی:\n"
                    f"• RSI: {rsi:.2f} ({rsi_signal})\n"
                    f"• MACD: {macd_status}\n"
                    f"• بولینگر: {bb_signal}\n"
                    f"• استوکاستیک: {stoch_signal}\n"
                    f"• ADX: {adx:.2f} ({adx_signal})\n"
                    f"• EMA12/26: {ema12_26_signal}\n"
                    f"• EMA50/200: {ema50_200_signal}\n"
                    f"• ابر ایچیموکو: {ichimoku_signal}\n"
                    f"• سوپرترند: {supertrend_signal}\n\n"
                    f"🐋 تحلیل نهنگ‌ها: {whale_signal}\n"
                    f"📊 احساسات بازار: {sentiment_signal}\n\n"
                    f"💰 قیمت فعلی: ${current_price:.4f}\n"
                    f"🛑 حد ضرر: ${stop_loss:.4f} (+{stop_loss_pct:.2f}%)\n"
                    f"🎯 حد سود: ${take_profit:.4f} (-{take_profit_pct:.2f}%)\n"
                    f"⚖️ اهرم پیشنهادی: {leverage}x\n"
                    f"🔒 اطمینان: {confidence*100:.1f}%"
                )
                
                return signal, confidence, stop_loss, take_profit, leverage, analysis_msg
                
            else:
                signal = "HOLD"
                confidence = 0.5
                
                analysis_msg = (
                    f"⏸️ سیگنال نگهداری برای {symbol}\n\n"
                    f"🔍 تحلیل فنی:\n"
                    f"• RSI: {rsi:.2f} ({rsi_signal})\n"
                    f"• MACD: {macd_status}\n"
                    f"• بولینگر: {bb_signal}\n"
                    f"• استوکاستیک: {stoch_signal}\n"
                    f"• ADX: {adx:.2f} ({adx_signal})\n"
                    f"• EMA12/26: {ema12_26_signal}\n"
                    f"• EMA50/200: {ema50_200_signal}\n"
                    f"• ابر ایچیموکو: {ichimoku_signal}\n"
                    f"• سوپرترند: {supertrend_signal}\n\n"
                    f"🐋 تحلیل نهنگ‌ها: {whale_signal}\n"
                    f"📊 احساسات بازار: {sentiment_signal}\n\n"
                    f"💰 قیمت فعلی: ${current_price:.4f}\n"
                    f"📈 بازار در حالت بلاتکلیف است. بهتر است منتظر سیگنال واضح‌تری بمانید."
                )
                
                return signal, confidence, 0, 0, 1, analysis_msg
                
        except Exception as e:
            print(f"خطا در تولید سیگنال برای {symbol}: {e}")
            return "HOLD", 0.5, 0, 0, 1, f"خطا در تحلیل {symbol}: {str(e)}"

# ایجاد نمونه آنالایزر
analyzer = AdvancedCryptoAnalyzer()

# دستورات ربات
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """دستور شروع"""
    welcome_text = (
        "🤖 به ربات تحلیلگر پیشرفته ارزهای دیجیتال خوش آمدید!\n\n"
        "این ربات با استفاده از 25 اندیکاتور تکنیکال، تحلیل نهنگ‌ها و احساسات بازار، "
        "سیگنال‌های دقیق خرید و فروش ارائه می‌دهد.\n\n"
        "دستورات موجود:\n"
        "/analyze [نماد] - تحلیل یک ارز خاص (مثلاً /analyze BTC)\n"
        "/analyze_all - تحلیل همه ارزهای اصلی\n"
        "/strategies - مشاهده استراتژی‌های معاملاتی"
    )
    await update.message.reply_text(welcome_text)

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تحلیل یک ارز خاص"""
    try:
        if not context.args:
            await update.message.reply_text("لطفاً نماد ارز را وارد کنید. مثال: /analyze BTC")
            return
        
        symbol = context.args[0].upper()
        if symbol not in CRYPTO_SYMBOLS:
            await update.message.reply_text(f"ارز {symbol} پشتیبانی نمی‌شود. ارزهای قابل تحلیل: {', '.join(CRYPTO_SYMBOLS)}")
            return
        
        # ارسال پیام در حال پردازش
        processing_msg = await update.message.reply_text(f"🔍 در حال تحلیل {symbol}... لطفاً منتظر بمانید.")
        
        # دریافت داده‌ها
        df = await analyzer.get_crypto_data(symbol)
        if df is None:
            await processing_msg.edit_text(f"خطا در دریافت داده‌های {symbol}. لطفاً دوباره تلاش کنید.")
            return
        
        # محاسبه اندیکاتورها
        indicators = analyzer.calculate_advanced_indicators(df)
        
        # دریافت اطلاعات نهنگ‌ها و احساسات بازار
        whale_activity = await analyzer.get_whale_activity(symbol)
        market_sentiment = await analyzer.get_market_sentiment(symbol)
        
        # تولید سیگنال
        signal, confidence, stop_loss, take_profit, leverage, analysis_msg = await analyzer.generate_advanced_signal(
            symbol, indicators, whale_activity, market_sentiment
        )
        
        # ارسال نتیجه
        await processing_msg.edit_text(analysis_msg)
        
    except Exception as e:
        print(f"خطا در تحلیل ارز: {e}")
        await update.message.reply_text(f"خطا در تحلیل: {str(e)}")

async def analyze_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تحلیل همه ارزهای اصلی"""
    try:
        processing_msg = await update.message.reply_text("🔍 در حال تحلیل همه ارزها... لطفاً منتظر بمانید.")
        
        results = []
        for symbol in CRYPTO_SYMBOLS:
            try:
                df = await analyzer.get_crypto_data(symbol)
                if df is None:
                    continue
                
                indicators = analyzer.calculate_advanced_indicators(df)
                whale_activity = await analyzer.get_whale_activity(symbol)
                market_sentiment = await analyzer.get_market_sentiment(symbol)
                
                signal, confidence, _, _, _, _ = await analyzer.generate_advanced_signal(
                    symbol, indicators, whale_activity, market_sentiment
                )
                
                current_price = indicators.get('close', 0)
                results.append((symbol, signal, confidence, current_price))
                
            except Exception as e:
                print(f"خطا در تحلیل {symbol}: {e}")
                continue
        
        # مرتب‌سازی نتایج بر اساس اعتماد
        results.sort(key=lambda x: x[2], reverse=True)
        
        # ایجاد گزارش خلاصه
        report = "📊 گزارش تحلیل همه ارزها:\n\n"
        for symbol, signal, confidence, price in results:
            emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            report += f"{emoji} {symbol}: {signal} (اطمینان: {confidence*100:.1f}%) - ${price:.4f}\n"
        
        report += "\n💡 برای تحلیل دقیق‌تر هر ارز از دستور /analyze استفاده کنید."
        
        await processing_msg.edit_text(report)
        
    except Exception as e:
        print(f"خطا در تحلیل همه ارزها: {e}")
        await update.message.reply_text(f"خطا در تحلیل: {str(e)}")

async def strategies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """نمایش استراتژی‌های معاملاتی"""
    strategies_text = (
        "🎯 استراتژی‌های معاملاتی:\n\n"
        "1. محافظه‌کارانه: ریسک کم، اهرم پایین، سود کمتر اما مطمئن‌تر\n"
        "2. متعادل: ترکیبی از ریسک و بازده، اهرم متوسط\n"
        "3. تهاجمی: ریسک بالا، اهرم بالا، پتانسیل سود بیشتر\n\n"
        "برای تغییر استراتژی، در کد ربات تنظیمات INDICATOR_WEIGHTS را تغییر دهید."
    )
    await update.message.reply_text(strategies_text)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """مدیریت خطاهای ربات"""
    print(f"خطا رخ داده: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text("متأسفانه خطایی رخ داده است. لطفاً دوباره تلاش کنید.")

async def cleanup(context: ContextTypes.DEFAULT_TYPE):
    """تمیز کردن منابع قبل از خروج"""
    await analyzer.safe_close_session()

def main():
    """تابع اصلی"""
    print("🤖 در حال راه‌اندازی ربات تحلیلگر ارزهای دیجیتال...")
    
    # ایجاد برنامه تلگرام
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # اضافه کردن دستورات
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze))
    application.add_handler(CommandHandler("analyze_all", analyze_all))
    application.add_handler(CommandHandler("strategies", strategies))
    
    # مدیریت خطاها
    application.add_error_handler(error_handler)
    
    # تمیز کردن منابع
    application.job_queue.run_once(cleanup, when=0)
    
    # شروع ربات
    print("✅ ربات راه‌اندازی شد. برای توقف Ctrl+C را بفشارید.")
    application.run_polling()

if __name__ == "__main__":
    main()
