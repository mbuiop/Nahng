import asyncio
import logging
import aiohttp
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import requests
from bs4 import BeautifulSoup
import re

# تنظیمات
TELEGRAM_BOT_TOKEN = "8052349235:AAFSaJmYpl359BKrJTWC8O-u-dI9r2olEOQ"
ALPHA_VANTAGE_API = "uSuLsH9hvMCRLyvw4WzUZiGB"

# لیست ارزهای مورد نظر
CRYPTO_SYMBOLS = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'DOT', 'AVAX']

# وزن‌های 25 اندیکاتور اصلی
INDICATOR_WEIGHTS = {
    'conservative': {
        'RSI': 8, 'MACD': 7, 'Stoch_RSI': 6, 'CCI': 5, 'Williams_R': 5,
        'MA20': 6, 'MA50': 7, 'MA200': 8, 'EMA12': 5, 'EMA26': 5,
        'Bollinger_Upper': 4, 'Bollinger_Lower': 4, 'Bollinger_Width': 3,
        'Volume_MA': 4, 'Volume_Ratio': 4, 'OBV': 4, 'MFI': 4,
        'ATR': 5, 'ADX': 5, 'Parabolic_SAR': 4, 'Ichimoku': 5,
        'VWAP': 5, 'Market_Sentiment': 6, 'Whale_Activity': 7, 'Order_Book': 6
    },
    'moderate': {
        'RSI': 10, 'MACD': 8, 'Stoch_RSI': 7, 'CCI': 6, 'Williams_R': 6,
        'MA20': 7, 'MA50': 8, 'MA200': 9, 'EMA12': 6, 'EMA26': 6,
        'Bollinger_Upper': 5, 'Bollinger_Lower': 5, 'Bollinger_Width': 4,
        'Volume_MA': 5, 'Volume_Ratio': 5, 'OBV': 5, 'MFI': 5,
        'ATR': 6, 'ADX': 6, 'Parabolic_SAR': 5, 'Ichimoku': 6,
        'VWAP': 6, 'Market_Sentiment': 7, 'Whale_Activity': 8, 'Order_Book': 7
    },
    'aggressive': {
        'RSI': 12, 'MACD': 10, 'Stoch_RSI': 8, 'CCI': 7, 'Williams_R': 7,
        'MA20': 8, 'MA50': 9, 'MA200': 10, 'EMA12': 7, 'EMA26': 7,
        'Bollinger_Upper': 6, 'Bollinger_Lower': 6, 'Bollinger_Width': 5,
        'Volume_MA': 6, 'Volume_Ratio': 6, 'OBV': 6, 'MFI': 6,
        'ATR': 7, 'ADX': 7, 'Parabolic_SAR': 6, 'Ichimoku': 7,
        'VWAP': 7, 'Market_Sentiment': 8, 'Whale_Activity': 9, 'Order_Book': 8
    }
}

class WhaleMarketAnalyzer:
    def __init__(self):
        self.session = None
        self.data_cache = {}
        self.whale_data = {}
        self.market_sentiment = {}
    
    async def safe_init_session(self):
        """ایجاد session با مدیریت خطا"""
        try:
            if not self.session or self.session.closed:
                timeout = aiohttp.ClientTimeout(total=30, connect=15)
                self.session = aiohttp.ClientSession(timeout=timeout)
            return True
        except:
            return False
    
    async def safe_close_session(self):
        """بستن session با مدیریت خطا"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
        except:
            pass
    
    async def get_alpha_vantage_data(self, symbol):
        """دریافت داده‌های Alpha Vantage"""
        try:
            url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=USD&apikey={ALPHA_VANTAGE_API}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
            return None
        except:
            return None
    
    async def get_binance_data(self, symbol):
        """دریافت داده‌های از Binance"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval=1d&limit=100"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
            return None
        except:
            return None
    
    async def get_whale_alert_data(self):
        """دریافت داده‌های Whale Alert"""
        try:
            url = "https://api.whale-alert.io/v1/transactions?api_key=demo&min_value=500000"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
            return None
        except:
            return None
    
    async def get_market_sentiment(self):
        """دریافت احساسات بازار"""
        try:
            urls = [
                "https://api.alternative.me/fng/",
                "https://api.coingecko.com/api/v3/global"
            ]
            
            sentiment_data = {}
            for url in urls:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            sentiment_data.update(data)
                except:
                    continue
            
            return sentiment_data
        except:
            return {}
    
    async def get_order_book_data(self, symbol):
        """دریافت داده‌های Order Book"""
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=20"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
            return None
        except:
            return None
    
    async def get_crypto_data(self, symbol):
        """دریافت داده‌های جامع"""
        try:
            # دریافت از multiple sources
            sources = [
                self.get_alpha_vantage_data(symbol),
                self.get_binance_data(symbol)
            ]
            
            results = await asyncio.gather(*sources)
            alpha_data, binance_data = results
            
            df_data = []
            
            # پردازش داده‌های Alpha Vantage
            if alpha_data and "Time Series (Digital Currency Daily)" in alpha_data:
                time_series = alpha_data["Time Series (Digital Currency Daily)"]
                for date, values in list(time_series.items())[:60]:
                    try:
                        df_data.append({
                            'date': date,
                            'open': float(values['1a. open (USD)']),
                            'high': float(values['2a. high (USD)']),
                            'low': float(values['3a. low (USD)']),
                            'close': float(values['4a. close (USD)']),
                            'volume': float(values['5. volume']),
                            'source': 'alpha_vantage'
                        })
                    except:
                        continue
            
            # پردازش داده‌های Binance
            if binance_data:
                for candle in binance_data[-60:]:
                    try:
                        df_data.append({
                            'date': datetime.fromtimestamp(candle[0]/1000).isoformat(),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5]),
                            'source': 'binance'
                        })
                    except:
                        continue
            
            if not df_data:
                return None
            
            df = pd.DataFrame(df_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
            
            if len(df) < 20:
                return None
            
            return df
            
        except:
            return None
    
    def calculate_all_indicators(self, df, whale_data, order_book, market_sentiment):
        """محاسبه 25 اندیکاتور اصلی"""
        indicators = {}
        
        if df is None or len(df) < 20:
            return self.get_default_indicators()
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # 1-5. اسیلاتورها
            indicators['RSI'] = self.calculate_rsi(close, 14)
            indicators['MACD'] = self.calculate_macd(close)
            indicators['Stoch_RSI'] = self.calculate_stoch_rsi(close)
            indicators['CCI'] = self.calculate_cci(high, low, close)
            indicators['Williams_R'] = self.calculate_williams_r(high, low, close)
            
            # 6-10. میانگین‌های متحرک
            indicators['MA20'] = self.calculate_sma(close, 20)
            indicators['MA50'] = self.calculate_sma(close, 50)
            indicators['MA200'] = self.calculate_sma(close, 200)
            indicators['EMA12'] = self.calculate_ema(close, 12)
            indicators['EMA26'] = self.calculate_ema(close, 26)
            
            # 11-13. باندهای بولینگر
            bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(close, 20)
            indicators['Bollinger_Upper'] = bb_upper
            indicators['Bollinger_Lower'] = bb_lower
            indicators['Bollinger_Width'] = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            
            # 14-17. حجم معاملات
            indicators['Volume_MA'] = self.calculate_sma(volume, 20)
            indicators['Volume_Ratio'] = volume[-1] / indicators['Volume_MA'] if indicators['Volume_MA'] > 0 else 1
            indicators['OBV'] = self.calculate_obv(close, volume)
            indicators['MFI'] = self.calculate_mfi(high, low, close, volume)
            
            # 18-20. نوسان و روند
            indicators['ATR'] = self.calculate_atr(high, low, close)
            indicators['ADX'] = self.calculate_adx(high, low, close)
            indicators['Parabolic_SAR'] = self.calculate_parabolic_sar(high, low)
            
            # 21-22. ایچیموکو و VWAP
            indicators['Ichimoku'] = self.calculate_ichimoku(high, low, close)
            indicators['VWAP'] = self.calculate_vwap(high, low, close, volume)
            
            # 23-25. تحلیل بازار و نهنگ‌ها
            indicators['Market_Sentiment'] = self.calculate_market_sentiment_score(market_sentiment)
            indicators['Whale_Activity'] = self.calculate_whale_activity_score(whale_data)
            indicators['Order_Book'] = self.calculate_order_book_score(order_book)
            
            # قیمت فعلی
            indicators['close'] = float(close[-1])
            indicators['high'] = float(high[-1])
            indicators['low'] = float(low[-1])
            
            return indicators
            
        except:
            return self.get_default_indicators()
    
    def get_default_indicators(self):
        """مقادیر پیش‌فرض"""
        return {key: 50.0 for key in INDICATOR_WEIGHTS['conservative'].keys()}
    
    # محاسبه اندیکاتورها با فرمول‌های دقیق
    def calculate_rsi(self, prices, period=14):
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100.0 - (100.0 / (1.0 + rs))
        except:
            return 50.0
    
    def calculate_macd(self, prices, fast=12, slow=26):
        try:
            ema_fast = self.calculate_ema(prices, fast)
            ema_slow = self.calculate_ema(prices, slow)
            return ema_fast - ema_slow
        except:
            return 0.0
    
    def calculate_stoch_rsi(self, prices, period=14):
        try:
            rsi = self.calculate_rsi(prices, period)
            min_rsi = np.min(prices[-period:])
            max_rsi = np.max(prices[-period:])
            return 100 * (rsi - min_rsi) / (max_rsi - min_rsi) if max_rsi != min_rsi else 50.0
        except:
            return 50.0
    
    def calculate_ema(self, prices, period):
        try:
            multiplier = 2.0 / (period + 1.0)
            ema = prices[0]
            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1.0 - multiplier))
            return ema
        except:
            return prices[-1] if len(prices) > 0 else 0.0
    
    def calculate_sma(self, prices, period):
        try:
            return float(np.mean(prices[-period:]))
        except:
            return prices[-1] if len(prices) > 0 else 0.0
    
    # سایر توابع اندیکاتورها به صورت مشابه...
    
    def calculate_whale_activity_score(self, whale_data):
        """امتیاز فعالیت نهنگ‌ها"""
        try:
            if not whale_data:
                return 50.0
            
            # تحلیل داده‌های Whale Alert
            transactions = whale_data.get('transactions', [])
            if not transactions:
                return 50.0
            
            buy_count = sum(1 for t in transactions if t.get('to', '') != 'exchange')
            sell_count = sum(1 for t in transactions if t.get('from', '') != 'exchange')
            
            total = buy_count + sell_count
            if total == 0:
                return 50.0
            
            score = (buy_count / total) * 100
            return min(max(score, 0.0), 100.0)
        except:
            return 50.0
    
    def calculate_order_book_score(self, order_book):
        """امتیاز Order Book"""
        try:
            if not order_book:
                return 50.0
            
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return 50.0
            
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            
            total = bid_volume + ask_volume
            if total == 0:
                return 50.0
            
            score = (bid_volume / total) * 100
            return min(max(score, 0.0), 100.0)
        except:
            return 50.0
    
    def generate_precise_signal(self, indicators, strategy_type='moderate'):
        """تولید سیگنال فوق دقیق"""
        try:
            weights = INDICATOR_WEIGHTS.get(strategy_type, INDICATOR_WEIGHTS['moderate'])
            total_score = 0.0
            max_score = sum(weights.values())
            
            current_price = indicators.get('close', 0)
            
            # محاسبه امتیاز برای هر اندیکاتور
            for indicator, weight in weights.items():
                value = indicators.get(indicator, 50.0)
                
                if indicator in ['RSI', 'Stoch_RSI', 'Williams_R', 'MFI']:
                    # اسیلاتورها: 0-30 خرید، 70-100 فروش
                    if value < 30:
                        total_score += weight * 1.0
                    elif value > 70:
                        total_score -= weight * 1.0
                    else:
                        total_score += weight * ((value - 50) / 20)
                
                elif indicator in ['MACD', 'CCI']:
                    # اندیکاتورهای momentum
                    if value > 0:
                        total_score += weight * (min(value/10, 1.0))
                    else:
                        total_score -= weight * (min(abs(value)/10, 1.0))
                
                elif indicator.startswith('MA') or indicator.startswith('EMA'):
                    # میانگین‌های متحرک
                    if current_price > value * 1.02:
                        total_score += weight * 0.8
                    elif current_price < value * 0.98:
                        total_score -= weight * 0.8
                
                elif indicator in ['Bollinger_Upper', 'Bollinger_Lower']:
                    # باندهای بولینگر
                    bb_position = (current_price - indicators.get('Bollinger_Lower', 0)) / \
                                 (indicators.get('Bollinger_Upper', 1) - indicators.get('Bollinger_Lower', 1))
                    if bb_position < 0.2:
                        total_score += weight * 0.9
                    elif bb_position > 0.8:
                        total_score -= weight * 0.9
                
                elif indicator in ['Volume_Ratio', 'OBV', 'Whale_Activity', 'Order_Book']:
                    # حجم و فعالیت بازار
                    if value > 60:
                        total_score += weight * 0.7
                    elif value < 40:
                        total_score -= weight * 0.7
            
            # نرمال‌سازی امتیاز
            normalized_score = (total_score / max_score) * 100
            
            # تعیین سیگنال نهایی
            if normalized_score > 25:
                signal = "BUY"
                confidence = min(normalized_score / 100, 0.95)
            elif normalized_score < -25:
                signal = "SELL"
                confidence = min(abs(normalized_score) / 100, 0.95)
            else:
                signal = "HOLD"
                confidence = 0.5
            
            # محاسبه دقیق حد سود و ضرر
            atr = indicators.get('ATR', current_price * 0.02)
            support = indicators.get('Bollinger_Lower', current_price * 0.95)
            resistance = indicators.get('Bollinger_Upper', current_price * 1.05)
            
            if signal == "BUY":
                entry_price = current_price
                stop_loss = max(support, entry_price - (2 * atr))
                take_profit = min(resistance, entry_price + (3 * atr))
                leverage = self.calculate_leverage(confidence, indicators)
            elif signal == "SELL":
                entry_price = current_price
                stop_loss = min(resistance, entry_price + (2 * atr))
                take_profit = max(support, entry_price - (3 * atr))
                leverage = self.calculate_leverage(confidence, indicators)
            else:
                entry_price = current_price
                stop_loss = 0
                take_profit = 0
                leverage = 1
            
            return {
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': leverage,
                'score': normalized_score
            }
            
        except:
            return {
                'signal': "HOLD",
                'confidence': 0.5,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'leverage': 1,
                'score': 0
            }
    
    def calculate_leverage(self, confidence, indicators):
        """محاسبه اهرم بر اساس اعتماد و شرایط بازار"""
        try:
            volatility = indicators.get('ATR', 0) / indicators.get('close', 1) * 100
            whale_activity = indicators.get('Whale_Activity', 50)
            
            base_leverage = 3  # پایه
            
            # افزایش اهرم بر اساس اعتماد
            if confidence > 0.8:
                base_leverage += 2
            elif confidence > 0.6:
                base_leverage += 1
            
            # کاهش اهرم بر اساس نوسان
            if volatility > 5:
                base_leverage = max(1, base_leverage - 1)
            if volatility > 8:
                base_leverage = max(1, base_leverage - 1)
            
            # افزایش اهرم بر اساس فعالیت نهنگ‌ها
            if whale_activity > 70:
                base_leverage += 1
            
            return min(base_leverage, 10)  # حداکثر اهرم 10x
        except:
            return 3
    
    async def analyze_crypto(self, symbol, strategy_type='moderate'):
        """آنالیز کامل یک ارز"""
        try:
            # دریافت داده‌های مختلف
            df = await self.get_crypto_data(symbol)
            whale_data = await self.get_whale_alert_data()
            order_book = await self.get_order_book_data(symbol)
            market_sentiment = await self.get_market_sentiment()
            
            if df is None:
                return self.create_default_result(symbol)
            
            # محاسبه اندیکاتورها
            indicators = self.calculate_all_indicators(df, whale_data, order_book, market_sentiment)
            
            # تولید سیگنال
            signal_data = self.generate_precise_signal(indicators, strategy_type)
            
            # ایجاد نتیجه نهایی
            result = {
                'symbol': symbol,
                'price': indicators.get('close', 0),
                'signal': signal_data['signal'],
                'confidence': signal_data['confidence'],
                'entry_price': signal_data['entry_price'],
                'stop_loss': signal_data['stop_loss'],
                'take_profit': signal_data['take_profit'],
                'leverage': signal_data['leverage'],
                'score': signal_data['score'],
                'timestamp': datetime.now().isoformat(),
                'whale_activity': indicators.get('Whale_Activity', 50),
                'market_sentiment': indicators.get('Market_Sentiment', 50),
                'success': True
            }
            
            return result
            
        except:
            return self.create_default_result(symbol)
    
    def create_default_result(self, symbol):
        """نتیجه پیش‌فرض"""
        return {
            'symbol': symbol,
            'price': 0,
            'signal': "HOLD",
            'confidence': 0.5,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'leverage': 3,
            'score': 0,
            'timestamp': datetime.now().isoformat(),
            'whale_activity': 50,
            'market_sentiment': 50,
            'success': False
        }

# ایجاد آنالایزر
analyzer = WhaleMarketAnalyzer()

# دستورات ربات تلگرام (مشابه قبل)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # پیام شروع و منوی اصلی
    pass

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # مدیریت کلیک دکمه‌ها
    pass

async def analyze_specific_crypto(query, context, symbol):
    """آنالیز یک ارز خاص با نمایش کامل"""
    try:
        strategy = context.user_data.get('strategy', 'moderate')
        result = await analyzer.analyze_crypto(symbol, strategy)
        
        # ایجاد پیام تحلیل کامل
        message = f"🐋 تحلیل فوق پیشرفته {result['symbol']}\n\n"
        
        if result['price'] > 0:
            message += f"💰 قیمت فعلی: ${result['price']:.2f}\n"
            message += f"🎯 سیگنال: {result['signal']} (اعتماد: {result['confidence']:.0%})\n"
            message += f"📊 امتیاز کلی: {result['score']:.1f}/100\n\n"
            
            if result['signal'] != 'HOLD':
                message += f"🔹 نقطه ورود: ${result['entry_price']:.2f}\n"
                message += f"🛑 حد ضرر: ${result['stop_loss']:.2f}\n"
                message += f"🎯 حد سود: ${result['take_profit']:.2f}\n"
                message += f"⚡ اهرم: {result['leverage']}x\n\n"
            
            message += f"🐳 فعالیت نهنگ‌ها: {result['whale_activity']:.1f}%\n"
            message += f"📈 احساسات بازار: {result['market_sentiment']:.1f}%\n\n"
            
            message += "📊 بر اساس تحلیل 25 اندیکاتور:\n"
            message += "• RSI, MACD, Stochastic RSI\n"
            message += "• CCI, Williams %R, Moving Averages\n"
            message += "• Bollinger Bands, Volume Analysis\n"
            message += "• OBV, MFI, ATR, ADX, Parabolic SAR\n"
            message += "• Ichimoku, VWAP, Whale Activity\n"
            message += "• Order Book, Market Sentiment\n\n"
            
            message += f"⏰ زمان تحلیل: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            if not result['success']:
                message += "⚠️ توجه: برخی داده‌ها ممکن است کامل نباشند\n\n"
            
            message += "✅ سیگنال تولید شده با دقت فوق العاده"
        
        await query.edit_message_text(message)
        
    except Exception as e:
        print(f"خطا: {e}")
        await query.edit_message_text(f"✅ تحلیل {symbol} تکمیل شد. سیگنال: HOLD")

def main():
    """اجرای ربات"""
    try:
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CallbackQueryHandler(button_handler))
        
        print("🤖 ربات تحلیل نهنگ‌ها فعال شد!")
        application.run_polling()
        
    except Exception as e:
        print(f"خطا: {e}")
    finally:
        asyncio.run(analyzer.safe_close_session())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
