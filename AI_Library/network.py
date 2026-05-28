# network.py
import csv
import io
import os
import re
from datetime import datetime
from urllib.parse import urlparse

import requests
from langchain_community.llms import Ollama

class InternetEnabledAI:
    """具备网络访问能力的AI系统"""
    
    def __init__(self, model_name="deepseek-r1:7b", temperature=0.3):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=temperature)
        # 定义可用的网络工具函数
        self.available_functions = {
            "get_weather": self.get_weather,
            "get_stock_price": self.get_stock_price,
            "search_news": self.search_news,
            "get_current_time": self.get_current_time
        }
    
    def get_weather(self, city):
        """获取城市天气信息 using Open-Meteo"""
        try:
            # 城市坐标映射（可扩展为动态查询）
            city_coords = {
                "北京": {"lat": 39.9042, "lon": 116.4074},
                "上海": {"lat": 31.2304, "lon": 121.4737},
                "广州": {"lat": 23.1291, "lon": 113.2644},
                "深圳": {"lat": 22.5431, "lon": 114.0579},
                "杭州": {"lat": 30.2741, "lon": 120.1551},
                "成都": {"lat": 30.5728, "lon": 104.0668}
            }
            coords = city_coords.get(city, {"lat": 39.9042, "lon": 116.4074})  # 默认北京
            url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current_weather=true"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            # 天气代码映射（WMO 代码）
            weather_codes = {
                0: "晴", 1: "多云", 2: "多云", 3: "阴", 
                45: "雾", 48: "霾", 
                51: "小雨", 61: "小雨", 63: "中雨", 65: "大雨",
                71: "小雪", 73: "中雪", 75: "大雪"
            }
            
            weather = data["current_weather"]
            weather_desc = weather_codes.get(weather["weathercode"], "未知")
            return f"""
{city}天气信息:
- 温度: {weather['temperature']}℃
- 天气: {weather_desc}
- 风速: {weather['windspeed']} km/h
- 风向: {weather['winddirection']}°
- 是否白天: {'是' if weather['is_day'] else '否'}
"""
        except Exception as e:
            return f"天气查询失败: {str(e)}"
    
    def _normalize_stock_symbol(self, symbol):
        """Normalize common user inputs to Stooq symbols."""
        aliases = {
            "苹果": "AAPL.US",
            "特斯拉": "TSLA.US",
            "微软": "MSFT.US",
            "英伟达": "NVDA.US",
            "谷歌": "GOOGL.US",
            "亚马逊": "AMZN.US",
            "Meta": "META.US",
            "META": "META.US",
        }
        raw_symbol = (symbol or "AAPL").strip()
        if raw_symbol in aliases:
            return aliases[raw_symbol]

        cleaned = re.sub(r"[^A-Za-z0-9.\-]", "", raw_symbol).upper()
        if not cleaned:
            return "AAPL.US"
        if "." not in cleaned and re.fullmatch(r"[A-Z]{1,5}", cleaned):
            return f"{cleaned}.US"
        return cleaned

    def get_stock_price(self, symbol):
        """通过 Stooq 查询股票价格，无需 API Key。"""
        try:
            stooq_symbol = self._normalize_stock_symbol(symbol)
            response = requests.get(
                "https://stooq.com/q/l/",
                params={
                    "s": stooq_symbol.lower(),
                    "f": "sd2t2ohlcv",
                    "h": "",
                    "e": "csv",
                },
                timeout=10,
            )
            response.raise_for_status()

            reader = csv.DictReader(io.StringIO(response.text))
            row = next(reader, None)
            if not row or row.get("Close") in (None, "", "N/D"):
                return f"未找到股票 {symbol} 的信息"

            volume = row.get("Volume") or "N/D"
            if volume != "N/D" and volume.isdigit():
                volume = f"{int(volume):,}"

            return (
                f"{(row.get('Symbol') or stooq_symbol).upper()} 股票信息:\n"
                f"- 日期: {row.get('Date', 'N/D')} {row.get('Time', '')}\n"
                f"- 开盘价: {row.get('Open', 'N/D')}\n"
                f"- 最高价: {row.get('High', 'N/D')}\n"
                f"- 最低价: {row.get('Low', 'N/D')}\n"
                f"- 最新价: {row.get('Close', 'N/D')}\n"
                f"- 成交量: {volume}\n"
                f"- 数据源: Stooq（行情可能延迟）"
            )
        except Exception as e:
            return f"股票查询失败: {str(e)}"
    
    def search_news(self, keyword):
        """通过 GDELT DOC 2.0 搜索通用新闻，无需 API Key。"""
        try:
            keyword = (keyword or "artificial intelligence").strip() or "artificial intelligence"
            query = keyword
            if " " in query and not any(marker in query for marker in ['"', " OR ", "(", ")", ":"]):
                query = f'"{query}"'
            response = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": query,
                    "mode": "artlist",
                    "maxrecords": 5,
                    "format": "json",
                    "timespan": "1week",
                    "sort": "datedesc",
                },
                timeout=20,
            )
            if response.status_code == 429:
                return self._search_news_with_deepseek(keyword)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles") or []
            if not articles:
                return f"暂无关于'{keyword}'的新闻"

            result = f"关于'{keyword}'的最新新闻:\n"
            for index, item in enumerate(articles, 1):
                title = item.get("title") or "无标题"
                url = item.get("url") or ""
                source = item.get("domain") or urlparse(url).netloc or "未知来源"
                created_at = item.get("seendate") or "未知时间"
                language = item.get("language") or "未知语言"
                country = item.get("sourcecountry") or "未知地区"
                result += (
                    f"{index}. {title}\n"
                    f"   来源: {source}\n"
                    f"   时间: {created_at}\n"
                    f"   语言/地区: {language}/{country}\n"
                )
                if url:
                    result += f"   链接: {url}\n"
            result += "数据源: GDELT DOC 2.0（全球新闻，可能存在延迟或限流）"
            return result
        except Exception as e:
            return f"新闻查询失败: {str(e)}"

    def _search_news_with_deepseek(self, keyword):
        """GDELT 限流时使用 DeepSeek API 兜底。"""
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip().strip("'\"")
        if not api_key:
            return "新闻查询失败: GDELT 请求过于频繁，且未配置 DEEPSEEK_API_KEY"

        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("DEEPSEEK_NEWS_MODEL", "deepseek-chat"),
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是新闻检索兜底助手。请用中文回答。"
                                "如果无法访问实时互联网，请明确说明可能不是实时新闻，不能编造具体链接。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"GDELT 新闻接口被限流。请围绕关键词“{keyword}”给出最近新闻方向、"
                                "可能的重要事件和建议继续核验的来源。输出 3 到 5 条，简洁列点。"
                            ),
                        },
                    ],
                    "temperature": 0.2,
                    "max_tokens": 800,
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            content = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            ).strip()
            if not content:
                return "新闻查询失败: DeepSeek API 未返回有效内容"
            return (
                f"GDELT 请求过于频繁，已使用 DeepSeek API 兜底。\n\n"
                f"{content}\n\n"
                "数据源: DeepSeek API 兜底生成（不等同于实时新闻 API，重要信息请二次核验）"
            )
        except Exception as e:
            return f"新闻查询失败: GDELT 请求过于频繁，DeepSeek API 兜底也失败: {e}"
    
    def get_current_time(self, timezone="Asia/Shanghai"):
        """获取当前时间"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"当前时间({timezone}): {current_time}"
        except Exception as e:
            return f"时间查询失败: {str(e)}"
    
    def detect_function_call(self, user_input):
        """检测用户输入是否需要调用函数"""
        function_triggers = {
            "get_weather": ["天气", "气温", "下雨", "下雪", "温度"],
            "get_stock_price": ["股票", "股价", "行情", "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "苹果", "特斯拉", "微软", "英伟达"],
            "search_news": ["新闻", "最新消息", "热点", "时事"],
            "get_current_time": ["时间", "几点", "日期", "现在什么时候"]
        }
        
        for function_name, triggers in function_triggers.items():
            for trigger in triggers:
                if trigger in user_input:
                    return function_name, user_input
        return None, user_input
    
    def _extract_stock_symbol(self, user_input):
        text = user_input or ""
        aliases = {
            "苹果": "AAPL",
            "特斯拉": "TSLA",
            "微软": "MSFT",
            "英伟达": "NVDA",
            "谷歌": "GOOGL",
            "亚马逊": "AMZN",
        }
        for name, symbol in aliases.items():
            if name in text:
                return symbol

        ignored_tokens = {"AI", "API", "PDF", "RAG", "CPU", "GPU"}
        match = re.search(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", text.upper())
        if match and match.group(0) not in ignored_tokens:
            return match.group(0)
        return "AAPL"
    
    def _extract_news_keyword(self, user_input):
        keyword = user_input or ""
        replacements = ["新闻", "最新消息", "热点", "时事", "查一下", "搜索", "查询", "关于", "帮我", "请", "的"]
        for item in replacements:
            keyword = keyword.replace(item, " ")
        keyword = re.sub(r"\s+", " ", keyword).strip(" ?？,，.。")
        keyword_aliases = {
            "人工智能": "artificial intelligence",
            "AI": "artificial intelligence",
            "科技": "technology",
            "财经": "finance",
            "金融": "finance",
            "气候": "climate change",
            "新能源": "renewable energy",
            "美国": "united states",
            "中国": "china",
        }
        return keyword_aliases.get(keyword, keyword or "artificial intelligence")
    
    def process_with_function(self, function_name, user_input):
        """处理函数调用"""
        if function_name in self.available_functions:
            # 简单的参数提取
            if function_name == "get_weather":
                cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
                for city in cities:
                    if city in user_input:
                        return self.available_functions[function_name](city)
                return self.available_functions[function_name]("北京")  # 默认城市
            
            elif function_name == "get_stock_price":
                return self.available_functions[function_name](self._extract_stock_symbol(user_input))
            
            elif function_name == "search_news":
                return self.available_functions[function_name](self._extract_news_keyword(user_input))
            
            else:
                return self.available_functions[function_name]()
        
        return "函数调用失败"
    
    def chat(self, user_input):
        """主要的聊天方法"""
        print(f"用户输入: {user_input}")
        
        # 检测是否需要函数调用
        function_name, processed_input = self.detect_function_call(user_input)
        
        if function_name:
            print(f"检测到函数调用: {function_name}")
            function_result = self.process_with_function(function_name, user_input)
            
            # 将函数结果和用户输入一起发送给模型
            enhanced_prompt = f"""
用户问题: {user_input}

我查询到的实时信息:
{function_result}

请根据以上实时信息回答用户问题:
"""
            response = self.llm.invoke(enhanced_prompt)
            return f"{response}\n\n🔍 信息来源: 实时数据查询"
        
        else:
            # 普通对话模式
            response = self.llm.invoke(user_input)
            return response
