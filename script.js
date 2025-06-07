// Gold Market Documentation Content
const goldMarketContent = [
    {
        id: 'introduction',
        title: 'Introduction to Gold Market Analysis',
        subtitle: 'Understanding the fundamentals of gold trading and market dynamics',
        section: 'Foundations',
        readingTime: '8 min read',
        content: `
            <div class="content">
                <div class="chapter-meta">
                    <span class="chapter-number">Chapter 1</span>
                    <div class="reading-time">
                        <i class="fas fa-clock"></i>
                        <span>8 min read</span>
                    </div>
                </div>

                <h1>Introduction to Gold Market Analysis</h1>
                <p>Gold has been a store of value and medium of exchange for thousands of years. In modern financial markets, gold serves as a hedge against inflation, currency devaluation, and economic uncertainty. This comprehensive guide will walk you through the complete process of analyzing gold market data across multiple timeframes.</p>
                
                <h2>Why Gold Market Analysis Matters</h2>
                <p>The gold market operates 24/7 across global exchanges, generating massive amounts of data every second. Understanding this data and extracting meaningful insights is crucial for:</p>
                
                <ul>
                    <li><strong>Investment Decisions:</strong> Making informed choices about when to buy or sell gold</li>
                    <li><strong>Risk Management:</strong> Understanding market volatility and potential downside risks</li>
                    <li><strong>Portfolio Optimization:</strong> Determining the optimal allocation of gold in investment portfolios</li>
                    <li><strong>Market Timing:</strong> Identifying optimal entry and exit points based on technical and fundamental analysis</li>
                </ul>

                <div class="data-card">
                    <h3>Market Data Timeframes</h3>
                    <p>Throughout this guide, we'll work with data across multiple timeframes:</p>
                    <ul>
                        <li><strong>Ultra High-Frequency:</strong> 1min, 5min data for scalping strategies</li>
                        <li><strong>Intraday:</strong> 15min, 1hr, 4hr data for day trading</li>
                        <li><strong>Swing Trading:</strong> Daily and weekly data for medium-term positions</li>
                        <li><strong>Position Trading:</strong> Monthly and yearly data for long-term investment</li>
                    </ul>
                </div>

                <h2>Market Structure and Participants</h2>
                <p>The gold market consists of several key participants:</p>
                
                <ul>
                    <li><strong>Central Banks:</strong> Major holders and occasional buyers/sellers</li>
                    <li><strong>Institutional Investors:</strong> Hedge funds, pension funds, sovereign wealth funds</li>
                    <li><strong>Retail Investors:</strong> Individual traders and investors</li>
                    <li><strong>Mining Companies:</strong> Primary producers of physical gold</li>
                    <li><strong>Jewelry Industry:</strong> Largest consumer of gold globally</li>
                </ul>

                <h2>Key Factors Affecting Gold Prices</h2>
                <p>Gold prices are influenced by numerous macroeconomic and market factors:</p>

                <div class="info-card">
                    <h3>Economic Indicators</h3>
                    <ul>
                        <li><strong>Inflation Rates:</strong> Higher inflation typically drives gold prices up</li>
                        <li><strong>Interest Rates:</strong> Lower rates make gold more attractive as it doesn't pay interest</li>
                        <li><strong>Currency Strength:</strong> Particularly the US Dollar, which has an inverse relationship with gold</li>
                        <li><strong>Economic Uncertainty:</strong> Geopolitical events, recessions, and market volatility</li>
                    </ul>
                </div>

                <h2>Data Sources and Quality</h2>
                <p>Reliable data is the foundation of any successful analysis. We'll be working with data from:</p>
                
                <ul>
                    <li><strong>COMEX:</strong> Primary gold futures market</li>
                    <li><strong>LBMA:</strong> London Bullion Market Association</li>
                    <li><strong>Spot Markets:</strong> Real-time physical gold pricing</li>
                    <li><strong>ETF Data:</strong> Gold-backed exchange-traded funds</li>
                </ul>

                <h2>What You'll Learn</h2>
                <p>By the end of this guide, you will have:</p>
                
                <ul>
                    <li>Comprehensive understanding of gold market data structures</li>
                    <li>Skills to collect, clean, and preprocess market data</li>
                    <li>Ability to perform technical and fundamental analysis</li>
                    <li>Knowledge of machine learning applications in gold trading</li>
                    <li>Experience building automated trading systems</li>
                    <li>Capability to deploy solutions to cloud infrastructure</li>
                </ul>

                <div class="warning-card">
                    <h3>Risk Disclaimer</h3>
                    <p>Trading gold and other financial instruments involves substantial risk and may not be suitable for all investors. Past performance is not indicative of future results. Always conduct your own research and consider seeking advice from qualified financial professionals.</p>
                </div>
            </div>
        `
    },
    {
        id: 'market-data-overview',
        title: 'Market Data Overview',
        subtitle: 'Understanding different timeframes and data structures',
        section: 'Foundations',
        readingTime: '12 min read',
        content: `
            <div class="content">
                <div class="chapter-meta">
                    <span class="chapter-number">Chapter 2</span>
                    <div class="reading-time">
                        <i class="fas fa-clock"></i>
                        <span>12 min read</span>
                    </div>
                </div>

                <h1>Market Data Overview</h1>
                <p>Gold market data comes in various formats and timeframes, each serving different analytical purposes. Understanding the structure and characteristics of each timeframe is essential for effective analysis and strategy development.</p>

                <h2>Timeframe Classifications</h2>
                
                <h3>High-Frequency Data (1min - 5min)</h3>
                <p>Ultra-short timeframes used primarily for algorithmic trading and scalping strategies:</p>
                
                <div class="data-card">
                    <h4>1-Minute Data Characteristics</h4>
                    <ul>
                        <li><strong>Volume:</strong> ~525,600 candles per year</li>
                        <li><strong>Noise Level:</strong> Very high, requires significant filtering</li>
                        <li><strong>Use Cases:</strong> Scalping, market microstructure analysis</li>
                        <li><strong>Storage Requirements:</strong> ~50MB per year (OHLCV format)</li>
                    </ul>
                </div>

                <pre data-language="python"><code># Sample 1-minute data structure
{
    "timestamp": "2024-01-15T10:30:00Z",
    "open": 2034.50,
    "high": 2035.20,
    "low": 2033.80,
    "close": 2034.90,
    "volume": 1250,
    "spread": 0.40,
    "tick_count": 45
}</code></pre>

                <h3>Intraday Data (15min - 4hr)</h3>
                <p>Medium-frequency data ideal for day trading and short-term analysis:</p>

                <div class="info-card">
                    <h4>15-Minute Data Benefits</h4>
                    <ul>
                        <li>Balanced between noise and signal quality</li>
                        <li>Suitable for most technical indicators</li>
                        <li>Manageable data size for backtesting</li>
                        <li>Captures intraday price movements effectively</li>
                    </ul>
                </div>

                <h3>Daily and Weekly Data</h3>
                <p>Standard timeframes for swing trading and medium-term analysis:</p>

                <table>
                    <thead>
                        <tr>
                            <th>Timeframe</th>
                            <th>Candles/Year</th>
                            <th>Best For</th>
                            <th>Typical Strategy</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Daily</td>
                            <td>~365</td>
                            <td>Swing Trading</td>
                            <td>3-30 day holds</td>
                        </tr>
                        <tr>
                            <td>Weekly</td>
                            <td>~52</td>
                            <td>Position Trading</td>
                            <td>1-6 month holds</td>
                        </tr>
                        <tr>
                            <td>Monthly</td>
                            <td>12</td>
                            <td>Long-term Investment</td>
                            <td>6+ month holds</td>
                        </tr>
                    </tbody>
                </table>

                <h2>Data Quality Considerations</h2>
                
                <h3>Missing Data Handling</h3>
                <p>Market data often contains gaps due to:</p>
                <ul>
                    <li>Exchange closures (weekends, holidays)</li>
                    <li>Technical outages</li>
                    <li>Low liquidity periods</li>
                    <li>Data feed interruptions</li>
                </ul>

                <pre data-language="python"><code># Common data quality checks
def validate_market_data(df):
    # Check for missing values
    missing_data = df.isnull().sum()
    
    # Validate price relationships
    invalid_prices = df[df['high'] < df['low']]
    
    # Check for outliers (> 3 standard deviations)
    price_outliers = df[
        (df['close'] - df['close'].mean()).abs() > 
        3 * df['close'].std()
    ]
    
    # Validate volume data
    negative_volume = df[df['volume'] < 0]
    
    return {
        'missing_data': missing_data,
        'invalid_prices': len(invalid_prices),
        'outliers': len(price_outliers),
        'negative_volume': len(negative_volume)
    }</code></pre>

                <h2>Data Storage Formats</h2>
                
                <h3>CSV Format</h3>
                <p>Standard format for historical data storage:</p>
                
                <pre data-language="csv"><code>timestamp,open,high,low,close,volume
2024-01-15 09:00:00,2034.50,2036.20,2033.80,2035.90,2580
2024-01-15 09:01:00,2035.90,2037.10,2035.40,2036.50,1950
2024-01-15 09:02:00,2036.50,2036.80,2035.20,2035.60,1740</code></pre>

                <h3>Parquet Format</h3>
                <p>Optimized columnar storage for large datasets:</p>
                
                <div class="success-card">
                    <h4>Parquet Advantages</h4>
                    <ul>
                        <li><strong>Compression:</strong> 80-90% size reduction vs CSV</li>
                        <li><strong>Performance:</strong> Faster read/write operations</li>
                        <li><strong>Schema:</strong> Built-in data type definitions</li>
                        <li><strong>Compatibility:</strong> Works with Pandas, Spark, BigQuery</li>
                    </ul>
                </div>

                <h2>Real-Time Data Streams</h2>
                <p>For live trading applications, you'll need real-time data feeds:</p>

                <pre data-language="python"><code># WebSocket connection for real-time data
import websocket
import json

class GoldDataStream:
    def __init__(self, api_key):
        self.api_key = api_key
        self.ws = None
        
    def on_message(self, ws, message):
        data = json.loads(message)
        if data['type'] == 'tick':
            self.process_tick(data)
            
    def process_tick(self, tick_data):
        # Process real-time tick data
        price = tick_data['price']
        timestamp = tick_data['timestamp']
        volume = tick_data['volume']
        
        # Update indicators, check signals
        self.update_indicators(price, timestamp, volume)
        
    def connect(self):
        ws_url = f"wss://api.goldmarket.com/stream?key={self.api_key}"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()</code></pre>

                <h2>Data Synchronization</h2>
                <p>When working with multiple timeframes, synchronization is crucial:</p>

                <div class="warning-card">
                    <h3>Timezone Considerations</h3>
                    <p>Gold trades across multiple global markets. Always use UTC timestamps and be aware of daylight saving time changes. Major trading sessions overlap in London and New York, creating higher volatility periods.</p>
                </div>

                <h2>Market Sessions</h2>
                <p>Understanding global trading sessions helps in data analysis:</p>

                <ul>
                    <li><strong>Asian Session:</strong> 23:00 - 08:00 UTC (Lower volatility)</li>
                    <li><strong>London Session:</strong> 08:00 - 17:00 UTC (High volatility)</li>
                    <li><strong>New York Session:</strong> 13:00 - 22:00 UTC (Highest volatility)</li>
                    <li><strong>Overlap Periods:</strong> 08:00-10:00 and 13:00-17:00 UTC (Maximum activity)</li>
                </ul>

                <h2>Next Steps</h2>
                <p>Now that you understand the different types of market data available, we'll move on to setting up data collection systems and understanding the metadata that accompanies market data feeds.</p>
            </div>
        `
    },
    {
        id: 'data-collection',
        title: 'Data Collection & Sources',
        subtitle: 'Setting up reliable data pipelines and API integrations',
        section: 'Data Infrastructure',
        readingTime: '15 min read',
        content: `
            <div class="content">
                <div class="chapter-meta">
                    <span class="chapter-number">Chapter 3</span>
                    <div class="reading-time">
                        <i class="fas fa-clock"></i>
                        <span>15 min read</span>
                    </div>
                </div>

                <h1>Data Collection & Sources</h1>
                <p>Reliable data collection is the foundation of any successful gold market analysis system. This chapter covers various data sources, API integrations, and best practices for building robust data pipelines.</p>

                <h2>Primary Data Sources</h2>
                
                <h3>Professional Data Providers</h3>
                <p>High-quality, institutional-grade data sources:</p>

                <div class="data-card">
                    <h4>Bloomberg Terminal API</h4>
                    <ul>
                        <li><strong>Coverage:</strong> Real-time and historical data</li>
                        <li><strong>Cost:</strong> $2,000+ per month</li>
                        <li><strong>Quality:</strong> Institutional grade</li>
                        <li><strong>Latency:</strong> <1ms for real-time data</li>
                    </ul>
                </div>

                <div class="info-card">
                    <h4>Refinitiv Eikon (formerly Thomson Reuters)</h4>
                    <ul>
                        <li>Comprehensive market data and news</li>
                        <li>Real-time streaming capabilities</li>
                        <li>Advanced analytics and charting tools</li>
                        <li>Integration with Excel and Python</li>
                    </ul>
                </div>

                <h3>Free and Low-Cost Sources</h3>
                <p>Budget-friendly options for individual traders and researchers:</p>

                <table>
                    <thead>
                        <tr>
                            <th>Source</th>
                            <th>Cost</th>
                            <th>Real-time</th>
                            <th>Historical</th>
                            <th>API Limits</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Alpha Vantage</td>
                            <td>Free/Paid</td>
                            <td>15min delay</td>
                            <td>20+ years</td>
                            <td>5 calls/min</td>
                        </tr>
                        <tr>
                            <td>Yahoo Finance</td>
                            <td>Free</td>
                            <td>15min delay</td>
                            <td>5+ years</td>
                            <td>Unofficial limits</td>
                        </tr>
                        <tr>
                            <td>Quandl</td>
                            <td>Free/Paid</td>
                            <td>No</td>
                            <td>30+ years</td>
                            <td>50 calls/day</td>
                        </tr>
                        <tr>
                            <td>MetalAPI</td>
                            <td>Free/Paid</td>
                            <td>1min delay</td>
                            <td>10+ years</td>
                            <td>100 calls/month</td>
                        </tr>
                    </tbody>
                </table>

                <h2>API Integration Examples</h2>
                
                <h3>Alpha Vantage Integration</h3>
                <p>Complete example for fetching gold price data:</p>

                <pre data-language="python"><code>import requests
import pandas as pd
import time
from datetime import datetime, timedelta

class AlphaVantageCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_gold_prices(self, interval='1min', outputsize='compact'):
        """
        Fetch gold price data from Alpha Vantage
        
        Args:
            interval: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
            outputsize: compact (last 100 data points) or full (20+ years)
        """
        params = {
            'function': 'FX_INTRADAY' if 'min' in interval else 'FX_DAILY',
            'from_symbol': 'XAU',
            'to_symbol': 'USD',
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Handle rate limiting
            if 'Note' in data:
                print("Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                return self.get_gold_prices(interval, outputsize)
                
            return self._parse_response(data, interval)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
            
    def _parse_response(self, data, interval):
        """Parse API response into pandas DataFrame"""
        time_series_key = f"Time Series FX ({interval})"
        if 'daily' in interval.lower():
            time_series_key = "Time Series FX (Daily)"
            
        if time_series_key not in data:
            print(f"Error: {data}")
            return None
            
        df = pd.DataFrame.from_dict(
            data[time_series_key], 
            orient='index'
        )
        
        # Rename columns
        df.columns = ['open', 'high', 'low', 'close']
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df.sort_index()
        
        return df

# Usage example
collector = AlphaVantageCollector('YOUR_API_KEY')
gold_data = collector.get_gold_prices(interval='15min', outputsize='full')</code></pre>

                <h3>WebSocket Real-Time Data</h3>
                <p>For live trading applications, implement WebSocket connections:</p>

                <pre data-language="python"><code>import websocket
import json
import threading
from queue import Queue

class RealTimeGoldFeed:
    def __init__(self, api_key):
        self.api_key = api_key
        self.data_queue = Queue()
        self.is_connected = False
        
    def on_open(self, ws):
        print("WebSocket connection opened")
        self.is_connected = True
        
        # Subscribe to gold price updates
        subscribe_msg = {
            "action": "subscribe",
            "symbols": ["XAUUSD"],
            "api_key": self.api_key
        }
        ws.send(json.dumps(subscribe_msg))
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            if data.get('type') == 'price_update':
                # Process price update
                price_data = {
                    'symbol': data['symbol'],
                    'price': float(data['price']),
                    'timestamp': pd.Timestamp(data['timestamp']),
                    'volume': data.get('volume', 0),
                    'bid': float(data.get('bid', 0)),
                    'ask': float(data.get('ask', 0))
                }
                
                self.data_queue.put(price_data)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing message: {e}")
            
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.is_connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.is_connected = False
        
    def start_stream(self):
        ws_url = "wss://api.metalsapi.com/v1/stream"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run WebSocket in separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
    def get_latest_data(self):
        """Get latest price data from queue"""
        if not self.data_queue.empty():
            return self.data_queue.get()
        return None</code></pre>

                <h2>Data Storage Architecture</h2>
                
                <h3>Time Series Database Setup</h3>
                <p>Efficient storage for high-frequency financial data:</p>

                <div class="success-card">
                    <h4>InfluxDB for Time Series</h4>
                    <ul>
                        <li><strong>Performance:</strong> Optimized for time-stamped data</li>
                        <li><strong>Compression:</strong> 90%+ compression ratio</li>
                        <li><strong>Queries:</strong> SQL-like query language</li>
                        <li><strong>Retention:</strong> Automatic data retention policies</li>
                    </ul>
                </div>

                <pre data-language="python"><code>from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

class GoldDataStorage:
    def __init__(self, url, token, org, bucket):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org
        
    def store_price_data(self, timestamp, price_data):
        """Store OHLCV data in InfluxDB"""
        point = (
            Point("gold_prices")
            .tag("symbol", "XAUUSD")
            .tag("timeframe", price_data.get('timeframe', '1m'))
            .field("open", price_data['open'])
            .field("high", price_data['high'])
            .field("low", price_data['low'])
            .field("close", price_data['close'])
            .field("volume", price_data['volume'])
            .time(timestamp)
        )
        
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        
    def query_price_data(self, start_time, end_time, timeframe='1m'):
        """Query historical price data"""
        query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_time}, stop: {end_time})
                |> filter(fn: (r) => r._measurement == "gold_prices")
                |> filter(fn: (r) => r.timeframe == "{timeframe}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        result = self.query_api.query(org=self.org, query=query)
        
        # Convert to pandas DataFrame
        data = []
        for table in result:
            for record in table.records:
                data.append({
                    'timestamp': record.get_time(),
                    'open': record.values.get('open'),
                    'high': record.values.get('high'),
                    'low': record.values.get('low'),
                    'close': record.values.get('close'),
                    'volume': record.values.get('volume')
                })
                
        return pd.DataFrame(data)</code></pre>

                <h2>Data Quality Monitoring</h2>
                
                <h3>Automated Quality Checks</h3>
                <p>Implement continuous monitoring to ensure data integrity:</p>

                <pre data-language="python"><code>class DataQualityMonitor:
    def __init__(self, storage_client):
        self.storage = storage_client
        self.quality_metrics = {}
        
    def run_quality_checks(self, data_df):
        """Comprehensive data quality assessment"""
        checks = {}
        
        # 1. Completeness Check
        checks['missing_data_pct'] = (data_df.isnull().sum().sum() / 
                                     (len(data_df) * len(data_df.columns))) * 100
        
        # 2. Validity Check - Price relationships
        invalid_ohlc = data_df[
            (data_df['high'] < data_df['low']) |
            (data_df['high'] < data_df['open']) |
            (data_df['high'] < data_df['close']) |
            (data_df['low'] > data_df['open']) |
            (data_df['low'] > data_df['close'])
        ]
        checks['invalid_ohlc_count'] = len(invalid_ohlc)
        
        # 3. Outlier Detection
        for col in ['open', 'high', 'low', 'close']:
            z_scores = np.abs(stats.zscore(data_df[col].dropna()))
            outliers = data_df[z_scores > 3]
            checks[f'{col}_outliers'] = len(outliers)
            
        # 4. Data Gaps Detection
        time_diff = data_df.index.to_series().diff()
        expected_interval = time_diff.mode()[0]
        gaps = time_diff[time_diff > expected_interval * 2]
        checks['time_gaps'] = len(gaps)
        
        # 5. Volume Validation
        checks['zero_volume_count'] = len(data_df[data_df['volume'] == 0])
        checks['negative_volume_count'] = len(data_df[data_df['volume'] < 0])
        
        return checks
        
    def alert_on_quality_issues(self, quality_results):
        """Send alerts for data quality issues"""
        critical_issues = []
        
        if quality_results['missing_data_pct'] > 5:
            critical_issues.append(f"High missing data: {quality_results['missing_data_pct']:.2f}%")
            
        if quality_results['invalid_ohlc_count'] > 0:
            critical_issues.append(f"Invalid OHLC data: {quality_results['invalid_ohlc_count']} records")
            
        if quality_results['time_gaps'] > 10:
            critical_issues.append(f"Multiple time gaps detected: {quality_results['time_gaps']}")
            
        if critical_issues:
            self.send_alert(critical_issues)
            
    def send_alert(self, issues):
        """Send notification about data quality issues"""
        # Implement your alerting mechanism here
        # Could be email, Slack, SMS, etc.
        print(f"DATA QUALITY ALERT: {', '.join(issues)}")</code></pre>

                <h2>Rate Limiting and Error Handling</h2>
                
                <div class="warning-card">
                    <h3>API Rate Limiting Best Practices</h3>
                    <ul>
                        <li>Implement exponential backoff for retries</li>
                        <li>Cache frequently requested data</li>
                        <li>Use batch requests when available</li>
                        <li>Monitor your API usage quotas</li>
                        <li>Have backup data sources ready</li>
                    </ul>
                </div>

                <pre data-language="python"><code>import time
import random
from functools import wraps

def rate_limited_retry(max_retries=3, backoff_factor=1):
    """Decorator for handling rate-limited API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limited
                        wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limited. Waiting {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise e
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = backoff_factor * (2 ** attempt)
                    print(f"Request failed. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

# Usage
@rate_limited_retry(max_retries=5, backoff_factor=2)
def fetch_gold_data_with_retry(api_client, symbol, timeframe):
    return api_client.get_data(symbol, timeframe)</code></pre>

                <h2>Data Backup and Recovery</h2>
                <p>Ensure business continuity with robust backup strategies:</p>

                <ul>
                    <li><strong>Automated Daily Backups:</strong> Store to cloud storage (S3, Google Cloud)</li>
                    <li><strong>Redundant Sources:</strong> Multiple data providers for critical feeds</li>
                    <li><strong>Local Caching:</strong> Cache recent data for offline analysis</li>
                    <li><strong>Disaster Recovery:</strong> Document recovery procedures and test regularly</li>
                </ul>

                <h2>Next Steps</h2>
                <p>With reliable data collection in place, we'll next explore the metadata that accompanies market data and how to interpret and utilize this additional information for better analysis.</p>
            </div>
        `
    },
    {
        id: 'metadata-understanding',
        title: 'Understanding Metadata',
        subtitle: 'Extracting insights from data about data',
        section: 'Data Infrastructure',
        readingTime: '10 min read',
        content: `
            <div class="content">
                <div class="chapter-meta">
                    <span class="chapter-number">Chapter 4</span>
                    <div class="reading-time">
                        <i class="fas fa-clock"></i>
                        <span>10 min read</span>
                    </div>
                </div>

                <h1>Understanding Metadata</h1>
                <p>Metadata - data about data - provides crucial context for market analysis. This chapter explores how to extract, interpret, and utilize metadata to enhance your gold market analysis capabilities.</p>

                <h2>Types of Market Metadata</h2>
                
                <h3>Temporal Metadata</h3>
                <p>Time-related information that provides context about when and how data was collected:</p>

                <div class="data-card">
                    <h4>Timestamp Information</h4>
                    <ul>
                        <li><strong>Collection Time:</strong> When data was recorded by the exchange</li>
                        <li><strong>Processing Time:</strong> When data was processed by the data provider</li>
                        <li><strong>Transmission Time:</strong> When data was sent to your system</li>
                        <li><strong>Reception Time:</strong> When your system received the data</li>
                    </ul>
                </div>

                <pre data-language="python"><code># Example metadata structure
metadata_example = {
    "symbol": "XAUUSD",
    "timeframe": "1m",
    "exchange": "COMEX",
    "collection_timestamp": "2024-01-15T10:30:00.000Z",
    "processing_timestamp": "2024-01-15T10:30:00.125Z",
    "transmission_timestamp": "2024-01-15T10:30:00.250Z",
    "reception_timestamp": "2024-01-15T10:30:00.300Z",
    "latency_ms": 300,
    "data_quality": {
        "completeness": 1.0,
        "accuracy": 0.99,
        "freshness": "real-time"
    },
    "source_details": {
        "provider": "AlphaVantage",
        "api_version": "v1.2",
        "rate_limit_remaining": 45,
        "data_tier": "premium"
    }
}</code></pre>

                <h2>Exchange-Specific Metadata</h2>
                
                <h3>COMEX Gold Futures</h3>
                <p>The Chicago Mercantile Exchange provides extensive metadata:</p>

                <table>
                    <thead>
                        <tr>
                            <th>Field</th>
                            <th>Description</th>
                            <th>Example</th>
                            <th>Usage</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Contract Code</td>
                            <td>Specific futures contract</td>
                            <td>GCZ24</td>
                            <td>Contract identification</td>
                        </tr>
                        <tr>
                            <td>Open Interest</td>
                            <td>Total open contracts</td>
                            <td>245,678</td>
                            <td>Market participation gauge</td>
                        </tr>
                        <tr>
                            <td>Settlement Price</td>
                            <td>Official closing price</td>
                            <td>$2,034.50</td>
                            <td>Daily P&L calculations</td>
                        </tr>
                        <tr>
                            <td>Trading Session</td>
                            <td>Market session identifier</td>
                            <td>RTH, ETH</td>
                            <td>Volume analysis</td>
                        </tr>
                    </tbody>
                </table>

                <h3>Spot Gold Metadata</h3>
                <p>Physical gold markets provide different metadata:</p>

                <pre data-language="python"><code># Spot gold metadata structure
spot_metadata = {
    "market": "LBMA",
    "fixing_time": "15:00 GMT",
    "participant_count": 12,
    "fix_type": "PM_FIX",
    "currency": "USD",
    "unit": "troy_ounce",
    "purity": "99.5%",
    "delivery_location": "London",
    "settlement_period": "T+2",
    "bid_ask_spread": 0.50,
    "market_makers": [
        "HSBC", "JP Morgan", "UBS", "Goldman Sachs"
    ]
}</code></pre>

                <h2>Data Quality Metadata</h2>
                
                <h3>Completeness Metrics</h3>
                <p>Understanding data coverage and gaps:</p>

                <div class="info-card">
                    <h4>Completeness Indicators</h4>
                    <ul>
                        <li><strong>Coverage Ratio:</strong> Percentage of expected data points received</li>
                        <li><strong>Gap Duration:</strong> Length of missing data periods</li>
                        <li><strong>Update Frequency:</strong> How often data is refreshed</li>
                        <li><strong>Staleness:</strong> Age of the most recent data point</li>
                    </ul>
                </div>

                <pre data-language="python"><code>class MetadataAnalyzer:
    def __init__(self):
        self.quality_thresholds = {
            'completeness_min': 0.95,
            'staleness_max_minutes': 5,
            'latency_max_ms': 1000
        }
    
    def analyze_data_quality(self, df, metadata):
        """Analyze data quality using metadata"""
        quality_report = {}
        
        # Calculate completeness
        expected_points = self._calculate_expected_points(
            df.index[0], df.index[-1], metadata['timeframe']
        )
        actual_points = len(df)
        quality_report['completeness'] = actual_points / expected_points
        
        # Calculate average latency
        if 'latency_ms' in metadata:
            quality_report['avg_latency_ms'] = metadata['latency_ms']
        
        # Check for data staleness
        latest_timestamp = df.index[-1]
        current_time = pd.Timestamp.now(tz='UTC')
        staleness_minutes = (current_time - latest_timestamp).total_seconds() / 60
        quality_report['staleness_minutes'] = staleness_minutes
        
        # Identify data gaps
        time_diffs = df.index.to_series().diff()
        expected_interval = self._get_expected_interval(metadata['timeframe'])
        gaps = time_diffs[time_diffs > expected_interval * 2]
        quality_report['gap_count'] = len(gaps)
        
        # Overall quality score
        quality_report['quality_score'] = self._calculate_quality_score(quality_report)
        
        return quality_report
    
    def _calculate_quality_score(self, report):
        """Calculate overall quality score (0-1)"""
        scores = []
        
        # Completeness score
        completeness_score = min(report['completeness'], 1.0)
        scores.append(completeness_score * 0.4)  # 40% weight
        
        # Latency score (lower is better)
        if 'avg_latency_ms' in report:
            latency_score = max(0, 1 - (report['avg_latency_ms'] / 1000))
            scores.append(latency_score * 0.3)  # 30% weight
        
        # Staleness score (lower is better)
        staleness_score = max(0, 1 - (report['staleness_minutes'] / 60))
        scores.append(staleness_score * 0.3)  # 30% weight
        
        return sum(scores)</code></pre>

                <h2>API Response Metadata</h2>
                
                <h3>Rate Limiting Information</h3>
                <p>Monitor and manage API usage:</p>

                <pre data-language="python"><code>class APIMetadataTracker:
    def __init__(self):
        self.api_stats = {}
        
    def extract_rate_limit_info(self, response_headers):
        """Extract rate limiting info from API response headers"""
        rate_info = {}
        
        # Common rate limit headers
        header_mappings = {
            'x-ratelimit-limit': 'limit_per_period',
            'x-ratelimit-remaining': 'remaining_calls',
            'x-ratelimit-reset': 'reset_timestamp',
            'x-ratelimit-used': 'used_calls',
            'retry-after': 'retry_after_seconds'
        }
        
        for header, key in header_mappings.items():
            if header in response_headers:
                rate_info[key] = response_headers[header]
                
        return rate_info
    
    def update_api_stats(self, provider, rate_info):
        """Update API usage statistics"""
        if provider not in self.api_stats:
            self.api_stats[provider] = {
                'total_calls': 0,
                'successful_calls': 0,
                'rate_limited_calls': 0,
                'last_reset': None
            }
        
        self.api_stats[provider]['total_calls'] += 1
        
        if rate_info.get('remaining_calls'):
            remaining = int(rate_info['remaining_calls'])
            if remaining < 10:  # Warning threshold
                print(f"WARNING: Only {remaining} API calls remaining for {provider}")
                
    def should_throttle_requests(self, provider):
        """Determine if requests should be throttled"""
        if provider in self.api_stats:
            stats = self.api_stats[provider]
            rate_limited_ratio = stats['rate_limited_calls'] / max(stats['total_calls'], 1)
            return rate_limited_ratio > 0.1  # Throttle if >10% of calls are rate limited
        return False</code></pre>

                <h2>Market Microstructure Metadata</h2>
                
                <h3>Order Book Information</h3>
                <p>Level 2 market data provides deeper insights:</p>

                <div class="success-card">
                    <h4>Order Book Metadata</h4>
                    <ul>
                        <li><strong>Bid-Ask Spread:</strong> Liquidity indicator</li>
                        <li><strong>Market Depth:</strong> Volume at different price levels</li>
                        <li><strong>Order Count:</strong> Number of orders at each level</li>
                        <li><strong>Market Maker IDs:</strong> Who's providing liquidity</li>
                    </ul>
                </div>

                <pre data-language="python"><code># Order book metadata example
order_book_metadata = {
    "timestamp": "2024-01-15T10:30:00.000Z",
    "symbol": "XAUUSD",
    "bid_ask_spread": 0.50,
    "total_bid_volume": 1250.5,
    "total_ask_volume": 980.2,
    "level_count": 10,
    "market_makers": {
        "MM001": {"bid_volume": 125.5, "ask_volume": 98.2},
        "MM002": {"bid_volume": 220.0, "ask_volume": 156.8}
    },
    "liquidity_metrics": {
        "effective_spread": 0.48,
        "impact_cost_100oz": 0.25,
        "resilience_score": 0.85
    }
}</code></pre>

                <h2>Regulatory and Compliance Metadata</h2>
                
                <h3>MiFID II Requirements</h3>
                <p>European regulations require specific metadata tracking:</p>

                <ul>
                    <li><strong>Systematic Internalizer (SI) flag:</strong> Trade execution venue</li>
                    <li><strong>Best Execution indicators:</strong> Price improvement metrics</li>
                    <li><strong>Transaction Reporting:</strong> Regulatory reporting requirements</li>
                    <li><strong>Market Data Fees:</strong> Cost attribution for data usage</li>
                </ul>

                <h2>Using Metadata for Analysis</h2>
                
                <h3>Quality-Weighted Analysis</h3>
                <p>Incorporate data quality into your analysis:</p>

                <pre data-language="python"><code>def quality_weighted_analysis(price_data, quality_metadata):
    """
    Perform analysis weighted by data quality scores
    """
    # Calculate quality weights
    quality_scores = [metadata.get('quality_score', 1.0) 
                     for metadata in quality_metadata]
    
    # Apply quality weights to price data
    weighted_prices = price_data['close'] * quality_scores
    
    # Calculate quality-weighted moving average
    quality_weighted_ma = weighted_prices.rolling(20).sum() / \
                         pd.Series(quality_scores).rolling(20).sum()
    
    # Quality-adjusted volatility calculation
    quality_mask = pd.Series(quality_scores) > 0.8  # High quality data only
    high_quality_returns = price_data['close'].pct_change()[quality_mask]
    quality_adjusted_volatility = high_quality_returns.std() * np.sqrt(252)
    
    return {
        'quality_weighted_ma': quality_weighted_ma,
        'quality_adjusted_volatility': quality_adjusted_volatility,
        'avg_quality_score': np.mean(quality_scores)
    }</code></pre>

                <h2>Metadata Storage and Retrieval</h2>
                
                <div class="warning-card">
                    <h3>Storage Considerations</h3>
                    <p>Metadata can grow large quickly. Consider separate storage for metadata vs. price data. Use document databases (MongoDB) for flexible metadata schemas, or dedicated columns in time-series databases.</p>
                </div>

                <pre data-language="python"><code># MongoDB metadata storage example
from pymongo import MongoClient

class MetadataStore:
    def __init__(self, connection_string, database_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.metadata_collection = self.db.metadata
        
    def store_metadata(self, symbol, timeframe, timestamp, metadata):
        """Store metadata document"""
        document = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': timestamp,
            'metadata': metadata,
            'created_at': pd.Timestamp.now()
        }
        
        self.metadata_collection.insert_one(document)
        
    def query_metadata(self, symbol, start_date, end_date):
        """Query metadata for analysis"""
        query = {
            'symbol': symbol,
            'timestamp': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        
        return list(self.metadata_collection.find(query))</code></pre>

                <h2>Next Steps</h2>
                <p>Understanding metadata is crucial for building robust analysis systems. Next, we'll dive into data preprocessing techniques that will clean and prepare your gold market data for analysis, taking into account the quality indicators we've learned about.</p>
            </div>
        `
    },
    {
        id: 'preprocessing',
        title: 'Data Preprocessing Pipeline',
        subtitle: 'Cleaning and preparing data for analysis',
        section: 'Data Processing',
        readingTime: '18 min read',
        content: `
            <div class="content">
                <div class="chapter-meta">
                    <span class="chapter-number">Chapter 5</span>
                    <div class="reading-time">
                        <i class="fas fa-clock"></i>
                        <span>18 min read</span>
                    </div>
                </div>

                <h1>Data Preprocessing Pipeline</h1>
                <p>Raw market data requires extensive preprocessing before analysis. This chapter covers comprehensive data cleaning, validation, normalization, and transformation techniques specifically designed for gold market data.</p>

                <h2>Preprocessing Pipeline Overview</h2>
                
                <div class="data-card">
                    <h3>Pipeline Stages</h3>
                    <ol>
                        <li><strong>Data Validation:</strong> Check for format consistency and basic errors</li>
                        <li><strong>Outlier Detection:</strong> Identify and handle anomalous price movements</li>
                        <li><strong>Gap Filling:</strong> Handle missing data points intelligently</li>
                        <li><strong>Normalization:</strong> Standardize price formats and time zones</li>
                        <li><strong>Feature Engineering:</strong> Create derived indicators and metrics</li>
                        <li><strong>Quality Assessment:</strong> Final validation and scoring</li>
                    </ol>
                </div>

                <h2>Data Validation and Cleaning</h2>
                
                <h3>OHLCV Validation</h3>
                <p>Ensure price data follows logical relationships:</p>

                <pre data-language="python"><code>import pandas as pd
import numpy as np
from scipy import stats
import warnings

class GoldDataPreprocessor:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        
    def _default_config(self):
        return {
            'outlier_threshold': 3.0,  # Z-score threshold
            'max_gap_minutes': 60,     # Maximum gap to fill
            'min_volume': 0,           # Minimum valid volume
            'price_change_limit': 0.1,  # 10% max price change per period
            'business_hours_only': False
        }
    
    def validate_ohlcv_data(self, df):
        """
        Comprehensive OHLCV validation
        """
        validation_results = {
            'total_records': len(df),
            'invalid_records': 0,
            'issues': []
        }
        
        # Create boolean masks for different validation rules
        masks = {}
        
        # 1. High >= Low
        masks['high_low'] = df['high'] >= df['low']
        
        # 2. High >= Open and High >= Close
        masks['high_bounds'] = (df['high'] >= df['open']) & (df['high'] >= df['close'])
        
        # 3. Low <= Open and Low <= Close
        masks['low_bounds'] = (df['low'] <= df['open']) & (df['low'] <= df['close'])
        
        # 4. Volume >= 0
        masks['volume_positive'] = df['volume'] >= 0
        
        # 5. No null values in critical columns
        masks['no_nulls'] = df[['open', 'high', 'low', 'close']].notna().all(axis=1)
        
        # 6. Reasonable price ranges (no negative prices)
        masks['positive_prices'] = (df[['open', 'high', 'low', 'close']] > 0).all(axis=1)
        
        # Combine all masks
        all_valid = pd.Series(True, index=df.index)
        for name, mask in masks.items():
            invalid_count = (~mask).sum()
            if invalid_count > 0:
                validation_results['issues'].append(f"{name}: {invalid_count} invalid records")
                all_valid &= mask
        
        validation_results['invalid_records'] = (~all_valid).sum()
        validation_results['clean_data'] = df[all_valid].copy()
        
        return validation_results
    
    def detect_price_outliers(self, df, method='zscore'):
        """
        Detect price outliers using various methods
        """
        outliers = pd.DataFrame(index=df.index)
        
        if method == 'zscore':
            for col in ['open', 'high', 'low', 'close']:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers[f'{col}_outlier'] = z_scores > self.config['outlier_threshold']
                
        elif method == 'iqr':
            for col in ['open', 'high', 'low', 'close']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
        elif method == 'price_change':
            # Detect sudden price changes
            for col in ['open', 'high', 'low', 'close']:
                pct_change = df[col].pct_change().abs()
                outliers[f'{col}_outlier'] = pct_change > self.config['price_change_limit']
        
        # Combine outlier indicators
        outliers['any_outlier'] = outliers.any(axis=1)
        
        return outliers</code></pre>

                <h2>Missing Data Handling</h2>
                
                <h3>Intelligent Gap Filling</h3>
                <p>Different strategies for different types of missing data:</p>

                <pre data-language="python"><code>    def handle_missing_data(self, df):
        """
        Handle missing data with multiple strategies
        """
        original_length = len(df)
        
        # 1. Identify gaps in time series
        gaps = self._identify_time_gaps(df)
        
        # 2. Fill small gaps (< max_gap_minutes)
        df_filled = self._fill_small_gaps(df, gaps)
        
        # 3. Handle larger gaps with forward fill or interpolation
        df_filled = self._handle_large_gaps(df_filled, gaps)
        
        # 4. Remove periods with too much missing data
        df_cleaned = self._remove_poor_quality_periods(df_filled)
        
        fill_report = {
            'original_length': original_length,
            'final_length': len(df_cleaned),
            'gaps_filled': len(gaps['small_gaps']),
            'large_gaps_handled': len(gaps['large_gaps']),
            'records_removed': original_length - len(df_cleaned)
        }
        
        return df_cleaned, fill_report
    
    def _identify_time_gaps(self, df):
        """Identify different types of time gaps"""
        time_diffs = df.index.to_series().diff()
        
        # Determine expected interval
        mode_interval = time_diffs.mode()[0]
        
        # Small gaps (fillable)
        small_gap_mask = (time_diffs > mode_interval) & \
                        (time_diffs <= pd.Timedelta(minutes=self.config['max_gap_minutes']))
        
        # Large gaps (may need special handling)
        large_gap_mask = time_diffs > pd.Timedelta(minutes=self.config['max_gap_minutes'])
        
        return {
            'small_gaps': df.index[small_gap_mask].tolist(),
            'large_gaps': df.index[large_gap_mask].tolist(),
            'expected_interval': mode_interval
        }
    
    def _fill_small_gaps(self, df, gaps):
        """Fill small gaps using interpolation"""
        if not gaps['small_gaps']:
            return df
            
        # Create complete time index
        full_index = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq=gaps['expected_interval']
        )
        
        # Reindex and interpolate
        df_reindexed = df.reindex(full_index)
        
        # Use different interpolation methods for different columns
        df_reindexed[['open', 'high', 'low', 'close']] = \
            df_reindexed[['open', 'high', 'low', 'close']].interpolate(method='time')
        
        # For volume, use forward fill or zero
        df_reindexed['volume'] = df_reindexed['volume'].fillna(method='ffill').fillna(0)
        
        return df_reindexed
    
    def _handle_large_gaps(self, df, gaps):
        """Handle large gaps with market session awareness"""
        for gap_start in gaps['large_gaps']:
            gap_end_idx = df.index.get_loc(gap_start)
            
            # Check if gap spans market close (weekend/holiday)
            if self._is_market_closure_gap(gap_start, df.index[gap_end_idx-1]):
                # For market closure gaps, forward fill last price
                df.loc[gap_start, ['open', 'high', 'low', 'close']] = \
                    df.iloc[gap_end_idx-1][['open', 'high', 'low', 'close']].values
                df.loc[gap_start, 'volume'] = 0
            else:
                # For unexpected gaps, mark for review
                df.loc[gap_start, 'data_quality_flag'] = 'large_gap'
        
        return df
    
    def _is_market_closure_gap(self, current_time, previous_time):
        """Check if gap corresponds to market closure"""
        # Gold markets typically close Friday 17:00 EDT, open Sunday 18:00 EDT
        time_diff = current_time - previous_time
        
        # Weekend gap (Friday close to Sunday open)
        if time_diff >= pd.Timedelta(hours=48) and time_diff <= pd.Timedelta(hours=72):
            return True
            
        # Holiday gaps (longer than typical weekend)
        if time_diff >= pd.Timedelta(days=1):
            return True
            
        return False</code></pre>

                <h2>Data Normalization</h2>
                
                <h3>Price and Volume Normalization</h3>
                <p>Standardize data for analysis across different time periods:</p>

                <div class="info-card">
                    <h4>Normalization Techniques</h4>
                    <ul>
                        <li><strong>Min-Max Scaling:</strong> Scale to 0-1 range</li>
                        <li><strong>Z-Score Normalization:</strong> Mean=0, Std=1</li>
                        <li><strong>Robust Scaling:</strong> Use median and IQR</li>
                        <li><strong>Log Transformation:</strong> For skewed distributions</li>
                    </ul>
                </div>

                <pre data-language="python"><code>from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class DataNormalizer:
    def __init__(self):
        self.scalers = {}
        
    def normalize_prices(self, df, method='robust', fit_period=None):
        """
        Normalize price data using various methods
        """
        price_cols = ['open', 'high', 'low', 'close']
        df_normalized = df.copy()
        
        # Use specific period for fitting scaler (e.g., training data)
        fit_data = df.loc[fit_period] if fit_period else df
        
        if method == 'minmax':
            scaler = MinMaxScaler()
            df_normalized[price_cols] = scaler.fit_transform(fit_data[price_cols])
            
        elif method == 'standard':
            scaler = StandardScaler()
            df_normalized[price_cols] = scaler.fit_transform(fit_data[price_cols])
            
        elif method == 'robust':
            scaler = RobustScaler()
            df_normalized[price_cols] = scaler.fit_transform(fit_data[price_cols])
            
        elif method == 'log_returns':
            # Log returns normalization
            for col in price_cols:
                df_normalized[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
            
        elif method == 'percentage_returns':
            # Percentage returns
            for col in price_cols:
                df_normalized[f'{col}_pct_return'] = df[col].pct_change()
        
        # Store scaler for inverse transformation
        self.scalers[method] = scaler if method in ['minmax', 'standard', 'robust'] else None
        
        return df_normalized
    
    def normalize_volume(self, df, method='log_zscore'):
        """
        Normalize volume data (often highly skewed)
        """
        df_normalized = df.copy()
        
        if method == 'log_transform':
            # Log transformation for skewed volume data
            df_normalized['volume_log'] = np.log1p(df['volume'])  # log(1+x) to handle zeros
            
        elif method == 'log_zscore':
            # Log transform then z-score normalize
            log_volume = np.log1p(df['volume'])
            df_normalized['volume_normalized'] = (log_volume - log_volume.mean()) / log_volume.std()
            
        elif method == 'percentile_rank':
            # Convert to percentile ranks
            df_normalized['volume_percentile'] = df['volume'].rank(pct=True)
            
        return df_normalized</code></pre>

                <h2>Feature Engineering</h2>
                
                <h3>Technical Indicators</h3>
                <p>Create derived features for analysis:</p>

                <pre data-language="python"><code>import talib

class FeatureEngineer:
    def __init__(self):
        self.feature_config = {
            'sma_periods': [5, 10, 20, 50, 200],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2
        }
    
    def create_technical_features(self, df):
        """
        Create comprehensive technical indicators
        """
        features = df.copy()
        
        # Moving Averages
        for period in self.feature_config['sma_periods']:
            features[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            
        for period in self.feature_config['ema_periods']:
            features[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # RSI
        features['rsi'] = talib.RSI(df['close'], timeperiod=self.feature_config['rsi_period'])
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'],
            fastperiod=self.feature_config['macd_fast'],
            slowperiod=self.feature_config['macd_slow'],
            signalperiod=self.feature_config['macd_signal']
        )
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df['close'],
            timeperiod=self.feature_config['bb_period'],
            nbdevup=self.feature_config['bb_std'],
            nbdevdn=self.feature_config['bb_std']
        )
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        return features
    
    def create_price_features(self, df):
        """
        Create price-based features
        """
        features = df.copy()
        
        # Price spreads and ranges
        features['hl_spread'] = df['high'] - df['low']
        features['oc_spread'] = abs(df['open'] - df['close'])
        features['price_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Price position within range
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility (rolling standard deviation of returns)
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
        
        # Price momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        return features
    
    def create_volume_features(self, df):
        """
        Create volume-based features
        """
        features = df.copy()
        
        # Volume moving averages
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
        
        # Volume ratio (current vs average)
        features['volume_ratio_10'] = df['volume'] / features['volume_sma_10']
        
        # On-Balance Volume
        features['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volume-Price Trend
        features['vpt'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Money Flow Index
        features['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        return features</code></pre>

                <h2>Time-Based Features</h2>
                
                <h3>Market Session and Calendar Features</h3>
                <p>Extract temporal patterns:</p>

                <pre data-language="python"><code>    def create_time_features(self, df):
        """
        Create time-based features for pattern recognition
        """
        features = df.copy()
        
        # Basic time components
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # Market session indicators
        features['asian_session'] = ((df.index.hour >= 23) | (df.index.hour <= 8)).astype(int)
        features['london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 17)).astype(int)
        features['ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 22)).astype(int)
        
        # Overlap periods (higher volatility)
        features['london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 17)).astype(int)
        
        # Market open/close indicators
        features['market_open'] = (df.index.hour == 18).astype(int)  # Sunday 18:00 UTC
        features['market_close'] = (df.index.hour == 17).astype(int)  # Friday 17:00 UTC
        
        # Holiday indicators (simplified - would need holiday calendar)
        features['is_holiday'] = 0  # Placeholder for holiday detection
        
        # Cyclical encoding for better ML performance
        features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        return features</code></pre>

                <h2>Quality Scoring and Validation</h2>
                
                <h3>Final Data Quality Assessment</h3>
                <p>Comprehensive quality scoring for processed data:</p>

                <div class="success-card">
                    <h4>Quality Metrics</h4>
                    <ul>
                        <li><strong>Completeness:</strong> Percentage of non-missing values</li>
                        <li><strong>Consistency:</strong> Logical relationships maintained</li>
                        <li><strong>Accuracy:</strong> Outlier and error detection</li>
                        <li><strong>Timeliness:</strong> Data freshness assessment</li>
                    </ul>
                </div>

                <pre data-language="python"><code>class DataQualityScorer:
    def __init__(self):
        self.weights = {
            'completeness': 0.3,
            'consistency': 0.25,
            'accuracy': 0.25,
            'timeliness': 0.2
        }
    
    def score_data_quality(self, df, metadata=None):
        """
        Comprehensive data quality scoring
        """
        scores = {}
        
        # Completeness Score
        non_null_ratio = df.notna().sum().sum() / (len(df) * len(df.columns))
        scores['completeness'] = min(non_null_ratio, 1.0)
        
        # Consistency Score
        consistency_checks = [
            (df['high'] >= df['low']).mean(),
            (df['high'] >= df['open']).mean(),
            (df['high'] >= df['close']).mean(),
            (df['low'] <= df['open']).mean(),
            (df['low'] <= df['close']).mean(),
            (df['volume'] >= 0).mean()
        ]
        scores['consistency'] = np.mean(consistency_checks)
        
        # Accuracy Score (inverse of outlier ratio)
        outlier_detector = self.detect_outliers(df)
        outlier_ratio = outlier_detector['any_outlier'].mean()
        scores['accuracy'] = max(0, 1 - outlier_ratio)
        
        # Timeliness Score
        if metadata and 'staleness_minutes' in metadata:
            staleness = metadata['staleness_minutes']
            # Score decreases as staleness increases
            scores['timeliness'] = max(0, 1 - (staleness / 60))  # 1 hour = 0 score
        else:
            scores['timeliness'] = 1.0  # Assume fresh if no metadata
        
        # Overall Quality Score
        overall_score = sum(
            scores[metric] * self.weights[metric] 
            for metric in scores.keys()
        )
        
        return {
            'individual_scores': scores,
            'overall_score': overall_score,
            'quality_grade': self._get_quality_grade(overall_score)
        }
    
    def _get_quality_grade(self, score):
        """Convert numeric score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'</code></pre>

                <h2>Complete Preprocessing Pipeline</h2>
                
                <h3>Putting It All Together</h3>
                <p>Complete pipeline implementation:</p>

                <pre data-language="python"><code>class GoldDataPipeline:
    def __init__(self, config=None):
        self.preprocessor = GoldDataPreprocessor(config)
        self.normalizer = DataNormalizer()
        self.feature_engineer = FeatureEngineer()
        self.quality_scorer = DataQualityScorer()
        
    def process(self, raw_data, metadata=None):
        """
        Complete preprocessing pipeline
        """
        pipeline_log = {
            'start_time': pd.Timestamp.now(),
            'input_records': len(raw_data),
            'steps': []
        }
        
        try:
            # Step 1: Validate and clean
            validation_results = self.preprocessor.validate_ohlcv_data(raw_data)
            clean_data = validation_results['clean_data']
            pipeline_log['steps'].append({
                'step': 'validation',
                'records_removed': validation_results['invalid_records'],
                'issues': validation_results['issues']
            })
            
            # Step 2: Handle missing data
            filled_data, fill_report = self.preprocessor.handle_missing_data(clean_data)
            pipeline_log['steps'].append({
                'step': 'gap_filling',
                'report': fill_report
            })
            
            # Step 3: Outlier detection and handling
            outliers = self.preprocessor.detect_price_outliers(filled_data)
            # Option to remove or flag outliers
            processed_data = filled_data.copy()
            processed_data['outlier_flag'] = outliers['any_outlier']
            
            # Step 4: Feature engineering
            featured_data = self.feature_engineer.create_technical_features(processed_data)
            featured_data = self.feature_engineer.create_price_features(featured_data)
            featured_data = self.feature_engineer.create_volume_features(featured_data)
            featured_data = self.feature_engineer.create_time_features(featured_data)
            
            # Step 5: Normalization (optional)
            # normalized_data = self.normalizer.normalize_prices(featured_data, method='robust')
            
            # Step 6: Quality assessment
            quality_report = self.quality_scorer.score_data_quality(featured_data, metadata)
            
            pipeline_log['end_time'] = pd.Timestamp.now()
            pipeline_log['processing_time'] = (pipeline_log['end_time'] - pipeline_log['start_time']).total_seconds()
            pipeline_log['final_records'] = len(featured_data)
            pipeline_log['quality_score'] = quality_report['overall_score']
            
            return {
                'data': featured_data,
                'quality_report': quality_report,
                'pipeline_log': pipeline_log
            }
            
        except Exception as e:
            pipeline_log['error'] = str(e)
            raise e

# Usage example
pipeline = GoldDataPipeline()
result = pipeline.process(raw_gold_data)

print(f"Processing completed in {result['pipeline_log']['processing_time']:.2f} seconds")
print(f"Data quality score: {result['quality_report']['overall_score']:.3f}")
print(f"Final dataset shape: {result['data'].shape}")</code></pre>

                <h2>Performance Optimization</h2>
                
                <div class="warning-card">
                    <h3>Large Dataset Considerations</h3>
                    <ul>
                        <li>Use chunked processing for datasets > 1M records</li>
                        <li>Implement parallel processing for independent operations</li>
                        <li>Consider using Dask for out-of-core processing</li>
                        <li>Cache intermediate results to disk</li>
                        <li>Use efficient data types (int32 vs int64, float32 vs float64)</li>
                    </ul>
                </div>

                <h2>Next Steps</h2>
                <p>With clean, processed data in hand, we can now move on to cloud infrastructure setup. The next chapter will cover deploying your data processing pipeline to the cloud for scalable, automated operation.</p>
            </div>
        `
    },
    {
        id: 'cloud-setup',
        title: 'Cloud Infrastructure Setup',
        subtitle: 'Deploying scalable gold market analysis systems',
        section: 'Infrastructure',
        readingTime: '20 min read',
        content: `
            <div class="content">
                <div class="chapter-meta">
                    <span class="chapter-number">Chapter 6</span>
                    <div class="reading-time">
                        <i class="fas fa-clock"></i>
                        <span>20 min read</span>
                    </div>
                </div>

                <h1>Cloud Infrastructure Setup</h1>
                <p>Building a scalable, reliable cloud infrastructure is essential for processing large volumes of gold market data. This chapter covers complete cloud deployment strategies using AWS, Google Cloud, and Azure.</p>

                <h2>Architecture Overview</h2>
                
                <div class="data-card">
                    <h3>Cloud Architecture Components</h3>
                    <ul>
                        <li><strong>Data Ingestion:</strong> API gateways, streaming services, message queues</li>
                        <li><strong>Data Processing:</strong> Serverless functions, container orchestration</li>
                        <li><strong>Data Storage:</strong> Time-series databases, data lakes, caching layers</li>
                        <li><strong>Analytics:</strong> Machine learning services, real-time analytics</li>
                        <li><strong>Monitoring:</strong> Logging, metrics, alerting systems</li>
                        <li><strong>Security:</strong> Identity management, encryption, network security</li>
                    </ul>
                </div>

                <h2>AWS Infrastructure Setup</h2>
                
                <h3>Core Services Architecture</h3>
                <p>AWS-based gold market analysis platform:</p>

                <pre data-language="yaml"><code># CloudFormation template for gold market infrastructure
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Gold Market Analysis Platform Infrastructure'

Parameters:
  Environment:
    Type: String
    Default: 'dev'
    AllowedValues: ['dev', 'staging', 'prod']
  
Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-gold-market-vpc'

  # Private Subnets for databases and processing
  PrivateSubnetA:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.1.0/24
      Tags:
        - Key: Name
          Value: !Sub '${Environment}-private-subnet-a'

  PrivateSubnetB:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.2.0/24

  # Public Subnets for load balancers and NAT gateways
  PublicSubnetA:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.101.0/24
      MapPublicIpOnLaunch: true

  # Time Series Database (TimeStream)
  GoldPriceDatabase:
    Type: AWS::Timestream::Database
    Properties:
      DatabaseName: !Sub '${Environment}-gold-market-db'
      
  GoldPriceTable:
    Type: AWS::Timestream::Table
    Properties:
      DatabaseName: !Ref GoldPriceDatabase
      TableName: 'price-data'
      RetentionProperties:
        MemoryStoreRetentionPeriodInHours: 24
        MagneticStoreRetentionPeriodInDays: 365

  # S3 Bucket for data lake
  DataLakeBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${Environment}-gold-market-datalake'
      VersioningConfiguration:
        Status: Enabled
      LifecycleConfiguration:
        Rules:
          - Id: ArchiveOldData
            Status: Enabled
            Transitions:
              - TransitionInDays: 30
                StorageClass: STANDARD_IA
              - TransitionInDays: 90
                StorageClass: GLACIER

  # Kinesis for real-time data streaming
  PriceDataStream:
    Type: AWS::Kinesis::Stream
    Properties:
      Name: !Sub '${Environment}-gold-price-stream'
      ShardCount: 2
      RetentionPeriodHours: 168  # 7 days

  # Lambda functions for data processing
  DataIngestionFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub '${Environment}-gold-data-ingestion'
      Runtime: python3.9
      Handler: lambda_function.lambda_handler
      Code:
        ZipFile: |
          # Placeholder - actual code would be deployed separately
          def lambda_handler(event, context):
              return {'statusCode': 200}
      Timeout: 300
      MemorySize: 1024</code></pre>

                <h3>Data Processing Pipeline</h3>
                <p>AWS Lambda-based processing functions:</p>

                <pre data-language="python"><code>import json
import boto3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class AWSGoldDataProcessor:
    def __init__(self):
        self.timestream = boto3.client('timestream-write')
        self.s3 = boto3.client('s3')
        self.kinesis = boto3.client('kinesis')
        self.cloudwatch = boto3.client('cloudwatch')
        
        self.database_name = 'gold-market-db'
        self.table_name = 'price-data'
        self.bucket_name = 'gold-market-datalake'
    
    def lambda_handler(self, event, context):
        """
        Main Lambda handler for processing gold price data
        """
        try:
            # Parse incoming data
            if 'Records' in event:
                # Kinesis stream trigger
                records = self._parse_kinesis_records(event['Records'])
            else:
                # Direct API trigger
                records = [json.loads(event['body'])]
            
            # Process each record
            processed_records = []
            for record in records:
                processed = self._process_price_record(record)
                if processed:
                    processed_records.append(processed)
            
            # Batch write to TimeStream
            if processed_records:
                self._write_to_timestream(processed_records)
                
            # Archive to S3 for long-term storage
            self._archive_to_s3(processed_records)
            
            # Send metrics to CloudWatch
            self._send_metrics(len(processed_records))
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'processed_records': len(processed_records),
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
            
        except Exception as e:
            print(f"Error processing records: {str(e)}")
            self._send_error_metric(str(e))
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }
    
    def _process_price_record(self, record):
        """
        Process individual price record with validation and enrichment
        """
        try:
            # Validate required fields
            required_fields = ['timestamp', 'symbol', 'price', 'volume']
            if not all(field in record for field in required_fields):
                print(f"Missing required fields in record: {record}")
                return None
            
            # Data validation
            if record['price'] <= 0 or record['volume'] < 0:
                print(f"Invalid price/volume data: {record}")
                return None
            
            # Enrich with additional metadata
            processed_record = {
                'timestamp': record['timestamp'],
                'symbol': record['symbol'],
                'price': float(record['price']),
                'volume': int(record['volume']),
                'bid': float(record.get('bid', 0)),
                'ask': float(record.get('ask', 0)),
                'spread': float(record.get('ask', 0)) - float(record.get('bid', 0)),
                'source': record.get('source', 'unknown'),
                'processed_at': datetime.utcnow().isoformat()
            }
            
            return processed_record
            
        except Exception as e:
            print(f"Error processing record {record}: {str(e)}")
            return None
    
    def _write_to_timestream(self, records):
        """
        Write processed records to AWS TimeStream
        """
        if not records:
            return
        
        # Prepare records for TimeStream format
        timestream_records = []
        for record in records:
            ts_record = {
                'Time': record['timestamp'],
                'TimeUnit': 'MILLISECONDS',
                'Dimensions': [
                    {'Name': 'symbol', 'Value': record['symbol']},
                    {'Name': 'source', 'Value': record['source']}
                ],
                'MeasureName': 'price_data',
                'MeasureValueType': 'MULTI',
                'MeasureValues': [
                    {'Name': 'price', 'Value': str(record['price']), 'Type': 'DOUBLE'},
                    {'Name': 'volume', 'Value': str(record['volume']), 'Type': 'BIGINT'},
                    {'Name': 'bid', 'Value': str(record['bid']), 'Type': 'DOUBLE'},
                    {'Name': 'ask', 'Value': str(record['ask']), 'Type': 'DOUBLE'},
                    {'Name': 'spread', 'Value': str(record['spread']), 'Type': 'DOUBLE'}
                ]
            }
            timestream_records.append(ts_record)
        
        # Write in batches (TimeStream limit: 100 records per request)
        batch_size = 100
        for i in range(0, len(timestream_records), batch_size):
            batch = timestream_records[i:i + batch_size]
            
            try:
                self.timestream.write_records(
                    DatabaseName=self.database_name,
                    TableName=self.table_name,
                    Records=batch
                )
                print(f"Successfully wrote {len(batch)} records to TimeStream")
                
            except Exception as e:
                print(f"Error writing to TimeStream: {str(e)}")
                raise e
    
    def _archive_to_s3(self, records):
        """
        Archive processed data to S3 for long-term storage and analytics
        """
        if not records:
            return
        
        # Create DataFrame for efficient storage
        df = pd.DataFrame(records)
        
        # Partition by date for efficient querying
        date_str = datetime.utcnow().strftime('%Y/%m/%d')
        hour_str = datetime.utcnow().strftime('%H')
        
        # Save as Parquet for efficient compression and querying
        key = f"price-data/year={date_str[:4]}/month={date_str[5:7]}/day={date_str[8:10]}/hour={hour_str}/data_{int(datetime.utcnow().timestamp())}.parquet"
        
        # Convert DataFrame to Parquet bytes
        parquet_buffer = df.to_parquet(index=False)
        
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=parquet_buffer,
                ContentType='application/octet-stream'
            )
            print(f"Successfully archived {len(records)} records to S3: {key}")
            
        except Exception as e:
            print(f"Error archiving to S3: {str(e)}")
            # Don't raise - archiving failure shouldn't stop real-time processing</code></pre>

                <h2>Container Orchestration with EKS</h2>
                
                <h3>Kubernetes Deployment</h3>
                <p>For more complex processing workloads, use Amazon EKS:</p>

                <pre data-language="yaml"><code># Kubernetes deployment for gold market analytics
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gold-analytics-processor
  namespace: gold-market
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gold-analytics
  template:
    metadata:
      labels:
        app: gold-analytics
    spec:
      containers:
      - name: analytics-processor
        image: your-account.dkr.ecr.region.amazonaws.com/gold-analytics:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: AWS_REGION
          value: "us-east-1"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: gold-analytics-service
  namespace: gold-market
spec:
  selector:
    app: gold-analytics
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gold-analytics-hpa
  namespace: gold-market
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gold-analytics-processor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80</code></pre>

                <h2>Google Cloud Platform Setup</h2>
                
                <h3>BigQuery and Cloud Functions</h3>
                <p>GCP-based analytics platform:</p>

                <pre data-language="python"><code>from google.cloud import bigquery, pubsub_v1, storage
from google.cloud import functions_v1
import json

class GCPGoldAnalytics:
    def __init__(self, project_id):
        self.project_id = project_id
        self.bigquery_client = bigquery.Client()
        self.pubsub_client = pubsub_v1.PublisherClient()
        self.storage_client = storage.Client()
        
        # BigQuery dataset and table setup
        self.dataset_id = 'gold_market_data'
        self.table_id = 'price_data'
        
    def setup_bigquery_infrastructure(self):
        """
        Set up BigQuery dataset and tables for gold market data
        """
        # Create dataset
        dataset_ref = self.bigquery_client.dataset(self.dataset_id)
        
        try:
            dataset = self.bigquery_client.get_dataset(dataset_ref)
            print(f"Dataset {self.dataset_id} already exists")
        except:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset = self.bigquery_client.create_dataset(dataset)
            print(f"Created dataset {self.dataset_id}")
        
        # Create price data table
        table_ref = dataset_ref.table(self.table_id)
        
        try:
            table = self.bigquery_client.get_table(table_ref)
            print(f"Table {self.table_id} already exists")
        except:
            schema = [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("price", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("volume", "INT64", mode="REQUIRED"),
                bigquery.SchemaField("bid", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("ask", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("spread", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("market_session", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("processed_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            table = bigquery.Table(table_ref, schema=schema)
            
            # Partition by timestamp for better performance
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
            
            # Cluster by symbol for better query performance
            table.clustering_fields = ["symbol", "source"]
            
            table = self.bigquery_client.create_table(table)
            print(f"Created table {self.table_id}")
    
    def cloud_function_handler(self, request):
        """
        Google Cloud Function for processing gold price data
        """
        try:
            # Parse incoming data
            request_json = request.get_json()
            
            if not request_json or 'price_data' not in request_json:
                return {'error': 'Invalid request format'}, 400
            
            price_data = request_json['price_data']
            
            # Process and validate data
            processed_records = []
            for record in price_data:
                if self._validate_record(record):
                    processed_record = self._enrich_record(record)
                    processed_records.append(processed_record)
            
            # Insert into BigQuery
            if processed_records:
                self._insert_to_bigquery(processed_records)
                
                # Publish to Pub/Sub for real-time processing
                self._publish_to_pubsub(processed_records)
            
            return {
                'status': 'success',
                'processed_records': len(processed_records)
            }, 200
            
        except Exception as e:
            print(f"Error in cloud function: {str(e)}")
            return {'error': str(e)}, 500
    
    def _insert_to_bigquery(self, records):
        """
        Insert processed records into BigQuery
        """
        table_ref = self.bigquery_client.dataset(self.dataset_id).table(self.table_id)
        table = self.bigquery_client.get_table(table_ref)
        
        errors = self.bigquery_client.insert_rows_json(table, records)
        
        if errors:
            print(f"BigQuery insert errors: {errors}")
            raise Exception(f"Failed to insert records: {errors}")
        else:
            print(f"Successfully inserted {len(records)} records into BigQuery")
    
    def _publish_to_pubsub(self, records):
        """
        Publish records to Pub/Sub for real-time processing
        """
        topic_path = self.pubsub_client.topic_path(self.project_id, 'gold-price-updates')
        
        for record in records:
            message_data = json.dumps(record).encode('utf-8')
            future = self.pubsub_client.publish(topic_path, message_data)
            
        print(f"Published {len(records)} messages to Pub/Sub")</code></pre>
</invoke>