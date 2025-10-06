from abc import ABC, abstractmethod
import json
import requests
import finnhub
from datetime import datetime, timedelta
from newspaper import Article
import time

from src.util.env_property import get_env_property

FINNHUB_API_KEY = get_env_property("FINNHUB_API_KEY")
TWELVE_DATA_API_KEY = get_env_property("TWELVE_DATA_API_KEY")


class StockPrice():
  def __init__(self, ticker: str, last_price: float, day_change:float, date: str = None):
    self.ticker = ticker
    self.last_price = last_price
    self.day_change = day_change
    self.date = date

  @classmethod
  def init_with_price(cls, ticker, last_price, before_last_price, date=None):
    if last_price is None or before_last_price is None:
      return None
    day_change = (last_price - before_last_price) / before_last_price * 100

    return StockPrice(ticker, last_price, day_change, date)

  @classmethod
  def init_with_change(cls, ticker, last_price, day_change, date=None):
    if last_price is None or day_change is None:
      return None

    return StockPrice(ticker, last_price, day_change, date)

  def __repr__(self):
    return f"StockPrice(ticker={self.ticker}, price={self.last_price}, change={self.day_change}, date={self.date})"

class PriceFactory():

  def __init__(self, providers, tickers: []):
    """
    Initialize the PriceFactory with a specific price provider.
    :param provider: An instance of a class that implements PriceProvider.
    """
    self.providers : [] = providers
    self.tickers = tickers
    self.ticker_counter = 0
    self.try_counter = 0

  def if_next_ticker_exist(self):
    if self.ticker_counter >= len(self.tickers):
      return False
    return True

  def get_price(self):
    """
    Get the next ticker from the list of tickers.
    :return: The next ticker symbol.
    """

    ticker = self.tickers[self.ticker_counter]
    provider = self.providers[self.try_counter % len(self.providers)]

    return provider.get_last_price(ticker)

  def get_ticker_change_map(self):
    """
    Get a map of tickers to their last prices and changes.
    :return: A dictionary mapping ticker symbols to StockPrice objects.
    """
    ticker_change_map = {}
    while self.if_next_ticker_exist():
      stock_price: StockPrice = self.get_price()
      time.sleep(1)

      fail_counter = 0
      while stock_price is None:
        time.sleep(2)

        if fail_counter > 5:
          print(f"Failed to fetch price for {self.tickers[self.ticker_counter]}")
          break

        stock_price = self.get_price()
        fail_counter += 1
        self.try_counter += 1

      if stock_price:
        ticker_change_map[stock_price.ticker] = stock_price.day_change

      self.ticker_counter += 1
      self.try_counter += 1

    return ticker_change_map




class PriceProvider(ABC):

  @abstractmethod
  def get_last_price(self, ticker: str) -> StockPrice:
    """
    Fetch the last price of the stock.
    :param ticker: Stock ticker symbol.
    :return: StockPrice object with the last price and change.
    """
    pass


class PriceProviderTwelveData(PriceProvider):
  def __init__(self, api_key: str):
    """
    Initialize the Twelve Data price provider with the API key.
    :param api_key: API key for Twelve Data.
    """
    self.api_key = api_key

  def get_last_price(self, ticker: str) -> StockPrice:
    response = requests.get(f"https://api.twelvedata.com/time_series?apikey={self.api_key}&interval=1day&format=JSON&symbol={ticker}&outputsize=2")
    data = json.loads(response.content.decode('utf-8'))
    try:
      today_price = float(data['values'][0]['close'])
      yesterday_price = float(data['values'][1]['close'])
      print("get data from twelvedata for ticker: " + ticker)

      return StockPrice.init_with_price(ticker, today_price, yesterday_price)
    except Exception as e:
      print(f"{type(self)} Failed to fetch price for {ticker}. response is: {data}")
      return None


class PriceProviderFinHub(PriceProvider):

  def __init__(self, api_key: str):
    """
    Initialize the Twelve Data price provider with the API key.
    :param api_key: API key for Twelve Data.
    """
    self.api_key = api_key

  def get_last_price(self, ticker: str) -> StockPrice:
    response = requests.get(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.api_key}")
    data = json.loads(response.content.decode('utf-8'))
    try:
      today_price = float(data['c'])
      change = float(data['dp'])
      print("get data from finhub for ticker: " + ticker)
      return StockPrice.init_with_change(ticker, today_price, change)
    except Exception:
      print(f"{type(self)} Failed to fetch price for {ticker}. response is: {data}")
      return None

price_provider1 = PriceProviderFinHub(FINNHUB_API_KEY)
price_provider2 = PriceProviderTwelveData(TWELVE_DATA_API_KEY)



def get_price_change_for_tickers(tickers: [str]):
  factory = PriceFactory([price_provider1, price_provider2], tickers)
  return factory.get_ticker_change_map()
  # example return {'AAPL': -3.5, 'MU': -2.3, 'SOFI': -3.1, 'UBER': -1.8, 'PGY': -3}
