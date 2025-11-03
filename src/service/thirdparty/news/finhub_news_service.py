from newspaper import Article

from src.util.env_property import get_env_property
import finnhub
from datetime import datetime, timedelta
import time
from src.util.logger import logger

FINNHUB_API_KEY = get_env_property("FINNHUB_API_KEY", "sandbox_c0m8n2qad3i8e1f5g5g0")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

ticker_to_company_name = {
  'AMZN' : 'Amazon',
  'MU' : 'Micron',
  'AVGO' : 'Broadcom',
  'NVDA' : 'Nvidia',
  'MSFT' : 'Microsoft',
  'SOFI' : 'SoFi',
  'SOUN' : 'SoundHound',
  'UBER' : 'Uber',
  'ASML' : 'ASML',
  'ORCL' : 'Orac',
  'CSCO' : 'Cisco',
  'RKLB' : 'Rocket',
  'PLTR' : 'Palantir',
  'SMCI' : 'Super Micro',
  'AMD' : 'AMD',
  'ACHR' : 'Archer Aviation',
  'NFLX' : 'Netflix',
  'AAPL' : 'Apple',
  'TSLA' : 'Tesla',
  'GOOGL' : 'Alphabet',
  'META': 'Meta',
  'AMGN': 'Amgen',
  'HIMS': 'Hims',
  'ANF': 'Abercrombie',
  'UPST': 'Upstart',
}



def form_article_test(article_data):
  article_result = ""
  url = article_data.get("url")
  try:
    article = Article(url)
    article.download()
    article.parse()

    article_text = article.text
    article_result+=article_text
    logger.info(f"Fetched article {url} with length {len(article_text)}")
  except Exception as e:
    logger.info(f"Failed fetch article {url}. exception is {e} Return summary instead.")
    alternative_summary = article_data.get('headline', 'No headline') + ". " + article_data.get("summary", "No summary available.")
    article_result += alternative_summary

  return article_result

# --HEADLINES
def fetch_company_news(ticker, filter_by_name=True, later_than_hours_filter=24, max_articles=5):
  counter = 0
  ticker = ticker.strip()
  now = datetime.utcnow()
  six_hours_ago = now - timedelta(hours=later_than_hours_filter)
  yesterday = now - timedelta(days=1)
  today_str = now.strftime("%Y-%m-%d")
  yesterday_str = yesterday.strftime("%Y-%m-%d")

  # Fetch all news from today (UTC-based)
  news_items = finnhub_client.company_news(ticker, _from=yesterday_str, to=today_str)
  logger.info(f"HEADLINES: {news_items}")

  result = []
  for item in news_items:
    if len(result) >= max_articles:
      break

    if filter_by_name:
      if (ticker not in item.get("headline", "") and ticker not in item.get("summary", "")
          and ticker_to_company_name.get(ticker, '!@#!@#') not in item.get("headline", "")
          and ticker_to_company_name.get(ticker, '23423') not in item.get('summary', "")):
        logger.info(f"Skipping as no naming for {ticker}, headline: {item.get('headline', '')}.")
        continue

    timestamp = datetime.utcfromtimestamp(item.get("datetime", 0))
    if timestamp < six_hours_ago:
      continue

    identificator = str(counter)
    counter += 1

    entity = {
      "id": identificator + ticker,
      "url": item.get("url"),
      "headline": item.get("headline"),
      "summary": item.get("summary")
    }

    time.sleep(2)
    entity['text'] = form_article_test(entity)
    result.append(entity)

  return result



if __name__ == "__main__":
  logger.info(fetch_company_news("TSLA", filter_by_name=True))
