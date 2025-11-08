from newsapi import NewsApiClient
from collections import Counter
from tweetGenerator import TwitterBotGenerator

class newsBeast:
    def __init__(self, news_api_key, model_path=r"..\my_twitter_bot_model"):
        self.news_key = news_api_key
        self.llm_generator = TwitterBotGenerator(model_path)
        # Common words that are ignored, add more if you want better results
        self.common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'a', 'is', 'it', 'as',
            'photos', 'images', 'pictures', 'image', 'photo', 'pic', 'face', 'portrait', 'head', 'as', 'this',
            'news', 'article', 'story', 'report', 'update', 'latest', 'new', 'old', 'young', 'from',
            'man', 'woman', 'person', 'people', 'celebrity', 'famous', 'star', 'actor', 'actress',
            'wikipedia', 'wiki', 'biography', 'bio', 'profile', 'page', 'site', 'website', '–', '-',
            'com', 'org', 'net', 'www', 'http', 'https', 'html', 'jpg', 'png', 'gif', 'after', 'its', 'says',
            'file', 'files', 'download', 'upload', 'view', 'click', 'here', 'more', 'info', 'out', 'be',
            'information', 'details', 'facts', 'born', 'age', 'years', 'year', 'old', 'will', 'how',
            'early', 'life', 'career', 'work', 'best', 'known', 'famous', 'popular', 'up', 'no', 'your', 'what',
            'american', 'british', 'canadian', 'australian', 'english', 'french', 'german',
            'singer', 'musician', 'director', 'producer', 'writer', 'author', 'politician',
            'athlete', 'model', 'businessman', 'entrepreneur', 'ceo', 'founder', 'announces'
        }

    def get_news(self):
        newsapi = NewsApiClient(api_key=self.news_key)
        # sources_real = "bbc-news,al-jazeera-english,reuters,associated-press,politico,the-hill,independent,bloomberg,cnbc"
        # sources_gossip = "mtv-news,entertainment-weekly,the-lad-bible,mashable,vice-news"

        top_headlines = newsapi.get_top_headlines(
                                                 language='en'
                                                 )
        all_news = newsapi.get_everything(q='breaking news', sort_by='publishedAt', language='en')

        return all_news


    def news_compacter(self, news: dict[str]):
        news_gloop = ""
        max_chars = 500

        all_words = [word.lower().strip('.,!?') for article in news['articles']
                     for word in article['title'].split()]

        awesome_words = [word for word in all_words if word not in self.common_words]

        counts = Counter(awesome_words)
        print(counts)
        if not counts:
            return ""

        if counts:
            most_common = counts.most_common(1)[0][0]
        else:
            most_common = ""

        counter = 0
        x = 1
        new_line = ""
        used_articles = set()

        # Process words by frequency
        for word, frequency in counts.most_common():
            for i, article in enumerate(news['articles']):
                if i in used_articles:
                    continue

                if word.lower() in article['title'].lower():
                    new_line = "\n" + article['title'] if news_gloop else article['title']

                    if len(news_gloop + new_line) > max_chars:
                        break

                    news_gloop += new_line
                    used_articles.add(i)

    #Most of this is "Pre-prompt", the only thing that's tweeted is generated headline and after"
        prompt = f"""
Based on these news headlines, generate a long, compelling headline that captures the main themes and mood of today's news.
        
IMPORTANT RULES:
- Do not include any URLs, links, or web addresses
- Do not include bit.ly, .com, http, https, or www
- Write only the headline text
- End with proper punctuation
        
        Headlines:
        {news_gloop}
Generated Headline:
{most_common}"""

        response = self.llm_generator.generate_headline(len(news_gloop),prompt)
        return response

# Don't Question My Naming Scheme
# if __name__ == "__main__":
#     burger = newsBeast("APIKEY")
#     all_news = burger.get_news()
#     print(burger.news_compacter(all_news))
