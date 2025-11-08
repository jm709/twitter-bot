import time
import tweepy
from tweetGenerator import TwitterBotGenerator
import artMaker
from newsGetter import newsBeast
import sys


# Twitter (x) API keys
api_key = ""
api_secret = ""
bearer_token = ""
access_token = ""
access_secret = ""

bot = TwitterBotGenerator()

client = tweepy.Client(bearer_token, api_key, api_secret, access_token, access_secret)

auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

def rand_tweet():
    tweet = bot.generate_tweet()
    client.create_tweet(text=tweet)

def news_tweet():
    # Generate the prompt
    newsPrompt = newsBeast("")

    print("Getting news")
    all_news = newsPrompt.get_news()
    prompt = newsPrompt.news_compacter(all_news)

    # Generate the art
    print("Genning art")
    print(prompt)
    gen_art = artMaker.generate_image(prompt[19:])

    # Post to twitter
    print("Posting")
    media_id = api.media_upload(filename=f"{gen_art}.png").media_id_string
    print(media_id)

    # Create the tweet
    time.sleep(1)
    # time.sleep is essential for ensuring that the upload goes thru.
    client.create_tweet(text=prompt, media_ids=[media_id])

def main():
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        if function_name == "rand_tweet":
            rand_tweet()
        elif function_name == "news_tweet":
            news_tweet()
        else:
            print(f"Unknown function: {function_name}")
    else:
        print("No function specified")

if __name__ == "__main__":
    main()
