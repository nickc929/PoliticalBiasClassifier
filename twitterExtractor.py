import twitter
import requests
from requests_oauthlib import OAuth1
from twitterKeys import twitterKeys

consumer_key = twitterKeys['consumer_key'],
consumer_secret = twitterKeys['consumer_secret'],
access_token_key = twitterKeys['access_token_key'],
access_token_secret = twitterKeys['access_token_secret']

api = twitter.Api(consumer_key=consumer_key,
                  consumer_secret=consumer_secret,
                  access_token_key=access_token_key,
                  access_token_secret=access_token_secret)


url = '''https://api.twitter.com/1.1/tweets/search/fullarchive/
         development.json?query=from%3ASenTomCotton&fromDate=201701220000&toDate=201711300000'''
auth = OAuth1(consumer_key, consumer_secret, access_token_key, access_token_secret)


r = requests.get(url, auth=auth)
outFile = open("tomCottonTweets.txt", "w")
for tweet in r.json()['results']:
    print(tweet['text'].encode("utf-8"), end="\t", file=outFile)
    print(tweet['created_at'].encode("utf-8"), end="\t", file=outFile)
    print("0", file=outFile)
