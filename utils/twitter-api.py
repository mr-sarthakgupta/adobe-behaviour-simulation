import time
import json
import requests
import tweepy
import pandas as pd

df = pd.read_excel("/content/behaviour_simulation_test_company.xlsx")

max_unique_usernames = df["username"].unique().tolist()

print(max_unique_usernames)
print(len(max_unique_usernames))
print("=" * 50)

f = pd.DataFrame(max_unique_usernames, columns='username')

response = requests.get(
    "https://api.twitter.com/graphql/NimuplG1OB7Fd2btCLdBOw/UserByScreenName",
)


# Set up Twitter API credentials
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

start_value = 1
for i, username in enumerate(f["username"][start_value:]):
    followers = tweepy.Cursor(api.followers, screen_name=username).items()
    f.at[i, "followers"] = followers

f.to_csv("./username_new.csv")
