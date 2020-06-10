#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:10:36 2020

# Code to download raw tweets from twitter

to install tweepy:
conda install -c conda-forge tweepy~

@author: charles mégnin
"""

# https://github.com/tweepy/tweepy
import tweepy
import csv

# user must enter credentials as access_key, access_secret, consumer_key, consumer_secret
import twitter_credentials


def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(twitter_credentials.consumer_key,
                               twitter_credentials.consumer_secret)
    auth.set_access_token(twitter_credentials.access_key,
                          twitter_credentials.access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,
                                   count = 200,
                                   include_rts = False,
                                   tweet_mode = 'extended')

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,
                                       count = 200,
                                       max_id = oldest,
                                       include_rts = False,
                                       tweet_mode = 'extended'
                                       )

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [[tweet.user.screen_name, tweet.created_at, tweet.full_text]
                 for tweet in alltweets]

    # write the csv
    with open('%s_tweets.csv' % screen_name, 'wt') as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerow(["screen_name", "date", "tweet"])
        writer.writerows(outtweets)


if __name__ == '__main__':
    # handle of the target user
    handle = 'MoDem'
    get_all_tweets(handle)
