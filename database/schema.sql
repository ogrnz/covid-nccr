CREATE TABLE IF NOT EXISTS "tweets" (
	"tweet_id"	integer NOT NULL UNIQUE,
	"covid_theme"	INTEGER,
	"created_at"	TEXT NOT NULL,
	"handle"	TEXT NOT NULL,
	"name"	TEXT NOT NULL,
	"oldtext"	TEXT,
	"text"	TEXT,
	"url"	TEXT NOT NULL,
	"type"	TEXT NOT NULL,
	"retweets"	INTEGER,
	"favorites"	INTEGER,
	PRIMARY KEY("tweet_id")
);
