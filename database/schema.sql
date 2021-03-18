CREATE TABLE IF NOT EXISTS "tweets" (
	"tweet_id"	TEXT UNIQUE,
	"covid_theme"	INTEGER,
	"created_at"	TEXT,
	"handle"	TEXT,
	"name"	TEXT,
	"oldtext"	TEXT,
	"text"	TEXT,
	"url"	TEXT,
	"type"	TEXT,
	"retweets"	TEXT,
	"favorites"	TEXT,
    "topic" TEXT,
    "subcat" TEXT,
    "position" TEXT,
    "frame" TEXT,
	PRIMARY KEY("tweet_id")
);
