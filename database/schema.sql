CREATE TABLE IF NOT EXISTS "tweets" (
	"tweet_id"	TEXT UNIQUE,
	"covid_theme"	INTEGER,
	"created_at"	TEXT,
	"handle"	TEXT,
	"name" TEXT,
	"old_text"	TEXT,
	"text"	TEXT,
	"url"	TEXT,
	"type"	TEXT,
	"retweets"	TEXT,
	"favorites"	TEXT,
    "topic" TEXT,
    "subcat" TEXT,
    "position" TEXT,
    "frame" TEXT,
    "theme_hardcoded" INTEGER,
	PRIMARY KEY("tweet_id")
);
