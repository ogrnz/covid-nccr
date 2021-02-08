CREATE TABLE "tweets" (
	"tweet_id"	integer NOT NULL UNIQUE,
	"created_at"	TEXT NOT NULL,
	"handle"	TEXT NOT NULL,
	"name"	TEXT NOT NULL,
	"oldtext"	TEXT NOT NULL,
	"text"	TEXT NOT NULL,
	"url"	TEXT NOT NULL,
	"type"	TEXT NOT NULL,
	"retweets"	INTEGER,
	"favorites"	INTEGER,
	PRIMARY KEY("tweet_id")
);