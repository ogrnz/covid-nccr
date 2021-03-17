"""
Database module
"""

import sqlite3

from common.app import App


class Database:
    """
    Class that handles database queries
    """

    def __init__(self, db_name: str, app: App):
        self.app = app
        self.db_name = db_name
        self.conn = None
        self.sql_schema = self.__retrieve_schema()
        self.create_table()
        self.db_name = db_name

    def __enter__(self):
        self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def connect(self):
        """
        Establish connection to database
        """

        try:
            self.conn = sqlite3.connect(f"{self.app.root_dir}/database/{self.db_name}")
        except sqlite3.Error as error:
            print(f"Error establishing connection to database {self.db_name}\n", error)

    def __retrieve_schema(self):
        """
        Get sql tweets table schema
        """

        with open(f"{self.app.root_dir}/database/schema.sql", "r") as sql:
            sql_lines = sql.readlines()
        return "".join(sql_lines)

    def create_table(self):
        """
        Create an SQL table
        """

        try:
            self.connect()
            cur = self.conn.cursor()
            cur.execute(self.sql_schema)
        except sqlite3.Error as error:
            print("Error creating table", error)
        finally:
            if self.conn:
                self.conn.close()

    def get_all_tweets(self):
        """
        Retrieve all tweets
        """

        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM tweets")

            return cur.fetchall()
        except sqlite3.Error as error:
            print("Error", error)
        finally:
            cur.close()

    def get_fields(self, fields: list, only_covid: bool = False, limit=None):
        """
        Retrieve desired fields from all tweets
        """

        where_cond = "WHERE covid_theme=1"
        if not only_covid:
            where_cond = None

        fields = ",".join(fields)
        try:
            cur = self.conn.cursor()
            if limit is not None:
                cur.execute(f"SELECT {fields} FROM tweets {where_cond} LIMIT {limit}")
            else:
                cur.execute(f"SELECT {fields} FROM tweets {where_cond}")

            return cur.fetchall()
        except sqlite3.Error as error:
            print("Error", error)
        finally:
            cur.close()

    def get_by_type(self, tweet_type: str):
        """
        Retrieve tweets by type
        :param type: available types are ['Retweet','New', 'Reply']
        """

        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM tweets WHERE type=?", (tweet_type,))

            return cur.fetchall()
        except sqlite3.Error as error:
            print("Error", error)
        finally:
            cur.close()

    def update_tweet_by_id(self, tweet_id: int, field: str, content: any):
        """
        Update tweet in database by id
        """

        sql = f"""UPDATE tweets SET {field} = ? WHERE tweet_id = ?"""
        cur = self.conn.cursor()

        try:
            cur.execute(
                sql,
                (
                    content,
                    tweet_id,
                ),
            )
            self.conn.commit()
        except sqlite3.Error as error:
            print(f"Error updating tweet {tweet_id} ", error)
        finally:
            cur.close()

    def update_theme_many(self, params):
        """
        Update tweets in bulk
        """

        sql = """UPDATE tweets SET covid_theme = ? WHERE tweet_id = ?"""
        cur = self.conn.cursor()

        try:
            cur.executemany(
                sql,
                params,
            )
            self.conn.commit()

            return cur.rowcount
        except sqlite3.Error as error:
            print("Error updating tweets", error)
        finally:
            cur.close()

    def insert_tweet(self, tweet):
        """
        Insert new tweet into database
        """
        sql = """ INSERT OR IGNORE INTO tweets(
                        tweet_id,
                        covid_theme,
                        type,
                        created_at,
                        handle,
                        name,
                        oldtext,
                        text,
                        url,
                        retweets,
                        favorites,
                        topic,
                        subcat,
                        position,
                        frame)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) """
        cur = self.conn.cursor()

        try:
            cur.execute(sql, tweet)
            self.conn.commit()

            return cur.lastrowid
        except sqlite3.Error as error:
            print(f"Error inserting new tweet \n {tweet} \n {error}")
        finally:
            cur.close()

    def insert_many(self, tweets):
        """
        Insert new tweet into database
        """
        sql = """ INSERT OR IGNORE INTO tweets
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) """
        cur = self.conn.cursor()

        try:
            cur.executemany(sql, tweets)
            self.conn.commit()

            return cur.rowcount
        except sqlite3.Error as error:
            print(f"Error inserting new tweets \n {error}")
        finally:
            cur.close()

    def get_last_id_by_handle(self, screen_name):
        """
        Retrieve last inserted tweet id in database for a given
        username
        """

        screen_name = f"@{screen_name}"
        cur = self.conn.cursor()

        try:
            cur.execute(
                """
                SELECT tweet_id FROM tweets
                WHERE handle=?
                ORDER BY tweet_id DESC""",
                (screen_name,),
            )

            return cur.fetchone()
        except sqlite3.Error as error:
            print(
                f"Error retrieving last tweet in db for {screen_name}\
                    \n {error}"
            )
        finally:
            cur.close()


if __name__ == "__main__":

    # db = Database("tweets_01.db")

    # print(db.sql_schema)
    pass
