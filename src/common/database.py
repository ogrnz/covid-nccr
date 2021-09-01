"""
Database module
"""

import sqlite3

from common.app import App
from common.helpers import Helpers


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
        self.prep_req = self.__prepare_req()

    def __enter__(self):
        self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def __prepare_req(self):
        """
        Generate a string with number of necessary "?" to prepare requests.
        returns str("(?,?,?)")
        """

        prep_req_tmp = tuple("?" for _ in range(len(Helpers.schema_cols)))
        return str(prep_req_tmp).replace("'", "")

    def __retrieve_schema(self):
        """
        Get sql tweets table schema
        """

        with open(f"{self.app.root_dir}/database/schema.sql", "r") as sql:
            sql_lines = sql.readlines()
        return "".join(sql_lines)

    def connect(self):
        """
        Establish connection to database
        """

        try:
            self.conn = sqlite3.connect(f"{self.app.root_dir}/database/{self.db_name}")
        except sqlite3.Error as error:
            print(f"Error establishing connection to database {self.db_name}\n", error)

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

    def get_all_tweets(
        self,
        condition: tuple = None,
    ):
        """
        Retrieve all tweets
        condition = tuple("condition_field", condition_value)
        """

        try:
            cur = self.conn.cursor()
            sql = "SELECT * FROM tweets"

            if condition:
                sql = f"SELECT * FROM tweets WHERE {condition[0]} = ?"

                cur.execute(
                    sql,
                    (condition[1],),
                )

            else:
                cur.execute(sql)

            return cur.fetchall()
        except sqlite3.Error as error:
            print("get_all_tweets: Error", error)
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
            print("get_fields: Error", error)
        finally:
            cur.close()

    def get_by_type(self, tweet_type: str):
        """
        Retrieve tweets by type
        :param type: available types are ['Retweet', 'New', 'Reply']
        """

        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM tweets WHERE type=?", (tweet_type,))

            return cur.fetchall()
        except sqlite3.Error as error:
            print("get_by_type: Error", error)
        finally:
            cur.close()

    def get_tweet_by_id(self, tweet_id):
        """
        Retrieve tweet by id
        """

        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM tweets WHERE tweet_id=?", (tweet_id,))

            return cur.fetchall()
        except sqlite3.Error as error:
            print("get_tweet_by_id: Error", error)
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
            print(f"update_tweet_by_id: Error updating tweet {tweet_id} ", error)
        finally:
            cur.close()

    def update_theme_many(self, params):
        """
        Update tweets in bulk
        BEWARE covid_theme comes first, tweet_id second
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
            print("update_theme_many: Error updating tweets", error)
        finally:
            cur.close()

    def update_many(self, fields, cond, values):
        """
        Update tweets in bulk

        If fields is a list, multiple fields will be updated.
        The condition value must be the last value of values.
        values can be an iterable such as tuple(field1_value, field2_value, cond_value)
        """

        if cond not in Helpers.schema_cols:
            print(
                "InvalidConditionError: This condition is not valid. \
                There is no column of that name."
            )

        if isinstance(fields, list):
            cols = str([field + " = ?" for field in fields])
            cols = cols.replace("[", "")
            cols = cols.replace("]", "")
            cols = cols.replace("'", "")
            sql = f"""UPDATE tweets SET {cols} WHERE {cond} = ?"""
        else:
            sql = f"""UPDATE tweets SET {fields} = ? WHERE {cond} = ?"""

        cur = self.conn.cursor()

        try:
            cur.executemany(
                sql,
                values,
            )
            self.conn.commit()

            return cur.rowcount
        except sqlite3.Error as error:
            print("update_many: Error updating tweets", error)
        finally:
            cur.close()

    def insert_tweet(self, tweet):
        """
        Insert new tweet into database
        """

        sql = f"""INSERT OR IGNORE INTO tweets{Helpers.get_cols_as_tuple_str()}
                VALUES{self.prep_req}"""
        cur = self.conn.cursor()

        try:
            cur.execute(sql, tweet)
            self.conn.commit()

            return cur.lastrowid
        except sqlite3.Error as error:
            print(sql)
            print(f"insert_tweet: Error inserting new tweet \n {tweet} \n {error}")
        finally:
            cur.close()

    def insert_many(self, tweets):
        """
        Insert new tweets into database.
        Tweets already in the database are not inserted.
        """

        sql = f"""INSERT OR IGNORE INTO tweets
                  VALUES{self.prep_req}"""
        cur = self.conn.cursor()

        try:
            cur.executemany(sql, tweets)
            self.conn.commit()

            return cur.rowcount
        except sqlite3.Error as error:
            print(f"insert_many: Error inserting new tweets \n {error}")
        finally:
            cur.close()

    def insert_or_replace_many(self, tweets):
        """
        Insert new tweets into database. If tweet already exists, values are overwritten.
        """

        sql = f"""INSERT OR REPLACE INTO tweets
                  VALUES{self.prep_req}"""
        cur = self.conn.cursor()

        try:
            cur.executemany(sql, tweets)
            self.conn.commit()

            return cur.rowcount
        except sqlite3.Error as error:
            print(f"insert_or_replace_many: Error inserting new tweets \n {error}")
        finally:
            cur.close()

    def get_last_id_by_handle(self, screen_name):
        """
        Retrieve last inserted tweet id in database for a given
        username.
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
                f"get_last_id_by_handle: Error retrieving last tweet in db for {screen_name}\
                    \n {error}"
            )
        finally:
            cur.close()

    def delete_by_id(self, tweet_id):
        """
        Remove entry from database by id.
        If tweet_id is a list, remove multiple entries.
        """

        try:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM tweets WHERE tweet_id = ?", (tweet_id,))
            self.conn.commit()
        except sqlite3.Error as error:
            print("delete_by_id: Error", error)
        finally:
            cur.close()


if __name__ == "__main__":

    # db = Database("tweets_01.db")

    # print(db.sql_schema)
    pass
