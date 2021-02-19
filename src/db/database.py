import sqlite3

class Database:
    """
    Class that handles database queries
    """     

    def __init__(self, db_name: str):
        """Initialize object"""

        self.path = f'../../database/{db_name}'
        self.connect_sqlite()

    def __del__(self):
        if self.conn:
            self.conn.close()

    def create_table(self, table_sql):
        try:
            c = self.conn.cursor()
            c.execute(table_sql)
        except sqlite3.Error as e:
            print('Error creating table', e)

    def connect_sqlite(self):
        """Establish connection to database"""

        self.conn = None
        try:
            self.conn = sqlite3.connect(self.path)
        except sqlite3.Error as e:
            print(e)

    def get_by_type(self, type: str):
        """
        Retrieve tweets by type
        :param type: available types are ['Retweet','New', 'Reply']
        """

        try:
            cur = self.conn.cursor()
            cur.execute(f"SELECT * FROM tweets WHERE type=?", (type,))
            
            return cur.fetchall()   
        except sqlite3.Error as e:
            print('Error', e)
        finally:
            cur.close()

    def update_tweet_by_id(self, id: int, field: str, content: any):
        """
        Update tweet in database by id
        """

        sql = f'''UPDATE tweets
             SET {field} = ?
             WHERE tweet_id = ?'''
        cur = self.conn.cursor()

        try:
            cur.execute(sql, (content, id,))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f'Error updating tweet {id} ', e)
        finally:
            cur.close()

    def insert_tweet(self, tweet):
        """
        Insert new tweet into database
        """
        sql = ''' INSERT OR IGNORE INTO tweets(
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
                        favorites)
                VALUES(?,?,?,?,?,?,?,?,?,?,?) '''
        cur = self.conn.cursor()

        try:
            cur.execute(sql, tweet)
            self.conn.commit()

            return cur.lastrowid
        except sqlite3.Error as e:
            print(f'Error inserting new tweet \n {tweet} \n {e}')
        finally:
            cur.close()

    def get_last_id_by_handle(self, screen_name):
        """
        Retrieve last inserted tweet id in database for a given
        username
        """

        screen_name = f'@{screen_name}'
        cur = self.conn.cursor()

        try:
            cur.execute('''
                SELECT tweet_id 
                FROM tweets 
                WHERE handle=?
                ORDER BY tweet_id DESC''', 
                (screen_name,)
            )

            return cur.fetchone()
        except sqlite3.Error as e:
            print(f'Error retrieving last tweet in db for {screen_name}\
                    \n {e}')
        finally:
            cur.close()

if __name__ == "__main__":
    """Some tests"""

    db = Database('tweets-tests.db')

    print(len(db.get_by_type('Reply')))

    db.update_tweet_by_id(1341331429100294144, 'text', None)
    #id doesn't exist, why no error raised?

    tweet = (123, 0, 'New', '19/02/2021 17:00:01', '@Fake', 'Fake', None, 'tweet', 'google.com', 0, 100)
    db.insert_tweet(tweet)

    print(db.get_last_id_by_handle('EU_Commission'))

    del db