import sqlite3
from contextlib import closing

dbname = "database.db"

# データベースと接続
with closing(sqlite3.connect(dbname)) as conn:
    # カーソルオブジェクト生成
    c = conn.cursor()

    # sql文
    create_table = "create table users (id int, name varchar(64), age int, gender varchar(32))"
    # 実行
    c.execute(create_table)

    sql = "insert into users (id, name, age, gender) values (?,?,?,?)"
    users = [
        (1, "kazuki", 21, "male"),
        (2, "layla", 30, "female")
    ]
    c.executemany(sql, users)

    # 変更の保存に必要
    conn.commit()

    select_sql = "select * from users"
    for row in c.execute(select_sql):
        print(row)
