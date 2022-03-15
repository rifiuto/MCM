import csv
import pymysql
# with open("players_stats3.csv", "r") as file:
#     reader = csv.reader(file)
#     next(reader)
#     for i in reader:
#         print(i)
db = pymysql.connect(host="localhost",
                     user="root",
                     password="271xufei.",
                     db="test",)
cursor = db.cursor()
cursor.execute("show tables");
print(cursor.fetchall())
cursor.execute("create table if not exists hh("
               "id int,"
               "name varchar(4))")
cursor.execute(f"insert table hh values({'1'}, {'chae'})")
#
# with open("players_stats3.csv", "r") as file:
#     teams = {}
#     reader = csv.DictReader(file)
#     for item in reader:
#         team = item['Team']
#         # if team in teams:
#         #     teams[team] += 1
#         # else:
#         #     teams[team] = 1
#         if team not in teams:
#             teams[team] = 0
#         teams[team] += 1
#     for team in sorted(teams, key=lambda x: teams[x], reverse=True):
#         print(team, teams[team])
#
# print("b" in {"a": "b"})
# print("a" in {"a": "b"})
