import pymysql
from sqlalchemy import create_engine
import sqlalchemy



engine = create_engine(
      "mysql+pymysql://USERNAME:PASSWORD@HOSTNAME/databasename?charset=utf8mb4")

con = engine.connect()




