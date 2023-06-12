import pymysql

# 创建数据库连接
conn = pymysql.connect(
    host = 'localhost', # 连接主机, 默认127.0.0.1
    user = 'root',      # 用户名
    passwd = 'root',    # 密码
    port = 3306,        # 端口，默认为3306
    db = 'modis',       # 数据库名称
    charset = 'utf8'    # 字符编码
)

# 生成游标对象 cursor
cursor = conn.cursor()

# 查询数据库版本
cursor.execute("select version()") # 返回值是查询到的数据数量
# 通过 fetchall方法获得数据
data = cursor.fetchone()
print("Database Version: %s" % data)

cursor.close()  # 关闭游标
conn.close()    # 关闭连接
