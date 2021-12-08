import numpy as np

import mysql.connector

db = mysql.connector.connect(host="localhost", user="root", passwd="123456", database="SCHOOL")  # 连接到数据库
cursor = db.cursor()  # 获取游标


def saveDataBase(table_name, datalist):  # 保存数据到数据库
     for data in datalist:
          if data != ['']:
               try:
                    sql = 'INSERT INTO ' + table_name + ' VALUES (' + str(data) + ');'
                    print(sql)
                    cursor.execute(sql)  # 插入数据
                    db.commit()
               except:
                    pass

def saveDataBase1(table_name, datalist):  # 保存数据到数据库
     for data in datalist:
          if data != ['']:
               sql = 'INSERT INTO ' + table_name + '(ID,DATA) VALUES ' + str(data) + ';'
               print(sql)
               cursor.execute(sql)  # 插入数据
               db.commit()


s = ["'1101','李明',1,'1993-03-06','上海','13613005486','02'",
     "'1102','刘晓明',1,'1992-12-08','安徽','18913457890','01'",
     "'1103','张颖',0,'1993-01-05','江苏','18826490423','01'",
     "'1104','刘晶晶',0,'1994-11-06','上海','13331934111','01'",
     "'1105','刘成刚',1,'1991-06-07','上海','18015872567','01'",
     "'1106','李二丽',0,'1993-05-04','江苏','18107620945','01'",
     "'1107','张晓峰',1,'1992-08-16','浙江','13912341078','01'"]

d = ["'01','计算机学院','上大东校区三号楼','65347567'",
     "'02','通讯学院','上大东校区二号楼','65341234'",
     "'03','材料学院','上大东校区四号楼','65347890'"]

t = ["'0101','陈迪茂',1,'1973-03-06','副教授',3567.00,'01'",
     "'0102','马小红',0,'1972-12-08','讲师',2845.00,'01'",
     "'0201','张心颖',0,'1960-01-05','教授',4200.00,'02'",
     "'0103','吴宝钢',1,'1980-11-06','讲师',2554.00,'01'"]

c = ["'08305001','离散数学',4,40,'01'",
     "'08305002','数据库原理',4,50,'01'",
     "'08305003','数据结构',4,50,'01'",
     "'08305004','系统结构',6,60,'01'",
     "'08301001','分子物理学',4,40,'03'",
     "'08302001','通信学',3,30,'02'"]

o = ["'2012-2013秋季','08305001','0103','星期三5-8'",
     "'2012-2013冬季','08305002','0101','星期三1-4'",
     "'2012-2013冬季','08305002','0102','星期三1-4'",
     "'2012-2013冬季','08305002','0103','星期三1-4'",
     "'2012-2013冬季','08305003','0102','星期五5-8'",
     "'2013-2014秋季','08305004','0101','星期二1-4'",
     "'2013-2014秋季','08305001','0102','星期一5-8'",
     "'2013-2014冬季','08302001','0201','星期一5-8'"]

e = ["'1101','2012-2013秋季','08305001','0103',60,60,60",
     "'1102','2012-2013秋季','08305001','0103',87,87,87",
     "'1102','2012-2013冬季','08305002','0101',82,82,82",
     "'1102','2013-2014秋季','08305004','0101',null,null,null",
     "'1103','2012-2013秋季','08305001','0103',56,56,56",
     "'1103','2012-2013冬季','08305002','0102',75,75,75",
     "'1103','2012-2013冬季','08305003','0102',84,84,84",
     "'1103','2013-2014秋季','08305001','0102',null,null,null",
     "'1103','2013-2014秋季','08305004','0101',null,null,null",
     "'1104','2012-2013秋季','08305001','0103',74,74,74",
     "'1104','2013-2014冬季','08302001','0201',null,null,null",
     "'1106','2012-2013秋季','08305001','0103',85,85,85",
     "'1106','2012-2013冬季','08305002','0103',66,66,66",
     "'1107','2012-2013秋季','08305001','0103',90,90,90",
     "'1107','2012-2013冬季','08305003','0102',79,79,79",
     "'1107','2013-2014秋季','08305004','0101',null,null,null"]

# saveDataBase('D', d)
# saveDataBase('S', s)
# saveDataBase('T', t)
# saveDataBase('C', c)
# saveDataBase('O', o)
# saveDataBase('E', e)

q = []

for i in range(10000):
     q = []
     print(i/10000)
     sql = ""
     for j in range(1000):
          if j > 0:
               sql = sql + ','
          sql = sql + "('" + str(1000*i + j) + "','" + str(np.random.randint(1, 10000000)) + "')"
     q.append(sql)
     saveDataBase1('TEST', q)

db.close()  # 关闭数据库