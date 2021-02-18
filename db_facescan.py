import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    user='root',
    password='mySQL_PASS_1234',
    auth_plugin='mysql_native_password',
    database='facialdb'
)
print(mydb)
my_cursor = mydb.cursor()


def create_db():
    my_cursor.execute("CREATE DATABASE facialdb")

    my_cursor.execute("SHOW DATABASES")
    print(my_cursor)
    for db in my_cursor:
        print(db)

def create_table():

    my_cursor.execute("CREATE TABLE tblresults \
                      (id int(11) NOT NULL,\
                        resultNo varchar(50) NOT NULL, \
                      category varchar(50) NOT NULL, \
                      sub_category varchar(50) NOT NULL, \
                      value double(15,2) NOT NULL, \
                      sub_value double(15,2) NOT NULL, \
                      created_at timestamp NULL DEFAULT current_timestamp())")


def insert_db(time_now,p_dict, w_dict, l_dict):
    client_num = 'client-'+time_now

    sql = "INSERT INTO tblresults " \
          "(id, resultNo, category, sub_category, value, sub_value, created_at) " \
          "VALUES (%s, %s, %s, %s, %s, %s, %s)"

    val = [
            (str(1), client_num,'Pores', 'Upper', str(p_dict['pores']['upper_part']),
                str(0), time_now),
            (2, client_num,'Pores', 'Middle', p_dict['pores']['middle_part'],
                0, time_now),
           (3, client_num, 'Pores', 'Lower', p_dict['pores']['lower_part'],
            0, time_now),
           (4, client_num, 'Wrinkles', 'Upper', w_dict['wrinkles']['upper_part'],
            0, time_now),
           (5, client_num, 'Wrinkles', 'Middle', w_dict['wrinkles']['middle_part'],
            0, time_now),
           (6, client_num, 'Wrinkles', 'Lower', w_dict['wrinkles']['lower_part'],
            0, time_now),
           (7, client_num, 'Moisture', 'Lips', l_dict['moisture']['lips'],
            0, time_now),
           ]
    my_cursor.executemany(sql, val)

    mydb.commit()

    print(my_cursor.rowcount, "record inserted.")
