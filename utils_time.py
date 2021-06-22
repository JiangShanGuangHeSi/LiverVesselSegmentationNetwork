import time

def get_time_str():
    # 格式化成2016-03-20 11:45:39形式
    # return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 

if __name__ == '__main__':
    print(get_time_str())