import os
import re

def mkdir(path):
    '''
    创建目录
    ————————————————
    版权声明：本文为CSDN博主「FanWinter」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/MuWinter/article/details/77215768
    '''
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    path=path.rstrip("/")
    # 判断路径是否存在
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        '''
        os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
        '''
        #此处路径最好使用utf-8解码，否则在磁盘中可能会出现乱码的情况
        # os.makedirs(path.decode('utf-8')) 
        os.makedirs(path) 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False

def get_filename_number(filename, num):
    # 提取文件名中的编号
    return int(re.findall("\d+",filename)[num]) 

def get_filename_info(filefullname):
    # 提取文件名的文件夹路径、文件名、文件短名称、文件扩展名
    (folderpath,filename) = os.path.split(filefullname)
    (shotname,extension) = os.path.splitext(filename) # 文件名和后缀
    return folderpath, filename, shotname, extension