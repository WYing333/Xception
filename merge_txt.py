import os
import os.path

#normal merge
filedir = '/home/ywang/txt_files'  # 填入要合并的文件夹名字
filenames = os.listdir(filedir)  # 获取文件夹内每个文件的名字
f = open('/home/ywang/all.txt', 'w')  # 以写的方式打开文件，没有则创建

# 对每个文件进行遍历
for filename in filenames:
    filepath = filedir + '/' + filename  # 将文件夹路径和文件名字合并
    for line in open(filepath):  # 循环遍历对每一个文件内的数据
        f.writelines(line)  # 将数据每次按行写入f打开的文件中

f.close()  # 关闭


# # original*2
# filedir = '/home/ywang/txt_files'  # 填入要合并的文件夹名字
# filenames = os.listdir(filedir)  # 获取文件夹内每个文件的名字
# keyword= "org"

# f = open('/home/ywang/all.txt', 'a')  # 继续写

# # 对每个文件进行遍历
# for filename in filenames:
#     if keyword in filename.split('_')[0]:
#         filepath = filedir + '/' + filename  # 将文件夹路径和文件名字合并
#         for line in open(filepath):  # 循环遍历对每一个文件内的数据
#             f.writelines(line)  # 将数据每次按行写入f打开的文件中
#             f.writelines(line)  #写2遍
# f.close()  # 关闭
