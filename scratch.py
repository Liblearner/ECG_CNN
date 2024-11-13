"""
进行输入权重与测试数据的量化，量化为0-127范围后在FPGA上进行运算
"""

import csv

def float_to_binary(float_num):
    # 将-1到1的浮点数映射到0到127的整数范围
    mapped_value = int((float_num + 1) * 63.5)  # 63.5 是将127除以2的结果
    if mapped_value < 0:
        mapped_value = 0
    elif mapped_value > 127:
        mapped_value = 127

    # 将整数转换为7位二进制数
    binary_str = format(mapped_value, '07b')

    # 决定符号位
    if float_num < 0:
        binary_str = '1' + binary_str
    else:
        binary_str = '0' + binary_str

    return binary_str


# # 读取 Excel 文件
# wb = openpyxl.load_workbook('weight/conv1_w.csv')
# sheet = wb.active
#
# wb1 = openpyxl.load_workbook('weight/conv2_w.csv')
# sheet1 = wb1.active
#
# wb2 = openpyxl.load_workbook('weight/conv3_w.csv')
# sheet2 = wb2.active
#
# wb3 = openpyxl.load_workbook('weight/fc_w.csv')
# sheet3 = wb3.active

with open('weight/conv1_w.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    with open('weight/conv1_w.txt','w') as txtfile:
        for row in reader:
            line = ''.join(float_to_binary(float(row[0])))
            txtfile.write(line+'\n')
print("Writing conv1_w done \n")

with open('weight/conv1_b.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    with open('weight/conv1_b.txt','w') as txtfile:
        for row in reader:
            line = ''.join(float_to_binary(float(row[0])))
            txtfile.write(line+'\n')
print("Writing conv1_b done \n")


with open('weight/conv2_w.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    with open('weight/conv2_w.txt','w') as txtfile:
        for row in reader:
            line = ''.join(float_to_binary(float(row[0])))
            txtfile.write(line+'\n')
print("Writing conv2_w done \n")

with open('weight/conv2_b.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    with open('weight/conv2_b.txt','w') as txtfile:
        for row in reader:
            line = ''.join(float_to_binary(float(row[0])))
            txtfile.write(line+'\n')
print("Writing conv2_b done \n")

with open('weight/conv3_w.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    with open('weight/conv3_w.txt','w') as txtfile:
        for row in reader:
            line = ''.join(float_to_binary(float(row[0])))
            txtfile.write(line+'\n')
print("Writing conv3_w done \n")

with open('weight/conv3_b.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    with open('weight/conv3_b.txt','w') as txtfile:
        for row in reader:
            line = ''.join(float_to_binary(float(row[0])))
            txtfile.write(line+'\n')
print("Writing conv3_b done \n")

with open('weight/fc_w.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    with open('weight/fc_w.txt','w') as txtfile:
        for row in reader:
            line = ''.join(float_to_binary(float(row[0])))
            txtfile.write(line+'\n')
print("Writing fc_w done \n")

with open('weight/fc_b.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    with open('weight/fc_b.txt','w') as txtfile:
        for row in reader:
            line = ''.join(float_to_binary(float(row[0])))
            txtfile.write(line+'\n')
print("Writing fc_b done \n")

with open('dataset/NAVLR_rawdata_310.csv', 'r') as csv_file,  open('test_input/test_input1.txt', 'w') as txtfile:
    reader = csv.reader(csv_file)
    for i,row in enumerate(reader):
        if i >65:
            break
        if i>0:
            for data in row:
                try:
                    data_int = int((float(data)+1)*63.5)
                except ValueError:
                    data_int = 0
                data_int = max(0, min(data_int, 127))
                binary_data = format(data_int, '07b')
                if float(data) < 0:
                    binary_data = '1' + binary_data
                else:
                    binary_data = '0' + binary_data
                txtfile.write(binary_data + ' ')
            txtfile.write("\n")
print("Writing test_input1 done \n")


# # 创建并打开文本文件以写入二进制数据
# with open('weight/conv1_w.txt', 'w') as f:
#     # 遍历 Excel 表格的每一行和每一列
#     for row in sheet.iter_rows(min_row=1, max_row=32, min_col=1, max_col=4, values_only=True):
#         binary_row = []  # 存储当前行的二进制数据
#         for value in row:
#             # 将 Excel 单元格中的数据转换为浮点数并进行转换为二进制
#             binary_representation = float_to_binary(value)
#             binary_row.append(binary_representation)  # 将二进制数据添加到当前行
#
#             # 每4个数据一组，组成一行
#             if len(binary_row) == 4:
#                 # 将4个数据连接起来，用空格隔开
#                 line = ' '.join(binary_row)
#                 # 写入到文件并换行
#                 f.write(line + '\n')
#                 binary_row = []  # 重置当前行的二进制数据列表
#
#         # 处理剩余不足4个数据的情况
#         if binary_row:
#             # 将剩余的数据连接起来，用空格隔开
#             line = ' '.join(binary_row)
#             # 写入到文件并换行
#             f.write(line + '\n')
#
# print("Writing Process 1 is over")
#
#
# with open('weight/conv2_w.txt', 'w') as f:
#     # 遍历 Excel 表格的每一行和每一列
#     for row in sheet1.iter_rows(min_row=1, max_row=512, min_col=1, max_col=4, values_only=True):
#         binary_row = []  # 存储当前行的二进制数据
#         for value in row:
#             # 将 Excel 单元格中的数据转换为浮点数并进行转换为二进制
#             binary_representation = float_to_binary(value)
#             binary_row.append(binary_representation)  # 将二进制数据添加到当前行
#
#             # 每4个数据一组，组成一行
#             if len(binary_row) == 4:
#                 # 将4个数据连接起来，用空格隔开
#                 line = ' '.join(binary_row)
#                 # 写入到文件并换行
#                 f.write(line + '\n')
#                 binary_row = []  # 重置当前行的二进制数据列表
#
#         # 处理剩余不足4个数据的情况
#         if binary_row:
#             # 将剩余的数据连接起来，用空格隔开
#             line = ' '.join(binary_row)
#             # 写入到文件并换行
#             f.write(line + '\n')
#
# print("Writing Process 2 is over")
#
# with open('weight/conv3_w.txt', 'w') as f:
#     # 遍历 Excel 表格的每一行和每一列
#     for row in sheet2.iter_rows(min_row=1, max_row=2048, min_col=1, max_col=4, values_only=True):
#         binary_row = []  # 存储当前行的二进制数据
#         for value in row:
#             # 将 Excel 单元格中的数据转换为浮点数并进行转换为二进制
#             binary_representation = float_to_binary(value)
#             binary_row.append(binary_representation)  # 将二进制数据添加到当前行
#
#             # 每4个数据一组，组成一行
#             if len(binary_row) == 4:
#                 # 将4个数据连接起来，用空格隔开
#                 line = ' '.join(binary_row)
#                 # 写入到文件并换行
#                 f.write(line + '\n')
#                 binary_row = []  # 重置当前行的二进制数据列表
#
#         # 处理剩余不足4个数据的情况
#         if binary_row:
#             # 将剩余的数据连接起来，用空格隔开
#             line = ' '.join(binary_row)
#             # 写入到文件并换行
#             f.write(line + '\n')
#
# print("Writing Process 3 is over")
#
#
#
# with open('test_input/weight/fc_w.txt', 'w') as f:
#     # 遍历 Excel 表格的每一行和每一列
#     for row in sheet3.iter_rows(min_row=1793, max_row=2048, min_col=1, max_col=300, values_only=True):
#         binary_row = []  # 存储当前行的二进制数据
#         for value in row:
#             # 将 Excel 单元格中的数据转换为浮点数并进行转换为二进制
#             binary_representation = float_to_binary(value)
#             binary_row.append(binary_representation)  # 将二进制数据添加到当前行
#
#             # 每4个数据一组，组成一行
#             if len(binary_row) == 30:
#                 # 将4个数据连接起来，用空格隔开
#                 line = ''.join(binary_row)
#                 # 写入到文件并换行
#                 f.write(line + '\n')
#                 binary_row = []  # 重置当前行的二进制数据列表
#
#         # 处理剩余不足4个数据的情况
#         if binary_row:
#             # 将剩余的数据连接起来，用空格隔开
#             line = ''.join(binary_row)
#             # 写入到文件并换行
#             f.write(line + '\n')
#
# print("Writing Process 4 is over")