
import csv


def read_csv(file_path,header=False):
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # 读取表头（如果有）
        headers = next(csv_reader)
        if header:
            data.append(headers)
        # 读取数据行
        for row in csv_reader:
            data.append(row)
            # print("data,",data)
    return data

# file_path = "/home/aa/Desktop/WJL/VTRAG/Data/ssvtp/train.csv"
# read_csv(file_path)