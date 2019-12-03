# -*- coding:utf-8 -*-
import glob

txt_filenames = glob.glob('F:\\aclImdb_v1\\aclImdb\\train\\pos\\*.txt')
txt_te_pos = open("F:\\aclImdb_v1\\aclImdb\\train\\trainpos_total.txt", "a", encoding='UTF-8')
for filename in txt_filenames:
    txt_file = open(filename, 'r', encoding='UTF-8')
    buf = txt_file.read()  # the context of txt file saved to buf
    txt_te_pos.write(buf + "\n")


print("done!")
txt_file.close()
txt_te_pos.close()