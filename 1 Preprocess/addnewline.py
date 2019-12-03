import os


if __name__=='__main__':
    path = 'F:\\aclImdb_v1\\IMDB'
    #path = 'F:\\aclImdb_v1\\IMDB\\LJY'
    orifile = 'neg\\neg_total.txt'
    newfile = 'neg\\neg_rmpunc.txt'

    orif = open(path + '\\' + orifile,'r', encoding='utf-8')
    newf = open(path + '\\' + newfile,'w', encoding='utf-8')
    sent_list = []
    lines = orif.readlines()
    for line in lines:
        line = line.replace('.' , '\n')
        sent_list.append(line)

    for sent in sent_list:
        newf.write(sent)

    orif.close()
    newf.close()



