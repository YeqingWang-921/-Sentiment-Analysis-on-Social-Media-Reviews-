if __name__=='__main__':
    '''
    reduce the original file by half
    '''
    #path = 'F:\\aclImdb_v1\\pros-cons'
    #path = 'F:\\aclImdb_v1\\IMDB\\LJY'
    orifile = 'F:\\aclImdb_v1\\LY\\ProsCons.txt'
    newfile = 'F:\\aclImdb_v1\\LY\\newProsCons.txt'

    orif = open(orifile,'r', encoding='utf-8')
    newf = open(newfile,'w', encoding='utf-8')
    lines=[]
    readlines = orif.readlines()
    for line in readlines:
        lines.append(line)
    #random.shuffle(lines)

    count = 0
    for line in lines:
        if (count%2 == 0 ):
            newf.write(line)
        count += 1
    orif.close()
    newf.close()