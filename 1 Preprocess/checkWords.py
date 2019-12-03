import string
import enchant


def checkLines(line):
    ''' replace useless character and check word spell
    Args:
        line: string
    Returns:
        string, right words  with 1 space
    '''
    # remove punctuation
    punctuationTable = dict.fromkeys(
        [ord(x) for x in string.punctuation], ' ')  # generate punctuation table
    '''punctuation table
    {
        key:value,
        be replaced: replace,
        ord('<'):' ' 

    }

    '''
    # dont reomve '  caz: words like  it's are right
    punctuationTable.pop(ord("'"))
    punctuationTable.pop(ord("-"))
    line = line.translate(punctuationTable)

    numericalTable = dict.fromkeys(
        [ord(x) for x in string.digits], ''
    )
    line = line.translate(numericalTable)

    # remove Cons or Pros
    line = line.replace('Cons', ' ')
    line = line.replace('Pros', ' ')

    # check whether the item is a word
    wordDict = enchant.Dict("en_US")
    res = ''
    for item in line.split(' '):
        item = item.strip("'")
        item = item.lower()
        if item:
            # remove ' of 2 sides  eg. 'smart' => smart
            if wordDict.check(item):
                res += item + ' '
            elif '-' in item:  # if  words connected with -
                flag = True
                for word in item.split('-'):
                    if word:
                        if wordDict.check(word):
                            continue
                        else:
                            flag = False
                            break
                if flag: res += item + ' '

    return res.strip()


def wirteRes(line):
    ''' write reuslt in file
    Args:
        line: string
    '''
    with open(f'{path}Final-{filename}', 'a') as f:  # f表达式
        f.write(line)
    print(line)


def main(file, row_sign):
    '''run the code
    Args:
        file: string, file path
        row_sign: string, '1' or '-1'
    '''
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = checkLines(line)
            if line:  # if line not empty  空值可以当作布尔型
                line = f'{row_sign}\t{line}\n'
                wirteRes(line)


if __name__ == "__main__":
    path = "/Users/a839/Desktop/"
    filename = "IntegratedPros.txt"
    main(path + filename, "1")
