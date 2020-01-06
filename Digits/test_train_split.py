import os


if __name__ == '__main__':
    file = open('mfeat-pix.txt', 'r')
    lines = file.readlines()
    file.close()

    test = [lines[i] for i in range(0, len(lines), 2)]
    train = [lines[i] for i in range(1, len(lines), 2)]
    print(len(test))
    print(len(train))

    test = '\n'.join(map(str, test))
    train = '\n'.join(map(str, train))

    try:
        os.mkdir('testdata')
    except OSError:
        print("Did not make testdata dir, perhaps it already exists")
    try:
        os.mkdir('traindata')
    except OSError:
        print("Did not make traindata dir, perhaps it already exists")

    file = open('testdata/testdata.txt', 'w')
    file.write(test)
    file.close()

    file = open('traindata/traindata.txt', 'w')
    file.write(train)
    file.close()
