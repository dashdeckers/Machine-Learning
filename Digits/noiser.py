import sys
from random import seed
from random import gauss

if __name__ == '__main__':
    if(len(sys.argv) < 1):
        print("Please specify a filesource")
    filename = sys.argv[1]
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    seed(1)
    output = str()

    #pool = mp.Pool(mp.cpu_count())
    for line in lines:
        for digit in line.split():
            if digit.isdigit():
                value = int(digit)
                value = value +gauss(0,1)
                if(value > 6):
                    value = 6
                if(value < 0):
                    value = 0
                output = output + str(value) + " "
        output = output + "\n"
    print(output)

    file = open("Copyof_"+ filename, 'w')
    file.write(output)
    file.close()


