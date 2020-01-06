import sys
import multiprocessing as mp
from random import seed
from random import gauss

def noise_line(arg):
    (line, spread, copies) = arg
    output = str()
    for i in range(0,copies):
        for digit in line.split():
            if digit.isdigit():
                value = int(digit)
                value = value +gauss(0,spread)
                if(value > 6):
                    value = 6
                if(value < 0):
                    value = 0
                output = output + str(value) + " "
        output = output + "\n"
    return output

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("Please specify parameters spread(std. deviation of noise) and copies(number of replicas to be created)")
    spread = float(sys.argv[1])
    copies = int(sys.argv[2])
    file = open('traindata/traindata.txt', 'r')
    lines = file.readlines()
    file.close()

    seed()
    output = str()
    args = [(line, spread, copies) for line in lines]
    pool = mp.Pool(mp.cpu_count())
    output = pool.map(noise_line, args)
    output = ''.join(map(str, output))

    file = open("traindata/traindata_s_"+ str(spread) + '_c_' + str(copies) + ".txt", 'w')
    file.write(output)
    file.close()


