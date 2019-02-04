from data_loader import *


def create_file(sen1, sen2):
    file = open("testie.txt", 'w')
    file.write("header line")
    for i in range(257):
        file.write("1\txx\txx\t{}\t{}\n".format(sen1, sen2))


if __name__ == '__main__':
    create_file("my name is shree", "they call me shree ")
