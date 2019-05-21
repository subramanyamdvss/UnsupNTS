import argparse
import sys
def main():
    parser = argparse.ArgumentParser(description='noredund')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    args = parser.parse_args()
    fl = open(args.input)
    for sent in fl:
        lsent = sent.strip().split()
        if len(lsent) == 0:
            print("")
            continue
        lst = [lsent[0]]
        prev_word = lsent[0]
        for word in lsent:
            if word==prev_word:
                prev_word=word
            else:
                lst.append(word)
                prev_word=word
        print(' '.join(lst))
if __name__ == '__main__':
    main()