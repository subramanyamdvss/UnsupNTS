import argparse
import sys
from fuzzywuzzy import fuzz

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='edit distance')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    parser.add_argument('-src','--source',default="",help='source file with which edit distance should be calculated.')
    args = parser.parse_args()
    infile = open(args.input,'r')
    srcfile = open(args.source,'r')
    inplines = [ln.strip() for ln in infile]
    srclines = [ln.strip() for ln in srcfile]
    ratio = 0
    n = 0
    for i in range(len(inplines)):
        try:
            ratio+=fuzz.ratio(inplines[i],srclines[i])/100
            n+=1
        except Exception:
            continue
    ratio/=n
    print("edit distance between {} and {} is: {:.4f}".format(args.input.split("/")[-1],args.source.split("/")[-1],ratio))

if __name__ == '__main__':
    main()