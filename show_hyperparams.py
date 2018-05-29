from pprint import pprint
import ast

with open('hyperparam-scores.txt', 'r') as infile:
    content = infile.read()
content = content.replace(' ', '').replace('\n','')
tups = eval(content)
tups.sort(key=lambda tup: tup[0])
pprint(tups)

