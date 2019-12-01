'''
Created on 1 Dec 2019

this code creates the list of all
vocal and non-vocal stems in the 
ccMixter dataset.

@author: Saumitra
'''

import os

dir_path = '../deep_inversion/datasets/ccMixter/audio'
res_path = '../deep_inversion/datasets/ccMixter/'
fn = []
for path, subdirs, files in os.walk(dir_path):
    for name in files:
        print os.path.join(path, name)
        temp = os.path.join(path, name)
        fn.append(temp)

print fn

fn2 = []
for ele in fn:
    a = ele.split('/')
    if 'source-01.wav' in a or 'source-02.wav' in a:
        fn2.append(ele)
    else:
        pass
    
print fn2
print "length:%d"%(len(fn2))


fn3= []
for ele in fn2:
    x = ele.split("/")
    x1 = x[-2] + '/' + x[-1]
    fn3.append(x1)
    
print fn3
print "length:%d" %(len(fn3))

with open(res_path+'test', 'w') as fp:
    for ele in fn3:
        fp.write("%s\n" %ele)