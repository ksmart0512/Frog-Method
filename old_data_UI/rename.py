import os
files = os.listdir('.')
files = [f for f in files if '.txt' in f]
for f in files:
 root = [i for i in f.split('.')[0] if not i.isdigit()]
 num = ''.join([i for i in f.split('.')[0] if i.isdigit()])
 frog = ''.join(root)
 os.rename(f, "%s%s.txt"%(frog,num.zfill(2)))

