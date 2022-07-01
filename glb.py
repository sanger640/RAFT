# ignore file, for experimentation purpose
import glob
import os
#counter = 0
for name in glob.glob('/road/rgb-images/*'):
    x = os.path.basename(name)
    os.mkdir('/road/opf/'+ x)
    #for file in  glob.glob(name+'/*'):
    #    print(file)
    break
    #break