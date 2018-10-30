#out = open("2dlabels.txt", 'w')
o = open("2dDataTest.txt", 'w')


with open("2dPoints.txt") as f:
    lines = f.readlines()
    for line in lines:
        o.write(line.rstrip() + "\n")
        '''if "Y" in line:
            out.write("1.0\n")
        else:
            out.write("0.0\n")
       '''
        
#out.close()
o.close()