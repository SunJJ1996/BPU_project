import os

path = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/image_google'
fileList = os.listdir(path)
n = 0
o = 12284
k = 0
a = 0
b = 0
for i in fileList:
    if i[-3:] == 'jpg':
        n += 1
        m = str(n)
        oldname = path + os.sep + i
        newname = path + os.sep + 'image_google_' + m + '.jpg'
        os.rename(oldname, newname)
        print(oldname, '======>', newname)
        a += 1
    if i[-3:] == 'txt':
        o += 1
        m = str(o)
        oldname = path + os.sep + i
        newname = path + os.sep + m + '.txt'
        os.rename(oldname, newname)
        print(oldname, '======>', newname)
        b += 1
print(a, b)
