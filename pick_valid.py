import os
import shutil
import sys

path = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/BPUtrain200706/image_train/'
out = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/BPUtrain200706/image_valid'
fileList = os.listdir(path)
num_jpg = 0
num_txt = 0




for i in fileList:
    if i[-3:] == 'jpg':

        if i[:8] == 'BUPvideo':
            if i.count("_") == 2:
                num_jpg += 1
                try:
                    shutil.copy(path+i, out)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())
                if os.path.exists(path+i):  # 如果文件存在
                    # 删除文件，可使用以下两种方法。
                    os.remove(path+i)
                    # os.unlink(path)
                else:
                    print('no such file'+i)
        if i[:12] == 'image_google':
            if i.count("_") == 2:
                num_jpg += 1
                try:
                    shutil.copy(path+i, out)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())
                if os.path.exists(path+i):  # 如果文件存在
                    # 删除文件，可使用以下两种方法。
                    os.remove(path+i)
                    # os.unlink(path)
                else:
                    print('no such file'+i)

    if i[-3:] == 'txt':
        if i[:8] == 'BUPvideo':
            if i.count("_") == 2:
                num_txt += 1
                try:
                    shutil.copy(path + i, out)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())
                if os.path.exists(path+i):  # 如果文件存在
                    # 删除文件，可使用以下两种方法。
                    os.remove(path+i)
                    # os.unlink(path)
                else:
                    print('no such file'+i)
        if i[:12] == 'image_google':
            if i.count("_") == 2:
                num_txt += 1
                try:
                    shutil.copy(path+i, out)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())
                if os.path.exists(path+i):  # 如果文件存在
                    # 删除文件，可使用以下两种方法。
                    os.remove(path+i)
                    # os.unlink(path)
                else:
                    print('no such file'+i)
print(num_txt)
print(num_jpg)
