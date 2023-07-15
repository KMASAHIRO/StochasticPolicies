import shutil

if __name__ == "__main__":
    x_list = [1000, 2484, 3000, 4000]
    #x_list = [1000, 1473, 2000, 3000]

    for x in x_list:
        for i in range(1400):
            if i == 0:
                continue
            dir_name = "./" + str(x) + "/environments/"
            shutil.copy(dir_name + "arterial4x4/arterial4x4_1.rou.xml", dir_name + "arterial4x4/arterial4x4_" + str(i+1) + ".rou.xml")
            #shutil.copy(dir_name + "grid4x4/grid4x4_1.rou.xml", dir_name + "grid4x4/grid4x4_" + str(i+1) + ".rou.xml")
