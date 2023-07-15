import os
import shutil

if __name__ == "__main__":
    parent_dir = "environments_flow"

    dir_dict = {
        "cologne1": [1000, 2015, 3000, 4000],
        "cologne3": [3000, 4494, 5000, 6000],
        "cologne8": [1000, 2046, 3000, 4000],
        "ingolstadt1": [1000, 1716, 2000, 3000],
        "ingolstadt7": [2000, 3031, 4000, 5000],
        "ingolstadt21": [3000, 4283, 5000, 6000],
        "arterial4x4": [1000, 2484, 3000, 4000],
        "grid4x4": [1000, 1473, 2000, 3000]
    }

    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    for map_name in dir_dict:
        for x in dir_dict[map_name]:
            path = os.path.join(parent_dir, str(x), "environments")
            if not os.path.exists(path):
                os.makedirs(path)

            if map_name == "cologne1":
                path = os.path.join(parent_dir, str(x), "environments", "cologne1")
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy("../environments/cologne1/cologne1.sumocfg", path)
                shutil.copy("../environments/cologne1/cologne1.net.xml", path)

            if map_name == "cologne3":
                path = os.path.join(parent_dir, str(x), "environments", "cologne3")
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy("../environments/cologne3/cologne3.sumocfg", path)
                shutil.copy("../environments/cologne3/cologne3.net.xml", path)
            
            if map_name == "cologne8":
                path = os.path.join(parent_dir, str(x), "environments", "cologne8")
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy("../environments/cologne8/cologne8.sumocfg", path)
                shutil.copy("../environments/cologne8/cologne8.net.xml", path)

            if map_name == "ingolstadt1":
                path = os.path.join(parent_dir, str(x), "environments", "ingolstadt1")
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy("../environments/ingolstadt1/ingolstadt1.sumocfg", path)
                shutil.copy("../environments/ingolstadt1/ingolstadt1.net.xml", path)

            if map_name == "ingolstadt7":
                path = os.path.join(parent_dir, str(x), "environments", "ingolstadt7")
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy("../environments/ingolstadt7/ingolstadt7.sumocfg", path)
                shutil.copy("../environments/ingolstadt7/ingolstadt7.net.xml", path)

            if map_name == "ingolstadt21":
                path = os.path.join(parent_dir, str(x), "environments", "ingolstadt21")
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy("../environments/ingolstadt21/ingolstadt21.sumocfg", path)
                shutil.copy("../environments/ingolstadt21/ingolstadt21.net.xml", path)

            if map_name == "arterial4x4":
                path = os.path.join(parent_dir, str(x), "environments", "arterial4x4")
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy("../environments/arterial4x4/arterial4x4.net.xml", path)

            if map_name == "grid4x4":
                path = os.path.join(parent_dir, str(x), "environments", "grid4x4")
                if not os.path.exists(path):
                    os.mkdir(path)
                shutil.copy("../environments/grid4x4/grid4x4.net.xml", path)