import xml.etree.ElementTree as ET
import numpy as np

def make_route(map_name, veh_num, begin, end, write_path):
    if map_name == "arterial4x4" or map_name == "grid4x4":
        path = "./" + map_name + "_1.rou.xml"
        base_path = "./base_" + map_name + "_1.rou.xml"
    else:
        path = "./" + map_name +  ".rou.xml"
        base_path = "./base_" + map_name +  ".rou.xml"

    tree = ET.parse(path)
    root = tree.getroot()

    route_pattern = dict()
    for name in root:
        if map_name == "cologne3" or map_name == "arterial4x4":
            for name_2 in name:
                attrs = name_2.attrib
                if "edges" in attrs.keys():
                    veh_type = name.attrib["type"]
                    route = attrs["edges"]
                    key = veh_type + "+" + route
                    if key in route_pattern.keys():
                        route_pattern[key] += 1
                    else:
                        route_pattern[key] = 1
        elif map_name == "grid4x4":
            for name_2 in name:
                attrs = name_2.attrib
                if "edges" in attrs.keys():
                    route = attrs["edges"]
                    key = route
                    if key in route_pattern.keys():
                        route_pattern[key] += 1
                    else:
                        route_pattern[key] = 1
        else:
            attrs = name.attrib
            if 'from' in attrs.keys() and 'to' in attrs.keys():
                veh_type = attrs["type"]
                route = attrs['from'] + '+' + attrs['to']
                key = veh_type + "+" + route
                if key in route_pattern.keys():
                    route_pattern[key] += 1
                else:
                    route_pattern[key] = 1
    
    sorted_route_pattern = sorted(route_pattern.items(), key=lambda item: item[1])
    sorted_route_pattern.reverse()
    
    routes = list()
    choice_num = list()
    for x in sorted_route_pattern:
        routes.append(x[0])
        choice_num.append(x[1])
    
    choice_num = np.asarray(choice_num)
    choice_prob = choice_num/np.sum(choice_num)

    write_tree = ET.parse(base_path)
    write_root = write_tree.getroot()
    interval = (end - begin) / veh_num
    current_time = begin

    veh_id = 0
    while current_time < end:
        route_type = np.random.choice(routes, p=choice_prob)
        route_sep = route_type.split("+")
        if map_name == "cologne3" or map_name == "arterial4x4":
            vehicle = ET.SubElement(write_root, "vehicle")
            route = ET.SubElement(vehicle, "route")
            vehicle.set("id", str(veh_id))
            vehicle.set("type", route_sep[0])
            vehicle.set("depart", str(current_time))
            route.set("edges", route_sep[1])
        elif map_name == "grid4x4":
            vehicle = ET.SubElement(write_root, "vehicle")
            route = ET.SubElement(vehicle, "route")
            vehicle.set("id", str(veh_id))
            vehicle.set("depart", str(current_time))
            route.set("edges", route_sep[0])
        else:
            trip = ET.SubElement(write_root, "trip")
            trip.set("id", str(veh_id))
            trip.set("type", route_sep[0])
            trip.set("depart", str(current_time))
            trip.set("from", route_sep[1])
            trip.set("to", route_sep[2])
        
        veh_id += 1
        
        current_time += interval
    
    write_tree.write(write_path)

if __name__ == "__main__":
    param_dict = {
        "cologne1":{"begin": 25200, "end": 28800, "veh_num":[1000, 2015, 3000, 4000]},
        "cologne3":{"begin": 25200, "end": 28800, "veh_num":[3000, 4494, 5000, 6000]},
        "cologne8":{"begin": 25200, "end": 28800, "veh_num":[1000, 2046, 3000, 4000]},
        "ingolstadt1":{"begin": 57600, "end": 61200, "veh_num":[1000, 1716, 2000, 3000]},
        "ingolstadt7":{"begin": 57600, "end": 61200, "veh_num":[2000, 3031, 4000, 5000]},
        "ingolstadt21":{"begin": 57600, "end": 61200, "veh_num":[3000, 4283, 5000, 6000]},
        "arterial4x4":{"begin": 0, "end": 3600, "veh_num":[1000, 2484, 3000, 4000]},
        "grid4x4":{"begin": 0, "end": 3600, "veh_num":[1000, 1473, 2000, 3000]}
    }

    for map_name in param_dict.keys():
        for x in param_dict[map_name]["veh_num"]:
            if map_name == "arterial4x4" or map_name == "grid4x4":
                write_path = "./environments_flow/" + str(x) + "/environments/" + map_name + "/" + map_name + "_1.rou.xml"
            else:
                write_path = "./environments_flow/" + str(x) + "/environments/" + map_name + "/" + map_name + ".rou.xml"

            make_route(map_name, x, param_dict[map_name]["begin"], param_dict[map_name]["end"], write_path)