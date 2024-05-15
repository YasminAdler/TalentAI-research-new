import csv
import random



def mean_generator(K, values):
    print("in mean_generator")
   # print(len(values))
   # print(values)
   # exit()
    items = random.sample(list(values), K)
    listed_items = [list(arr) for arr in items]
    return listed_items

def csv_to_nested_list(file_name):
    with open(file_name, 'r') as read_obj:
        # Return a reader object which will
        # iterate over lines in the given csvfile
        csv_reader = csv.reader(read_obj)
        # convert string to list
        list_of_csv = list(csv_reader)
        new_lst = [[int(x) for x in inner] for inner in list_of_csv]
        return new_lst
