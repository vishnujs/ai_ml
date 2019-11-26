import csv
import pandas as pd


def covert_file_to_csv():
    #Check if the data can be read or not
    # with open('/home/system.log', 'r') as in_file:
    pass
#     print([line for line in in_file], end='\n')
with open('/home/system.csv','w') as out_file,open('/home/system.log', 'r') as in_file:

    data_list = []
    writer = csv.writer(out_file)
    writer.writerow(['date','time','app_Name','severity','information'])
    
    #Read the rows in the input file and split the first 4 spaces and store them
    
    for line in in_file:
        columns = line[:-1].split(' ', 4);
        # print(columns)
        # columns[4]=' '.join(columns[4:])
        # print(columns)
        # break
        # writer.writerow(columns)
        data_list.append(columns)
    # print(data_list)
    data_set = pd.DataFrame(data_list,columns = ['date','time','app_Name','severity','information'])
    print(data_set.head(100))
    data_set.to_csv(r'/home/system.csv')

def main():
    data_set = pd.read_csv('system.csv', 'r', error_bad_lines=False,skiprows=1)
    print(data_set.head())


covert_file_to_csv()
# main()




