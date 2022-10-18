import json

#Test Paper
# with open("/home/ywang/splits/test.json",'r') as load_f:
#     load_dict = json.load(load_f)

# f = open('/home/ywang/all.txt', 'r') 
# ft = open('/home/ywang/test_paper.txt', 'w')

# real_keyword = "original"

# for line in f:
#     for pair in load_dict:
#         first = pair[0]
#         second = pair[1]
#         path_pair = first + '_' + second 
#         path_pair_reverse = second + '_' + first

#         if path_pair in line:
#             ft.writelines(line)
#         if path_pair_reverse in line:
#             ft.writelines(line)

#         if real_keyword in line:
#             train_face = line.split('/')[8]
#             if first in train_face:
#                 ft.writelines(line)
#             if second in train_face:
#                 ft.writelines(line)

# ft.close()
# f.close() 


#Val Paper
# with open("/home/ywang/splits/val.json",'r') as load_f:
#     load_dict = json.load(load_f)

# f = open('/home/ywang/all.txt', 'r') 
# ft = open('/home/ywang/validation_paper.txt', 'w')

# real_keyword = "original"

# for line in f:
#     for pair in load_dict:
#         first = pair[0]
#         second = pair[1]
#         path_pair = first + '_' + second 
#         path_pair_reverse = second + '_' + first

#         if path_pair in line:
#             ft.writelines(line)
#         if path_pair_reverse in line:
#             ft.writelines(line)

#         if real_keyword in line:
#             train_face = line.split('/')[8]
#             if first in train_face:
#                 ft.writelines(line)
#             if second in train_face:
#                 ft.writelines(line)

# ft.close()
# f.close() 


# #Train Paper
# with open("/home/ywang/splits/train.json",'r') as load_f:
#     load_dict = json.load(load_f)

# f = open('/home/ywang/all.txt', 'r') 
# ft = open('/home/ywang/train_paper.txt', 'w')

# real_keyword = "original"

# for line in f:
#     for pair in load_dict:
#         first = pair[0]
#         second = pair[1]
#         path_pair = first + '_' + second 
#         path_pair_reverse = second + '_' + first

#         if path_pair in line:
#             ft.writelines(line)
#         if path_pair_reverse in line:
#             ft.writelines(line)

#         if real_keyword in line:
#             train_face = line.split('/')[8]
#             if first in train_face:
#                 ft.writelines(line)
#             if second in train_face:
#                 ft.writelines(line)

# ft.close()
# f.close() 



# Add Label
ftr1 = open('/home/ywang/train_paper.txt', 'r') 
fte1 = open('/home/ywang/test_paper.txt', 'r')
fv1 = open('/home/ywang/validation_paper.txt', 'r')
ftr = open('/home/ywang/train_paper_label.txt', 'w') 
fte = open('/home/ywang/test_paper_label.txt', 'w')
fv = open('/home/ywang/validation_paper_label.txt', 'w')

fake_keyword = "manipulated"
real_keyword = "original"

for line in ftr1:
    line=line.strip('\n')
    if fake_keyword in line:
        label = 1  #fake=1

    if real_keyword in line:
        label = 0

    line = line + " " + str(label) + "\n"
    ftr.write(line)

for line in fte1:
    line=line.strip('\n')
    if fake_keyword in line:
        label = 1  #fake=1

    if real_keyword in line:
        label = 0

    line = line + " " + str(label) + "\n"
    fte.write(line)

for line in fv1:
    line=line.strip('\n')
    if fake_keyword in line:
        label = 1  #fake=1

    if real_keyword in line:
        label = 0

    line = line + " " + str(label) + "\n"
    fv.write(line)

    
ftr.close()
fte.close()
fv.close()
ftr1.close()
fte1.close()
fv1.close()

