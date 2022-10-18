fte = open('/home/ywang/test_paper_label.txt', 'r') 

fte_DF = open('/home/ywang/test_DF_label.txt', 'r') #改模式
fte_DF_ = open('/home/ywang/test_DF_label_.txt', 'w')#加新文件 最后改名


real_keyword = "original"
face_pairs=[]

for line_ in fte_DF:
    face_pair_str = line_.split('/')[8]
    face_pair = face_pair_str.split('_')
    if face_pair[0] not in face_pairs:
        face_pairs.append(face_pair[0])
    if face_pair[1] not in face_pairs:
        face_pairs.append(face_pair[1])

    fte_DF_.writelines(line_)


for line in fte:
    if real_keyword in line:
        train_face = line.split('/')[8]
        if train_face in face_pairs:
            fte_DF_.writelines(line)
        elif train_face in face_pairs:
            fte_DF_.writelines(line)



fte_DF.close()
fte_DF_.close()

fte.close()

