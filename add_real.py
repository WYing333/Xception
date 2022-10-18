ftr = open('/home/ywang/train_paper_label.txt', 'r') 
fte = open('/home/ywang/test_paper_label.txt', 'r')
fv = open('/home/ywang/validation_paper_label.txt', 'r')

#train
# ftr_DF = open('/home/ywang/train_DF_label.txt', 'r')
# ftr_DF_ = open('/home/ywang/train_DF_label_.txt', 'w')
# ftr_F2F = open('/home/ywang/train_F2F_label.txt', 'r') 
# ftr_F2F_ = open('/home/ywang/train_F2F_label_.txt', 'w')
# ftr_FS = open('/home/ywang/train_FS_label.txt', 'r') 
# ftr_FS_ = open('/home/ywang/train_FS_label_.txt', 'w')
ftr_NT = open('/home/ywang/train_NT_label.txt', 'r') 
ftr_NT_ = open('/home/ywang/train_NT_label_.txt', 'w')

#test
# fte_DF = open('/home/ywang/test_DF_label.txt', 'r') #改模式
# fte_DF_ = open('/home/ywang/test_DF_label_.txt', 'w')#加新文件 最后改名
# fte_F2F = open('/home/ywang/test_F2F_label.txt', 'r') 
# fte_F2F_ = open('/home/ywang/test_F2F_label_.txt', 'w')
# fte_FS = open('/home/ywang/test_FS_label.txt', 'r') 
# fte_FS_ = open('/home/ywang/test_FS_label_.txt', 'w')
fte_NT = open('/home/ywang/test_NT_label.txt', 'r') 
fte_NT_ = open('/home/ywang/test_NT_label_.txt', 'w')

#val
# fv_DF = open('/home/ywang/val_DF_label.txt', 'r')
# fv_DF_ = open('/home/ywang/val_DF_label_.txt', 'w')
# fv_F2F = open('/home/ywang/val_F2F_label.txt', 'r') 
# fv_F2F_ = open('/home/ywang/val_F2F_label_.txt', 'w')
# fv_FS = open('/home/ywang/val_FS_label.txt', 'r') 
# fv_FS_ = open('/home/ywang/val_FS_label_.txt', 'w')
fv_NT = open('/home/ywang/val_NT_label.txt', 'r') 
fv_NT_ = open('/home/ywang/val_NT_label_.txt', 'w')


real_keyword = "original"

#train
train_face_pairs=[]
for line_ in ftr_NT:
    train_face_pair_str = line_.split('/')[8]
    train_face_pair = train_face_pair_str.split('_')
    if train_face_pair[0] not in train_face_pairs:
        train_face_pairs.append(train_face_pair[0])
    if train_face_pair[1] not in train_face_pairs:
        train_face_pairs.append(train_face_pair[1])

    ftr_NT_.writelines(line_)

for line in ftr:
    if real_keyword in line:
        train_face = line.split('/')[8]
        if train_face in train_face_pairs:
            ftr_NT_.writelines(line)
        elif train_face in train_face_pairs:
            ftr_NT_.writelines(line)


#test
test_face_pairs=[]
for line_ in fte_NT:
    test_face_pair_str = line_.split('/')[8]
    test_face_pair = test_face_pair_str.split('_')
    if test_face_pair[0] not in test_face_pairs:
        test_face_pairs.append(test_face_pair[0])
    if test_face_pair[1] not in test_face_pairs:
        test_face_pairs.append(test_face_pair[1])

    fte_NT_.writelines(line_)


for line in fte:
    if real_keyword in line:
        test_face = line.split('/')[8]
        if test_face in test_face_pairs:
            fte_NT_.writelines(line)
        elif test_face in test_face_pairs:
            fte_NT_.writelines(line)

#val
val_face_pairs=[]
for line_ in fv_NT:
    val_face_pair_str = line_.split('/')[8]
    val_face_pair = val_face_pair_str.split('_')
    if val_face_pair[0] not in val_face_pairs:
        val_face_pairs.append(val_face_pair[0])
    if val_face_pair[1] not in val_face_pairs:
        val_face_pairs.append(val_face_pair[1])

    fv_NT_.writelines(line_)


for line in fv:
    if real_keyword in line:
        val_face = line.split('/')[8]
        if val_face in val_face_pairs:
            fv_NT_.writelines(line)
        elif val_face in val_face_pairs:
            fv_NT_.writelines(line)



#ftr_DF.close()
#ftr_F2F.close()
#ftr_FS.close()
ftr_NT.close()
#fte_DF.close()
#fte_F2F.close()
#fte_FS.close()
fte_NT.close()
#fv_DF.close()
#fv_F2F.close()
#fv_FS.close()
fv_NT.close()

#ftr_DF_.close()
#ftr_F2F_.close()
#ftr_FS_.close()
ftr_NT_.close()
#fte_DF_.close()
#fte_F2F_.close()
#fte_FS_.close()
fte_NT_.close()
#fv_DF_.close()
#fv_F2F_.close()
#fv_FS_.close()
fv_NT_.close()

ftr.close()
fte.close()
fv.close()

