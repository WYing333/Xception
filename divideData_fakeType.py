ftr = open('/home/ywang/train_paper_label.txt', 'r') 
fte = open('/home/ywang/test_paper_label.txt', 'r')
fv = open('/home/ywang/validation_paper_label.txt', 'r')


ftr_DF = open('/home/ywang/train_DF_label.txt', 'w')
ftr_F2F = open('/home/ywang/train_F2F_label.txt', 'w')
ftr_FS = open('/home/ywang/train_FS_label.txt', 'w')
ftr_NT = open('/home/ywang/train_NT_label.txt', 'w')

fte_DF = open('/home/ywang/test_DF_label.txt', 'w')
fte_F2F = open('/home/ywang/test_F2F_label.txt', 'w')
fte_FS = open('/home/ywang/test_FS_label.txt', 'w')
fte_NT = open('/home/ywang/test_NT_label.txt', 'w')

fv_DF = open('/home/ywang/val_DF_label.txt', 'w')
fv_F2F = open('/home/ywang/val_F2F_label.txt', 'w')
fv_FS = open('/home/ywang/val_FS_label.txt', 'w')
fv_NT = open('/home/ywang/val_NT_label.txt', 'w')

#keywords
DF = "Deepfakes"
F2F = "Face2Face"
FS = "FaceSwap"
NT = "NeuralTextures"


#train
for line in ftr:
    if DF in line:
        ftr_DF.writelines(line)
    if F2F in line:
        ftr_F2F.writelines(line)
    if FS in line:
        ftr_FS.writelines(line)
    if NT in line:
        ftr_NT.writelines(line)


#test
for line in fte:
    if DF in line:
        fte_DF.writelines(line)
    if F2F in line:
        fte_F2F.writelines(line)
    if FS in line:
        fte_FS.writelines(line)
    if NT in line:
        fte_NT.writelines(line)

#val
for line in fv:
    if DF in line:
        fv_DF.writelines(line)
    if F2F in line:
        fv_F2F.writelines(line)
    if FS in line:
        fv_FS.writelines(line)
    if NT in line:
        fv_NT.writelines(line)



ftr_DF.close()
ftr_F2F.close()
ftr_FS.close()
ftr_NT.close()
fte_DF.close()
fte_F2F.close()
fte_FS.close()
fte_NT.close()
fv_DF.close()
fv_F2F.close()
fv_FS.close()
fv_NT.close()

ftr.close()
fte.close()
fv.close()