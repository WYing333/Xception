import numpy as np

def train_test_val_split(X, test_ratio, train_ratio, seed=None):
    """按test_size和train_ratio的比例分割训练集测试集"""
    #assert X.shape[0]==y.shape[0],'the shape of X must equal to y'
    assert 0.0<=test_ratio<=1.0,'test_size must be valid'
    assert 0.0<=train_ratio<=1.0,'train_size must be valid'

    if seed is not None:
        np.random.seed(seed)

    shuffled_indices=np.random.permutation(len(X))#随机索引
    test_size=int(len(X)*test_ratio)
    train_size=int(len(X)*train_ratio)
    #print(train_size)

    test_indices=shuffled_indices[:test_size]
    train_indices=shuffled_indices[test_size:train_size+test_size]
    validation_indices=shuffled_indices[train_size+test_size:]
   # print(test_indices)

    X_train = []
    X_test = []
    X_validation = []
    #X_train,y_train=X[train_indices],y[train_indices]
    for train_idx in train_indices:
        X_train.append(X[train_idx][0])
        X_train.append(X[train_idx][1])
    #print(X_train)

    for test_idx in test_indices:
        X_test.append(X[test_idx][0])
        X_test.append(X[test_idx][1])

    for validation_idx in validation_indices:
        X_validation.append(X[validation_idx][0])
        X_validation.append(X[validation_idx][1])

    return X_train, X_test, X_validation



f = open('/home/ywang/all.txt', 'r') 
keyword = "manipulated"
face_pairs = []

for line in f:
    if keyword in line:
        face_pair_str = line.split('/')[8]
        face_pair = face_pair_str.split('_')
        if int(face_pair[0]) > int(face_pair[1]):
            face_pair[0], face_pair[1] = face_pair[1], face_pair[0]
            #print(face_pair)
        if not face_pair in face_pairs:
            face_pairs.append(face_pair)
            #print(face_pairs)
            #print("\n")

#print(face_pairs)
#print(len(face_pairs))
#print(face_pairs[:500])
f.close() 


X_train, X_test, X_validation = train_test_val_split(face_pairs, 0.2, 0.7, None)

#print(X_train)
#print(X_test)
#print(X_validation)



# Train, Test, Validation Data Path
f1 = open('/home/ywang/all.txt', 'r') 
ftr = open('/home/ywang/train.txt', 'w')
fte = open('/home/ywang/test.txt', 'w')
fv = open('/home/ywang/validation.txt', 'w')

fake_keyword = "manipulated"
real_keyword = "original"
train_face_pairs = []
test_face_pairs = []
validation_face_pairs = []

for line in f1:
    if fake_keyword in line:
        train_face_pair_str = line.split('/')[8]
        train_face_pair = train_face_pair_str.split('_')
        if train_face_pair[0] in X_train:
            ftr.writelines(line)
        elif train_face_pair[0] in X_test:
            fte.writelines(line)
        elif train_face_pair[0] in X_validation:
            fv.writelines(line)

        # if train_face_pair[1] in X_train:
        #     ftr.writelines(line)
        # elif train_face_pair[1] in X_test:
        #     fte.writelines(line)
        # elif train_face_pair[1] in X_validation:
        #     fv.writelines(line)

    
    if real_keyword in line:
        train_face = line.split('/')[8]
        if train_face in X_train:
            ftr.writelines(line)
        elif train_face in X_test:
            fte.writelines(line)
        elif train_face in X_validation:
            fv.writelines(line)
        
     
ftr.close()
fte.close()
fv.close()
f1.close() 



#Data Balance in Train
train_real = 0
train_fake = 0

f = open('/home/ywang/train.txt', 'r') 

for line in f:
    if fake_keyword in line:
        train_fake = train_fake + 1
    
    if real_keyword in line:
        train_real = train_real + 1

f.close()

fake_real_ratio = float(train_fake/train_real)

print ("Real number of faces in Train is: %d" %(train_real))
print ("Fake number of faces in Train is: %d" %(train_fake))
print("Fake : Real = %10.3f" %(fake_real_ratio))


# Do the actual balance in train
f1 = open('/home/ywang/train.txt', 'r') 
f = open('/home/ywang/train_balanced.txt', 'w') 
balanced_keyword = "original"

for line in f1:
    f.writelines(line)
    if balanced_keyword in line:
        for i in range(1,int(fake_real_ratio)):
            f.writelines(line)
    
f.close()
f1.close()




# Add Label
ftr1 = open('/home/ywang/train_balanced.txt', 'r') 
fte1 = open('/home/ywang/test.txt', 'r')
fv1 = open('/home/ywang/validation.txt', 'r')
ftr = open('/home/ywang/train_balanced_label.txt', 'w') 
fte = open('/home/ywang/test_label.txt', 'w')
fv = open('/home/ywang/validation_label.txt', 'w')

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


