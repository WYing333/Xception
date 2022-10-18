import os
import glob

#f=open('/home/ywang/org_actor.txt', 'w')
#f=open('/home/ywang/org_youtube.txt', 'w')
#f=open('/home/ywang/man_Deepfakes.txt', 'w')
#f=open('/home/ywang/man_Face2Face.txt', 'w')
#f=open('/home/ywang/man_FaceShifter.txt', 'w')
#f=open('/home/ywang/man_FaceSwap.txt', 'w')
#f=open('/home/ywang/man_NeuralTextures.txt', 'w')
f=open('/home/ywang/man_DeepFakeDetection.txt', 'w')

#filenames = glob.glob(r"/home/ywang/XceptionExtract/orriginal_sequences/actors/c23/videos/*/*.jpg")
#filenames = glob.glob(r"/home/ywang/XceptionExtract/original_sequences/youtube/c23/videos/*/*.jpg")
#filenames = glob.glob(r"/home/ywang/XceptionExtract/manipulated_sequences/Deepfakes/c23/videos/*/*.jpg")
#filenames = glob.glob(r"/home/ywang/XceptionExtract/manipulated_sequences/Face2Face/c23/videos/*/*.jpg")
#filenames = glob.glob(r"/home/ywang/XceptionExtract/manipulated_sequences/FaceShifter/c23/videos/*/*.jpg")
#filenames = glob.glob(r"/home/ywang/XceptionExtract/manipulated_sequences/NeuralTextures/c23/videos/*/*.jpg")
filenames = glob.glob(r"/home/ywang/XceptionExtract/manipulated_sequences/DeepFakeDetection/c23/videos/*/*.jpg")

filenames.sort()

for filename in filenames:
    #print(filename)
    f.write(filename+'\n')
f.close()

