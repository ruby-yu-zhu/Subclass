import pickle
path='./class_split.pkl'   #path='/root/……/aus_openface.pkl'   pkl文件所在路径
       
f=open(path,'rb')
data=pickle.load(f)
        
print(data)


