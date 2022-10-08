import numpy
from csv import writer

from sklearn import preprocessing
vesa = numpy.array([2,2,3,5,2,3,5,3,2,2,2,4,1,1,1,5]) #веса каждого элемента
nv = ((numpy.around(preprocessing.normalize([vesa], norm="l1"), 4)).tolist())[0]
#print(nv)
#vector1 = [1,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,1,0] #вектор1 - исходная ситуация - по ней выставляются веса
vector1 = [0,1,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0] #вектор1 - исходная ситуация - по ней выставляются веса
s = numpy.array(vector1)
#print(ss)
s1 = numpy.array([1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1]) #вектор2 - сравниваемая ситуация
rs = numpy.array([0,1,0,1,0,0.3,0.8,1,0,0.3,0.8,1,0,0.3,0.8,1,0,0.3,0.8,1,0,0.3,0.8,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]) #расстояния между состояниям
k = [2,2,4,4,4,4,4,2,2,2,2,2,2,2,2,2] #размеры векторов по каждому элементу
print(vesa.shape)


train = numpy.loadtxt('/Users/Public/files/proba1111.csv', delimiter=",")
shape = train.shape[0]
#print(s1)

for sss in range(600):
    itog = []
    print(sss*1)
    #print(train[sss].astype(int))
    s1 = train[sss*1].astype(int)
    #print(s1)
    #print((train[s1].astype(int)).shape)
    for l, v in zip((range(16)),nv):
        #print(sum(k[0:(l+1)]))
        X = abs((s[sum(k[0:l]):sum(k[0:(l+1)])]-s1[sum(k[0:l]):sum(k[0:(l+1)])]))
        #print(X)
        R = rs[sum(k[0:l]):sum(k[0:(l+1)])]
        #print(R)
        itog.append(sum(abs(R*X))*v)
        #rint(sum(abs(R*X))*v)
    print(numpy.around(1-sum(itog), decimals=3))
    sim = numpy.array([(numpy.around(1-sum(itog), decimals=3))])
    d = numpy.concatenate((s, train[sss*1].astype(int),sim), axis=0)
    print(d)
    with open('/Users/Public/files/train.csv', 'a', newline='') as f_object:
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(d)
        # writer_object.writerow(vector1+sum(vector,[])+[sim])
        # Close the file object
        f_object.close()


