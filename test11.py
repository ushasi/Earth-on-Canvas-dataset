
# coding: utf-8

# In[1]:
import warnings
warnings.filterwarnings("ignore")

import random
import scipy.io as sio
import numpy as np
import tensorflow as tf
import os
#from spektral2.spektral.layers import GraphConv
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from flip_gradient import flip_gradient
import seaborn as sns
tfd = tf.contrib.distributions
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

nb_classes = 220

# In[2]:
os.environ["CUDA_VISIBLE_DEVICES"]="3"

datasetX = sio.loadmat('dataset/vggsketch.mat') 
sketches = np.array(datasetX['feature'])
datasetY = sio.loadmat('dataset/vggphoto.mat') #sketchmat2048,sketchmatnew
photos = np.array(datasetY['feature'])  


'''
datasetX = sio.loadmat('dataset/imagehmat.mat') 
sketches = np.array(datasetX['feature'])
datasetY = sio.loadmat('dataset/sketchmat.mat')#sketchmat2048,sketchmatnew
photos = np.array(datasetY['feature'])  
'''
label_im = sio.loadmat('dataset/img_labels.mat')
label_im = np.squeeze(np.array(label_im['label']))
lab_sk =  sio.loadmat('dataset/sk_labels.mat')
lab_sk = np.squeeze(np.array(lab_sk['label']))


label_im = np.expand_dims(label_im.flatten(), axis=1)
lab_sk = np.expand_dims(lab_sk, axis=1)


print(photos.shape)
print(label_im.shape)

print(sketches.shape)
print(lab_sk.shape)

'''
tsned = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results3 = tsned.fit_transform(sketches)

dx = sns.scatterplot(tsne_results3[:,0], tsne_results3[:,1], np.squeeze(lab_sk), palette=sns.color_palette("hls", 30), alpha=0.6)
plt.savefig('sketchsne.png', bbox_inches='tight')
plt.clf()
'''

# In[3]:




print('sketches shape 1',sketches.shape)
print('label shape 1',lab_sk.shape)
# In[4]:


X = photos
Y = sketches
Lx = label_im
Ly = lab_sk


# In[5]:

X1 = X[0:17600,:]
Y1 = Y[0:175098,:]
Lx1 = Lx[0:17600] - 1
Ly1 = Ly[0:175098] - 1

X2 = X[17601:20000,:]
Y2 = Y[175099:204071,:]
Lx2 = Lx[17601:20000] - 1
Ly2 = Ly[175099:204071] - 1
# In[6]:

print(Y1.shape)
print(Ly1.shape)

tsnec = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results2 = tsnec.fit_transform(X1)

cx = sns.scatterplot(tsne_results2[:,0], tsne_results2[:,1], np.squeeze(Lx1), palette=sns.color_palette("hls", 220), alpha=0.6)
plt.savefig('phototsne.png', bbox_inches='tight')
plt.clf()

train_X, test_X, train_Lx, test_Lx = train_test_split(X1, Lx1, test_size=0.1)

train_Y, test_Y, train_Ly, test_Ly = train_test_split(Y1, Ly1, test_size=0.1)


# In[7]:


dataset = sio.loadmat('dataset/wv_anjan_semantic_labels.mat')
wv = np.array(dataset['wv'], np.float32)  # glove features 

train_wvx1 = np.empty([0, 300], np.float32)
test_wvx1 = np.empty([0, 300], np.float32)
train_wvy1 = np.empty([0, 300], np.float32)
test_wvy1 = np.empty([0, 300], np.float32)

train_wvx1 = wv[train_Lx,:]
train_wvx1 = np.reshape(train_wvx1, [train_wvx1.shape[0], train_wvx1.shape[1] * train_wvx1.shape[2]])

train_wvy1 = wv[train_Ly,:]
train_wvy1 = np.reshape(train_wvy1, [train_wvy1.shape[0], train_wvy1.shape[1] * train_wvy1.shape[2]])

test_wvx1 = wv[test_Lx,:]
test_wvx1 = np.reshape(test_wvx1, [test_wvx1.shape[0], test_wvx1.shape[1] * test_wvx1.shape[2]])

test_wvy1 = wv[test_Ly,:]
test_wvy1 = np.reshape(test_wvy1, [test_wvy1.shape[0], test_wvy1.shape[1] * test_wvy1.shape[2]])

unseen_wvx1 = wv[Lx2,:]
unseen_wvx1 = np.reshape(unseen_wvx1, [unseen_wvx1.shape[0], unseen_wvx1.shape[1] * unseen_wvx1.shape[2]])

unseen_wvy1 = wv[Ly2,:]
unseen_wvy1 = np.reshape(unseen_wvy1, [unseen_wvy1.shape[0], unseen_wvy1.shape[1] * unseen_wvy1.shape[2]])



def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = data#np.array(int(data)).reshape(-1)
    #targets = np.array([x[0]-1 for x in data])
    return np.eye(nb_classes)[targets]


#onehot_encoder = OneHotEncoder(sparse=False)

onehot_train_Lx = indices_to_one_hot(train_Lx,nb_classes)
onehot_test_Lx = indices_to_one_hot(test_Lx,nb_classes)
onehot_train_Ly = indices_to_one_hot(train_Ly,nb_classes)
onehot_test_Ly = indices_to_one_hot(test_Ly,nb_classes)

onehot_train_Lx = np.squeeze(onehot_train_Lx)
onehot_test_Lx= np.squeeze(onehot_test_Lx)
onehot_train_Ly = np.squeeze(onehot_train_Ly)
onehot_test_Ly = np.squeeze(onehot_test_Ly)

print('label shape 2',onehot_test_Lx.shape)
print('label shape 3',onehot_train_Lx.shape)
print('label shape 2',onehot_test_Ly.shape)
print('label shape 3',onehot_train_Ly.shape)

# In[9]:

anchor = np.empty([0,256], np.float32) # anchor means photo
anchor_wv = np.empty([0,300], np.float32)

pos = np.empty([0,256], np.float32) # pos means sketch
pos_wv = np.empty([0,300], np.float32)

neg = np.empty([0,256], np.float32)
neg_wv = np.empty([0,300], np.float32)

anchor_l = np.empty([0,220], np.float32)
pos_l = np.empty([0,220], np.float32)
neg_l = np.empty([0,220], np.float32)

lab = np.unique(train_Lx)
print(lab.shape)


for l in np.unique(train_Lx):
    sub = [];sub1 = []

    r,c = np.where(train_Lx == l)
    r1,c1 = np.where(train_Ly == l)
    
    diff = np.setdiff1d(lab,l)
    r2,c2 = np.where(train_Ly == diff)   
    
    print(l,len(r),len(r1),len(r2))
    for p in range(25):
       sub = sub + random.sample(list(r),2)
    for p in range(25):
       sub1 = sub1 + random.sample(list(r1),2)
    sub2 = random.sample(list(r2),50)
    
    temp_data = train_X[sub,:]
    temp_data1 = train_Y[sub1,:]
    temp_data2 = train_Y[sub2,:]

    
    temp_data_l = onehot_train_Lx[sub,:]
    temp_data1_l = onehot_train_Ly[sub1,:]
    temp_data2_l = onehot_train_Ly[sub2,:]
    
    temp_wvx = train_wvx1[sub,:]
    temp_wvy1 = train_wvy1[sub1,:]
    temp_wvy2 = train_wvy1[sub2,:]
    
    anchor = np.vstack((anchor,temp_data)) # tensor for the photo features
    pos = np.vstack((pos,temp_data1)) # tensor for the sketch with similarr class labels
    neg = np.vstack((neg, temp_data2))
    
    anchor_l = np.vstack((anchor_l,temp_data_l)) # photo label
    pos_l = np.vstack((pos_l,temp_data1_l)) # sketch label
    neg_l = np.vstack((neg_l,temp_data2_l))
    
    anchor_wv = np.vstack((anchor_wv,temp_wvx)) # photo glove
    pos_wv = np.vstack((pos_wv,temp_wvy1)) # skech glove
    neg_wv = np.vstack((neg_wv,temp_wvy2))

# In[10]:


p = np.zeros([20,1], np.float32)
s = np.ones([20,1], np.float32)

d_lab = np.vstack((p,s))

onehot_encoder = OneHotEncoder(sparse=False)

onehot_d_lab = onehot_encoder.fit_transform(d_lab)

'''
# In[11]:
# sliced wasserstein computation use
def get_theta(embedding_dim, num_samples=50):
    theta = [w/np.sqrt((w**2).sum())
             for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor).cuda()


def sliced_wasserstein_distance(source_z, target_z, num_projections=50, p=2):
    # theta is vector represents the projection directoin
    batch_size = target_z.shape[0]
    theta = get_theta(64*9*9, num_projections)
    proj_target = tf.matmul(target_z)
    proj_target = target_z.matmul(theta.transpose(0, 1))
    proj_source = source_z.matmul(theta.transpose(0, 1))
    w_distance = torch.sort(proj_target.transpose(0, 1), dim=1)[
        0]-torch.sort(proj_source.transpose(0, 1), dim=1)[0]

    # calculate by the definition of p-Wasserstein distance
    w_distance_p = torch.pow(w_distance, p)

return w_distance_p.mean()
'''
def next_batch(s, e, input1, input2, input3, l1, l2, w1, w2):
    
    inp1 = input1[s:e,:]
    inp2 = input2[s:e,:]
    inp3 = input3[s:e,:]
    
    lab1 = l1[s:e,:]
    lab2 = l2[s:e,:]
    
    wv1 = w1[s:e, :]
    wv2 = w2[s:e, :]
        
    return inp1, inp2, inp3, lab1, lab2, wv1, wv2

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# In[12]:


feature_dim = np.shape(anchor)[1]
n_class = np.shape(anchor_l)[1]
w2v_dim = 300
h1 = 128
h2 = 64
h1by2 = 32

x1 = tf.placeholder(tf.float32, [None, feature_dim]) #photo feature
x2 = tf.placeholder(tf.float32, [None, feature_dim]) #sketch feature
x3 = tf.placeholder(tf.float32, [None, feature_dim]) #sketch feature


y1 = tf.placeholder(tf.float32, [None, n_class]) # photo label
y2 = tf.placeholder(tf.float32, [None, n_class]) # sketch label

v1 = tf.placeholder(tf.float32, [None, w2v_dim]) # photo glove
v2 = tf.placeholder(tf.float32, [None, w2v_dim]) # sketch glove

d1 = tf.placeholder(tf.float32, [None, 2]) #domain label


# In[13]:


# Encoder for photo stream
w11 = tf.Variable(tf.random_normal([feature_dim, h1], stddev = 0.01), name= 'w11')
w12 = tf.Variable(tf.random_normal([h1, h2], stddev = 0.01), name= 'w12')

b11 = tf.Variable(tf.random_normal([h1], stddev = 0.01), name= 'b11')
b12 = tf.Variable(tf.random_normal([h2], stddev = 0.01), name= 'b12')

#Encoder for sketch stream
w21 = tf.Variable(tf.random_normal([feature_dim, h1], stddev = 0.01), name= 'w21')
w22 = tf.Variable(tf.random_normal([h1, h2], stddev = 0.01), name= 'w22')

b21 = tf.Variable(tf.random_normal([h1], stddev = 0.01), name= 'b21')
b22 = tf.Variable(tf.random_normal([h2], stddev = 0.01), name= 'b22')


#classifier

w_c = tf.Variable(tf.random_normal([h2, n_class], stddev = 0.01), name= 'w1_c') # classifier weight
b_c = tf.Variable(tf.random_normal([n_class], stddev = 0.01), name= 'b_c') # classifier bias

#For semantic attention branch, one FC layer
w31 = tf.Variable(tf.random_normal([w2v_dim, h2], stddev = 0.01), name= 'w31')
b31 = tf.Variable(tf.random_normal([h2], stddev = 0.01), name= 'b31')

#For semantic branch, one FC layer
w41 = tf.Variable(tf.random_normal([w2v_dim, h2], stddev = 0.01), name= 'w31')
b41 = tf.Variable(tf.random_normal([h2], stddev = 0.01), name= 'b31')

#Domain classifier parameter (for photo and semantic)
w_d = tf.Variable(tf.random_normal([h2,2], stddev = 0.01), name= 'w_p')
b_d = tf.Variable(tf.random_normal([2], stddev = 0.01), name= 'b_p')

l = tf.placeholder(tf.float32, [])

#Decoder features

w111 = tf.Variable(tf.random_normal([h2, feature_dim], stddev = 0.01), name= 'w111')
b111 = tf.Variable(tf.random_normal([feature_dim], stddev = 0.01), name= 'b111')

w112 = tf.Variable(tf.random_normal([h2, feature_dim], stddev = 0.01), name= 'w112')
b112 = tf.Variable(tf.random_normal([feature_dim], stddev = 0.01), name= 'b112')

# In[14]:


is_training = tf.placeholder_with_default(False, (), 'is_training')

#Photo
x1_h1 = (tf.add(tf.matmul(x1, w11), b11))
x1_h1 = tf.layers.batch_normalization(x1_h1,training = is_training)
x1_h2 = (tf.add(tf.matmul(x1_h1, w12), b12))
mu1, rho1 = x1_h2[:, :h1by2], x1_h2[:, h1by2:]

#sketch
x2_h1 = (tf.add(tf.matmul(x2, w21), b21))
#x2_h1 = tf.layers.batch_normalization(x2_h1,training = is_training)
x2_h2 = (tf.add(tf.matmul(x2_h1, w22), b22))
mu2, rho2 = x2_h2[:, :h1by2], x2_h2[:, h1by2:]

#sketch neg
x3_h1 = (tf.add(tf.matmul(x3, w21), b21))
#x2_h1 = tf.layers.batch_normalization(x2_h1,training = is_training)
x3_h2 = (tf.add(tf.matmul(x3_h1, w22), b22))


#Semantic attention network
v1_h = (tf.add(tf.matmul(v1, w31), b31))
v2_h = (tf.add(tf.matmul(v2, w31), b31))
#mu3, rho3 = v1_h[:, :h1by2], v1_h[:, h1by2:]
#mu4, rho4 = v2_h[:, :h1by2], v2_h[:, h1by2:]

# k-l divergence between anchor and positive sample
'''  
rho1 = tf.math.softplus(rho1) + 1e-7 
rho2 = tf.math.softplus(rho2) + 1e-7 
rho3 = tf.math.softplus(rho3) + 1e-7 
rho4 = tf.math.softplus(rho4) + 1e-7 
z1v1 =tfd.Independent(tfd.Normal(loc=mu1,scale=rho1),1)  
z2v2 =tfd.Independent(tfd.Normal(loc=mu2,scale=rho2),1)
z3v3 =tfd.Independent(tfd.Normal(loc=mu1,scale=rho3),1)  
z4v4 =tfd.Independent(tfd.Normal(loc=mu2,scale=rho4),1)

z1 = z1v1.sample()
z2 = z2v2.sample()
z3 = z3v3.sample()
z4 = z4v4.sample()
 
# Symmetrized Kullback-Leibler divergence 
#between image and sketch
kl_1_2 = z1v1.log_prob(z1) - z2v2.log_prob(z1)
kl_2_1 = z2v2.log_prob(z2) - z1v1.log_prob(z2)
skl_a = tf.reduce_mean(tf.add(kl_1_2, kl_2_1))

#between image and semantic
kl_1_3 = z1v1.log_prob(z1) - z3v3.log_prob(z1)
kl_3_1 = z3v3.log_prob(z3) - z1v1.log_prob(z3)
skl_b = tf.reduce_mean(tf.add(kl_1_3, kl_3_1))

#between sketch and semantic
kl_4_2 = z4v4.log_prob(z4) - z2v2.log_prob(z4)
kl_4_1 = z2v2.log_prob(z2) - z4v4.log_prob(z2)
skl_c = tf.reduce_mean(tf.add(kl_4_2, kl_4_1))


#skl_a = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
skl_a = tfd.kl_divergence(z1,z2, allow_nan_stats=True)
skl_b = tfd.kl_divergence(z1,z3, allow_nan_stats=True)
skl_c = tfd.kl_divergence(z2,z4, allow_nan_stats=True)

skl =  skl_a +  skl_b +  skl_c 
'''
#Attention

x1_h21 = tf.add(x1_h2, tf.multiply(x1_h2, tf.nn.sigmoid(v1_h)))
x2_h21 = tf.add(x2_h2, tf.multiply(x2_h2, tf.nn.sigmoid(v2_h)))

#Cross-decoder

x1_he_d = (tf.add(tf.matmul(x1_h21, w111), b111))
x2_he_d = (tf.add(tf.matmul(x2_h21, w112), b112))


#Semantic branch

v1_h2 = tf.nn.leaky_relu(tf.add(tf.matmul(v1, w41), b41), alpha = 0.01)
v2_h2 = tf.nn.leaky_relu(tf.add(tf.matmul(v2, w41), b41), alpha = 0.01)

#Classifier

x1_h2_c = tf.add(tf.matmul(x1_h21, w_c), b_c)
x2_h2_c = tf.add(tf.matmul(x2_h21, w_c), b_c)

#W-Decoder - photo

x_stack = tf.concat([x1_h21, x2_h21], axis = 0)

x_stack1 = flip_gradient(x_stack, l)

x_stack_d = tf.matmul(x_stack1, w_d) + b_d


# In[15]:


#Domain loss

domain_pred = tf.nn.softmax(x_stack_d)
domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=x_stack_d, labels=d1)


# In[16]:
#cross triplet

d_pos = tf.reduce_sum(tf.square(x2_h2 - x1_h2), 1)
d_neg = tf.reduce_sum(tf.square(x2_h2 - x3_h2), 1)

t_loss = tf.maximum(0.0, 0.5 + d_pos - d_neg)
triplet_loss = tf.reduce_mean(t_loss)

#Classification

p_c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = x1_h2_c, labels = y1)) #Photo
s_c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = x2_h2_c, labels = y2)) #Sketch




#semantic loss

sem_loss = tf.reduce_mean(tf.losses.mean_squared_error(v1_h2, x1_h21) + tf.losses.mean_squared_error(v2_h2, x2_h21))

#Domain loss

d_loss = tf.reduce_mean(domain_loss) + triplet_loss

#Decoder loss

dec_loss = tf.reduce_mean(tf.losses.mean_squared_error(x1_he_d, x2)) + tf.reduce_mean(tf.losses.mean_squared_error(x2_he_d, x1))

#Reg

reg1 = tf.reduce_mean(tf.norm(x1_h21)) + tf.reduce_mean(tf.norm(x2_h21))  #+ tf.reduce_mean(tf.norm(w11-w21)) + tf.reduce_mean(tf.norm(w12-w22)) + tf.reduce_mean(tf.norm(b11-b21)) + tf.reduce_mean(tf.norm(b12-b22))
reg2 = tf.reduce_mean(tf.norm(v1_h2)) + tf.reduce_mean(tf.norm(v2_h2))
#Tot_loss

#FINAL loss = s_c_loss + p_c_loss + d_loss + sem_loss + dec_loss + 0.1 * (reg1 + reg2) 
loss = 0.1*s_c_loss + 0.1*p_c_loss + 0.01*d_loss + sem_loss + dec_loss #+ 0.1 * (reg1 + reg2) 
# In[17]:


#accuracy calculation for classification

a_p = tf.nn.softmax(x1_h2_c) #photo
a_s = tf.nn.softmax(x2_h2_c) # sketch

correct_prediction1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a_p, 1), tf.argmax(y1, 1)), tf.float32))
correct_prediction2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a_s, 1), tf.argmax(y2, 1)), tf.float32))

pic_acc = correct_prediction1
sk_acc = correct_prediction2

#Domain accuracy

correct_domain_pred = tf.equal(tf.argmax(domain_pred, 1), tf.argmax(onehot_d_lab, 1))
domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

# In[18]:


loss_t = tf.train.AdamOptimizer(0.001).minimize(loss) #For the semantic and classification losses together
loss_t1 = tf.train.AdamOptimizer(0.0001).minimize(d_loss) #For the semantic and classification losses together


init=tf.global_variables_initializer()


# In[20]:


num_epoch=150
batch_size = 20

with tf.Session() as sess:
    
    sess.run(init)
    total_batch = int(len(anchor)/batch_size)
    
    for epoch in range(num_epoch):
        
        p1 = float(epoch) / num_epoch
        p2 = 2. / (1. + np.exp(-10. * p1))

        cl_p = 0.0
        cl_s = 0.0
        triplet = 0.0
        
        p_acc = 0.0
        s_acc = 0.0
        d_acc = 0.0
        
        sem_l = 0.0
        
        for batch in range(total_batch):
            
            s = batch * batch_size
            e = (batch + 1) * batch_size
            
            #Create next batch
            i1, i2, i3, l1, l2, u1, u2 =  next_batch(s, e, anchor, pos, neg, anchor_l, pos_l, anchor_wv, pos_wv)
            
            
            #Total loss
            
            _, l6 = sess.run([loss_t1,d_loss], feed_dict = {x1: i1, x2: i2, x3: i3, v1: u1, v2: u2, d1: onehot_d_lab, l: p2})
            
            _, l11, l21, l31, l41, l51, l61, l71 = sess.run([loss_t, p_c_loss, s_c_loss, domain_acc, pic_acc, sk_acc, sem_loss, triplet_loss], feed_dict = {x1: i1, x2: i2, x3: i3, y1: l1, y2: l2, v1: u1, v2: u2, d1: onehot_d_lab, l: p2})
             
            
            cl_p = cl_p + l11
            cl_s = cl_s + l21
            
            d_acc = d_acc + l31
            
            p_acc = p_acc + l41
            s_acc = s_acc + l51
            
            sem_l = sem_l + l61
            triplet = triplet + l71
            

        
        d_acc = d_acc / total_batch
        
        cl_p = cl_p / total_batch
        cl_s = cl_s / total_batch
        
        s_acc = s_acc / total_batch
        p_acc = p_acc / total_batch
        
        sem_l = sem_l / total_batch
        
        
        #testing
        test_pic, test_sk = sess.run([pic_acc, sk_acc], feed_dict = {x1: test_X, x2: test_Y, y1: onehot_test_Lx, y2: onehot_test_Ly, v1: test_wvx1, v2: test_wvy1})


        #print
        
        print("Photo Epoch " + str(epoch) + ", sem_acc= " + "{:.3f}".format(sem_l)+  ", classification= " + "{:.3f}".format(cl_p)+ ",  pic_acc= " + "{:.3f}".format(p_acc))
        print("Test Pic ", str(test_pic))
        print("Sketch Epoch " + str(epoch) + ", domain_acc= " + "{:.3f}".format(d_acc)+  ", classification= " + "{:.3f}".format(cl_s)+ ", sk_acc= " + "{:.3f}".format(s_acc))
        print("Test Sk ", str(test_sk))
        print("triplet-loss", triplet)
        
        if s_acc >= 0.6 and p_acc >= 0.6 and sem_l < 0.5:
            sk, pic, wv1, wv2 = sess.run([x2_h2, x1_h2, v1_h2, v2_h2], feed_dict={x1: test_X, x2: test_Y, v1: test_wvx1, v2:test_wvy1})

            pic_s, sk_s = sess.run([x1_h2, x2_h2], feed_dict = {x1: train_X, x2: train_Y, v1: train_wvx1, v2: train_wvy1})
            pic_un, sk_un = sess.run([x1_h2, x2_h2], feed_dict = {x1: X2, x2: Y2, v1: unseen_wvx1, v2: unseen_wvy1})
            #pic_s = np.array(pic_s, np.float32)
            sk_s = np.array(sk_s, np.float32)
            pic_un = np.array(pic_un, np.float32)
            sk_un = np.array(sk_un, np.float32)
    sio.savemat('feat_un.mat', {'photo_un': pic_un, 'sketch_un':sk_un, 'photo_lab': Lx2, 'sketch_lab':Ly2, 'glove':unseen_wvx1})


print("Training Complete")
    
sess.close() 


# In[23]:

#-----------------PLOT -------------------------------------
from sklearn.manifold import TSNE
'''
tsnea = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsnea.fit_transform(sk)

tsneb = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results1 = tsneb.fit_transform(pic)
'''
tsnec = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results2 = tsnec.fit_transform(pic_un)

tsned = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results3 = tsned.fit_transform(sk_un)


# In[28]:



'''
ax = sns.scatterplot(tsne_results[:,0], tsne_results[:,1], np.squeeze(test_Ly), palette=sns.color_palette("hls", 220), alpha=0.6)
plt.savefig('sketchtsne.png', bbox_inches='tight')
plt.clf()

# In[29]:

bx = sns.scatterplot(tsne_results1[:,0], tsne_results1[:,1], np.squeeze(test_Lx), palette=sns.color_palette("hls", 220), alpha=0.6)
plt.savefig('imagetsne.png', bbox_inches='tight')
plt.clf()
'''
cx = sns.scatterplot(tsne_results2[:,0], tsne_results2[:,1], np.squeeze(Lx2), palette=sns.color_palette("hls", 30), alpha=0.6)
plt.savefig('u_imagetsne.png', bbox_inches='tight')
plt.clf()

dx = sns.scatterplot(tsne_results3[:,0], tsne_results3[:,1], np.squeeze(Ly2), palette=sns.color_palette("hls", 30), alpha=0.6)
plt.savefig('u_sketchsne.png', bbox_inches='tight')
plt.clf()
# In[30]:


#-----------------DISTILLATION -------------------------------------


feature_dim = 256
h1 = 128
h2 = 64

xk = tf.placeholder(tf.float32, [None, h2]) #photo feature
yk = tf.placeholder(tf.float32, [None, h2]) #sketch feature
x = tf.placeholder(tf.float32, [None, feature_dim]) #photo feature
y = tf.placeholder(tf.float32, [None, feature_dim]) #sketch feature

# Encoder for photo stream
w11k = tf.Variable(tf.random_normal([feature_dim, h1], stddev = 0.01), name= 'w11k')
w12k = tf.Variable(tf.random_normal([h1, h2], stddev = 0.01), name= 'w12k')

b11k = tf.Variable(tf.random_normal([h1], stddev = 0.01), name= 'b11k')
b12k = tf.Variable(tf.random_normal([h2], stddev = 0.01), name= 'b12k')

#Encoder for sketch stream
w21k = tf.Variable(tf.random_normal([feature_dim, h1], stddev = 0.01), name= 'w21k')
w22k = tf.Variable(tf.random_normal([h1, h2], stddev = 0.01), name= 'w22k')

b21k = tf.Variable(tf.random_normal([h1], stddev = 0.01), name= 'b21k')
b22k = tf.Variable(tf.random_normal([h2], stddev = 0.01), name= 'b22k')

is_training = tf.placeholder_with_default(False, (), 'is_training')

#Photo
x1_h1k = (tf.add(tf.matmul(x, w11k), b11k))
x1_h1k = tf.layers.batch_normalization(x1_h1k,training = is_training)
x1_h2k = (tf.add(tf.matmul(x1_h1k, w12k), b12k))

#sketch
x2_h1k = (tf.add(tf.matmul(y, w21k), b21k))
x2_h1k = tf.layers.batch_normalization(x2_h1k,training = is_training)
x2_h2k = (tf.add(tf.matmul(x2_h1k, w22k), b22k))


lossk1 = tf.reduce_mean(tf.abs(tf.subtract(tf.math.square(x1_h2k),tf.math.square(xk))))
lossk2 = tf.reduce_mean(tf.abs(tf.subtract(tf.math.square(x2_h2k),tf.math.square(yk))))
lossk = lossk1 + lossk2

loss_k = tf.train.AdamOptimizer(0.01).minimize(lossk)


def next_batch_kd(s, e, input1, input2, l1, l2):
    
    inp1 = input1[s:e,:]
    inp2 = input2[s:e,:]
    
    lab1 = l1[s:e,:]
    lab2 = l2[s:e,:]
        
    return inp1, inp2, lab1, lab2


init=tf.global_variables_initializer()


# In[20]:


num_epoch=20
batch_size = 20

with tf.Session() as sess:
    
    sess.run(init)
    total_batch = int(len(train_X)/batch_size)
    
    for epoch in range(num_epoch):
        mse_l1 = 0.0 
        mse_l2 = 0.0      
        for batch in range(total_batch):
            
            s = batch * batch_size
            e = (batch + 1) * batch_size
            
            #Create next batch
            k1, k2,i1, i2 =  next_batch_kd(s, e, pic_s, sk_s, train_X, train_Y)
            
            #Total loss          
            _,m1, m2 = sess.run([loss_k,lossk1,lossk2], feed_dict = {x: i1, y: i2, xk: k1, yk: k2})     
            mse_l1 = mse_l1 + m1
            mse_l2 = mse_l2 + m2           
        
        mse_l1 = mse_l1 / total_batch
        mse_l2 = mse_l2 / total_batch
        
        
        #testing
        _,test_pic, test_sk = sess.run([loss_k,lossk1,lossk2], feed_dict = {x: test_X, y: test_Y, xk:pic, yk:sk })

        #print  
        print("Photo Epoch " + str(epoch) + ", Train loss= " + "{:.3f}".format(mse_l1)+ "Sketch Epoch " + str(epoch) + ", Train loss = " + "{:.3f}".format(mse_l2))     
        print("Photo Epoch " + str(epoch) + ", mse_l1= " + "{:.3f}".format(test_pic)+ "Sketch Epoch " + str(epoch) + ", mse_l2= " + "{:.3f}".format(test_sk))
        
        #if s_acc >= 0.90 and p_acc >= 0.90 and sem_l < 12.0:
        #sk_kd, pic_kd = sess.run(loss_k, feed_dict={x: test_X, y: test_Y})
        
    pic_un, sk_un = sess.run([x1_h2k, x2_h2k], feed_dict = {x:X2, y:Y2})
    pic_un = np.array(pic_un, np.float32)
    sk_un = np.array(sk_un, np.float32)
    sio.savemat('feat_kd_un.mat', {'photo_feat_un': pic_un, 'sketch_un':sk_un, 'photo_lab': Lx2, 'sketch_lab':Ly2})


print("Distillation Complete")
    
sess.close() 

#mean_ap = eval_cls_map(sk_un, pic_un, Ly2, Lx2)
#print(mean_ap)
'''
tsnea = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsnea.fit_transform(sk_kd)

tsneb = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results1 = tsneb.fit_transform(pic_kd)

tsnec = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results2 = tsnec.fit_transform(pic_un)

tsned = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results3 = tsned.fit_transform(sk_un)


# In[28]:


ax = sns.scatterplot(tsne_results[:,0], tsne_results[:,1], np.squeeze(test_Ly), palette=sns.color_palette("hls", 220), alpha=0.6)
plt.savefig('kdsketchtsne.png', bbox_inches='tight')
plt.clf()

bx = sns.scatterplot(tsne_results1[:,0], tsne_results1[:,1], np.squeeze(test_Lx), palette=sns.color_palette("hls", 220), alpha=0.6)
plt.savefig('kdimagetsne.png', bbox_inches='tight')
plt.clf()

cx = sns.scatterplot(tsne_results2[:,0], tsne_results2[:,1], np.squeeze(Lx2), palette=sns.color_palette("hls", 30), alpha=0.6)
plt.savefig('kdu_imagetsne.png', bbox_inches='tight')
plt.clf()

dx = sns.scatterplot(tsne_results3[:,0], tsne_results3[:,1], np.squeeze(Ly2), palette=sns.color_palette("hls", 30), alpha=0.6)
plt.savefig('kdu_sketchsne.png', bbox_inches='tight')
plt.clf()
'''


def gen_sim_mat(class_list1, class_list2):
    """
    Generates a similarity matrix
    Args:
        class_list1: row-ordered class indicators
        class_list2:
    Returns: [N1 N2]
    """
    c1 = np.asarray(class_list1)
    c2 = np.asarray(class_list2)
    sim_mat = np.matmul(c1, c2.T)
    sim_mat[sim_mat > 0] = 1
    return sim_mat



