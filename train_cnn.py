from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
train_images = X_train[:1500]
train_labels = y_train[:1500]
test_images = X_test[:1500]
test_labels = y_test[:1500]
test_images=torch.tensor(test_images)
test_lables=torch.tensor(test_labels)

conv=Conv_operation(8,3,1,1)
pool=Maxpooling(2)
softmax=Softmax(14*14*8,10)

def cnn_forward_prop(image,label):
    out= conv.forward_prop((image/255) -0.5)
    out=torch.tensor(out)
    out=pool.forward_prop(out)
    out=softmax.forward_prop(out)
    
    #calculate cross entropy loss and accuracy
    loss=-torch.log(out[label])
    acc=torch.where(torch.argmax(out)==label,1,0)
    
#     out_p=torch.argmax(out_p)
#     label=label.float()
#     out_p=out_p.float()*(label>0).float()
#     accuracy_eval=out_p*(out_p==label).float()
    
#     accuracy_eval=1 if torch.argmax(out_p)==label else 0
#     correct+=(out_p==label).sum().item()
#     accuracy_eval=100*correct/total
    
    return out,loss,acc
def train_cnn(image,label,learning_rate=0.000000005):
    #forward
    out,loss,acc=cnn_forward_prop(image,label)
    #calculate initial gradient
    gradient=torch.zeros(10)
    gradient[label]=-1/out[label]
    
    #backward
    grad_back=softmax.backward_prop(gradient,learning_rate)
    grad_back=pool.backward_prop(grad_back)
    grad_back=conv.backward_prop(grad_back,learning_rate)
    return loss,acc
  
  
  for epoch1 in range(2):
  print('Epoch %d ->'% (epoch1 +1))
  

  # shuffle the training data
  shuffle_data = torch.randperm(len(train_images))
  train_images = train_images[shuffle_data]
  train_labels = train_labels[shuffle_data]
  train_images=torch.tensor(train_images)
  test_images=torch.tensor(test_images)

  

  #training the CNN
  loss = 0.0
  num_correct = 0

  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 0:
      print('%d steps out of 100 steps: Average Loss %.3f and Accuracy: %d%%' %(i+1, loss/100, num_correct))
      loss = 0
      num_correct = 0

    l1, acc = train_cnn(im, label)
    loss += l1

    num_correct +=acc
