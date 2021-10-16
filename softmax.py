class Softmax:
    def __init__(self,input_node,softmax_node):
        self.weight=torch.randn(input_node,softmax_node)/input_node
        self.bias=torch.zeros(softmax_node)
    def forward_prop(self,image):
        self.last_input_shape=image.shape
        new_image=image.flatten()
        self.modified_input=new_image #to be  used in backpropagation
#         input_node,softmax_node=self.weight.shape
#         new_image=torch.squeeze(new_image,dim=1)
        new_image=new_image.cpu().detach().numpy()
        self.weight=self.weight.cpu().detach().numpy()
        self.bias=self.bias.cpu().detach().numpy()
        output_val=np.dot(new_image,self.weight)+self.bias
        self.weight=torch.from_numpy(self.weight)
        self.bias=torch.from_numpy(self.bias)
      
        output_val=torch.from_numpy(output_val)
        self.out=output_val
        exp_out=torch.exp(output_val)
        return exp_out/torch.sum(exp_out)
    
    def backward_prop(self,d_L_dout,learning_rate):
        for i, gradient in enumerate(d_L_dout):
            if gradient==0:
                continue
            
    #out(c)=e^tc/summation(e^ti)
    #   where S=summation(e^ti)
        t_exp=torch.exp(self.out)
        #SUM OF ALL e^totals
        S=torch.sum(t_exp)
        #gradients of output[i] against totals
        dout_dt=-t_exp[i]*t_exp/(S**2)
        dout_dt[i]=t_exp[i]*(S - t_exp[i]) / (S ** 2)
        
        #gradients of totals against weights,biases, input
        dt_dw=self.modified_input
        dt_db=1
        dt_dinput=self.weight
        
        #gradients of loss against totals
        dL_dt=gradient*dout_dt
        
        #gradients of loss against weights, biases and input
#         dt_dw=torch.unsqueeze(dt_dw,dim=0)
#         dl_dt=torch.unsqueeze(dl_dt,dim=0)
        dl_dw=torch.matmul(dt_dw.unsqueeze(0).t(),dL_dt.unsqueeze(0))
        dl_db=torch.mul(dL_dt,dt_db)
        dl_dinput=torch.matmul(dt_dinput,dL_dt)
        
        #update weights biases
        
        self.weight-=torch.mul(learning_rate,dl_dw)
        self.bias-=torch.mul(learning_rate,dl_db)
        
        # return dl_dinput.reshape(self.last_input_shape)
        return torch.reshape(dl_dinput,self.last_input_shape)
