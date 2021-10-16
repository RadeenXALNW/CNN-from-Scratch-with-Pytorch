class Conv_operation:
    def __init__(self,num_filters,filter_size,stride_size,padding_size):
        self.num_filters=num_filters
        self.filter_size=filter_size
        self.padding_size=padding_size
        self.stride_size=stride_size
        self.conv_filter=torch.rand(num_filters,filter_size,filter_size)/(filter_size*filter_size)
    #image patching 
    def image_region(self,image):
        height,width=image.shape
        self.image=image
        for j in range((height-self.filter_size)+1):
            for k in range((width-self.filter_size)+1):
                image_patch=image[j:(j+self.filter_size),k:(k+self.filter_size)]
                yield image_patch,j,k
    def forward_prop(self,image):
#         assert single_sample.dim()==3, f'Input not 2D, given {single_sample.dim()}D'
        # image=torch.squeeze(image)
        height,width=image.shape
        padding_size=(self.filter_size-1)//2
        conv_out=torch.zeros(((height-self.filter_size+2*padding_size)//self.stride_size)+1,((width-self.filter_size+2*padding_size)//self.stride_size)+1,self.num_filters)
        for image_path,i,j in self.image_region(image):
            conv_out[i,j]= torch.sum(image_path*self.conv_filter)
        # conv_out = 1. / (1. + torch.exp(-conv_out))
        return conv_out
    def padding_size(self):
        return(self.filter_size-1)//2

        
    def relu(self,xa,derive=False):
      if derive:
        return torch.ceil(torch.clamp(xa,min=0,max=1)).detach
      return torch.clamp(xa,min=0).detach()


    def backward_prop(self,d_L_dout,learning_rate):
        dL_dF_params=torch.zeros(self.conv_filter.shape)
        for image_patch,i,j in self.image_region(self.image):
            for k in range(self.num_filters):
                dL_dF_params[k]+=image_patch*d_L_dout[i,j,k]
#         learning_rate=torch.tensor(learning_rate)
#         dL_dF_params=torch.tensor(dL_dF_params)
        self.conv_filter-=learning_rate*dL_dF_params
        return dL_dF_params
