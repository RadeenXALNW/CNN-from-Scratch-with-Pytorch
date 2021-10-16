class Maxpooling:
    def __init__(self,filter_size):
        self.filter_size=filter_size
    def image_region(self,image):
        new_height=image.shape[0]//self.filter_size
        new_width=image.shape[1]//self.filter_size
        self.image=image
        for i in range(new_height):


          for j in range(new_width):


            image_patch=image[(i*self.filter_size):(i*self.filter_size+self.filter_size),
                                 (j*self.filter_size):(j*self.filter_size+self.filter_size)]
            yield image_patch,i,j
    def forward_prop(self,image):

      height,width,num_filters=image.shape
      output=torch.zeros(height//self.filter_size,width//self.filter_size,num_filters)
      
      for image_patch, i, j in self.image_region(image):
        # image_patch=torch.flatten(image_patch,start_dim=0,end_dim=1)
        output[i,j]=torch.amax(image_patch)
      return output
    

    def backward_prop(self,d_L_dout): #dL_dout is the input from softmax layer
      dl_dinput=torch.zeros(self.image.shape)
      for image_patch,i,j in self.image_region(self.image):
        height,width,num_filters=image_patch.shape
        # max_val=torch.max(image_patch,dim=0)

  
            
        for i1 in range(height):

          for j1 in range(width):

                for k1 in range(num_filters):
                  x_pool=dl_dinput[i*self.filter_size+i1,j*self.filter_size+j1,k1]
                  mask=(x_pool==torch.amax(x_pool))
                  dl_dinput[i*self.filter_size+i1,j*self.filter_size+j1,k1]=mask*d_L_dout[i,j,k1]
                                  
                           
        return dl_dinput
