import torch
import torch.nn as nn
#linear
layer=nn.linear(in_features=3,out_features=5,bias=True)
t1=torch.Tensor([1,2,3])
t2=torch.Tensor([[1,2,3]])

output2=layer(t2)

print(output2)

#view 
t=torch.tensor([1,2,3,4,5,6],[7,8,9,10,11,12])  #[2,6]
t_view1=t.view(3,4)
print(t_view1)

t_view2=t.view(4,3)
print(t_view2)
