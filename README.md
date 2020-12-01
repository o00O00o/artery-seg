# Methods that might bring about improvement
1. instead of using one frame per branch, use 2.5D per branch  
2. replace the DeepLabV3 architecture with Unet or other lighter structures
3. recode the validation phase using the method in paper or code
4. use mean teacher semi-supervised method to ultilize the consistency between frames 
5. use self-supervised setting as pretraining
# Current obtained results
original cosnet without testing phase: 0.855715, [0.99164621 0.85714468 0.88446912 0.82553191]  
cosunet without testing phase: 0.842708, [0.9920026  0.85655325 0.85530143 0.81627028]  
Vnet: 0.897314, [0.99243155 0.85889641 0.86579253 0.96725222]  

四类分别是：背景，血管内径，钙化斑块，软斑块  

num_labeled_data: 1366
num_unlabeled_data: 18858
num_validattion_data: 5104

model: Vnet(plane UNet)
epoch: 800

baseline:
class: [0.99097141 0.85512442 0.83146037 0.74127659]
mean: 0.8547 

tcsm:
class: [0.99504425 0.90335097 0.88608155 0.861352]
mean: 0.9114

consistency-weight 分别设置成：5， 10， 15. 得到的结果分别是：87， 86， 85.

