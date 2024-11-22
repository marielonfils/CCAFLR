from SemaClassifier.classifier.Breast import breast_classifier as bc
from SemaClassifier.classifier.Images import ImageClassifier as ic
model=bc.MobileNet(0.001,0.001)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

model2=ic.ConvNet(14)
pytorch_total_params2 = sum(p.numel() for p in model2.parameters())
print(pytorch_total_params2)