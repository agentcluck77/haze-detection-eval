Jian Hong Tan, [9/11/25 11:33]
I think your model skews towards normal/cloud class. it seems to misclassify smoke as cloud

Jian Hong Tan, [9/11/25 11:34]
i think around 196 out of 488 smoke images were classified as cloud. FYI.

Jian Hong Tan, [9/11/25 11:35]
if you are retraining, do fix the class mapping.

Jian Hong Tan, [9/11/25 10:55]
hmm I think then class mapping is wrong for your inference script

Jian Hong Tan, [9/11/25 10:55]
0 = smoke (includes wildfire)
1 = haze
2 = normal (cloud, land, seaside, dust)