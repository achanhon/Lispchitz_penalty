# Lispchitz_penalty

## see https://hal.archives-ouvertes.fr/hal-01773170

## this code is for research only. Contact ONERA for any other uses.


baseline detection

test perf : 
dfc/test 99.97749668094104 83.245932236481
vedai/test 99.99354538247503 65.53665586577912
saclay/test 99.96784363963216 62.04334517330438
little_xview/test 99.9371195024978 56.63833718422258
dota/test 99.9605172236759 70.94145036562024
isprs/test 99.81418968158602 69.46556834739555
isprs/train 99.86852320504936 68.15148221965569

test on testing crop
accuracy and IoU 99.88812307463672 70.79918231512892

test on training crop
accuracy and IoU 99.97540974924385 89.31688568798543


=> both 
a severe overfitting
and an issue with tilling
