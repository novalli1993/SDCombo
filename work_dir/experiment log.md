# Experiment log
Database: Stanford2D3D

## SDBottleneck
simply concatenate the seg and depth
|No.|Mark|Set|Pre-trained|Initial lr|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230803_204453|[Set 1][^set1]|None|0.001|52.9|19.8|The loss shocks|
|2|20230804_123637|[6], [3]|[ade20k][^ade20k]|0.001|64.9|34.3|not so good, some classes 0|
|3|20230804_204744|[6], [3]|[ade20k][^ade20k]|0.0001|72.7|44.9|4 epochs on small dataset, looks good|
|4|20230804_222235|[Set 1][^set1]|[ade20k][^ade20k]|0.0001|65.1|36.4|best on epoch 7, lr=2.140e-5, but still really bad|
|5|20230805_154709|[6], [3]|[ade20k][^ade20k]|5e-05|76.2|53.6|looks good|
|6|20230805_202214|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|70.4|43.8|best on epoch 4, lr=2.938e-5. evaluating on size=1080 has no benefit but more time|
|7|20230806_090018|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|66.9|40.7|best on epoch 7, lr=5.94e-6. not good. maybe the crop size is too small|
|8|20230806_134041|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|68.4|40.3|without depth. almost the same. best on epoch 5, lr: 2.072e-5|

## SDHead
|No.|Mark|Set|Pre-trained|Initial lr|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230806_134041|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|68.4|40.3|model is much big, set the batch size to 4|
|1|20230806_211145|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|63.6|32.7|bad|

## SDBottleneck (update 0807)
Add some convolution layers
|No.|Mark|Set|Pre-trained|Initial lr|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230807_164017|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|68.1|40.9|best on epoch 7, lr=1.6e-6. not so good, two classes got 0.|
|2|20230807_204708|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|67.8|41.6|interrupted on epoch 8. best on epoch 5, lr=2.072e-5. bigger data only a little improvement.|
|3|20230808_085650|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|67.1|37.8|test the Focal Loss. no obvious increase on memory and training time. not good, and still have classes with 0 acc.|
|3|20230808_131252|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|68.4|40.5|change the mix in Neck from fixed channel=32 to channel=input. a little slow. unstable, and need more epochs to train. best on epoch 7, lr=5.94e-6.|
|3|20230808_205841|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|68.1|40.6|best on epoch 7, lr=5.94e-6.|
|4|20230809_120044|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|69.9|43.4|set resize=1~0.25, crop size=256. best on epoch 9, lr=1e-7. looks good, but not enough. some classes is 0, which is strange.|
|5|20230809_205650|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|67.2|41.6|FocalLoss(alpha=1, gamma=4, weight=None, ignore_index=0). best on epoch 9, lr=1e-7. different class got different effect of the loss function changing, at least no class is 0.|
|6|20230810_130727|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|67.2|41.4|set resize=0.5~0.25.|
|7|20230810_213258|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|64.7|35.8|set resize=1~0.2, crop size=125, batch-size=32, val.batch-size=4, epochs=20, cos=[5,1e-7]. FocalLoss(alpha=0.5, gamma=2, weight=None, ignore_index=0). totally bad.|
|8|20230811_064658|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|69.4|43.7|set resize=1~0.25, crop size=256, batch-size=8, val.batch-size=1, epochs=10, cos=[9,1e-7]. normalize depth instead of simple mean. best on epoch 7, lr=1.6e-6. need to change the model.|

## SDBottleneck (change model)
Add some convolution layers
|No.|Mark|Set|Pre-trained|Initial lr|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230811_112905|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|69.9|44.1|add a convolution layer on depth. use "adaptive_avg_pool2d" instead of "interpolate" when concatenate two tensor. best on epoch 9, lr=1e-7. got some improvement. mIoU and mAcc shake.|
|2|20230811_153904|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|70.1|44.4|use a convolution layer to downsample the depth before concatenate. best on epoch 8, lr=1.6e-7. mIoU and mAcc shake.|
|3|20230811_200817|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|73.9|47.9|crop size=512, batch size=2. best on epoch 7, lr=5.94e-6. better, but still imbalance.|

## SDBottleneck, SDHead
Use SDHead instead of UperNet
|No.|Mark|Set|Pre-trained|Initial lr|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230812_124122|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|67.1|39.8|crop size=256, batch size=16, val.batch_size=int(batch size/2). best on epoch 3, lr=3.753e-5. mIoU and mAcc shake.|
|2|20230812_210904|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|66.8|38.5|val.batch_size=1. epochs=20, cos=[19,1e-7]. best on epoch 2, lr=4.966e-5. mIoU and mAcc shake.|
|3|20230813_070127|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|68.2|42.0|epochs=10, cos=[9,1e-7]. losses = nn.functional.cross_entropy(inputs, target, ignore_index=0). best on epoch 4, lr=2.938e-5. mIoU and mAcc shake.|
|4|20230813_090649|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|69.1|42.7|without depth. best on epoch 6, lr=1.258e-5. no obvious difference. depth don't make any effect.|
|5|20230813_130704|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|67.8|42.3|change a lot. best on epoch 2, lr=4.85e-5.|
|6|20230813_152854|[Set 4][^set4]|None|5e-05|48.7|17.9|No pretrain. bad. imbalance.|
|7|20230813_203334|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|66.4|39.6|pretrain. FocalLoss. resize=0.5~0.2. best on epoch 8, lr=1.6e-6. mIoU and mAcc shake.|
|8|20230814_070247|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|65.9|37.3|freeze the InternImage. interrupt on epoch 3. imbalance.|
|9|20230814_074356|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|69.2|42.3|unfreeze the InternImage. best on epoch 3, lr=3.7525e-05. imbalance.|

## ResNet Fusion
Based on ResNet. Fuse with depth.
|No.|Mark|Set|Pre-trained|Initial lr|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230814_154152|[Set 4][^set4]|None|5e-05|55.0|19.0|batch-size=8. best on epoch 3, lr=3.7525e-05. imbalance, bad.|
|2|20230814_194721|[Set 2][^set2]|None|5e-05|46.3|9.8|ResNet of [3, 3, 5, 2]. batch-size=6. too slow. interrupted on epoch 2.|
|2|20230815_064644|[Set 4][^set4]|None|5e-03|48.4|11.4|ResNet of [3, 3, 5, 2]. batch-size=4. imbalance, bad.|
|3|20230816_082802|[Set 4][^set4]|None|5e-03|46.4|9.9|model is wrong. batch-size=4. imbalance, bad.|
|4|20230816_123716|[Set 4][^set4]|None|5e-05|38.2|6.0|corrected model. base size=1024, crop size=126, max=base/8, min=base/4, val size=crop*2, batch size=16. epochs=10, cos=[9,1e-7]. best on epoch 3, lr=3.7525e-05. imbalance, bad.|
|5|20230816_153418|[Set 4][^set4]|None|5e-05|51.9|16.1|crop size=256, max=base/4, min=base/1, val size=crop, batch size=4. interrupted on epoch 1, lr=5.0000e-05. bigger crop size better.|
|6|20230816_162627|[Set 4][^set4]|None|5e-05|52.8|18.6|add downsample in the begin, like InternImage. batch size=16. best on epoch 1, lr=4.8495e-05. terrible.|

## InternImage Fusion
Based on InternImage. Fuse with depth.
|No.|Mark|Set|Pre-trained|Initial lr|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230816_180352|[Set 4][^set4]|None|5e-05|54.5|20.6|without pretrained. best on epoch 3, lr=3.7525e-05.|
|2|20230816_210331|[Set 1][^set1]|[ade20k][^ade20k]|5e-05|68.7|41.0|with pretrained. batch size=8 best on epoch 2, lr=4.4163e-05. acc and mIoU shake. best IoU of 'beam' on epoch 2, but still only 1.1.|
|3-1|20230817_093254|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|65.6|35.8|InternImage * 2, f_rgb = in_rgb[i] + rgb, 1 epoch.|
|3-2|20230817_100407|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|68.6|38.7|InternImage * 2, f_rgb = in_rgb[i], 1 epoch.|
|3-3|20230817_103124|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|66.6|34.4|InternImage * 2, f_rgb = in_rgb[i], 1 epoch.|
|4|20230817_112655|[Set 4][^set4]|[ade20k][^ade20k]|5e-05|70.4|43.7|II4sII, f_rgb = in_rgb[i], 10 epoch. best on epoch 4, lr: 2.9383e-05. imbalance.|


## Control group
|No.|Mark|Set|Pre-trained|Initial lr|Best epoch|Best LR|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230818_132716|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|5|1.2575e-05|59.2|29.5|test: depth = rgb. control group. imbalance.|
|2|20230818_182654|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|7|5.9372e-06|59.8|32.0|DeepLabV3 with MobileNet Large. control group.|
|3|20230819_174932|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|54.7|24.5|test: depth = rgb. control group.|
|4|20230824_120706|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|63.4|38.8|DeepLabV3 with MobileNet Large. control group. 20230823_144153: same set|
|5|20230824_164806|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|3|3.7525e-05|62.3|38.4|DeepLabV3 with MobileNet Large. control group. 20230823_144153: same set|
|6|DL_20230827_185154|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|7|5.9372e-06|64.8|40.6|DeepLabV3 with MobileNet Large. control group.|
|7|DL_20230830_072554|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|19|1.0000e-07|56.9|28.8|DeepLabV3 with MobileNet Large. control group. **no pretrained.**|
|78|DL_20230831_070701|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|18|4.4029e-07|63.9|40.2|DeepLabV3 with MobileNet Large. control group.|

## DeepLab Fusion
Based on DeepLabV3_MobileNet_large. Fuse with depth.
|No.|Mark|Set|Pre-trained|Initial lr|Best epoch|Best LR|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230817_171840|[Set 4][^set4]|None|5e-05|7|5.9372e-06|60.1|26.6|imbalance.|
|2|20230817_205825|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|8|1.6047e-06|68.0|41.1|big dataset matters.|
|3|20230818_082204|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|7|5.9372e-06|66.9|36.6|update fusion part. batch size = 4. imbalance.|
|4|20230819_020022|[Set 1][^set1]|None|5e-05|9|1.0000e-07|61.1|31.2|deep change on DeepLabV3 with MobileNet Large. looks good.|
|5|20230819_124508|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|5|2.0717e-05|62.8|31.3|not better.|
|6|20230819_231747|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|64.8|35.9|not enough.|
|7|20230821_104849|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|63.2|35.1|20230821_062516: same set.|
|8|20230827_224813|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|64.4|38.6|20230828_033835: same set.|
|9|20230830_211613|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|63.5|37.0|20230830_151212|


## DeepLab Fusion test with HHA
Based on DeepLabV3_MobileNet_large. Fuse with depth oof HHA.
|No.|Mark|Set|Pre-trained|Initial lr|Best epoch|Best LR|Best acc|Best IoU|Notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-|
|1|20230819_153749|3|[DeepLabV3][^DeepLabV3]|5e-05|8|1.6047e-06|~~75.3~~|~~44.6~~|**this is just a test. train and eval on same data.**|
|2|20230820_083812|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|66.8|35.9|imbalance|
|3|20230820_102239|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|8|1.6047e-06|67.5|37.5|focal loss $\rightarrow$ cross entropy loss. imbalance. focal loss seems no effect.|
|4|20230820_130824|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|7|5.9372e-06|67.1|37.4|change focal loss. don't get much difference.|
|5|20230820_183221|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|69.9|43.7|from model_20230820_130824_9. imbalance. strange 0.|
|6|20230820_213959|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|70.3|46.8|crop size = 516. batch size=4.|
|7|20230821_062516|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|8|1.6047e-06|66.2|40.6|20230820_130824: weight decay=0.01.|
|8|20230821_152810|[Set 4][^set4]|None|5e-05|6|1.2575e-05|61.7|29.7|no pretrained. no difference on imbalance classes.|
|9|20230821_205512|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|49|2.1246e-05|70.4|44.9|cos=[9,1e-6], 50 epochs. more epochs improve less than more data.|
|10|20230822_133737|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|7|5.9372e-06|66.6|38.9|cos=[9,1e-7], 10 epochs. Cross Entropy Loss.|
|11|20230822_182357|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|13|1.1404e-05|71.7|47.7|cos=[19,1e-7], 20 epochs. Focal loss.|
|12|20230823_144153|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|66.9|43.0||
|13|20230823_213627|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|||70.0|47.9|crop size=512|
|14|20230824_211913|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|8|1.6047e-06|66.5|41.9|10 epoch test 1.|
|14|20230825_225522|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|16|3.1071e-06|67.5|44.1|20 epoch test 1. interrupted on 16.|
|15|20230826_102647|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|7|5.9372e-06|70.8|46.8|crop size=512|
|16|20230827_053357|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|19|1.0000e-07|68.5|44.4|20 epoch test 2.|
|17|20230828_033835|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|8|1.6047e-06|66.6|41.3|20230827_224813: same set. min = max = base size|
|18|20230828_124303|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|63.6|36.7|min = max = base size=512|
|19|20230828_195406|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|66.6|41.0|fix the bug: ~~min = max = base size~~.|
|20|20230829_071557|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|28|3.2760e-05|68.3|43.6|30 epochs, cos=[5, 1e-7]|
|21|20230830_151212|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|8|1.0000e-07|66.7|41.0|10 epochs, 20230830_211613|
|22|20230831_220805|[Set 4][^set4]|[DeepLabV3][^DeepLabV3]|5e-05|34|2.0965e-06|70.4|45.6|40 epochs|
|23|20230901_131623|[Set 1][^set1]|[DeepLabV3][^DeepLabV3]|5e-05|9|1.0000e-07|70.7|47.2|crop_size: 512|

SDCombo with HHA: 20230830_151212_8
SDCombo with Depth: 20230830_211613_9
DeepLabV3: DL_20230831_070701_18
SDCombo with HHA 512: model_20230901_131623_9
SDCombo with HHA 40 epochs: model_20230831_220805_34

[^set1]: [1, 2, 3, 4, 6], [5]

[^set2]: [1, 3, 5, 6], [2, 4]

[^set3]: [2, 4, 5], [1, 3, 6]

[^set4]: [4], [3]

[^ade20k]: init_from_ade20k_internimage

[^DeepLabV3]: init_from_deeplabv3_mobilenet_v3_large
