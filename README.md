# Quick Start

+ [1.環境準備](#1-environment)
+ [2.套件安裝](#2-package-install)
+ [3.STD](#3-STD)
+ [4.STR](#4-STR)
+ [5.STD+STR](#5-STD+STR)

<a name = '1-environment'></a>
## 1. 環境準備
- **作業系統**
    - Ubuntu 18.04.6
- **環境介紹**
    - Python 3.8.8
    - GPU RTX2080 * 1 + V100 * 1
    - CUDA Version 10.2

<a name = '2-package-install'></a>
## 2. 套件安裝
```bash
    # Linux example
    conda create --name STDR 
    source activate STDR

    # clone repo
    git clone https://github.com/D641593/STDSTR.git
    cd STDSTR

    # this installs the right pip and dependencies for the fresh python
    conda install ipython pip

    # python dependencies
    pip install -r requirement.txt

    # install PyTorch with cuda-10.1
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

    # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
    # you need to delete the build directory before you re-build it.
    echo $CUDA_HOME
    # if $CUDA_HOME is not your cuda path, you have to define cuda path
    # >> export CUDA_HOME='/usr/local/cuda'  
    cd STD/assets/ops/dcn/
    python setup.py build_ext --inplace
```

<a name = '3-STD'></a>
## 3. Scene Text Detection ( STD )
```
# 此階段指令皆須在STD資料夾下執行
cd STD
```
- **STD 資料集**<br>
    請建立datasets資料夾
    ```
    mkdir datasets
    ```
    須在STD/datasets/logoHigh/放置訓練資料集，架構如下<br>
     ```
    |-logoHigh
        |- train_list.txt
        |- test_list.txt
        |- train_images
            |- img_1.jpg
            |- img_2.jpg
            |- img_3.jpg
            | ...
        |- train_gts
            |- img_1.txt
            |- img_2.txt
            |- img_3.txt
            | ...
        |- test_images
            |- img_1.jpg
            |- img_2.jpg
            |- img_3.jpg
            | ...
        |- test_gts
            |- img_1.txt
            |- img_2.txt
            |- img_3.txt
            | ...
    ```
    資料集下載如下<br>
    [datasets_STD](https://drive.google.com/file/d/1vJMsHk2TknBXF9sv0wXaAyKAtEwFF89T/view?usp=sharing)<br>

    
- **訓練** <br>
    訓練時需要編寫yml檔案，定義訓練時的架構與訓練資料<br>
    ```
        yml檔案名稱                 
    experiments/seg_detector/base_logoHigh.yaml                 ->  定義資料集
    experiments/seg_detector/logoHigh_resnet152_interd2v2.yaml  ->  定義架構    
    ```
    定義完成後，可使用以下指令開始訓練
    ```bash
    # example
    CUDA_VISIBLE_DEVICES=0 python train.py experiments/seg_detector/logoHigh_resnet152_interd2v2.yaml
    ```
    訓練完成後模型儲存在STD/workspace/SegDetectorModel-seg_detector/resnet152/L1BalanceCELoss/model<br>

- **偵測**<br>
    已訓練模型參數如下<br>
    [STD trained models](https://drive.google.com/drive/folders/1HORS6VOe6v_sb3AB4zwE8M39VoJqE4mF?usp=sharing)<br>
    請建立weight資料夾，將下載模型參數放置於此資料夾
    ```
    mkdir weight
    ```
    偵測可單張圖片或一整個資料夾，須給定yaml檔案和模型參數檔案<br>
    ```bash
    # 單張圖片
    CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/logoHigh_resnet152_interd2v2.yaml --image_path datasets/logoHigh/private/img_21000.jpg --resume weight/model_epoch_474_minibatch_810000 --box_thresh 0.5 --thresh 0.5
    # 資料夾 
    CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/logoHigh_resnet152_interd2v2.yaml --image_path datasets/logoHigh/private --resume weight/model_epoch_474_minibatch_810000 --box_thresh 0.5 --thresh 0.5
    # 輸出帶框圖片 --visualize
    CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/logoHigh_resnet152_interd2v2.yaml --image_path datasets/logoHigh/private --resume weight/model_epoch_474_minibatch_810000 --box_thresh 0.5 --thresh 0.5 --visualize
    ```
    輸出位置為 STD/demo_results 
<a name = '4-STR'></a>
## 4. Scene Text Recognition ( STR )
```
# 此階段指令皆須在STR資料夾下執行
cd ..
cd STR
```
- **字典**  <br>
    字典用以表示所有字對應的編號，若有一字典如下
    ```
    漫
    髮
    型
    工
    作
    室
    ```
    則標記為「漫」的圖片會對應到[0]，而標記為「髮型工作室」的圖片則會對應到[1 2 3 4 5]<br>
    本次使用的字典為
    ``` 
    STR/train_data/myDict.txt
    ```
- **STR 資料集**<br>
    須在STR/train_data/放置訓練資料集，架構如下<br>
     ```
    |-train_data
        |- train_list.txt
        |- train_crop
            |- img_1_1.jpg
            |- img_1_2.jpg
            |- img_2_1.jpg
            | ...
    ```
    資料集下載如下<br>
    [datasets_STR](https://drive.google.com/file/d/15PG4GS-vw-wxTDbuG0nOozIGkKpN-w--/view?usp=sharing)<br>

- **訓練**<br>
    須定義train.yaml
    已訓練模型參數下載連結如下<br>
    [STR trained models](https://drive.google.com/drive/folders/1Pi4mc6Q3wrQ2SB1f9YFuZ_UWn5j_bNFG?usp=sharing)<br>

    ```bash
    python train.py --yaml_file train.yaml
    ```
    訓練完成模型儲存於train_models/xxxxxx，xxxxxx資料夾於train.yaml內可定義

- **辨識**<br>
    須定義predict.yaml，可定義
    ```
    1. 預測圖片資料夾
    2. 圖片csv檔
    3. 輸出csv檔
    4. 模型參數檔
    5. 字典
    ```
    執行predict.py進行預測
    ```bash
    python predict.py --yaml_file predict.yaml
    ```

<a name = '5-STD+STR'></a>
## 5. STD + STR
- ****
    若須完整的偵測＋辨識，請將圖片放置STD/datasets/private資料夾內，並執行辨識指令
    ```
    cd STD
    # 資料夾 
    CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/logoHigh_resnet152_interd2v2.yaml --image_path datasets/logoHigh/private --resume weight/model_epoch_474_minibatch_810000 --box_thresh 0.5 --thresh 0.5
    ```
    執行 STDtoSTR.py，將STD/demo_results內的結果整合成empty.csv，並根據辨識結果切割STD/datasets/logoHigh/private的圖片至STR/train_data/private_high_crop
    ```
    cd ..
    python STDtoSTR.py --res_img_dir STD/demo_results/ --img_dir STD/datasets/private --output_dir STR/train_data/private_high_crop
    ```
    完成後定義predict.yaml，執行
    ```bash
    cd STR
    python predict.py --yaml_file predict.yaml
    ```
    得到ans.csv，為最終預測結果

    
    