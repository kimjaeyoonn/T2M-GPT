# Utterance to Motion
1인 유튜버 영상으로부터 한국어 발화&모션 데이터셋 'KTubeUM'을 구축할 수 있고, 한국어 발화로부터 모션을 생성하는 모델을 학습 및 평가할 수 있습니다. 학습 모델은 [VL-KE-T5](https://github.com/AIRC-KETI/VL-KE-T5)의 언어인코더와 [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)의 기존 아키텍처를 결합하여 구성하였습니다. 모든 프로세스는 단일 GPU TITAN XP에서 진행힐 수 있습니다. 아래의 두 과정은 각각의 가상환경을 구성하여 진행할 수 있습니다.


## I. 데이터 추출 및 KTubeUM 데이터셋 구성
'construct_KTubeUM' 폴더에서 아래의 절차를 통해 학습에 필요한 한국어 발화 데이터셋을 구축할 수 있습니다.
### 1. 필요 패키지 설치 및 가상환경 설정
```bash
cd construct_KTubeUM
conda env create -f environment.yml
conda activate ktubeum
```
### 2. 데이터셋 구축 프로세스
'pose_json' 폴더 내에는 유튜브 영상 속 발화자에 대한 자세 추정 값이 SMPL-X 형식으로 존재합니다. 해당 데이터로부터 아래의 과정을 통해 학습에 필요한 데이터셋을 구성할 수 있습니다.
1. **json_to_npz.ipynb**

   모든 셀을 실행하여 **.**/pose_json 폴더에 존재하는 모든 json 확장자의 파일을 npz 확장자의 파일로 변환하여 ./my_pose 폴더에 저장합니다. 이는 다음 프로세스의 연산을 위한 과정이며, 기존 json 파일에서 cam_trans 값과 자세 추정 값 중 root 부분을 모든 프레임이 동일값을 가질 수 있도록 고정시킵니다. 또한, Low Pass Filter 연산을 적용하여 프레임 변환시 자세 추정 값의 떨림 현상을 보간합니다.
    
3. **raw_pose_processing.ipynb**
    
    모든 셀을 실행하여 ./my_data 폴더에 존재하는 모든 npz 확장자의 파일에 대해 연산을 수행한 뒤, 변환된 npy 파일을 ./joints 폴더에 저장합니다.
    
4. **motion_representation.ipynb**
    
    모든 셀을 실행하여 ./joints 폴더에 존재하는 모든 npy 확장자의 파일에 대해 연산을 수행한뒤, 변환된 npy 파일을 ./HumanML3D/new_joints 폴더에 저장합니다.
    
5. **cal_mean_variance.ipynb**
    
    mean, std 결과를 확인하는 단계입니다.
    
6. **animation.ipynb**
    
    모든 셀을 실행하여 ./HumanML3D/new_joints 폴더에 존재하는 모든 npy 확장자의 파일에 대해 ,mp4 영상을 저장하고 확인할 수 있습니다.

아래의 경로에서 최종적인 데이터 구축 결과를 확인할 수 있습니다.
```
./KTubeUM/
├── new_joint_vecs/
├── new_joints/
├── texts/
├── Mean.npy
├── Std.npy
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```

## II. 한국어 발화로부터 모션을 생성하는 모델 학습
'utterance_to_gesture' 폴더에서 아래의 절차를 통해 모델 학습 및 평가를 진행할 수 있습니다.
### 1. 필요 패키지 설치 및 가상환경 설정
```bash
conda env create -f environment.yml
conda activate u2m-gpt
```
발화 텍스트의 토큰화 및 임베딩을 위해 glove 모델을 다운로드합니다. 'glove' 폴더에 세 종류의 파일이 생성됩니다:
```bash
bash dataset/prepare/download_glove.sh
```
생성된 모션을 평가하기 위한 모션&발화 Feature extractors를 다운로드합니다. 'checkpoints' 폴더에 두 종류의 폴더가 생성됩니다:
```bash
bash dataset/prepare/download_extractor.sh
```
GT 모션과 예측된 모션의 joint를 SMPL 포멧으로 렌더링하기 위해 아래의 패키지를 다운로드합니다:
```bash
sudo sh dataset/prepare/download_smpl.sh
conda install -c menpo osmesa
conda install h5py
conda install -c conda-forge shapely pyrender trimesh mapbox_earcut
```
HumanML3D 데이터셋 기반의 사전 학습된 모델을 다운로드할 수 있습니다. 'pretrained' 폴더에 두 종류의 모델이 생성됩니다:
```bash
bash dataset/prepare/download_model.sh
```

### 2. 데이터셋 구성
학습에 사용할 데이터셋을 아래의 경로에 이동시켜 줍니다:
```
./dataset/KTubeUM/
├── new_joint_vecs/
├── texts/
├── Mean.npy
├── Std.npy
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```
HumanML3D 데이터셋 사용을 원하는 경우, [여기](https://github.com/EricGuo5513/HumanML3D/tree/main)를 참고하여 동일한 프로세스를 통해 구축할 수 있습니다.

### 3. 모델 학습

HumanML3D 데이터셋을 사용하여 학습을 진행한다면, '--dataname humanml3d'로 설정합니다.

### 3.1 VQ-VAE 모듈 학습

학습 결과에 대한 모든 파일은 'output' 폴더에 생성됩니다.

<details>
<summary>
VQ training
</summary>

```bash
python3 train_vq.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 100000 \
--lr-scheduler 50000 \
--nb-code 128 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname ktube \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE
```

</details>

### 3.2 GPT 모듈 학습

학습 결과에 대한 모든 파일은 'output' 폴더에 생성됩니다.

<details>
<summary>
GPT training
</summary>

```bash
python3 train_t2m_trans.py  \
--exp-name GPT \
--batch-size 16 \
--num-layers 9 \
--clip-dim 768 \
--embed-dim-gpt 1536 \
--nb-code 128 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth output/VQVAE/net_best_fid.pth \
--vq-name VQVAE \
--out-dir output \
--total-iter 100000 \
--lr-scheduler 50000 \
--lr 0.0001 \
--dataname ktube \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--eval-iter 10000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu
```

</details>

## 4. 모델 평가

### 4.1 VQ-VAE 모듈 평가
<details>
<summary>
VQ eval
</summary>

```bash
python3 VQ_eval.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 10000 \
--lr-scheduler 5000 \
--nb-code 128 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname ktube \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name TEST_VQVAE \
--resume-pth output/VQVAE/net_best_fid.pth
```

</details>

### 4.2. GPT 모듈 평가

<details>
<summary>
GPT eval
</summary>

```bash
python3 GPT_eval_multi.py  \
--exp-name TEST_GPT \
--batch-size 128 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 512 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth output/VQVAE/net_last.pth \
--vq-name VQVAE \
--out-dir output \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.0001 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--eval-iter 10000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu \
--resume-trans output/GPT/net_best_fid.pth
```

</details>


## 5. SMPL 렌더링
<details>
<summary>
SMPL Mesh Rendering 
</summary>

생성된 모션을 SMPL 렌더링하여 확인할 수 있습니다:

```bash
python3 render_final.py --filedir output/TEST_GPT/ --motion-list 000019 005485
```

</details>
