# 01 安裝準備
**請確認是否已安裝git**</br>
**github目前沒有資料集的圖片，請自己下載後把整個資料夾放到melanoma_external_256\train**</br>
[Dataset](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-224x224-resized)
申辦github帳號，然後給我你的githubID或email，我會幫你加入團隊

# 02 github設定
**請先安裝 git for windows**</br>
安裝完成後，開啟cmd輸入
>git config --global user.name "name"</br>
git config --global user.email "Email"</br>

## 設定個人token
1. 到github網站 GitHub Settings -> Developer settings -> Personal access tokens (classic)
2. 點擊 Generate new token (classic)
3. Note 寫 my_project，期限選 No expiration
4. **勾選 repo (所有權限)**
5. 產生的 Token 請先複製下來存好


# 03 安裝新的annaconda
開啟command line 用cd指令(舉例來說，要切換到桌面，輸入 cd C:\Users\使用者名稱\desktop)切換到你要的資料夾，輸入</br>
> git clone https://github.com/thefirstwangry/multimodal-skin-lesion-classification.git</br>

找到資料夾中 **pytorch_conda.yml** 檔案，確認有內容後回到annaconda prompt 輸入</br>
> conda env create -f pytorch_conda.yml</br>
>conda activate <環境名稱></br>

然後驗證是否有安裝好cuda(事前請先更新顯卡驅動)</br>

> python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'顯卡型號: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"無\"}')"</br> 

接下來請**把資料集搬到melanoma_external_256\train**
然後創兩個資料夾 
weights, logs</br>
安裝大功告成

# 04 版本控制
## 開發前準備
**每次要改東西前請先使用**</br>
> git checkout main</br>
> git pull origin main</br>
> git checkout -b describe_ur_request</br>

**在describe_ur_request那邊，簡述你要更新的東西是什麼**</br>

## 開發完成後
> git add filename </br>
> git commit -m "text"</br>

**在text那邊，一句話敘述本次修改的主要工作**
> git push origin describe_ur_request

把describe_ur_request換成剛剛checkout -b 後面輸入的東西  





