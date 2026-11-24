import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from datasets import load_dataset

def download_lsun_samples_from_hf(max_per_class=500):
    os.makedirs("./LSUN_origin/bedroom", exist_ok=True)
    os.makedirs("./LSUN_origin/cat", exist_ok=True)
    
    # 1. 寝室（Bedroom）データのダウンロード
    # ※ Hugging Face上の公開サンプルリポジトリを利用
    print("Hugging Faceから寝室(bedroom)データを取得中...")
    try:
        bedroom_ds = load_dataset("pcuenq/lsun-bedrooms", split="train", streaming=True)
        count = 0
        for item in tqdm(bedroom_ds, desc="Saving bedrooms", total=max_per_class):
            if count >= max_per_class:
                break
            # 画像オブジェクトを取得して保存
            image = item["image"]
            image.save(f"./LSUN_origin/bedroom/{count:05d}.jpg")
            count += 1
    except Exception as e:
        print(f"bedroomのダウンロード中にエラーが発生しました: {e}")

    # 2. 猫（Cat）データのダウンロード
    print("Hugging Faceから猫(cat)データを取得中...")
    try:
        # 例として猫の画像が含まれるオープンなデータセットを使用
        # （別データセットやカスタムの猫画像セットに変更可能です）
        cat_ds = load_dataset("cats_vs_dogs", split="train", streaming=True) # もしくは別の猫データセット
        count = 0
        for item in tqdm(cat_ds, desc="Saving cats", total=max_per_class):
            if count >= max_per_class:
                break
            # cats_vs_dogsの場合はラベル0がネコ、1がイヌなどのフィルタリングが可能
            # ここではシンプルに画像を取得
            image = item["image"]
            image.save(f"./LSUN_origin/cat/{count:05d}.jpg")
            count += 1
    except Exception as e:
        print(f"catのダウンロード中にエラーが発生しました: {e}")

    print("すべての自動ダウンロードと保存が完了しました！")
def prepare_and_save_images(
    data_dir="./LSUN_origin", save_dir="./dataset/val_256",max_images=1000
):
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),        
    ])
    print("ダウンロード開始")
    
    dataset=datasets.ImageFolder(root=data_dir,transform=transform)
    loader=DataLoader(dataset, batch_size=32, shuffle=False,num_workers=4)
    
    os.makedirs(save_dir,exist_ok=True)
    count=0
    
    print(f"画像を256*256にリサイズして{save_dir}に保存します。")
    for batch_idx, (images,label)in enumerate(tqdm(loader)):
        for i in range(images.size(0)):
            if count >= max_images:
                break
            class_name=dataset.classes[label[i].item()]
            class_folder=os.path.join(save_dir,class_name)
            os.makedirs(class_folder,exist_ok=True)
            
            img_path=os.path.join(class_folder,f"{count:05d}.png")
            save_image(images[i],img_path)
            count+=1
            
            if count >= max_images:
                break
            
    print(f"完了！合計{count}枚の画像を保存しました。")
def main():
    prepare_and_save_images()
    print("稼働")
    
if __name__=="__main__":
    main()