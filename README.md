# RagRohtoCompetition

- [該当コンペ](https://raggle.jp/competition/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae#overview)

## 仮想環境作成
Anacondaのパスを通し、condaコマンドが利用可能か確認する。  
以下のコマンドで仮想環境を作成する。  
```shell
conda create -n {環境名} python=3.9
```
仮想環境を起動する  
```shell
conda activate {環境名}
```
ライブラリをインストール  
- `scripts/iTransformer.ipynb` を実行する場合
```shell
pip install -r require_itrans.txt --user
```
- `scripts/Prophet.ipynb` を実行する場合
```shell
pip install -r require_prophet.txt --user
```