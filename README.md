# 製薬企業向け高度なRAGシステムの精度改善コンペ(RAGGLE)

## 概要
製薬企業の多様なドキュメント（Well-beingレポート、財務諸表、商品紹介資料、研究論文など）を対象とした高度なRAG（Retrieval Augmented Generation）システムの構築と精度改善が目的。
- [該当サイトURL](https://raggle.jp/competition/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae#overview)

![image](https://github.com/user-attachments/assets/63cfcd85-327b-452c-ba11-6c7c57f6b51c)


## 仮想環境作成
Anacondaのパスを通し、condaコマンドが利用可能か確認する。  
以下のコマンドで仮想環境を作成する。  
```shell
conda create -n {環境名} python=3.11.9
```
仮想環境を起動する  
```shell
conda activate {環境名}
```
ライブラリをインストール  
```shell
pip install -r requirements.txt
```

## APIキーの設定
クローンしたrepositoryを同じ階層に`.env`ファイルを作成しAPIキーを設定してください

```
# OpenAI
OPENAI_API_KEY={OpenAIのAPIキー}

# LangSmith
LANGCHAIN_API_KEY={LangSmithのAPIキー}
```
- OpenAI APIの取得方法
  - [OpenAIのAPIキー取得方法|2024年7月最新版|料金体系や注意事項](https://qiita.com/kurata04/items/a10bdc44cc0d1e62dad3)
- LangSmith APIの取得方法
  - [LangSmith】アカウント作成からAPIキーの発行方法を解説！](https://highreso.jp/edgehub/machinelearning/langsmithapi.html)
