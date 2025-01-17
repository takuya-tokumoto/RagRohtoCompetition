import json
import sys
import time

import pdfplumber
import requests
from dotenv import load_dotenv
from langchain import callbacks
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from main_0011 import pdf_file_urls, rag_implementation


def evaluate_rag(csv_path: str, output_path: str) -> None:
    """
    RAGモデルの評価を行う関数

    Args:
        csv_path (str): 質問と期待する回答が含まれるCSVファイルのパス
        output_path (str): 評価結果を保存するCSVファイルのパス
    """
    # 評価データを読み込む
    data = pd.read_csv(csv_path)

    # PDFファイルのURLリストを表示（必要に応じて使用）
    print("Evaluating with the following PDF files:")
    for url in pdf_file_urls:
        print(url)

    # 評価メトリックを設定
    rouge = load_metric("rouge")

    # 評価結果を格納するリスト
    results = []

    # データセットをループ処理
    for _, row in data.iterrows():
        question = row["question"]
        expected_answer = row["expected_answer"]

        # rag_implementation関数を呼び出して回答を生成
        generated_answer = rag_implementation(question)

        # メトリックを計算
        score = rouge.compute(predictions=[generated_answer], references=[expected_answer])
        results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "rouge-l": score["rougeL"].mid.fmeasure,
            }
        )

    # 結果をデータフレームに変換
    results_df = pd.DataFrame(results)

    # 評価結果を保存
    results_df.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    load_dotenv()

    querys = [
        "存在意義（パーパス）は、なんですか？",
        "事務連絡者の電話番号は？",
        "Vロートプレミアムは、第何類の医薬品ですか？",
        "肌ラボ 極潤ヒアルロン液の詰め替え用には、何mLが入っていますか？",
        "LN211E8は、どのようなhiPSCの分化において、どのように作用しますか？",
    ]
    groundtrurhs = [
        "世界の人々に商品やサービスを通じて「健康」をお届けすることによって、当社を取り巻くすべての人や社会を「Well-being」へと導き、明日の世界を元気にすることです。",
        "（06）6758-1235です。",
        "第2類医薬品です。",
        "170mLが入っています。",
        "Wnt 活性化を通じて神経堤細胞への分化を促進します。",
    ]

    for i, (query, groundtrurh) in enumerate(zip(querys, groundtrurhs)):
        print(f"Question {i+1}.")
        ans = rag_implementation(query)
        print(query)

        print(" ■正解：")
        print(groundtrurh)
        print(" ■予測文章：")
        print(ans)
        print("=" * 10)
