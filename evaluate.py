import csv

from dotenv import load_dotenv

from main_0008 import rag_implementation as rag_implementation_pre  # 後で消す
from main_0012 import rag_implementation


def load_csv(file_path: str) -> list:
    """
    CSVファイルを読み込む関数。

    Args:
        file_path (str): CSVファイルのパス。

    Returns:
        list: 辞書形式の質問と正解のリスト。
    """
    qa_pairs = []
    with open(file_path, mode="r", encoding="utf-8-sig") as file:  # エンコーディングをutf-8-sigに変更
        reader = csv.DictReader(file)
        for row in reader:
            qa_pairs.append({"query": row["query"], "groundtruth": row["groundtruth"]})
    return qa_pairs


if __name__ == "__main__":
    load_dotenv()

    # 外部CSVファイルの読み込み
    csv_path = "/root/git_work/90_adhoc/RagRohtoCompetition/validation_dataset_v001.csv"  # CSVファイルのパス
    qa_pairs = load_csv(csv_path)

    # RAG処理
    for i, qa_pair in enumerate(qa_pairs):
        query = qa_pair["query"]
        groundtruth = qa_pair["groundtruth"]

        print(f"Question {i+1}.")
        ans = rag_implementation(query)
        ans_pre = rag_implementation_pre(query)

        print(query)

        print(" ■正解：")
        print(groundtruth)
        print(" ■(改修前)予測文章：")
        print(ans_pre)
        print(" ■(改修後)予測文章：")
        print(ans)
        print("=" * 10)
