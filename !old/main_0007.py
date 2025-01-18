import json
import sys
import time
from typing import Any, Callable, Dict, List

import pdfplumber
import requests
from dotenv import load_dotenv
from langchain import callbacks
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from sudachipy import dictionary, tokenizer

# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Financial_Statements_2023.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Hada_Labo_Gokujun_Lotion_Overview.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Shibata_et_al_Research_Article.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/V_Rohto_Premium_Product_Information.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Well-Being_Report_2024.pdf",
]
# ==============================================================================


# ==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
# ==============================================================================
def rag_implementation(question: str) -> str:
    """
    ロート製薬の製品・企業情報に関する質問に対して回答を生成する関数
    この関数は与えられた質問に対してRAGパイプラインを用いて回答を生成します。

    Args:
        question (str): ロート製薬の製品・企業情報に関する質問文字列

    Returns:
        answer (str): 質問に対する回答

    Note:
        - デバッグ出力は標準出力に出力しないでください
        - model 変数 と pdf_file_urls 変数は編集しないでください
        - 回答は日本語で生成してください
    """

    # 戻り値として質問に対する回答を返却してください。
    def download_and_load_pdfs(urls: list) -> list:
        """
        PDFファイルをダウンロードして読み込む関数

        Args:
            urls (list): PDFファイルのURLリスト

        Returns:
            documents (list): PDFファイルのテキストデータを含むDocumentオブジェクトのリスト

        Raises:
            Exception: ダウンロードまたは読み込みに失敗した場合に発生する例外

        Examples:
            >>> urls = ["https://example.com/example.pdf"]
            >>> download_and_load_pdfs(urls)
            [Document(page_content="...", metadata={"source": "https://example.com/example.pdf"})]
        """
        try:

            def download_pdf(url, save_path):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                else:
                    raise Exception(f"Failed to download {url}")

            documents = []

            for i, url in enumerate(urls):
                tmp_path = f"pdf_{i}.pdf"
                download_pdf(url, tmp_path)

                with pdfplumber.open(tmp_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"

                    documents.append(Document(page_content=full_text, metadata={"source": url}))
            return documents
        except Exception as e:
            raise Exception(f"Error reading {url}: {e}")

    def document_transformer(
        docs_list: List[Document], chunk_size: int = 1100, chunk_overlap: int = 100, separator: str = "\n"
    ) -> List[Document]:
        """ドキュメントリストを LangChain の Document 型に変換し、指定されたサイズで分割する。

        Args:
            docs_list (List[Document]): `content` と `metadata` を持つドキュメントのリスト。
            chunk_size (int): 分割時のチャンクサイズ。
            chunk_overlap (int): 分割時のチャンクのオーバーラップサイズ。
            separator (str): チャンク間の区切り文字。

        Returns:
            List[Document]: 分割後の LangChain の Document 型のリスト。
        """
        # # Step 1: Convert to Document type
        # docs_list_converted = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in docs_list]

        # Step 2: Initialize text splitter
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Step 3: Split documents
        # doc_splits = text_splitter.split_documents(docs_list_converted)
        doc_splits = text_splitter.split_documents(docs_list)

        return doc_splits

    def generate_word_ngrams(text, i, j, binary=False):
        """テキストから単語単位のn-gramを生成します。

        Args:
            text (str): n-gramを生成する元のテキスト。
            i (int): n-gramの最小サイズ。
            j (int): n-gramの最大サイズ。
            binary (bool, optional): 重複を削除するかどうか。Trueの場合、重複を削除します。
                                    デフォルトはFalse。

        Returns:
            List[Tuple[str]]: 生成されたn-gramのリスト。各n-gramはタプル形式で表されます。
        """
        tokenizer_obj = dictionary.Dictionary(dict="core").create()
        mode = tokenizer.Tokenizer.SplitMode.A
        tokens = tokenizer_obj.tokenize(text, mode)
        words = [token.surface() for token in tokens]

        ngrams = []

        for n in range(i, j + 1):
            for k in range(len(words) - n + 1):
                ngram = tuple(words[k : k + n])
                ngrams.append(ngram)

        if binary:
            ngrams = list(set(ngrams))  # 重複を削除

        return ngrams

    def preprocess_word_func(text: str) -> List[str]:
        return generate_word_ngrams(text, 1, 1, True)

    def generate_character_ngrams(text, i, j, binary=False):
        """テキストから文字単位のn-gramを生成します。

        Args:
            text (str): n-gramを生成する元のテキスト。
            i (int): n-gramの最小サイズ。
            j (int): n-gramの最大サイズ。
            binary (bool, optional): 重複を削除するかどうか。Trueの場合、重複を削除します。
                                    デフォルトはFalse。

        Returns:
            List[str]: 生成されたn-gramのリスト。各n-gramは文字列形式で表されます。
        """
        ngrams = []

        for n in range(i, j + 1):
            for k in range(len(text) - n + 1):
                ngram = text[k : k + n]
                ngrams.append(ngram)

        if binary:
            ngrams = list(set(ngrams))  # 重複を削除

        return ngrams

    def preprocess_char_func(text: str) -> List[str]:
        i, j = 1, 3
        if len(text) < i:
            return [text]
        return generate_character_ngrams(text, i, j, True)

    def create_bm25_retrievers(
        doc_splits: List,
        word_preprocess_func: Callable,
        char_preprocess_func: Callable,
        k_value: int = 4,
        word_weight: float = 0.7,
        char_weight: float = 0.3,
    ) -> EnsembleRetriever:
        """BM25Retriever を単語と文字レベルで作成し、それを EnsembleRetriever で統合する。

        Args:
            doc_splits (List): 分割されたドキュメントのリスト。
            word_preprocess_func (Callable): 単語レベルの前処理関数。
            char_preprocess_func (Callable): 文字レベルの前処理関数。
            k_value (int): 各 BM25Retriever に設定する `k` の値。
            word_weight (float): EnsembleRetriever における単語リトリーバーの重み。
            char_weight (float): EnsembleRetriever における文字リトリーバーの重み。

        Returns:
            EnsembleRetriever: 作成された EnsembleRetriever オブジェクト。
        """
        # Create word-level BM25 retriever
        word_retriever = BM25Retriever.from_documents(doc_splits, preprocess_func=word_preprocess_func)
        word_retriever.k = k_value

        # Create char-level BM25 retriever
        char_retriever = BM25Retriever.from_documents(doc_splits, preprocess_func=char_preprocess_func)
        char_retriever.k = k_value

        # Create EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[word_retriever, char_retriever], weights=[word_weight, char_weight]
        )

        return ensemble_retriever

    docs = download_and_load_pdfs(pdf_file_urls)
    doc_splits = document_transformer(docs)
    ensemble_retriever = create_bm25_retrievers(doc_splits, preprocess_word_func, preprocess_char_func)

    template = """
    # ゴール
    私は、参考文章と質問を提供します。
    あなたは、参考文章に基づいて、質問に対する回答を生成してください。

    # 質問
    {question}

    # 参考文章
    {context}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chat = ChatOpenAI(model=model)
    chain = {"context": ensemble_retriever, "question": RunnablePassthrough()} | prompt | chat | StrOutputParser()
    answer = chain.invoke(question)

    return answer


# ==============================================================================


# ==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
# ==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        for attempt in range(2):  # 最大2回試行
            try:
                run_id = cb.traced_runs[0].id
                break
            except IndexError:
                if attempt == 0:  # 1回目の失敗時のみ
                    time.sleep(3)  # 3秒待機して再試行
                else:  # 2回目も失敗した場合
                    raise RuntimeError("Failed to get run_id after 2 attempts")

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
# ==============================================================================
