import json
import os
import re
import sys
import time
from functools import partial
from operator import itemgetter
from typing import Any, Callable, Dict, List

import easyocr
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from langchain import callbacks
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseOutputParser, BaseRetriever
from langchain.schema import Document as LCDocument
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
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

    # ---------------------------
    # 1) PDFダウンロード＆読み込み
    # ---------------------------
    def download_and_load_pdfs(urls: list) -> list:
        """
        指定されたURLからPDFファイルをダウンロードし、テキストを抽出する関数。

        この関数は、与えられたPDFファイルのURLリストからファイルをダウンロードし、PyMuPDFまたはEasyOCRを
        使用してPDFファイルからテキストを抽出します。

        Args:
            urls (list): ダウンロード対象のPDFファイルのURLリスト。

        Raises:
            Exception: ダウンロードまたはPDFの読み取りに失敗した場合に発生。

        Returns:
            list: Documentオブジェクトのリスト。
                各Documentにはテキスト（`page_content`）とURLソース（`metadata`）が含まれる。

        Examples:
            >>> urls = ["https://example.com/sample1.pdf", "https://example.com/sample2.pdf"]
            >>> documents = download_and_load_pdfs(urls)
            >>> for doc in documents:
            >>>     print(doc.page_content)

        Note:
            'V_Rohto_Premium_Product_Information.pdf'はテキストすべてが画像化されているため
            EasyOCRを適応している。それ以外のPDFファイルはPyMuPDFを適応。
        """

        try:

            def download_pdf(url: str, save_path: str) -> None:
                """
                指定されたURLからPDFファイルをダウンロードしてローカルに保存する関数。

                この関数は、指定されたURLからPDFファイルをダウンロードし、
                ローカルストレージに保存します。HTTPステータスコードが200でない場合は
                例外をスローします。

                Args:
                    url (str): ダウンロード元のPDFファイルのURL。
                    save_path (str): ダウンロードしたPDFファイルを保存するローカルパス。

                Raises:
                    Exception: ダウンロードが失敗した場合（HTTPステータスコードが200以外）。

                Examples:
                    >>> download_pdf("https://example.com/sample.pdf", "sample.pdf")
                    ファイル "sample.pdf" が現在の作業ディレクトリに保存されます。
                """

                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                else:
                    raise Exception(f"Failed to download {url}")

            def read_pdf_by_pymupdf(pdf_path: str) -> list:
                """
                PyMuPDFを使用してPDFファイルのテキストを抽出する関数。

                この関数は、指定されたPDFファイルを開き、すべてのページから
                テキストを抽出して1つの文字列として結合します。

                Args:
                    pdf_path (str): 読み取るPDFファイルのパス。

                Returns:
                    str: PDF内の全ページから抽出されたテキストを結合した文字列。

                Examples:
                    >>> text = read_pdf_by_pymupdf("example.pdf")
                    >>> print(text)
                    "ページ1のテキスト\nページ2のテキスト\n..."

                Note:
                    - ファイルが存在しない場合や無効なPDFの場合は例外が発生します。
                    - Unicodeの特殊文字を含む場合、正しく処理されない場合があります。
                """

                with fitz.open(pdf_path) as pdf:
                    full_text = ""
                    for page in pdf:
                        text = page.get_text()  # プレーンテキストを取得する
                        if text:
                            full_text += text

                return full_text

            def read_pdf_by_easyocr(pdf_path: str) -> list:
                """
                EasyOCRを使用してPDFファイルの画像からテキストを抽出する関数。

                Args:
                    pdf_path (str): 読み取るPDFファイルのパス。

                Returns:
                    str: PDF内の全ページから抽出されたテキストを結合した文字列。
                """

                def pdf_to_images(pdf_path: str, output_folder: str = "pdf_images", dpi: int = 300) -> list:
                    """
                    PDFの各ページを画像ファイルとして保存する関数。

                    この関数は指定されたPDFファイルを開き、各ページを指定された解像度（dpi）で
                    画像化し、指定したフォルダに保存します。

                    Args:
                        pdf_path (str): 画像化する対象のPDFファイルのパス。
                        output_folder (str, optional): 画像ファイルを保存するフォルダのパス。デフォルトは "pdf_images"。
                        dpi (int, optional): 出力画像の解像度（ドット・パー・インチ）。デフォルトは300。

                    Returns:
                        list: 保存された画像ファイルのパスリスト。

                    Raises:
                        Exception: PDFファイルが存在しない場合や、保存時にエラーが発生した場合。

                    Examples:
                        >>> images = pdf_to_images("example.pdf", "output_folder", dpi=200)
                        >>> print(images)
                        ['output_folder/page_1.png', 'output_folder/page_2.png']

                    Note:
                        - 出力フォルダが存在しない場合は自動的に作成されます。
                        - 画像はPNG形式で保存されます。
                    """

                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    pdf_document = fitz.open(pdf_path)
                    image_paths = []

                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        matrix = fitz.Matrix(dpi / 72, dpi / 72)
                        pix = page.get_pixmap(matrix=matrix)
                        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
                        pix.save(image_path)
                        image_paths.append(image_path)

                    pdf_document.close()
                    return image_paths

                def extract_text_from_images(image_paths):
                    """
                    画像からテキストを抽出する関数。

                    Args:
                        image_paths (list): 画像ファイルのパスリスト。

                    Returns:
                        str: 抽出されたテキストを結合した文字列。
                    """
                    reader = easyocr.Reader(["ja", "en"])  # 日本語と英語をサポート
                    extracted_text = ""

                    for image_path in image_paths:
                        results = reader.readtext(image_path)
                        for _, text, _ in results:
                            extracted_text += text + "\n"

                    return extracted_text

                def pdf_text_extraction_pipeline(pdf_path):
                    """
                    PDFファイルからテキストを抽出するパイプライン。

                    Args:
                        pdf_path (str): PDFファイルのパス。

                    Returns:
                        str: PDF全体から抽出されたテキスト。
                    """

                    image_paths = pdf_to_images(pdf_path)
                    extracted_text = extract_text_from_images(image_paths)

                    return extracted_text

                return pdf_text_extraction_pipeline(pdf_path)

            documents = []

            for i, url in enumerate(urls):
                tmp_path = f"pdf_{i}.pdf"
                download_pdf(url, tmp_path)

                if i != 3:  # V_Rohto_Premium_Product_Information.pdf のみOCRでテキスト抽出
                    full_text = read_pdf_by_pymupdf(tmp_path)
                else:
                    full_text = read_pdf_by_easyocr(tmp_path)

                documents.append(Document(page_content=full_text, metadata={"source": url}))

            return documents

        except Exception as e:
            raise Exception(f"Error reading {url}: {e}")

    # ---------------------------
    # 2) 文書のチャンク分割
    # ---------------------------
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

        # Step 1: Initialize text splitter
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Step 2: Split documents
        doc_splits = text_splitter.split_documents(docs_list)

        return doc_splits

    # ---------------------------
    # 3) クエリ生成モデル (複数検索クエリの拡張)
    # ---------------------------
    class Query(BaseModel):
        """質問のリスト"""

        query: list[str] = Field(description="質問のリスト")

    def query_generator(original_query: dict) -> list[str]:
        """
        指定された質問に基づいて、関連する複数の検索クエリを生成する関数。

        この関数は、入力された質問（original_query["question"]）を元に、
        クエリ生成用のプロンプトを通じて、検索に使用可能なクエリリストを作成します。
        生成されるクエリは、元の質問を大きく変えずに、関連するコンテキストを付加した形式になります。

        Args:
            original_query (dict):
                - 質問が含まれる辞書形式の入力データ。
                - 必須キー: `"question"`（生成元となる質問の文字列）。

        Returns:
            list[str]:
                - 元の質問を含む、関連する複数の検索クエリのリスト。

        Example:
            >>> original_query = {"question": "What is RAG?"}
            >>> query_generator(original_query)
            [
                "What is RAG?",
                "RAGの詳細とは？",
                "RAGの仕組みを教えてください",
                "RAGについての簡単な説明"
            ]

        Note:
            - この関数は LangChain を使用してプロンプトを定義し、ChatOpenAI モデルを用いてクエリを生成します。
            - ChatOpenAI モデルは事前に定義された `model` を使用します。
            - 日本語のコンテキストを付加したクエリを生成するよう設計されています。
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that generates multiple search queries based on a single input query.",
                ),
                (
                    "user",
                    "Generate multiple search queries related to: {question}. When creating queries, please refine or add closely related contextual information in Japanese, without significantly altering the original query's meaning",
                ),
                ("user", "OUTPUT (4 queries):"),
            ]
        )

        user_query = original_query.get("question")

        llm = ChatOpenAI(
            model_name=model,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(Query)

        def query_to_list(query):
            query_list = []
            query_list.append(user_query)
            query_list.extend(query.query)
            return query_list

        query_generator_chain = prompt | structured_llm | query_to_list
        queries = query_generator_chain.invoke({"question": user_query})

        return queries

    # ---------------------------
    # 4-1) BM25関連
    # ---------------------------
    # ※ 以下、必要に応じて形態素解析器の設定が必要です
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
        doc_splits: List[Document],
        word_preprocess_func: Callable,
        char_preprocess_func: Callable,
        k_value: int = 4,
        word_weight: float = 0.7,
        char_weight: float = 0.3,
    ) -> EnsembleRetriever:
        """
        BM25Retriever(単語版, 文字版) を EnsembleRetriever でまとめる
        """
        # Word-level BM25
        word_retriever = BM25Retriever.from_documents(doc_splits, preprocess_func=word_preprocess_func)
        word_retriever.k = k_value

        # Char-level BM25
        char_retriever = BM25Retriever.from_documents(doc_splits, preprocess_func=char_preprocess_func)
        char_retriever.k = k_value

        # Ensembleにまとめる
        ensemble_retriever = EnsembleRetriever(
            retrievers=[word_retriever, char_retriever], weights=[word_weight, char_weight]
        )
        return ensemble_retriever

    # ---------------------------
    # 4-2) Chroma検索関連
    # ---------------------------

    def create_vectorstore(doc_splits: list) -> Chroma:
        """
        テキストデータからベクトルストアを生成する関数

        Args:
            docs (list): Documentオブジェクトのリスト

        Returns:
            vectorstore (Chroma): ベクトルストア
        """
        try:
            embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = Chroma.from_documents(doc_splits, embedding_function)
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {e}")

    # ---------------------------
    # 5) RRFによるマージ
    # ---------------------------
    def reciprocal_rank_fusion(results: list[list], k=60) -> List[str]:
        """
        Reciprocal Rank Fusion (RRF) アルゴリズムを使用して、複数のランキング結果を統合する関数。

        この関数は、複数のランキング結果（`results`）を受け取り、各ドキュメントに対するスコアを
        計算して統合します。統合されたスコアを基に、ドキュメントを再ランク付けし、上位数件のドキュメントを返します。

        RRFでは、各ランキングリスト内の順位に基づいてスコアを計算します。順位が低い（高順位）ドキュメントに
        より高いスコアが付与されます。計算式は以下の通りです：

        \[
        \text{score} = \sum_{i=1}^{n} \frac{1}{\text{rank}_i + k}
        \]
        - `rank_i`: 各リストにおけるドキュメントの順位（0から始まる）。
        - `k`: 安定性を調整するための定数（デフォルトは60）。

        Args:
            results (list[list]): 複数のランキングリスト。各リストにはドキュメントが順位付けされています。
            k (int, optional): スコア計算時の調整パラメータ。デフォルトは60。

        Returns:
            list: RRFアルゴリズムを通じて再ランク付けされた、上位4件のドキュメントのリスト。

        Example:
            >>> results = [
            ...     [{"id": 1}, {"id": 2}, {"id": 3}],
            ...     [{"id": 2}, {"id": 3}, {"id": 4}],
            ...     [{"id": 1}, {"id": 4}, {"id": 3}]
            ... ]
            >>> reciprocal_rank_fusion(results)
            [{"id": 2}, {"id": 1}, {"id": 3}, {"id": 4}]

        Note:
            - `Document.to_json(doc)` を使用して、ドキュメントを JSON 形式に変換して統合しています。
            - 上位数件のドキュメントを返しますが、必要に応じて数を調整できます。
        """

        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = json.dumps(Document.to_json(doc), ensure_ascii=False)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (doc, score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return [x[0] for x in reranked_results[:8]]

    def format_docs(docs: list) -> str:
        """ドキュメントのリストを整形し、フォーマット済みのテキストを返す関数。

        Args:
            docs (list): JSON形式の文字列を含むドキュメントのリスト。

        Returns:
            str: 整形されたドキュメントのテキスト。各ドキュメントの内容（`page_content`）を
                抜き出し、2つの改行文字で区切った単一の文字列として返します。

        Example:
            >>> docs = [
            ...     '{"kwargs": {"page_content": "Content of doc 1"}}',
            ...     '{"kwargs": {"page_content": "Content of doc 2"}}'
            ... ]
            >>> format_docs(docs)
            'Content of doc 1\\n\\nContent of doc 2'
        """

        format_docs = []
        for n in docs:
            docs_json = json.loads(n)
            format_docs.append(docs_json["kwargs"]["page_content"])
        return "\n\n".join(format_docs)

    # ---------------------------------------------------------------
    # ここからハイブリッド検索（BM25 + ベクトル検索）を準備
    # ---------------------------------------------------------------

    # 1) PDFを読み込む
    docs = download_and_load_pdfs(pdf_file_urls)
    # 2) BM25用にドキュメント分割
    doc_splits = document_transformer(docs, chunk_size=1200, chunk_overlap=100)

    # 3) ベクトルストアを作成
    db = create_vectorstore(doc_splits)
    vector_retriever = db.as_retriever(
        search_kwargs={"k": 8}
    )  # search_kwargs:類似検索時に返すDocumentsオブジェクトの数

    # 4) BM25 Retriever (Word + Char Ensemble)
    bm25_ensemble_retriever = create_bm25_retrievers(
        doc_splits,
        preprocess_word_func,
        preprocess_char_func,
        k_value=8,
        word_weight=0.7,
        char_weight=0.3,  # k_value: 返す件数
    )

    # ---------------------------------------------------------------
    # 5) RAGパイプライン:
    #    (1) 質問をクエリ群に拡張 → (2) BM25で検索 → (3) ベクトル検索 → (4) RRFで融合
    # ---------------------------------------------------------------

    # (A) 質問を複数クエリに変換する Runnable
    query_gen = RunnableLambda(query_generator)

    # (B) BM25 & ベクトル検索の「map()」: 各クエリに対して invoke を実行
    def bm25_map(queries: list[str]) -> list[List[LCDocument]]:
        """
        複数のクエリをBM25 EnsembleRetrieverで検索し、それぞれに対応する検索結果を返します。

        Args:
            queries (list[str]):
                検索したいクエリ文字列のリスト。

        Returns:
            list[list[LCDocument]]:
                各クエリに対する検索結果が格納された二次元リスト。
                具体的には、`all_results` は以下のような構造になります。

                [
                    [LCDocument, LCDocument, ...],  # 1つ目のクエリに対する検索結果
                    [LCDocument, LCDocument, ...],  # 2つ目のクエリに対する検索結果
                    ...
                ]

                - 外側のリスト: クエリごとの検索結果をまとめたもの
                - 内側のリスト: 各クエリに対して BM25 EnsembleRetriever が返す LCDocument のリスト
        """
        all_results = []
        for q in queries:
            docs_ = bm25_ensemble_retriever.invoke(q)
            all_results.append(docs_)
        return all_results

    def vector_map(queries: list[str]) -> list[List[LCDocument]]:
        """
        複数のクエリをベクトル検索リトリーバで検索し、それぞれに対応する検索結果を返します。

        Args:
            queries (list[str]):
                検索したいクエリ文字列のリスト。

        Returns:
            list[list[LCDocument]]:
                各クエリに対する検索結果が格納された二次元リスト。
                具体的には、`all_results` は以下のような構造になります。

                [
                    [LCDocument, LCDocument, ...],  # 1つ目のクエリに対する検索結果
                    [LCDocument, LCDocument, ...],  # 2つ目のクエリに対する検索結果
                    ...
                ]

                - 外側のリスト: クエリごとの検索結果をまとめたもの
                - 内側のリスト: 各クエリに対して ベクトル検索リトリーバ が返す LCDocument のリスト
        """
        all_results = []
        for q in queries:
            docs_ = vector_retriever.invoke(q)
            all_results.append(docs_)
        return all_results

    bm25_search = RunnableLambda(bm25_map)
    vector_search = RunnableLambda(vector_map)

    # (C) RRFでマージ
    def rrf_merge(results_dict: dict) -> list[str]:
        """
        results_dict: {"bm25": [...], "vector": [...]}
            - それぞれがクエリごとの検索結果 list[List[Document]] を持つ
        """
        bm25_results = results_dict["bm25"]  # list[List[Document]]
        vec_results = results_dict["vector"]
        # クエリ数は同じはずなので zip で回しながらRRF
        final_docs = []
        for bm25_list, vec_list in zip(bm25_results, vec_results):
            fused = reciprocal_rank_fusion([bm25_list, vec_list], k=60)
            final_docs.extend(fused)
        return final_docs

    rrf_fusion = RunnableLambda(rrf_merge)

    # (D) format_docs で文字列化
    # (E) LLM（ChatOpenAI）で最終回答

    # -- LangChain の Runnableを組み合わせる --
    rag_fusion_retriever = (
        {"question": itemgetter("question")} | query_gen | {"bm25": bm25_search, "vector": vector_search} | rrf_fusion
    )

    template = """
    # ゴール
    私は、参考文章と質問を提供します。
    あなたは、参考文章に基づいて、質問に回答できそうな事実を網羅しつつ回答文を生成してください。

    # 質問
    {question}

    # 参考文章
    {context}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chat = ChatOpenAI(model=model, temperature=0)
    chain = (
        {"context": rag_fusion_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat
        | StrOutputParser()
    )
    answer = chain.invoke({"question": question})

    # ---------------------------------------------------------------
    # 6) 回答内容を再整理:
    #    (1) 質問をクエリ群に拡張 → (2) BM25で検索 → (3) ベクトル検索 → (4) RRFで融合
    # ---------------------------------------------------------------

    def create_reanswer(question: str, answer: str) -> str:
        """
        質問文に対する回答文を簡潔に要約する関数。

        この関数は、プロンプトテンプレートを使用して回答文をリライトし、
        質問文に対する最も重要で簡潔な情報を抽出します。

        Args:
            question (str): 質問文。
            answer (str): 質問文に対する回答文。

        Returns:
            str: 回答文の中から抽出された簡潔で重要な情報。
        """

        simplify_template = """
        あなたは製薬企業の広報担当者です。
        以下に質問文に対する回答案があります。
        回答案は冗長になりがちなため、以下ルールにしたがってブラッシュアップし、新たな回答文を生成してください。
        1. 質問文で問われている内容のみを回答文から抽出し、適切な回答文に修正してください
        2. 回答文の中で重複して同じ内容が繰り返し言及される個所については、一つにまとめて読みやすくしてください。
        3. 重要ワードを言い換えたりすることなく事実のみを的確に回答するようにしてください。

        # 質問文
        {question}	

        # 回答案
        {answer}
        """

        llm = ChatOpenAI(
            model_name=model,
            temperature=0,
        )
        simplify_prompt = ChatPromptTemplate.from_template(simplify_template)
        simplify_chain = simplify_prompt | llm | StrOutputParser()
        result_pre = simplify_chain.invoke({"question": question, "answer": answer})

        remove_sensitive_word_template = """
        あなたは製薬企業の広報担当者です。
        以下に質問文に対する回答文があります。
        回答文から、有害性、攻撃性、個人情報が含まれている可能性のある部分を削除した回答に修正してください。  
        修正の際は、以下の基準に従ってください：

        1. **有害性の排除**: 他者を傷つける、誤解を招く可能性のある表現を削除または書き換えます。
        2. **攻撃性の回避**: 他者を批判・非難するようなトーンや内容を取り除きます。
        3. **個人情報の除去**: 公人を除き、個人のプライベート情報は削除します。

        # 質問文
        {question}	

        # 回答文
        {answer}
        """

        remove_sensitive_word_prompt = PromptTemplate.from_template(remove_sensitive_word_template)
        remove_sensitive_word_chain = remove_sensitive_word_prompt | llm | StrOutputParser()
        result = remove_sensitive_word_chain.invoke({"question": question, "answer": result_pre})

        return result

    answer = create_reanswer(question, answer)

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
