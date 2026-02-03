"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct
import csv


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response


def integrate_csv_to_document(csv_file_path):
    """
    CSVファイルを1つの統合されたドキュメントに変換する。
    各行の情報を文脈を保持した形で結合する。
    """
    integrated_document = ""

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # 各行の情報を結合して1つの文書に統合
            integrated_document += f"社員ID: {row['社員ID']}, 氏名: {row['氏名（フルネーム）']}, 部署: {row['部署']}, 役職: {row['役職']}, スキル: {row['スキルセット']}, 資格: {row['保有資格']}\n"

    return integrated_document

def index_documents(search_engine, csv_file_path):
    """
    統合されたドキュメントを検索エンジンにインデックス化する。
    """
    document = integrate_csv_to_document(csv_file_path)
    search_engine.index_document(document)


def optimize_query(query):
    """
    検索クエリを最適化する。

    Args:
        query: ユーザーからの検索クエリ

    Returns:
        最適化された検索クエリ
    """
    # 例: 「人事部に所属している従業員情報を一覧化して」を「人事部 AND 従業員」に変換
    optimized_query = query.replace("に所属している", "AND").replace("情報を一覧化して", "").strip()
    return optimized_query

def context_aware_search(search_engine, query, context):
    """
    文脈を考慮した検索を実行する。

    Args:
        search_engine: 検索エンジンのインスタンス
        query: ユーザーからの検索クエリ
        context: 文脈情報

    Returns:
        検索結果のリスト
    """
    # 文脈をクエリに追加して検索
    context_query = f"{query} AND {context}"
    results = search_engine.search(context_query)
    return results