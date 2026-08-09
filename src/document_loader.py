import os
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MultiFormatDocumentLoader:
    """Handles loading and chunking for PDF, DOCX, TXT, and XLSX files."""

    def __init__(self, file_path: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.file_path = file_path

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def _load_excel_rows(self, file_path: str) -> List[Document]:
        """Loads each Excel row as its own explicitly-labeled document
        (e.g. 'Region: Europe | Product: Widget Pro | ...'), instead of
        letting a generic loader flatten the whole sheet into one blob of
        text. Flattening loses which value belongs to which row, which
        causes the LLM to cross-attribute data (e.g. mixing up which sales
        rep belongs to which region) when it later reads the chunk."""
        docs = []
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name).fillna("")
            for i, row in df.iterrows():
                row_text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
                content = f"[Sheet: {sheet_name}, Row {i + 2}] {row_text}"
                docs.append(Document(
                    page_content=content,
                    metadata={"source": file_path, "sheet": sheet_name, "row": i + 2}
                ))
        return docs

    def load_document(self, file_path: str = None) -> List[Document]:
        """Loads a single document based on file extension."""
        target_path = file_path or self.file_path
        if not target_path:
            raise ValueError("No file path provided to load_document.")

        ext = os.path.splitext(target_path)[-1].lower()

        if ext == ".pdf":
            docs = PyPDFLoader(target_path).load()
        elif ext == ".txt":
            docs = TextLoader(target_path, encoding="utf-8").load()
        elif ext in [".docx", ".doc"]:
            docs = UnstructuredWordDocumentLoader(target_path).load()
        elif ext in [".xlsx", ".xls"]:
            docs = self._load_excel_rows(target_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        print(f"[INFO] Successfully loaded '{os.path.basename(target_path)}' ({len(docs)} raw pages/elements).")
        return docs

    def load_and_split(self, file_path: str = None) -> List[Document]:
        """Convenience method for single file ingestion & chunking (used by the Gradio app)."""
        target_path = file_path or self.file_path
        docs = self.load_document(target_path)
        chunks = self.text_splitter.split_documents(docs)
        print(f"[INFO] Split '{os.path.basename(target_path)}' into {len(chunks)} chunks.")
        return chunks

    def process_directory(self, data_dir: str) -> List[Document]:
        """Ingests all supported files from a folder and splits them into chunks."""
        all_docs = []
        supported_exts = {".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls"}

        for root, _, files in os.walk(data_dir):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in supported_exts:
                    full_path = os.path.join(root, file)
                    try:
                        docs = self.load_document(full_path)
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"[ERROR] Failed to load {file}: {e}")

        chunks = self.text_splitter.split_documents(all_docs)
        print(f"[INFO] Total text chunks generated: {len(chunks)}")
        return chunks