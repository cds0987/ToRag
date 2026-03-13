from .load import loader_manager

# Import loaders so they self-register
from .csv_loader import csvLoader
from .tsv_loader import tsvLoader
from .json_loader import jsonLoader
from .jsonl_loader import jsonlLoader
from .txt_loader import txtLoader
from .html_loader import htmlLoader
from .xml_loader import xmlLoader
from .md_loader import mdLoader
from .excel_loader import excelLoader
from .docx_loader import docxLoader
from .pdf_loader import pdfLoader
from .parquet_loader import parquetLoader
from .hf_loader import HuggingFaceLoader
